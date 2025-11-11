// Standalone 2-node network probe test
// This tests the actual NetworkProbeManager without requiring full XLA initialization
//
// Build:
//   bazel build --config=rocm //xla/backends/profiler/gpu:network_probe_standalone_test
//
// Run 2-node test:
//   Terminal 1: ./bazel-bin/.../network_probe_standalone_test --node_id=0 --num_nodes=2
//   Terminal 2: ./bazel-bin/.../network_probe_standalone_test --node_id=1 --num_nodes=2

#include "xla/backends/profiler/gpu/network_probe.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <csignal>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"
#include "absl/strings/str_cat.h"

ABSL_FLAG(int, node_id, -1, "Node ID (0 or 1)");
ABSL_FLAG(int, num_nodes, 2, "Total number of nodes");
ABSL_FLAG(std::string, node0_addr, "10.7.76.147:12345", "Node 0 address");
ABSL_FLAG(std::string, node1_addr, "10.227.7.189:12346", "Node 1 address");
// ABSL_FLAG(std::string, node0_addr, "127.0.0.1:12345", "Node 0 address");
// ABSL_FLAG(std::string, node1_addr, "127.0.0.1:12346", "Node 1 address");
ABSL_FLAG(int, duration_sec, 10, "Test duration in seconds");
ABSL_FLAG(int, probe_cadence_us, 800, "Probe cadence in microseconds");
ABSL_FLAG(int, probe_window_s, 4, "Probe window in seconds");

namespace xla {
namespace profiler {

volatile sig_atomic_t g_shutdown = 0;

void signal_handler(int signal) {
  g_shutdown = 1;
  LOG(INFO) << "Caught signal " << signal << ", shutting down...";
}

// Create a test configuration for 2-node setup
DistributedProfilerContext CreateTestConfig(int node_id, int num_nodes) {
  DistributedProfilerContext config;
  
  config.node_id = node_id;
  config.num_nodes = num_nodes;
  
  // Hardcoded addresses
  config.node_addresses = {
    absl::GetFlag(FLAGS_node0_addr),
    absl::GetFlag(FLAGS_node1_addr)
  };
  
  // Hardcoded directed graph: 0 -> 1 (node 0 probes node 1)
  if (node_id == 0) {
    config.neighbors = {1};        // OUT: Node 0 probes node 1
    config.in_neighbors = {};      // IN: Nobody probes node 0
  } else if (node_id == 1) {
    config.neighbors = {};         // OUT: Node 1 doesn't probe anyone
    config.in_neighbors = {0};     // IN: Node 0 probes node 1
  }
  
  // Hardcoded port assignments (simulating master assignment)
  // Edge 0->1: Node 1 listens on 20100, Node 0 receives responses on 20000
  config.edge_ports["probe_edge:0->1"] = {20100, 20000};
  
  // Probe parameters
  config.probe_cadence_us = absl::GetFlag(FLAGS_probe_cadence_us);
  config.probe_window_s = absl::GetFlag(FLAGS_probe_window_s);
  config.enable_probe_export = true;
  config.enable_clock_snapshots = false;
  config.enable_socket_timestamping = true;
  config.timestamp_sync_timeout = absl::Seconds(5);
  config.graph_policy = "test_directed";
  
  return config;
}

// Print configuration for debugging
void PrintConfig(const DistributedProfilerContext& config) {
  LOG(INFO) << "=== Test Configuration ===";
  LOG(INFO) << "Node ID: " << config.node_id;
  LOG(INFO) << "Num nodes: " << config.num_nodes;
  LOG(INFO) << "Addresses:";
  for (int i = 0; i < config.node_addresses.size(); ++i) {
    LOG(INFO) << "  Node " << i << ": " << config.node_addresses[i];
  }
  LOG(INFO) << "OUT-neighbors: " << absl::StrJoin(config.neighbors, ", ");
  LOG(INFO) << "IN-neighbors: " << absl::StrJoin(config.in_neighbors, ", ");
  LOG(INFO) << "Edge ports:";
  for (const auto& [key, ports] : config.edge_ports) {
    LOG(INFO) << "  " << key << ": listen=" << ports.first 
              << ", response=" << ports.second;
  }
  LOG(INFO) << "Probe cadence: " << config.probe_cadence_us << " us";
  LOG(INFO) << "Probe window: " << config.probe_window_s << " s";
  LOG(INFO) << "==========================";
}

// Read and print CSV results
void PrintResults(int node_id) {
  std::string csv_file = absl::StrCat("/tmp/alpha_beta_node", node_id, ".csv");
  
  LOG(INFO) << "\n=== Results ===";
  LOG(INFO) << "Reading: " << csv_file;
  
  std::ifstream file(csv_file);
  if (!file.is_open()) {
    LOG(WARNING) << "Could not open results file: " << csv_file;
    return;
  }
  
  std::string line;
  while (std::getline(file, line)) {
    LOG(INFO) << line;
  }
  file.close();
  LOG(INFO) << "===============";
}

absl::Status RunTest() {
  int node_id = absl::GetFlag(FLAGS_node_id);
  int num_nodes = absl::GetFlag(FLAGS_num_nodes);
  int duration_sec = absl::GetFlag(FLAGS_duration_sec);
  
  if (node_id < 0 || node_id >= num_nodes) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid node_id: ", node_id, " (must be 0 to ", num_nodes - 1, ")"));
  }
  
  LOG(INFO) << "Starting standalone network probe test";
  LOG(INFO) << "Node ID: " << node_id << " / " << num_nodes;
  
  // Create test configuration
  auto config = CreateTestConfig(node_id, num_nodes);
  PrintConfig(config);
  
  // Create NetworkProbeManager
  LOG(INFO) << "\nInitializing NetworkProbeManager...";
  auto probe_manager = std::make_unique<NetworkProbeManager>(config);
  
  // Initialize (sets up sockets and listener threads)
  auto status = probe_manager->Initialize();
  if (!status.ok()) {
    return status;
  }
  LOG(INFO) << "✅ NetworkProbeManager initialized";
  
  // Start probing
  if (config.neighbors.size() > 0) {
    LOG(INFO) << "\nStarting probe threads...";
    status = probe_manager->Start();
    if (!status.ok()) {
      return status;
    }
    LOG(INFO) << "✅ Probe threads started";
  } else {
    LOG(INFO) << "\nNo OUT-neighbors, running in listener-only mode";
  }
  
  // Wait for test duration or signal
  LOG(INFO) << "\n🔬 Running test for " << duration_sec << " seconds...";
  LOG(INFO) << "Press Ctrl+C to stop early\n";
  
  for (int i = 0; i < duration_sec && !g_shutdown; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Print progress every few seconds
    if ((i + 1) % 5 == 0) {
      LOG(INFO) << "Test progress: " << (i + 1) << "/" << duration_sec << " seconds";
    }
  }
  
  // Export data
  LOG(INFO) << "\n📊 Exporting data...";
  status = probe_manager->ExportData();
  if (!status.ok()) {
    LOG(WARNING) << "Export failed: " << status;
  } else {
    LOG(INFO) << "✅ Data exported";
  }
  
  // Shutdown
  LOG(INFO) << "\n🛑 Shutting down...";
  probe_manager->Shutdown();
  LOG(INFO) << "✅ Shutdown complete";
  
  // Print results
  PrintResults(node_id);
  
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace xla

int main(int argc, char** argv) {
  // Print to stderr first to confirm execution
  std::cerr << "=== Network Probe Standalone Test Starting ===" << std::endl;
  
  // Initialize logging with stderr
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  
  std::cerr << "Parsing command line..." << std::endl;
  absl::ParseCommandLine(argc, argv);
  
  std::cerr << "Setting up signal handlers..." << std::endl;
  // Set up signal handler
  std::signal(SIGINT, xla::profiler::signal_handler);
  std::signal(SIGTERM, xla::profiler::signal_handler);
  
  std::cerr << "Running test..." << std::endl;
  // Run test
  auto status = xla::profiler::RunTest();
  
  if (!status.ok()) {
    LOG(ERROR) << "❌ Test failed: " << status;
    std::cerr << "❌ Test failed: " << status << std::endl;
    return 1;
  }
  
  LOG(INFO) << "\n✅ Test completed successfully!";
  std::cerr << "✅ Test completed successfully!" << std::endl;
  return 0;
}


