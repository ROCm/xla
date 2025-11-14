// Standalone multi-node network probe test
// This tests the actual NetworkProbeManager without requiring full XLA initialization
//
// Build:
//   bazel build --config=rocm //xla/backends/profiler/gpu:network_probe_standalone_test
//
// Run 2-node test:
//   Terminal 1: ./bazel-bin/.../network_probe_standalone_test --rank=0 --num_nodes=2 --addresses=10.7.76.147:12345,10.227.7.189:12346
//   Terminal 2: ./bazel-bin/.../network_probe_standalone_test --rank=1 --num_nodes=2 --addresses=10.7.76.147:12345,10.227.7.189:12346
//
// Run 4-node test:
//   Terminal 1: ./bazel-bin/.../network_probe_standalone_test --rank=0 --num_nodes=4 --addresses=addr0:port0,addr1:port1,addr2:port2,addr3:port3
//   Terminal 2: ./bazel-bin/.../network_probe_standalone_test --rank=1 --num_nodes=4 --addresses=addr0:port0,addr1:port1,addr2:port2,addr3:port3
//   Terminal 3: ./bazel-bin/.../network_probe_standalone_test --rank=2 --num_nodes=4 --addresses=addr0:port0,addr1:port1,addr2:port2,addr3:port3
//   Terminal 4: ./bazel-bin/.../network_probe_standalone_test --rank=3 --num_nodes=4 --addresses=addr0:port0,addr1:port1,addr2:port2,addr3:port3

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
#include "absl/strings/str_split.h"

ABSL_FLAG(int, rank, -1, "Node rank/ID (0 to num_nodes-1)");
ABSL_FLAG(int, num_nodes, 2, "Total number of nodes");
ABSL_FLAG(std::string, addresses, "10.7.76.147:12345,10.227.7.189:12346", "Comma-separated list of all node addresses (e.g., '10.7.76.147:12345,10.227.7.189:12346')");
ABSL_FLAG(int, duration_sec, 10, "Test duration in seconds");
ABSL_FLAG(int, probe_cadence_us, 800, "Probe cadence in microseconds");
ABSL_FLAG(int, probe_window_s, 4, "Probe window in seconds");
ABSL_FLAG(std::string, topology, "ring", "Probe topology: 'ring' (each node probes next) or 'directed' (node 0 probes all others)");

namespace xla {
namespace profiler {

volatile sig_atomic_t g_shutdown = 0;

void signal_handler(int signal) {
  g_shutdown = 1;
  LOG(INFO) << "Caught signal " << signal << ", shutting down...";
}

// Create a dynamic test configuration for N-node setup
DistributedProfilerContext CreateTestConfig(int rank, int num_nodes, 
                                           const std::vector<std::string>& addresses,
                                           const std::string& topology) {
  DistributedProfilerContext config;
  
  config.node_id = rank;
  config.num_nodes = num_nodes;
  config.node_addresses = addresses;
  
  // Configure topology
  if (topology == "ring") {
    // Ring topology: each node probes the next one (last node probes first)
    int next_node = (rank + 1) % num_nodes;
    int prev_node = (rank - 1 + num_nodes) % num_nodes;
    
    config.neighbors = {next_node};       // OUT: probe the next node
    config.in_neighbors = {prev_node};    // IN: previous node probes us
    
    // Assign ports for the edge from this node to next
    std::string edge_key = absl::StrCat("probe_edge:", rank, "->", next_node);
    // Each edge uses unique ports: target node listens on 40000+target*100, sender receives on 40000+sender*100
    config.edge_ports[edge_key] = {40000 + next_node * 100, 40000 + rank * 100};
    
    config.graph_policy = "test_ring";
  } else if (topology == "directed") {
    // Directed topology: node 0 probes all others
    if (rank == 0) {
      // Node 0 probes all other nodes
      for (int i = 1; i < num_nodes; ++i) {
        config.neighbors.push_back(i);
        std::string edge_key = absl::StrCat("probe_edge:0->", i);
        config.edge_ports[edge_key] = {40000 + i * 100, 40000};
      }
      config.in_neighbors = {};  // Nobody probes node 0
    } else {
      // Other nodes only receive probes from node 0
      config.neighbors = {};         // Don't probe anyone
      config.in_neighbors = {0};     // Node 0 probes us
      std::string edge_key = absl::StrCat("probe_edge:", "0->", rank);
      config.edge_ports[edge_key] = {40000 + rank * 100, 40000};
    }
    config.graph_policy = "test_directed";
    config.probe_participants = {0};
  }
  else if(topology == "loop_2"){
    if(rank == 0){
      config.neighbors = {1};
      config.in_neighbors = {2,3};
      config.edge_ports["probe_edge:0->1"] = {40000 + 1 * 100, 40000};  // {40100, 40000}
      config.edge_ports["probe_edge:2->0"] = {40002, 40000 + 2 * 100}; // {40200, 40002}
      config.edge_ports["probe_edge:3->0"] = {40003, 40000 + 3 * 100}; // {40300, 40003}
    }
    else if(rank == 1){
      config.neighbors = {2, 3};
      config.in_neighbors = {0};
      config.edge_ports["probe_edge:1->2"] = {40000 + 2 * 100 + 1, 40000 + 1 * 100 + 2}; // {40201, 40102}
      config.edge_ports["probe_edge:1->3"] = {40000 + 3 * 100 + 1, 40000 + 1 * 100 + 3}; // {40301, 40103}
      config.edge_ports["probe_edge:0->1"] = {40000 + 1 * 100, 40000};
    }
    else if(rank == 2){
      config.neighbors = {0};
      config.in_neighbors = {1};
      config.edge_ports["probe_edge:2->0"] = {40002, 40000 + 2 * 100}; // {40200, 40002}
      config.edge_ports["probe_edge:1->2"] = {40000 + 2 * 100 + 1, 40000 + 1 * 100 + 2}; // {40201, 40102}
    }
    else if(rank == 3){
      config.neighbors = {0};
      config.in_neighbors = {1};
      config.edge_ports["probe_edge:3->0"] = {40003, 40000 + 3 * 100};
      config.edge_ports["probe_edge:1->3"] = {40000 + 3 * 100 + 1, 40000 + 1 * 100 + 3};
    }

  }
  
  // Probe parameters
  config.probe_cadence_us = absl::GetFlag(FLAGS_probe_cadence_us);
  config.probe_window_s = absl::GetFlag(FLAGS_probe_window_s);
  config.enable_probe_export = true;
  config.enable_clock_snapshots = false;
  config.enable_socket_timestamping = true;
  config.probe_participants = {0, 1, 2, 3};
  config.has_probe_senders = true;
  config.timestamp_sync_timeout = absl::Seconds(5);
  
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
  int rank = absl::GetFlag(FLAGS_rank);
  int num_nodes = absl::GetFlag(FLAGS_num_nodes);
  int duration_sec = absl::GetFlag(FLAGS_duration_sec);
  std::string addresses_str = absl::GetFlag(FLAGS_addresses);
  std::string topology = absl::GetFlag(FLAGS_topology);
  
  // Validate rank
  if (rank < 0 || rank >= num_nodes) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid rank: ", rank, " (must be 0 to ", num_nodes - 1, ")"));
  }
  
  // Parse addresses
  if (addresses_str.empty()) {
    return absl::InvalidArgumentError("--addresses flag is required");
  }
  std::vector<std::string> addresses = absl::StrSplit(addresses_str, ',');
  
  // Validate addresses count
  if (addresses.size() != num_nodes) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of addresses (", addresses.size(), 
                    ") does not match num_nodes (", num_nodes, ")"));
  }
  
  LOG(INFO) << "Starting standalone network probe test";
  LOG(INFO) << "Rank: " << rank << " / " << num_nodes;
  LOG(INFO) << "Topology: " << topology;
  
  // Create test configuration
  auto config = CreateTestConfig(rank, num_nodes, addresses, topology);
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
  if (config.neighbors.size() > 0 || config.in_neighbors.size() > 0) {
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
  PrintResults(rank);
  
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


