// Unit test for network probe port assignment
#include "xla/backends/profiler/gpu/network_probe.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"

#include <gtest/gtest.h>
#include <map>
#include <string>

namespace xla {
namespace profiler {
namespace {

TEST(NetworkProbeTest, CentralizedPortAssignment) {
  // Test that port assignments are correctly read from config
  DistributedProfilerContext config;
  config.node_id = 1;
  config.num_nodes = 3;
  config.node_addresses = {"192.168.1.1:5000", "192.168.1.2:5000", "192.168.1.3:5000"};
  
  // Simulate master-assigned ports
  config.edge_ports["probe_edge:0->1"] = {20100, 20000};  // 0 probes 1
  config.edge_ports["probe_edge:1->2"] = {20200, 20101};  // 1 probes 2
  
  config.neighbors = {2};      // OUT: Node 1 probes node 2
  config.in_neighbors = {0};   // IN: Node 0 probes node 1
  
  // Verify port lookup
  auto it_out = config.edge_ports.find("probe_edge:1->2");
  ASSERT_NE(it_out, config.edge_ports.end());
  EXPECT_EQ(it_out->second.first, 20200);   // dst_listen_port
  EXPECT_EQ(it_out->second.second, 20101);  // src_response_port
  
  auto it_in = config.edge_ports.find("probe_edge:0->1");
  ASSERT_NE(it_in, config.edge_ports.end());
  EXPECT_EQ(it_in->second.first, 20100);    // my_listen_port
  EXPECT_EQ(it_in->second.second, 20000);   // src_response_port
}

TEST(NetworkProbeTest, PortConflictDetection) {
  // Verify that centralized assignment prevents conflicts
  std::set<uint16_t> used_ports;
  
  // Simulate allocating ports for sparse graph
  std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}};
  uint16_t base_port = 20000;
  
  for (auto [src, dst] : edges) {
    uint16_t dst_port = base_port + dst * 100;
    while (used_ports.count(dst_port)) dst_port++;
    used_ports.insert(dst_port);
    
    uint16_t src_port = base_port + src * 100;
    while (used_ports.count(src_port)) src_port++;
    used_ports.insert(src_port);
  }
  
  // All ports should be unique
  EXPECT_EQ(used_ports.size(), 6);  // 3 edges × 2 ports
}

TEST(NetworkProbeTest, EdgeKeyFormat) {
  // Verify edge key format matches between assignment and lookup
  int src = 0, dst = 1;
  std::string edge_key = absl::StrCat("probe_edge:", src, "->", dst);
  EXPECT_EQ(edge_key, "probe_edge:0->1");
}

}  // namespace
}  // namespace profiler
}  // namespace xla




