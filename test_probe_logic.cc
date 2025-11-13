// Standalone test program for network probe logic
// Build: g++ -std=c++17 -o test_probe test_probe_logic.cc -I. -lpthread
//
// This lets you test port assignment logic without rebuilding XLA

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

struct EdgePorts {
  uint16_t dst_listen_port;
  uint16_t src_response_port;
};

// Simulate the master's port assignment
std::map<std::string, EdgePorts> AssignPorts(
    const std::vector<std::pair<int, int>>& edges, int num_nodes) {
  constexpr uint16_t kBasePort = 20000;
  constexpr uint16_t kPortsPerNode = 100;
  
  std::vector<std::set<uint16_t>> used_ports(num_nodes);
  std::map<std::string, EdgePorts> result;
  
  for (auto [src, dst] : edges) {
    // Assign dst_listen_port
    uint16_t dst_base = kBasePort + dst * kPortsPerNode;
    uint16_t dst_listen = dst_base;
    while (used_ports[dst].count(dst_listen)) {
      dst_listen++;
      if (dst_listen >= dst_base + kPortsPerNode) {
        throw std::runtime_error("Node ran out of ports");
      }
    }
    used_ports[dst].insert(dst_listen);
    
    // Assign src_response_port
    uint16_t src_base = kBasePort + src * kPortsPerNode;
    uint16_t src_response = src_base;
    while (used_ports[src].count(src_response)) {
      src_response++;
      if (src_response >= src_base + kPortsPerNode) {
        throw std::runtime_error("Node ran out of ports");
      }
    }
    used_ports[src].insert(src_response);
    
    std::string key = "probe_edge:" + std::to_string(src) + "->" + std::to_string(dst);
    result[key] = {dst_listen, src_response};
    
    std::cout << "Edge " << src << "->" << dst << ": "
              << "dst_listen=" << dst_listen << ", src_response=" << src_response << "\n";
  }
  
  return result;
}

int main() {
  std::cout << "Testing centralized port assignment...\n\n";
  
  // Test sparse graph: 0->1, 1->2, 2->0 (ring)
  std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}};
  
  try {
    auto ports = AssignPorts(edges, 3);
    
    std::cout << "\n✅ All ports assigned successfully!\n";
    std::cout << "Total unique edge keys: " << ports.size() << "\n";
    
    // Verify node 1's ports
    std::cout << "\nNode 1 port usage:\n";
    if (ports.count("probe_edge:0->1")) {
      auto& p = ports["probe_edge:0->1"];
      std::cout << "  IN from 0: listen on " << p.dst_listen_port << "\n";
    }
    if (ports.count("probe_edge:1->2")) {
      auto& p = ports["probe_edge:1->2"];
      std::cout << "  OUT to 2: receive responses on " << p.src_response_port << "\n";
    }
    
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "❌ Error: " << e.what() << "\n";
    return 1;
  }
}





