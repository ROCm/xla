/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/pjrt/gpu/distributed_context_setup.h"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <set>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "xla/backends/profiler/gpu/distributed_timestamp_sync.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {

absl::StatusOr<std::vector<std::string>> DistributedContextSetup::ExchangeNodeAddresses(
    int node_id, int num_nodes, 
    KeyValueStoreInterface* kv_store) {
  std::vector<std::string> addresses;
  for (int i = 0; i < num_nodes; ++i) {
    auto addr_key = absl::StrCat("node_addresses:", i);
    absl::StatusOr<std::string> output =
        kv_store->TryGet(absl::string_view(addr_key));
    if (!output.ok()) {
      return absl::UnavailableError(
          absl::StrCat("Failed to get address for node ", i));
    }
    addresses.push_back(*output);
  }
  VLOG(1) << "Exchanged node addresses: " << absl::StrJoin(addresses, ", ");
  return addresses;  
}

absl::StatusOr<std::pair<std::vector<int>, std::vector<int>>> 
DistributedContextSetup::GenerateDirectedNeighbors(
    int node_id, int num_nodes, KeyValueStoreInterface* kv_store) {
  // Only master (node 0) generates the full graph
  if (node_id == 0) {
    std::mt19937 rng(12345);  // Fixed seed for reproducibility
    std::vector<absl::flat_hash_set<int>> out_edges(num_nodes);
    std::vector<absl::flat_hash_set<int>> in_edges(num_nodes);
    
    VLOG(1) << "Master node generating directed probe graph for " << num_nodes << " nodes";
    
    // For each node, pick random out-degree in [5, 10] (capped by num_nodes-1)
    for (int src = 0; src < num_nodes; ++src) {
      int max_degree = std::min(num_nodes - 1, 10);
      int min_degree = std::min(num_nodes - 1, 5);
      
      // Skip graph generation for single-node jobs
      if (max_degree == 0) {
        VLOG(1) << "Single-node job detected, skipping probe graph generation";
        break;
      }
      
      std::uniform_int_distribution<> degree_dist(min_degree, max_degree);
      int out_degree = degree_dist(rng);
      
      // Sample out_degree unique targets (excluding self)
      std::vector<int> candidates;
      for (int i = 0; i < num_nodes; ++i) {
        if (i != src) candidates.push_back(i);
      }
      std::shuffle(candidates.begin(), candidates.end(), rng);
      
      // Add edges, avoiding bidirectional conflicts
      for (int i = 0; i < out_degree && i < static_cast<int>(candidates.size()); ++i) {
        int dst = candidates[i];
        // Only add edge if reverse doesn't exist
        if (out_edges[dst].find(src) == out_edges[dst].end()) {
          out_edges[src].insert(dst);
          in_edges[dst].insert(src);
        }
      }
      
      VLOG(2) << "Node " << src << " assigned " << out_edges[src].size() << " out-neighbors";
    }
    
    // Assign ports centrally and store in KV
    constexpr uint16_t kBasePort = 20000;
    constexpr uint16_t kPortsPerNode = 100;
    std::vector<std::set<uint16_t>> used_ports(num_nodes);
    
    // Assign ports for each edge: src->dst
    for (int src = 0; src < num_nodes; ++src) {
      for (int dst : out_edges[src]) {
        uint16_t dst_base = kBasePort + dst * kPortsPerNode;
        uint16_t src_base = kBasePort + src * kPortsPerNode;
        
        // Assign dst_listen_port (port on dst where src sends probes)
        uint16_t dst_listen_port = dst_base;
        while (used_ports[dst].count(dst_listen_port)) {
          dst_listen_port++;
          if (dst_listen_port >= dst_base + kPortsPerNode) {
            return absl::ResourceExhaustedError(
                absl::StrCat("Node ", dst, " ran out of available ports"));
          }
        }
        used_ports[dst].insert(dst_listen_port);
        
        // Assign src_response_port (port on src where dst sends responses)
        uint16_t src_response_port = src_base;
        while (used_ports[src].count(src_response_port)) {
          src_response_port++;
          if (src_response_port >= src_base + kPortsPerNode) {
            return absl::ResourceExhaustedError(
                absl::StrCat("Node ", src, " ran out of available ports"));
          }
        }
        used_ports[src].insert(src_response_port);
        
        // Store edge port assignment in KV
        std::string edge_key = absl::StrCat("probe_edge:", src, "->", dst);
        std::string edge_value = absl::StrCat(dst_listen_port, ",", src_response_port);
        TF_RETURN_IF_ERROR(kv_store->Set(edge_key, edge_value));
        
        VLOG(2) << "Edge " << src << "->" << dst << " assigned ports: "
                << "dst_listen=" << dst_listen_port << ", src_response=" << src_response_port;
      }
    }
    
    // Store neighbor lists (for backward compatibility and easy lookup)
    for (int i = 0; i < num_nodes; ++i) {
      std::string key = absl::StrCat("probe_neighbors:", i);
      std::string in_key = absl::StrCat("backward_neighbors:", i);
      std::vector<std::string> out_neighbor_strs;
      std::vector<std::string> in_neighbor_strs;
      for (int neighbor : out_edges[i]) {
        out_neighbor_strs.push_back(absl::StrCat(neighbor));
      }
      for (int neighbor : in_edges[i]) {
        in_neighbor_strs.push_back(absl::StrCat(neighbor));
      }
      std::string out_value = absl::StrJoin(out_neighbor_strs, ",");
      std::string in_value = absl::StrJoin(in_neighbor_strs, ",");
      TF_RETURN_IF_ERROR(kv_store->Set(key, out_value));
      TF_RETURN_IF_ERROR(kv_store->Set(in_key, in_value));
      VLOG(2) << "Stored neighbors for node " << i << ": out=" << out_value << " in=" << in_value;
    }

    // Persist list of nodes that run probe senders (non-empty out-degree)
    std::vector<std::string> participant_tokens;
    participant_tokens.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      if (!out_edges[i].empty()) {
        participant_tokens.push_back(absl::StrCat(i));
      }
    }
    TF_RETURN_IF_ERROR(
        kv_store->Set("probe_participants",
                      absl::StrJoin(participant_tokens, ",")));
    
    // Master writes a sentinel key to signal graph generation is complete
    TF_RETURN_IF_ERROR(kv_store->Set("probe_graph_ready", absl::StrCat(num_nodes)));
    VLOG(1) << "Master node completed probe graph generation and KV storage";
  }
  
  // All nodes wait for master to signal completion
  VLOG(1) << "Node " << node_id << " waiting for probe graph generation to complete...";
  absl::Duration barrier_timeout = absl::Minutes(2);
  absl::StatusOr<std::string> ready_signal = 
      kv_store->Get("probe_graph_ready", barrier_timeout);
  if (!ready_signal.ok()) {
    return absl::UnavailableError(
        absl::StrCat("Timeout waiting for probe graph generation. "
                     "Master may have failed. Status: ", ready_signal.status().ToString()));
  }
  VLOG(1) << "Node " << node_id << " probe graph is ready, reading neighbor data...";
  
  // All nodes read back their neighbors
  std::string key = absl::StrCat("probe_neighbors:", node_id);
  std::string in_key = absl::StrCat("backward_neighbors:", node_id);
  
  absl::StatusOr<std::string> value = kv_store->TryGet(key);
  absl::StatusOr<std::string> in_value = kv_store->TryGet(in_key);
  
  std::vector<int> neighbors;
  std::vector<int> in_neighbors;
  if (value.ok() && !value->empty()) {
    std::vector<std::string> tokens = absl::StrSplit(*value, ',');
    for (const auto& token : tokens) {
      int neighbor;
      if (absl::SimpleAtoi(token, &neighbor)) {
        neighbors.push_back(neighbor);
      } else {
        return absl::InternalError(
            absl::StrCat("Failed to parse neighbor ID from token: ", token));
      }
    }
  }
  if (in_value.ok() && !in_value->empty()) {
    std::vector<std::string> in_tokens = absl::StrSplit(*in_value, ',');
    for (const auto& token : in_tokens) {
      int neighbor;
      if (absl::SimpleAtoi(token, &neighbor)) {
        in_neighbors.push_back(neighbor);
      } else {
        return absl::InternalError(
            absl::StrCat("Failed to parse in-neighbor ID from token: ", token));
      }
    }
  }

  VLOG(1) << "Node " << node_id << " retrieved " << neighbors.size() 
          << " out-neighbors and " << in_neighbors.size() << " in-neighbors";
  if (neighbors.empty() && in_neighbors.empty()) {
    return absl::InternalError(
        absl::StrCat("Node ", node_id, " has no neighbors (empty or single-node job)"));
  }
  return std::make_pair(neighbors, in_neighbors);
}

absl::StatusOr<std::map<std::string, std::pair<uint16_t, uint16_t>>> 
DistributedContextSetup::ReadEdgePorts(
    int node_id,
    const std::vector<int>& out_neighbors,
    const std::vector<int>& in_neighbors,
    KeyValueStoreInterface* kv_store) {
  std::map<std::string, std::pair<uint16_t, uint16_t>> edge_ports;
  
  // Read port assignments for all OUT-edges
  for (int dst : out_neighbors) {
    std::string edge_key = absl::StrCat("probe_edge:", node_id, "->", dst);
    absl::StatusOr<std::string> port_value = kv_store->TryGet(edge_key);
    if (!port_value.ok()) {
      return absl::UnavailableError(
          absl::StrCat("Failed to get port assignment for edge ", edge_key));
    }
    std::vector<std::string> ports = absl::StrSplit(*port_value, ',');
    if (ports.size() != 2) {
      return absl::InternalError(
          absl::StrCat("Invalid port format for edge ", edge_key, ": ", *port_value));
    }
    uint16_t dst_listen_port, src_response_port;
    if (!absl::SimpleAtoi(ports[0], &dst_listen_port) ||
        !absl::SimpleAtoi(ports[1], &src_response_port)) {
      return absl::InternalError(
          absl::StrCat("Failed to parse ports for edge ", edge_key));
    }
    edge_ports[edge_key] = {dst_listen_port, src_response_port};
  }
  
  // Read port assignments for all IN-edges
  for (int src : in_neighbors) {
    std::string edge_key = absl::StrCat("probe_edge:", src, "->", node_id);
    absl::StatusOr<std::string> port_value = kv_store->TryGet(edge_key);
    if (!port_value.ok()) {
      return absl::UnavailableError(
          absl::StrCat("Failed to get port assignment for edge ", edge_key));
    }
    std::vector<std::string> ports = absl::StrSplit(*port_value, ',');
    if (ports.size() != 2) {
      return absl::InternalError(
          absl::StrCat("Invalid port format for edge ", edge_key, ": ", *port_value));
    }
    uint16_t my_listen_port, src_response_port;
    if (!absl::SimpleAtoi(ports[0], &my_listen_port) ||
        !absl::SimpleAtoi(ports[1], &src_response_port)) {
      return absl::InternalError(
          absl::StrCat("Failed to parse ports for edge ", edge_key));
    }
    edge_ports[edge_key] = {my_listen_port, src_response_port};
  }
  
  return edge_ports;
}

absl::StatusOr<std::vector<int>> DistributedContextSetup::ReadProbeParticipants(
    KeyValueStoreInterface* kv_store) {
  std::vector<int> participants;
  absl::StatusOr<std::string> raw = kv_store->TryGet("probe_participants");
  if (!raw.ok() || raw->empty()) {
    return participants;
  }
  std::vector<std::string> tokens = absl::StrSplit(*raw, ',');
  for (const auto& token : tokens) {
    if (token.empty()) continue;
    int id;
    if (!absl::SimpleAtoi(token, &id)) {
      return absl::InternalError(
          absl::StrCat("Failed to parse probe participant id from token: ", token));
    }
    participants.push_back(id);
  }
  return participants;
}

absl::Status DistributedContextSetup::Initialize(
    int node_id,
    int num_nodes,
    KeyValueStoreInterface* kv_store) {
  if (kv_store == nullptr || num_nodes <= 1) {
    VLOG(1) << "Distributed context setup skipped (single-node or no KV store)";
    return absl::OkStatus();
  }

  VLOG(1) << "Initializing distributed context for node " << node_id;
  
  // Exchange node addresses
  auto addresses_or = ExchangeNodeAddresses(node_id, num_nodes, kv_store);
  if (!addresses_or.ok()) {
    return addresses_or.status();
  }
  std::vector<std::string> addresses = std::move(*addresses_or);
  
  // Generate and distribute directed probe graph
  auto neighbor_pair_or = GenerateDirectedNeighbors(node_id, num_nodes, kv_store);
  if (!neighbor_pair_or.ok()) {
    return neighbor_pair_or.status();
  }
  auto neighbor_pair = std::move(*neighbor_pair_or);
  std::vector<int>& neighbors = neighbor_pair.first;
  std::vector<int>& in_neighbors = neighbor_pair.second;
  
  // Read port assignments from KV store
  auto edge_ports_or = ReadEdgePorts(node_id, neighbors, in_neighbors, kv_store);
  if (!edge_ports_or.ok()) {
    return edge_ports_or.status();
  }
  auto edge_ports = std::move(*edge_ports_or);
  
  auto probe_participants_or = ReadProbeParticipants(kv_store);
  if (!probe_participants_or.ok()) {
    return probe_participants_or.status();
  }
  std::vector<int> probe_participants = std::move(*probe_participants_or);
  
  // Read probe config from env
  uint64_t probe_cadence_us = 800;
  if (const char* env_val = std::getenv("XLA_PROBE_CADENCE_US")) {
    (void)absl::SimpleAtoi(env_val, &probe_cadence_us);
  }
  uint64_t probe_window_s = 4;
  if (const char* env_val = std::getenv("XLA_PROBE_WINDOW_S")) {
    (void)absl::SimpleAtoi(env_val, &probe_window_s);
  }
  bool enable_probe_export = true;
  if (const char* env_val = std::getenv("XLA_ENABLE_PROBE_EXPORT")) {
    enable_probe_export = (std::string(env_val) != "0" && std::string(env_val) != "false");
  }
  bool enable_clock_snapshots = false;
  if (const char* env_val = std::getenv("XLA_ENABLE_CLOCK_SNAPSHOTS")) {
    enable_clock_snapshots = (std::string(env_val) == "1" || std::string(env_val) == "true");
  }
  std::string graph_policy = "random_graph";
  if (const char* env_val = std::getenv("XLA_GRAPH_POLICY")) {
    graph_policy = env_val;
  }
  std::string output_dir = "/tmp/xla_dist_prof";
  if (const char* env_val = std::getenv("XLA_DIST_PROF_OUTPUT_DIR")) {
    output_dir = env_val;
  }
  
  // Build complete context with all fields
  using profiler::DistributedProfilerContext;
  DistributedProfilerContext dist_ctx;
  dist_ctx.node_id = node_id;
  dist_ctx.num_nodes = num_nodes;
  dist_ctx.node_addresses = addresses;
  dist_ctx.enable_socket_timestamping = true;
  dist_ctx.timestamp_sync_timeout = absl::Seconds(5);
  dist_ctx.neighbors = neighbors;
  dist_ctx.in_neighbors = in_neighbors;
  dist_ctx.edge_ports = edge_ports;
  dist_ctx.probe_cadence_us = probe_cadence_us;
  dist_ctx.probe_window_s = probe_window_s;
  dist_ctx.enable_probe_export = enable_probe_export;
  dist_ctx.enable_clock_snapshots = enable_clock_snapshots;
  dist_ctx.graph_policy = graph_policy;
  dist_ctx.output_dir = output_dir;
  dist_ctx.probe_participants = probe_participants;
  dist_ctx.has_probe_senders = !neighbors.empty();
  
  // Store in singleton
  profiler::DistributedProfilerContextManager::Get().SetDistributedContext(dist_ctx);
  
  VLOG(1) << "Distributed context stored with " << neighbors.size() 
          << " out-neighbors and " << in_neighbors.size() << " in-neighbors";
  
  return absl::OkStatus();
}

}  // namespace xla

