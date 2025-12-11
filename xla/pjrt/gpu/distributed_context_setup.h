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

#ifndef XLA_PJRT_GPU_DISTRIBUTED_CONTEXT_SETUP_H_
#define XLA_PJRT_GPU_DISTRIBUTED_CONTEXT_SETUP_H_

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"

namespace xla {

// Utility class for setting up distributed profiling context.
// This encapsulates the graph generation, port assignment, and context
// initialization that was previously embedded in se_gpu_pjrt_client.cc.
class DistributedContextSetup {
 public:
  // Initialize distributed profiling context for a node.
  // This exchanges addresses, generates the probe graph, assigns ports,
  // and stores the context in the DistributedProfilerContextManager singleton.
  static absl::Status Initialize(
      int node_id,
      int num_nodes,
      KeyValueStoreInterface* kv_store);

 private:
  // Exchange node addresses via KV store
  static absl::StatusOr<std::vector<std::string>> ExchangeNodeAddresses(
      int node_id, int num_nodes, 
      KeyValueStoreInterface* kv_store);

  // Generate directed random graph for network probing
  // Each node gets 5-10 out-neighbors (no bidirectional edges)
  static absl::StatusOr<std::pair<std::vector<int>, std::vector<int>>> 
  GenerateDirectedNeighbors(
      int node_id, int num_nodes, KeyValueStoreInterface* kv_store);

  // Read edge port assignments from KV store
  static absl::StatusOr<std::map<std::string, std::pair<uint16_t, uint16_t>>> 
  ReadEdgePorts(
      int node_id,
      const std::vector<int>& out_neighbors,
      const std::vector<int>& in_neighbors,
      KeyValueStoreInterface* kv_store);

  // Read list of nodes that participate in probing
  static absl::StatusOr<std::vector<int>> ReadProbeParticipants(
      KeyValueStoreInterface* kv_store);
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_DISTRIBUTED_CONTEXT_SETUP_H_

