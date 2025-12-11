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

#ifndef XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_PROFILER_CONTEXT_H_
#define XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_PROFILER_CONTEXT_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace xla {
namespace profiler {

// Configuration context for distributed profiling across multiple nodes.
// This struct holds all the configuration needed for multi-node timestamp
// synchronization, network probing, and probe data export.
//
// Typical usage:
//   1. PJRT layer populates this during client initialization
//   2. Stored in DistributedProfilerContextManager singleton
//   3. Accessed by profiler components (RocmCollector, NetworkProbeManager, etc.)
//
// Configuration sources (in order of precedence):
//   1. Environment variables (XLA_PROBE_CADENCE_US, XLA_DIST_PROF_OUTPUT_DIR, etc.)
//   2. Config file (via XLA_DIST_PROF_CONFIG env var)
//   3. Default values
struct DistributedProfilerContext {
  // === Node Identity ===
  int node_id = -1;                          // Current node's index
  int num_nodes = 1;                         // Total number of nodes
  std::vector<std::string> node_addresses;   // e.g., ["192.168.1.10:5000", ...]

  // === Socket Configuration ===
  bool enable_socket_timestamping = false;   // Enable SO_TIMESTAMPING
  absl::Duration timestamp_sync_timeout = absl::Seconds(5);

  // === Probe Configuration ===
  uint64_t probe_cadence_us = 800;           // Time between probes (µs)
  uint64_t probe_window_s = 4;               // Window duration (s)
  bool enable_probe_export = true;           // Export probe data to files
  bool enable_clock_snapshots = false;       // Enable clock snapshot collection
  std::string graph_policy = "random_graph"; // Probe graph topology policy

  // === Neighbor Configuration (populated by PJRT layer) ===
  std::vector<int> neighbors;                // Out-neighbors for probing
  std::vector<int> in_neighbors;             // In-neighbors (nodes that probe us)
  std::vector<int> probe_participants;       // Nodes that run ProbeSender
  bool has_probe_senders = false;            // Does this node send probes?

  // === Master Sync Configuration ===
  bool enable_master_sync = true;
  int master_node_id = 0;
  uint16_t master_control_port = 36000;
  uint16_t master_sync_port = 37000;

  // === Port Assignments ===
  // Centrally allocated by master node during initialization.
  // Key format: "src->dst", Value: (dst_listen_port, src_response_port)
  std::map<std::string, std::pair<uint16_t, uint16_t>> edge_ports;

  // === Output Configuration ===
  std::string output_dir = "/tmp/xla_dist_prof";

  // === Runtime Metadata (populated by profiler) ===
  std::optional<uint64_t> collector_start_walltime_ns;
  std::optional<uint64_t> collector_start_gpu_ns;
};

// Singleton manager for DistributedProfilerContext.
// Provides thread-safe access to the distributed profiling configuration.
//
// Lifecycle:
//   1. PJRT client calls SetDistributedContext() during initialization
//   2. Profiler components call GetDistributedContext() when needed
//   3. Context is immutable after being set (subsequent Set calls are ignored)
class DistributedProfilerContextManager {
 public:
  // Get the singleton instance
  static DistributedProfilerContextManager& Get();

  // Set the distributed context (called from BuildDistributedDevices)
  // Can only be called once; subsequent calls are ignored with a warning
  void SetDistributedContext(const DistributedProfilerContext& ctx);

  // Get the distributed context if available
  std::optional<DistributedProfilerContext> GetDistributedContext() const;

  // Check if context is available
  bool HasDistributedContext() const;

  // Reset context (for testing only)
  void ResetContext();

 private:
  DistributedProfilerContextManager() = default;
  ~DistributedProfilerContextManager() = default;

  // Prevent copy/move
  DistributedProfilerContextManager(const DistributedProfilerContextManager&) = delete;
  DistributedProfilerContextManager& operator=(const DistributedProfilerContextManager&) = delete;

  mutable absl::Mutex mu_;
  std::optional<DistributedProfilerContext> context_ ABSL_GUARDED_BY(mu_);
  bool context_set_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_PROFILER_CONTEXT_H_

