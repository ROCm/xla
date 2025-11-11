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

#ifndef XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_TIMESTAMP_SYNC_H_
#define XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_TIMESTAMP_SYNC_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"

namespace xla::profiler {

// Forward declarations
class NetworkProbeManager;

// ============================================================================
// SINGLETON: DistributedProfilerContextManager
// ============================================================================
// Thread-safe singleton to store and manage distributed profiler context.
// This is initialized EARLY in BuildDistributedDevices() when addresses are
// exchanged via KV store, then accessed later when ROCM profiler starts.
//
// Usage Flow:
//   1. BuildDistributedDevices() exchanges addresses via KV store
//   2. BuildDistributedDevices() calls SetDistributedContext() on singleton
//   3. Later, CreateGpuTracer() or rocm_collector initializer calls
//      GetDistributedContext() to access the stored context
//   4. DistributedTimestampSynchronizer is initialized with this context

class DistributedProfilerContextManager {
 public:
  // Get the singleton instance
  static DistributedProfilerContextManager& Get();

  // Set the distributed context (called from BuildDistributedDevices)
  // Can only be called once; subsequent calls are ignored
  void SetDistributedContext(const DistributedProfilerContext& ctx);

  // Get the distributed context if available
  std::optional<DistributedProfilerContext> GetDistributedContext() const;

  // Check if context is available
  bool HasDistributedContext() const;

  // Reset context (for testing)
  void ResetContext();

 private:
  DistributedProfilerContextManager() = default;
  ~DistributedProfilerContextManager() = default;

  // Prevent copy/move
  DistributedProfilerContextManager(const DistributedProfilerContextManager&) =
      delete;
  DistributedProfilerContextManager& operator=(
      const DistributedProfilerContextManager&) = delete;

  mutable absl::Mutex mu_;
  std::optional<DistributedProfilerContext> context_ ABSL_GUARDED_BY(mu_);
  bool context_set_ ABSL_GUARDED_BY(mu_) = false;
};

// ============================================================================
// Struct representing a synchronized timestamp from a specific node
struct SyncedTimestamp {
  int node_id;
  uint64_t local_ns;        // Local timestamp (GPU clock)
  uint64_t wall_ns;         // Wall clock timestamp
  uint64_t socket_ts_ns;    // Socket-based hardware timestamp (SOF_TIMESTAMPING)
};

// ============================================================================
// DistributedTimestampSynchronizer
// ============================================================================
// Handles multi-node timestamp synchronization using socket communication
// with SOF_TIMESTAMPING_RX_SOFTWARE for precision.
//
// Architecture:
// 1. Each node establishes listening socket on its address
// 2. Nodes exchange timestamps via sockets with hardware timestamping enabled
// 3. Clock offsets are computed and stored locally
// 4. GPU timestamps are adjusted during profiling to align with global clock

class DistributedTimestampSynchronizer {
 public:
  explicit DistributedTimestampSynchronizer(
      const DistributedProfilerContext& config);
  ~DistributedTimestampSynchronizer();

  // Initialize synchronization: establish connections and exchange initial timestamps
  absl::Status Initialize();

  // Get the clock offset for this node (to convert local to global time)
  // Returns: global_time_ns = local_time_ns + GetClockOffset()
  int64_t GetClockOffset() const;

  // Exchange timestamps with all other nodes
  // This should be called periodically to refine clock synchronization
  absl::Status SyncTimestamps();

  // NEW: Start the network probing for clock synchronization
  absl::Status StartProbing();

  // Convert local GPU timestamp to synchronized global timestamp
  uint64_t LocalToGlobal(uint64_t local_ns) const;

  // Get detailed sync information (for debugging)
  std::vector<SyncedTimestamp> GetLastSyncTimestamps() const;

  // Export probe data (CSV files with timing measurements)
  absl::Status ExportProbeData();

 private:
  DistributedProfilerContext config_;
  int64_t clock_offset_ns_ = 0;  // Offset to apply to local timestamps
  std::vector<SyncedTimestamp> last_sync_timestamps_;

  // NEW: Probe management
  std::unique_ptr<NetworkProbeManager> probe_manager_;

  // Helper: setup socket with SOF_TIMESTAMPING_RX_SOFTWARE
  absl::Status CreateTimestampSocket();

  // Helper: exchange timestamps with a remote node
  absl::Status ExchangeTimestampWithNode(int remote_node_id, int socket_fd);
};

}  // namespace xla::profiler

#endif  // XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_TIMESTAMP_SYNC_H_
