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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_UTILS_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_UTILS_H_

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <unistd.h>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/container/node_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "xla/tsl/profiler/utils/time_utils.h"

#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "absl/time/time.h"

namespace xla {
namespace profiler {

struct MemcpyDetails {
  // The amount of data copied for memcpy events.
  size_t num_bytes;
  // The destination device for peer-2-peer communication (memcpy). The source
  // device is implicit: it's the current device.
  uint32_t destination;
  // Whether or not the memcpy is asynchronous.
  bool async;
};

struct MemAllocDetails {
  // The amount of data requested for cudaMalloc events.
  uint64_t num_bytes;
};

struct MemsetDetails {
  // The number of memory elements getting set
  size_t num_bytes;
  // Whether or not the memset is asynchronous.
  bool async;
};

struct KernelDetails {
  // The amount of private memory used by kernel, 
  // number of register per thread (register spillage if > 0)
  uint32_t private_segment_size;
  // The amount of shared memory (SMEM)
  uint32_t group_segment_size;
  // X-dimension of a workgroup (grid.x*block.x)
  uint32_t workgroup_x;
  // Y-dimension of a workgroup (grid.x*block.x)
  uint32_t workgroup_y;
  // Z-dimension of a workgroup (grid.x*block.x)
  uint32_t workgroup_z;
  // X-dimension of a grid.
  uint32_t grid_x;
  // Y-dimension of a grid.
  uint32_t grid_y;
  // Z-dimension of a grid.
  uint32_t grid_z;

  // kernel address. Used for calculating core occupancy
  void* func_ptr;
};

enum class RocmTracerEventType {
  Unsupported = 0,
  Kernel,
  MemcpyH2D,
  MemcpyD2H,
  MemcpyD2D,
  MemcpyP2P,
  MemcpyOther,
  MemoryAlloc,
  MemoryFree,
  Memset,
  Synchronization,
  Generic,
};

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type);

enum class RocmTracerEventSource {
  Invalid = 0,
  ApiCallback,
  Activity,
};

const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source);

enum class RocmTracerEventDomain {
  InvalidDomain = 0,
  HIP_API,
  HIP_OPS,
};

const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain);

// RocmTracerSyncTypes forward declaration
enum class RocmTracerSyncTypes;

struct SynchronizationDetails {
  RocmTracerSyncTypes sync_type;
};

struct RocmTracerEvent {
  static constexpr uint32_t kInvalidDeviceId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidThreadId =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint32_t kInvalidCorrelationId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidStreamId =
      std::numeric_limits<uint64_t>::max();
  RocmTracerEventType type;
  RocmTracerEventSource source = RocmTracerEventSource::Invalid;
  RocmTracerEventDomain domain;
  std::string name;
  // This points to strings in AnnotationMap, which should outlive the point
  // where serialization happens.
  absl::string_view annotation;
  absl::string_view roctx_range;
  uint64_t start_time_ns = 0;
  uint64_t end_time_ns = 0;
  uint32_t device_id = kInvalidDeviceId;
  uint32_t correlation_id = kInvalidCorrelationId;
  uint64_t thread_id = kInvalidThreadId;
  uint64_t stream_id = kInvalidStreamId;

  union {
    MemcpyDetails memcpy_info;                    // If type == Memcpy*
    MemsetDetails memset_info;                    // If type == Memset*
    MemAllocDetails memalloc_info;                // If type == MemoryAlloc
    KernelDetails kernel_info;                    // If type == Kernel
    SynchronizationDetails synchronization_info;  // If type == Synchronization
  };
};

// Distributed profiling context for multi-node timestamp synchronization
struct DistributedProfilerContext {
  // Network addresses of all nodes in the distributed job
  std::vector<std::string> node_addresses;  // e.g., ["192.168.1.10:5000", "192.168.1.11:5000"]
  
  // Current node's index in the distributed setup
  int node_id = -1;
  
  // Total number of nodes
  int num_nodes = 1;
  
  // Socket configuration for timestamp exchange
  // Enable SOF_TIMESTAMPING_RX_SOFTWARE for hardware clock synchronization
  bool enable_socket_timestamping = false;
  
  // Timeout for timestamp exchange operations
  absl::Duration timestamp_sync_timeout = absl::Seconds(5);

  // Probe configuration for network synchronization
  uint64_t probe_cadence_us = 800;
  uint64_t probe_window_s = 4;
  bool enable_probe_export = true;
  bool enable_clock_snapshots = false;
  std::string graph_policy = "random_graph";
  std::vector<int> neighbors;  // Resolved neighbors for probing
  std::vector<int> in_neighbors;  // Resolved in-neighbors for probing
  std::vector<int> probe_participants;  // Nodes expected to run ProbeSender
  bool has_probe_senders = false;
  bool enable_master_sync = true;
  int master_node_id = 0;
  uint16_t master_control_port = 36000;
  uint16_t master_sync_port = 37000;
  
  // Port assignments for directed edges: key="src->dst", value=(dst_listen_port, src_response_port)
  std::map<std::string, std::pair<uint16_t, uint16_t>> edge_ports;

  // Optional profiling-session metadata populated by the collector
  std::optional<uint64_t> collector_start_walltime_ns;
  std::optional<uint64_t> collector_start_gpu_ns;
};

struct RocmTraceCollectorOptions {
  // Maximum number of events to collect from callback API; if -1, no limit.
  // if 0, the callback API is enabled to build a correlation map, but no
  // events are collected.
  uint64_t max_callback_api_events;
  // Maximum number of events to collect from activity API; if -1, no limit.
  uint64_t max_activity_api_events;
  // Maximum number of annotation strings that we can accommodate.
  uint64_t max_annotation_strings;
  // Number of GPUs involved.
  uint32_t num_gpus;
  
  // NEW: Distributed profiling context
  std::optional<DistributedProfilerContext> distributed_context;

  // Snapshot period in milliseconds
  int snapshot_period_ms = 4000;
};

class AnnotationMap {
 public:
  explicit AnnotationMap(uint64_t max_size) : max_size_(max_size) {}
  void Add(uint32_t correlation_id, const std::string& annotation);
  absl::string_view LookUp(uint32_t correlation_id);

 private:
  struct AnnotationMapImpl {
    // The population/consumption of annotations might happen from multiple
    // callback/activity api related threads.
    absl::Mutex mutex;
    // Annotation tends to be repetitive, use a hash_set to store the strings,
    // an use the reference to the string in the map.
    absl::node_hash_set<std::string> annotations;
    absl::flat_hash_map<uint32_t, absl::string_view> correlation_map;
  };
  const uint64_t max_size_;
  AnnotationMapImpl map_;

 public:
  // Disable copy and move.
  AnnotationMap(const AnnotationMap&) = delete;
  AnnotationMap& operator=(const AnnotationMap&) = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_UTILS_H_
