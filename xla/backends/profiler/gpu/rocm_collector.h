/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

inline std::string ToXStat(const KernelDetails& kernel_info,
                           double occupancy_pct) {
  uint32_t grid_x = kernel_info.workgroup_x != 0
                        ? kernel_info.grid_x / kernel_info.workgroup_x
                        : 0,
           grid_y = kernel_info.workgroup_y != 0
                        ? kernel_info.grid_y / kernel_info.workgroup_y
                        : 0,
           grid_z = kernel_info.workgroup_z != 0
                        ? kernel_info.grid_z / kernel_info.workgroup_z
                        : 0;

  return absl::StrCat(" grid:", grid_x, ",", grid_y, ",", grid_z,
                      " block:", kernel_info.workgroup_x, ",",
                      kernel_info.workgroup_y, ",", kernel_info.workgroup_z,
                      " private_mem:", kernel_info.private_segment_size,
                      " group_mem:", kernel_info.group_segment_size,
                      " occ_pct:", occupancy_pct);
}

class RocmTraceCollector {
 public:
  explicit RocmTraceCollector(const RocmTraceCollectorOptions& options)
      : options_(options) {}
  virtual ~RocmTraceCollector() {}

  virtual void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32_t num_events) = 0;
  virtual void Flush() = 0;
  virtual void Export(tsl::profiler::XSpace* space) = 0;
  virtual void SetScopeRangeIdTree(ScopeRangeIdTree tree) {}

 protected:
  RocmTraceCollectorOptions options_;

 public:
  // Disable copy and move.
  RocmTraceCollector(const RocmTraceCollector&) = delete;
  RocmTraceCollector& operator=(const RocmTraceCollector&) = delete;
};

class PerDeviceCollector {
 public:
  void Export(uint64_t start_walltime_ns, uint64_t start_gputime_ns,
              uint64_t end_gputime_ns,
              tsl::profiler::XPlaneBuilder* device_plane,
              tsl::profiler::XPlaneBuilder* host_plane);

  PerDeviceCollector() = default;

  void AddEvent(RocmTracerEvent&& event);
  void GetDeviceCapabilities(int32_t device_ordinal,
                             tsl::profiler::XPlaneBuilder* device_plane,
                             uint32_t simd_per_cu, uint32_t max_waves_per_simd,
                             uint32_t gfx_target_version);

 private:
  double ComputeTheoreticalOccupancy(const KernelDetails& ki);
  void CreateXEvent(const RocmTracerEvent& event,
                    tsl::profiler::XPlaneBuilder* plane, uint64_t start_gpu_ns,
                    uint64_t end_gpu_ns, tsl::profiler::XLineBuilder* line);
  void SortByStartTime();
  bool IsHostEvent(const RocmTracerEvent& event, int64_t* line_id);

 private:
  absl::Mutex events_mutex_;
  std::vector<RocmTracerEvent> events_ ABSL_GUARDED_BY(events_mutex_);
  hipDeviceProp_t device_properties_;
  struct OccupancyKey {
    uint32_t arch_vgpr_count;
    uint32_t sgpr_count;
    uint32_t group_segment_size;
    uint32_t block_size;

    template <typename H>
    friend H AbslHashValue(H h, const OccupancyKey& k) {
      return H::combine(std::move(h), k.arch_vgpr_count, k.sgpr_count,
                        k.group_segment_size, k.block_size);
    }
    friend bool operator==(const OccupancyKey& a, const OccupancyKey& b) {
      return a.arch_vgpr_count == b.arch_vgpr_count &&
             a.sgpr_count == b.sgpr_count &&
             a.group_segment_size == b.group_segment_size &&
             a.block_size == b.block_size;
    }
  };

  absl::flat_hash_map<OccupancyKey, double> theoretical_occupancy_cache_;
  uint32_t simd_per_cu_ = 0;
  uint32_t max_waves_per_simd_ = 0;
  uint32_t gfx_target_version_ = 0;
};  // PerDeviceCollector

class RocmTraceCollectorImpl : public RocmTraceCollector {
 public:
  RocmTraceCollectorImpl(const RocmTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : RocmTraceCollector(options),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gputime_ns_(start_gputime_ns),
        num_gpus_(options.num_gpus) {}

  void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) override;
  void Flush() override;
  void Export(tsl::profiler::XSpace* space) override;
  void SetScopeRangeIdTree(ScopeRangeIdTree tree) override {
    scope_range_id_tree_ = std::move(tree);
  }

  void OnEventsDropped(const std::string& reason,
                       uint32_t correlation_id) override {
    VLOG(2) << "RocmTracerEvent dropped (correlation_id=" << correlation_id
            << ",) : " << reason << ".";
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64_t start_walltime_ns_;
  uint64_t start_gputime_ns_;
  int num_gpus_;

  absl::Mutex event_maps_mutex_;
  absl::flat_hash_map<uint32_t, RocmTracerEvent> api_events_map_
      ABSL_GUARDED_BY(event_maps_mutex_);

  /* Some apis such as MEMSETD32 (based on an observation with ResNet50),
   trigger multiple HIP ops domain activities. We keep them in a vector and
   merge them with api activities at flush time.
 */
  absl::flat_hash_map<uint32_t, std::vector<RocmTracerEvent>>
      activity_ops_events_map_ ABSL_GUARDED_BY(event_maps_mutex_);
  // This is for the APIs that we track because we need some information from
  // them to populate the corresponding activity that we actually track.
  absl::flat_hash_map<uint32_t, RocmTracerEvent> auxiliary_api_events_map_
      ABSL_GUARDED_BY(event_maps_mutex_);

  std::vector<RocmTracerEvent> ApiActivityInfoExchange()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(event_maps_mutex_);

  void ExportScopeRangeIdTree(tsl::profiler::XSpace* space);

  absl::node_hash_map<uint32_t, PerDeviceCollector> per_device_collector_;
  ScopeRangeIdTree scope_range_id_tree_;
};  // RocmTraceCollectorImpl

std::unique_ptr<RocmTraceCollector> CreateRocmCollector(
    const RocmTraceCollectorOptions& options, uint64_t start_walltime_ns,
    uint64_t start_gputime_ns);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_
