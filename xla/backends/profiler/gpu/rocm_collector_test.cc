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

#include "xla/backends/profiler/gpu/rocm_collector.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace test {

using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::StatType;
using tsl::profiler::XSpace;

TEST(RocmCollectorTest, TestAddKernelEventAndExport) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 2000;

  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  constexpr uint32_t kCorrelationId = 42;
  constexpr uint64_t kStartTimeNs = 3000;
  constexpr uint64_t kEndTimeNs = 4000;

  // === 1. Add API Callback Event ===
  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::Kernel;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "test_rocm_kernel";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  api_event.kernel_info = KernelDetails{};
  api_event.kernel_info.private_segment_size = 32;
  api_event.kernel_info.group_segment_size = 1024;
  api_event.kernel_info.workgroup_x = 256;
  api_event.kernel_info.workgroup_y = 1;
  api_event.kernel_info.workgroup_z = 1;
  api_event.kernel_info.grid_x = 100;
  api_event.kernel_info.grid_y = 1;
  api_event.kernel_info.grid_z = 1;
  api_event.kernel_info.num_regs = 32;

  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  // === 2. Add Activity Event ===
  RocmTracerEvent activity_event;
  activity_event.type = RocmTracerEventType::Kernel;
  activity_event.source = RocmTracerEventSource::Activity;
  activity_event.domain = RocmTracerEventDomain::HIP_OPS;
  activity_event.name = "test_rocm_kernel";
  activity_event.correlation_id = kCorrelationId;
  activity_event.start_time_ns = kStartTimeNs;
  activity_event.end_time_ns = kEndTimeNs;
  activity_event.device_id = 100;
  activity_event.stream_id = 123;

  collector.AddEvent(std::move(activity_event), /*is_auxiliary=*/false);

  // === 3. Finalize and Export ===
  collector.Flush();

  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  // === 4. Check results ===
  ASSERT_GE(space.planes_size(), 1);
  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);

  ASSERT_GT(gpu_plane->lines_size(), 0);
  const auto& line = gpu_plane->lines(0);
  ASSERT_GT(line.events_size(), 0);

  const auto& event = line.events(0);
  EXPECT_EQ(event.offset_ps(), (kStartTimeNs - kStartGpuTimeNs) * 1000);
  EXPECT_EQ(event.duration_ps(), (kEndTimeNs - kStartTimeNs) * 1000);
  EXPECT_EQ(gpu_plane->event_metadata().at(event.metadata_id()).name(),
            "test_rocm_kernel");
}

// Regression test for the .front()-only iteration bug in
// ApiActivityInfoExchange. When N activity events share one
// correlation_id (the rocprofiler-sdk pattern for hipGraphLaunch-replayed
// kernels), all N must reach the exported XPlane, not just the first.
TEST(RocmCollectorTest, MultipleActivitiesPerCorrelationIdAllExported) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 2000;
  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  // Single correlation_id shared by all events -- mirrors a hipGraphLaunch
  // that replays a captured graph: one API call, many kernel-dispatch
  // records emitted by rocprofiler-sdk under the same correlation_id.
  constexpr uint32_t kCorrelationId = 7;
  constexpr uint32_t kDeviceId = 100;
  constexpr uint64_t kStreamId = 123;

  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::Kernel;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "hipGraphLaunch";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  api_event.kernel_info = KernelDetails{};
  api_event.kernel_info.num_regs = 32;
  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  // Three GPU activity records, same correlation_id, same stream (so
  // they land on the same XLine), distinct names and timestamps.
  struct ActivityShape {
    const char* name;
    uint64_t start_ns;
    uint64_t end_ns;
  };
  constexpr ActivityShape kActivities[] = {
      {"kernel_a", 3000, 3500},
      {"kernel_b", 3500, 4000},
      {"kernel_c", 4000, 4500},
  };
  for (const auto& shape : kActivities) {
    RocmTracerEvent activity;
    activity.type = RocmTracerEventType::Kernel;
    activity.source = RocmTracerEventSource::Activity;
    activity.domain = RocmTracerEventDomain::HIP_OPS;
    activity.name = shape.name;
    activity.correlation_id = kCorrelationId;
    activity.start_time_ns = shape.start_ns;
    activity.end_time_ns = shape.end_ns;
    activity.device_id = kDeviceId;
    activity.stream_id = kStreamId;
    collector.AddEvent(std::move(activity), /*is_auxiliary=*/false);
  }

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);

  // Pre-fix (.front()-only) would emit just one event here. The fix
  // iterates the entire vector, so all three activity records must
  // appear on the stream line.
  size_t total_kernel_events = 0;
  absl::flat_hash_set<std::string> seen_names;
  for (const auto& line : gpu_plane->lines()) {
    if (line.id() != static_cast<int64_t>(kStreamId)) {
      continue;
    }
    total_kernel_events += line.events_size();
    for (const auto& ev : line.events()) {
      seen_names.insert(
          gpu_plane->event_metadata().at(ev.metadata_id()).name());
    }
  }

  EXPECT_EQ(total_kernel_events, 3u)
      << "Expected all 3 activity records to be emitted under the same "
         "correlation_id; got "
      << total_kernel_events
      << " (this is the "
         "regression the .front()-only iteration introduced).";
  EXPECT_TRUE(seen_names.contains("kernel_a"));
  EXPECT_TRUE(seen_names.contains("kernel_b"));
  EXPECT_TRUE(seen_names.contains("kernel_c"));
}

// ============================================================================
// Occupancy unit tests
// ============================================================================

// Helper: build a RocmTraceCollectorImpl and inject a paired API + Activity
// kernel event, optionally with GPU agent data for occupancy.
struct OccupancyTestFixture {
  RocmTraceCollectorImpl collector;

  static RocmTraceCollectorOptions MakeOpts() {
    RocmTraceCollectorOptions o;
    o.max_callback_api_events = 100;
    o.max_activity_api_events = 100;
    o.max_annotation_strings = 100;
    o.num_gpus = 1;
    return o;
  }

  OccupancyTestFixture()
      : collector(MakeOpts(), /*start_walltime_ns=*/1000,
                  /*start_gputime_ns=*/2000) {}

  void AddKernelPair(uint32_t num_regs, uint32_t wg_x, uint32_t wg_y,
                     uint32_t wg_z, uint32_t smem, uint64_t start_ns,
                     uint64_t end_ns, uint32_t corr_id = 1) {
    RocmTracerEvent api;
    api.type = RocmTracerEventType::Kernel;
    api.source = RocmTracerEventSource::ApiCallback;
    api.domain = RocmTracerEventDomain::HIP_API;
    api.name = "test_kernel";
    api.correlation_id = corr_id;
    api.thread_id = 1;
    api.kernel_info = KernelDetails{};
    api.kernel_info.num_regs = num_regs;
    api.kernel_info.workgroup_x = wg_x;
    api.kernel_info.workgroup_y = wg_y;
    api.kernel_info.workgroup_z = wg_z;
    api.kernel_info.group_segment_size = smem;
    collector.AddEvent(std::move(api), false);

    RocmTracerEvent act;
    act.type = RocmTracerEventType::Kernel;
    act.source = RocmTracerEventSource::Activity;
    act.domain = RocmTracerEventDomain::HIP_OPS;
    act.name = "test_kernel";
    act.correlation_id = corr_id;
    act.start_time_ns = start_ns;
    act.end_time_ns = end_ns;
    act.device_id = 0;
    act.stream_id = 1;
    collector.AddEvent(std::move(act), false);
  }
};

// When num_regs == 0 the occupancy block must not be written.
TEST(RocmCollectorOccupancyTest, ZeroNumRegsSkipsOccupancyStats) {
  OccupancyTestFixture f;
  f.AddKernelPair(/*num_regs=*/0, 256, 1, 1, 0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  absl::string_view kOccKey =
      GetStatTypeStr(StatType::kTheoreticalOccupancyPct);
  for (const auto& [id, smd] : gpu->stat_metadata()) {
    EXPECT_NE(smd.name(), kOccKey)
        << "kTheoreticalOccupancyPct must not appear when num_regs == 0";
  }
}

// When agent data has not been set (max_waves_per_cu_ == 0), occupancy must
// not be written even if num_regs is non-zero.
TEST(RocmCollectorOccupancyTest, ZeroWavesPerCuSkipsOccupancyStats) {
  OccupancyTestFixture f;
  // num_regs=32 but no agent data injected → max_waves_per_cu_ stays 0.
  f.AddKernelPair(/*num_regs=*/32, 256, 1, 1, 0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  absl::string_view kOccKey =
      GetStatTypeStr(StatType::kTheoreticalOccupancyPct);
  for (const auto& [id, smd] : gpu->stat_metadata()) {
    EXPECT_NE(smd.name(), kOccKey)
        << "kTheoreticalOccupancyPct must not appear when max_waves_per_cu=0";
  }
}

// kKernelDetails stat must always be present for kernel activity events,
// regardless of whether occupancy was computed.
TEST(RocmCollectorOccupancyTest, KernelDetailsAlwaysPresent) {
  OccupancyTestFixture f;
  f.AddKernelPair(/*num_regs=*/0, 64, 1, 1, 512, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  absl::string_view kDetailsKey = GetStatTypeStr(StatType::kKernelDetails);
  bool found_details = false;
  for (const auto& [id, smd] : gpu->stat_metadata()) {
    if (smd.name() == kDetailsKey) {
      found_details = true;
      break;
    }
  }
  EXPECT_TRUE(found_details) << "kKernelDetails stat must be present for "
                                "kernel events regardless of occupancy";
}

// kKernelDetails string must contain "occ_pct:0" when num_regs == 0.
TEST(RocmCollectorOccupancyTest, KernelDetailsContainsZeroOccupancyWhenNoRegs) {
  OccupancyTestFixture f;
  f.AddKernelPair(/*num_regs=*/0, 128, 1, 1, 0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  absl::string_view kDetailsKey = GetStatTypeStr(StatType::kKernelDetails);
  int64_t details_stat_id = -1;
  for (const auto& [id, smd] : gpu->stat_metadata()) {
    if (smd.name() == kDetailsKey) {
      details_stat_id = id;
      break;
    }
  }
  ASSERT_GE(details_stat_id, 0) << "kKernelDetails stat metadata not found";

  bool found_occ_pct_zero = false;
  for (const auto& line : gpu->lines()) {
    for (const auto& ev : line.events()) {
      for (const auto& stat : ev.stats()) {
        if (stat.metadata_id() != details_stat_id) continue;
        int64_t ref = stat.ref_value();
        auto it = gpu->stat_metadata().find(ref);
        if (it == gpu->stat_metadata().end()) continue;
        if (it->second.name().find("occ_pct:0") != std::string::npos) {
          found_occ_pct_zero = true;
        }
      }
    }
  }
  EXPECT_TRUE(found_occ_pct_zero)
      << "KernelDetails string must contain 'occ_pct:0' when num_regs == 0";
}

// RocmDeviceOccupancyParams equality and hashing must work correctly so that
// the occupancy cache deduplicates across identical kernel launches.
TEST(RocmCollectorOccupancyTest, OccupancyParamsCacheKey) {
  RocmDeviceOccupancyParams a{};
  a.num_regs = 32;
  a.block_size = 256;
  a.smem_bytes = 0;
  a.max_waves_per_cu = 32;
  a.wave_front_size = 64;
  a.max_waves_per_simd = 8;
  a.simd_per_cu = 4;
  a.lds_size_bytes = 65536;

  RocmDeviceOccupancyParams b = a;
  EXPECT_EQ(a, b) << "Identical params must compare equal";

  b.num_regs = 64;
  EXPECT_NE(a, b) << "Params with different num_regs must not be equal";

  b = a;
  b.block_size = 128;
  EXPECT_NE(a, b) << "Params with different block_size must not be equal";

  b = a;
  b.max_waves_per_cu = 16;
  EXPECT_NE(a, b) << "Params with different max_waves_per_cu must not be equal";

  b = a;
  b.wave_front_size = 32;
  EXPECT_NE(a, b) << "Params with different wave_front_size must not be equal";
}

// ============================================================================
// Direct formula validation (GetOccupancy is a free function)
// ============================================================================

// gfx942 (MI300/CDNA3): 512 VGPRs/SIMD, 4 SIMDs/CU, 8 waves/SIMD,
// 32 waves/CU, 64 threads/wave, 64 KiB LDS.
// A kernel with 128 VGPRs/thread, block_size=256, no LDS:
//   waves_per_block     = ceil(256/64) = 4
//   waves_per_simd_vgpr = 512 / 128 = 4
//   waves_per_cu_vgpr   = 4 * 4 SIMDs = 16
//   active_waves  = min(16, 32 max) = 16
//   occupancy_pct = 16/32 = 50%
TEST(RocmCollectorOccupancyTest, FormulaGfx942VgprLimited) {
  RocmDeviceOccupancyParams params{};
  params.num_regs = 128;
  params.block_size = 256;
  params.smem_bytes = 0;
  params.max_waves_per_cu = 32;
  params.wave_front_size = 64;
  params.max_waves_per_simd = 8;
  params.simd_per_cu = 4;
  params.lds_size_bytes = 65536;

  OccupancyStats stats = GetOccupancy(params);
  EXPECT_DOUBLE_EQ(stats.occupancy_pct, 50.0);
  EXPECT_EQ(stats.suggested_block_size, 256);
  EXPECT_GT(stats.min_grid_size, 0);
}

// Same GPU, low register pressure (32 VGPRs/thread):
//   waves_per_simd_vgpr = 512 / 32 = 16
//   waves_per_cu_vgpr   = 16 * 4 SIMDs = 64
//   active_waves  = min(64, 32 max) = 32  (hardware-capped)
//   occupancy_pct = 32/32 = 100%
TEST(RocmCollectorOccupancyTest, FormulaGfx942FullOccupancy) {
  RocmDeviceOccupancyParams params{};
  params.num_regs = 32;
  params.block_size = 256;
  params.smem_bytes = 0;
  params.max_waves_per_cu = 32;
  params.wave_front_size = 64;
  params.max_waves_per_simd = 8;
  params.simd_per_cu = 4;
  params.lds_size_bytes = 65536;

  OccupancyStats stats = GetOccupancy(params);
  EXPECT_DOUBLE_EQ(stats.occupancy_pct, 100.0);
  EXPECT_EQ(stats.suggested_block_size, 256);
}

// LDS-limited: 64 KiB LDS, each block uses 32 KiB → only 2 blocks fit.
// 2 blocks × 4 waves/block = 8 active waves out of 32 max → 25%.
TEST(RocmCollectorOccupancyTest, FormulaLdsLimited) {
  RocmDeviceOccupancyParams params{};
  params.num_regs = 16;
  params.block_size = 256;
  params.smem_bytes = 32768;
  params.max_waves_per_cu = 32;
  params.wave_front_size = 64;
  params.max_waves_per_simd = 8;
  params.simd_per_cu = 4;
  params.lds_size_bytes = 65536;

  OccupancyStats stats = GetOccupancy(params);
  EXPECT_DOUBLE_EQ(stats.occupancy_pct, 25.0);
}

// All-zero params must return empty stats, not crash.
TEST(RocmCollectorOccupancyTest, FormulaZeroParamsReturnsEmpty) {
  RocmDeviceOccupancyParams params{};
  OccupancyStats stats = GetOccupancy(params);
  EXPECT_DOUBLE_EQ(stats.occupancy_pct, 0.0);
  EXPECT_EQ(stats.suggested_block_size, 0);
  EXPECT_EQ(stats.min_grid_size, 0);
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
