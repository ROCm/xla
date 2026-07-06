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
  api_event.kernel_info.func_ptr = reinterpret_cast<void*>(0xdeadbeef);

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
  api_event.kernel_info.func_ptr = reinterpret_cast<void*>(0xdeadbeef);
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

  void AddKernelPair(void* func_ptr, uint32_t wg_x, uint32_t wg_y,
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
    api.kernel_info.func_ptr = func_ptr;
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

// When func_ptr is null the occupancy stats block must not be written —
// the three occupancy XEvent stats must be absent.
TEST(RocmCollectorOccupancyTest, NullFuncPtrSkipsOccupancyStats) {
  OccupancyTestFixture f;
  f.AddKernelPair(/*func_ptr=*/nullptr, 256, 1, 1, 0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  // Occupancy stat metadata must not appear if func_ptr was null.
  const std::string kOccKey =
      GetStatTypeStr(StatType::kTheoreticalOccupancyPct);
  for (const auto& [id, smd] : gpu->stat_metadata()) {
    EXPECT_NE(smd.name(), kOccKey)
        << "kTheoreticalOccupancyPct must not appear when func_ptr is null";
  }
}

// When a non-null func_ptr is provided but no agent data has been set
// (max_waves_per_cu_ == 0), occupancy stats must still not be written.
TEST(RocmCollectorOccupancyTest, ZeroWavesPerCuSkipsOccupancyStats) {
  OccupancyTestFixture f;
  // Use a non-null but obviously invalid pointer. hipOccupancyMax* will fail,
  // and even before that the max_waves_per_cu guard in CreateXEvent fires.
  f.AddKernelPair(reinterpret_cast<void*>(0xdeadbeef), 256, 1, 1, 0, 3000,
                  4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const std::string kOccKey =
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
  f.AddKernelPair(/*func_ptr=*/nullptr, 64, 1, 1, 512, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  const std::string kDetailsKey = GetStatTypeStr(StatType::kKernelDetails);
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

// kKernelDetails string must contain "occ_pct:0" when func_ptr is null.
TEST(RocmCollectorOccupancyTest, KernelDetailsContainsZeroOccupancyWhenNoPtr) {
  OccupancyTestFixture f;
  f.AddKernelPair(/*func_ptr=*/nullptr, 128, 1, 1, 0, 3000, 4000);

  f.collector.Flush();
  tensorflow::profiler::XSpace space;
  f.collector.Export(&space);

  const auto* gpu = FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu, nullptr);

  // Find the kKernelDetails stat metadata id, then locate its ref-value string.
  const std::string kDetailsKey = GetStatTypeStr(StatType::kKernelDetails);
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
        // kKernelDetails is written as a ref_value pointing to the string.
        int64_t ref = stat.ref_value();
        auto it = gpu->stat_metadata().find(ref);
        if (it == gpu->stat_metadata().end()) continue;
        const std::string& details = it->second.name();
        if (details.find("occ_pct:0") != std::string::npos) {
          found_occ_pct_zero = true;
        }
      }
    }
  }
  EXPECT_TRUE(found_occ_pct_zero)
      << "KernelDetails string must contain 'occ_pct:0' when func_ptr is null";
}

// RocmDeviceOccupancyParams equality and hashing must work correctly so that
// the occupancy cache deduplicates across identical kernel launches.
TEST(RocmCollectorOccupancyTest, OccupancyParamsCacheKey) {
  RocmDeviceOccupancyParams a{};
  a.block_size = 256;
  a.dynamic_smem_size = 0;
  a.func_ptr = reinterpret_cast<void*>(0x1000);
  a.max_waves_per_cu = 40;
  a.wave_front_size = 64;

  RocmDeviceOccupancyParams b = a;

  EXPECT_EQ(a, b) << "Identical params must compare equal";

  b.block_size = 128;
  EXPECT_NE(a, b) << "Params with different block_size must not be equal";

  b = a;
  b.func_ptr = reinterpret_cast<void*>(0x2000);
  EXPECT_NE(a, b) << "Params with different func_ptr must not be equal";

  b = a;
  b.max_waves_per_cu = 20;
  EXPECT_NE(a, b) << "Params with different max_waves_per_cu must not be equal";

  b = a;
  b.wave_front_size = 32;
  EXPECT_NE(a, b) << "Params with different wave_front_size must not be equal";
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
