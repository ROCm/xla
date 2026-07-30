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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace test {

using tsl::profiler::FindOrAddMutablePlaneWithName;
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

// Regression test for the kernel_info round-trip in ApiActivityInfoExchange.
// The merge used to overwrite each activity event's KernelDetails with the API
// event's copy, which itself came from front(). Every dispatch under one
// correlation_id therefore reported the *first* dispatch's grid/block geometry
// -- on a real llama2-7b trace this corrupted all 987 multi-kernel correlation
// groups, e.g. 16 elementwise kernels all reporting a sibling GEMM's
// grid:1408,1,1 block:256,1,1 group_mem:61440.
TEST(RocmCollectorTest, DistinctKernelsUnderOneCorrelationIdKeepOwnDetails) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 2000;
  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  constexpr uint32_t kCorrelationId = 7;
  constexpr uint32_t kDeviceId = 100;
  constexpr uint64_t kStreamId = 123;

  // One hipGraphLaunch API event. Its own kernel_info is deliberately given
  // the "poison" geometry the merge used to smear over every dispatch.
  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::Kernel;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "hipGraphLaunch";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  api_event.kernel_info = KernelDetails{};
  api_event.kernel_info.workgroup_x = 999;
  api_event.kernel_info.grid_x = 999;
  api_event.kernel_info.group_segment_size = 999;
  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  // Two dispatches under that one correlation_id with genuinely different
  // geometry -- a wide GEMM-like launch followed by a small elementwise one.
  struct DispatchShape {
    const char* name;
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t workgroup_x;
    uint32_t grid_x;
    uint32_t group_segment_size;
  };
  constexpr DispatchShape kDispatches[] = {
      {"gemm_kernel", 3000, 3500, 256, 360448, 61440},
      {"elementwise_kernel", 3500, 4000, 64, 4096, 0},
  };
  for (const auto& shape : kDispatches) {
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
    activity.kernel_info = KernelDetails{};
    activity.kernel_info.workgroup_x = shape.workgroup_x;
    activity.kernel_info.workgroup_y = 1;
    activity.kernel_info.workgroup_z = 1;
    activity.kernel_info.grid_x = shape.grid_x;
    activity.kernel_info.grid_y = 1;
    activity.kernel_info.grid_z = 1;
    activity.kernel_info.group_segment_size = shape.group_segment_size;
    collector.AddEvent(std::move(activity), /*is_auxiliary=*/false);
  }

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);

  // Collect the kernel_details string emitted for each dispatch by name.
  absl::flat_hash_map<std::string, std::string> details_by_kernel;
  for (const auto& line : gpu_plane->lines()) {
    for (const auto& ev : line.events()) {
      const std::string& kernel_name =
          gpu_plane->event_metadata().at(ev.metadata_id()).name();
      for (const auto& stat : ev.stats()) {
        const auto& stat_name =
            gpu_plane->stat_metadata().at(stat.metadata_id()).name();
        if (stat_name != "kernel_details") continue;
        details_by_kernel[kernel_name] =
            gpu_plane->stat_metadata().at(stat.ref_value()).name();
      }
    }
  }

  ASSERT_TRUE(details_by_kernel.contains("gemm_kernel"));
  ASSERT_TRUE(details_by_kernel.contains("elementwise_kernel"));

  // Each dispatch must report the geometry it was created with. ToXStat
  // renders grid as grid_x / workgroup_x, so 360448/256 = 1408 and
  // 4096/64 = 64.
  EXPECT_EQ(details_by_kernel["gemm_kernel"],
            " grid:1408,1,1 block:256,1,1 private_mem:0 group_mem:61440"
            " occ_pct:0");
  EXPECT_EQ(details_by_kernel["elementwise_kernel"],
            " grid:64,1,1 block:64,1,1 private_mem:0 group_mem:0 occ_pct:0");

  // The headline invariant: distinct dispatches must not collapse onto one
  // shared kernel_details value, and neither may inherit the API event's.
  EXPECT_NE(details_by_kernel["gemm_kernel"],
            details_by_kernel["elementwise_kernel"]);
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
