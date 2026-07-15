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

TEST(RocmCollectorTest, RoctxEventsLandInDedicatedPlane) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 1000;

  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  // Generic event with roctx_range — must land in /host:ROCTX.
  RocmTracerEvent roctx_event;
  roctx_event.type = RocmTracerEventType::Generic;
  roctx_event.source = RocmTracerEventSource::ApiCallback;
  roctx_event.domain = RocmTracerEventDomain::HIP_API;
  roctx_event.name = "train_step";
  roctx_event.roctx_range = "train_step";
  roctx_event.start_time_ns = 2000;
  roctx_event.end_time_ns = 3000;
  roctx_event.thread_id = 42;
  roctx_event.device_id = RocmTracerEvent::kInvalidDeviceId;
  roctx_event.stream_id = RocmTracerEvent::kInvalidStreamId;
  roctx_event.correlation_id = RocmTracerEvent::kInvalidCorrelationId;
  collector.AddEvent(std::move(roctx_event), /*is_auxiliary=*/false);

  // Generic event without roctx_range — must land in /host:ROCTRACER.
  RocmTracerEvent generic_event;
  generic_event.type = RocmTracerEventType::Generic;
  generic_event.source = RocmTracerEventSource::ApiCallback;
  generic_event.domain = RocmTracerEventDomain::HIP_API;
  generic_event.name = "no_roctx";
  generic_event.roctx_range = {};
  generic_event.start_time_ns = 2000;
  generic_event.end_time_ns = 3000;
  generic_event.thread_id = 43;
  generic_event.device_id = RocmTracerEvent::kInvalidDeviceId;
  generic_event.stream_id = RocmTracerEvent::kInvalidStreamId;
  generic_event.correlation_id = RocmTracerEvent::kInvalidCorrelationId;
  collector.AddEvent(std::move(generic_event), /*is_auxiliary=*/false);

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  // Find the ROCTX plane.
  bool found_roctx_plane = false;
  bool roctx_event_in_roctx_plane = false;
  bool roctx_event_in_roctracer_plane = false;

  for (const auto& plane : space.planes()) {
    if (plane.name() == tsl::profiler::kRoctxPlaneName) {
      found_roctx_plane = true;
      for (const auto& line : plane.lines()) {
        for (const auto& ev : line.events()) {
          const auto& name =
              plane.event_metadata().at(ev.metadata_id()).name();
          if (name == "train_step") roctx_event_in_roctx_plane = true;
        }
      }
    }
    if (plane.name() == tsl::profiler::kRoctracerApiPlaneName) {
      for (const auto& line : plane.lines()) {
        for (const auto& ev : line.events()) {
          const auto& name =
              plane.event_metadata().at(ev.metadata_id()).name();
          if (name == "train_step") roctx_event_in_roctracer_plane = true;
        }
      }
    }
  }

  EXPECT_TRUE(found_roctx_plane) << "/host:ROCTX plane must exist";
  EXPECT_TRUE(roctx_event_in_roctx_plane)
      << "Generic event with roctx_range must land in /host:ROCTX";
  EXPECT_FALSE(roctx_event_in_roctracer_plane)
      << "Generic event with roctx_range must NOT be in /host:ROCTRACER";
}

TEST(RocmCollectorTest, GenericEventWithoutRoctxRangeLandsInRoctracerPlane) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 1000;

  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  RocmTracerEvent generic_event;
  generic_event.type = RocmTracerEventType::Generic;
  generic_event.source = RocmTracerEventSource::ApiCallback;
  generic_event.domain = RocmTracerEventDomain::HIP_API;
  generic_event.name = "no_roctx";
  generic_event.roctx_range = {};
  generic_event.start_time_ns = 2000;
  generic_event.end_time_ns = 3000;
  generic_event.thread_id = 44;
  generic_event.device_id = RocmTracerEvent::kInvalidDeviceId;
  generic_event.stream_id = RocmTracerEvent::kInvalidStreamId;
  generic_event.correlation_id = RocmTracerEvent::kInvalidCorrelationId;
  collector.AddEvent(std::move(generic_event), /*is_auxiliary=*/false);

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  bool event_in_roctracer = false;
  bool event_in_roctx = false;

  for (const auto& plane : space.planes()) {
    for (const auto& line : plane.lines()) {
      for (const auto& ev : line.events()) {
        const auto& name = plane.event_metadata().at(ev.metadata_id()).name();
        if (name != "no_roctx") continue;
        if (plane.name() == tsl::profiler::kRoctracerApiPlaneName)
          event_in_roctracer = true;
        if (plane.name() == tsl::profiler::kRoctxPlaneName)
          event_in_roctx = true;
      }
    }
  }

  EXPECT_TRUE(event_in_roctracer)
      << "Generic event without roctx_range must land in /host:ROCTRACER";
  EXPECT_FALSE(event_in_roctx)
      << "Generic event without roctx_range must NOT be in /host:ROCTX";
}


TEST(RocmCollectorTest, RoctxEventsLandInDedicatedPlane) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 1000;

  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  RocmTracerEvent roctx_event;
  roctx_event.type = RocmTracerEventType::Generic;
  roctx_event.source = RocmTracerEventSource::ApiCallback;
  roctx_event.domain = RocmTracerEventDomain::HIP_API;
  roctx_event.name = "train_step";
  roctx_event.roctx_range = "train_step";
  roctx_event.start_time_ns = 2000;
  roctx_event.end_time_ns = 3000;
  roctx_event.thread_id = 42;
  roctx_event.device_id = RocmTracerEvent::kInvalidDeviceId;
  roctx_event.stream_id = RocmTracerEvent::kInvalidStreamId;
  roctx_event.correlation_id = RocmTracerEvent::kInvalidCorrelationId;
  collector.AddEvent(std::move(roctx_event), /*is_auxiliary=*/false);

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  bool found_roctx_plane = false;
  bool roctx_event_in_roctx_plane = false;
  bool roctx_event_in_roctracer_plane = false;

  for (const auto& plane : space.planes()) {
    if (plane.name() == tsl::profiler::kRoctxPlaneName) {
      found_roctx_plane = true;
      for (const auto& line : plane.lines()) {
        for (const auto& ev : line.events()) {
          const auto& name =
              plane.event_metadata().at(ev.metadata_id()).name();
          if (name == "train_step") roctx_event_in_roctx_plane = true;
        }
      }
    }
    if (plane.name() == tsl::profiler::kRoctracerApiPlaneName) {
      for (const auto& line : plane.lines()) {
        for (const auto& ev : line.events()) {
          const auto& name =
              plane.event_metadata().at(ev.metadata_id()).name();
          if (name == "train_step") roctx_event_in_roctracer_plane = true;
        }
      }
    }
  }

  EXPECT_TRUE(found_roctx_plane) << "/host:ROCTX plane must exist";
  EXPECT_TRUE(roctx_event_in_roctx_plane)
      << "Generic event with roctx_range must land in /host:ROCTX";
  EXPECT_FALSE(roctx_event_in_roctracer_plane)
      << "Generic event with roctx_range must NOT be in /host:ROCTRACER";
}

TEST(RocmCollectorTest, GenericEventWithoutRoctxRangeLandsInRoctracerPlane) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 1000;

  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  RocmTracerEvent generic_event;
  generic_event.type = RocmTracerEventType::Generic;
  generic_event.source = RocmTracerEventSource::ApiCallback;
  generic_event.domain = RocmTracerEventDomain::HIP_API;
  generic_event.name = "no_roctx";
  generic_event.roctx_range = {};
  generic_event.start_time_ns = 2000;
  generic_event.end_time_ns = 3000;
  generic_event.thread_id = 44;
  generic_event.device_id = RocmTracerEvent::kInvalidDeviceId;
  generic_event.stream_id = RocmTracerEvent::kInvalidStreamId;
  generic_event.correlation_id = RocmTracerEvent::kInvalidCorrelationId;
  collector.AddEvent(std::move(generic_event), /*is_auxiliary=*/false);

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  bool event_in_roctracer = false;
  bool event_in_roctx = false;

  for (const auto& plane : space.planes()) {
    for (const auto& line : plane.lines()) {
      for (const auto& ev : line.events()) {
        const auto& name = plane.event_metadata().at(ev.metadata_id()).name();
        if (name != "no_roctx") continue;
        if (plane.name() == tsl::profiler::kRoctracerApiPlaneName)
          event_in_roctracer = true;
        if (plane.name() == tsl::profiler::kRoctxPlaneName)
          event_in_roctx = true;
      }
    }
  }

  EXPECT_TRUE(event_in_roctracer)
      << "Generic event without roctx_range must land in /host:ROCTRACER";
  EXPECT_FALSE(event_in_roctx)
      << "Generic event without roctx_range must NOT be in /host:ROCTX";
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
