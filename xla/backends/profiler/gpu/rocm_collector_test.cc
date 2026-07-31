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
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace test {

using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::XSpace;

// Returns the text of the string-valued stat `stat_name` on the first event
// named `event_name` in `plane`, or "" if there is no such event or stat.
// String stats are interned: the XStat holds a ref_value indexing the plane's
// stat_metadata rather than carrying the text inline.
std::string FindStringStat(const tensorflow::profiler::XPlane& plane,
                           absl::string_view event_name,
                           absl::string_view stat_name) {
  for (const auto& line : plane.lines()) {
    for (const auto& ev : line.events()) {
      if (plane.event_metadata().at(ev.metadata_id()).name() != event_name) {
        continue;
      }
      for (const auto& stat : ev.stats()) {
        if (plane.stat_metadata().at(stat.metadata_id()).name() != stat_name) {
          continue;
        }
        return plane.stat_metadata().at(stat.ref_value()).name();
      }
    }
  }
  return "";
}

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
  api_event.set_kernel_info(KernelDetails{
      .private_segment_size = 32,
      .group_segment_size = 1024,
      .workgroup_x = 256,
      .workgroup_y = 1,
      .workgroup_z = 1,
      .grid_x = 100,
      .grid_y = 1,
      .grid_z = 1,
      .func_ptr = reinterpret_cast<void*>(0xdeadbeef),
  });

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
  api_event.set_kernel_info(
      KernelDetails{.func_ptr = reinterpret_cast<void*>(0xdeadbeef)});
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
  // appear on the stream line. Dense stream remapping converts the raw
  // stream_id (123) to a sequential index (0), so we look for events on
  // any device line rather than matching a specific line ID.
  size_t total_kernel_events = 0;
  absl::flat_hash_set<std::string> seen_names;
  for (const auto& line : gpu_plane->lines()) {
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
  api_event.set_kernel_info(KernelDetails{
      .group_segment_size = 999,
      .workgroup_x = 999,
      .grid_x = 999,
  });
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
    activity.set_kernel_info(KernelDetails{
        .group_segment_size = shape.group_segment_size,
        .workgroup_x = shape.workgroup_x,
        .workgroup_y = 1,
        .workgroup_z = 1,
        .grid_x = shape.grid_x,
        .grid_y = 1,
        .grid_z = 1,
    });
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

// Regression test for the fabricated memcpy sizes. ROCm implements most
// hipMemcpy* calls as a ROCclr blit *kernel* dispatch rather than an SDMA
// transfer, so rocprofiler emits a KERNEL_DISPATCH record and no MEMORY_COPY
// record at all. The copy's byte count exists nowhere in the buffered data;
// only the HIP API callback arguments carry it, which is what CopyInfoMap now
// stashes and both event paths rejoin by correlation id.
//
// Both halves of the export are checked here:
//   - the host API row must report the real byte count. It used to adopt the
//     blit kernel's KernelDetails through the union and print the workgroup
//     dimension as a size ("size:512 dest:0 async:1") for a copy that in fact
//     moved a megabyte;
//   - the GPU dispatch row must report its true grid geometry *and* the copy
//     size, so a blit copy is not invisible to memcpy-aware tooling -- before
//     this it appeared only as an oddly-named kernel.
TEST(RocmCollectorTest, BlitCopyKernelReportsCopySizeAndGeometry) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 2000;
  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  constexpr uint32_t kCorrelationId = 11;
  constexpr uint32_t kDeviceId = 100;
  constexpr uint64_t kStreamId = 123;
  // 1 MiB: the size the standalone repro copies, and large enough that a
  // workgroup dimension read in its place is unmistakable.
  constexpr size_t kNumBytes = 1048576;

  // The hipMemcpy API row. HipApiEvent() recovers the direction and size from
  // the CopyInfoMap entry the callback stashed, so the event arrives here
  // already typed MemcpyH2D with a real byte count.
  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::MemcpyH2D;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "hipMemcpy";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  // The host row needs its own timestamps: CreateXEvent drops any event that
  // falls outside the session window, and the default 0 is before it.
  api_event.start_time_ns = 2500;
  api_event.end_time_ns = 3600;
  api_event.set_memcpy_info(MemcpyDetails{
      .num_bytes = kNumBytes,
      .destination = 0,
      .async = false,
  });
  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  // The GPU-side record for that copy is a kernel dispatch, with the grid
  // geometry ROCclr actually launched -- the geometry whose workgroup_x used
  // to be reported as the copy's size.
  RocmTracerEvent activity_event;
  activity_event.type = RocmTracerEventType::Kernel;
  activity_event.source = RocmTracerEventSource::Activity;
  activity_event.domain = RocmTracerEventDomain::HIP_OPS;
  activity_event.name = "__amd_rocclr_copyBuffer";
  activity_event.correlation_id = kCorrelationId;
  activity_event.start_time_ns = 3000;
  activity_event.end_time_ns = 3500;
  activity_event.device_id = kDeviceId;
  activity_event.stream_id = kStreamId;
  activity_event.set_kernel_info(KernelDetails{
      .workgroup_x = 512,
      .workgroup_y = 1,
      .workgroup_z = 1,
      .grid_x = 4096,
      .grid_y = 1,
      .grid_z = 1,
  });
  activity_event.blit_copy_info = CopyApiDetails{
      .type = RocmTracerEventType::MemcpyH2D,
      .details = {.num_bytes = kNumBytes, .destination = 0, .async = false},
  };
  collector.AddEvent(std::move(activity_event), /*is_auxiliary=*/false);

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  // Host API row: the size the user asked to move, not a grid dimension.
  const auto* host_plane =
      FindOrAddMutablePlaneWithName(&space, "/host:ROCTRACER");
  ASSERT_NE(host_plane, nullptr);
  EXPECT_EQ(FindStringStat(*host_plane, "hipMemcpy", "memcpy_details"),
            "kind:HtoD size:1048576 dest:0 async:0");

  // GPU dispatch row: both facts about a blit copy, neither displacing the
  // other.
  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);
  EXPECT_EQ(
      FindStringStat(*gpu_plane, "__amd_rocclr_copyBuffer", "kernel_details"),
      " grid:8,1,1 block:512,1,1 private_mem:0 group_mem:0 occ_pct:0");
  EXPECT_EQ(
      FindStringStat(*gpu_plane, "__amd_rocclr_copyBuffer", "memcpy_details"),
      "kind:HtoD size:1048576 dest:0 async:0");
}

// A kernel that is not a blit copy must not acquire a memcpy_details stat.
// blit_copy_info is populated only when the dispatch's correlation id belongs
// to a recorded copy API, so an ordinary dispatch leaves it empty.
TEST(RocmCollectorTest, OrdinaryKernelGetsNoMemcpyStat) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  RocmTraceCollectorImpl collector(options, /*start_walltime_ns=*/1000,
                                   /*start_gputime_ns=*/2000);

  constexpr uint32_t kCorrelationId = 12;

  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::Kernel;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "hipLaunchKernel";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  api_event.set_kernel_info(KernelDetails{});
  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  RocmTracerEvent activity_event;
  activity_event.type = RocmTracerEventType::Kernel;
  activity_event.source = RocmTracerEventSource::Activity;
  activity_event.domain = RocmTracerEventDomain::HIP_OPS;
  activity_event.name = "plain_kernel";
  activity_event.correlation_id = kCorrelationId;
  activity_event.start_time_ns = 3000;
  activity_event.end_time_ns = 3500;
  activity_event.device_id = 100;
  activity_event.stream_id = 123;
  activity_event.set_kernel_info(KernelDetails{
      .workgroup_x = 64,
      .grid_x = 4096,
  });
  collector.AddEvent(std::move(activity_event), /*is_auxiliary=*/false);

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);
  EXPECT_FALSE(
      FindStringStat(*gpu_plane, "plain_kernel", "kernel_details").empty());
  EXPECT_EQ(FindStringStat(*gpu_plane, "plain_kernel", "memcpy_details"), "");
}

// hipMemcpyPeer{,Async} are the first producers of MemcpyP2P on this backend:
// before ExtractCopyApiDetails() the copy branch of HipApiEvent() hardcoded
// MemcpyOther, so the enumerator existed but nothing emitted it. Every switch
// and type test the event then passes through has to know about it. The
// failure without that is not a mislabelled row: MemcpyP2P falls to the
// default arm of ApiActivityInfoExchange(), which drops the event outright.
TEST(RocmCollectorTest, PeerCopyApiRowSurvivesAndReportsDirection) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  RocmTraceCollectorImpl collector(options, /*start_walltime_ns=*/1000,
                                   /*start_gputime_ns=*/2000);

  constexpr uint32_t kCorrelationId = 13;
  constexpr size_t kNumBytes = 65536;
  // The peer device hipMemcpyPeer was asked to copy to, taken from the API
  // arguments -- the one field only the callback path can supply.
  constexpr uint32_t kDestinationDevice = 3;

  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::MemcpyP2P;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "hipMemcpyPeer";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  api_event.start_time_ns = 2500;
  api_event.end_time_ns = 3600;
  api_event.set_memcpy_info(MemcpyDetails{
      .num_bytes = kNumBytes,
      .destination = kDestinationDevice,
      .async = false,
  });
  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  // A peer copy is serviced by a ROCclr blit kernel like any other, so the
  // activity counterpart is a dispatch rather than a MEMORY_COPY record.
  RocmTracerEvent activity_event;
  activity_event.type = RocmTracerEventType::Kernel;
  activity_event.source = RocmTracerEventSource::Activity;
  activity_event.domain = RocmTracerEventDomain::HIP_OPS;
  activity_event.name = "__amd_rocclr_copyBuffer";
  activity_event.correlation_id = kCorrelationId;
  activity_event.start_time_ns = 3000;
  activity_event.end_time_ns = 3500;
  activity_event.device_id = 100;
  activity_event.stream_id = 123;
  activity_event.set_kernel_info(KernelDetails{
      .workgroup_x = 256,
      .grid_x = 65536,
  });
  activity_event.blit_copy_info = CopyApiDetails{
      .type = RocmTracerEventType::MemcpyP2P,
      .details = {.num_bytes = kNumBytes,
                  .destination = kDestinationDevice,
                  .async = false},
  };
  collector.AddEvent(std::move(activity_event), /*is_auxiliary=*/false);

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  const auto* host_plane =
      FindOrAddMutablePlaneWithName(&space, "/host:ROCTRACER");
  ASSERT_NE(host_plane, nullptr);
  EXPECT_EQ(FindStringStat(*host_plane, "hipMemcpyPeer", "memcpy_details"),
            "kind:PtoP size:65536 dest:3 async:0");

  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);
  EXPECT_EQ(
      FindStringStat(*gpu_plane, "__amd_rocclr_copyBuffer", "memcpy_details"),
      "kind:PtoP size:65536 dest:3 async:0");
}

// Every enumerator must have a name: GetRocmTracerEventTypeName() ends in
// DCHECK(false), so a missing arm aborts a debug build from inside the very
// logging the drop paths use to report what went wrong. Unsupported is
// reachable now that RocmTracerEvent::type is default-initialized to it.
TEST(RocmCollectorTest, EveryEventTypeHasAName) {
  for (const auto type : {
           RocmTracerEventType::Unsupported,
           RocmTracerEventType::Kernel,
           RocmTracerEventType::MemcpyH2D,
           RocmTracerEventType::MemcpyD2H,
           RocmTracerEventType::MemcpyD2D,
           RocmTracerEventType::MemcpyP2P,
           RocmTracerEventType::MemcpyOther,
           RocmTracerEventType::MemoryAlloc,
           RocmTracerEventType::MemoryFree,
           RocmTracerEventType::Memset,
           RocmTracerEventType::Synchronization,
           RocmTracerEventType::Generic,
       }) {
    EXPECT_STRNE(GetRocmTracerEventTypeName(type), "")
        << "unnamed event type " << static_cast<int>(type);
  }
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
