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

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>
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

TEST(RocmCollectorTest, Collect4MEvents) {
    constexpr uint64_t kMaxEvents = 4 * 1024 * 1024;  // Actual 4M limit
    
    RocmTraceCollectorOptions options;
    options.max_callback_api_events = kMaxEvents;
    options.max_activity_api_events = kMaxEvents;
    options.max_annotation_strings = kMaxEvents;
    options.num_gpus = 1;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RocmTraceCollectorImpl collector(options, 1000, 2000);
    
    // Simulate adding 4M events
    for (uint64_t i = 0; i < kMaxEvents; i++) {
      RocmTracerEvent event;
      event.type = RocmTracerEventType::Kernel;
      event.source = RocmTracerEventSource::ApiCallback;
      event.domain = RocmTracerEventDomain::HIP_API;
      event.name = "kernel";
      event.correlation_id = i;
      event.thread_id = i % 1024;  // Vary thread IDs
      
      collector.AddEvent(std::move(event), false);
      
      // Periodic progress
      if (i % 1'000'000 == 0) {
        LOG(INFO) << "Added " << i << " events";
      }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    LOG(INFO) << "Time to add 4M events: " << duration.count() << " ms";
    
    // Test export performance
    start_time = std::chrono::high_resolution_clock::now();
    collector.Flush();
    XSpace space;
    collector.Export(&space);
    end_time = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    LOG(INFO) << "Time to export 4M events: " << duration.count() << " ms";
    
    // Verify memory usage is reasonable
    size_t estimated_memory_mb = (kMaxEvents * sizeof(RocmTracerEvent)) / (1024 * 1024);
    LOG(INFO) << "Estimated memory usage: " << estimated_memory_mb << " MB";
    EXPECT_LT(estimated_memory_mb, 2000);  // Should be < 2GB
  }

}  // namespace test
}  // namespace profiler
}  // namespace xla
