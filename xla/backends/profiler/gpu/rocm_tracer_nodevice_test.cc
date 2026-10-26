/* Copyright 2026 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "rocm/include/rocprofiler-sdk/agent.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "xla/backends/profiler/gpu/rocm_device_map.h"
#include "xla/backends/profiler/gpu/rocm_tracer.h"

namespace xla {
namespace profiler {
namespace {

// The BUILD target sets ROCR_VISIBLE_DEVICES="", so HIP numbers no device
// although rocprofiler-sdk still lists every GPU.
TEST(RocmTracerNoDeviceTest, ReportsNoGpus) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();

  const HipDeviceMap& devices = tracer.EnsureHipDeviceMap();

  int num_sdk_gpus = 0;
  for (const auto& entry : tracer.agents_) {
    if (entry.second.type == ROCPROFILER_AGENT_TYPE_GPU) ++num_sdk_gpus;
  }
  ASSERT_GT(num_sdk_gpus, 0) << "with no SDK GPU, this test checks nothing";
  EXPECT_TRUE(devices.gpu_agents.empty());
  EXPECT_TRUE(devices.hip_device_of_agent.empty());
  EXPECT_EQ(tracer.NumGpus(), 0u);
  EXPECT_TRUE(tracer.GpuAgents().empty());
}

}  // namespace
}  // namespace profiler
}  // namespace xla
