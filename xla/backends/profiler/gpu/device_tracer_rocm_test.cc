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

// End-to-end tests for the ROCm GpuTracer's handling of
// ProfileOptions.advanced_configuration. Requires a ROCm GPU; tagged
// gpu + rocm-only like its siblings in this package.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

// Defined in device_tracer_rocm.cc, which is deliberately header-less. The
// comment there ("Not in anonymous namespace for testing purposes") is the
// standing invitation for this declaration.
std::unique_ptr<tsl::profiler::ProfilerInterface> CreateGpuTracer(
    const tensorflow::ProfileOptions& options);

namespace {

using ::tensorflow::ProfileOptions;
using ::tensorflow::profiler::XSpace;

constexpr char kBogusKey[] = "gpu_max_callbac_api_events";  // realistic typo

// A ProfileOptions the tracer factory will accept. version(1) matters: with
// version 0, ProfilerSession::GetOptions strips advanced_configuration before
// the tracer ever sees it, so a test built on a version-0 proto would pass
// while asserting nothing.
ProfileOptions MakeGpuOptions() {
  ProfileOptions options;
  options.set_version(1);
  options.set_device_type(ProfileOptions::GPU);
  return options;
}

void SetInt(ProfileOptions& options, const std::string& key, int64_t value) {
  ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_int64_value(value);
  (*options.mutable_advanced_configuration())[key] = config_value;
}

// A few HIP operations, enough to produce trace events. Copied in shape from
// RocmTracerTest.CapturesHipEvents; no custom kernel is needed because the
// memcpy API callbacks are what the caps and the exporter act on.
void RunSomeHipWork(int iterations) {
  constexpr size_t kNumFloats = 1024;
  constexpr size_t kSize = kNumFloats * sizeof(float);
  std::vector<float> host_data(kNumFloats, 1.0f);
  void* device_data = nullptr;
  ASSERT_EQ(hipMalloc(&device_data, kSize), hipSuccess);
  for (int i = 0; i < iterations; ++i) {
    ASSERT_EQ(
        hipMemcpy(device_data, host_data.data(), kSize, hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(host_data.data(), device_data, kSize, hipMemcpyDeviceToHost),
        hipSuccess);
  }
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(hipFree(device_data), hipSuccess);
}

bool HasGpu() {
  int device_count = 0;
  return hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0;
}

size_t CountGpuPlaneEvents(const XSpace& space) {
  size_t total = 0;
  for (const auto& plane : space.planes()) {
    if (!absl::StartsWith(plane.name(), "/device:GPU:")) {
      continue;
    }
    for (const auto& line : plane.lines()) {
      total += line.events_size();
    }
  }
  return total;
}

// This is the whole point of the PR in one test. On the CUPTI path today the
// equivalent assertion fails on both halves: DoStart returns an error, the
// session swallows it, and the user gets an empty GPU plane with an empty
// errors field and no way to tell that a typo was the cause.
TEST(DeviceTracerRocmTest, BadKeyStillProducesATraceAndSaysWhy) {
  if (!HasGpu()) GTEST_SKIP() << "No HIP devices available";

  ProfileOptions options = MakeGpuOptions();
  SetInt(options, kBogusKey, 5000);

  auto tracer = CreateGpuTracer(options);
  ASSERT_NE(tracer, nullptr);

  TF_ASSERT_OK(tracer->Start());
  RunSomeHipWork(/*iterations=*/4);
  absl::SleepFor(absl::Milliseconds(100));
  TF_ASSERT_OK(tracer->Stop());

  XSpace space;
  TF_ASSERT_OK(tracer->CollectData(&space));

  EXPECT_GT(CountGpuPlaneEvents(space), 0u)
      << "A bad advanced_configuration key must not cost the user the trace.";

  bool named = false;
  for (const auto& error : space.errors()) {
    named |= absl::StrContains(error, kBogusKey);
  }
  EXPECT_TRUE(named) << "XSpace.errors must name the offending key; got "
                     << space.errors_size() << " error(s).";
}

// The escape hatch. Callers that explicitly asked for start failures to be
// fatal keep getting them; the leniency above is a default, not a policy.
TEST(DeviceTracerRocmTest, RaiseErrorOnStartFailureMakesABadKeyFatal) {
  if (!HasGpu()) GTEST_SKIP() << "No HIP devices available";

  ProfileOptions options = MakeGpuOptions();
  options.set_raise_error_on_start_failure(true);
  SetInt(options, kBogusKey, 5000);

  auto tracer = CreateGpuTracer(options);
  ASSERT_NE(tracer, nullptr);

  const absl::Status status = tracer->Start();
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), kBogusKey)) << status;

  // Start failed before the tracer was enabled, so Stop is a no-op and the
  // singleton is left available for the next test.
  TF_EXPECT_OK(tracer->Stop());
}

// Pins the precedence order in BuildOptions, which is otherwise only asserted
// by reading it top to bottom. The flag default is 4M; if
// advanced_configuration were ignored, all the events below would survive.
TEST(DeviceTracerRocmTest, AdvancedConfigurationOverridesTheFlagDefault) {
  if (!HasGpu()) GTEST_SKIP() << "No HIP devices available";

  constexpr int64_t kCallbackCap = 3;
  constexpr int kIterations = 20;  // >= 40 API callbacks, far above the cap

  ProfileOptions options = MakeGpuOptions();
  SetInt(options, "gpu_max_callback_api_events", kCallbackCap);

  auto tracer = CreateGpuTracer(options);
  ASSERT_NE(tracer, nullptr);

  TF_ASSERT_OK(tracer->Start());
  RunSomeHipWork(kIterations);
  absl::SleepFor(absl::Milliseconds(100));
  TF_ASSERT_OK(tracer->Stop());

  XSpace space;
  TF_ASSERT_OK(tracer->CollectData(&space));

  EXPECT_TRUE(space.errors().empty())
      << "A documented key must not be reported as a problem.";
  EXPECT_LE(CountGpuPlaneEvents(space), static_cast<size_t>(kCallbackCap))
      << "gpu_max_callback_api_events did not take precedence over the "
         "xla_gpu_rocm_max_trace_events default.";
}

}  // namespace
}  // namespace profiler
}  // namespace xla
