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

#include <cstdlib>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_version_parser.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::gpu {
namespace {

void LogEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value != nullptr) {
    LOG(INFO) << "env: " << name << "=\"" << value << "\"";
  }
}

void EnsureRocmIsInitialized() {
  // Platform is intentionally leaked.
  // See the comment in platform_manager.h.
  absl::LeakCheckDisabler disabler;

  ASSERT_OK_AND_ASSIGN(
      stream_executor::Platform * platform,
      PlatformManager::PlatformWithName(stream_executor::GpuPlatformName()));
  CHECK_GT(platform->VisibleDeviceCount(), 0);
}

TEST(RocmDiagnosticsTest, DiagnosticRuns) {
  // Platform init is not under test; it only provides a working ROCm context.
  EnsureRocmIsInitialized();

  // Check basic ROCm system info: visibility env, runtime/driver version, and
  // device 0 identity.
  LogEnv("HIP_VISIBLE_DEVICES");
  LogEnv("ROCR_VISIBLE_DEVICES");

  int runtime_version = 0;
  ASSERT_EQ(hipRuntimeGetVersion(&runtime_version), hipSuccess);
  ASSERT_OK_AND_ASSIGN(stream_executor::SemanticVersion runtime,
                       stream_executor::ParseRocmVersion(runtime_version));
  LOG(INFO) << "hipRuntimeGetVersion packed=" << runtime_version
            << " parsed=" << runtime;

  int driver_version = 0;
  ASSERT_EQ(hipDriverGetVersion(&driver_version), hipSuccess);
  ASSERT_OK_AND_ASSIGN(stream_executor::SemanticVersion driver,
                       stream_executor::ParseRocmVersion(driver_version));
  LOG(INFO) << "hipDriverGetVersion packed=" << driver_version
            << " parsed=" << driver;

  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  ASSERT_GT(device_count, 0);

  hipDeviceProp_t props{};
  ASSERT_EQ(hipGetDeviceProperties(&props, 0), hipSuccess);
  LOG(INFO) << "device 0 name=\"" << props.name
            << "\" gcnArchName=" << props.gcnArchName;
}

}  // namespace
}  // namespace stream_executor::gpu
