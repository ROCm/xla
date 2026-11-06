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

#include "xla/stream_executor/rocm/rocm_capture_invalidation.h"

#include <array>
#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::MockFunction;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

TEST(CreateDnnSupportTest, CreatesDnnSupportOnEveryDeviceWhenProbeFails) {
  constexpr std::array<int, 3> kOrdinals = {3, 5, 6};
  std::array<NiceMock<MockStreamExecutor>, kOrdinals.size()> executors;
  std::array<StreamExecutor*, kOrdinals.size()> executor_ptrs;
  for (size_t i = 0; i < kOrdinals.size(); ++i) {
    ON_CALL(executors[i], device_ordinal()).WillByDefault(Return(kOrdinals[i]));
    EXPECT_CALL(executors[i], AsDnn());
    executor_ptrs[i] = &executors[i];
  }
  StrictMock<MockFunction<absl::StatusOr<bool>(int, int)>> probe;
  EXPECT_CALL(probe, Call(3, 3)).WillOnce(Return(true));
  EXPECT_CALL(probe, Call(3, 5))
      .WillOnce(Return(absl::InternalError("probe failed")));

  CreateDnnSupportUnlessCaptureInvalidationIsDeviceScoped(
      executor_ptrs, probe.AsStdFunction());
}

TEST(SyncCallInvalidatesCaptureTest,
     InvalidatesASameDeviceCaptureAndLeavesNoTrace) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
    GTEST_SKIP() << "Test requires a GPU";
  }
  const int last_device = device_count - 1;
  ASSERT_EQ(hipSetDevice(last_device), hipSuccess);
  ASSERT_EQ(hipGetLastError(), hipSuccess);

  EXPECT_THAT(SyncCallInvalidatesCapture(0, 0), IsOkAndHolds(true));
  EXPECT_THAT(SyncCallInvalidatesCapture(0, last_device), IsOk());

  int current_device = -1;
  EXPECT_EQ(hipGetDevice(&current_device), hipSuccess);
  EXPECT_EQ(current_device, last_device);
  EXPECT_EQ(hipGetLastError(), hipSuccess);
  hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
  EXPECT_EQ(hipStreamIsCapturing(nullptr, &status), hipSuccess);
  EXPECT_EQ(status, hipStreamCaptureStatusNone);

  // A capture left behind, even an invalidated one, fails synchronous calls.
  void* buffer = nullptr;
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipMalloc(&buffer, 4), hipSuccess);
  EXPECT_EQ(hipMemset(buffer, 0, 4), hipSuccess);
  EXPECT_EQ(hipFree(buffer), hipSuccess);
}

}  // namespace
}  // namespace stream_executor::gpu
