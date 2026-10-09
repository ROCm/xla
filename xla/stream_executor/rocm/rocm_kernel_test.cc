/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_kernel.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {
using testing::Ge;

TEST(RocmKernelTest, GetMaxOccupiedBlocksPerCore) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(KernelLoaderSpec add_kernel,
                       GetAddI32TestKernelSpec(rocm::kROCmPlatformId));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> rocm_kernel,
                       executor->LoadKernel(add_kernel));

  EXPECT_EQ(rocm_kernel->Arity(), 3);
  EXPECT_THAT(rocm_kernel->GetMaxOccupiedBlocksPerCore(
                  ThreadDim(1, 1, 1), /*dynamic_shared_memory_bytes=*/0),
              absl_testing::IsOkAndHolds(Ge(1)));
}

TEST(RocmKernelTest, CheckDynamicSharedMemoryBytes) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(auto kernel, LoadDynShmemTestKernel(executor));
  const auto& rocm_kernel = static_cast<const RocmKernel&>(*kernel);

  // DynShmemKernel has no static shared memory, so it may use all of it.
  int64_t limit =
      executor->GetDeviceDescription().shared_memory_per_block_optin();
  EXPECT_OK(rocm_kernel.CheckDynamicSharedMemoryBytes(limit));
  EXPECT_THAT(rocm_kernel.CheckDynamicSharedMemoryBytes(limit + 1),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RocmKernelTest, CheckDynamicSharedMemoryBytesWithoutLimit) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  // Not loaded through LoadKernel, so its limit was never queried.
  RocmKernel kernel(executor);
  EXPECT_OK(kernel.CheckDynamicSharedMemoryBytes(0));
  EXPECT_THAT(kernel.CheckDynamicSharedMemoryBytes(1),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));

  absl::StatusOr<uint64_t> failed_query = absl::InternalError("query failed");
  EXPECT_OK(CheckDynamicSharedMemoryBytes("kernel", 0, failed_query));
  EXPECT_THAT(CheckDynamicSharedMemoryBytes("kernel", 1, failed_query),
              absl_testing::StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace stream_executor::gpu
