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

#include "xla/stream_executor/rocm/rocm_executor.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
using testing::IsEmpty;
using testing::Not;

// Base test fixture for ROCm executor tests that ensures clean GPU state.
// Synchronizes the device after each test to prevent cross-process memory
class RocmExecutorTestFixture : public ::testing::Test {
 protected:
  void TearDown() override {
    auto platform_or = PlatformManager::PlatformWithName("ROCM");
    if (!platform_or.ok()) {
      LOG(WARNING) << "Failed to get ROCM platform: " << platform_or.status();
      return;
    }

    auto executor_or = platform_or.value()->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      LOG(WARNING) << "Failed to get executor for device 0: "
                   << executor_or.status();
      return;
    }

    // Use StreamExecutor's SynchronizeAllActivity which is platform-agnostic
    if (!executor_or.value()->SynchronizeAllActivity()) {
      LOG(WARNING) << "SynchronizeAllActivity failed";
    }
  }
};

TEST(RocmExecutorTest, CreateDeviceDescription) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceDescription> result,
                          RocmExecutor::CreateDeviceDescription(0));

  constexpr SemanticVersion kNullVersion{0, 0, 0};
  EXPECT_NE(result->runtime_version(), kNullVersion);
  EXPECT_NE(result->driver_version(), kNullVersion);
  EXPECT_NE(result->compile_time_toolkit_version(), kNullVersion);

  EXPECT_THAT(result->platform_version(), Not(IsEmpty()));
  EXPECT_THAT(result->name(), Not(IsEmpty()));
  EXPECT_THAT(result->model_str(), Not(IsEmpty()));
  EXPECT_THAT(result->device_vendor(), "Advanced Micro Devices, Inc");

  EXPECT_THAT(result->gpu_compute_capability()
                  .rocm_compute_capability()
                  ->gcn_arch_name(),
              Not(IsEmpty()));
}

TEST(RocmExecutorTest, GetRocmKernel) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("ROCM"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(KernelLoaderSpec add_kernel,
                          GetAddI32TestKernelSpec(rocm::kROCmPlatformId));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          executor->LoadKernel(add_kernel));

  auto rocm_executor = dynamic_cast<RocmExecutor*>(executor);
  ASSERT_NE(rocm_executor, nullptr);
  EXPECT_THAT(rocm_executor->GetRocmKernel(kernel.get()),
              absl_testing::IsOkAndHolds(kernel.get()));

  rocm_executor->UnloadKernel(kernel.get());
  EXPECT_THAT(rocm_executor->GetRocmKernel(kernel.get()),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));

  EXPECT_THAT(rocm_executor->GetRocmKernel(nullptr),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(RocmExecutorTestFixture, CreateUnifiedMemoryAllocatorWorks) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("ROCM"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemoryAllocator> allocator,
      executor->CreateMemoryAllocator(MemorySpace::kUnified));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
  allocation.reset();
}

TEST_F(RocmExecutorTestFixture, CreateHostMemoryAllocatorWorks) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("ROCM"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocator> allocator,
                          executor->CreateMemoryAllocator(MemorySpace::kHost));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
  allocation.reset();
}

TEST_F(RocmExecutorTestFixture, CreateCollectiveMemoryAllocatorWorks) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("ROCM"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemoryAllocator> allocator,
      executor->CreateMemoryAllocator(MemorySpace::kCollective));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          allocator->Allocate(1024));
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), 1024);
  allocation.reset();
}

TEST(RocmExecutorTest, CreateUnsupportedMemoryAllocatorsFail) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("ROCM"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  EXPECT_THAT(executor->CreateMemoryAllocator(MemorySpace::kDevice),
              Not(absl_testing::IsOk()));
}

}  // namespace
}  // namespace stream_executor::gpu
