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

// Covers the XLA_ROCM_VMM_EXPORTABLE_HANDLES=0 escape hatch.
//
// RocmVmmHandleType() resolves its answer once for the lifetime of the
// process, so the environment variable can only be observed by a binary that
// sets it before the first query. That is why this lives in its own test
// target instead of alongside rocm_vmm_handle_type_test.cc.

#include <unistd.h>

#include <cstdlib>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/rocm/rocm_vmm_handle_type.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

class RocmVmmHandleTypeDisabledTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("ROCM");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "ROCM platform not available";
    }
    auto executor_or = platform_or.value()->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "ROCM executor not available: " << executor_or.status();
    }
    executor_ = executor_or.value();

    std::unique_ptr<ActivateContext> activation = executor_->Activate();
    ASSERT_OK(ToStatus(hipDeviceGet(&device_, executor_->device_ordinal()),
                       "hipDeviceGet"));
  }

  StreamExecutor* executor_ = nullptr;
  hipDevice_t device_ = 0;
};

TEST_F(RocmVmmHandleTypeDisabledTest, FallsBackToNonExportableHandles) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();

  EXPECT_EQ(RocmVmmHandleType(), hipMemHandleTypeNone);
  EXPECT_EQ(MakeRocmVmmAllocationProps(device_).requestedHandleTypes,
            hipMemHandleTypeNone);
}

// The point of the escape hatch is to restore the pre-fix behavior exactly, so
// the allocation must genuinely be non-exportable - not merely requested that
// way.
TEST_F(RocmVmmHandleTypeDisabledTest, AllocationIsNotExportable) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmRawMemoryAllocation> allocation,
      RocmRawMemoryAllocation::Create(executor_, /*size=*/1024 * 1024));

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  int fd = -1;
  hipError_t result = hipMemExportToShareableHandle(
      &fd, allocation->GetHandle(), hipMemHandleTypePosixFileDescriptor, 0ULL);
  if (result == hipSuccess) {
    close(fd);
  }
  EXPECT_NE(result, hipSuccess);
}

}  // namespace
}  // namespace stream_executor::gpu

int main(int argc, char** argv) {
  // Must happen before any RocmVmmHandleType() call resolves the answer.
  setenv("XLA_ROCM_VMM_EXPORTABLE_HANDLES", "0", /*overwrite=*/1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
