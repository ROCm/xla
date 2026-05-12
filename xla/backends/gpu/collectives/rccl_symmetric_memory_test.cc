/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/rccl_symmetric_memory.h"

#include <cstddef>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_address.h"

#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200

namespace xla::gpu {
namespace {

namespace se = stream_executor;

using ::absl_testing::IsOk;
using ::testing::HasSubstr;
using ::testing::Not;

// Test fixture that creates a single-rank RCCL communicator and allocates a
// small GPU buffer. Tests are skipped if no ROCm device or RCCL is available.
class RcclSymmetricMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Skip if no ROCm/HIP GPU is present.
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
      GTEST_SKIP() << "No ROCm/HIP GPU device available";
    }
    if (hipSetDevice(0) != hipSuccess) {
      GTEST_SKIP() << "hipSetDevice(0) failed";
    }

    // Initialise a single-rank RCCL communicator on device 0.
    int device_ids[] = {0};
    ncclResult_t nccl_err = ncclCommInitAll(&comm_, 1, device_ids);
    if (nccl_err != ncclSuccess) {
      GTEST_SKIP() << "RCCL communicator initialisation failed: "
                   << ncclGetErrorString(nccl_err);
    }

    // Allocate a 4 KiB GPU buffer to use as the symmetric memory backing store.
    constexpr size_t kBufSize = 4096;
    buf_size_ = kBufSize;
    if (hipMalloc(&buf_ptr_, buf_size_) != hipSuccess) {
      ncclCommDestroy(comm_);
      comm_ = nullptr;
      FAIL() << "hipMalloc(" << kBufSize << ") failed";
    }
  }

  void TearDown() override {
    if (buf_ptr_ != nullptr) {
      (void)hipFree(buf_ptr_);
      buf_ptr_ = nullptr;
    }
    if (comm_ != nullptr) {
      ncclCommDestroy(comm_);
      comm_ = nullptr;
    }
  }

  ncclComm_t comm_ = nullptr;
  void* buf_ptr_ = nullptr;
  size_t buf_size_ = 0;
};

// Verifies that RcclSymmetricMemory::Create succeeds for a valid buffer.
TEST_F(RcclSymmetricMemoryTest, CreateSucceeds) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  ASSERT_OK_AND_ASSIGN(auto symm_mem, RcclSymmetricMemory::Create(comm_, addr));
  ASSERT_NE(symm_mem, nullptr);
}

// Verifies that addr() returns exactly the pointer and size that were
// registered.
TEST_F(RcclSymmetricMemoryTest, AddrMatchesRegisteredBuffer) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  ASSERT_OK_AND_ASSIGN(auto symm_mem, RcclSymmetricMemory::Create(comm_, addr));
  EXPECT_EQ(symm_mem->addr().opaque(), buf_ptr_);
  EXPECT_EQ(symm_mem->addr().size(), buf_size_);
}

// Verifies that ToString() contains key diagnostic fields.
TEST_F(RcclSymmetricMemoryTest, ToStringContainsExpectedFields) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  ASSERT_OK_AND_ASSIGN(auto symm_mem, RcclSymmetricMemory::Create(comm_, addr));
  const std::string str = symm_mem->ToString();
  EXPECT_THAT(str, HasSubstr("RcclSymmetricMemory"));
  EXPECT_THAT(str, HasSubstr("comm="));
  EXPECT_THAT(str, HasSubstr("win="));
  EXPECT_THAT(str, HasSubstr("ptr="));
}

// Verifies that PackKernelArg() returns a non-null handle, and that the
// win() accessor returns the same underlying ncclWindow_t.
TEST_F(RcclSymmetricMemoryTest, PackKernelArgReturnsValidWindowHandle) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  ASSERT_OK_AND_ASSIGN(auto symm_mem, RcclSymmetricMemory::Create(comm_, addr));
  // PackKernelArg() returns a void* (PackedKernelArg) wrapping the window.
  auto packed = symm_mem->PackKernelArg();
  EXPECT_NE(packed, nullptr);
  // win() returns the typed ncclWindow_t handle directly.
  EXPECT_NE(symm_mem->win(), nullptr);
}

// Verifies that multimem_addr() returns Unimplemented — RCCL does not support
// multimem, so the base-class default is expected.
TEST_F(RcclSymmetricMemoryTest, MultimemAddrNotSupported) {
  se::DeviceAddressBase addr(buf_ptr_, buf_size_);
  ASSERT_OK_AND_ASSIGN(auto symm_mem, RcclSymmetricMemory::Create(comm_, addr));
  auto result = symm_mem->multimem_addr();
  EXPECT_THAT(result, Not(IsOk()));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}

// Verifies that two independent windows created from the same communicator
// on different buffers have distinct ncclWindow_t handles.
TEST_F(RcclSymmetricMemoryTest, TwoWindowsHaveDistinctHandles) {
  void* buf2_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&buf2_ptr, buf_size_), hipSuccess);

  se::DeviceAddressBase addr1(buf_ptr_, buf_size_);
  se::DeviceAddressBase addr2(buf2_ptr, buf_size_);

  ASSERT_OK_AND_ASSIGN(auto symm1, RcclSymmetricMemory::Create(comm_, addr1));
  ASSERT_OK_AND_ASSIGN(auto symm2, RcclSymmetricMemory::Create(comm_, addr2));

  EXPECT_NE(symm1->win(), symm2->win());
  EXPECT_EQ(symm1->addr().opaque(), buf_ptr_);
  EXPECT_EQ(symm2->addr().opaque(), buf2_ptr);

  (void)hipFree(buf2_ptr);
}

}  // namespace
}  // namespace xla::gpu
