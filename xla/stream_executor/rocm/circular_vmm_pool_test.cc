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

#include "xla/stream_executor/rocm/circular_vmm_pool.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {
namespace {

class CircularVmmPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("ROCM");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "ROCm platform not available";
    }
    auto executor_or = platform_or.value()->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "ROCm executor not available";
    }
    executor_ = executor_or.value();
  }

  StreamExecutor* executor_ = nullptr;
};

TEST_F(CircularVmmPoolTest, CreateWithSingleSlot) {
  std::vector<uint64_t> buffer_sizes = {1024, 2048, 512};
  ASSERT_OK_AND_ASSIGN(auto pool,
                       CircularVmmPool::Create(executor_, buffer_sizes, 1));
  EXPECT_EQ(pool->num_slots(), 1);
}

TEST_F(CircularVmmPoolTest, CreateWithTwoSlots) {
  std::vector<uint64_t> buffer_sizes = {4096, 8192};
  ASSERT_OK_AND_ASSIGN(auto pool,
                       CircularVmmPool::Create(executor_, buffer_sizes, 2));
  EXPECT_EQ(pool->num_slots(), 2);
}

TEST_F(CircularVmmPoolTest, AddressesAreStablePerSlot) {
  std::vector<uint64_t> buffer_sizes = {1024, 2048};
  ASSERT_OK_AND_ASSIGN(auto pool,
                       CircularVmmPool::Create(executor_, buffer_sizes, 2));

  ASSERT_OK_AND_ASSIGN(auto addrs_iter0, pool->AcquireNextSlot(0));
  ASSERT_OK_AND_ASSIGN(auto addrs_iter2, pool->AcquireNextSlot(2));

  ASSERT_EQ(addrs_iter0.size(), 2);
  ASSERT_EQ(addrs_iter2.size(), 2);
  EXPECT_EQ(addrs_iter0[0].opaque(), addrs_iter2[0].opaque());
  EXPECT_EQ(addrs_iter0[1].opaque(), addrs_iter2[1].opaque());
}

TEST_F(CircularVmmPoolTest, DifferentSlotsHaveDifferentAddresses) {
  std::vector<uint64_t> buffer_sizes = {4096};
  ASSERT_OK_AND_ASSIGN(auto pool,
                       CircularVmmPool::Create(executor_, buffer_sizes, 2));

  ASSERT_OK_AND_ASSIGN(auto addrs_slot0, pool->AcquireNextSlot(0));
  ASSERT_OK_AND_ASSIGN(auto addrs_slot1, pool->AcquireNextSlot(1));

  ASSERT_EQ(addrs_slot0.size(), 1);
  ASSERT_EQ(addrs_slot1.size(), 1);
  EXPECT_NE(addrs_slot0[0].opaque(), addrs_slot1[0].opaque());
}

TEST_F(CircularVmmPoolTest, BufferSizesArePreserved) {
  std::vector<uint64_t> buffer_sizes = {1024, 2048, 512};
  ASSERT_OK_AND_ASSIGN(auto pool,
                       CircularVmmPool::Create(executor_, buffer_sizes, 1));

  ASSERT_OK_AND_ASSIGN(auto addrs, pool->AcquireNextSlot(0));
  ASSERT_EQ(addrs.size(), 3);
  EXPECT_EQ(addrs[0].size(), 1024);
  EXPECT_EQ(addrs[1].size(), 2048);
  EXPECT_EQ(addrs[2].size(), 512);
}

TEST_F(CircularVmmPoolTest, SlotCycling) {
  std::vector<uint64_t> buffer_sizes = {1024};
  ASSERT_OK_AND_ASSIGN(auto pool,
                       CircularVmmPool::Create(executor_, buffer_sizes, 3));

  ASSERT_OK_AND_ASSIGN(auto addrs0, pool->AcquireNextSlot(0));
  ASSERT_OK_AND_ASSIGN(auto addrs1, pool->AcquireNextSlot(1));
  ASSERT_OK_AND_ASSIGN(auto addrs2, pool->AcquireNextSlot(2));

  EXPECT_NE(addrs0[0].opaque(), addrs1[0].opaque());
  EXPECT_NE(addrs1[0].opaque(), addrs2[0].opaque());
  EXPECT_NE(addrs0[0].opaque(), addrs2[0].opaque());

  ASSERT_OK_AND_ASSIGN(auto addrs3, pool->AcquireNextSlot(3));
  EXPECT_EQ(addrs0[0].opaque(), addrs3[0].opaque());
}

TEST_F(CircularVmmPoolTest, ReleaseAndAcquireWithStream) {
  std::vector<uint64_t> buffer_sizes = {4096};
  ASSERT_OK_AND_ASSIGN(auto pool,
                       CircularVmmPool::Create(executor_, buffer_sizes, 2));

  ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());

  ASSERT_OK_AND_ASSIGN(auto addrs0, pool->AcquireNextSlot(0));
  EXPECT_OK(pool->ReleaseSlot(stream.get(), 0));

  ASSERT_OK_AND_ASSIGN(auto addrs1, pool->AcquireNextSlot(1));
  EXPECT_OK(pool->ReleaseSlot(stream.get(), 1));

  EXPECT_OK(stream->BlockHostUntilDone());

  ASSERT_OK_AND_ASSIGN(auto addrs2, pool->AcquireNextSlot(2));
  EXPECT_EQ(addrs0[0].opaque(), addrs2[0].opaque());
}

TEST_F(CircularVmmPoolTest, InvalidNumSlots) {
  std::vector<uint64_t> buffer_sizes = {1024};
  auto result = CircularVmmPool::Create(executor_, buffer_sizes, 0);
  EXPECT_FALSE(result.ok());
}

TEST_F(CircularVmmPoolTest, EmptyBufferSizes) {
  std::vector<uint64_t> buffer_sizes = {};
  auto result = CircularVmmPool::Create(executor_, buffer_sizes, 2);
  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace stream_executor::gpu
