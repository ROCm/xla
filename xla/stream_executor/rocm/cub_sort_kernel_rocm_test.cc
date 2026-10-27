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

#include "xla/stream_executor/rocm/cub_sort_kernel_rocm.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <memory>
#include <optional>
#include <thread>
#include <tuple>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::rocm {
namespace {

using ::absl_testing::IsOk;

class CubSortKernelRocmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<Platform*> platform =
        PlatformManager::PlatformWithName("ROCM");
    ASSERT_THAT(platform, IsOk());
    platform_ = *platform;
    absl::StatusOr<StreamExecutor*> executor =
        platform_->ExecutorForDevice(0);
    ASSERT_THAT(executor, IsOk());
    executor_ = *executor;
    absl::StatusOr<std::unique_ptr<Stream>> stream =
        executor_->CreateStream(std::nullopt);
    ASSERT_THAT(stream, IsOk());
    stream_ = *std::move(stream);
  }

  hipStream_t hip_stream() {
    return static_cast<hipStream_t>(stream_->platform_specific_handle().stream);
  }

  // Keys with many duplicates (to check stability), no zeros or NaNs.
  static std::vector<float> MakeKeys(size_t n) {
    std::vector<float> keys(n);
    for (size_t i = 0; i < n; ++i) {
      keys[i] = static_cast<float>(1 + (i * 7919 + i / 13) % 61);
    }
    return keys;
  }

  // Sorts `batch_size` equal segments of keys (and of the values 0, 1, ... if
  // `with_values`) on the device, and checks them against std::stable_sort.
  void RunSort(size_t batch_size, size_t segment_size, bool descending,
               bool with_values) {
    const size_t n = batch_size * segment_size;
    std::vector<float> keys = MakeKeys(n);
    std::vector<uint32_t> values(n);
    std::iota(values.begin(), values.end(), 0);

    std::vector<uint32_t> expected_values = values;
    for (size_t b = 0; b < batch_size; ++b) {
      auto begin = expected_values.begin() + b * segment_size;
      std::stable_sort(begin, begin + segment_size,
                       [&](uint32_t a, uint32_t c) {
                         return descending ? keys[a] > keys[c]
                                           : keys[a] < keys[c];
                       });
    }
    std::vector<float> expected_keys(n);
    for (size_t i = 0; i < n; ++i) expected_keys[i] = keys[expected_values[i]];

    size_t temp_bytes = 0;
    if (with_values) {
      ASSERT_THAT((CubSortPairs<float, uint32_t>(
                      nullptr, temp_bytes, nullptr, nullptr, nullptr, nullptr,
                      n, descending, batch_size, nullptr)),
                  IsOk());
    } else {
      ASSERT_THAT(CubSortKeys<float>(nullptr, temp_bytes, nullptr, nullptr, n,
                                     descending, batch_size, nullptr),
                  IsOk());
    }

    DeviceAddress<float> d_keys_in = executor_->AllocateArray<float>(n);
    DeviceAddress<float> d_keys_out = executor_->AllocateArray<float>(n);
    DeviceAddress<uint32_t> d_values_in = executor_->AllocateArray<uint32_t>(n);
    DeviceAddress<uint32_t> d_values_out =
        executor_->AllocateArray<uint32_t>(n);
    DeviceAddress<uint8_t> d_temp =
        executor_->AllocateArray<uint8_t>(std::max<size_t>(temp_bytes, 1));
    absl::Cleanup cleanup = [&] {
      executor_->Deallocate(&d_keys_in);
      executor_->Deallocate(&d_keys_out);
      executor_->Deallocate(&d_values_in);
      executor_->Deallocate(&d_values_out);
      executor_->Deallocate(&d_temp);
    };
    ASSERT_THAT(stream_->Memcpy(&d_keys_in, keys.data(), n * sizeof(float)),
                IsOk());
    ASSERT_THAT(
        stream_->Memcpy(&d_values_in, values.data(), n * sizeof(uint32_t)),
        IsOk());

    if (with_values) {
      ASSERT_THAT((CubSortPairs<float, uint32_t>(
                      d_temp.opaque(), temp_bytes, d_keys_in.opaque(),
                      d_keys_out.opaque(), d_values_in.opaque(),
                      d_values_out.opaque(), n, descending, batch_size,
                      hip_stream())),
                  IsOk());
    } else {
      ASSERT_THAT(CubSortKeys<float>(d_temp.opaque(), temp_bytes,
                                     d_keys_in.opaque(), d_keys_out.opaque(),
                                     n, descending, batch_size, hip_stream()),
                  IsOk());
    }

    std::vector<float> out_keys(n);
    std::vector<uint32_t> out_values(n);
    ASSERT_THAT(stream_->Memcpy(out_keys.data(), d_keys_out, n * sizeof(float)),
                IsOk());
    if (with_values) {
      ASSERT_THAT(stream_->Memcpy(out_values.data(), d_values_out,
                                  n * sizeof(uint32_t)),
                  IsOk());
    }
    ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
    EXPECT_EQ(out_keys, expected_keys);
    if (with_values) EXPECT_EQ(out_values, expected_values);
  }

  Platform* platform_;
  StreamExecutor* executor_;
  std::unique_ptr<Stream> stream_;
};

class CubSortKernelRocmShapeTest
    : public CubSortKernelRocmTest,
      public ::testing::WithParamInterface<
          std::tuple<std::tuple<size_t, size_t>, bool, bool>> {};

TEST_P(CubSortKernelRocmShapeTest, MatchesStableSort) {
  auto [shape, descending, with_values] = GetParam();
  auto [batch_size, segment_size] = shape;
  RunSort(batch_size, segment_size, descending, with_values);
}

// Batches of short, medium and long segments, below and above the segment
// counts at which rocPRIM would partition the segments by size.
INSTANTIATE_TEST_SUITE_P(
    Shapes, CubSortKernelRocmShapeTest,
    ::testing::Combine(::testing::Values(std::make_tuple(1, 100000),
                                         std::make_tuple(3, 5000),
                                         std::make_tuple(256, 32768),
                                         std::make_tuple(4096, 16),
                                         std::make_tuple(4096, 100),
                                         std::make_tuple(4096, 600)),
                       ::testing::Bool(), ::testing::Bool()));

// A batched sort must only enqueue work: it must not wait for the work queued
// before it on the stream. The stream is held by a host callback that waits
// for a notification, which is only sent once the sort call has returned (or
// after a timeout, which fails the test).
TEST_F(CubSortKernelRocmTest, BatchedSortDoesNotWaitForTheStream) {
  constexpr size_t kBatchSize = 4096;
  constexpr size_t kSegmentSize = 600;
  constexpr size_t n = kBatchSize * kSegmentSize;
  size_t temp_bytes = 0;
  ASSERT_THAT(CubSortKeys<float>(nullptr, temp_bytes, nullptr, nullptr, n,
                                 /*descending=*/true, kBatchSize, nullptr),
              IsOk());
  DeviceAddress<float> d_keys_in = executor_->AllocateArray<float>(n);
  DeviceAddress<float> d_keys_out = executor_->AllocateArray<float>(n);
  DeviceAddress<uint8_t> d_temp = executor_->AllocateArray<uint8_t>(temp_bytes);
  absl::Cleanup cleanup = [&] {
    executor_->Deallocate(&d_keys_in);
    executor_->Deallocate(&d_keys_out);
    executor_->Deallocate(&d_temp);
  };
  std::vector<float> keys = MakeKeys(n);
  ASSERT_THAT(stream_->Memcpy(&d_keys_in, keys.data(), n * sizeof(float)),
              IsOk());

  absl::Notification release_stream;
  ASSERT_THAT(stream_->DoHostCallback(
                  [&release_stream] { release_stream.WaitForNotification(); }),
              IsOk());

  absl::Notification sort_returned;
  absl::Status sort_status;
  std::thread sorter([&] {
    sort_status = CubSortKeys<float>(
        d_temp.opaque(), temp_bytes, d_keys_in.opaque(), d_keys_out.opaque(), n,
        /*descending=*/true, kBatchSize, hip_stream());
    sort_returned.Notify();
  });
  bool returned_while_blocked =
      sort_returned.WaitForNotificationWithTimeout(absl::Seconds(30));
  release_stream.Notify();
  sorter.join();
  EXPECT_TRUE(returned_while_blocked)
      << "the sort waited for the work queued before it on the stream";
  ASSERT_THAT(sort_status, IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());

  std::vector<float> out_keys(n);
  ASSERT_THAT(stream_->Memcpy(out_keys.data(), d_keys_out, n * sizeof(float)),
              IsOk());
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
  for (size_t b = 0; b < kBatchSize; ++b) {
    auto begin = keys.begin() + b * kSegmentSize;
    std::sort(begin, begin + kSegmentSize, std::greater<float>());
  }
  EXPECT_EQ(out_keys, keys);
}

}  // namespace
}  // namespace stream_executor::rocm
