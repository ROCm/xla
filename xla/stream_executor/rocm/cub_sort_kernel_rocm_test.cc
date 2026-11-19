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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status_matchers.h"
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
    ASSERT_OK_AND_ASSIGN(platform_, PlatformManager::PlatformWithName("ROCM"));
    ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
  }

  hipStream_t hip_stream() const {
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
      std::stable_sort(
          begin, begin + segment_size, [&](uint32_t a, uint32_t c) {
            return descending ? keys[a] > keys[c] : keys[a] < keys[c];
          });
    }
    std::vector<float> expected_keys(n);
    for (size_t i = 0; i < n; ++i) {
      expected_keys[i] = keys[expected_values[i]];
    }

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
      ASSERT_THAT(
          (CubSortPairs<float, uint32_t>(
              d_temp.opaque(), temp_bytes, d_keys_in.opaque(),
              d_keys_out.opaque(), d_values_in.opaque(), d_values_out.opaque(),
              n, descending, batch_size, hip_stream())),
          IsOk());
    } else {
      ASSERT_THAT(CubSortKeys<float>(d_temp.opaque(), temp_bytes,
                                     d_keys_in.opaque(), d_keys_out.opaque(), n,
                                     descending, batch_size, hip_stream()),
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
    if (with_values) {
      EXPECT_EQ(out_values, expected_values);
    }
  }

  Platform* platform_ = nullptr;
  StreamExecutor* executor_ = nullptr;
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

// A single sort, and batches of few and many short, medium and long segments.
INSTANTIATE_TEST_SUITE_P(
    Shapes, CubSortKernelRocmShapeTest,
    ::testing::Combine(::testing::Values(std::make_tuple(1, 100000),
                                         std::make_tuple(3, 5000),
                                         std::make_tuple(256, 32768),
                                         std::make_tuple(4096, 16),
                                         std::make_tuple(4096, 100),
                                         std::make_tuple(4096, 600)),
                       ::testing::Bool(), ::testing::Bool()));

}  // namespace
}  // namespace stream_executor::rocm
