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

#include "xla/stream_executor/rocm/rocm_pcie_bandwidth.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace stream_executor::gpu {
namespace {

TEST(ComputePcieBandwidthTest, Gen1x1) {
  // Gen1: 2500 MT/s, 8b/10b encoding (80%), 1 lane
  // 2500e6 * 1 / 8 * 0.8 = 250 MB/s
  int64_t bw = ComputePcieBandwidthFromSpeedAndWidth(2500, 1);
  EXPECT_EQ(bw, 250 * 1000 * 1000);
}

TEST(ComputePcieBandwidthTest, Gen2x16) {
  // Gen2: 5000 MT/s, 8b/10b encoding (80%), 16 lanes
  // 5000e6 * 16 / 8 * 0.8 = 8000 MB/s = 8 GB/s
  int64_t bw = ComputePcieBandwidthFromSpeedAndWidth(5000, 16);
  EXPECT_EQ(bw, static_cast<int64_t>(8000) * 1000 * 1000);
}

TEST(ComputePcieBandwidthTest, Gen3x16) {
  // Gen3: 8000 MT/s, 128b/130b encoding, 16 lanes
  // 8000e6 * 16 / 8 * (128/130) = ~15.754 GB/s
  int64_t bw = ComputePcieBandwidthFromSpeedAndWidth(8000, 16);
  // Expected: 8000e6 * 16 / 8 * 128/130 = 15753846153.846...
  EXPECT_GT(bw, 15700LL * 1000 * 1000);
  EXPECT_LT(bw, 15800LL * 1000 * 1000);
}

TEST(ComputePcieBandwidthTest, Gen4x16) {
  // Gen4: 16000 MT/s, 128b/130b encoding, 16 lanes
  // 16000e6 * 16 / 8 * (128/130) = ~31.508 GB/s
  int64_t bw = ComputePcieBandwidthFromSpeedAndWidth(16000, 16);
  EXPECT_GT(bw, 31400LL * 1000 * 1000);
  EXPECT_LT(bw, 31600LL * 1000 * 1000);
}

TEST(ComputePcieBandwidthTest, Gen5x16) {
  // Gen5: 32000 MT/s, 128b/130b encoding, 16 lanes
  // 32000e6 * 16 / 8 * (128/130) = ~63.015 GB/s
  int64_t bw = ComputePcieBandwidthFromSpeedAndWidth(32000, 16);
  EXPECT_GT(bw, 62900LL * 1000 * 1000);
  EXPECT_LT(bw, 63100LL * 1000 * 1000);
}

TEST(ComputePcieBandwidthTest, Gen4x8) {
  // Gen4 x8: half bandwidth of Gen4 x16
  int64_t bw_x16 = ComputePcieBandwidthFromSpeedAndWidth(16000, 16);
  int64_t bw_x8 = ComputePcieBandwidthFromSpeedAndWidth(16000, 8);
  EXPECT_EQ(bw_x8, bw_x16 / 2);
}

TEST(ComputePcieBandwidthTest, ZeroInputs) {
  EXPECT_EQ(ComputePcieBandwidthFromSpeedAndWidth(0, 16), 0);
  EXPECT_EQ(ComputePcieBandwidthFromSpeedAndWidth(16000, 0), 0);
  EXPECT_EQ(ComputePcieBandwidthFromSpeedAndWidth(0, 0), 0);
}

}  // namespace
}  // namespace stream_executor::gpu
