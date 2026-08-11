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

#include "xla/stream_executor/rocm/rocm_core_info_table.h"

#include <gtest/gtest.h>
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor {
namespace gpu {
namespace {

TEST(RocmCoreInfoTableTest, GetRocmFpusPerCore) {
  EXPECT_EQ(GetRocmFpusPerCore(RocmComputeCapability("gfx908")), 64);
  EXPECT_EQ(GetRocmFpusPerCore(RocmComputeCapability("gfx90a")), 128);
  EXPECT_EQ(GetRocmFpusPerCore(RocmComputeCapability("gfx942")), 128);
  EXPECT_EQ(GetRocmFpusPerCore(RocmComputeCapability("gfx950")), 128);
}

TEST(RocmCoreInfoTableTest, IgnoresArchFeatureSuffixes) {
  EXPECT_EQ(GetRocmFpusPerCore(RocmComputeCapability("gfx942:sramecc+:xnack-")),
            128);
}

TEST(RocmCoreInfoTableTest, UnknownArchKeepsPreviousDefault) {
  EXPECT_EQ(GetRocmFpusPerCore(RocmComputeCapability("gfx9999")), 128);
}

// Each entry should reproduce the vendor's published FP32 vector peak, since
// that is how it was derived: peak = cu_count * fpus * 2 * clock.
TEST(RocmCoreInfoTableTest, ReproducesPublishedFp32Peaks) {
  auto peak_tflops = [](int cu_count, int fpus, double clock_ghz) {
    return cu_count * fpus * 2 * clock_ghz / 1000.0;
  };

  EXPECT_NEAR(peak_tflops(120, GetRocmFpusPerCore(RocmComputeCapability(
                                   "gfx908")),
                          1.502),
              23.1, 0.2);
  EXPECT_NEAR(peak_tflops(104, GetRocmFpusPerCore(RocmComputeCapability(
                                   "gfx90a")),
                          1.7),
              45.3, 0.2);
  EXPECT_NEAR(peak_tflops(304, GetRocmFpusPerCore(RocmComputeCapability(
                                   "gfx942")),
                          2.1),
              163.4, 0.2);
  EXPECT_NEAR(peak_tflops(256, GetRocmFpusPerCore(RocmComputeCapability(
                                   "gfx950")),
                          2.4),
              157.3, 0.2);
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
