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

#include "xla/stream_executor/rocm/rocm_flops_table.h"

#include <optional>

#include <gtest/gtest.h>

namespace stream_executor {
namespace gpu {
namespace {

TEST(RocmFlopsTableTest, KnownArchReturnsRates) {
  std::optional<RocmFlopsPerCuPerCycle> rates =
      GetRocmFlopsPerCuPerCycle("gfx942");
  ASSERT_TRUE(rates.has_value());
  EXPECT_TRUE(rates->has_matrix_unit);
  EXPECT_GT(rates->bf16_matrix, 0.0);
  EXPECT_GT(rates->f32_vector, 0.0);
}

TEST(RocmFlopsTableTest, UnknownArchReturnsNullopt) {
  EXPECT_FALSE(GetRocmFlopsPerCuPerCycle("gfx0000").has_value());
  EXPECT_FALSE(GetRocmFlopsPerCuPerCycle("").has_value());
}

TEST(RocmFlopsTableTest, PeakUsesMatrixRateForCdna) {
  // gfx942 has a matrix unit; peak should be based on bf16_matrix.
  const uint32_t cu_count = 304;      // MI300X-ish
  const double clock_hz = 2.1e9;      // 2.1 GHz
  std::optional<RocmFlopsPerCuPerCycle> rates =
      GetRocmFlopsPerCuPerCycle("gfx942");
  ASSERT_TRUE(rates.has_value());

  double expected =
      static_cast<double>(cu_count) * rates->bf16_matrix * clock_hz / 1e12;
  double actual = GetRocmPeakTeraflopsPerSecond("gfx942", cu_count, clock_hz);
  EXPECT_NEAR(actual, expected, expected * 1e-9);
  EXPECT_GT(actual, 0.0);
}

TEST(RocmFlopsTableTest, PeakUsesVectorRateForRdna) {
  // gfx1100 has no matrix unit; peak should be based on f32_vector.
  const uint32_t cu_count = 96;
  const double clock_hz = 2.3e9;
  std::optional<RocmFlopsPerCuPerCycle> rates =
      GetRocmFlopsPerCuPerCycle("gfx1100");
  ASSERT_TRUE(rates.has_value());
  EXPECT_FALSE(rates->has_matrix_unit);

  double expected =
      static_cast<double>(cu_count) * rates->f32_vector * clock_hz / 1e12;
  double actual = GetRocmPeakTeraflopsPerSecond("gfx1100", cu_count, clock_hz);
  EXPECT_NEAR(actual, expected, expected * 1e-9);
}

TEST(RocmFlopsTableTest, ZeroInputsReturnZero) {
  EXPECT_EQ(GetRocmPeakTeraflopsPerSecond("gfx942", 0, 2.1e9), 0.0);
  EXPECT_EQ(GetRocmPeakTeraflopsPerSecond("gfx942", 304, 0.0), 0.0);
}

TEST(RocmFlopsTableTest, UnknownArchPeakReturnsZero) {
  EXPECT_EQ(GetRocmPeakTeraflopsPerSecond("gfx0000", 304, 2.1e9), 0.0);
}

// Pins the headline (matrix bf16, or vector fp32 for RDNA) peak against AMD's
// published per-product specs, using each product's real CU count and clock.
// This guards the back-calculated per-CU-per-cycle constants against edits.
TEST(RocmFlopsTableTest, HeadlinePeakMatchesPublishedSpecs) {
  struct Spec {
    absl::string_view gfx;
    uint32_t cu;
    double clock_hz;
    double expected_tflops;  // AMD-published headline peak
    double tol_tflops;
  };
  const Spec specs[] = {
      // MI100 FP16 matrix 184.6 TFLOPS (headline is bf16=92.3 on CDNA1).
      {"gfx908", 120, 1.502e9, 92.3, 0.5},
      // MI250 per-GCD BF16 matrix 362.1 TFLOPS.
      {"gfx90a", 104, 1.7e9, 362.1, 0.5},
      // MI300X BF16 matrix 1307.4 TFLOPS.
      {"gfx942", 304, 2.1e9, 1307.4, 1.0},
      // MI355X dense BF16 matrix "2.5 PFLOPS" @ 2.4 GHz. AMD rounds to 1 sig
      // fig; exact is 256*4096*2.4e9 = 2516.6 TFLOPS, so allow the rounding gap.
      {"gfx950", 256, 2.4e9, 2500.0, 20.0},
      // RX 6900 XT FP32 vector 23.0 TFLOPS.
      {"gfx1030", 80, 2.25e9, 23.0, 0.2},
      // RX 7900 XTX FP32 vector 61.4 TFLOPS.
      {"gfx1100", 96, 2.5e9, 61.4, 0.2},
  };
  for (const Spec& s : specs) {
    double actual = GetRocmPeakTeraflopsPerSecond(s.gfx, s.cu, s.clock_hz);
    EXPECT_NEAR(actual, s.expected_tflops, s.tol_tflops)
        << "arch " << s.gfx;
  }
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
