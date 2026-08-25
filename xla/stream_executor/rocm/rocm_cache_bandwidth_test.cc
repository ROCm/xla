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

#include "xla/stream_executor/rocm/rocm_cache_bandwidth.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor::gpu {
namespace {

// MI300X: 8 XCDs at a 2100 MHz peak engine clock. The CDNA3 whitepaper states
// both aggregate figures outright, which makes them a citable check on the
// width constants rather than a restatement of the implementation.
constexpr int kMI300XNumXcd = 8;
constexpr double kMI300XClockGhz = 2.1;

// Tolerance covers the whitepaper's own rounding to three significant figures.
constexpr double kToleranceFraction = 0.01;

testing::AssertionResult IsNear(int64_t actual, double expected) {
  double diff = actual > expected ? actual - expected : expected - actual;
  if (diff <= expected * kToleranceFraction) return testing::AssertionSuccess();
  return testing::AssertionFailure()
         << actual << " is not within " << kToleranceFraction * 100
         << "% of " << expected;
}

TEST(RocmCacheBandwidthTest, MatchesCdna3WhitepaperL2Figure) {
  // CDNA3 whitepaper p.9: the AMD CDNA 3 architecture "has collectively up to
  // eight instances and up to 34.4 TB/s aggregate read bandwidth".
  EXPECT_TRUE(IsNear(
      GetRocmL2CacheBandwidth(RocmComputeCapability("gfx942"), kMI300XNumXcd,
                              kMI300XClockGhz),
      34.4e12));
}

TEST(RocmCacheBandwidthTest, MatchesCdna3WhitepaperInfinityCacheFigure) {
  // CDNA3 whitepaper p.10: "The peak bandwidth from the Infinity Cache is an
  // astounding 17.2 TB/s".
  EXPECT_TRUE(
      IsNear(GetRocmLastLevelCacheBandwidth(RocmComputeCapability("gfx942"),
                                            kMI300XClockGhz),
             17.2e12));
}

TEST(RocmCacheBandwidthTest, Cdna4SharesCdna3Widths) {
  // The CDNA4 whitepaper describes the same channel counts and widths, so the
  // same clock must yield the same bandwidth on gfx950.
  EXPECT_EQ(GetRocmL2CacheBandwidth(RocmComputeCapability("gfx950"),
                                    kMI300XNumXcd, kMI300XClockGhz),
            GetRocmL2CacheBandwidth(RocmComputeCapability("gfx942"),
                                    kMI300XNumXcd, kMI300XClockGhz));
  EXPECT_EQ(GetRocmLastLevelCacheBandwidth(RocmComputeCapability("gfx950"),
                                           kMI300XClockGhz),
            GetRocmLastLevelCacheBandwidth(RocmComputeCapability("gfx942"),
                                           kMI300XClockGhz));
}

TEST(RocmCacheBandwidthTest, L2ScalesWithEngineClockAcrossSkus) {
  // MI350X and MI355X share gfx950 and differ only in peak engine clock
  // (2200 vs 2400 MHz), so their L2 bandwidths must differ by that ratio.
  // This is the per-SKU behavior the queried clock exists to provide.
  RocmComputeCapability gfx950("gfx950");
  int64_t mi350x = GetRocmL2CacheBandwidth(gfx950, 8, 2.2);
  int64_t mi355x = GetRocmL2CacheBandwidth(gfx950, 8, 2.4);

  EXPECT_TRUE(IsNear(mi350x, 36.0e12));
  EXPECT_TRUE(IsNear(mi355x, 39.3e12));
  EXPECT_GT(mi355x, mi350x);
}

TEST(RocmCacheBandwidthTest, UnmodeledArchitecturesReturnZero) {
  // Zero means "no information", which keeps the cost model on its legacy
  // memory-bandwidth scaling instead of inventing a rate.
  for (const char* gfx : {"gfx908", "gfx90a", "gfx1030", "gfx1100", "gfx1201"}) {
    RocmComputeCapability cc{gfx};
    EXPECT_EQ(GetRocmL2CacheBandwidth(cc, 8, 2.1), 0) << gfx;
    EXPECT_EQ(GetRocmLastLevelCacheBandwidth(cc, 2.1), 0) << gfx;
  }
}

TEST(RocmCacheBandwidthTest, ImplausibleInputsReturnZero) {
  RocmComputeCapability gfx950("gfx950");
  EXPECT_EQ(GetRocmL2CacheBandwidth(gfx950, 0, 2.2), 0);
  EXPECT_EQ(GetRocmL2CacheBandwidth(gfx950, -1, 2.2), 0);
  EXPECT_EQ(GetRocmL2CacheBandwidth(gfx950, 8, 0.0), 0);
  // A caller passing MHz instead of GHz must not produce a plausible-looking
  // number.
  EXPECT_EQ(GetRocmL2CacheBandwidth(gfx950, 8, 2200.0), 0);
}

TEST(RocmCacheBandwidthTest, LastLevelFallsBackToDocumentedFabricClock) {
  // AMDSMI_CLK_TYPE_DF reports a zero peak clock on MI350X, so an unavailable
  // clock is the common case rather than an error. It must still produce the
  // documented bandwidth instead of 0, otherwise the tier silently disappears.
  RocmComputeCapability gfx950("gfx950");
  EXPECT_TRUE(IsNear(GetRocmLastLevelCacheBandwidth(gfx950, 0.0), 17.2e12));
  // A caller passing MHz is a bug, not a missing value, but it lands in the
  // same guard and gets the documented figure rather than a 1000x overestimate.
  EXPECT_TRUE(IsNear(GetRocmLastLevelCacheBandwidth(gfx950, 2100.0), 17.2e12));
  // An unmodeled architecture still gets nothing.
  EXPECT_EQ(GetRocmLastLevelCacheBandwidth(RocmComputeCapability("gfx90a"), 0.0),
            0);
}

}  // namespace
}  // namespace stream_executor::gpu
