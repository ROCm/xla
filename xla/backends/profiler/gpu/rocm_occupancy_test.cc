/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

// Deviceless tests for the theoretical-occupancy model.
//
// This file has no ROCm dependency at all, by design: it runs in CPU CI on
// every platform. That is the only way this arithmetic gets continuous
// protection, since the collector it used to live in cannot be built without
// a ROCm toolchain.
//
// The golden table below is the load-bearing test. Its expectations come from
// EXTERNAL oracles, not from restating the implementation's own arithmetic:
//   H = measured with hipOccupancyMaxActiveBlocksPerMultiprocessor on a live
//       MI300X.
//   S = derived from the LLVM source functions this model ports
//       (AMDGPUBaseInfo.cpp / AMDGPUSubtarget.cpp / AMDGPUTargetParser.cpp).
//
// Do not regenerate these numbers from the code under test, and do not use
// llc's "; Occupancy:" comment as an oracle for the LDS or workgroup terms --
// the local /opt/rocm/llvm applies no LDS granule at all, and the asm printer
// rounds up across EUs. Both traps are documented at GetOccupancy().

#include "xla/backends/profiler/gpu/rocm_occupancy.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"

namespace xla {
namespace profiler {
namespace {

constexpr uint32_t kGfx90a = 90010;
constexpr uint32_t kGfx942 = 90402;
constexpr uint32_t kGfx950 = 90500;

RocmDeviceOccupancyParams MakeParams(uint32_t gfx, uint32_t arch,
                                     uint32_t accum, uint32_t sgpr,
                                     uint32_t block, uint32_t smem) {
  RocmDeviceOccupancyParams p;
  p.gfx_target_version = gfx;
  p.arch_vgpr_count = arch;
  p.accum_vgpr_count = accum;
  p.sgpr_count = sgpr;
  p.block_size = block;
  p.smem_bytes = smem;
  return p;
}

// ===========================================================================
// The golden table
// ===========================================================================

struct GoldenRow {
  const char* name;
  uint32_t gfx;
  uint32_t arch_vgpr;
  uint32_t accum_vgpr;
  uint32_t sgpr;
  uint32_t block;
  uint32_t smem;
  double expected_pct;
  OccupancyLimiter expected_limiter;
  const char* what_it_pins;
};

const GoldenRow kGoldenTable[] = {
    // --- The row that would have caught the CRITICAL bug on day one. An MFMA
    // kernel with 5 arch VGPRs and 128 AGPRs: charging arch only reports
    // 100%, the hardware truth measured on MI300X is 37.5%. alignTo(5,4)+128
    // = 136 -> alignTo(136,8) = 136 -> 512/136 = 3 waves/SIMD.
    {"mfma_agpr", kGfx942, 5, 128, 16, 256, 0, 37.5, OccupancyLimiter::kVGPR,
     "AGPRs excluded from the unified VGPR file (reports 100% before the fix)"},
    {"mfma_balanced", kGfx942, 128, 128, 16, 256, 0, 25.0,
     OccupancyLimiter::kVGPR, "same defect, 2x overstatement"},
    {"flash_attn", kGfx942, 168, 248, 16, 256, 0, 12.5, OccupancyLimiter::kVGPR,
     "same defect, 3x overstatement"},

    // --- Workgroup quantization. Hardware cannot resident a partial
    // workgroup; a 320-thread block is 5 waves, and only 6 such blocks fit in
    // 32 slots, leaving 2 slots idle. Measured on MI300X at 6 blocks/CU.
    {"wg320", kGfx942, 4, 0, 14, 320, 0, 93.75, OccupancyLimiter::kWorkgroup,
     "no workgroup quantization (reports 100% before the fix)"},
    {"wg768", kGfx942, 4, 0, 14, 768, 0, 75.0, OccupancyLimiter::kWorkgroup,
     "same defect at a larger block size"},

    // --- The register bound must be re-quantized to whole workgroups too.
    // 176 VGPRs gives 2 waves/SIMD = 8 slots, but a 384-thread block needs 6
    // waves, so exactly one block fits and 2 slots go unused.
    {"vgpr_wg_quant", kGfx942, 176, 0, 16, 384, 0, 18.75,
     OccupancyLimiter::kVGPR,
     "VGPR budget not rounded down to whole workgroups (reports 25%)"},

    // --- VGPR allocation granule. 100 arch VGPRs are charged as 104.
    {"granule100", kGfx942, 100, 0, 16, 256, 0, 50.0, OccupancyLimiter::kVGPR,
     "missing alignTo(NumVGPRs, 8) (reports 62.5%)"},

    // --- LDS allocation granule. alignTo(13000, 512) = 13312, and
    // 65536/13312 = 4 workgroups, not the 5 an unrounded divide gives.
    {"lds_unaligned", kGfx942, 4, 0, 14, 256, 13000, 50.0,
     OccupancyLimiter::kLDS, "missing alignTo(LDS, granule) (reports 62.5%)"},
    {"lds32k", kGfx942, 3, 0, 9, 256, 32768, 25.0, OccupancyLimiter::kLDS,
     "LDS baseline, measured"},

    // --- The best cross-architecture assertion available without MI355
    // silicon: identical inputs, 4x the answer, because gfx950's LDS pool is
    // 160 KiB rather than 64 KiB.
    {"lds40k_gfx942", kGfx942, 3, 0, 9, 256, 40960, 12.5,
     OccupancyLimiter::kLDS, "gfx942 LDS pool size"},
    {"lds40k_gfx950", kGfx950, 3, 0, 9, 256, 40960, 50.0,
     OccupancyLimiter::kLDS,
     "gfx950 LDS pool size, same input as the row above"},
    // Note 50.0 and not 62.5. alignTo(32768, 1280) = 33280 and
    // 163840/33280 = 4. The 62.5% figure is 163840/32768 = 5, the
    // un-granulated answer -- which is exactly what the local /opt/rocm/llvm
    // oracle produces, and exactly why that oracle is invalid for this term.
    {"lds32k_gfx950", kGfx950, 3, 0, 9, 256, 32768, 50.0,
     OccupancyLimiter::kLDS, "gfx950's 1280-byte LDS granule"},

    // --- The SGPR term is live on gfx9 (isSGPROccupancyLimited is Major < 10).
    // alignTo(112,16) = 112, and 800/112 = 7 waves/SIMD.
    {"sgpr112", kGfx942, 4, 0, 112, 256, 0, 87.5, OccupancyLimiter::kSGPR,
     "missing SGPR bound (reports 100%)"},

    // --- Guards against someone adding a barrier term that binds. One-wave
    // workgroups get the full 32 slots; getMaxWorkGroupsPerCU returns MaxWaves
    // rather than 16 in that case.
    {"wg64_full", kGfx942, 4, 0, 14, 64, 0, 100.0, OccupancyLimiter::kNone,
     "the 16-workgroup barrier cap must not bind on CDNA"},

    // --- LDS larger than a CU has. LLVM's contract is one workgroup, not
    // zero; the pre-fix code returned empty stats and emitted nothing.
    {"lds_over_cap", kGfx942, 4, 0, 14, 256, 81920, 12.5,
     OccupancyLimiter::kLDS, "smem > LDS capacity returned {} before the fix"},

    // --- gfx90a shares every constant with gfx942, so this row is really an
    // assertion about the 90010 encoding: the step renders as hex in the
    // agent *name* only, so gfx90a is 90010 and must not fall through to the
    // inexact generic branch.
    {"mfma_agpr_gfx90a", kGfx90a, 5, 128, 16, 256, 0, 37.5,
     OccupancyLimiter::kVGPR, "the gfx90a target-version encoding"},

    // --- A block size that is not a multiple of the wavefront. 100 threads
    // occupy 2 whole waves, so 16 workgroups fill all 32 slots: the limiter is
    // correctly kNone (no resource is short), yet occupancy_pct is 78.125,
    // not 100. That pairing looks contradictory and is not -- occupancy_pct
    // counts THREADS per CU (CUPTI's definition, see the UNITS note in the
    // header), and 28 of every 128 thread slots are idle inside the partial
    // wave. The deficit is intra-wave waste, which no amount of resource
    // tuning recovers; only a block size that is a multiple of 64 does.
    // Every other row here uses a multiple of 64, so without this one the
    // interaction between partial waves and limiter attribution is untested.
    {"partial_wave", kGfx942, 4, 0, 14, 100, 0, 78.125, OccupancyLimiter::kNone,
     "partial-wave thread waste, which the limiter cannot express"},
};

TEST(RocmOccupancyGoldenTest, AllRows) {
  for (const GoldenRow& row : kGoldenTable) {
    SCOPED_TRACE(absl::StrCat(row.name, " pins: ", row.what_it_pins));
    std::optional<OccupancyStats> occ =
        GetOccupancy(MakeParams(row.gfx, row.arch_vgpr, row.accum_vgpr,
                                row.sgpr, row.block, row.smem),
                     /*cu_count=*/304);
    ASSERT_TRUE(occ.has_value());
    EXPECT_DOUBLE_EQ(occ->occupancy_pct, row.expected_pct);
    EXPECT_EQ(occ->limiter, row.expected_limiter)
        << "got " << OccupancyLimiterName(occ->limiter) << ", want "
        << OccupancyLimiterName(row.expected_limiter);
  }
}

// The derived fields must stay consistent with the percentage, so that a
// future edit cannot fix one and silently break the others.
TEST(RocmOccupancyGoldenTest, DerivedFieldsAreConsistent) {
  for (const GoldenRow& row : kGoldenTable) {
    SCOPED_TRACE(row.name);
    std::optional<OccupancyStats> occ =
        GetOccupancy(MakeParams(row.gfx, row.arch_vgpr, row.accum_vgpr,
                                row.sgpr, row.block, row.smem),
                     /*cu_count=*/304);
    ASSERT_TRUE(occ.has_value());
    // All three golden targets are 4 SIMDs/CU, 8 waves/SIMD, 64-wide waves.
    EXPECT_DOUBLE_EQ(occ->waves_per_simd * 4,
                     static_cast<double>(occ->active_waves_per_cu));
    EXPECT_LE(occ->active_waves_per_cu, 32u);
    EXPECT_GE(occ->active_blocks_per_cu, 1u);
    EXPECT_EQ(occ->min_grid_size, occ->active_blocks_per_cu * 304);
    EXPECT_GT(occ->occupancy_pct, 0.0);
    EXPECT_LE(occ->occupancy_pct, 100.0);
  }
}

// ===========================================================================
// Graceful degradation -- nullopt must mean "emit nothing", never zero
// ===========================================================================

TEST(RocmOccupancyTest, UnknownTargetReturnsNullopt) {
  // gfx1100. Wave32, a 1536-entry VGPR file and different granules; we have
  // no validated constants, so the caller must emit no occupancy at all.
  EXPECT_FALSE(
      GetOccupancy(MakeParams(110000, 40, 0, 20, 256, 0), 304).has_value());
  EXPECT_FALSE(LookupTargetConstants(110000).has_value());
  EXPECT_FALSE(LookupTargetConstants(100300).has_value());
  EXPECT_FALSE(LookupTargetConstants(120000).has_value());
  EXPECT_FALSE(LookupTargetConstants(0).has_value());
}

TEST(RocmOccupancyTest, UnknownGfx9DegradesInexactly) {
  // gfx906 (MI50). Not in the table, but still gfx9: fall back to the
  // pre-GFX90A LLVM values, which under-report rather than over-report.
  std::optional<AmdGpuTargetConstants> tc = LookupTargetConstants(90600);
  ASSERT_TRUE(tc.has_value());
  EXPECT_FALSE(tc->exact);
  EXPECT_FALSE(tc->unified_vgpr_file);
  EXPECT_EQ(tc->total_vgprs, 256u);
  EXPECT_EQ(tc->vgpr_granule, 4u);
  EXPECT_EQ(tc->max_waves_per_simd, 10u);
}

TEST(RocmOccupancyTest, MissingInputsReturnNullopt) {
  // No symbol data at all -- the code-object lookup missed. Zero is a wrong
  // answer here, not a conservative one.
  EXPECT_FALSE(
      GetOccupancy(MakeParams(kGfx942, 0, 0, 16, 256, 0), 304).has_value());
  // No block size.
  EXPECT_FALSE(
      GetOccupancy(MakeParams(kGfx942, 64, 0, 16, 0, 0), 304).has_value());
  // A workgroup larger than a CU's entire wave-slot budget is not a geometry
  // the hardware would have accepted.
  EXPECT_FALSE(
      GetOccupancy(MakeParams(kGfx942, 64, 0, 16, 4096, 0), 304).has_value());
}

TEST(RocmOccupancyTest, UnknownCuCountLeavesMinGridSizeZero) {
  std::optional<OccupancyStats> occ =
      GetOccupancy(MakeParams(kGfx942, 5, 128, 16, 256, 0), /*cu_count=*/0);
  ASSERT_TRUE(occ.has_value());
  EXPECT_EQ(occ->min_grid_size, 0u);
  EXPECT_DOUBLE_EQ(occ->occupancy_pct, 37.5);
}

// ===========================================================================
// UnifiedVgprCount -- the gfx908 trap
// ===========================================================================

TEST(RocmOccupancyTest, UnifiedVgprCountSumsOnUnifiedTargets) {
  const AmdGpuTargetConstants tc = *LookupTargetConstants(kGfx942);
  // alignTo(5,4) + 128. This is the CRITICAL fix in one expression.
  EXPECT_EQ(UnifiedVgprCount(tc, 5, 128), 136u);
  EXPECT_EQ(UnifiedVgprCount(tc, 128, 128), 256u);
  EXPECT_EQ(UnifiedVgprCount(tc, 168, 248), 416u);
  // accum == 0 must take the max() branch, not add zero to a rounded-up arch.
  EXPECT_EQ(UnifiedVgprCount(tc, 100, 0), 100u);
}

TEST(RocmOccupancyTest, UnifiedVgprCountDoesNotDoubleCountNonUnified) {
  // On gfx908 the SDK's accum_vgpr_count() returns arch_vgpr_count(), so an
  // unconditional sum reports exactly 2x. The generic gfx9 fallback is
  // non-unified, so max() protects us.
  const AmdGpuTargetConstants tc = *LookupTargetConstants(90800);
  ASSERT_FALSE(tc.unified_vgpr_file);
  EXPECT_EQ(UnifiedVgprCount(tc, 128, 128), 128u);
}

// ===========================================================================
// Version-skew guard
// ===========================================================================

// On a unified-file target the hardware allocation is a multiple of the
// 8-register granule. A sum that is not one means the installed SDK decoded
// the code object with a scheme we do not recognise, and correct arithmetic
// on wrong input is still wrong.
TEST(RocmOccupancyTest, NonGranulatedUnifiedCountIsSuppressed) {
  // alignTo(4,4) + 1 = 5, not a multiple of 8.
  EXPECT_FALSE(
      GetOccupancy(MakeParams(kGfx942, 4, 1, 16, 256, 0), 304).has_value());
}

// ...but the guard must NOT fire on the max() branch. arch counts are
// granulated to 4, not 8, so ordinary low-register kernels reach it with an
// odd multiple of 4 and must still be modelled. This is the case that makes
// wg320, granule100 and every other accum-free row work.
TEST(RocmOccupancyTest, GuardDoesNotFireWithoutAgprs) {
  for (uint32_t arch : {4u, 12u, 20u, 36u, 100u, 172u}) {
    SCOPED_TRACE(absl::StrCat("arch_vgpr_count=", arch));
    EXPECT_TRUE(GetOccupancy(MakeParams(kGfx942, arch, 0, 16, 256, 0), 304)
                    .has_value());
  }
}

// ===========================================================================
// Degenerate input
//
// The version-skew guard exists because the SDK is known to hand back register
// counts we cannot decode. That makes "garbage in" a real path, not a
// hypothetical one, and the model must degrade rather than crash: it runs
// inside the profiled process, so a SIGFPE here takes down the user's training
// job. These rows pin the saturating arithmetic in AlignTo/SaturatingAdd.
// ===========================================================================

TEST(RocmOccupancyTest, HugeRegisterCountsDoNotCrashOrOverReport) {
  constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();

  // Wrapping AlignTo(kMax, 8) == 0 divided by zero in the VGPR bound. The
  // accum=0 path takes max(), so the skew guard does not intercept it.
  std::optional<OccupancyStats> vgpr =
      GetOccupancy(MakeParams(kGfx942, kMax, 0, 16, 256, 0), 304);
  ASSERT_TRUE(vgpr.has_value());
  EXPECT_EQ(vgpr->active_blocks_per_cu, 1u)
      << "an impossible register demand must floor at one workgroup";

  // The sum branch saturates, and kMax % 8 != 0, so the skew guard rejects it
  // outright -- the stronger of the two outcomes.
  EXPECT_FALSE(
      GetOccupancy(MakeParams(kGfx942, kMax, 1, 16, 256, 0), 304).has_value());

  // Wrapping AlignTo in the LDS bound made lds_per_wg 0, which read as "LDS
  // never binds" and reported 100%. It must read as "LDS always binds".
  std::optional<OccupancyStats> lds =
      GetOccupancy(MakeParams(kGfx942, 4, 0, 16, 256, kMax - 255), 304);
  ASSERT_TRUE(lds.has_value());
  EXPECT_EQ(lds->active_blocks_per_cu, 1u);
  EXPECT_EQ(lds->limiter, OccupancyLimiter::kLDS);
  EXPECT_LT(lds->occupancy_pct, 100.0)
      << "an impossible LDS demand must not report full occupancy";
}

// ===========================================================================
// Target constants
// ===========================================================================

TEST(RocmOccupancyTest, TargetConstantsMatchLlvm) {
  const AmdGpuTargetConstants gfx942 = *LookupTargetConstants(kGfx942);
  EXPECT_TRUE(gfx942.exact);
  EXPECT_TRUE(gfx942.unified_vgpr_file);
  EXPECT_TRUE(gfx942.sgpr_limited);  // IsaVersion.Major < 10
  EXPECT_EQ(gfx942.max_waves_per_simd, 8u);
  EXPECT_EQ(gfx942.simd_per_cu, 4u);
  EXPECT_EQ(gfx942.wave_front_size, 64u);
  EXPECT_EQ(gfx942.total_vgprs, 512u);
  EXPECT_EQ(gfx942.vgpr_granule, 8u);
  EXPECT_EQ(gfx942.arch_vgpr_granule, 4u);
  EXPECT_EQ(gfx942.lds_per_cu, 65536u);
  EXPECT_EQ(gfx942.lds_granule, 512u);
  EXPECT_EQ(gfx942.total_sgprs, 800u);
  EXPECT_EQ(gfx942.sgpr_granule, 16u);

  // gfx950 differs from gfx942 in exactly two constants.
  const AmdGpuTargetConstants gfx950 = *LookupTargetConstants(kGfx950);
  EXPECT_TRUE(gfx950.exact);
  EXPECT_EQ(gfx950.lds_per_cu, 163840u);
  EXPECT_EQ(gfx950.lds_granule, 1280u);
  EXPECT_EQ(gfx950.total_vgprs, gfx942.total_vgprs);
  EXPECT_EQ(gfx950.max_waves_per_simd, gfx942.max_waves_per_simd);

  // gfx90a: 90010, decimal decode, step 10.
  const AmdGpuTargetConstants gfx90a = *LookupTargetConstants(kGfx90a);
  EXPECT_TRUE(gfx90a.exact);
  EXPECT_EQ(gfx90a.lds_per_cu, 65536u);
  EXPECT_EQ(gfx90a.lds_granule, 512u);
}

TEST(RocmOccupancyTest, LimiterNamesAreStable) {
  // No production caller emits these yet -- see the CONSUMERS note on
  // OccupancyStats. They are pinned here so that the follow-up surfacing
  // `limiter` as an XStat inherits a stable vocabulary rather than inventing
  // one, and so a rename shows up as a test change rather than silently
  // becoming a user-visible string later.
  EXPECT_STREQ(OccupancyLimiterName(OccupancyLimiter::kNone), "none");
  EXPECT_STREQ(OccupancyLimiterName(OccupancyLimiter::kVGPR), "vgpr");
  EXPECT_STREQ(OccupancyLimiterName(OccupancyLimiter::kLDS), "lds");
  EXPECT_STREQ(OccupancyLimiterName(OccupancyLimiter::kSGPR), "sgpr");
  EXPECT_STREQ(OccupancyLimiterName(OccupancyLimiter::kWorkgroup), "workgroup");
  EXPECT_STREQ(OccupancyLimiterName(OccupancyLimiter::kBarrier), "barrier");
}

// ===========================================================================
// Cache key
// ===========================================================================

TEST(RocmOccupancyTest, ParamsCacheKeyDiscriminatesEveryField) {
  const RocmDeviceOccupancyParams base =
      MakeParams(kGfx942, 5, 128, 16, 256, 0);
  EXPECT_EQ(base, MakeParams(kGfx942, 5, 128, 16, 256, 0));

  EXPECT_NE(base, MakeParams(kGfx942, 6, 128, 16, 256, 0));
  EXPECT_NE(base, MakeParams(kGfx942, 5, 129, 16, 256, 0));
  EXPECT_NE(base, MakeParams(kGfx942, 5, 128, 17, 256, 0));
  EXPECT_NE(base, MakeParams(kGfx942, 5, 128, 16, 512, 0));
  EXPECT_NE(base, MakeParams(kGfx942, 5, 128, 16, 256, 1024));
  // The field the pre-fix cache key discarded entirely, which is why the old
  // model could not tell a gfx90a from a gfx1100.
  EXPECT_NE(base, MakeParams(kGfx950, 5, 128, 16, 256, 0));
}

}  // namespace
}  // namespace profiler
}  // namespace xla
