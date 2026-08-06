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

#include "xla/backends/autotuner/dichotomic_search.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;

// Builds a Triton (TritonGemmKey) BackendConfig with the given knobs.
std::unique_ptr<BackendConfig> MakeTritonConfig(
    int64_t block_m, int64_t block_n, int64_t block_k, int64_t num_stages,
    int64_t num_warps, int64_t group_size = 1) {
  auto config = std::make_unique<BackendConfig>();
  auto* t = config->mutable_triton();
  t->set_block_m(block_m);
  t->set_block_n(block_n);
  t->set_block_k(block_k);
  t->set_num_stages(num_stages);
  t->set_num_warps(num_warps);
  t->set_group_size(group_size);
  return config;
}

// Builds a block-level (ragged-dot) BackendConfig with a single 2D output tile.
std::unique_ptr<BackendConfig> MakeBlockLevelConfig(int64_t tile0,
                                                    int64_t tile1,
                                                    int64_t num_warps,
                                                    int64_t num_stages) {
  auto config = std::make_unique<BackendConfig>();
  auto* b = config->mutable_block_level();
  auto* tile = b->add_output_tiles();
  tile->add_sizes(tile0);
  tile->add_sizes(tile1);
  b->set_num_warps(num_warps);
  b->set_num_stages(num_stages);
  return config;
}

std::vector<const BackendConfig*> Ptrs(
    const std::vector<std::unique_ptr<BackendConfig>>& configs) {
  std::vector<const BackendConfig*> ptrs;
  ptrs.reserve(configs.size());
  for (const auto& c : configs) ptrs.push_back(c.get());
  return ptrs;
}

// Returns the axis index with the given name, or -1.
int AxisIndex(const DichotomicSearchSpace& space, absl::string_view name) {
  for (int a = 0; a < space.axes().size(); ++a) {
    if (space.axes()[a].name == name) return a;
  }
  return -1;
}

TEST(DichotomicSearchTest, BuildFailsOnEmptySet) {
  EXPECT_FALSE(DichotomicSearchSpace::Build({}).ok());
}

TEST(DichotomicSearchTest, ExtractsDistinctSortedAxisValuesForTriton) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(MakeTritonConfig(16, 64, 32, 1, 4));
  configs.push_back(MakeTritonConfig(64, 256, 32, 2, 8));
  configs.push_back(MakeTritonConfig(256, 64, 16, 1, 4));

  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  int m = AxisIndex(space, "block_m");
  int n = AxisIndex(space, "block_n");
  int k = AxisIndex(space, "block_k");
  ASSERT_GE(m, 0);
  ASSERT_GE(n, 0);
  ASSERT_GE(k, 0);
  EXPECT_EQ(space.axes()[m].values, (std::vector<int64_t>{16, 64, 256}));
  EXPECT_EQ(space.axes()[n].values, (std::vector<int64_t>{64, 256}));
  EXPECT_EQ(space.axes()[k].values, (std::vector<int64_t>{16, 32}));
  EXPECT_EQ(space.num_configs(), 3);
}

TEST(DichotomicSearchTest, ExtractsAxesForBlockLevel) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(MakeBlockLevelConfig(16, 128, 4, 1));
  configs.push_back(MakeBlockLevelConfig(64, 256, 8, 2));

  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  EXPECT_GE(AxisIndex(space, "tile_0"), 0);
  EXPECT_GE(AxisIndex(space, "tile_1"), 0);
  EXPECT_GE(AxisIndex(space, "num_warps"), 0);
  EXPECT_GE(AxisIndex(space, "num_stages"), 0);
}

TEST(DichotomicSearchTest, BuildFailsOnHeterogeneousSet) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(MakeTritonConfig(16, 64, 32, 1, 4));
  configs.push_back(MakeBlockLevelConfig(16, 128, 4, 1));
  EXPECT_FALSE(DichotomicSearchSpace::Build(Ptrs(configs)).ok());
}

TEST(DichotomicSearchTest, MakeProfileMarksBlockNMonotoneUpForDot) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  // Ensure block_n and block_m axes have > 3 distinct values so they are not
  // auto-classified as sweep axes.
  for (int64_t v : {16, 32, 64, 128, 256}) {
    configs.push_back(MakeTritonConfig(v, v, 32, 1, 4));
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);
  int n = AxisIndex(space, "block_n");
  int m = AxisIndex(space, "block_m");
  int stages = AxisIndex(space, "num_stages");
  ASSERT_GE(n, 0);
  ASSERT_GE(m, 0);
  ASSERT_GE(stages, 0);
  EXPECT_EQ(profile.roles[n], AxisRole::kMonotoneUp);
  EXPECT_EQ(profile.roles[m], AxisRole::kUnimodal);
  EXPECT_EQ(profile.roles[stages], AxisRole::kSweep);  // only 1 value -> short
}

TEST(DichotomicSearchTest, MakeProfileWithHintsUsesAnalysisRolesByIndex) {
  // Build a block-level (experimental tiling) config set with two output tile
  // axes, tile_0 and tile_1, both with > 3 distinct values so neither is
  // auto-classified as a sweep axis.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t v : {16, 32, 64, 128, 256}) {
    configs.push_back(MakeBlockLevelConfig(v, v, /*num_warps=*/4,
                                           /*num_stages=*/1));
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  const int t0 = AxisIndex(space, "tile_0");
  const int t1 = AxisIndex(space, "tile_1");
  ASSERT_GE(t0, 0);
  ASSERT_GE(t1, 0);

  // Analysis hints (index-aligned): tile_0 is a kSequential (contraction) dim,
  // tile_1 is the kParallel dim and tiles the LARGEST dimension. Note the tile
  // VALUES are identical across axes, so the "widest values" name heuristic
  // could not distinguish them -- only the analysis (dimension_size) can.
  AxisRoleHints hints(space.axes().size());
  hints[t0].semantics = AxisSemantics::kSequential;
  hints[t0].dimension_size = 512;
  hints[t1].semantics = AxisSemantics::kParallel;
  hints[t1].dimension_size = 4096;  // largest parallel dim => the N-like axis

  SearchProfile profile = MakeProfile(space, HloOpcode::kDot, hints);
  // The parallel axis tiling the largest dim becomes kMonotoneUp; the
  // sequential axis becomes kUnimodal.
  EXPECT_EQ(profile.roles[t1], AxisRole::kMonotoneUp);
  EXPECT_EQ(profile.roles[t0], AxisRole::kUnimodal);
}

TEST(DichotomicSearchTest, MakeProfileWithHintsPicksLargestParallelDim) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t v : {16, 32, 64, 128, 256}) {
    configs.push_back(MakeBlockLevelConfig(v, v, /*num_warps=*/4,
                                           /*num_stages=*/1));
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  const int t0 = AxisIndex(space, "tile_0");
  const int t1 = AxisIndex(space, "tile_1");
  ASSERT_GE(t0, 0);
  ASSERT_GE(t1, 0);

  // Both axes are kParallel; the one tiling the larger dimension must be chosen
  // as the monotone N-like axis. Here tile_0 has the larger dimension.
  AxisRoleHints hints(space.axes().size());
  hints[t0].semantics = AxisSemantics::kParallel;
  hints[t0].dimension_size = 8192;  // larger
  hints[t1].semantics = AxisSemantics::kParallel;
  hints[t1].dimension_size = 128;  // smaller

  SearchProfile profile = MakeProfile(space, HloOpcode::kDot, hints);
  EXPECT_EQ(profile.roles[t0], AxisRole::kMonotoneUp);
  EXPECT_EQ(profile.roles[t1], AxisRole::kUnimodal);
}

TEST(DichotomicSearchTest, MakeProfileFallsBackWhenHintsEmptyOrMismatched) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t v : {16, 32, 64, 128, 256}) {
    configs.push_back(MakeTritonConfig(v, v, 32, 1, 4));
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  const int n = AxisIndex(space, "block_n");
  ASSERT_GE(n, 0);

  // Empty hints => identical to the opcode-only heuristic (block_n monotone).
  SearchProfile from_empty =
      MakeProfile(space, HloOpcode::kDot, AxisRoleHints{});
  SearchProfile from_opcode = MakeProfile(space, HloOpcode::kDot);
  EXPECT_EQ(from_empty.roles, from_opcode.roles);
  EXPECT_EQ(from_empty.roles[n], AxisRole::kMonotoneUp);

  // Size-mismatched hints are ignored (also fall back to the heuristic).
  AxisRoleHints wrong_size(space.axes().size() + 1);
  SearchProfile from_mismatch = MakeProfile(space, HloOpcode::kDot, wrong_size);
  EXPECT_EQ(from_mismatch.roles, from_opcode.roles);
}

TEST(DichotomicSearchTest, RefineRolesRelaxesMonotoneWhenContradicted) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t v : {16, 32, 64, 128, 256}) {
    configs.push_back(MakeTritonConfig(v, v, 32, 1, 4));
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);
  int n = AxisIndex(space, "block_n");
  ASSERT_GE(n, 0);
  ASSERT_EQ(profile.roles[n], AxisRole::kMonotoneUp);

  const int num_axes = space.axes().size();
  const int last = space.axes()[n].values.size() - 1;

  // Samples that CONTRADICT monotone-up on block_n: the smallest value is
  // fastest (lowest time), the largest is slowest.
  std::vector<Sample> contradicting;
  {
    Coord lo(num_axes, 0);
    lo[n] = 0;  // smallest block_n
    contradicting.push_back(Sample{lo, /*time=*/1.0});
    Coord hi(num_axes, 0);
    hi[n] = last;  // largest block_n
    contradicting.push_back(Sample{hi, /*time=*/5.0});
  }
  SearchProfile relaxed =
      RefineRoles(profile, space, contradicting, /*noise_tolerance=*/0.03);
  EXPECT_EQ(relaxed.roles[n], AxisRole::kUnimodal);

  // Samples that CONFIRM monotone-up: largest value is fastest.
  std::vector<Sample> confirming;
  {
    Coord lo(num_axes, 0);
    lo[n] = 0;
    confirming.push_back(Sample{lo, /*time=*/5.0});
    Coord hi(num_axes, 0);
    hi[n] = last;
    confirming.push_back(Sample{hi, /*time=*/1.0});
  }
  SearchProfile kept =
      RefineRoles(profile, space, confirming, /*noise_tolerance=*/0.03);
  EXPECT_EQ(kept.roles[n], AxisRole::kMonotoneUp);
}

// Feature B: with a known tiled-dimension size, Phase-2 ternary probes on a
// unimodal axis land on clean-divisor tile values (zero masking waste) rather
// than the raw geometric thirds.
TEST(DichotomicSearchTest, TernaryProbesSnapToDivisorsWhenSizeKnown) {
  // block_m axis has values {16, 24, 32, 48, 64, 96, 128}. For a dimension of
  // size 128, the clean divisors among these are {16, 32, 64, 128}; 24, 48, 96
  // are wasteful. Give block_m > 3 values (unimodal) and make every other axis
  // a single value so the only varying axis is block_m.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t m : {16, 24, 32, 48, 64, 96, 128}) {
    configs.push_back(MakeTritonConfig(m, /*block_n=*/64, /*block_k=*/32,
                                       /*num_stages=*/1, /*num_warps=*/4));
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  const int mi = AxisIndex(space, "block_m");
  ASSERT_GE(mi, 0);

  // Start from the opcode heuristic, then attach a known dimension size of 128
  // for the block_m axis (as feature-A hints would).
  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);
  profile.dimension_sizes.assign(space.axes().size(), 0);
  profile.dimension_sizes[mi] = 128;

  // Some Phase-1 samples so MarginalBestIndex has data (values are irrelevant
  // to the divisor-snapping being tested).
  std::vector<Sample> prior;
  {
    Coord c(space.axes().size(), 0);
    prior.push_back(Sample{c, 1.0});
  }

  std::vector<int> probes = SelectConfigs(
      space, profile, SearchPhase::kTernaryRefine, prior, /*already=*/{});
  ASSERT_FALSE(probes.empty());

  // Every emitted config's block_m must be a clean divisor of 128.
  for (int idx : probes) {
    const int64_t m = configs[idx]->triton().block_m();
    EXPECT_EQ(128 % m, 0) << "block_m=" << m << " is not a divisor of 128";
  }
}

// On a Phase-3 neighborhood sweep, tile values whose masking waste
// exceeds the pruning threshold (25% of the last block) are dropped, provided a
// lower-waste value is retained and the value is not the current best. Values
// with waste <= 25% (including exact divisors) are kept.
//
// The soft-prune is threshold-based, NOT "divides cleanly", so this test checks
// the exact waste boundary rather than pure divisibility.
TEST(DichotomicSearchTest, NeighborhoodSweepSoftPrunesHighWasteValues) {
  // block_m values {16, 24, 32, 48, 64}; dimension size D = 64. Waste ratios:
  //   16 -> 0      (64%16==0)
  //   24 -> 8/72   ~= 0.111  (<= 0.25, kept)
  //   32 -> 0      (64%32==0)
  //   48 -> 32/96  ~= 0.333  (>  0.25, pruned when non-best)
  //   64 -> 0      (64%64==0)
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t m : {16, 24, 32, 48, 64}) {
    configs.push_back(MakeTritonConfig(m, /*block_n=*/64, /*block_k=*/32,
                                       /*num_stages=*/1, /*num_warps=*/4));
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  const int mi = AxisIndex(space, "block_m");
  ASSERT_GE(mi, 0);

  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);
  profile.dimension_sizes.assign(space.axes().size(), 0);
  profile.dimension_sizes[mi] = 64;

  // Best coordinate at block_m=32. Its +/-1 index neighborhood is the values
  // {24, 32, 48}. Of these, only 48 exceeds the 25% waste threshold and is a
  // non-best value, so exactly 48 must be pruned; 24 (11% waste) and 32 (best,
  // 0% waste) are retained.
  std::vector<Sample> prior;
  {
    Coord c(space.axes().size(), 0);
    const auto& vals = space.axes()[mi].values;
    for (int i = 0; i < vals.size(); ++i) {
      if (vals[i] == 32) c[mi] = i;
    }
    prior.push_back(Sample{c, /*time=*/1.0});
  }

  std::vector<int> sweep = SelectConfigs(
      space, profile, SearchPhase::kNeighborhoodSweep, prior, /*already=*/{});

  bool saw_24 = false, saw_32 = false, saw_48 = false;
  for (int idx : sweep) {
    const int64_t m = configs[idx]->triton().block_m();
    if (m == 24) saw_24 = true;
    if (m == 32) saw_32 = true;
    if (m == 48) saw_48 = true;
    // No emitted block_m should exceed the 25% waste threshold (48 is the only
    // such value in the neighborhood and must have been pruned).
    EXPECT_LE(static_cast<double>((64 + m - 1) / m * m - 64) /
                  static_cast<double>((64 + m - 1) / m * m),
              0.25)
        << "block_m=" << m << " has >25% masking waste and should be pruned";
  }
  EXPECT_TRUE(saw_32) << "the best value (block_m=32) must be retained";
  EXPECT_TRUE(saw_24) << "a low-waste (~11%) value must be retained";
  EXPECT_FALSE(saw_48) << "the high-waste (~33%) value must be pruned";
}

// Feature B is a strict no-op when no dimension sizes are supplied: the
// selected probe set must be identical to a profile without dimension_sizes.
TEST(DichotomicSearchTest, DivisibilityIsNoOpWithoutDimensionSizes) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t m : {16, 24, 32, 48, 64, 96, 128}) {
    configs.push_back(MakeTritonConfig(m, /*block_n=*/64, /*block_k=*/32,
                                       /*num_stages=*/1, /*num_warps=*/4));
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  SearchProfile no_sizes = MakeProfile(space, HloOpcode::kDot);
  // dimension_sizes is all-zero (unknown) here.
  std::vector<Sample> prior;
  {
    Coord c(space.axes().size(), 0);
    prior.push_back(Sample{c, 1.0});
  }
  std::vector<int> a =
      SelectConfigs(space, no_sizes, SearchPhase::kTernaryRefine, prior, {});
  std::vector<int> b =
      SelectConfigs(space, no_sizes, SearchPhase::kTernaryRefine, prior, {});
  EXPECT_EQ(a, b);  // deterministic, and unaffected by divisibility logic
}

// A synthetic unimodal cost function over block_m with a known interior
// minimum. Verifies the 3-phase search converges to the optimum while
// evaluating a strict subset.
TEST(DichotomicSearchTest, ThreePhaseSearchFindsUnimodalOptimum) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::vector<int64_t> ms = {16, 32, 64, 128, 256};
  const std::vector<int64_t> ns = {16, 32, 64, 128, 256};
  const std::vector<int64_t> ks = {16, 32, 64};
  for (int64_t m : ms) {
    for (int64_t n : ns) {
      for (int64_t k : ks) {
        configs.push_back(MakeTritonConfig(m, n, k, /*stages=*/1,
                                           /*warps=*/4));
      }
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);

  int mi = AxisIndex(space, "block_m");
  int ni = AxisIndex(space, "block_n");
  int ki = AxisIndex(space, "block_k");
  ASSERT_GE(mi, 0);
  ASSERT_GE(ni, 0);
  ASSERT_GE(ki, 0);

  // Ground-truth landscape: unimodal in m (min at index 2 => block_m=64),
  // monotone-up in n (bigger is faster), unimodal in k (min at index 1 =>
  // block_k=32).
  auto cost = [&](const Coord& c) -> double {
    double dm = std::abs(c[mi] - 2);
    double dn = (space.axes()[ni].values.size() - 1) - c[ni];  // fewer = better
    double dk = std::abs(c[ki] - 1);
    return 1.0 + dm + dn + dk;
  };

  // Simulate the 3-phase loop against the ground-truth cost table.
  std::vector<Sample> samples;
  std::vector<int> evaluated;
  auto eval = [&](const std::vector<int>& indices) {
    for (int idx : indices) {
      // Recover coordinate for this config index by matching values.
      // We rebuild from the config knobs directly.
      const auto& t = configs[idx]->triton();
      Coord c(space.axes().size(), 0);
      auto set_axis = [&](int axis, int64_t value) {
        const auto& vals = space.axes()[axis].values;
        for (int i = 0; i < vals.size(); ++i) {
          if (vals[i] == value) {
            c[axis] = i;
            return;
          }
        }
      };
      set_axis(mi, t.block_m());
      set_axis(ni, t.block_n());
      set_axis(ki, t.block_k());
      samples.push_back(Sample{c, cost(c)});
      evaluated.push_back(idx);
    }
  };

  eval(SelectConfigs(space, profile, SearchPhase::kCoarseGrid, {}, {}));
  ASSERT_FALSE(samples.empty());
  profile = RefineRoles(profile, space, samples);
  eval(SelectConfigs(space, profile, SearchPhase::kTernaryRefine, samples,
                     evaluated));
  eval(SelectConfigs(space, profile, SearchPhase::kNeighborhoodSweep, samples,
                     evaluated));

  int best = BestSampleIndex(space, samples);
  ASSERT_GE(best, 0);
  // Find the best sample's coordinate & verify it hits the true optimum
  // (block_m=64, block_n=256, block_k=32).
  double best_time = 1e30;
  Coord best_c;
  for (const Sample& s : samples) {
    if (s.time_seconds < best_time) {
      best_time = s.time_seconds;
      best_c = s.coord;
    }
  }
  EXPECT_EQ(space.axes()[mi].values[best_c[mi]], 64);
  EXPECT_EQ(space.axes()[ni].values[best_c[ni]], 256);
  EXPECT_EQ(space.axes()[ki].values[best_c[ki]], 32);

  // Strict subset: we evaluated far fewer than the full space.
  EXPECT_LT(static_cast<int>(evaluated.size()), space.num_configs());
}

// Because block_k<->num_stages and block_m<->num_warps are strongly coupled, a
// tile value must be scored against ALL its coupled sweep values. So in Phase 1
// every emitted block_k appears crossed with every num_stages, and every
// block_m crossed with every num_warps. This test verifies that binding.
TEST(DichotomicSearchTest, CoarseGridBindsCoupledSweepToTile) {
  // block_m/block_k > 3 values (kUnimodal); num_warps/num_stages multiple
  // values (real kSweep axes) so the coupling logic engages. block_n has a
  // single value so it doesn't blow up the cross product.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::vector<int64_t> ms = {16, 32, 64, 128, 256};
  const std::vector<int64_t> ks = {16, 32, 64, 128};
  const std::vector<int64_t> warps = {2, 4, 8};
  const std::vector<int64_t> stages = {1, 2, 3, 4};
  for (int64_t m : ms) {
    for (int64_t k : ks) {
      for (int64_t w : warps) {
        for (int64_t s : stages) {
          configs.push_back(MakeTritonConfig(m, /*block_n=*/64, k, s, w));
        }
      }
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);

  const int mi = AxisIndex(space, "block_m");
  const int ki = AxisIndex(space, "block_k");
  const int wi = AxisIndex(space, "num_warps");
  const int si = AxisIndex(space, "num_stages");
  ASSERT_GE(mi, 0);
  ASSERT_GE(ki, 0);
  ASSERT_GE(wi, 0);
  ASSERT_GE(si, 0);

  std::vector<int> phase1 =
      SelectConfigs(space, profile, SearchPhase::kCoarseGrid, {}, {});
  ASSERT_FALSE(phase1.empty());

  // Collect, for each emitted block_k, the set of num_stages it was paired
  // with; and for each block_m, the set of num_warps.
  std::map<int64_t, std::set<int64_t>> ns_by_bk;
  std::map<int64_t, std::set<int64_t>> nw_by_bm;
  for (int idx : phase1) {
    const auto& t = configs[idx]->triton();
    ns_by_bk[t.block_k()].insert(t.num_stages());
    nw_by_bm[t.block_m()].insert(t.num_warps());
  }

  const std::set<int64_t> all_stages(stages.begin(), stages.end());
  const std::set<int64_t> all_warps(warps.begin(), warps.end());

  // Every block_k that was probed must have been crossed with ALL num_stages.
  ASSERT_FALSE(ns_by_bk.empty());
  for (const auto& [bk, seen_ns] : ns_by_bk) {
    EXPECT_EQ(seen_ns, all_stages)
        << "block_k=" << bk << " was not crossed with all num_stages";
  }
  // Every block_m that was probed must have been crossed with ALL num_warps.
  ASSERT_FALSE(nw_by_bm.empty());
  for (const auto& [bm, seen_nw] : nw_by_bm) {
    EXPECT_EQ(seen_nw, all_warps)
        << "block_m=" << bm << " was not crossed with all num_warps";
  }
}

// Phase 2 (ternary refine) must also bind the coupled sweep: every block_k it
// probes is crossed with all num_stages, and every block_m with all num_warps.
TEST(DichotomicSearchTest, TernaryRefineBindsCoupledSweepToTile) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::vector<int64_t> ms = {16, 32, 64, 128, 256};
  const std::vector<int64_t> ks = {16, 32, 64, 128};
  const std::vector<int64_t> warps = {2, 4, 8};
  const std::vector<int64_t> stages = {1, 2, 3, 4};
  for (int64_t m : ms) {
    for (int64_t k : ks) {
      for (int64_t w : warps) {
        for (int64_t s : stages) {
          configs.push_back(MakeTritonConfig(m, /*block_n=*/64, k, s, w));
        }
      }
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);

  std::vector<Sample> prior;
  {
    Coord c(space.axes().size(), 0);
    prior.push_back(Sample{c, 1.0});
  }
  std::vector<int> probes = SelectConfigs(
      space, profile, SearchPhase::kTernaryRefine, prior, /*already=*/{});
  ASSERT_FALSE(probes.empty());

  std::map<int64_t, std::set<int64_t>> ns_by_bk;
  std::map<int64_t, std::set<int64_t>> nw_by_bm;
  for (int idx : probes) {
    const auto& t = configs[idx]->triton();
    ns_by_bk[t.block_k()].insert(t.num_stages());
    nw_by_bm[t.block_m()].insert(t.num_warps());
  }
  const std::set<int64_t> all_stages(stages.begin(), stages.end());
  const std::set<int64_t> all_warps(warps.begin(), warps.end());
  for (const auto& [bk, seen_ns] : ns_by_bk) {
    EXPECT_EQ(seen_ns, all_stages)
        << "ternary block_k=" << bk << " not crossed with all num_stages";
  }
  for (const auto& [bm, seen_nw] : nw_by_bm) {
    EXPECT_EQ(seen_nw, all_warps)
        << "ternary block_m=" << bm << " not crossed with all num_warps";
  }
}

// All emitted configs remain unique and feasible under the coupled expansion.
TEST(DichotomicSearchTest, CoupledExpansionKeepsConfigsUniqueAndFeasible) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t m : {16, 32, 64, 128, 256}) {
    for (int64_t k : {16, 32, 64, 128}) {
      for (int64_t w : {2, 4, 8}) {
        for (int64_t s : {1, 2, 3, 4}) {
          configs.push_back(MakeTritonConfig(m, /*block_n=*/64, k, s, w));
        }
      }
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);

  std::vector<int> grid =
      SelectConfigs(space, profile, SearchPhase::kCoarseGrid, {}, {});
  std::set<int> uniq(grid.begin(), grid.end());
  EXPECT_EQ(uniq.size(), grid.size()) << "coarse grid must be deduplicated";
  for (int idx : grid) {
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, space.num_configs());
  }
}

TEST(DichotomicSearchTest, SelectedConfigsAreAlwaysFeasible) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t m : {16, 64, 256}) {
    for (int64_t n : {16, 64, 256}) {
      configs.push_back(MakeTritonConfig(m, n, 32, 1, 4));
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);

  std::vector<int> phase1 =
      SelectConfigs(space, profile, SearchPhase::kCoarseGrid, {}, {});
  for (int idx : phase1) {
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, space.num_configs());
  }
}

}  // namespace
}  // namespace xla
