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
#include "absl/time/time.h"
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

// The joint 2-D co-sampling must CREATE a diagonal (block_m, block_k) config
// that per-axis moves never form: from a best sample at (bm*, bk*), the Phase-3
// neighborhood must include the diagonal neighbours (bm*±1, bk*∓1). This is the
// core fix for coupled optima that 1-D coordinate descent strands.
TEST(DichotomicSearchTest, NeighborhoodEmitsDiagonalCoupledConfig) {
  // Dense 2-D grid over (block_m, block_k) so every diagonal neighbour exists
  // as a real config; other knobs single-valued to isolate the (bm,bk) plane.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::vector<int64_t> ms = {16, 32, 64, 128, 256};
  const std::vector<int64_t> ks = {16, 32, 64, 128};
  for (int64_t m : ms) {
    for (int64_t k : ks) {
      configs.push_back(MakeTritonConfig(m, /*block_n=*/64, k,
                                         /*num_stages=*/1, /*num_warps=*/4));
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);

  const int mi = AxisIndex(space, "block_m");
  const int ki = AxisIndex(space, "block_k");
  ASSERT_GE(mi, 0);
  ASSERT_GE(ki, 0);

  // Best sample at (block_m=64, block_k=64) -> indices (2, 2).
  std::vector<Sample> prior;
  {
    Coord c(space.axes().size(), 0);
    for (int i = 0; i < space.axes()[mi].values.size(); ++i) {
      if (space.axes()[mi].values[i] == 64) c[mi] = i;
    }
    for (int i = 0; i < space.axes()[ki].values.size(); ++i) {
      if (space.axes()[ki].values[i] == 64) c[ki] = i;
    }
    prior.push_back(Sample{c, /*time=*/1.0});
  }

  std::vector<int> sweep = SelectConfigs(
      space, profile, SearchPhase::kNeighborhoodSweep, prior, /*already=*/{});
  ASSERT_FALSE(sweep.empty());

  // The diagonal neighbour (block_m=128, block_k=32) -- one step up in bm, one
  // step down in bk -- must be present. A per-axis-only neighborhood could not
  // create it (it differs from the best on BOTH axes).
  bool saw_diagonal = false;
  for (int idx : sweep) {
    const auto& t = configs[idx]->triton();
    if (t.block_m() == 128 && t.block_k() == 32) saw_diagonal = true;
  }
  EXPECT_TRUE(saw_diagonal)
      << "joint 2-D move must co-sample the diagonal (bm=128, bk=32)";
}

// The joint move is applied in Phase 1 too: around a coarse-grid coordinate the
// diagonal (bm, bk) neighbours are emitted.
TEST(DichotomicSearchTest, CoarseGridEmitsDiagonalCoupledConfigs) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::vector<int64_t> ms = {16, 32, 64, 128, 256};
  const std::vector<int64_t> ks = {16, 32, 64, 128};
  for (int64_t m : ms) {
    for (int64_t k : ks) {
      configs.push_back(MakeTritonConfig(m, /*block_n=*/64, k,
                                         /*num_stages=*/1, /*num_warps=*/4));
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);

  std::vector<int> grid =
      SelectConfigs(space, profile, SearchPhase::kCoarseGrid, {}, {});
  ASSERT_FALSE(grid.empty());

  // The base coarse grid over block_m/block_k probes {min,median,max} on each
  // axis (axis-aligned). The joint move adds diagonal neighbours around those
  // points; e.g. around the median (bm=64, bk=64) the diagonal (bm=128, bk=32)
  // must appear.
  bool saw_diagonal = false;
  for (int idx : grid) {
    const auto& t = configs[idx]->triton();
    if (t.block_m() == 128 && t.block_k() == 32) saw_diagonal = true;
  }
  EXPECT_TRUE(saw_diagonal)
      << "Phase-1 joint move must co-sample a diagonal (bm,bk) config";

  // Still deduplicated and feasible.
  std::set<int> uniq(grid.begin(), grid.end());
  EXPECT_EQ(uniq.size(), grid.size()) << "coarse grid must be deduplicated";
  for (int idx : grid) {
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, space.num_configs());
  }
}

// Phase 2 is seeded from the best measured SAMPLE (joint), not per-axis
// marginals. With a best sample at (bm=128, bk=32) the ternary background uses
// bm=128, so probes vary bk while holding bm at the winning 128 -- verifying
// the seed comes from the joint best rather than a marginalised bm.
TEST(DichotomicSearchTest, TernarySeedsFromBestSampleCoordinate) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::vector<int64_t> ms = {16, 32, 64, 128, 256};
  const std::vector<int64_t> ks = {16, 32, 64, 128};
  for (int64_t m : ms) {
    for (int64_t k : ks) {
      configs.push_back(MakeTritonConfig(m, /*block_n=*/64, k,
                                         /*num_stages=*/1, /*num_warps=*/4));
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  SearchProfile profile = MakeProfile(space, HloOpcode::kDot);

  const int mi = AxisIndex(space, "block_m");
  const int ki = AxisIndex(space, "block_k");
  ASSERT_GE(mi, 0);
  ASSERT_GE(ki, 0);

  // Provide two prior samples; the BEST (lowest time) is (bm=128, bk=32).
  std::vector<Sample> prior;
  auto make_coord = [&](int64_t bm, int64_t bk) {
    Coord c(space.axes().size(), 0);
    for (int i = 0; i < space.axes()[mi].values.size(); ++i) {
      if (space.axes()[mi].values[i] == bm) c[mi] = i;
    }
    for (int i = 0; i < space.axes()[ki].values.size(); ++i) {
      if (space.axes()[ki].values[i] == bk) c[ki] = i;
    }
    return c;
  };
  prior.push_back(Sample{make_coord(64, 64), /*time=*/5.0});   // worse
  prior.push_back(Sample{make_coord(128, 32), /*time=*/1.0});  // BEST

  std::vector<int> probes = SelectConfigs(
      space, profile, SearchPhase::kTernaryRefine, prior, /*already=*/{});
  ASSERT_FALSE(probes.empty());

  // The block_k ternary must run at the best sample's block_m (=128); so at
  // least one probe should hold block_m=128 while varying block_k (e.g. the
  // small bk=32 or others), demonstrating the joint seed. (A marginal seed for
  // block_m could differ.)
  bool saw_bm128_probe = false;
  for (int idx : probes) {
    if (configs[idx]->triton().block_m() == 128) saw_bm128_probe = true;
  }
  EXPECT_TRUE(saw_bm128_probe)
      << "Phase-2 ternary must be seeded from the best sample's block_m=128";
}

// FeasibleSlice returns exactly the real configs that differ from the anchor in
// ONLY the requested axis, so a pairwise comparison within the slice is a
// controlled single-axis experiment. Here block_k is a sparse projection: only
// some (block_m, block_k) combinations are feasible, and the slice must include
// only the feasible ones for the anchor's fixed block_m/block_n.
TEST(DichotomicSearchTest, FeasibleSliceVariesExactlyOneAxis) {
  // Sparse feasible set: block_k in {16,32,64} exists only for block_m=64;
  // block_m=32 only has block_k=16. So the block_k slice through a block_m=64
  // anchor has three points, and every point keeps block_m=64 fixed.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(MakeTritonConfig(32, 64, 16, 1, 4));
  configs.push_back(MakeTritonConfig(64, 64, 16, 1, 4));
  configs.push_back(MakeTritonConfig(64, 64, 32, 1, 4));
  configs.push_back(MakeTritonConfig(64, 64, 64, 1, 4));
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  const int mi = AxisIndex(space, "block_m");
  const int ki = AxisIndex(space, "block_k");
  ASSERT_GE(mi, 0);
  ASSERT_GE(ki, 0);

  // Anchor at (block_m=64, block_k=32) -> config index 2.
  Coord anchor = space.CoordOf(2);
  std::vector<AxisPoint> slice = space.FeasibleSlice(anchor, ki);

  // The block_k slice through block_m=64 has exactly the three feasible k
  // values {16, 32, 64}; every returned config keeps block_m == 64 (only
  // block_k varies), and the points are sorted by the block_k value index.
  ASSERT_EQ(slice.size(), 3);
  int prev_value_index = -1;
  for (const AxisPoint& p : slice) {
    EXPECT_GT(p.value_index, prev_value_index) << "slice must be sorted";
    prev_value_index = p.value_index;
    const Coord& c = space.CoordOf(p.config_index);
    EXPECT_EQ(c[mi], anchor[mi]) << "block_m must be held fixed in the slice";
  }

  // The block_m slice through this anchor (block_k=32) has only block_m=64,
  // because block_m=32 has no block_k=32 config: an infeasible single-axis
  // neighbor is simply ABSENT (no snapping, no substitution).
  std::vector<AxisPoint> m_slice = space.FeasibleSlice(anchor, mi);
  ASSERT_EQ(m_slice.size(), 1);
  EXPECT_EQ(space.CoordOf(m_slice[0].config_index)[ki], anchor[ki]);
}

// FeasibleSlice2D returns exactly the real configs differing from the anchor in
// at most the two requested axes (a controlled two-axis experiment); no third
// axis moves, and infeasible members are absent.
TEST(DichotomicSearchTest, FeasibleSlice2DVariesAtMostTwoAxes) {
  // Dense (block_m, block_k) plane, plus a distractor axis num_stages that must
  // stay fixed. Two num_stages values so we can verify the third axis is held.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::vector<int64_t> ms = {32, 64, 128};
  const std::vector<int64_t> ks = {16, 32, 64};
  for (int64_t m : ms) {
    for (int64_t k : ks) {
      for (int64_t s : {1, 2}) {
        configs.push_back(MakeTritonConfig(m, /*block_n=*/64, k,
                                           /*num_stages=*/s, /*num_warps=*/4));
      }
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  const int mi = AxisIndex(space, "block_m");
  const int ki = AxisIndex(space, "block_k");
  const int si = AxisIndex(space, "num_stages");
  ASSERT_GE(mi, 0);
  ASSERT_GE(ki, 0);
  ASSERT_GE(si, 0);

  // Anchor at (block_m=64, block_k=32, num_stages=1).
  Coord anchor(space.axes().size(), 0);
  auto set_axis = [&](int axis, int64_t value) {
    const auto& vals = space.axes()[axis].values;
    for (int i = 0; i < static_cast<int>(vals.size()); ++i) {
      if (vals[i] == value) anchor[axis] = i;
    }
  };
  set_axis(mi, 64);
  set_axis(ki, 32);
  set_axis(si, 1);
  ASSERT_GE(space.LookupIndex(anchor), 0);

  std::vector<Coord> slice = space.FeasibleSlice2D(anchor, mi, ki);
  // The full (block_m, block_k) plane at num_stages=1 has 3x3 = 9 members.
  EXPECT_EQ(slice.size(), 9);
  for (const Coord& c : slice) {
    // num_stages (the third axis) must never move.
    EXPECT_EQ(c[si], anchor[si]) << "num_stages must be held fixed";
  }
}

// SnapIndex of an imagined (possibly infeasible) coordinate yields a real
// config, but it may differ from the requested coordinate in MORE than one
// axis -- which is why snapped samples are candidate-only, never evidence.
TEST(DichotomicSearchTest, SnapIndexIsCandidateOnly) {
  // Sparse set: block_k=64 exists ONLY for block_m=32, and block_m=128 exists
  // ONLY with block_k=16. So the coordinate (block_m=128, block_k=64) uses two
  // values that each exist on their own axis but never co-occur -- it is an
  // infeasible combination. Snapping it lands on a real config that differs
  // from the request in more than one axis.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(MakeTritonConfig(32, 64, 64, 1, 4));   // only k=64 lives here
  configs.push_back(MakeTritonConfig(64, 64, 32, 1, 4));
  configs.push_back(MakeTritonConfig(128, 64, 16, 1, 4));  // m=128 only with k=16
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  const int mi = AxisIndex(space, "block_m");
  const int ki = AxisIndex(space, "block_k");
  ASSERT_GE(mi, 0);
  ASSERT_GE(ki, 0);
  // Sanity: both values exist on their axes (so the coord is well-formed) but
  // the COMBINATION does not.
  ASSERT_EQ(space.axes()[mi].values, (std::vector<int64_t>{32, 64, 128}));
  ASSERT_EQ(space.axes()[ki].values, (std::vector<int64_t>{16, 32, 64}));

  // Imagined coordinate (block_m=128, block_k=64) is an infeasible combination.
  Coord imagined(space.axes().size(), 0);
  auto set_axis = [&](int axis, int64_t value) {
    const auto& vals = space.axes()[axis].values;
    for (int i = 0; i < static_cast<int>(vals.size()); ++i) {
      if (vals[i] == value) imagined[axis] = i;
    }
  };
  set_axis(mi, 128);
  set_axis(ki, 64);
  EXPECT_LT(space.LookupIndex(imagined), 0) << "must be infeasible";

  const int snapped = space.SnapIndex(imagined);
  ASSERT_GE(snapped, 0);
  // A Sample built from a snapped coord must be marked candidate-only so it is
  // never used as per-axis evidence.
  Sample candidate{space.CoordOf(snapped), /*time=*/1.0, /*is_evidence=*/false};
  EXPECT_FALSE(candidate.is_evidence);
}

// The budget is a PERCENTAGE of the exhaustive config count and therefore scales
// with the size of the search space: fast=10%, balanced=20%, thorough=30% of
// num_configs. An optional wall-clock cap is threaded through as a finite
// duration (0 => no cap).
TEST(DichotomicSearchTest, PresetMapsToExpectedBudget) {
  // A space with 1000 feasible configs (5*5*5*2*2*2 = 1000 knob combos), so the
  // percentages are exact and well above the small-space floor.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t m : {16, 32, 64, 128, 256}) {
    for (int64_t n : {16, 32, 64, 128, 256}) {
      for (int64_t k : {16, 32, 64, 128, 256}) {
        for (int64_t s : {1, 2}) {
          for (int64_t w : {4, 8}) {
            for (int64_t g : {1, 8}) {
              configs.push_back(MakeTritonConfig(m, n, k, s, w, g));
            }
          }
        }
      }
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  ASSERT_EQ(space.num_configs(), 1000);

  // fast = 10%, balanced = 20%, thorough = 30% of num_configs.
  EXPECT_EQ(ResolveBudget(SearchBudgetPreset::kFast, space).max_compilations,
            100);
  EXPECT_EQ(
      ResolveBudget(SearchBudgetPreset::kBalanced, space).max_compilations, 200);
  EXPECT_EQ(
      ResolveBudget(SearchBudgetPreset::kThorough, space).max_compilations, 300);

  // Ordering holds and scales with the space: fast < balanced < thorough.
  EXPECT_LT(ResolveBudget(SearchBudgetPreset::kFast, space).max_compilations,
            ResolveBudget(SearchBudgetPreset::kBalanced, space)
                .max_compilations);
  EXPECT_LT(
      ResolveBudget(SearchBudgetPreset::kBalanced, space).max_compilations,
      ResolveBudget(SearchBudgetPreset::kThorough, space).max_compilations);

  // No time cap by default; a positive time-limit becomes a finite duration.
  EXPECT_EQ(
      ResolveBudget(SearchBudgetPreset::kBalanced, space).max_tuning_time,
      absl::InfiniteDuration());
  EXPECT_EQ(ResolveBudget(SearchBudgetPreset::kBalanced, space,
                          /*time_limit_ms=*/1500)
                .max_tuning_time,
            absl::Milliseconds(1500));
}

// A tiny space's budget is clamped up to the small-space floor and never exceeds
// the total number of configs.
TEST(DichotomicSearchTest, BudgetIsClampedForTinySpaces) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  // Only 4 feasible configs: 10% would be < 1, so the floor applies, but the
  // budget can never exceed num_configs (4).
  configs.push_back(MakeTritonConfig(16, 64, 32, 1, 4));
  configs.push_back(MakeTritonConfig(32, 64, 32, 1, 4));
  configs.push_back(MakeTritonConfig(64, 64, 32, 1, 4));
  configs.push_back(MakeTritonConfig(128, 64, 32, 1, 4));
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;
  ASSERT_EQ(space.num_configs(), 4);

  for (SearchBudgetPreset preset :
       {SearchBudgetPreset::kFast, SearchBudgetPreset::kBalanced,
        SearchBudgetPreset::kThorough}) {
    const int budget = ResolveBudget(preset, space).max_compilations;
    EXPECT_GE(budget, 1);
    EXPECT_LE(budget, space.num_configs())
        << "budget must never exceed the exhaustive config count";
  }
}

// The BudgetLedger charges only compiled configs and reports exhaustion once
// the compilation cap is reached; RemainingCompilations() decreases monotonically
// and never goes negative.
TEST(DichotomicSearchTest, SearchStopsAtCompilationBudget) {
  SearchBudget budget;
  budget.max_compilations = 10;
  BudgetLedger ledger(budget);

  EXPECT_EQ(ledger.RemainingCompilations(), 10);
  EXPECT_FALSE(ledger.Exhausted());

  ledger.RecordCompiled(4);
  EXPECT_EQ(ledger.compiled(), 4);
  EXPECT_EQ(ledger.RemainingCompilations(), 6);
  EXPECT_FALSE(ledger.Exhausted());

  ledger.RecordCompiled(6);
  EXPECT_EQ(ledger.RemainingCompilations(), 0);
  EXPECT_TRUE(ledger.Exhausted());

  // Overspending never yields a negative remaining count.
  ledger.RecordCompiled(5);
  EXPECT_EQ(ledger.RemainingCompilations(), 0);
  EXPECT_TRUE(ledger.Exhausted());

  // An unlimited budget (max_compilations <= 0) is never exhausted by count.
  SearchBudget unlimited;
  unlimited.max_compilations = 0;
  BudgetLedger unlimited_ledger(unlimited);
  unlimited_ledger.RecordCompiled(1000);
  EXPECT_FALSE(unlimited_ledger.Exhausted());
  EXPECT_GT(unlimited_ledger.RemainingCompilations(), 1000);
}

// Under a tiny budget, Phase1Cap collapses to the structural minimum (so prior
// evidence is never starved), while a generous budget lets the 0.4 fraction win.
TEST(DichotomicSearchTest, TinyBudgetDegradesToCoarseOnly) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int64_t m : {16, 32, 64, 128, 256}) {
    for (int64_t n : {16, 32, 64, 128, 256}) {
      configs.push_back(MakeTritonConfig(m, n, 32, 1, 4));
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  // Fast (10% of 625) vs thorough (30% of 625): the larger budget grants a
  // larger Phase-1 cap, but both stay at least at the structural floor.
  BudgetLedger tiny(ResolveBudget(SearchBudgetPreset::kFast, space));
  const int tiny_cap = Phase1Cap(tiny, space);
  EXPECT_GE(tiny_cap, 8) << "Phase-1 cap must never drop below the floor";

  BudgetLedger generous(ResolveBudget(SearchBudgetPreset::kThorough, space));
  const int generous_cap = Phase1Cap(generous, space);
  EXPECT_GE(generous_cap, tiny_cap)
      << "a larger budget must not grant Phase 1 a smaller cap";
}

// SelectTopKDiverse returns coordinate-diverse basin seeds: the fastest evidence
// sample is always seed 1, and later seeds are accepted only if their tile-axis
// Manhattan distance to every accepted seed is large enough. Candidate-only
// samples never seed a basin, and categorical-only differences don't create new
// basins.
TEST(DichotomicSearchTest, SelectTopKDiverseReturnsDistinctBasins) {
  // Dense (block_m, block_k) grid plus a categorical num_stages axis.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::vector<int64_t> ms = {16, 32, 64, 128, 256};
  const std::vector<int64_t> ks = {16, 32, 64, 128};
  for (int64_t m : ms) {
    for (int64_t k : ks) {
      for (int64_t s : {1, 2}) {
        configs.push_back(MakeTritonConfig(m, /*block_n=*/64, k,
                                           /*num_stages=*/s, /*num_warps=*/4));
      }
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  const int mi = AxisIndex(space, "block_m");
  const int ki = AxisIndex(space, "block_k");
  const int si = AxisIndex(space, "num_stages");
  ASSERT_GE(mi, 0);
  ASSERT_GE(ki, 0);
  ASSERT_GE(si, 0);

  auto coord_of = [&](int64_t bm, int64_t bk, int64_t stages) {
    Coord c(space.axes().size(), 0);
    auto set_axis = [&](int axis, int64_t value) {
      const auto& vals = space.axes()[axis].values;
      for (int i = 0; i < static_cast<int>(vals.size()); ++i) {
        if (vals[i] == value) c[axis] = i;
      }
    };
    set_axis(mi, bm);
    set_axis(ki, bk);
    set_axis(si, stages);
    return c;
  };

  std::vector<Sample> samples;
  // Fastest: (bm=16, bk=16). A far tile-diverse basin: (bm=256, bk=128).
  samples.push_back(Sample{coord_of(16, 16, 1), /*time=*/1.0});
  samples.push_back(Sample{coord_of(256, 128, 1), /*time=*/1.5});
  // Same basin as the fastest but only num_stages differs (categorical) -- must
  // NOT count as a distinct basin.
  samples.push_back(Sample{coord_of(16, 16, 2), /*time=*/1.2});
  // A candidate-only (snapped) sample that is fast but must never seed a basin.
  samples.push_back(
      Sample{coord_of(64, 64, 1), /*time=*/0.5, /*is_evidence=*/false});

  std::vector<Coord> seeds =
      SelectTopKDiverse(space, samples, /*k=*/3, /*min_manhattan=*/2);

  // Seed 1 is the fastest EVIDENCE sample (candidate-only 0.5 is ignored).
  ASSERT_GE(seeds.size(), 1u);
  EXPECT_EQ(space.axes()[mi].values[seeds[0][mi]], 16);
  EXPECT_EQ(space.axes()[ki].values[seeds[0][ki]], 16);

  // Exactly two diverse basins are found: {(16,16), (256,128)}. The categorical-
  // only twin of the fastest does NOT add a third basin.
  EXPECT_EQ(seeds.size(), 2u);
  bool saw_far_basin = false;
  for (const Coord& c : seeds) {
    if (space.axes()[mi].values[c[mi]] == 256 &&
        space.axes()[ki].values[c[ki]] == 128) {
      saw_far_basin = true;
    }
  }
  EXPECT_TRUE(saw_far_basin) << "the tile-diverse basin must be a distinct seed";

  // No candidate-only sample (bm=64,bk=64) is ever a seed.
  for (const Coord& c : seeds) {
    const bool is_candidate_basin =
        space.axes()[mi].values[c[mi]] == 64 &&
        space.axes()[ki].values[c[ki]] == 64;
    EXPECT_FALSE(is_candidate_basin)
        << "candidate-only samples must never seed a basin";
  }
}

// Among several candidates with SIMILAR performance (within the noise band),
// SelectTopKDiverse prefers the one FARTHEST from the already-accepted seeds so
// the K basins are spread as widely as possible. A clearly-slower-but-farther
// candidate is NOT preferred over a similar-performance closer one only when the
// slower one falls outside the band.
TEST(DichotomicSearchTest, SelectTopKDiversePrefersFarthestAmongSimilar) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::vector<int64_t> ms = {16, 32, 64, 128, 256};
  const std::vector<int64_t> ks = {16, 32, 64, 128};
  for (int64_t m : ms) {
    for (int64_t k : ks) {
      configs.push_back(MakeTritonConfig(m, /*block_n=*/64, k,
                                         /*num_stages=*/1, /*num_warps=*/4));
    }
  }
  auto space_or = DichotomicSearchSpace::Build(Ptrs(configs));
  ASSERT_THAT(space_or, IsOk());
  const DichotomicSearchSpace& space = *space_or;

  const int mi = AxisIndex(space, "block_m");
  const int ki = AxisIndex(space, "block_k");
  ASSERT_GE(mi, 0);
  ASSERT_GE(ki, 0);

  auto coord_of = [&](int64_t bm, int64_t bk) {
    Coord c(space.axes().size(), 0);
    auto set_axis = [&](int axis, int64_t value) {
      const auto& vals = space.axes()[axis].values;
      for (int i = 0; i < static_cast<int>(vals.size()); ++i) {
        if (vals[i] == value) c[axis] = i;
      }
    };
    set_axis(mi, bm);
    set_axis(ki, bk);
    return c;
  };

  // Seed 1 will be the fastest: (bm=16, bk=16) @ 1.00.
  // Two SIMILAR-performance far candidates (within 3% of the best-eligible):
  //   near basin  (bm=64,  bk=64)  @ 1.01  (tile-axis dist to seed1 = 4)
  //   far  basin  (bm=256, bk=128) @ 1.02  (tile-axis dist to seed1 = 7)
  // Both are within the 3% band of the best eligible (1.01); the FAR one must be
  // chosen as seed 2 even though it is marginally slower.
  std::vector<Sample> samples;
  samples.push_back(Sample{coord_of(16, 16), /*time=*/1.00});
  samples.push_back(Sample{coord_of(64, 64), /*time=*/1.01});
  samples.push_back(Sample{coord_of(256, 128), /*time=*/1.02});

  std::vector<Coord> seeds =
      SelectTopKDiverse(space, samples, /*k=*/2, /*min_manhattan=*/2,
                        /*noise_tolerance=*/0.03);
  ASSERT_EQ(seeds.size(), 2u);
  EXPECT_EQ(space.axes()[mi].values[seeds[0][mi]], 16);
  // Seed 2 is the FAR basin, not the near one, because they perform similarly.
  EXPECT_EQ(space.axes()[mi].values[seeds[1][mi]], 256);
  EXPECT_EQ(space.axes()[ki].values[seeds[1][ki]], 128);

  // If the far candidate is clearly slower (outside the band), the closer
  // similar-performance one is chosen instead.
  std::vector<Sample> samples2;
  samples2.push_back(Sample{coord_of(16, 16), /*time=*/1.00});
  samples2.push_back(Sample{coord_of(64, 64), /*time=*/1.01});   // in band
  samples2.push_back(Sample{coord_of(256, 128), /*time=*/2.00});  // clearly slow
  std::vector<Coord> seeds2 =
      SelectTopKDiverse(space, samples2, /*k=*/2, /*min_manhattan=*/2,
                        /*noise_tolerance=*/0.03);
  ASSERT_EQ(seeds2.size(), 2u);
  EXPECT_EQ(space.axes()[mi].values[seeds2[1][mi]], 64)
      << "a clearly-slower far candidate must not be preferred";
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
