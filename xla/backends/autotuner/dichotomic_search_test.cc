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
#include <memory>
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
