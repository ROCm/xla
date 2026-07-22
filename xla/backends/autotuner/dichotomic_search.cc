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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/hlo/ir/hlo_opcode.h"

#include "absl/log/log.h"

namespace xla {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Packs a coordinate into a stable string key for hashing.
std::string CoordKey(const Coord& coord) { return absl::StrJoin(coord, ","); }

// Returns true if the config populates the `triton` (TritonGemmKey) oneof case.
bool IsTritonGemmConfig(const BackendConfig& config) {
  return config.config_case() == BackendConfig::kTriton;
}

// Returns true if the config populates the `block_level` oneof case.
bool IsBlockLevelConfig(const BackendConfig& config) {
  return config.config_case() == BackendConfig::kBlockLevel;
}

// Returns true for dot-like ops that share the "larger parallel (N) tile is
// better" property empirically verified in ragged_dot_autotuner_analysis.md.
bool IsDotLikeOpcode(HloOpcode opcode) {
  return opcode == HloOpcode::kDot || opcode == HloOpcode::kRaggedDot ||
         opcode == HloOpcode::kConvolution;
}

// Extracts the named knob values from a config in a deterministic order.
// `axis_names` is filled on first call (when empty) and must be consistent
// across all configs in a set.
//
// For the `triton` (TritonGemmKey) case the axes are
//   {block_m, block_n, block_k, num_stages, num_warps, group_size}.
// For the `block_level` (BlockLevelFusionConfig) case the axes are
//   {tile_0..tile_n (from output_tiles(0).sizes), num_warps, num_ctas,
//    num_stages, group_size}.
std::vector<int64_t> ExtractKnobs(const BackendConfig& config,
                                  std::vector<std::string>* axis_names) {
  std::vector<int64_t> values;
  auto add = [&](const std::string& name, int64_t value) {
    if (axis_names->size() < values.size() + 1) {
      axis_names->push_back(name);
    }
    values.push_back(value);
  };

  if (IsTritonGemmConfig(config)) {
    const auto& t = config.triton();
    add("block_m", t.block_m());
    add("block_n", t.block_n());
    add("block_k", t.block_k());
    add("num_stages", t.num_stages());
    add("num_warps", t.num_warps());
    add("group_size", t.group_size());
  } else if (IsBlockLevelConfig(config)) {
    const auto& b = config.block_level();
    int tile_index = 0;
    if (b.output_tiles_size() > 0) {
      const auto& tile = b.output_tiles(0);
      for (int i = 0; i < tile.sizes_size(); ++i) {
        add(absl::StrCat("tile_", tile_index++), tile.sizes(i));
      }
    }
    add("num_warps", b.num_warps());
    add("num_ctas", b.num_ctas());
    add("num_stages", b.num_stages());
    add("group_size", b.group_size());
  }
  return values;
}

// Returns the marginal-best index on `axis` given `samples`: the axis value
// whose best-observed time (minimizing over all other axes) is smallest.
int MarginalBestIndex(const DichotomicSearchSpace& space, int axis,
                      absl::Span<const Sample> samples) {
  const int n = space.axes()[axis].values.size();
  std::vector<double> best_for_value(n, kInf);
  for (const Sample& s : samples) {
    if (axis >= s.coord.size()) continue;
    int idx = s.coord[axis];
    if (idx >= 0 && idx < n) {
      best_for_value[idx] = std::min(best_for_value[idx], s.time_seconds);
    }
  }
  int best = 0;
  double best_time = kInf;
  for (int i = 0; i < n; ++i) {
    if (best_for_value[i] < best_time) {
      best_time = best_for_value[i];
      best = i;
    }
  }
  return best;
}

// Returns the coordinate of the best sample, or a center coordinate if none.
Coord BestCoordOrCenter(const DichotomicSearchSpace& space,
                        absl::Span<const Sample> samples) {
  int best_time_idx = -1;
  double best_time = kInf;
  for (int i = 0; i < samples.size(); ++i) {
    if (samples[i].time_seconds < best_time) {
      best_time = samples[i].time_seconds;
      best_time_idx = i;
    }
  }
  if (best_time_idx >= 0) {
    return samples[best_time_idx].coord;
  }
  Coord center(space.axes().size());
  for (int a = 0; a < space.axes().size(); ++a) {
    center[a] = space.axes()[a].values.size() / 2;
  }
  return center;
}

// Representative indices {min, median, max} for an ordered axis.
std::vector<int> RepresentativeIndices(const ParameterAxis& axis) {
  const int n = axis.values.size();
  std::set<int> reps;
  reps.insert(0);
  reps.insert(n - 1);
  reps.insert(n / 2);
  return std::vector<int>(reps.begin(), reps.end());
}

// Central index for a sweep axis in the early phases.
int CentralIndex(const ParameterAxis& axis) { return axis.values.size() / 2; }

// Appends `coord` (snapped to a real config) to `out` if not already in `seen`.
void AddSnapped(const DichotomicSearchSpace& space, const Coord& coord,
                absl::flat_hash_set<int>* seen, std::vector<int>* out) {
  int idx = space.SnapIndex(coord);
  if (idx >= 0 && seen->insert(idx).second) {
    out->push_back(idx);
  }
}

// Emits the Cartesian product of per-axis `candidates` (index lists), snapping
// each to a real config and skipping `seen`. Stops early at `max_configs`
// (max_configs <= 0 means unlimited).
std::vector<int> CartesianProduct(
    const DichotomicSearchSpace& space,
    const std::vector<std::vector<int>>& candidates,
    absl::flat_hash_set<int>* seen, int max_configs) {
  const int num_axes = candidates.size();
  std::vector<int> result;
  Coord coord(num_axes, 0);
  std::vector<int> pos(num_axes, 0);
  while (true) {
    for (int a = 0; a < num_axes; ++a) coord[a] = candidates[a][pos[a]];
    AddSnapped(space, coord, seen, &result);
    if (max_configs > 0 && result.size() >= max_configs) break;

    int a = num_axes - 1;
    while (a >= 0) {
      if (++pos[a] < candidates[a].size()) break;
      pos[a] = 0;
      --a;
    }
    if (a < 0) break;  // odometer wrapped around => done
  }
  return result;
}

// ---- Per-phase implementations. --------------------------------------------

std::vector<int> SelectCoarseGrid(const DichotomicSearchSpace& space,
                                  const SearchProfile& profile,
                                  int max_configs) {
  const auto& axes = space.axes();
  const int num_axes = axes.size();
  std::vector<std::vector<int>> candidates(num_axes);
  for (int a = 0; a < num_axes; ++a) {
    if (profile.roles[a] == AxisRole::kSweep) {
      candidates[a] = {CentralIndex(axes[a])};
    } else {
      candidates[a] = RepresentativeIndices(axes[a]);
    }
  }
  absl::flat_hash_set<int> seen;
  return CartesianProduct(space, candidates, &seen, max_configs);
}

std::vector<int> SelectTernaryRefine(const DichotomicSearchSpace& space,
                                     const SearchProfile& profile,
                                     absl::Span<const Sample> prior_samples,
                                     absl::Span<const int> already_evaluated) {
  const auto& axes = space.axes();
  const int num_axes = axes.size();

  absl::flat_hash_set<int> seen(already_evaluated.begin(),
                                already_evaluated.end());
  std::vector<int> result;

  // Seed each axis from its marginal best (or the monotone extreme).
  Coord seed(num_axes);
  for (int a = 0; a < num_axes; ++a) {
    if (profile.roles[a] == AxisRole::kMonotoneUp) {
      seed[a] = axes[a].values.size() - 1;  // largest
    } else if (profile.roles[a] == AxisRole::kMonotoneDown) {
      seed[a] = 0;  // smallest
    } else {
      seed[a] = MarginalBestIndex(space, a, prior_samples);
    }
  }

  Coord cur = seed;
  for (int iter = 0; iter < 2; ++iter) {
    for (int a = 0; a < num_axes; ++a) {
      if (profile.roles[a] != AxisRole::kUnimodal) continue;
      const int n = axes[a].values.size();
      if (n <= 2) continue;
      int lo = 0, hi = n - 1;
      for (int step = 0; step < 3 && hi - lo > 1; ++step) {
        int i1 = lo + (hi - lo) / 3;
        int i2 = hi - (hi - lo) / 3;
        Coord c1 = cur;
        c1[a] = i1;
        AddSnapped(space, c1, &seen, &result);
        Coord c2 = cur;
        c2[a] = i2;
        AddSnapped(space, c2, &seen, &result);
        lo = i1;
        hi = i2;
      }
    }
    // Re-seed unimodal axes to their marginal best for the next iteration.
    for (int a = 0; a < num_axes; ++a) {
      if (profile.roles[a] == AxisRole::kUnimodal) {
        cur[a] = MarginalBestIndex(space, a, prior_samples);
      }
    }
  }
  return result;
}

std::vector<int> SelectNeighborhoodSweep(
    const DichotomicSearchSpace& space, const SearchProfile& profile,
    absl::Span<const Sample> prior_samples,
    absl::Span<const int> already_evaluated) {
  const auto& axes = space.axes();
  const int num_axes = axes.size();

  absl::flat_hash_set<int> seen(already_evaluated.begin(),
                                already_evaluated.end());

  Coord best = BestCoordOrCenter(space, prior_samples);
  if (best.size() != num_axes) {
    best.assign(num_axes, 0);
    for (int a = 0; a < num_axes; ++a) best[a] = axes[a].values.size() / 2;
  }

  std::vector<std::vector<int>> candidates(num_axes);
  for (int a = 0; a < num_axes; ++a) {
    const int n = axes[a].values.size();
    if (profile.roles[a] == AxisRole::kSweep) {
      candidates[a].resize(n);
      for (int i = 0; i < n; ++i) candidates[a][i] = i;
    } else if (profile.roles[a] == AxisRole::kMonotoneUp) {
      std::set<int> s;
      s.insert(n - 1);
      if (n >= 2) s.insert(n - 2);
      candidates[a].assign(s.begin(), s.end());
    } else if (profile.roles[a] == AxisRole::kMonotoneDown) {
      std::set<int> s;
      s.insert(0);
      if (n >= 2) s.insert(1);
      candidates[a].assign(s.begin(), s.end());
    } else {  // kUnimodal
      std::set<int> s;
      int b = std::clamp(best[a], 0, n - 1);
      s.insert(b);
      if (b - 1 >= 0) s.insert(b - 1);
      if (b + 1 < n) s.insert(b + 1);
      candidates[a].assign(s.begin(), s.end());
    }
  }

  return CartesianProduct(space, candidates, &seen, /*max_configs=*/0);
}

}  // namespace

absl::StatusOr<DichotomicSearchSpace> DichotomicSearchSpace::Build(
    absl::Span<const BackendConfig* const> configs) {
  if (configs.empty()) {
    return absl::InvalidArgumentError(
        "DichotomicSearchSpace::Build: empty config set.");
  }
  const BackendConfig::ConfigCase first_case = configs.front()->config_case();
  if (first_case != BackendConfig::kTriton &&
      first_case != BackendConfig::kBlockLevel) {
    return absl::InvalidArgumentError(
        "DichotomicSearchSpace::Build: configs are not Triton configs.");
  }
  for (const BackendConfig* c : configs) {
    if (c->config_case() != first_case) {
      return absl::InvalidArgumentError(
          "DichotomicSearchSpace::Build: heterogeneous config set.");
    }
  }

  DichotomicSearchSpace space;
  space.num_configs_ = configs.size();

  std::vector<std::string> axis_names;
  std::vector<std::vector<int64_t>> per_config_values;
  per_config_values.reserve(configs.size());
  for (const BackendConfig* c : configs) {
    per_config_values.push_back(ExtractKnobs(*c, &axis_names));
  }

  const int num_axes = axis_names.size();
  if (num_axes == 0) {
    return absl::InvalidArgumentError(
        "DichotomicSearchSpace::Build: no tunable knobs extracted.");
  }

  std::vector<std::set<int64_t>> distinct(num_axes);
  for (const auto& v : per_config_values) {
    if (v.size() != num_axes) {
      return absl::InvalidArgumentError(
          "DichotomicSearchSpace::Build: inconsistent knob count.");
    }
    for (int a = 0; a < num_axes; ++a) distinct[a].insert(v[a]);
  }

  space.axes_.resize(num_axes);
  for (int a = 0; a < num_axes; ++a) {
    ParameterAxis& axis = space.axes_[a];
    axis.name = axis_names[a];
    axis.values.assign(distinct[a].begin(), distinct[a].end());
    axis.ordered = true;
  }

  std::vector<std::map<int64_t, int>> value_to_index(num_axes);
  for (int a = 0; a < num_axes; ++a) {
    for (int i = 0; i < space.axes_[a].values.size(); ++i) {
      value_to_index[a][space.axes_[a].values[i]] = i;
    }
  }

  space.coords_.reserve(configs.size());
  for (int c = 0; c < configs.size(); ++c) {
    Coord coord(num_axes);
    for (int a = 0; a < num_axes; ++a) {
      coord[a] = value_to_index[a][per_config_values[c][a]];
    }
    space.coord_to_index_[CoordKey(coord)] = c;
    space.coords_.push_back(std::move(coord));
  }

  return space;
}

int DichotomicSearchSpace::LookupIndex(const Coord& coord) const {
  auto it = coord_to_index_.find(CoordKey(coord));
  return it == coord_to_index_.end() ? -1 : it->second;
}

int DichotomicSearchSpace::SnapIndex(const Coord& coord) const {
  int exact = LookupIndex(coord);
  if (exact >= 0) return exact;
  int best = 0;
  int64_t best_dist = std::numeric_limits<int64_t>::max();
  for (int c = 0; c < coords_.size(); ++c) {
    int64_t dist = 0;
    for (int a = 0; a < coord.size() && a < coords_[c].size(); ++a) {
      dist += std::llabs(static_cast<int64_t>(coords_[c][a]) - coord[a]);
    }
    if (dist < best_dist) {
      best_dist = dist;
      best = c;
    }
  }
  return best;
}

SearchProfile MakeProfile(const DichotomicSearchSpace& space,
                          HloOpcode opcode) {
  SearchProfile profile;
  const auto& axes = space.axes();
  profile.roles.resize(axes.size(), AxisRole::kUnimodal);

  auto is_short = [](const ParameterAxis& axis) {
    return axis.values.size() <= 3;
  };

  const bool dot_like = IsDotLikeOpcode(opcode);
  const bool is_block_level =
      // block-level configs expose tile_* axes; TritonGemmKey exposes block_n.
      [&] {
        for (const ParameterAxis& a : axes) {
          if (a.name.rfind("tile_", 0) == 0) return true;
        }
        return false;
      }();

  // Identify the "N-like" parallel axis, if any.
  //  - TritonGemmKey (dot/scaled-dot): "block_n".
  //  - block-level (ragged-dot): the widest output tile axis.
  int n_like = -1;
  if (!is_block_level) {
    for (int a = 0; a < axes.size(); ++a) {
      if (axes[a].name == "block_n") {
        n_like = a;
        break;
      }
    }
  } else {
    int64_t best_max = -1;
    for (int a = 0; a < axes.size(); ++a) {
      if (axes[a].name.rfind("tile_", 0) == 0 && !axes[a].values.empty()) {
        int64_t mx = axes[a].values.back();
        if (mx > best_max) {
          best_max = mx;
          n_like = a;
        }
      }
    }
  }

  for (int a = 0; a < axes.size(); ++a) {
    const ParameterAxis& axis = axes[a];
    // Small / categorical knobs are swept.
    if (axis.name == "num_stages" || axis.name == "num_warps" ||
        axis.name == "num_ctas" || axis.name == "group_size" ||
        axis.name == "split_k" || is_short(axis)) {
      profile.roles[a] = AxisRole::kSweep;
      continue;
    }
    // The N-like parallel axis is monotone-up for dot-like ops (the strongly
    // verified "larger tile_n is better" prior). For non-dot ops we keep it
    // unimodal as a safe default.
    if (a == n_like && dot_like) {
      profile.roles[a] = AxisRole::kMonotoneUp;
      continue;
    }
    // block_m, block_k, other tile axes: unimodal.
    profile.roles[a] = AxisRole::kUnimodal;
  }
  return profile;
}

SearchProfile RefineRoles(const SearchProfile& profile,
                          const DichotomicSearchSpace& space,
                          absl::Span<const Sample> phase1_samples,
                          double noise_tolerance) {
  SearchProfile refined = profile;
  const auto& axes = space.axes();

  for (int a = 0; a < axes.size(); ++a) {
    AxisRole role = refined.roles[a];
    if (role != AxisRole::kMonotoneUp && role != AxisRole::kMonotoneDown) {
      continue;
    }
    const int n = axes[a].values.size();
    if (n < 2) continue;

    std::vector<double> best_at(n, kInf);
    for (const Sample& s : phase1_samples) {
      if (a >= s.coord.size()) continue;
      int idx = s.coord[a];
      if (idx >= 0 && idx < n) {
        best_at[idx] = std::min(best_at[idx], s.time_seconds);
      }
    }

    double t_lo = best_at.front();
    double t_hi = best_at.back();
    if (!std::isfinite(t_lo) || !std::isfinite(t_hi)) continue;

    auto within = [noise_tolerance](double smaller, double larger) {
      return smaller <= larger * (1.0 + noise_tolerance);
    };

    bool holds = (role == AxisRole::kMonotoneUp) ? within(t_hi, t_lo)
                                                 : within(t_lo, t_hi);
    if (!holds) {
      refined.roles[a] = AxisRole::kUnimodal;  // relax only
    }
  }
  return refined;
}

std::vector<int> SelectConfigs(const DichotomicSearchSpace& space,
                               const SearchProfile& profile, SearchPhase phase,
                               absl::Span<const Sample> prior_samples,
                               absl::Span<const int> already_evaluated,
                               int max_configs) {
  switch (phase) {
    case SearchPhase::kCoarseGrid:
      return SelectCoarseGrid(space, profile, max_configs);
    case SearchPhase::kTernaryRefine:
      return SelectTernaryRefine(space, profile, prior_samples,
                                 already_evaluated);
    case SearchPhase::kNeighborhoodSweep:
      return SelectNeighborhoodSweep(space, profile, prior_samples,
                                     already_evaluated);
  }
  return {};
}

int BestSampleIndex(const DichotomicSearchSpace& space,
                    absl::Span<const Sample> samples) {
  int best = -1;
  double best_time = kInf;
  for (const Sample& s : samples) {
    if (s.time_seconds < best_time) {
      best_time = s.time_seconds;
      best = space.LookupIndex(s.coord);
    }
  }
  VLOG(-1) << "BestSampleIndex = " << best;
  return best;
}

}  // namespace xla
