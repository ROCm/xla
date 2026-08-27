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
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/hlo/ir/hlo_opcode.h"

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

// ---- Divisibility-aware ("masking waste") helpers (feature B). --------------
//
// A tile of size `v` applied to a dimension of size `D` launches
// ceil(D / v) blocks; the last block is masked for (ceil(D/v)*v - D) elements.
// The "waste ratio" is that masked fraction of the last block's work. Tiles
// that divide D cleanly (D % v == 0) have zero waste and are empirically at
// least as good on contraction axes (see ragged_dot_autotuner_analysis.md).
//
// `dim_size <= 0` means the size is unknown (no tiling analysis for this axis);
// callers must treat that as "divisibility logic disabled" for the axis.

// Returns true if value index `i` on `axis` divides `dim_size` cleanly.
bool DividesCleanly(const ParameterAxis& axis, int64_t dim_size, int i) {
  if (dim_size <= 0 || i < 0 || i >= axis.values.size()) return false;
  const int64_t v = axis.values[i];
  return v > 0 && dim_size % v == 0;
}

// Given a starting index `i` on `axis`, returns the index of the nearest value
// (by |index distance|, ties preferring the larger value) that divides
// `dim_size` cleanly. Returns `i` itself when `i` already divides cleanly, and
// also returns `i` unchanged when the size is unknown or no clean divisor
// exists among the axis values.
int NearestDivisorIndex(const ParameterAxis& axis, int64_t dim_size, int i) {
  const int n = axis.values.size();
  if (dim_size <= 0 || n == 0) return i;
  if (DividesCleanly(axis, dim_size, i)) return i;
  for (int d = 1; d < n; ++d) {
    const int hi = i + d;  // prefer the larger value on ties
    if (hi < n && DividesCleanly(axis, dim_size, hi)) return hi;
    const int lo = i - d;
    if (lo >= 0 && DividesCleanly(axis, dim_size, lo)) return lo;
  }
  return i;  // no clean divisor available; leave the geometric probe in place.
}

// ---- Budget constants and helpers. -----------------------------------------

// The compilation budget is a PERCENTAGE OF THE EXHAUSTIVE CONFIG COUNT, so it
// scales automatically with the size of the search space instead of being a
// hard-coded constant. The preset only selects the coverage fraction.
constexpr double kFastBudgetFraction = 0.10;      // 10% of num_configs
constexpr double kBalancedBudgetFraction = 0.20;  // 20% of num_configs
constexpr double kThoroughBudgetFraction = 0.30;  // 30% of num_configs

// Absolute floor on the compilation budget so that very small spaces still get
// enough probes to run the coarse grid + a little refinement.
constexpr int kMinBudget = 8;

// The fraction of the global compilation budget Phase 1 (coarse grid) may
// request before handing off to refinement, and the absolute floor on that
// request so tiny budgets still gather some prior evidence.
constexpr double kPhase1BudgetFraction = 0.4;
constexpr int kPhase1Floor = 8;

// A-priori structural minimum number of coarse-grid probes needed so that
// RefineRoles has >= 2 measured points on the representative slices it checks.
// Ordered (non-sweep) axes contribute {min, median, max} => up to 3 reps; a
// couple of probes per such axis is enough to seed the single/two-axis slices.
int MinPhase1Configs(const DichotomicSearchSpace& space) {
  int ordered_axes = 0;
  for (const ParameterAxis& axis : space.axes()) {
    if (axis.values.size() >= 2) ++ordered_axes;
  }
  // At least 2 endpoints per ordered axis, floored at kPhase1Floor.
  return std::max(kPhase1Floor, 2 * ordered_axes);
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

// Returns the axis index whose name equals `name`, or -1 if absent.
int AxisIndexByName(const DichotomicSearchSpace& space,
                    absl::string_view name) {
  const auto& axes = space.axes();
  for (int a = 0; a < static_cast<int>(axes.size()); ++a) {
    if (axes[a].name == name) return a;
  }
  return -1;
}

// A pair of ordered TILE axes that are strongly coupled, so that their joint
// optimum can sit off the per-axis coordinate lines (e.g. a smaller block_k
// with a larger block_m/block_n on large "square" shapes, where a larger
// block_k costs an extra launch wave via SMEM occupancy). A 1-D coordinate
// search optimizes each axis with the other frozen and therefore never
// *co-samples* such a diagonal optimum. The Phase-3 joint move below evaluates
// the 2-D neighborhood of these pairs around the best measured sample to create
// and measure those diagonal configs. Purely runtime-driven (no HW model).
struct CoupledTilePair {
  int a;  // first tile axis
  int b;  // second tile axis
};

// Derives the coupled tile-axis pairs to co-sample. Uses axis names for the
// canonical Triton GEMM knobs (block_m/block_n/block_k). Returns pairs
// (block_m, block_k) and (block_n, block_k) when those axes exist with >= 2
// distinct values. Empty for non-GEMM/blocklevel spaces (the joint move then
// degrades to a no-op and Phase 3 keeps its standard behavior).
std::vector<CoupledTilePair> DeriveCoupledTilePairs(
    const DichotomicSearchSpace& space) {
  const auto& axes = space.axes();
  auto usable = [&](int a) -> bool {
    return a >= 0 && a < static_cast<int>(axes.size()) &&
           axes[a].values.size() >= 2;
  };
  const int m = AxisIndexByName(space, "block_m");
  const int n = AxisIndexByName(space, "block_n");
  const int k = AxisIndexByName(space, "block_k");
  std::vector<CoupledTilePair> pairs;
  if (usable(m) && usable(k)) pairs.push_back({m, k});
  if (usable(n) && usable(k)) pairs.push_back({n, k});
  return pairs;
}

// 2-D hill-climb move: for each coupled tile pair, emits the 3x3 index
// neighborhood on those two axes around `center` (all other axes held at
// `center`), snapped to real configs and de-duplicated via `seen`. This
// co-samples DIAGONAL joint configs (e.g. smaller block_k together with larger
// block_m) that per-axis 1-D moves can never create, which is exactly where the
// coupled optima on large/square shapes live. O(1) extra probes per pair
// (<= 9, minus dedup); purely runtime-driven (PickBestConfig keeps a joint move
// only if it profiles faster). Applied in EVERY phase so the diagonal is always
// reachable, and combined with best-config seeding it makes the whole search a
// bounded 2-D local search rather than 1-D coordinate descent.
void AddJointCoupledNeighborhood(const DichotomicSearchSpace& space,
                                 const Coord& center,
                                 absl::flat_hash_set<int>* seen,
                                 std::vector<int>* out) {
  const auto& axes = space.axes();
  if (static_cast<int>(center.size()) != static_cast<int>(axes.size())) return;
  const std::vector<CoupledTilePair> pairs = DeriveCoupledTilePairs(space);
  for (const CoupledTilePair& p : pairs) {
    const int na = static_cast<int>(axes[p.a].values.size());
    const int nb = static_cast<int>(axes[p.b].values.size());
    const int ba = std::clamp(center[p.a], 0, na - 1);
    const int bb = std::clamp(center[p.b], 0, nb - 1);
    for (int da = -1; da <= 1; ++da) {
      const int ia = ba + da;
      if (ia < 0 || ia >= na) continue;
      for (int db = -1; db <= 1; ++db) {
        const int ib = bb + db;
        if (ib < 0 || ib >= nb) continue;
        Coord c = center;
        c[p.a] = ia;
        c[p.b] = ib;
        AddSnapped(space, c, seen, out);
      }
    }
  }
}

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
  std::vector<int> result =
      CartesianProduct(space, candidates, &seen, max_configs);

  // Also co-sample the joint 2-D neighborhood of coupled tile pairs around each
  // coarse-grid coordinate, so Phase 1 already surfaces diagonal
  // (bm,bk)/(bn,bk) combinations rather than only the axis-aligned grid points.
  const std::vector<int> base(result);
  for (int idx : base) {
    AddJointCoupledNeighborhood(space, space.CoordOf(idx), &seen, &result);
    if (max_configs > 0 && static_cast<int>(result.size()) >= max_configs) {
      break;
    }
  }
  return result;
}

std::vector<int> SelectTernaryRefine(const DichotomicSearchSpace& space,
                                     const SearchProfile& profile,
                                     absl::Span<const Sample> prior_samples,
                                     absl::Span<const int> already_evaluated) {
  const auto& axes = space.axes();
  const int num_axes = axes.size();
  const bool have_sizes =
      static_cast<int>(profile.dimension_sizes.size()) == num_axes;
  auto dim_size = [&](int a) -> int64_t {
    return have_sizes ? profile.dimension_sizes[a] : 0;
  };

  absl::flat_hash_set<int> seen(already_evaluated.begin(),
                                already_evaluated.end());
  std::vector<int> result;

  // Seed the background from the best measured SAMPLE coordinate (a real,
  // co-measured config), NOT from per-axis marginals. For coupled axes this is
  // essential: a ternary probe on one tile axis is then scored against the
  // value of the OTHER axes that actually won TOGETHER, instead of a
  // marginalised value that may never have co-occurred with the probe (which is
  // how the 1-D search stranded joint optima like (block_m large, block_k
  // small)). Monotone axes still pin to their extreme.
  Coord cur = BestCoordOrCenter(space, prior_samples);
  if (static_cast<int>(cur.size()) != num_axes) {
    cur.assign(num_axes, 0);
    for (int a = 0; a < num_axes; ++a) cur[a] = axes[a].values.size() / 2;
  }
  for (int a = 0; a < num_axes; ++a) {
    if (profile.roles[a] == AxisRole::kMonotoneUp) {
      cur[a] = axes[a].values.size() - 1;  // largest
    } else if (profile.roles[a] == AxisRole::kMonotoneDown) {
      cur[a] = 0;  // smallest
    }
  }

  // Coordinate-wise ternary refinement of each unimodal axis, bracketed
  // geometrically over its sorted index range [0, n-1], with the other axes
  // held at the best-sample background `cur`. When the axis's tiled-dimension
  // size is known, each geometric probe index is snapped to the nearest
  // clean-divisor value.
  for (int a = 0; a < num_axes; ++a) {
    if (profile.roles[a] != AxisRole::kUnimodal) continue;
    const int n = axes[a].values.size();
    if (n <= 2) continue;
    const int64_t D = dim_size(a);
    int lo = 0, hi = n - 1;
    for (int step = 0; step < 3 && hi - lo > 1; ++step) {
      int i1 = lo + (hi - lo) / 3;
      int i2 = hi - (hi - lo) / 3;
      int p1 = NearestDivisorIndex(axes[a], D, i1);
      int p2 = NearestDivisorIndex(axes[a], D, i2);
      Coord c1 = cur;
      c1[a] = p1;
      AddSnapped(space, c1, &seen, &result);
      Coord c2 = cur;
      c2[a] = p2;
      AddSnapped(space, c2, &seen, &result);
      lo = i1;
      hi = i2;
    }
  }

  // Joint 2-D co-sampling of coupled tile pairs around the best-sample seed, so
  // Phase 2 can also CREATE diagonal (bm,bk)/(bn,bk) configs that the per-axis
  // ternary above never forms.
  AddJointCoupledNeighborhood(space, cur, &seen, &result);
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

  // Anchor on the best measured SAMPLE coordinate (joint), so the neighborhood
  // is taken around the config that actually won together.
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

  std::vector<int> result =
      CartesianProduct(space, candidates, &seen, /*max_configs=*/0);

  // Joint 2-D co-sampling of coupled tile pairs around the best sample -- the
  // key move that CREATES diagonal joint configs (e.g. smaller block_k together
  // with larger block_m) that the axis-aligned neighborhood above cannot form.
  AddJointCoupledNeighborhood(space, best, &seen, &result);
  return result;
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

bool DichotomicSearchSpace::CoordForConfig(const BackendConfig& config,
                                           Coord* coord) const {
  // Extract this config's knobs in the same deterministic order used to build
  // the axes. `local_names` must line up with axes_ by name.
  std::vector<std::string> local_names;
  std::vector<int64_t> values = ExtractKnobs(config, &local_names);
  if (static_cast<int>(values.size()) != static_cast<int>(axes_.size())) {
    return false;
  }
  coord->assign(axes_.size(), 0);
  for (int a = 0; a < static_cast<int>(axes_.size()); ++a) {
    // Names must match positionally (ExtractKnobs is deterministic), but verify
    // to be safe against any future divergence.
    if (a < static_cast<int>(local_names.size()) &&
        local_names[a] != axes_[a].name) {
      return false;
    }
    const std::vector<int64_t>& vals = axes_[a].values;
    // vals is small and sorted ascending; a linear scan is fine and avoids
    // building a per-axis map. (Axes typically have <= ~10 distinct values.)
    int found = -1;
    for (int i = 0; i < static_cast<int>(vals.size()); ++i) {
      if (vals[i] == values[a]) {
        found = i;
        break;
      }
    }
    if (found < 0) return false;
    (*coord)[a] = found;
  }
  return true;
}

std::vector<AxisPoint> DichotomicSearchSpace::FeasibleSlice(const Coord& fixed,
                                                            int axis) const {
  std::vector<AxisPoint> slice;
  const int num_axes = static_cast<int>(axes_.size());
  if (axis < 0 || axis >= num_axes ||
      static_cast<int>(fixed.size()) != num_axes) {
    return slice;
  }
  // Scan all real configs; keep those that agree with `fixed` on every axis
  // except `axis`. Each kept config differs from `fixed` in at most that single
  // axis, so pairwise comparisons within the returned slice are controlled
  // single-axis experiments.
  for (int c = 0; c < static_cast<int>(coords_.size()); ++c) {
    const Coord& coord = coords_[c];
    bool match = true;
    for (int j = 0; j < num_axes; ++j) {
      if (j == axis) continue;
      if (coord[j] != fixed[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      slice.push_back({/*value_index=*/coord[axis], /*config_index=*/c});
    }
  }
  std::sort(slice.begin(), slice.end(),
            [](const AxisPoint& a, const AxisPoint& b) {
              return a.value_index < b.value_index;
            });
  return slice;
}

std::vector<Coord> DichotomicSearchSpace::FeasibleSlice2D(const Coord& fixed,
                                                          int a, int b) const {
  std::vector<Coord> slice;
  const int num_axes = static_cast<int>(axes_.size());
  if (a == b || a < 0 || b < 0 || a >= num_axes || b >= num_axes ||
      static_cast<int>(fixed.size()) != num_axes) {
    return slice;
  }
  // Keep all real configs that agree with `fixed` on every axis except `{a,b}`,
  // i.e. those differing from `fixed` in at most those two axes. This is a
  // controlled two-axis experiment (no snapping, no third axis moving).
  for (const Coord& coord : coords_) {
    bool match = true;
    for (int j = 0; j < num_axes; ++j) {
      if (j == a || j == b) continue;
      if (coord[j] != fixed[j]) {
        match = false;
        break;
      }
    }
    if (match) slice.push_back(coord);
  }
  return slice;
}

SearchProfile MakeProfile(const DichotomicSearchSpace& space,
                          HloOpcode opcode) {
  return MakeProfile(space, opcode, AxisRoleHints{});
}

SearchProfile MakeProfile(const DichotomicSearchSpace& space, HloOpcode opcode,
                          const AxisRoleHints& hints) {
  SearchProfile profile;
  const auto& axes = space.axes();
  const int num_axes = axes.size();
  profile.roles.resize(num_axes, AxisRole::kUnimodal);
  // Carry per-axis tiled-dimension sizes from the analysis hints (0 = unknown)
  // so the ternary/neighborhood phases can do divisibility-aware placement.
  profile.dimension_sizes.assign(num_axes, 0);
  // Carry per-axis semantic roles (kParallel/kSequential/kUnknown) from the
  // hints so later phases (e.g. the correlated coarse-grid probes) can pair a
  // tile axis with its coupled sweep knob by semantics rather than axis name.
  profile.semantics.assign(num_axes, AxisSemantics::kUnknown);
  if (static_cast<int>(hints.size()) == num_axes) {
    for (int a = 0; a < num_axes; ++a) {
      profile.dimension_sizes[a] = hints[a].dimension_size;
      profile.semantics[a] = hints[a].semantics;
    }
  }

  auto is_short = [](const ParameterAxis& axis) {
    return axis.values.size() <= 3;
  };

  // Analysis hints are used only when index-aligned with the axes. An empty or
  // mismatched vector => pure name/opcode heuristic (unchanged legacy path).
  const bool have_hints = static_cast<int>(hints.size()) == num_axes && [&] {
    for (const AxisRoleHint& h : hints) {
      if (h.semantics != AxisSemantics::kUnknown) {
        return true;
      }
    }
    return false;
  }();

  const bool dot_like = IsDotLikeOpcode(opcode);

  // Identify the single "N-like" parallel axis (the one declared kMonotoneUp
  // for dot-like ops).
  //  - With analysis hints: among axes whose tiled dimension is kParallel, the
  //    one tiling the LARGEST dimension (from DimensionInfo::dimension_size).
  //    This is a purely analysis-driven choice by axis INDEX -- no axis names.
  //  - Without hints (fallback): the legacy axis-name heuristic.
  int n_like = -1;
  if (have_hints) {
    int64_t best_dim = -1;
    for (int a = 0; a < num_axes; ++a) {
      if (hints[a].semantics == AxisSemantics::kParallel &&
          hints[a].dimension_size > best_dim) {
        best_dim = hints[a].dimension_size;
        n_like = a;
      }
    }
  } else {
    const bool is_block_level = [&] {
      for (const ParameterAxis& a : axes) {
        if (a.name.rfind("tile_", 0) == 0) return true;
      }
      return false;
    }();
    if (!is_block_level) {
      for (int a = 0; a < num_axes; ++a) {
        if (axes[a].name == "block_n") {
          n_like = a;
          break;
        }
      }
    } else {
      int64_t best_max = -1;
      for (int a = 0; a < num_axes; ++a) {
        if (axes[a].name.rfind("tile_", 0) == 0 && !axes[a].values.empty()) {
          int64_t mx = axes[a].values.back();
          if (mx > best_max) {
            best_max = mx;
            n_like = a;
          }
        }
      }
    }
  }

  for (int a = 0; a < num_axes; ++a) {
    const ParameterAxis& axis = axes[a];
    // Small / categorical knobs are swept (independent of analysis). Note these
    // knobs (num_stages/warps/ctas/group_size/split_k) are not tiling axes, so
    // the analysis carries no dimension for them; keep the name/size test.
    if (axis.name == "num_stages" || axis.name == "num_warps" ||
        axis.name == "num_ctas" || axis.name == "group_size" ||
        axis.name == "split_k" || is_short(axis)) {
      profile.roles[a] = AxisRole::kSweep;
      continue;
    }
    // The parallel (N-like) axis is monotone-up for dot-like ops (the strongly
    // verified "larger parallel tile is better" prior). For non-dot ops we keep
    // it unimodal as a safe default.
    if (a == n_like && dot_like) {
      profile.roles[a] = AxisRole::kMonotoneUp;
      continue;
    }
    // Every other ordered tiling axis (a kSequential contraction axis, or a
    // non-N parallel axis such as M) has an interior optimum: unimodal. With
    // hints this is decided by the analysis role; without hints it is the same
    // safe default the legacy heuristic used.
    profile.roles[a] = AxisRole::kUnimodal;
  }
  return profile;
}

namespace {

// Collects the feasible slices along `axis` that have >= 2 measured points,
// grouped by the anchor (all-other-axes) coordinate. Each returned inner map is
// value_index_on_axis -> best measured time on that value within that single-
// axis slice.
//
// RefineRoles only CONFIRMS/RELAXES a prior (it never places the next probe),
// so it may safely use ALL evaluated samples -- not just evidence-grade ones.
// Grouping by anchor still guarantees each comparison is a controlled single-
// axis one: every point in a returned slice shares the same values on all axes
// except `axis`.
std::vector<std::map<int, double>> SlicesAlongAxis(
    const DichotomicSearchSpace& space, int axis,
    absl::Span<const Sample> samples) {
  const int num_axes = static_cast<int>(space.axes().size());
  // Key the slice by the coordinate with `axis` masked out (the anchor).
  std::map<std::string, std::map<int, double>> by_anchor;
  for (const Sample& s : samples) {
    if (static_cast<int>(s.coord.size()) != num_axes) continue;
    if (axis < 0 || axis >= num_axes) continue;
    const int v = s.coord[axis];
    if (v < 0 || v >= static_cast<int>(space.axes()[axis].values.size())) {
      continue;
    }
    std::string key;
    key.reserve(num_axes * 3);
    for (int j = 0; j < num_axes; ++j) {
      if (j == axis) continue;
      absl::StrAppend(&key, s.coord[j], ",");
    }
    auto& slice = by_anchor[key];
    auto it = slice.find(v);
    if (it == slice.end() || s.time_seconds < it->second) {
      slice[v] = s.time_seconds;
    }
  }
  std::vector<std::map<int, double>> slices;
  for (auto& [key, slice] : by_anchor) {
    if (slice.size() >= 2) slices.push_back(std::move(slice));
  }
  return slices;
}

}  // namespace

SearchProfile RefineRoles(const SearchProfile& profile,
                          const DichotomicSearchSpace& space,
                          absl::Span<const Sample> phase1_samples,
                          double noise_tolerance) {
  SearchProfile refined = profile;
  const auto& axes = space.axes();
  const int num_axes = static_cast<int>(axes.size());

  for (int a = 0; a < num_axes; ++a) {
    const AxisRole role = refined.roles[a];
    if (role != AxisRole::kMonotoneUp && role != AxisRole::kMonotoneDown) {
      continue;
    }
    const int n = static_cast<int>(axes[a].values.size());
    if (n < 2) continue;

    // Gather the feasible single-axis slices through this axis that carry >= 2
    // measured points. A prior can only be RELAXED, never verified.
    const std::vector<std::map<int, double>> slices =
        SlicesAlongAxis(space, a, phase1_samples);
    if (slices.empty()) continue;

    bool contradicted = false;
    for (const std::map<int, double>& slice : slices) {
      // Relative noise band: two times are "clearly different" only if they
      // differ by more than `noise_tolerance` (e.g. 3%). A prior is contradicted
      // when the extreme value in its preferred direction is beaten -- beyond
      // noise -- by a value on the WRONG side of the axis.
      auto clearly_faster = [&](double faster, double slower) {
        // `faster` beats `slower` by more than the noise band.
        return faster < slower * (1.0 - noise_tolerance);
      };

      const int top_v = slice.rbegin()->first;     // highest value index
      const int bottom_v = slice.begin()->first;   // lowest value index
      const double top_t = slice.rbegin()->second;
      const double bottom_t = slice.begin()->second;

      if (role == AxisRole::kMonotoneUp) {
        // "Larger is better" is contradicted iff some LOWER value is clearly
        // faster than the top (highest) value.
        for (const auto& [v, t] : slice) {
          if (v < top_v && clearly_faster(t, top_t)) {
            contradicted = true;
            break;
          }
        }
      } else {  // kMonotoneDown
        // "Smaller is better" is contradicted iff some HIGHER value is clearly
        // faster than the bottom (lowest) value.
        for (const auto& [v, t] : slice) {
          if (v > bottom_v && clearly_faster(t, bottom_t)) {
            contradicted = true;
            break;
          }
        }
      }
      if (contradicted) break;  // one clean counter-example is enough.
    }

    if (contradicted) {
      refined.roles[a] = AxisRole::kUnimodal;  // relaxation only ever widens.
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

namespace {

// Returns true if `axis` names a TILE axis (block_m/block_n/block_k or tile_*),
// i.e. a spatial tiling dimension that defines a "basin". Categorical knobs
// (num_warps/num_stages/num_ctas/group_size/split_k) are NOT tile axes: two
// seeds differing only in such a knob belong to the SAME basin.
bool IsTileAxis(const ParameterAxis& axis) {
  return axis.name == "block_m" || axis.name == "block_n" ||
         axis.name == "block_k" || axis.name.rfind("tile_", 0) == 0;
}

// Manhattan distance (in value-index space) between `a` and `b` over TILE axes
// only. Categorical axes are ignored so they cannot create false basin
// diversity.
int TileAxisManhattan(const DichotomicSearchSpace& space, const Coord& a,
                      const Coord& b) {
  const auto& axes = space.axes();
  const int num_axes = static_cast<int>(axes.size());
  int dist = 0;
  for (int j = 0; j < num_axes; ++j) {
    if (j >= static_cast<int>(a.size()) || j >= static_cast<int>(b.size())) {
      continue;
    }
    if (!IsTileAxis(axes[j])) continue;
    dist += std::abs(a[j] - b[j]);
  }
  return dist;
}

}  // namespace

std::vector<Coord> SelectTopKDiverse(const DichotomicSearchSpace& space,
                                     absl::Span<const Sample> samples, int k,
                                     int min_manhattan, double noise_tolerance) {
  std::vector<Coord> seeds;
  if (k <= 0) return seeds;

  // Default diversity threshold: half the tile-axis count (>= 1).
  if (min_manhattan <= 0) {
    int tile_axes = 0;
    for (const ParameterAxis& axis : space.axes()) {
      if (IsTileAxis(axis)) ++tile_axes;
    }
    min_manhattan = std::max(1, tile_axes / 2);
  }

  // Only evidence-grade samples may seed a basin. Sort a stable copy by measured
  // time ascending (fastest first). Keep the config index for deterministic tie
  // breaks.
  struct Cand {
    const Sample* sample;
    int config_index;
  };
  std::vector<Cand> evidence;
  evidence.reserve(samples.size());
  for (const Sample& s : samples) {
    if (s.is_evidence && std::isfinite(s.time_seconds)) {
      evidence.push_back({&s, space.LookupIndex(s.coord)});
    }
  }
  std::stable_sort(evidence.begin(), evidence.end(),
                   [](const Cand& a, const Cand& b) {
                     return a.sample->time_seconds < b.sample->time_seconds;
                   });
  if (evidence.empty()) return seeds;

  // Tracks which candidates have already been accepted as seeds.
  std::vector<bool> used(evidence.size(), false);

  // The min tile-axis distance from candidate `i` to every accepted seed;
  // +inf when no seeds accepted yet (so seed 1 is unconstrained).
  auto min_dist_to_seeds = [&](const Coord& c) -> int {
    if (seeds.empty()) return INT_MAX;
    int best = INT_MAX;
    for (const Coord& seed : seeds) {
      best = std::min(best, TileAxisManhattan(space, c, seed));
    }
    return best;
  };

  // Seed 1 = the globally fastest evidence sample.
  seeds.push_back(evidence.front().sample->coord);
  used[0] = true;

  // Subsequent seeds: among ELIGIBLE candidates (min tile-axis distance to all
  // accepted seeds >= min_manhattan), restrict to those WITHIN a noise band of
  // the best eligible time, and pick the one FARTHEST from the accepted seeds.
  // This spreads the basins as widely as possible when performances are similar,
  // while never preferring a clearly-slower candidate.
  while (static_cast<int>(seeds.size()) < k) {
    // Best eligible time (fastest candidate that is far enough from all seeds).
    double best_eligible_time = kInf;
    for (int i = 0; i < static_cast<int>(evidence.size()); ++i) {
      if (used[i]) continue;
      if (min_dist_to_seeds(evidence[i].sample->coord) < min_manhattan) {
        continue;
      }
      best_eligible_time =
          std::min(best_eligible_time, evidence[i].sample->time_seconds);
    }
    if (!std::isfinite(best_eligible_time)) break;  // no eligible candidate.

    const double time_band = best_eligible_time * (1.0 + noise_tolerance);

    // Among eligible candidates within the noise band of best_eligible_time,
    // pick max min-distance to seeds (ties: faster time, then lower index).
    int chosen = -1;
    int chosen_dist = -1;
    for (int i = 0; i < static_cast<int>(evidence.size()); ++i) {
      if (used[i]) continue;
      const Coord& c = evidence[i].sample->coord;
      const int dist = min_dist_to_seeds(c);
      if (dist < min_manhattan) continue;
      if (evidence[i].sample->time_seconds > time_band) continue;
      const bool better =
          dist > chosen_dist ||
          (dist == chosen_dist && chosen >= 0 &&
           (evidence[i].sample->time_seconds <
                evidence[chosen].sample->time_seconds ||
            (evidence[i].sample->time_seconds ==
                 evidence[chosen].sample->time_seconds &&
             evidence[i].config_index < evidence[chosen].config_index)));
      if (chosen < 0 || better) {
        chosen = i;
        chosen_dist = dist;
      }
    }
    if (chosen < 0) break;
    seeds.push_back(evidence[chosen].sample->coord);
    used[chosen] = true;
  }
  return seeds;
}

int BudgetLedger::RemainingCompilations() const {
  if (budget_.max_compilations <= 0) return INT_MAX;  // unlimited
  return std::max(0, budget_.max_compilations - compiled_);
}

bool BudgetLedger::Exhausted() const {
  if (budget_.max_compilations > 0 && compiled_ >= budget_.max_compilations) {
    return true;
  }
  if (budget_.max_tuning_time != absl::InfiniteDuration() &&
      absl::Now() - start_time_ >= budget_.max_tuning_time) {
    return true;
  }
  return false;
}

SearchBudget ResolveBudget(SearchBudgetPreset preset,
                           const DichotomicSearchSpace& space,
                           int64_t time_limit_ms) {
  double fraction = kBalancedBudgetFraction;
  switch (preset) {
    case SearchBudgetPreset::kFast:
      fraction = kFastBudgetFraction;
      break;
    case SearchBudgetPreset::kThorough:
      fraction = kThoroughBudgetFraction;
      break;
    case SearchBudgetPreset::kBalanced:
      fraction = kBalancedBudgetFraction;
      break;
  }

  // Budget = ceil(fraction * num_configs), clamped to [floor, num_configs].
  // The floor is kMinBudget, but never more than num_configs itself (a budget
  // can never exceed the exhaustive config count), so tiny spaces just get the
  // whole space.
  const int num_configs = space.num_configs();
  SearchBudget budget;
  if (num_configs <= 0) {
    budget.max_compilations = 0;  // empty space => nothing to compile.
  } else {
    const int floor = std::min(kMinBudget, num_configs);
    const int scaled = static_cast<int>(
        std::ceil(fraction * static_cast<double>(num_configs)));
    budget.max_compilations = std::clamp(scaled, floor, num_configs);
  }

  budget.max_tuning_time = time_limit_ms > 0
                               ? absl::Milliseconds(time_limit_ms)
                               : absl::InfiniteDuration();
  return budget;
}

int Phase1Cap(const BudgetLedger& ledger, const DichotomicSearchSpace& space) {
  const int max_compilations = ledger.budget().max_compilations;
  const int structural_min = MinPhase1Configs(space);
  if (max_compilations <= 0) {
    // Unlimited budget: request at least the structural minimum; the coarse
    // grid's own size otherwise bounds it.
    return structural_min;
  }
  const int fractional = static_cast<int>(
      std::ceil(kPhase1BudgetFraction * static_cast<double>(max_compilations)));
  return std::max(structural_min, fractional);
}

}  // namespace xla
