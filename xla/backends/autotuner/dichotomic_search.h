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

#ifndef XLA_BACKENDS_AUTOTUNER_DICHOTOMIC_SEARCH_H_
#define XLA_BACKENDS_AUTOTUNER_DICHOTOMIC_SEARCH_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/hlo/ir/hlo_opcode.h"

// This file implements an EXPERIMENTAL, Triton-only adaptive "dichotomic"
// (bisection / ternary) autotuning search. It is a generic engine that operates
// on abstract, ordered parameter axes extracted from the set of configs that
// the Triton backend already produces for exhaustive search. Because the axes
// are derived from that set, the search boundaries are exactly the feasible
// boundaries of the exhaustive search (no hardware feasibility formula is ever
// re-implemented here), and every config the engine selects is guaranteed to be
// a member of the original set.
//
// The design is documented in triton_dichotomic_search_design.md.
//
// The three phases are:
//   1. Coarse grid + role verification (prologue).
//   2. Coordinate-wise ternary refinement of unimodal axes.
//   3. +/-1 neighborhood sweep + full sweep of small ("sweep") axes.
//
// The engine itself is op-agnostic. The only op-specific knowledge is the
// per-axis "role" prior returned by MakeProfile(), which is verified and can
// only be *relaxed* (never promoted) by Phase-1 measurements.

namespace xla {

class HloInstruction;

// Alias matching xla/backends/autotuner/codegen_backend.h. The dichotomic
// search operates on the BackendConfig oneof proto (specifically the `triton`
// and `block_level` cases produced by the Triton backend).
using BackendConfig = autotuner::BackendConfig;

// The empirically-observed shape of the performance function along one ordered
// axis, holding all other axes fixed. See triton_dichotomic_search_design.md
// section 6 for the formal definitions and the safety ordering
// (kMonotone* is a special case of kUnimodal, which is a special case of
// kSweep -- roles may only be relaxed toward the safer end at runtime).
enum class AxisRole {
  // Larger value is faster: optimum is at the top end of the axis.
  kMonotoneUp,
  // Smaller value is faster: optimum is at the bottom end of the axis.
  kMonotoneDown,
  // Single interior optimum: performance degrades at both extremes. Amenable to
  // ternary search.
  kUnimodal,
  // Short axis or no reliable ordering: enumerate all values.
  kSweep,
};

// The three phases of the dichotomic search.
enum class SearchPhase {
  // Coarse grid over {min, median, max} of each ordered axis (also the
  // role-verification prologue).
  kCoarseGrid = 1,
  // Coordinate-wise ternary refinement of unimodal axes.
  kTernaryRefine = 2,
  // +/-1 neighborhood sweep around the best coordinate plus a full sweep of
  // all "sweep" axes.
  kNeighborhoodSweep = 3,
};

// One tunable dimension, with the DISTINCT values that actually appear across
// the supported configs (sorted ascending). Boundaries come "for free" from
// these values.
struct ParameterAxis {
  std::string name;
  std::vector<int64_t> values;  // sorted ascending, distinct
  bool ordered = true;          // false => categorical
};

// A coordinate in the discretized search space: one index per axis into
// ParameterAxis::values.
using Coord = std::vector<int>;

// Device-independent semantic role of the HLO dimension that a tuning axis
// tiles, as derived from indexing/tiling analysis. kParallel = an
// output/parallel dimension (N-like); kSequential = a contraction/reduction/
// scan dimension (K-like). kUnknown means "no analysis available for this axis"
// and causes MakeProfile to fall back to the name-based heuristic for it.
enum class AxisSemantics { kUnknown, kParallel, kSequential };

// A per-axis role assignment (the "search profile").
struct SearchProfile {
  // One role per axis, index-aligned with DichotomicSearchSpace::axes().
  std::vector<AxisRole> roles;

  // Full size of the HLO dimension each axis tiles, index-aligned with
  // DichotomicSearchSpace::axes(). Populated from tiling-analysis hints (see
  // AxisRoleHint::dimension_size); <= 0 means "unknown" (no analysis for that
  // axis). Used by the ternary/neighborhood phases for divisibility-aware
  // ("masking waste") probe placement and soft-pruning: a tile value v on an
  // axis of size D wastes fraction (ceil(D/v)*v - D)/(ceil(D/v)*v) of the last
  // block. When the size is unknown the divisibility logic is skipped, so this
  // is a strict, opt-in refinement that never changes behavior without hints.
  std::vector<int64_t> dimension_sizes;

  // Per-axis semantic role of the tiled HLO dimension (kParallel / kSequential
  // / kUnknown), index-aligned with DichotomicSearchSpace::axes(). Populated
  // from tiling-analysis hints (AxisRoleHint::semantics); kUnknown when no
  // analysis is available. Used by the static grid/work Pareto soft-prune to
  // know which axes contribute to the launch grid (only kParallel dims) vs the
  // reduction footprint. When semantics/dimension_sizes are unknown the static
  // proxy is disabled, so this is a strict, opt-in refinement.
  std::vector<AxisSemantics> semantics;
};

// Per-axis analysis fact for one tuning axis, taken straight from the
// indexing/tiling analysis
// (xla::gpu::experimental::TilingSpace::DimensionInfo:: {type, dimension_size})
struct AxisRoleHint {
  AxisSemantics semantics = AxisSemantics::kUnknown;
  int64_t dimension_size = 0;  // full extent of the tiled dim; <= 0 => unknown.
};

// Optional per-axis hints, INDEX-ALIGNED with DichotomicSearchSpace::axes()
// (i.e. hints[a] describes axis a), produced by the caller from tiling
// analysis. An empty vector, a size mismatch, or a kUnknown entry all cause
// MakeProfile to fall back to the name/opcode heuristic (globally for empty/
// mismatch, per-axis for kUnknown). When hints are present, the parallel axis
// tiling the LARGEST dimension is treated as the "N-like" monotone axis -- an
// analysis-driven choice (largest parallel dimension), not an axis-name guess.
using AxisRoleHints = std::vector<AxisRoleHint>;

// A single measured (coordinate, time-in-seconds) sample. Failed measurements
// should be represented with a large/infinite time and excluded by callers.
struct Sample {
  Coord coord;
  double time_seconds;
};

// The discretized search space built from a set of Triton BackendConfigs, plus
// an index mapping coordinates back to the original (feasible) configs.
class DichotomicSearchSpace {
 public:
  // Builds the axes from the DISTINCT values present in `configs`. All configs
  // must be Triton configs (the `triton` oneof case). Returns an error if the
  // set is empty or not Triton.
  static absl::StatusOr<DichotomicSearchSpace> Build(
      absl::Span<const BackendConfig* const> configs);

  const std::vector<ParameterAxis>& axes() const { return axes_; }

  // Returns the index into the original config vector for the exact coordinate,
  // or -1 if no such config exists in the supported set.
  int LookupIndex(const Coord& coord) const;

  // Snaps an arbitrary coordinate to the nearest existing coordinate (per-axis
  // nearest index, then a nearest-neighbor search among real configs). Always
  // returns a valid index into the original config vector.
  int SnapIndex(const Coord& coord) const;

  // Returns the coordinate of the config at `index` (into the original config
  // vector). `index` must be in [0, num_configs()). Inverse of LookupIndex.
  const Coord& CoordOf(int index) const { return coords_[index]; }

  // Maps an arbitrary Triton/block-level `config` to its coordinate in this
  // space by extracting the same knobs used to build the axes and looking up
  // each knob value's index on its axis. Returns false (and leaves `*coord`
  // unspecified) if the config's knob set is inconsistent with this space or a
  // value is not present on its axis. This avoids rebuilding a whole
  // DichotomicSearchSpace just to recover one config's coordinate.
  bool CoordForConfig(const BackendConfig& config, Coord* coord) const;

  int num_configs() const { return num_configs_; }

 private:
  std::vector<ParameterAxis> axes_;
  // Maps a coordinate (packed as a string key) to an index into the original
  // config vector.
  absl::flat_hash_map<std::string, int> coord_to_index_;
  // All valid coordinates, in the same order as the original configs.
  std::vector<Coord> coords_;
  int num_configs_ = 0;
};

// Returns the conservative hardcoded role prior for a Triton config, keyed on
// the instruction's opcode.
//
// The N-like parallel axis (block_n for TritonGemmKey configs; the widest
// output tile axis for block-level/ragged configs) is declared kMonotoneUp for
// all dot-like ops (kDot, kRaggedDot, and other matmul-shaped Triton ops),
// reflecting the strongly and universally verified "larger tile_n is better"
// finding. All other ordered axes default to kUnimodal and small/categorical
// axes to kSweep. For opcodes without a clear parallel dimension the N-like
// axis is left kUnimodal (safe fallback).
SearchProfile MakeProfile(const DichotomicSearchSpace& space, HloOpcode opcode);

// Same as above, but uses per-axis semantic roles derived from tiling/indexing
// analysis (`hints`, index-aligned with space.axes()) instead of the axis-name
// heuristic to identify the parallel (N-like) and sequential (contraction)
// axes:
//   - the dot-like op's parallel axis tiling the LARGEST dimension (the N-like
//     axis) -> kMonotoneUp (the verified "larger parallel tile is better"
//     prior),
//   - any other kSequential/kParallel ordered axis -> kUnimodal,
//   - small/categorical axes -> kSweep (unchanged).
// If `hints` is empty or its size != space.axes().size(), this behaves exactly
// like the opcode-only overload (name heuristic). Per-axis kUnknown entries
// also fall back to the name heuristic for that axis. This is a strict superset
// and never regresses the heuristic path. Roles are still verified/relaxed by
// RefineRoles from Phase-1 measurements.
SearchProfile MakeProfile(const DichotomicSearchSpace& space, HloOpcode opcode,
                          const AxisRoleHints& hints);

// Verifies the monotone priors against Phase-1 samples and RELAXES any that are
// contradicted (kMonotone* -> kUnimodal). Never promotes a role.
// `noise_tolerance` is the relative slack allowed when checking monotonicity
// (e.g. 0.03 for 3%).
SearchProfile RefineRoles(const SearchProfile& profile,
                          const DichotomicSearchSpace& space,
                          absl::Span<const Sample> phase1_samples,
                          double noise_tolerance = 0.03);

// Single entry point for config selection across all phases.
//
// Returns the indices (into the original config vector) that should be
// compiled+profiled in `phase`, excluding any index in `already_evaluated`.
//
//  - kCoarseGrid:        `prior_samples` is ignored. `max_configs` caps the
//                        coarse grid size.
//  - kTernaryRefine:     uses `prior_samples` (Phase-1 results) to seed
//                        coordinate-wise ternary probes.
//  - kNeighborhoodSweep: uses `prior_samples` (Phase-1+2 results) to locate the
//                        best coordinate, then sweeps its +/-1 neighborhood and
//                        all sweep axes.
//
// All returned indices correspond to real, feasible configs (via Snap/Lookup).
std::vector<int> SelectConfigs(const DichotomicSearchSpace& space,
                               const SearchProfile& profile, SearchPhase phase,
                               absl::Span<const Sample> prior_samples = {},
                               absl::Span<const int> already_evaluated = {},
                               int max_configs = 64);

// Returns the index (into space's config vector) of the best sample, or -1 if
// `samples` is empty.
int BestSampleIndex(const DichotomicSearchSpace& space,
                    absl::Span<const Sample> samples);

// The {grid, bytes} static proxy for one config, computed from the EXPERIMENTAL
// tiling analysis. `valid` is false on any analysis failure (config not
// prunable). Exposed so callers can memoize it across the search phases.
struct StaticCost {
  int64_t grid = 0;
  int64_t bytes = 0;
  bool valid = false;
};

// Memoization cache mapping a config index (into the original config vector) to
// its computed StaticCost. A config's static cost is phase-independent, so the
// caller should keep ONE cache for the whole dichotomic search and pass it to
// every ParetoPruneByStaticCost call -- this avoids rebuilding the (expensive)
// TilingSpace + tile propagation for the same config in each of the 3 phases.
using StaticCostCache = absl::flat_hash_map<int, StaticCost>;

// Drops statically Pareto-dominated candidates from `indices` using
// a device-model-FREE proxy computed from the EXPERIMENTAL tiling analysis of
// `instr`:
//
//   grid  = TiledHloComputation::num_output_tiles()  (occupancy pressure)
//   bytes = sum over fusion-operand tiles of
//           num_blocks * product(tile_sizes) * ByteWidth(elt)  (masking-aware
//           bytes read, i.e. memory traffic)
//
// A candidate is dropped iff some other candidate in `indices` dominates it on
// BOTH proxies (grid' <= grid AND bytes' <= bytes, at least one strictly less).
// `configs` must be INDEX-ALIGNED with the original config vector (configs[i]
// is the BackendConfig of config index i). `keep_index` (if >= 0) is never
// pruned (the current best). Returns the surviving subset of `indices`, order
// preserved. On any analysis failure, or if no candidate is dominated, returns
// `indices` unchanged -- so this is a strict, opt-in prune that can never
// remove the Pareto frontier (where the runtime optimum lives) nor the current
// best.
//
// `cache` (if non-null) memoizes per-config StaticCost across calls so the
// tiling analysis for a given config is performed at most once for the whole
// search rather than once per phase.
std::vector<int> ParetoPruneByStaticCost(
    const HloInstruction& instr, absl::Span<const BackendConfig* const> configs,
    absl::Span<const int> indices, int keep_index,
    StaticCostCache* cache = nullptr);

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_DICHOTOMIC_SEARCH_H_
