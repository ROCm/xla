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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/origami.hpp"
#include "origami/types.hpp"
#include "xla/backends/gpu/autotuner/triton/origami_cost_model.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::origami::config_t;
using ::origami::data_type_t;
using ::origami::dim3_t;
using ::origami::hardware_t;
using ::origami::problem_t;
using ::origami::transpose_t;

using TileShape = std::tuple<int, int, int>;
// What Origami actually consumes: tile shape, MFMA size, occupancy.
using OrigamiKey = std::tuple<int, int, int, int, int>;
// What SelectTopConfigs budgets over. Occupancy is excluded: it is derived
// from num_stages, so including it would spend the budget on knob variants of
// one tile instead of on distinct tiles.
using GroupKey = std::tuple<int, int, int, int>;

GroupKey KeyOf(const TritonGemmConfig& c) {
  return {c.block_m, c.block_n, c.block_k, c.mfma_size > 0 ? c.mfma_size : 16};
}

// Origami has no counterpart for these, so sweeping them multiplies the
// candidate set without the model being able to discriminate. Values follow
// tritonBLAS, which pins rather than searches them.
bool HasAllowedKnobs(const TritonGemmConfig& c) {
  if (c.is_tma_allowed || c.is_warp_specialization_allowed) return false;
  if (c.num_warps != 4 && c.num_warps != 8) return false;
  if (c.num_stages != 2 && c.num_stages != 3) return false;
  return true;
}

struct MatrixInstruction {
  int64_t m;
  int64_t n;
  int64_t k;
};

// gfx950 inserts PADDING elements after every INTERVAL for 16-bit operands on
// the async-copy path (Triton's composePaddedLayoutForAsyncCopyCDNA4).
constexpr int64_t kGfx950PadInterval = 512;
constexpr int64_t kGfx950PadElems = 16;
constexpr int64_t kCacheLineBytes = 128;
constexpr int kMaxOccupancy = 8;

std::optional<data_type_t> ToOrigamiDtype(PrimitiveType type) {
  switch (type) {
    case F32:
      return data_type_t::Float;
    case BF16:
      return data_type_t::BFloat16;
    case F16:
      return data_type_t::Half;
    case F8E4M3FNUZ:
      return data_type_t::Float8_fnuz;
    case F8E5M2FNUZ:
      return data_type_t::BFloat8_fnuz;
    case F8E4M3FN:
      return data_type_t::Float8;
    case F8E5M2:
      return data_type_t::BFloat8;
    case S8:
      return data_type_t::Int8;
    case F4E2M1FN:
      return data_type_t::Float4;
    default:
      return std::nullopt;
  }
}

// mfma_size gives MI's M and N; K comes from this table.
std::optional<MatrixInstruction> GetMatrixInstruction(absl::string_view gfx,
                                                      int64_t operand_bits,
                                                      int64_t mfma_size) {
  const int64_t mn = mfma_size > 0 ? mfma_size : 16;
  std::optional<int64_t> k;
  if (gfx == "gfx950") {
    if (operand_bits == 32) k = 4;
    if (operand_bits == 16) k = 32;
    if (operand_bits <= 8) k = 128;
  } else if (gfx == "gfx942") {
    if (operand_bits == 32) k = 4;
    if (operand_bits == 16) k = 16;
    if (operand_bits == 8) k = 32;
  } else if (gfx == "gfx90a") {
    if (operand_bits == 32) k = 4;
    if (operand_bits == 16) k = 16;
  }
  if (!k.has_value()) return std::nullopt;
  return MatrixInstruction{mn, mn, *k};
}

int64_t PaddedElems(int64_t n) {
  return n + ((n - 1) / kGfx950PadInterval) * kGfx950PadElems;
}

// Triton's shared-memory allocation, which differs from Origami's raw-tile
// assumption.
int64_t TritonSharedMemoryBytes(const TritonGemmConfig& config,
                                int64_t lhs_bits, int64_t rhs_bits,
                                bool async_copy) {
  const int64_t a_bytes =
      CeilOfRatio<int64_t>(config.block_m * config.block_k * lhs_bits, 8);
  const int64_t b_bytes =
      CeilOfRatio<int64_t>(config.block_k * config.block_n * rhs_bits, 8);
  if (config.num_stages <= 1) return std::max(a_bytes, b_bytes);
  if (async_copy) {
    int64_t a_elems = config.num_stages * config.block_m * config.block_k;
    int64_t b_elems = config.num_stages * config.block_k * config.block_n;
    if (lhs_bits == 16) a_elems = PaddedElems(a_elems);
    if (rhs_bits == 16) b_elems = PaddedElems(b_elems);
    return CeilOfRatio<int64_t>(a_elems * lhs_bits, 8) +
           CeilOfRatio<int64_t>(b_elems * rhs_bits, 8);
  }
  return (config.num_stages - 1) * (a_bytes + b_bytes);
}

// Workgroups resident per CU, bounded by LDS alone. Origami has no view of
// Triton's allocation, so it cannot derive this itself.
int LdsOccupancy(int64_t lds_bytes, int64_t lds_capacity) {
  if (lds_bytes <= 0) return 1;
  return std::clamp<int>(lds_capacity / lds_bytes, 1, kMaxOccupancy);
}

// Tensile calls an operand transposed when its contracting dimension is
// minor-most. Row-major A[M,K] is T, row-major B[K,N] is N.
transpose_t ContractingIsMinor(const Shape& shape, int64_t contracting_dim) {
  if (!shape.has_layout() || shape.dimensions().size() < 2) {
    return transpose_t::N;
  }
  return shape.layout().minor_to_major(0) == contracting_dim ? transpose_t::T
                                                             : transpose_t::N;
}

struct DotShape {
  int64_t b, m, n, k;
  int64_t lhs_contracting, rhs_contracting;
  PrimitiveType lhs_type, rhs_type, out_type;
};

absl::StatusOr<DotShape> GetDotShape(const HloDotInstruction* dot) {
  const Shape& lhs = dot->operand(0)->shape();
  const Shape& rhs = dot->operand(1)->shape();
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();

  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1) {
    return absl::UnimplementedError("expected exactly one contracting dim");
  }
  DimensionVector lhs_nc = GetNonContractingDims(
      lhs.dimensions().size(), dnums.lhs_contracting_dimensions(),
      dnums.lhs_batch_dimensions());
  DimensionVector rhs_nc = GetNonContractingDims(
      rhs.dimensions().size(), dnums.rhs_contracting_dimensions(),
      dnums.rhs_batch_dimensions());
  if (lhs_nc.size() != 1 || rhs_nc.size() != 1) {
    return absl::UnimplementedError("expected exactly one non-contracting dim");
  }

  DotShape s;
  s.b = 1;
  for (int64_t d : dnums.lhs_batch_dimensions()) s.b *= lhs.dimensions(d);
  s.lhs_contracting = dnums.lhs_contracting_dimensions(0);
  s.rhs_contracting = dnums.rhs_contracting_dimensions(0);
  s.m = lhs.dimensions(lhs_nc[0]);
  s.n = rhs.dimensions(rhs_nc[0]);
  s.k = lhs.dimensions(s.lhs_contracting);
  s.lhs_type = lhs.element_type();
  s.rhs_type = rhs.element_type();
  s.out_type = dot->shape().element_type();
  return s;
}

}  // namespace

class OrigamiDotCostModel::Impl {
 public:
  Impl(hardware_t hardware, std::string gfx, int64_t shared_memory_limit,
       int64_t l2_bytes, double clock_ghz)
      : hardware_(std::move(hardware)),
        gfx_(std::move(gfx)),
        shared_memory_limit_(shared_memory_limit),
        l2_bytes_(l2_bytes),
        clock_ghz_(clock_ghz),
        async_copy_(gfx_ == "gfx950") {}

  absl::StatusOr<std::vector<TritonGemmConfig>> FilterNarrowReads(
      const HloDotInstruction* dot,
      absl::Span<const TritonGemmConfig> configs) const {
    ABSL_ASSIGN_OR_RETURN(DotShape shape, GetDotShape(dot));
    const transpose_t a_trans =
        ContractingIsMinor(dot->operand(0)->shape(), shape.lhs_contracting);
    const transpose_t b_trans =
        ContractingIsMinor(dot->operand(1)->shape(), shape.rhs_contracting);
    std::vector<TritonGemmConfig> kept;
    kept.reserve(configs.size());
    for (const TritonGemmConfig& c : configs) {
      if (!HasNarrowDramRead(c, shape, a_trans, b_trans)) kept.push_back(c);
    }
    return kept;
  }

  absl::StatusOr<absl::flat_hash_map<TritonGemmConfig, absl::Duration>>
  EstimateRunTimes(const HloDotInstruction* dot,
                   absl::Span<const TritonGemmConfig> configs) const {
    ABSL_ASSIGN_OR_RETURN(DotShape shape, GetDotShape(dot));
    ABSL_ASSIGN_OR_RETURN(problem_t problem, MakeProblem(dot, shape));

    const int64_t lhs_bits = primitive_util::BitWidth(shape.lhs_type);
    const int64_t rhs_bits = primitive_util::BitWidth(shape.rhs_type);
    const int64_t operand_bits = std::max(lhs_bits, rhs_bits);

    // num_warps and num_stages are not forwarded to Origami, so the distinct
    // inputs are far fewer than the configs. Shared-memory capacity does
    // depend on num_stages, so that check stays per config.
    absl::flat_hash_map<OrigamiKey, std::optional<absl::Duration>> priced;
    absl::flat_hash_map<TritonGemmConfig, absl::Duration> estimates;
    estimates.reserve(configs.size());
    for (const TritonGemmConfig& config : configs) {
      if (estimates.contains(config)) continue;
      if (config.block_m <= 0 || config.block_n <= 0 || config.block_k <= 0) {
        continue;
      }
      const int64_t lds_bytes =
          TritonSharedMemoryBytes(config, lhs_bits, rhs_bits, async_copy_);
      if (lds_bytes > shared_memory_limit_) continue;
      const int mfma_size = config.mfma_size > 0 ? config.mfma_size : 16;
      const int occupancy = LdsOccupancy(lds_bytes, shared_memory_limit_);
      auto [it, inserted] =
          priced.try_emplace(OrigamiKey{config.block_m, config.block_n,
                                        config.block_k, mfma_size, occupancy});
      if (inserted) {
        it->second =
            Estimate(problem, config, operand_bits, mfma_size, occupancy);
      }
      if (it->second.has_value()) estimates.emplace(config, *it->second);
    }
    VLOG(2) << "Origami priced " << priced.size() << " distinct inputs for "
            << configs.size() << " configs";
    return estimates;
  }

 private:
  absl::StatusOr<problem_t> MakeProblem(const HloDotInstruction* dot,
                                        const DotShape& shape) const {
    std::optional<data_type_t> lhs = ToOrigamiDtype(shape.lhs_type);
    std::optional<data_type_t> rhs = ToOrigamiDtype(shape.rhs_type);
    std::optional<data_type_t> out = ToOrigamiDtype(shape.out_type);
    if (!lhs.has_value() || !rhs.has_value() || !out.has_value()) {
      return absl::UnimplementedError(
          absl::StrCat("Origami does not model element types ",
                       PrimitiveType_Name(shape.lhs_type), "x",
                       PrimitiveType_Name(shape.rhs_type), "->",
                       PrimitiveType_Name(shape.out_type)));
    }

    problem_t problem;
    problem.size = dim3_t(shape.m, shape.n, shape.k);
    problem.batch = std::max<int64_t>(shape.b, 1);
    problem.a_transpose =
        ContractingIsMinor(dot->operand(0)->shape(), shape.lhs_contracting);
    problem.b_transpose =
        ContractingIsMinor(dot->operand(1)->shape(), shape.rhs_contracting);
    problem.a_dtype = *lhs;
    problem.b_dtype = *rhs;
    problem.c_dtype = *out;
    problem.d_dtype = *out;
    // The MFMA type follows the inputs, not the output.
    problem.mi_dtype = primitive_util::BitWidth(shape.lhs_type) <=
                               primitive_util::BitWidth(shape.rhs_type)
                           ? *lhs
                           : *rhs;
    problem.a_mx_block_size = 0;
    problem.b_mx_block_size = 0;
    problem.num_cus = 0;
    return problem;
  }

  // True when a tile reads an operand in strips narrower than a cache line
  // AND that operand is too large to stay resident in L2, so the wasted
  // fraction of each line reaches DRAM. Only the non-contracting dimension
  // counts: where the contracting dimension is minor-most, the rest of the
  // line is what the same workgroup reads on its next k-iteration. Origami
  // assumes the sharing holds either way and under-predicts such tilings by
  // ~2x. See docs/origami-upstream-issue.md.
  bool HasNarrowDramRead(const TritonGemmConfig& config, const DotShape& shape,
                         transpose_t a_trans, transpose_t b_trans) const {
    const int64_t lhs_bits = primitive_util::BitWidth(shape.lhs_type);
    const int64_t rhs_bits = primitive_util::BitWidth(shape.rhs_type);
    const int64_t batch = std::max<int64_t>(shape.b, 1);

    auto narrow = [&](transpose_t trans, int64_t tile_dim, int64_t problem_dim,
                      int64_t bits, int64_t elems) {
      if (trans == transpose_t::T) return false;
      return CeilOfRatio<int64_t>(tile_dim * bits, 8) < kCacheLineBytes &&
             CeilOfRatio<int64_t>(elems * bits, 8) > l2_bytes_ &&
             tile_dim < problem_dim;
    };
    return narrow(a_trans, config.block_m, shape.m, lhs_bits,
                  batch * shape.m * shape.k) ||
           narrow(b_trans, config.block_n, shape.n, rhs_bits,
                  batch * shape.k * shape.n);
  }

  std::optional<absl::Duration> Estimate(const problem_t& problem,
                                         const TritonGemmConfig& config,
                                         int64_t operand_bits, int mfma_size,
                                         int occupancy) const {
    std::optional<MatrixInstruction> mi =
        GetMatrixInstruction(gfx_, operand_bits, mfma_size);
    if (!mi.has_value()) return std::nullopt;
    if (config.block_m < mi->m || config.block_n < mi->n) return std::nullopt;

    config_t c;
    c.mt = dim3_t(config.block_m, config.block_n, config.block_k);
    c.mi = dim3_t(mi->m, mi->n, mi->k);
    c.occupancy = occupancy;
    c.stream_k = 0;
    c.grid_selection = origami::grid_selection_t::data_parallel;

    const double cycles =
        origami::gemm::compute_total_latency(problem, hardware_, c);
    if (!std::isfinite(cycles) || cycles <= 0.0 || cycles > 1e300) {
      return std::nullopt;
    }
    return absl::Nanoseconds(cycles / clock_ghz_);
  }

  hardware_t hardware_;
  std::string gfx_;
  int64_t shared_memory_limit_;
  int64_t l2_bytes_;
  double clock_ghz_;
  bool async_copy_;
};

OrigamiDotCostModel::OrigamiDotCostModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

OrigamiDotCostModel::~OrigamiDotCostModel() = default;

absl::StatusOr<std::unique_ptr<OrigamiDotCostModel>>
OrigamiDotCostModel::Create(const se::DeviceDescription& device_description) {
  const se::GpuComputeCapability& cc =
      device_description.gpu_compute_capability();
  if (!cc.IsRocm()) {
    return absl::UnimplementedError(
        "Origami cost model requires a ROCm target");
  }
  const std::string gfx = cc.rocm_compute_capability()->gfx_version();
  const auto arch = hardware_t::arch_name_to_enum(gfx);
  if (arch == hardware_t::architecture_t::Count) {
    return absl::UnimplementedError(
        absl::StrCat("Origami does not know architecture ", gfx));
  }

  const int64_t cus = device_description.core_count();
  const int64_t lds = device_description.shared_memory_per_core();
  const int64_t regs = device_description.registers_per_core_limit();
  const int64_t l2 = device_description.l2_cache_size();
  const float clock_ghz = device_description.clock_rate_ghz();
  if (cus <= 0 || lds <= 0 || regs <= 0 || l2 <= 0 || clock_ghz <= 0.f) {
    return absl::FailedPreconditionError(absl::StrCat(
        "incomplete device description for Origami: core_count=", cus,
        " shared_memory_per_core=", lds, " registers_per_core_limit=", regs,
        " l2_cache_size=", l2, " clock_rate_ghz=", clock_ghz));
  }

  hardware_t hardware = hardware_t::get_hardware_for_arch(
      arch, cus, lds, regs, l2, static_cast<int>(clock_ghz * 1e6));

  return absl::WrapUnique(new OrigamiDotCostModel(std::make_unique<Impl>(
      std::move(hardware), gfx, lds, l2, static_cast<double>(clock_ghz))));
}

absl::StatusOr<absl::flat_hash_map<TritonGemmConfig, absl::Duration>>
OrigamiDotCostModel::EstimateRunTimes(
    const HloDotInstruction* dot,
    absl::Span<const TritonGemmConfig> configs) const {
  return impl_->EstimateRunTimes(dot, configs);
}

absl::StatusOr<std::vector<TritonGemmConfig>>
OrigamiDotCostModel::SelectTopConfigs(
    const HloDotInstruction* dot, absl::Span<const TritonGemmConfig> configs,
    const SelectionOptions& options) const {
  std::vector<TritonGemmConfig> candidates;
  candidates.reserve(configs.size());
  for (const TritonGemmConfig& config : configs) {
    if (!options.restrict_knobs || HasAllowedKnobs(config))
      candidates.push_back(config);
  }
  if (candidates.empty()) candidates.assign(configs.begin(), configs.end());

  if (options.drop_narrow_reads) {
    absl::StatusOr<std::vector<TritonGemmConfig>> narrowed =
        impl_->FilterNarrowReads(dot, candidates);
    if (narrowed.ok() && !narrowed->empty()) {
      VLOG(2) << "Origami dropped " << candidates.size() - narrowed->size()
              << " configs reading an out-of-cache operand narrower than a "
                 "cache line";
      candidates = *std::move(narrowed);
    }
  }

  ABSL_ASSIGN_OR_RETURN(auto estimates, EstimateRunTimes(dot, candidates));

  absl::flat_hash_map<GroupKey, absl::Duration> best_per_key;
  for (const auto& [config, duration] : estimates) {
    auto [it, inserted] = best_per_key.try_emplace(KeyOf(config), duration);
    if (!inserted && duration < it->second) it->second = duration;
  }

  std::vector<std::pair<absl::Duration, GroupKey>> ranked;
  ranked.reserve(best_per_key.size());
  for (const auto& [key, duration] : best_per_key) {
    ranked.push_back({duration, key});
  }
  absl::c_sort(ranked);
  if (options.num_groups > 0 &&
      static_cast<int64_t>(ranked.size()) > options.num_groups) {
    ranked.resize(options.num_groups);
  }

  absl::flat_hash_set<GroupKey> keep;
  keep.reserve(ranked.size());
  for (const auto& [duration, key] : ranked) keep.insert(key);

  absl::flat_hash_set<TileShape> shapes;
  std::vector<TritonGemmConfig> result;
  result.reserve(candidates.size());
  for (const TritonGemmConfig& config : candidates) {
    if (!estimates.contains(config)) continue;
    if (keep.contains(KeyOf(config))) {
      result.push_back(config);
      shapes.insert({config.block_m, config.block_n, config.block_k});
    }
  }

  VLOG(2) << "Origami kept " << result.size() << " of " << configs.size()
          << " configs: " << keep.size() << " of " << best_per_key.size()
          << " estimated configurations, " << shapes.size() << " tile shapes";
  return result;
}

absl::Status IsSupportedByOrigami(
    const HloDotInstruction* dot,
    const se::DeviceDescription& device_description) {
  const se::GpuComputeCapability& cc =
      device_description.gpu_compute_capability();
  if (!cc.IsRocm()) {
    return absl::UnimplementedError("not a ROCm target");
  }
  if (hardware_t::arch_name_to_enum(
          cc.rocm_compute_capability()->gfx_version()) ==
      hardware_t::architecture_t::Count) {
    return absl::UnimplementedError(absl::StrCat(
        "unknown architecture ", cc.rocm_compute_capability()->gfx_version()));
  }
  ABSL_ASSIGN_OR_RETURN(DotShape shape, GetDotShape(dot));
  if (!ToOrigamiDtype(shape.lhs_type).has_value() ||
      !ToOrigamiDtype(shape.rhs_type).has_value() ||
      !ToOrigamiDtype(shape.out_type).has_value()) {
    return absl::UnimplementedError("unmodelled element type");
  }
  if (!GetMatrixInstruction(cc.rocm_compute_capability()->gfx_version(),
                            std::max(primitive_util::BitWidth(shape.lhs_type),
                                     primitive_util::BitWidth(shape.rhs_type)),
                            0)
           .has_value()) {
    return absl::UnimplementedError("no MFMA instruction for this dtype/arch");
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
