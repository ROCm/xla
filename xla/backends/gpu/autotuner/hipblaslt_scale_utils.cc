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

#include "xla/backends/gpu/autotuner/hipblaslt_scale_utils.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/decision.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

Decision IsContiguousLayout(const Shape& shape, int64_t element_size_in_bits) {
  if (!shape.has_layout()) {
    return Decision::Forbid("32x8 scales require assigned layouts");
  }
  const Layout& layout = shape.layout();
  return Decision(LayoutUtil::IsMonotonicWithDim0Major(layout) &&
                      layout.tiles().empty() && !layout.has_physical_shape() &&
                      layout.dynamic_shape_metadata_prefix_bytes() == 0 &&
                      (layout.element_size_in_bits() == element_size_in_bits ||
                       (element_size_in_bits == 8 &&
                        layout.element_size_in_bits() == 0)),
                  "32x8 scales require contiguous, untiled layouts with "
                  "native element storage");
}

Decision IsSupportedScaleShape(const Shape& shape) {
  int64_t rank = shape.dimensions().size();
  if (shape.element_type() != F8E8M0FNU || (rank != 2 && rank != 3) ||
      (rank == 3 && shape.dimensions(0) != 1) ||
      !IsContiguousLayout(shape, 8).IsAllowed() || shape.is_dynamic()) {
    return Decision::Forbid(
        "32x8 scales require static row-major E8M0 "
        "[MN, K/32] or [1, MN, K/32]");
  }
  int64_t rows = shape.dimensions(rank - 2);
  int64_t cols = shape.dimensions(rank - 1);
  return Decision(rows > 0 && rows % 32 == 0 && cols > 0 && cols % 8 == 0,
                  "32x8 scales require MN divisible by 32 and K divisible by "
                  "256; padding is not supported");
}

// AITER/hipBLASLt E8M0 shuffle:
// [rows/32, 2, 16, cols/8, 2, 4] -> permute(0, 3, 5, 2, 4, 1).
constexpr int64_t kScale32x8Permutation[] = {0, 3, 5, 2, 4, 1};

Shape SplitScaleShape(int64_t rows, int64_t cols) {
  return ShapeUtil::MakeShape(F8E8M0FNU, {rows / 32, 2, 16, cols / 8, 2, 4});
}

Shape SwizzledScaleShape(int64_t rows, int64_t cols) {
  return ShapeUtil::MakeShape(F8E8M0FNU, {rows / 32, cols / 8, 4, 16, 2, 2});
}

}  // namespace

Decision CanUseHipblasLtScale32x8(
    const HloInstruction& scaled_dot,
    const stream_executor::GpuComputeCapability& gpu_version) {
  const auto* rocm = gpu_version.rocm_compute_capability();
  if (rocm == nullptr || !rocm->gfx9_mi350()) {
    return Decision::Forbid("hipBLASLt 32x8 scales require gfx950");
  }
  if (scaled_dot.opcode() != HloOpcode::kScaledDot ||
      scaled_dot.operand_count() != 4) {
    return Decision::Forbid("Expected a scaled-dot with four operands");
  }
  PrimitiveType input_type = scaled_dot.operand(0)->shape().element_type();
  if ((input_type != F8E4M3FN && input_type != F4E2M1FN) ||
      scaled_dot.operand(1)->shape().element_type() != input_type) {
    return Decision::Forbid(
        "32x8 scales require matching E4M3FN or E2M1FN inputs");
  }
  // GPU FP4 buffers pack two elements per byte and must have E(4) layouts.
  // An unspecified element size instead denotes byte storage for FP4.
  int64_t input_element_size_in_bits = input_type == F4E2M1FN ? 4 : 8;
  const DotDimensionNumbers& dims = scaled_dot.dot_dimension_numbers();
  for (int side = 0; side != 2; ++side) {
    const Shape& input = scaled_dot.operand(side)->shape();
    const Shape& scale = scaled_dot.operand(side + 2)->shape();
    Decision scale_supported = IsSupportedScaleShape(scale);
    if (!scale_supported.IsAllowed()) return scale_supported;
    int64_t rank = input.dimensions().size();
    const auto& contracting = side == 0 ? dims.lhs_contracting_dimensions()
                                        : dims.rhs_contracting_dimensions();
    const auto& batch =
        side == 0 ? dims.lhs_batch_dimensions() : dims.rhs_batch_dimensions();
    if (rank != scale.dimensions().size() || input.is_dynamic() ||
        !IsContiguousLayout(input, input_element_size_in_bits).IsAllowed() ||
        contracting.size() != 1 || contracting[0] != rank - 1 ||
        (rank == 2 && !batch.empty()) ||
        (rank == 3 && (batch.size() != 1 || batch[0] != 0)) ||
        input.dimensions(rank - 1) != scale.dimensions(rank - 1) * 32) {
      return Decision::Forbid(
          "32x8 scales require row-major E4M3FN or packed E2M1FN inputs "
          "with the final dimension contracting");
    }
    for (int64_t dim = 0; dim < rank - 1; ++dim) {
      if (input.dimensions(dim) != scale.dimensions(dim)) {
        return Decision::Forbid("32x8 scale and input free dimensions differ");
      }
    }
  }
  return Decision::Allow();
}

absl::StatusOr<HloInstruction*> SwizzleHipblasLtScale32x8(
    HloInstruction* scale) {
  const Shape& shape = scale->shape();
  TF_RET_CHECK(IsSupportedScaleShape(shape).IsAllowed());
  int64_t rank = shape.dimensions().size();
  int64_t rows = shape.dimensions(rank - 2);
  int64_t cols = shape.dimensions(rank - 1);

  // ApplyConfig swizzles the scale operands of every 32x8 scaled-dot, and a
  // scale often feeds several of them (the q, k and v projections, or the
  // two MLP input projections). The swizzle is a pure byte permutation of its
  // operand, so an existing one is reused, unless control dependencies order
  // it against other instructions: a new user would inherit that ordering.
  for (HloInstruction* user : scale->users()) {
    if (IsHipblasLtScale32x8Swizzle(*user) &&
        user->control_predecessors().empty() &&
        user->control_successors().empty()) {
      return user;
    }
  }

  // Each scale byte is copied exactly; there is no floating-point arithmetic.
  HloComputation::Builder builder("hipblaslt_mx_scale_32x8");
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "scale"));
  HloInstruction* split = builder.AddInstruction(
      HloInstruction::CreateReshape(SplitScaleShape(rows, cols), input));
  HloInstruction* transpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          SwizzledScaleShape(rows, cols), split, kScale32x8Permutation));
  HloInstruction* result =
      builder.AddInstruction(HloInstruction::CreateReshape(shape, transpose));
  HloComputation* computation =
      scale->GetModule()->AddEmbeddedComputation(builder.Build(result));
  // Name the instruction before insertion so the module's name uniquer handles
  // both operand scales (and additional MX GEMMs in the same module).
  return scale->parent()->AddInstruction(
      HloInstruction::CreateFusion(shape, HloInstruction::FusionKind::kLoop,
                                   {scale}, computation),
      "hipblaslt_mx_scale_32x8");
}

bool IsHipblasLtScale32x8Swizzle(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion ||
      instr.fusion_kind() != HloInstruction::FusionKind::kLoop ||
      instr.operand_count() != 1) {
    return false;
  }
  const Shape& shape = instr.operand(0)->shape();
  if (!IsSupportedScaleShape(shape).IsAllowed() ||
      !ShapeUtil::Equal(instr.shape(), shape)) {
    return false;
  }
  int64_t rank = shape.dimensions().size();
  int64_t rows = shape.dimensions(rank - 2);
  int64_t cols = shape.dimensions(rank - 1);

  // reshape(transpose(reshape(parameter 0))), and nothing else.
  const HloComputation* computation = instr.fused_instructions_computation();
  if (computation->instruction_count() != 4) {
    return false;
  }
  const HloInstruction* merge = computation->root_instruction();
  if (merge->opcode() != HloOpcode::kReshape ||
      !ShapeUtil::Compatible(merge->shape(), shape)) {
    return false;
  }
  const HloInstruction* transpose = merge->operand(0);
  if (transpose->opcode() != HloOpcode::kTranspose ||
      transpose->dimensions() != absl::MakeConstSpan(kScale32x8Permutation) ||
      !ShapeUtil::Compatible(transpose->shape(),
                             SwizzledScaleShape(rows, cols))) {
    return false;
  }
  const HloInstruction* split = transpose->operand(0);
  if (split->opcode() != HloOpcode::kReshape ||
      !ShapeUtil::Compatible(split->shape(), SplitScaleShape(rows, cols))) {
    return false;
  }
  const HloInstruction* parameter = split->operand(0);
  return parameter->opcode() == HloOpcode::kParameter &&
         parameter->parameter_number() == 0;
}

}  // namespace xla::gpu
