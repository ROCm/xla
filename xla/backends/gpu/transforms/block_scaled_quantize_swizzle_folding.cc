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

#include "xla/backends/gpu/transforms/block_scaled_quantize_swizzle_folding.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/autotuner/hipblaslt_scale_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

bool IsBlockScaledQuantizeFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  absl::StatusOr<GpuBackendConfig> config =
      instr.backend_config<GpuBackendConfig>();
  return config.ok() && config->fusion_backend_config().kind() ==
                            kBlockScaledQuantizeFusionKind;
}

// Collects the swizzles that read `value` directly, through bitcasts or
// through elements of optimization barriers, if nothing else reads it.
Decision CollectSwizzlesOf(HloInstruction* value,
                           std::vector<HloInstruction*>& swizzles) {
  if (value->IsRoot()) {
    return Decision::Forbid("the result is a computation result");
  }
  for (HloInstruction* user : value->users()) {
    if (IsHipblasLtScale32x8Swizzle(*user)) {
      swizzles.push_back(user);
      continue;
    }
    if (user->opcode() == HloOpcode::kBitcast) {
      Decision decision = CollectSwizzlesOf(user, swizzles);
      if (decision.IsForbidden()) {
        return decision;
      }
      continue;
    }
    if (user->opcode() != HloOpcode::kTuple || user->user_count() != 1 ||
        user->users().front()->opcode() != HloOpcode::kOptimizationBarrier) {
      return Decision::Forbid("the result has a user other than a swizzle");
    }
    HloInstruction* barrier = user->users().front();
    for (HloInstruction* element : barrier->users()) {
      if (element->opcode() != HloOpcode::kGetTupleElement) {
        return Decision::Forbid("a barrier has a user other than an element");
      }
      if (user->operand(element->tuple_index()) != value) {
        continue;
      }
      Decision decision = CollectSwizzlesOf(element, swizzles);
      if (decision.IsForbidden()) {
        return decision;
      }
    }
    if (barrier->IsRoot()) {
      return Decision::Forbid("the result is a computation result");
    }
  }
  return Decision::Allow();
}

// Whether the reshape from `shape` to `split` only splits dimensions.
bool OnlySplitsDimensions(const Shape& shape, const Shape& split) {
  int64_t next = 0;
  for (int64_t dim : shape.dimensions()) {
    int64_t product = 1;
    while (product < dim && next < split.dimensions().size()) {
      product *= split.dimensions(next++);
    }
    if (product != dim) {
      return false;
    }
  }
  while (next < split.dimensions().size() && split.dimensions(next) == 1) {
    ++next;
  }
  return next == split.dimensions().size();
}

// Collects the swizzles that read result `index` of `fusion` if nothing else
// reads it and they all permute it alike.
Decision CollectSwizzles(HloInstruction* fusion, int64_t index,
                         std::vector<HloInstruction*>& swizzles) {
  for (HloInstruction* element : fusion->users()) {
    if (element->opcode() != HloOpcode::kGetTupleElement) {
      return Decision::Forbid("the fusion has a user other than an element");
    }
    if (element->tuple_index() != index) {
      continue;
    }
    Decision decision = CollectSwizzlesOf(element, swizzles);
    if (decision.IsForbidden()) {
      return decision;
    }
  }
  if (swizzles.empty()) {
    return Decision::Forbid("no swizzle reads the result");
  }
  const HloInstruction* transpose =
      swizzles.front()->fused_expression_root()->operand(0);
  for (const HloInstruction* swizzle : swizzles) {
    if (swizzle->HasControlDependencies()) {
      return Decision::Forbid("a swizzle has control dependencies");
    }
    const HloInstruction* other = swizzle->fused_expression_root()->operand(0);
    if (!ShapeUtil::Equal(swizzle->operand(0)->shape(),
                          swizzles.front()->operand(0)->shape()) ||
        other->dimensions() != transpose->dimensions() ||
        !ShapeUtil::Equal(other->operand(0)->shape(),
                          transpose->operand(0)->shape())) {
      return Decision::Forbid("the swizzles of the result differ");
    }
  }
  const Shape& result = fusion->shape().tuple_shapes(index);
  // The swizzle reads its row-major operand through a bitcast, so the result
  // must be row-major too for both to see the same element order.
  if (!LayoutUtil::IsMonotonicWithDim0Major(result.layout())) {
    return Decision::Forbid("the result is not row-major");
  }
  return Decision(OnlySplitsDimensions(result, transpose->operand(0)->shape()),
                  "the result does not split into the swizzle's shape");
}

// Wraps result `index` of `fusion` in the reshape, transpose and reshape of the
// swizzles, and replaces the swizzles by their operands.
absl::Status FoldSwizzles(HloInstruction* fusion, int64_t index,
                          absl::Span<HloInstruction* const> swizzles) {
  HloComputation* fused = fusion->fused_instructions_computation();
  HloInstruction* root = fused->root_instruction();
  HloInstruction* scale = root->mutable_operand(index);
  const PrimitiveType type = scale->shape().element_type();
  const HloInstruction* transpose =
      swizzles.front()->fused_expression_root()->operand(0);
  const HloInstruction* split = transpose->operand(0);
  HloInstruction* new_split =
      fused->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::ChangeElementType(split->shape(), type), scale));
  HloInstruction* new_transpose =
      fused->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::ChangeElementType(transpose->shape(), type), new_split,
          transpose->dimensions()));
  HloInstruction* merge = fused->AddInstruction(
      HloInstruction::CreateReshape(scale->shape(), new_transpose));
  ABSL_RETURN_IF_ERROR(root->ReplaceOperandWith(index, merge));
  for (HloInstruction* swizzle : swizzles) {
    HloComputation* swizzle_computation =
        swizzle->fused_instructions_computation();
    ABSL_RETURN_IF_ERROR(swizzle->parent()->ReplaceInstruction(
        swizzle, swizzle->mutable_operand(0)));
    if (swizzle_computation->caller_instructions().empty()) {
      ABSL_RETURN_IF_ERROR(
          fusion->GetModule()->RemoveEmbeddedComputation(swizzle_computation));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> BlockScaledQuantizeSwizzleFolding::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> fusions;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (IsBlockScaledQuantizeFusion(*instr)) {
        fusions.push_back(instr);
      }
    }
  }
  bool changed = false;
  for (HloInstruction* fusion : fusions) {
    for (int64_t index = 0; index < fusion->shape().tuple_shapes().size();
         ++index) {
      std::vector<HloInstruction*> swizzles;
      Decision decision = CollectSwizzles(fusion, index, swizzles);
      if (decision.IsForbidden()) {
        VLOG(3) << "Not folding a swizzle into result " << index << " of "
                << fusion->name() << ": " << decision.Explain();
        continue;
      }
      ABSL_RETURN_IF_ERROR(FoldSwizzles(fusion, index, swizzles));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla::gpu
