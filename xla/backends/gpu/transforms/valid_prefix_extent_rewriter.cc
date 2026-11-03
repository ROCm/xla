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

#include "xla/backends/gpu/transforms/valid_prefix_extent_rewriter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/frontend_attributes.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/valid_prefix_extent.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

constexpr absl::string_view kAllowDceSideEffectingOp =
    "xla_allow_dce_side_effecting_op";

bool IsEligibleFusion(const HloInstruction* fusion) {
  return fusion->fusion_kind() == HloInstruction::FusionKind::kLoop &&
         fusion->shape().IsArray() && !fusion->shape().dimensions().empty();
}

bool ConsumesValidPrefix(const HloInstruction* user,
                         const HloInstruction* producer) {
  return user->IsCustomCall(kCublasLtGroupedMatmulCallTarget) &&
         user->operand_count() > 0 && user->operand(0) == producer;
}

bool DependsOn(const HloInstruction* instruction,
               const HloInstruction* dependency) {
  std::vector<const HloInstruction*> worklist = {instruction};
  absl::flat_hash_set<const HloInstruction*> visited;
  while (!worklist.empty()) {
    const HloInstruction* current = worklist.back();
    worklist.pop_back();
    if (current == dependency) {
      return true;
    }
    if (visited.insert(current).second) {
      worklist.insert(worklist.end(), current->operands().begin(),
                      current->operands().end());
    }
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> ValidPrefixExtentRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  absl::flat_hash_map<std::string, HloInstruction*> extent_sources;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      std::optional<std::string> source =
          instruction->get_frontend_attribute(kXlaValidPrefixExtentSource);
      if (source.has_value() && instruction->shape().IsArray() &&
          instruction->shape().dimensions().empty() &&
          instruction->shape().element_type() == S32) {
        auto [it, inserted] = extent_sources.emplace(*source, instruction);
        if (!inserted && it->second == nullptr) {
          continue;
        }
        if (!inserted && DependsOn(instruction, it->second)) {
          it->second = instruction;
        } else if (!inserted && !DependsOn(it->second, instruction)) {
          it->second = nullptr;
        }
      }
    }
  }

  absl::flat_hash_map<HloInstruction*, std::string> candidates;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* fusion : computation->instructions()) {
      if (fusion->opcode() != HloOpcode::kFusion || !IsEligibleFusion(fusion)) {
        continue;
      }
      std::optional<std::string> id =
          fusion->get_frontend_attribute(kXlaValidPrefixExtentId);
      if (!id.has_value()) {
        continue;
      }
      auto source = extent_sources.find(*id);
      if (source != extent_sources.end() && source->second != nullptr &&
          source->second->parent() == fusion->parent()) {
        candidates.emplace(fusion, *id);
      }
    }
  }

  // A bounded fusion may only feed another fusion with the same extent. If an
  // ordinary consumer could observe the unspecified tail, keep the producer
  // on the existing capacity-sized path instead.
  bool removed_candidate;
  do {
    removed_candidate = false;
    std::vector<HloInstruction*> to_remove;
    for (const auto& [fusion, id] : candidates) {
      for (HloInstruction* user : fusion->users()) {
        auto user_candidate = candidates.find(user);
        const bool same_extent_fusion =
            user_candidate != candidates.end() && user_candidate->second == id;
        const bool materializes_dynamic_result =
            user->IsCustomCall("SliceToDynamic") &&
            user->operand_count() >= 2 && user->operand(0) == fusion;
        if (!same_extent_fusion && !materializes_dynamic_result &&
            !ConsumesValidPrefix(user, fusion)) {
          to_remove.push_back(fusion);
          break;
        }
      }
    }
    for (HloInstruction* fusion : to_remove) {
      removed_candidate |= candidates.erase(fusion) != 0;
    }
  } while (removed_candidate);

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* fusion : computation->instructions()) {
      auto candidate = candidates.find(fusion);
      if (candidate == candidates.end()) {
        continue;
      }
      HloInstruction* source = extent_sources.at(candidate->second);

      int64_t operand_index = fusion->operand_count();
      for (int64_t i = 0; i < fusion->operand_count(); ++i) {
        if (fusion->operand(i) == source) {
          operand_index = i;
          break;
        }
      }
      HloInstruction* extent_parameter;
      if (operand_index == fusion->operand_count()) {
        extent_parameter = fusion->AddFusionOperand(source);
      } else {
        extent_parameter = fusion->fused_parameter(operand_index);
      }
      HloComputation* fused_computation =
          fusion->fused_instructions_computation();
      HloInstruction* root = fused_computation->root_instruction();
      // Fusion parameters must have a data user. This identity select is added
      // after fusion simplification and keeps n verifier-visible; codegen uses
      // the operand directly for the valid-prefix indexing constraint.
      HloInstruction* predicate =
          fused_computation->AddInstruction(HloInstruction::CreateCompare(
              ShapeUtil::MakeShape(PRED, {}), extent_parameter,
              extent_parameter, ComparisonDirection::kEq));
      HloInstruction* broadcast =
          fused_computation->AddInstruction(HloInstruction::CreateBroadcast(
              ShapeUtil::ChangeElementType(root->shape(), PRED), predicate,
              {}));
      HloInstruction* new_root =
          fused_computation->AddInstruction(HloInstruction::CreateTernary(
              root->shape(), HloOpcode::kSelect, broadcast, root, root));
      fused_computation->set_root_instruction(new_root);
      ABSL_RETURN_IF_ERROR(SetValidPrefixExtent(fusion, operand_index));
      changed = true;
    }
  }

  // The association is now represented by a real operand. Remove transient
  // metadata, and allow unused sources without an eligible fusion to be DCE'd.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      const bool source =
          instruction->get_frontend_attribute(kXlaValidPrefixExtentSource)
              .has_value();
      changed |=
          instruction->erase_frontend_attribute(kXlaValidPrefixExtentId) != 0;
      changed |= instruction->erase_frontend_attribute(
                     kXlaValidPrefixExtentSource) != 0;
      if (source) {
        changed |= instruction->erase_frontend_attribute(
                       kAllowDceSideEffectingOp) != 0;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
