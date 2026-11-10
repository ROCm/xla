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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/valid_prefix_extent.h"

namespace xla {
namespace gpu {
namespace {

bool IsEligibleFusion(const HloInstruction* fusion) {
  return fusion->fusion_kind() == HloInstruction::FusionKind::kLoop &&
         fusion->shape().IsArray() && !fusion->shape().dimensions().empty();
}

bool ConsumesValidPrefix(const HloInstruction* user,
                         const HloInstruction* producer) {
  return user->IsCustomCall(kCublasLtGroupedMatmulCallTarget) &&
         user->operand_count() > 0 && user->operand(0) == producer;
}

absl::StatusOr<bool> RemoveValidPrefixMarkers(HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (!instruction->IsCustomCall(kValidPrefixCallTarget)) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(
        computation
            ->ReplaceInstruction(instruction, instruction->mutable_operand(0),
                                 /*preserve_sharding=*/false,
                                 /*relay_control_dependency=*/false,
                                 /*remove_unused_operands=*/false)
            .status());
    changed = true;
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> ValidPrefixExtentRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  absl::flat_hash_map<HloInstruction*, int64_t> candidates;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* fusion : computation->instructions()) {
      if (fusion->opcode() != HloOpcode::kFusion || !IsEligibleFusion(fusion)) {
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(std::optional<ValidPrefixExtent> extent,
                            GetValidPrefixExtent(*fusion));
      if (extent.has_value()) {
        candidates.emplace(fusion, extent->operand_index);
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
    for (const auto& [fusion, operand_index] : candidates) {
      if (fusion->user_count() == 0) {
        to_remove.push_back(fusion);
        continue;
      }
      for (HloInstruction* user : fusion->users()) {
        auto user_candidate = candidates.find(user);
        const bool same_extent_fusion = user_candidate != candidates.end() &&
                                        user->operand(user_candidate->second) ==
                                            fusion->operand(operand_index);
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
      if (fusion->opcode() != HloOpcode::kFusion ||
          candidates.contains(fusion)) {
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(
          bool removed,
          RemoveValidPrefixMarkers(fusion->fused_instructions_computation()));
      changed |= removed;
    }
  }

  // Markers that were not absorbed into a fusion are compile-time identities.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ABSL_ASSIGN_OR_RETURN(bool removed, RemoveValidPrefixMarkers(computation));
    changed |= removed;
  }
  if (changed) {
    HloDCE dce;
    ABSL_ASSIGN_OR_RETURN(bool dce_changed, dce.Run(module, execution_threads));
    changed |= dce_changed;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
