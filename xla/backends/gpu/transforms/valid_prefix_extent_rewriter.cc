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
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/valid_prefix_extent.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

bool IsEligibleFusion(const HloInstruction* fusion) {
  return fusion->fusion_kind() == HloInstruction::FusionKind::kLoop &&
         fusion->shape().IsArray() && !fusion->shape().dimensions().empty();
}

bool PreservesValidPrefix(const HloInstruction& user, int64_t operand_index) {
  const Shape& input = user.operand(operand_index)->shape();
  const Shape& output = user.shape();
  if (!input.IsArray() || input.dimensions().empty() || !output.IsArray() ||
      output.dimensions().empty() ||
      input.dimensions(0) != output.dimensions(0)) {
    return false;
  }
  if (user.IsElementwise()) {
    return input.dimensions() == output.dimensions();
  }
  switch (user.opcode()) {
    case HloOpcode::kCopy:
    case HloOpcode::kReshape:
      return operand_index == 0;
    case HloOpcode::kBroadcast:
      return operand_index == 0 && !user.dimensions().empty() &&
             user.dimensions(0) == 0;
    default:
      return false;
  }
}

std::optional<int64_t> GetValidPrefixExtentOperandIndex(
    const HloInstruction* fusion) {
  using OperandIndices = absl::flat_hash_set<int64_t>;
  absl::flat_hash_map<const HloInstruction*, OperandIndices> extents;
  std::vector<const HloInstruction*> worklist;

  for (const HloInstruction* instruction :
       fusion->fused_instructions_computation()->instructions()) {
    if (!instruction->IsCustomCall("xla.valid_prefix") ||
        instruction->operand_count() != 2 ||
        instruction->operand(1)->opcode() != HloOpcode::kParameter ||
        instruction->operand(1)->shape().element_type() != S32 ||
        !instruction->operand(1)->shape().dimensions().empty()) {
      continue;
    }
    extents[instruction].insert(instruction->operand(1)->parameter_number());
    worklist.push_back(instruction);
  }

  while (!worklist.empty()) {
    const HloInstruction* instruction = worklist.back();
    worklist.pop_back();
    for (const HloInstruction* user : instruction->users()) {
      bool preserves = false;
      for (int64_t i = 0; i < user->operand_count(); ++i) {
        preserves |=
            user->operand(i) == instruction && PreservesValidPrefix(*user, i);
      }
      if (!preserves) {
        continue;
      }
      OperandIndices& user_extents = extents[user];
      size_t old_size = user_extents.size();
      user_extents.insert(extents[instruction].begin(),
                          extents[instruction].end());
      if (user_extents.size() != old_size) {
        worklist.push_back(user);
      }
    }
  }

  const OperandIndices& root_extents =
      extents[fusion->fused_instructions_computation()->root_instruction()];
  if (root_extents.size() != 1) {
    return std::nullopt;
  }
  return *root_extents.begin();
}

bool ConsumesValidPrefix(const HloInstruction* user,
                         const HloInstruction* producer) {
  return user->IsCustomCall(kCublasLtGroupedMatmulCallTarget) &&
         user->operand_count() > 0 && user->operand(0) == producer;
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
      if (std::optional<int64_t> operand_index =
              GetValidPrefixExtentOperandIndex(fusion)) {
        candidates.emplace(fusion, *operand_index);
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
      for (HloInstruction* user : fusion->users()) {
        auto user_candidate = candidates.find(user);
        const bool same_extent_fusion =
            user_candidate != candidates.end() &&
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
      auto candidate = candidates.find(fusion);
      if (candidate == candidates.end()) {
        continue;
      }
      ABSL_RETURN_IF_ERROR(SetValidPrefixExtent(fusion, candidate->second));
      changed = true;
    }
  }

  // Markers that were not absorbed into a fusion are compile-time identities.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (!instruction->IsCustomCall("xla.valid_prefix")) {
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
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
