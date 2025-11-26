/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/nan_check_inserter.h"

#include <vector>

#include "absl/log/log.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/runtime_intrinsics.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

void NanCheckInserter::RunOnComputation(HloModule* module, HloComputation* comp,
                        WorkList& to_visit) {
  auto print_options = HloPrintOptions::ShortParsable()
                           .set_print_operand_shape(true)
                           .set_print_extra_attributes(true);

  auto is_async_start = [](const HloInstruction* inst) {
    return (HloPredicateIsOp<HloOpcode::kAllReduceStart, 
        HloOpcode::kAllGatherStart, HloOpcode::kCollectivePermuteStart,
        HloOpcode::kAsyncStart, HloOpcode::kCopyStart>(inst));
  };
  auto is_async_done = [](const HloInstruction* inst) {
    return (HloPredicateIsOp<HloOpcode::kAllReduceDone, 
        HloOpcode::kAllGatherDone, HloOpcode::kCollectivePermuteDone,
        HloOpcode::kAsyncDone, HloOpcode::kCopyDone>(inst));
  };

  for (HloInstruction* instruction : comp->instructions()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      VLOG(1) << "Found while loop: processing!";
      to_visit.push(instruction->while_condition());
      to_visit.push(instruction->while_body());
    } else if (instruction->opcode() == HloOpcode::kConditional) {
      VLOG(1) << "Found conditional: processing!";
      for (auto* branch_comp : instruction->branch_computations()) {
        to_visit.push(branch_comp);
      }
    }

    auto shape = instruction->shape();
    // TODO(rocm): Check if bitcast can be a copy, then it is not noop
    if (HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant,
                         HloOpcode::kBitcast>(instruction) ||
        is_async_start(instruction) || !shape.IsArray() ||
        !primitive_util::IsFloatingPointType(shape.element_type()))
      continue;

    int32_t index = 0;
    const auto* source = instruction;
    if (source->opcode() == HloOpcode::kGetTupleElement) {
      index = source->tuple_index();
      // now source points to an instruction returning a tuple
      source = source->operand(0); 
    }
    if (source->opcode() == HloOpcode::kWhile ||
        source->opcode() == HloOpcode::kConditional) {
      continue;
    }
    absl::InlinedVector<HloInstruction*, 8> args;
    args.push_back(instruction);

    if (is_async_done(source)) { 
      source = source->operand(0);
      CHECK(is_async_start(source));
    }
    for (auto op : source->operands()) {
      // op->shape().IsArray()
      if (!op->IsConstant()) args.push_back(op);
    }

    auto created =
        Cast<HloCustomCallInstruction>(instruction->parent()->AddInstruction(
            HloInstruction::CreateCustomCall(
                ShapeUtil::MakeTokenShape(), args,
                kXlaGpuNanCheckCustomCallTag, "", API_VERSION_ORIGINAL)));
    created->set_custom_call_has_side_effect(true);
  }  // for
}

absl::StatusOr<bool> NanCheckInserter::Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&) {
  WorkList to_visit;
  if (auto* comp = module->entry_computation()) {
    to_visit.push(comp);
  }

  while (!to_visit.empty()) {
    auto* comp = to_visit.top();
    to_visit.pop();
    RunOnComputation(module, comp, to_visit);
  }
  return true;
}

}  // namespace gpu
}  // namespace xla
