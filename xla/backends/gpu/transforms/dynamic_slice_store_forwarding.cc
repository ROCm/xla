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

#include "xla/backends/gpu/transforms/dynamic_slice_store_forwarding.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

std::optional<DynamicSliceConfig> GetDynamicSliceConfig(
    const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  if (!config.ok() || !config->has_dynamic_slice_config()) {
    return std::nullopt;
  }
  return config->dynamic_slice_config();
}

// As DynamicSliceFusionRewriterV2: a contiguous slice whose size keeps the
// buffer alignment.
bool IsAlignedUpdate(const HloInstruction* dus) {
  if (!IsContiguousSlice(*dus)) {
    return false;
  }
  const Shape& update = dus->operand(1)->shape();
  return ShapeUtil::ByteStrides(update).has_value() &&
         ShapeUtil::ByteSizeOfElements(update) %
                 kXlaAllocatedBufferAlignBytes ==
             0;
}

// The dynamic-update-slice's result is the computation's result or one of the
// root tuple's elements (e.g. a stacked buffer in a while loop's state).
bool FeedsRoot(const HloInstruction* dus) {
  const HloInstruction* root = dus->parent()->root_instruction();
  return root == dus || (root->opcode() == HloOpcode::kTuple &&
                         absl::c_linear_search(root->operands(), dus));
}

// Walks from `instr` up through bitcasts that have no other user. Fills
// `chain` with the bitcasts (nearest to `instr` last) and returns the first
// instruction that is not such a bitcast.
HloInstruction* SkipSingleUseBitcasts(HloInstruction* instr,
                                      std::vector<HloInstruction*>* chain) {
  while (instr->opcode() == HloOpcode::kBitcast && instr->user_count() == 1) {
    chain->push_back(instr);
    instr = instr->mutable_operand(0);
  }
  absl::c_reverse(*chain);
  return instr;
}

// True if `target` is among the transitive operands of `instr`.
bool IsAncestor(const HloInstruction* target, const HloInstruction* instr) {
  std::vector<const HloInstruction*> stack(instr->operands().begin(),
                                           instr->operands().end());
  absl::flat_hash_set<const HloInstruction*> seen;
  while (!stack.empty()) {
    const HloInstruction* cur = stack.back();
    stack.pop_back();
    if (cur == target) {
      return true;
    }
    if (!seen.insert(cur).second) {
      continue;
    }
    for (const HloInstruction* operand : cur->operands()) {
      stack.push_back(operand);
    }
  }
  return false;
}

// A user that can take dynamic-slice(stack', i) in place of the stored value
// without a copy: a hero (dynamic slice fusion slices its operand), or an
// instruction that fusion can absorb the dynamic-slice into.
bool CanReadSlice(const HloInstruction* user,
                  const DynamicSliceStoreForwarding::HeroPredicate& is_hero) {
  if (is_hero(user)) {
    return true;
  }
  switch (user->opcode()) {
    case HloOpcode::kCustomCall:
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
    case HloOpcode::kCall:
    case HloOpcode::kFusion:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kDynamicUpdateSlice:
      return false;
    default:
      break;
  }
  if (hlo_query::IsCollectiveCommunicationOp(user->opcode()) ||
      user->HasSideEffect()) {
    return false;
  }
  return user->IsFusible();
}

// `user` is a hero, or bitcasts that only reach heroes (which dynamic slice
// fusion looks through to a slice).
bool ReachesOnlyHeroes(
    const HloInstruction* user,
    const DynamicSliceStoreForwarding::HeroPredicate& is_hero) {
  if (is_hero(user)) {
    return true;
  }
  return user->opcode() == HloOpcode::kBitcast && user->user_count() > 0 &&
         absl::c_all_of(user->users(), [&](const HloInstruction* next) {
           return ReachesOnlyHeroes(next, is_hero);
         });
}

// A producer that fusion can put inside the dynamic-update-slice once it has a
// single user.
bool IsFusibleProducer(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kCustomCall:
    case HloOpcode::kParameter:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kConstant:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
    case HloOpcode::kCall:
    case HloOpcode::kFusion:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCopyDone:
    case HloOpcode::kRecvDone:
    case HloOpcode::kBitcast:
      return false;
    default:
      break;
  }
  return !hlo_query::IsCollectiveCommunicationOp(instr->opcode()) &&
         !instr->HasSideEffect() && instr->IsFusible() &&
         !instr->shape().IsTuple();
}

class StoreForwarder {
 public:
  StoreForwarder(HloComputation* computation,
                 const DynamicSliceStoreForwarding::HeroPredicate& is_hero)
      : computation_(computation), is_hero_(is_hero) {}

  absl::StatusOr<bool> Run() {
    std::vector<HloInstruction*> stores;
    for (HloInstruction* instr : computation_->instructions()) {
      if (instr->opcode() == HloOpcode::kDynamicUpdateSlice &&
          GetDynamicSliceConfig(instr).has_value() && IsAlignedUpdate(instr) &&
          FeedsRoot(instr)) {
        stores.push_back(instr);
      }
    }
    bool changed = false;
    for (HloInstruction* dus : stores) {
      ABSL_ASSIGN_OR_RETURN(bool forwarded, Forward(dus));
      changed |= forwarded;
    }
    return changed;
  }

 private:
  // dynamic-slice(stack, offsets of `dus`) with the shape of `dus`'s update,
  // bitcast to `shape`, carrying `dus`'s slice config and `like`'s metadata.
  absl::StatusOr<HloInstruction*> ReadBack(HloInstruction* stack,
                                           const HloInstruction* dus,
                                           const Shape& shape,
                                           const HloInstruction* like) {
    const Shape& update = dus->operand(1)->shape();
    std::vector<HloInstruction*> offsets(dus->operands().begin() + 2,
                                         dus->operands().end());
    HloInstruction* slice =
        computation_->AddInstruction(HloInstruction::CreateDynamicSlice(
            update, stack, offsets, update.dimensions()));
    GpuBackendConfig config;
    *config.mutable_dynamic_slice_config() = *GetDynamicSliceConfig(dus);
    ABSL_RETURN_IF_ERROR(slice->set_backend_config(config));
    slice->set_metadata(like->metadata());
    return Reshape(slice, shape);
  }

  HloInstruction* Reshape(HloInstruction* instr, const Shape& shape) {
    if (ShapeUtil::Equal(instr->shape(), shape)) {
      return instr;
    }
    HloInstruction* bitcast = computation_->AddInstruction(
        HloInstruction::CreateBitcast(shape, instr));
    bitcast->set_metadata(instr->metadata());
    return bitcast;
  }

  // Redirects the uses of `value` by `users` to `replacement`.
  static absl::Status Redirect(HloInstruction* value,
                               absl::Span<HloInstruction* const> users,
                               HloInstruction* replacement) {
    for (HloInstruction* user : users) {
      for (int64_t i = 0; i < user->operand_count(); ++i) {
        if (user->operand(i) == value) {
          ABSL_RETURN_IF_ERROR(user->ReplaceOperandWith(i, replacement));
        }
      }
    }
    return absl::OkStatus();
  }

  bool CanRedirect(absl::Span<HloInstruction* const> users,
                   const HloInstruction* dus, bool heroes_only) const {
    return absl::c_all_of(users, [&](const HloInstruction* user) {
      return (heroes_only ? ReachesOnlyHeroes(user, is_hero_)
                          : CanReadSlice(user, is_hero_)) &&
             !IsAncestor(user, dus);
    });
  }

  absl::StatusOr<bool> Forward(HloInstruction* dus) {
    std::vector<HloInstruction*> chain;
    HloInstruction* value =
        SkipSingleUseBitcasts(dus->mutable_operand(1), &chain);
    const HloInstruction* chain_head = chain.empty() ? dus : chain.front();
    if (value->shape().IsTuple() || absl::c_count(dus->operands(), value) > 1) {
      return false;
    }
    std::vector<HloInstruction*> others;
    for (HloInstruction* user : value->users()) {
      if (user != chain_head) {
        others.push_back(user);
      }
    }

    if (!is_hero_(value)) {
      // A fusible producer read by heroes as well: the heroes read the stored
      // slice, the producer fuses into the store.
      if (others.empty() || !IsFusibleProducer(value) ||
          !CanRedirect(others, dus, /*heroes_only=*/true)) {
        return false;
      }
      ABSL_ASSIGN_OR_RETURN(HloInstruction * read,
                            ReadBack(dus, dus, value->shape(), value));
      ABSL_RETURN_IF_ERROR(Redirect(value, others, read));
      VLOG(2) << "Forwarded store " << dus->name() << " of producer "
              << value->name() << " to " << others.size() << " hero(es)";
      return true;
    }

    auto* callable = DynCast<HloCallableInstruction>(value);
    const bool in_place =
        callable != nullptr && !callable->output_to_operand_aliasing().empty();
    if (!in_place) {
      if (others.empty() || !CanRedirect(others, dus, /*heroes_only=*/false)) {
        return false;
      }
      ABSL_ASSIGN_OR_RETURN(HloInstruction * read,
                            ReadBack(dus, dus, value->shape(), value));
      ABSL_RETURN_IF_ERROR(Redirect(value, others, read));
      VLOG(2) << "Forwarded store " << dus->name() << " of hero "
              << value->name() << " to " << others.size() << " user(s)";
      return true;
    }

    // An in-place hero: its result is its aliased operand, updated.
    const auto& aliasing = callable->output_to_operand_aliasing();
    if (aliasing.size() != 1 || !aliasing[0].first.empty() ||
        !aliasing[0].second.second.empty()) {
      return false;
    }
    const int64_t aliased = aliasing[0].second.first;
    HloInstruction* operand = value->mutable_operand(aliased);
    std::vector<HloInstruction*> operand_chain;
    HloInstruction* source = SkipSingleUseBitcasts(operand, &operand_chain);
    // The operand is the only use of another hero's result, so that hero can
    // write it straight into the stored slice; and the in-place hero reads
    // nothing else from it or from the stack.
    if (source->user_count() != 1 || !is_hero_(source) ||
        source->shape().IsTuple() || operand->user_count() != 1) {
      return false;
    }
    for (int64_t i = 0; i < value->operand_count(); ++i) {
      const HloInstruction* stripped = value->operand(i);
      while (stripped->opcode() == HloOpcode::kBitcast) {
        stripped = stripped->operand(0);
      }
      if (i != aliased && (stripped == source || stripped == dus->operand(0))) {
        return false;
      }
    }
    if (!others.empty() && !CanRedirect(others, dus, /*heroes_only=*/false)) {
      return false;
    }

    const Shape& update = dus->operand(1)->shape();
    std::vector<HloInstruction*> offsets(dus->operands().begin() + 2,
                                         dus->operands().end());
    auto store = [&](HloInstruction* stack, HloInstruction* stored) {
      std::vector<HloInstruction*> operands = {stack, Reshape(stored, update)};
      operands.insert(operands.end(), offsets.begin(), offsets.end());
      return computation_->AddInstruction(
          dus->CloneWithNewOperands(dus->shape(), operands));
    };
    // stack1 = dus(stack, operand); value' = hero(ds(stack1)); stack' =
    // dus(stack1, value').
    HloInstruction* stack1 = store(dus->mutable_operand(0), operand);
    ABSL_ASSIGN_OR_RETURN(HloInstruction * slice,
                          ReadBack(stack1, dus, operand->shape(), operand));
    std::vector<HloInstruction*> hero_operands(value->operands().begin(),
                                               value->operands().end());
    hero_operands[aliased] = slice;
    HloInstruction* new_value = computation_->AddInstruction(
        value->CloneWithNewOperands(value->shape(), hero_operands));
    ABSL_RETURN_IF_ERROR(new_value->CopyAllControlDepsFrom(value));
    HloInstruction* stack2 = store(stack1, new_value);
    if (!others.empty()) {
      ABSL_ASSIGN_OR_RETURN(HloInstruction * read,
                            ReadBack(stack2, dus, value->shape(), value));
      ABSL_RETURN_IF_ERROR(Redirect(value, others, read));
    }
    ABSL_RETURN_IF_ERROR(value->DropAllControlDeps());
    VLOG(2) << "Forwarded store " << dus->name() << " of in-place hero "
            << value->name() << " on " << source->name() << " ("
            << others.size() << " other user(s))";
    // The old store, its bitcasts and the old hero are dead now. Removing the
    // store removes them too, unless the hero has a side effect.
    const bool value_has_side_effect = value->HasSideEffect();
    ABSL_RETURN_IF_ERROR(computation_->ReplaceInstruction(dus, stack2));
    if (value_has_side_effect) {
      ABSL_RETURN_IF_ERROR(
          computation_->RemoveInstructionAndUnusedOperands(value));
    }
    return true;
  }

  HloComputation* computation_;
  const DynamicSliceStoreForwarding::HeroPredicate& is_hero_;
};

}  // namespace

absl::StatusOr<bool> DynamicSliceStoreForwarding::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ABSL_ASSIGN_OR_RETURN(bool computation_changed,
                          StoreForwarder(computation, is_hero_).Run());
    changed |= computation_changed;
  }
  return changed;
}

}  // namespace xla::gpu
