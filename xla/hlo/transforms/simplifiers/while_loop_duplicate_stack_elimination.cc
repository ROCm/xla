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

#include "xla/hlo/transforms/simplifiers/while_loop_duplicate_stack_elimination.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

// Every index is evaluated for every iteration, so longer loops are skipped.
constexpr int64_t kMaxTripCount = 4096;

// Skips copies and same-type converts, which pass the value through.
const HloInstruction* SkipCopies(const HloInstruction* h) {
  while (h->opcode() == HloOpcode::kCopy ||
         (h->opcode() == HloOpcode::kConvert &&
          h->shape().element_type() == h->operand(0)->shape().element_type())) {
    h = h->operand(0);
  }
  return h;
}

// Also skips reshapes: two chains of copies and reshapes from the same value
// to the same shape yield the same value (unlike bitcasts, which can change
// the layout and are not skipped).
const HloInstruction* SkipCopiesAndReshapes(const HloInstruction* h) {
  for (h = SkipCopies(h); h->opcode() == HloOpcode::kReshape;
       h = SkipCopies(h->operand(0))) {
  }
  return h;
}

bool IsGteOf(const HloInstruction* h, const HloInstruction* tuple,
             int64_t index) {
  return h->opcode() == HloOpcode::kGetTupleElement && h->operand(0) == tuple &&
         h->tuple_index() == index;
}

// A loop element that every iteration overwrites with a dynamic-update-slice
// of `update`, a slice of the buffer along at most one dimension.
struct StackWrite {
  int64_t index;
  const HloInstruction* dus;
  const HloInstruction* update;  // Copies and reshapes skipped.
};

std::optional<StackWrite> MatchStackWrite(const HloInstruction* while_op,
                                          int64_t index) {
  const HloComputation* body = while_op->while_body();
  const HloInstruction* dus =
      SkipCopies(body->root_instruction()->operand(index));
  if (dus->opcode() != HloOpcode::kDynamicUpdateSlice ||
      !IsGteOf(SkipCopies(dus->operand(0)), body->parameter_instruction(0),
               index)) {
    return std::nullopt;
  }
  return StackWrite{index, dus, SkipCopiesAndReshapes(dus->operand(1))};
}

bool SameWriteShapes(const StackWrite& a, const StackWrite& b) {
  return a.update == b.update &&
         ShapeUtil::Equal(a.dus->shape(), b.dus->shape()) &&
         ShapeUtil::Equal(a.dus->operand(1)->shape(),
                          b.dus->operand(1)->shape());
}

// The start offsets, clamped as the dynamic-update-slice clamps them, along
// the dimension the writes advance in, for every iteration, if the writes
// together cover the buffer so that no initial element reaches the output.
std::optional<std::vector<int64_t>> CoveringStarts(
    const HloInstruction* while_op, const StackWrite& write) {
  const Shape& shape = write.dus->shape();
  const Shape& update_shape = write.dus->operand(1)->shape();
  std::vector<int64_t> partial_dims;
  for (int64_t d = 0; d < shape.dimensions().size(); ++d) {
    if (update_shape.dimensions(d) < shape.dimensions(d)) {
      partial_dims.push_back(d);
    }
  }
  // Tiling more than one dimension is not handled. If every write covers the
  // buffer, dimension 0 is still evaluated, which checks that the loop runs.
  if (partial_dims.size() > 1 || shape.dimensions().empty()) {
    return std::nullopt;
  }
  const int64_t dim = partial_dims.empty() ? 0 : partial_dims[0];
  const int64_t size = update_shape.dimensions(dim);
  std::optional<std::vector<int64_t>> starts = EvaluateScalarPerIteration(
      while_op, write.dus->operand(2 + dim), kMaxTripCount);
  if (!starts.has_value()) {
    return std::nullopt;
  }
  for (int64_t& start : *starts) {
    start = std::clamp<int64_t>(start, 0, shape.dimensions(dim) - size);
  }
  std::vector<int64_t> sorted = *starts;
  absl::c_sort(sorted);
  int64_t covered = 0;
  for (int64_t start : sorted) {
    if (start > covered) {
      return std::nullopt;
    }
    covered = start + size;
  }
  return covered == shape.dimensions(dim) ? starts : std::nullopt;
}

// An initial value that only exists to be overwritten (a scan's stacked
// output); an element with one is the one to drop.
bool IsFreshInit(const HloInstruction* init) {
  init = SkipCopiesAndReshapes(init);
  return init->opcode() == HloOpcode::kBroadcast ||
         init->opcode() == HloOpcode::kConstant ||
         init->opcode() == HloOpcode::kIota ||
         init->IsCustomCall("AllocateBuffer");
}

absl::StatusOr<bool> RunOnLoop(HloInstruction* while_op) {
  HloComputation* parent = while_op->parent();
  if (parent->root_instruction() == while_op ||
      while_op->while_body()->root_instruction()->opcode() !=
          HloOpcode::kTuple ||
      !absl::c_all_of(while_op->users(), [](const HloInstruction* user) {
        return user->opcode() == HloOpcode::kGetTupleElement;
      })) {
    return false;
  }
  std::vector<StackWrite> writes;
  for (int64_t i = 0; i < while_op->shape().tuple_shapes().size(); ++i) {
    if (std::optional<StackWrite> write = MatchStackWrite(while_op, i)) {
      writes.push_back(*write);
    }
  }
  // Pairs with the same update and shapes, before anything is evaluated.
  std::vector<std::pair<int64_t, int64_t>> pairs;
  for (int64_t a = 0; a < writes.size(); ++a) {
    for (int64_t b = a + 1; b < writes.size(); ++b) {
      if (SameWriteShapes(writes[a], writes[b])) {
        pairs.push_back({a, b});
      }
    }
  }
  if (pairs.empty()) {
    return false;
  }
  // Evaluated once per element, and only for elements in a pair.
  absl::flat_hash_map<int64_t, std::optional<std::vector<int64_t>>> starts;
  for (auto [a, b] : pairs) {
    for (int64_t w : {a, b}) {
      if (!starts.contains(w)) {
        starts[w] = CoveringStarts(while_op, writes[w]);
      }
    }
  }

  // Each duplicate element is redirected to the first equal element kept.
  const HloInstruction* init = while_op->operand(0);
  std::vector<bool> dropped(writes.size(), false);
  bool changed = false;
  for (auto [a, b] : pairs) {
    if (dropped[a] || dropped[b] || !starts[a].has_value() ||
        starts[a] != starts[b]) {
      continue;
    }
    if (IsFreshInit(init->operand(writes[a].index)) &&
        !IsFreshInit(init->operand(writes[b].index))) {
      std::swap(a, b);
    }
    dropped[b] = true;
    const int64_t keep = writes[a].index;
    const int64_t drop = writes[b].index;
    std::vector<HloInstruction*> drop_gtes;
    HloInstruction* keep_gte = nullptr;
    for (HloInstruction* user : while_op->users()) {
      if (user->tuple_index() == drop) {
        drop_gtes.push_back(user);
      } else if (user->tuple_index() == keep && keep_gte == nullptr) {
        keep_gte = user;
      }
    }
    if (drop_gtes.empty()) {
      continue;
    }
    if (keep_gte == nullptr) {
      keep_gte = parent->AddInstruction(
          HloInstruction::CreateGetTupleElement(while_op, keep));
    }
    VLOG(2) << "While loop " << while_op->name() << ": output " << drop
            << " duplicates output " << keep;
    for (HloInstruction* gte : drop_gtes) {
      ABSL_RETURN_IF_ERROR(gte->ReplaceAllUsesWith(keep_gte));
      // WhileLoopSimplifier counts any get-tuple-element as a use.
      if (gte->IsDead() && !gte->HasControlDependencies()) {
        ABSL_RETURN_IF_ERROR(parent->RemoveInstruction(gte));
      }
    }
    changed = true;
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> WhileLoopDuplicateStackElimination::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> while_ops;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        while_ops.push_back(instruction);
      }
    }
  }
  bool changed = false;
  for (HloInstruction* while_op : while_ops) {
    ABSL_ASSIGN_OR_RETURN(bool loop_changed, RunOnLoop(while_op));
    changed |= loop_changed;
  }
  return changed;
}

}  // namespace xla
