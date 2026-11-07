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

#include "xla/backends/gpu/transforms/while_loop_duplicate_stack_elimination.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/value_range.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

// Loops with more iterations are skipped: every index is evaluated for every
// iteration.
constexpr int64_t kMaxTripCount = 1 << 16;
// Bounds the recursion of the index evaluator.
constexpr int kMaxEvalDepth = 64;

// Skips copies and same-type converts, which pass the value through.
const HloInstruction* SkipCopies(const HloInstruction* h) {
  while (h->opcode() == HloOpcode::kCopy ||
         (h->opcode() == HloOpcode::kConvert &&
          h->shape().element_type() ==
              h->operand(0)->shape().element_type())) {
    h = h->operand(0);
  }
  return h;
}

// Also skips reshapes. Two chains of copies and reshapes from the same value to
// the same shape yield the same value, since a reshape is a function of the
// logical value only (unlike a bitcast, which is not skipped).
const HloInstruction* SkipCopiesAndReshapes(const HloInstruction* h) {
  while (true) {
    h = SkipCopies(h);
    if (h->opcode() != HloOpcode::kReshape) {
      return h;
    }
    h = h->operand(0);
  }
}

bool IsGteOf(const HloInstruction* h, const HloInstruction* tuple,
             int64_t index) {
  return h->opcode() == HloOpcode::kGetTupleElement &&
         h->operand(0) == tuple && h->tuple_index() == index;
}

// The loop's facts the pass needs: the induction variable's tuple index, its
// value in the first iteration and its step, and the trip count.
struct Loop {
  HloInstruction* while_op;
  int64_t iv_index;
  int64_t iv_init;
  int64_t iv_step;
  int64_t trip_count;
};

std::optional<Loop> MatchLoop(HloInstruction* while_op) {
  if (while_op->operand(0)->opcode() != HloOpcode::kTuple ||
      while_op->while_body()->root_instruction()->opcode() !=
          HloOpcode::kTuple) {
    return std::nullopt;
  }
  std::optional<int64_t> iv_index = GetLoopInductionVarTupleIdx(while_op);
  if (!iv_index.has_value()) {
    return std::nullopt;
  }
  std::optional<Range> range = MatchTrivialLoopRange(while_op);
  if (!range.has_value() || range->IsEmpty() || !range->IsBounded() ||
      !range->IsStepKnown()) {
    return std::nullopt;
  }
  const int64_t init = range->min().GetSignedValue();
  const int64_t last = range->max()->GetSignedValue();
  const int64_t step = range->step()->GetSignedValue();
  if (step <= 0 || last < init) {
    return std::nullopt;
  }
  const int64_t trip_count = (last - init) / step + 1;
  if (trip_count <= 0 || trip_count > kMaxTripCount) {
    return std::nullopt;
  }
  return Loop{while_op, *iv_index, init, step, trip_count};
}

// Evaluates single-element integer expressions in the loop body for a given
// value of the induction variable. Returns nullopt for anything it does not
// understand, which makes the caller leave the loop alone.
class IndexEvaluator {
 public:
  explicit IndexEvaluator(const Loop& loop)
      : loop_(loop),
        body_param_(loop.while_op->while_body()->parameter_instruction(0)),
        body_root_(loop.while_op->while_body()->root_instruction()) {}

  std::optional<int64_t> Eval(const HloInstruction* h, int64_t iv,
                              int depth = 0) const {
    if (depth > kMaxEvalDepth || !h->shape().IsArray() ||
        ShapeUtil::ElementsIn(h->shape()) != 1) {
      return std::nullopt;
    }
    const PrimitiveType type = h->shape().element_type();
    if (!primitive_util::IsIntegralType(type) && type != PRED) {
      return std::nullopt;
    }
    std::optional<int64_t> v = EvalUnchecked(h, iv, depth);
    if (!v.has_value()) {
      return std::nullopt;
    }
    // Values that would wrap around in their type are not modeled.
    if (type == PRED) {
      return (*v == 0 || *v == 1) ? v : std::nullopt;
    }
    const int bits = primitive_util::BitWidth(type);
    if (bits >= 64) {
      return v;
    }
    const bool is_signed = primitive_util::IsSignedIntegralType(type);
    const int64_t lo = is_signed ? -(int64_t{1} << (bits - 1)) : 0;
    const int64_t hi = is_signed ? (int64_t{1} << (bits - 1)) - 1
                                 : (int64_t{1} << bits) - 1;
    if (*v < lo || *v > hi) {
      return std::nullopt;
    }
    return v;
  }

 private:
  std::optional<int64_t> EvalUnchecked(const HloInstruction* h, int64_t iv,
                                       int depth) const {
    auto operand = [&](int64_t i) {
      return Eval(h->operand(i), iv, depth + 1);
    };
    switch (h->opcode()) {
      case HloOpcode::kConstant:
        return h->literal().GetFirstInteger();
      case HloOpcode::kGetTupleElement:
        if (IsGteOf(h, body_param_, loop_.iv_index)) {
          return iv;
        }
        return std::nullopt;
      case HloOpcode::kCopy:
      case HloOpcode::kReshape:
      case HloOpcode::kBitcast:
      case HloOpcode::kConvert:
        // One element: the value is unchanged (converts are range-checked).
        return operand(0);
      case HloOpcode::kNegate: {
        std::optional<int64_t> a = operand(0);
        if (!a.has_value()) {
          return std::nullopt;
        }
        return -*a;
      }
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kDivide:
      case HloOpcode::kRemainder: {
        std::optional<int64_t> a = operand(0);
        std::optional<int64_t> b = operand(1);
        if (!a.has_value() || !b.has_value()) {
          return std::nullopt;
        }
        constexpr int64_t kBound = int64_t{1} << 31;
        if (std::abs(*a) > kBound || std::abs(*b) > kBound) {
          return std::nullopt;
        }
        switch (h->opcode()) {
          case HloOpcode::kAdd:
            return *a + *b;
          case HloOpcode::kSubtract:
            return *a - *b;
          case HloOpcode::kMultiply:
            return *a * *b;
          case HloOpcode::kMaximum:
            return std::max(*a, *b);
          case HloOpcode::kMinimum:
            return std::min(*a, *b);
          case HloOpcode::kDivide:
            // Truncating division, as XLA's for integers.
            return *b == 0 ? std::nullopt : std::optional<int64_t>(*a / *b);
          default:
            return *b == 0 ? std::nullopt : std::optional<int64_t>(*a % *b);
        }
      }
      case HloOpcode::kCompare: {
        // Unsigned operands are non-negative after the range check, so the
        // signed order is theirs too.
        std::optional<int64_t> a = operand(0);
        std::optional<int64_t> b = operand(1);
        if (!a.has_value() || !b.has_value()) {
          return std::nullopt;
        }
        switch (h->comparison_direction()) {
          case ComparisonDirection::kEq:
            return *a == *b;
          case ComparisonDirection::kNe:
            return *a != *b;
          case ComparisonDirection::kLt:
            return *a < *b;
          case ComparisonDirection::kLe:
            return *a <= *b;
          case ComparisonDirection::kGt:
            return *a > *b;
          case ComparisonDirection::kGe:
            return *a >= *b;
        }
        return std::nullopt;
      }
      case HloOpcode::kSelect: {
        std::optional<int64_t> p = operand(0);
        if (!p.has_value()) {
          return std::nullopt;
        }
        return operand(*p != 0 ? 1 : 2);
      }
      case HloOpcode::kClamp: {
        std::optional<int64_t> lo = operand(0);
        std::optional<int64_t> x = operand(1);
        std::optional<int64_t> hi = operand(2);
        if (!lo.has_value() || !x.has_value() || !hi.has_value()) {
          return std::nullopt;
        }
        return std::min(std::max(*x, *lo), *hi);
      }
      case HloOpcode::kDynamicSlice: {
        // One element of a 1-D loop-invariant array.
        const HloInstruction* array = h->operand(0);
        if (array->shape().dimensions().size() != 1 ||
            h->operand_count() != 2) {
          return std::nullopt;
        }
        std::optional<int64_t> start = operand(1);
        if (!start.has_value()) {
          return std::nullopt;
        }
        const int64_t n = array->shape().dimensions(0);
        return InvariantElement(array, std::clamp<int64_t>(*start, 0, n - 1),
                                depth + 1);
      }
      case HloOpcode::kSlice: {
        const HloInstruction* array = h->operand(0);
        if (array->shape().dimensions().size() != 1) {
          return std::nullopt;
        }
        return InvariantElement(array, h->slice_starts(0), depth + 1);
      }
      default:
        return std::nullopt;
    }
  }

  // Element `i` of a 1-D integer array that has the same value in every
  // iteration: a constant, an iota, a broadcast of a constant, or a loop
  // element passed through the body unchanged whose initial value is one of
  // those.
  std::optional<int64_t> InvariantElement(const HloInstruction* array,
                                          int64_t i, int depth) const {
    if (depth > kMaxEvalDepth) {
      return std::nullopt;
    }
    array = SkipCopies(array);
    if (array->shape().dimensions().size() != 1 || i < 0 ||
        i >= array->shape().dimensions(0) ||
        !primitive_util::IsIntegralType(array->shape().element_type())) {
      return std::nullopt;
    }
    switch (array->opcode()) {
      case HloOpcode::kConstant:
        return array->literal().GetIntegralAsS64({i});
      case HloOpcode::kIota:
        return i;
      case HloOpcode::kBroadcast:
        if (array->operand(0)->opcode() == HloOpcode::kConstant &&
            ShapeUtil::IsScalar(array->operand(0)->shape())) {
          return array->operand(0)->literal().GetFirstInteger();
        }
        return std::nullopt;
      case HloOpcode::kGetTupleElement: {
        if (array->operand(0) != body_param_) {
          return std::nullopt;
        }
        const int64_t index = array->tuple_index();
        if (index == loop_.iv_index ||
            !IsGteOf(SkipCopies(body_root_->operand(index)), body_param_,
                     index)) {
          return std::nullopt;
        }
        // The initial value is in the loop's computation; this recursion
        // stops there, since its get-tuple-elements are not of body_param_.
        return InvariantElement(loop_.while_op->operand(0)->operand(index), i,
                                depth + 1);
      }
      default:
        return std::nullopt;
    }
  }

  const Loop& loop_;
  const HloInstruction* body_param_;
  const HloInstruction* body_root_;
};

// A loop element that every iteration overwrites with a dynamic-update-slice
// of `update` at `starts[iteration]` (clamped as the dynamic-update-slice
// clamps them), and whose writes cover the whole buffer.
struct StackWrite {
  int64_t index;
  const HloInstruction* dus;
  const HloInstruction* update;  // Copies and reshapes skipped.
  std::vector<std::vector<int64_t>> starts;
};

std::optional<StackWrite> MatchStackWrite(const Loop& loop,
                                          const IndexEvaluator& eval,
                                          int64_t index) {
  HloComputation* body = loop.while_op->while_body();
  const HloInstruction* param = body->parameter_instruction(0);
  const HloInstruction* dus =
      SkipCopies(body->root_instruction()->operand(index));
  if (dus->opcode() != HloOpcode::kDynamicUpdateSlice ||
      !IsGteOf(SkipCopies(dus->operand(0)), param, index)) {
    return std::nullopt;
  }
  const Shape& shape = dus->shape();
  const Shape& update_shape = dus->operand(1)->shape();
  const int64_t rank = shape.dimensions().size();
  if (rank == 0 || update_shape.dimensions().size() != rank) {
    return std::nullopt;
  }
  // The single dimension the writes advance along; the update spans the
  // buffer in every other one.
  std::optional<int64_t> dim;
  for (int64_t d = 0; d < rank; ++d) {
    if (update_shape.dimensions(d) == shape.dimensions(d)) {
      continue;
    }
    if (dim.has_value()) {
      return std::nullopt;
    }
    dim = d;
  }
  if (!dim.has_value()) {
    return std::nullopt;
  }
  StackWrite write{index, dus, SkipCopiesAndReshapes(dus->operand(1)), {}};
  write.starts.reserve(loop.trip_count);
  std::vector<std::pair<int64_t, int64_t>> intervals;
  intervals.reserve(loop.trip_count);
  for (int64_t t = 0; t < loop.trip_count; ++t) {
    const int64_t iv = loop.iv_init + t * loop.iv_step;
    std::vector<int64_t> starts(rank);
    for (int64_t d = 0; d < rank; ++d) {
      std::optional<int64_t> start = eval.Eval(dus->operand(2 + d), iv);
      if (!start.has_value()) {
        return std::nullopt;
      }
      starts[d] = std::clamp<int64_t>(
          *start, 0, shape.dimensions(d) - update_shape.dimensions(d));
    }
    intervals.push_back(
        {starts[*dim], starts[*dim] + update_shape.dimensions(*dim)});
    write.starts.push_back(std::move(starts));
  }
  // The iterations' slices must cover the dimension, so that no initial
  // element survives the loop.
  std::sort(intervals.begin(), intervals.end());
  int64_t covered = 0;
  for (const auto& [lo, hi] : intervals) {
    if (lo > covered) {
      return std::nullopt;
    }
    covered = std::max(covered, hi);
  }
  if (covered < shape.dimensions(*dim)) {
    return std::nullopt;
  }
  return write;
}

bool SameWrites(const StackWrite& a, const StackWrite& b) {
  return a.update == b.update &&
         ShapeUtil::Equal(a.dus->shape(), b.dus->shape()) &&
         ShapeUtil::Equal(a.dus->operand(1)->shape(),
                          b.dus->operand(1)->shape()) &&
         a.starts == b.starts;
}

// An initial value that only exists to be overwritten (a scan's stacked
// output); the element that has one is the one to drop.
bool IsFreshInit(const HloInstruction* init) {
  init = SkipCopiesAndReshapes(init);
  return init->opcode() == HloOpcode::kBroadcast ||
         init->opcode() == HloOpcode::kConstant ||
         init->opcode() == HloOpcode::kIota ||
         init->IsCustomCall("AllocateBuffer");
}

absl::StatusOr<bool> RunOnLoop(HloInstruction* while_op) {
  HloComputation* parent = while_op->parent();
  if (parent->root_instruction() == while_op) {
    return false;
  }
  for (const HloInstruction* user : while_op->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
  }
  std::optional<Loop> loop = MatchLoop(while_op);
  if (!loop.has_value()) {
    return false;
  }
  IndexEvaluator eval(*loop);
  const HloInstruction* init = while_op->operand(0);
  std::vector<StackWrite> writes;
  for (int64_t i = 0; i < init->operand_count(); ++i) {
    if (i == loop->iv_index || !init->operand(i)->shape().IsArray()) {
      continue;
    }
    if (std::optional<StackWrite> w = MatchStackWrite(*loop, eval, i)) {
      writes.push_back(*std::move(w));
    }
  }

  // Each duplicate element is redirected to the first equal element kept.
  std::vector<std::pair<int64_t, int64_t>> redirects;  // {dropped, kept}
  std::vector<bool> dropped(writes.size(), false);
  for (int64_t a = 0; a < writes.size(); ++a) {
    if (dropped[a]) {
      continue;
    }
    for (int64_t b = a + 1; b < writes.size(); ++b) {
      if (dropped[b] || !SameWrites(writes[a], writes[b])) {
        continue;
      }
      int64_t keep = a;
      int64_t drop = b;
      if (IsFreshInit(init->operand(writes[a].index)) &&
          !IsFreshInit(init->operand(writes[b].index))) {
        std::swap(keep, drop);
      }
      dropped[drop] = true;
      redirects.push_back({writes[drop].index, writes[keep].index});
      if (drop == a) {
        break;
      }
    }
  }
  if (redirects.empty()) {
    return false;
  }

  bool changed = false;
  for (const auto& [drop, keep] : redirects) {
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

}  // namespace xla::gpu
