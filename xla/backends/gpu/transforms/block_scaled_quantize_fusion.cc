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

#include "xla/backends/gpu/transforms/block_scaled_quantize_fusion.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

constexpr int64_t kViewDimensionMultiple = 32;
constexpr int64_t kPairViewDimensionMultiple = 64;

using InstructionSet = absl::flat_hash_set<HloInstruction*>;
using PostOrderPositions = absl::flat_hash_map<const HloInstruction*, int64_t>;

// One block-scaled quantization of a source value in one orientation.
struct BlockQuantization {
  HloInstruction* reduce = nullptr;
  HloInstruction* source = nullptr;
  // Blocks run along `cols` of the [rows, cols] source, or along `rows`.
  bool column = false;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t block_size = 0;
  // The elements of the source that the blocks read contiguously and in order.
  int64_t contiguous_run = 0;
  // Everything but the source and its producers.
  std::vector<HloInstruction*> instructions;
  std::vector<HloInstruction*> scale_region;
  std::vector<HloInstruction*> data_outputs;
};

bool IsQuantizedType(PrimitiveType type) {
  return type == F8E4M3FN || type == F8E5M2 || type == F4E2M1FN;
}

bool IsBitcastOfSameSize(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBitcast &&
         ShapeUtil::ElementsIn(instr->shape()) ==
             ShapeUtil::ElementsIn(instr->operand(0)->shape());
}

bool IsConstantOperand(const HloInstruction* instr) {
  return hlo_query::IsScalarConstant(instr) ||
         hlo_query::IsBroadcastOfScalarConstant(*instr);
}

int64_t PhysicalStride(const Shape& shape, int64_t dim) {
  int64_t stride = 1;
  for (int64_t i = 0; shape.layout().minor_to_major(i) != dim; ++i) {
    stride *= shape.dimensions(shape.layout().minor_to_major(i));
  }
  return stride;
}

// Returns the scalar constant that `instr`, a while body's loop-invariant
// state element viewed through a bitcast, is a broadcast of in the loop's init.
HloInstruction* LoopInvariantScalar(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kBitcast ||
      instr->operand(0)->opcode() != HloOpcode::kGetTupleElement) {
    return nullptr;
  }
  const HloInstruction* element = instr->operand(0);
  const HloComputation* body = element->parent();
  absl::InlinedVector<HloInstruction*, 1> loops =
      body->caller_instructions(HloOpcode::kWhile);
  if (element->operand(0) != body->parameter_instruction(0) ||
      loops.size() != 1 || loops.front()->while_body() != body ||
      body->root_instruction()->opcode() != HloOpcode::kTuple ||
      loops.front()->while_init()->opcode() != HloOpcode::kTuple) {
    return nullptr;
  }
  const int64_t index = element->tuple_index();
  const HloInstruction* next = body->root_instruction()->operand(index);
  HloInstruction* init = loops.front()->while_init()->mutable_operand(index);
  if (next->opcode() != HloOpcode::kGetTupleElement ||
      next->operand(0) != element->operand(0) || next->tuple_index() != index ||
      !hlo_query::IsBroadcastOfScalarConstant(*init)) {
    return nullptr;
  }
  return init->mutable_operand(0);
}

// Returns how many elements, from the most-minor dimension, `transpose` keeps
// contiguous and in order.
int64_t ContiguousRun(const HloInstruction* transpose) {
  const Shape& operand_shape = transpose->operand(0)->shape();
  const Shape& shape = transpose->shape();
  int64_t run = 1;
  for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
    const int64_t operand_dim = operand_shape.layout().minor_to_major(i);
    if (transpose->dimensions(shape.layout().minor_to_major(i)) !=
        operand_dim) {
      break;
    }
    run *= operand_shape.dimensions(operand_dim);
  }
  return run;
}

// Returns the dimension of `instr`'s operand that holds a block along `dim`,
// if the block stays inside one operand dimension of a multiple of its size.
std::optional<int64_t> OperandBlockDimension(const HloInstruction* instr,
                                             int64_t dim, int64_t block_size) {
  if (instr->opcode() == HloOpcode::kTranspose) {
    return instr->dimensions(dim);
  }
  if (instr->opcode() != HloOpcode::kBitcast) {
    return dim;
  }
  const Shape& operand_shape = instr->operand(0)->shape();
  const int64_t stride = PhysicalStride(instr->shape(), dim);
  for (int64_t i = 0; i < operand_shape.dimensions().size(); ++i) {
    if (operand_shape.dimensions(i) % block_size == 0 &&
        PhysicalStride(operand_shape, i) == stride) {
      return i;
    }
  }
  return std::nullopt;
}

// Returns the converts, bitcasts and at most one transpose from `source`.
std::optional<std::vector<HloInstruction*>> ChainFromSource(
    HloInstruction* instr, const HloInstruction* source) {
  std::vector<HloInstruction*> chain;
  bool crossed_transpose = false;
  while (instr != source) {
    if (instr->opcode() == HloOpcode::kTranspose && !crossed_transpose) {
      crossed_transpose = true;
    } else if (instr->opcode() != HloOpcode::kConvert &&
               !IsBitcastOfSameSize(instr)) {
      return std::nullopt;
    }
    chain.push_back(instr);
    instr = instr->mutable_operand(0);
  }
  return chain;
}

// Whether `instr` broadcasts a per-block value back to the blocks' elements.
bool IsScaleBroadcast(const HloInstruction* instr, int64_t block_size) {
  if (instr->opcode() != HloOpcode::kBroadcast ||
      instr->dimensions().size() + 1 != instr->shape().dimensions().size()) {
    return false;
  }
  for (int64_t i = 0; i < instr->shape().dimensions().size(); ++i) {
    if (!absl::c_linear_search(instr->dimensions(), i)) {
      return instr->shape().dimensions(i) == block_size;
    }
  }
  return false;
}

Decision MatchBlockQuantization(HloInstruction* reduce,
                                BlockQuantization* quantization) {
  if (reduce->opcode() != HloOpcode::kReduce || reduce->operand_count() != 2 ||
      reduce->dimensions().size() != 1) {
    return Decision::Forbid("not a reduce of one operand over one dimension");
  }
  if (!primitive_util::IsFloatingPointType(reduce->shape().element_type()) ||
      !Match(reduce->to_apply()->root_instruction(),
             m::MaximumAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return Decision::Forbid("not a floating-point max reduction");
  }
  const HloInstruction* init = reduce->operand(1);
  if (!init->IsConstant() ||
      !(init->literal().IsAll(0) ||
        init->literal().IsAllFloat(-std::numeric_limits<float>::infinity()))) {
    return Decision::Forbid("the init value is not 0 or -inf");
  }
  HloInstruction* amax_operand = reduce->mutable_operand(0);
  const int64_t dim = reduce->dimensions(0);
  const int64_t block_size = amax_operand->shape().dimensions(dim);
  if (block_size != 16 && block_size != 32) {
    return Decision::Forbid("the block size is not 16 or 32");
  }
  if (ShapeUtil::IsZeroElementArray(amax_operand->shape())) {
    return Decision::Forbid("the reduce has no elements");
  }

  // The amax walk: converts, bitcasts, at most one transpose and one abs.
  std::vector<HloInstruction*> walk;
  bool crossed_abs = false;
  bool crossed_transpose = false;
  HloInstruction* source = amax_operand;
  while (true) {
    if (source->opcode() == HloOpcode::kAbs && !crossed_abs) {
      crossed_abs = true;
    } else if (source->opcode() == HloOpcode::kTranspose &&
               !crossed_transpose) {
      crossed_transpose = true;
    } else if (source->opcode() != HloOpcode::kConvert &&
               !IsBitcastOfSameSize(source)) {
      break;
    }
    walk.push_back(source);
    source = source->mutable_operand(0);
  }
  if (!crossed_abs) {
    return Decision::Forbid("the reduce does not read the abs of a value");
  }

  const int64_t elements = ShapeUtil::ElementsIn(source->shape());
  quantization->block_size = block_size;
  quantization->contiguous_run = elements;
  int64_t source_dim = dim;
  for (const HloInstruction* instr : walk) {
    if (instr->opcode() == HloOpcode::kTranspose) {
      quantization->contiguous_run = ContiguousRun(instr);
    }
    std::optional<int64_t> operand_dim =
        OperandBlockDimension(instr, source_dim, block_size);
    if (!operand_dim.has_value()) {
      return Decision::Forbid("a block crosses a dimension of the source");
    }
    source_dim = *operand_dim;
  }
  const Shape& source_shape = source->shape();
  const int64_t stride = PhysicalStride(source_shape, source_dim);
  quantization->column = stride != 1;
  quantization->cols =
      quantization->column
          ? stride
          : source_shape.dimensions(source_shape.layout().minor_to_major(0));
  quantization->rows = elements / quantization->cols;
  if (quantization->rows * quantization->cols != elements ||
      quantization->rows % kViewDimensionMultiple != 0 ||
      quantization->cols % kViewDimensionMultiple != 0) {
    return Decision::Forbid(
        "the source is not an [M, N] array with M and N multiples of 32");
  }

  // The scale region lies in the elementwise closure of the reduce.
  std::vector<HloInstruction*> closure = {reduce};
  InstructionSet in_closure = {reduce};
  for (int64_t i = 0; i < closure.size(); ++i) {
    for (HloInstruction* user : closure[i]->users()) {
      bool admitted = user->IsElementwise() || IsBitcastOfSameSize(user);
      if (user->opcode() == HloOpcode::kTranspose && !crossed_transpose) {
        crossed_transpose = true;
        admitted = true;
      }
      if (admitted && user->shape().element_type() != F8E8M0FNU &&
          in_closure.insert(user).second) {
        closure.push_back(user);
      }
    }
  }
  std::vector<HloInstruction*> broadcasts;
  for (HloInstruction* instr : closure) {
    for (HloInstruction* user : instr->users()) {
      if (IsScaleBroadcast(user, block_size)) {
        broadcasts.push_back(user);
      }
    }
  }

  std::vector<HloInstruction*> data_outputs;
  std::vector<HloInstruction*> worklist = broadcasts;
  InstructionSet visited(broadcasts.begin(), broadcasts.end());
  while (!worklist.empty()) {
    HloInstruction* instr = worklist.back();
    worklist.pop_back();
    for (HloInstruction* user : instr->users()) {
      if (!user->IsElementwise() || !visited.insert(user).second) {
        continue;
      }
      if (user->opcode() == HloOpcode::kConvert &&
          IsQuantizedType(user->shape().element_type())) {
        data_outputs.push_back(user);
      } else {
        worklist.push_back(user);
      }
    }
  }
  if (data_outputs.empty()) {
    return Decision::Forbid("no FP8 or FP4 convert reads the scale");
  }

  std::vector<HloInstruction*> data_region = data_outputs;
  InstructionSet in_data_region(data_outputs.begin(), data_outputs.end());
  InstructionSet source_chain;
  InstructionSet used_broadcasts;
  bool reads_source = false;
  worklist = data_outputs;
  while (!worklist.empty()) {
    HloInstruction* instr = worklist.back();
    worklist.pop_back();
    for (HloInstruction* operand : instr->mutable_operands()) {
      if (operand == source) {
        reads_source = true;
        continue;
      }
      if (in_data_region.contains(operand) || source_chain.contains(operand) ||
          IsConstantOperand(operand)) {
        continue;
      }
      if (absl::c_linear_search(broadcasts, operand)) {
        used_broadcasts.insert(operand);
        continue;
      }
      if (std::optional<std::vector<HloInstruction*>> chain =
              ChainFromSource(operand, source)) {
        reads_source = true;
        source_chain.insert(chain->begin(), chain->end());
        continue;
      }
      if (!operand->IsElementwise()) {
        return Decision::Forbid(
            "the data region reads a value other than the source, the scale "
            "and constants");
      }
      data_region.push_back(operand);
      in_data_region.insert(operand);
      worklist.push_back(operand);
    }
  }
  if (!reads_source) {
    return Decision::Forbid("the data region does not read the source");
  }
  auto in_region = [&](const HloInstruction* user) {
    return in_data_region.contains(user);
  };
  for (const HloInstruction* instr : data_region) {
    if (!absl::c_linear_search(data_outputs, instr) &&
        (instr->IsRoot() || !absl::c_all_of(instr->users(), in_region))) {
      return Decision::Forbid(
          "a value of the data region is used outside the quantization");
    }
  }
  for (const HloInstruction* instr : used_broadcasts) {
    if (instr->IsRoot() || !absl::c_all_of(instr->users(), in_region)) {
      return Decision::Forbid(
          "a broadcast scale is used outside the quantization");
    }
  }

  InstructionSet in_scale_region;
  for (HloInstruction* instr : used_broadcasts) {
    worklist.push_back(instr->mutable_operand(0));
  }
  while (!worklist.empty()) {
    HloInstruction* instr = worklist.back();
    worklist.pop_back();
    if (in_closure.contains(instr) && in_scale_region.insert(instr).second) {
      worklist.insert(worklist.end(), instr->operands().begin(),
                      instr->operands().end());
    }
  }
  for (HloInstruction* instr : closure) {
    if (!in_scale_region.contains(instr)) {
      continue;
    }
    for (const HloInstruction* operand : instr->operands()) {
      if (instr != reduce && !in_scale_region.contains(operand) &&
          !IsConstantOperand(operand) &&
          !hlo_query::IsEffectiveParameter(*operand)) {
        return Decision::Forbid(
            "the scale region reads a value other than the amax, constants "
            "and parameters");
      }
    }
    quantization->scale_region.push_back(instr);
  }

  InstructionSet instructions(walk.begin(), walk.end());
  instructions.insert(quantization->scale_region.begin(),
                      quantization->scale_region.end());
  instructions.insert(used_broadcasts.begin(), used_broadcasts.end());
  instructions.insert(data_region.begin(), data_region.end());
  instructions.insert(source_chain.begin(), source_chain.end());
  if (!source->shape().is_static() ||
      !absl::c_all_of(instructions, [](const HloInstruction* instr) {
        return instr->shape().is_static();
      })) {
    return Decision::Forbid("dynamic shape");
  }
  quantization->reduce = reduce;
  quantization->source = source;
  quantization->instructions.assign(instructions.begin(), instructions.end());
  quantization->data_outputs = std::move(data_outputs);
  return Decision::Allow();
}

// Whether every user of `instr` is fused and none of them also stays outside.
bool UsedOnlyInside(const HloInstruction* instr, const InstructionSet& fused,
                    const InstructionSet& duplicated) {
  return !instr->IsRoot() &&
         absl::c_all_of(instr->users(), [&](const HloInstruction* user) {
           return fused.contains(user) && !duplicated.contains(user);
         });
}

// Whether `slice` keeps whole the most-minor non-degenerate dimension of its
// operand, so that it reads the operand in contiguous runs.
bool ReadsContiguousRuns(const HloInstruction* slice) {
  const Shape& operand_shape = slice->operand(0)->shape();
  for (int64_t dim : operand_shape.layout().minor_to_major()) {
    if (operand_shape.dimensions(dim) > 1) {
      return slice->shape().dimensions(dim) == operand_shape.dimensions(dim);
    }
  }
  return true;
}

// Whether `instr` broadcasts a value other than a scalar constant to more
// dimensions.
bool IsExpandingBroadcast(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBroadcast &&
         !hlo_query::IsScalarConstant(instr->operand(0)) &&
         instr->operand(0)->shape().dimensions().size() <
             instr->shape().dimensions().size();
}

// A producer is absorbed if only the fusion uses it, or if duplicating it reads
// no more bytes than its result; below an expanding broadcast, only if it moves
// data, since the fusion evaluates it for each element of the broadcast.
Decision CanAbsorb(const HloInstruction* instr, const InstructionSet& fused,
                   bool used_only_inside, bool below_broadcast) {
  if (!instr->IsElementwise() && instr->opcode() != HloOpcode::kBitcast &&
      instr->opcode() != HloOpcode::kBroadcast &&
      instr->opcode() != HloOpcode::kDynamicSlice) {
    return Decision::Forbid(
        "not an elementwise op, bitcast, broadcast or dynamic-slice");
  }
  if (instr->opcode() == HloOpcode::kDynamicSlice &&
      !ReadsContiguousRuns(instr)) {
    return Decision::Forbid(
        "the dynamic-slice narrows the most-minor dimension of its operand");
  }
  if (below_broadcast &&
      !HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kConvert,
                        HloOpcode::kDynamicSlice>(instr)) {
    return Decision::Forbid(
        "a broadcast would recompute it for each of its elements");
  }
  if (absl::c_any_of(instr->operands(), [](const HloInstruction* operand) {
        return HloPredicateIsOp<HloOpcode::kTranspose, HloOpcode::kConcatenate,
                                HloOpcode::kSlice>(operand);
      })) {
    return Decision::Forbid(
        "absorbing it would materialize a transpose, concatenate or slice "
        "that it fuses today");
  }
  if (used_only_inside) {
    return Decision::Allow();
  }
  int64_t new_operand_bytes = 0;
  for (const HloInstruction* operand : instr->operands()) {
    if (!fused.contains(operand) && !IsConstantOperand(operand)) {
      new_operand_bytes += ShapeUtil::ByteSizeOf(operand->shape());
    }
  }
  return Decision(new_operand_bytes <= ShapeUtil::ByteSizeOf(instr->shape()),
                  "duplicating it would read more bytes");
}

// Adds `source` and the producers that `CanAbsorb` allows, users first.
void AbsorbProducers(HloInstruction* source,
                     const PostOrderPositions& positions,
                     InstructionSet& fused) {
  auto earlier = [&](const HloInstruction* a, const HloInstruction* b) {
    return positions.at(a) < positions.at(b);
  };
  std::priority_queue<HloInstruction*, std::vector<HloInstruction*>,
                      decltype(earlier)>
      candidates(earlier);
  InstructionSet queued = {source};
  InstructionSet duplicated;
  InstructionSet below_broadcast;
  candidates.push(source);
  while (!candidates.empty()) {
    HloInstruction* instr = candidates.top();
    candidates.pop();
    const bool used_only_inside = UsedOnlyInside(instr, fused, duplicated);
    Decision decision = CanAbsorb(instr, fused, used_only_inside,
                                  below_broadcast.contains(instr));
    if (decision.IsForbidden()) {
      VLOG(3) << "Not absorbing " << instr->name() << ": "
              << decision.Explain();
      continue;
    }
    fused.insert(instr);
    if (!used_only_inside) {
      duplicated.insert(instr);
    }
    const bool operands_below_broadcast =
        IsExpandingBroadcast(instr) || below_broadcast.contains(instr);
    for (HloInstruction* operand : instr->mutable_operands()) {
      if (operands_below_broadcast) {
        below_broadcast.insert(operand);
      }
      if (!fused.contains(operand) && queued.insert(operand).second) {
        candidates.push(operand);
      }
    }
  }
}

// Fuses one quantization, or a row and a column one; false if refused.
absl::StatusOr<bool> FuseQuantizations(
    absl::Span<const BlockQuantization> quantizations,
    const BlockScaledQuantizeFusion::CanEmit& can_emit,
    PostOrderPositions& positions) {
  HloInstruction* source = quantizations.front().source;
  // A pair takes the column's view; the row's blocks must tile it.
  const BlockQuantization& row = quantizations.front();
  const BlockQuantization& column = quantizations.back();
  if (quantizations.size() == 2 &&
      (column.cols % row.block_size != 0 ||
       row.contiguous_run % column.cols != 0 ||
       column.rows % kPairViewDimensionMultiple != 0 ||
       column.cols % kPairViewDimensionMultiple != 0)) {
    VLOG(2) << "Not fusing both orientations of " << source->name()
            << ": the row does not tile the column's view, or the view is not "
               "a multiple of 64";
    return false;
  }
  HloComputation* computation = source->parent();
  HloModule* module = computation->parent();
  auto by_position = [&](const HloInstruction* a, const HloInstruction* b) {
    return positions.at(a) < positions.at(b);
  };

  InstructionSet fused;
  for (const BlockQuantization& quantization : quantizations) {
    fused.insert(quantization.instructions.begin(),
                 quantization.instructions.end());
  }
  AbsorbProducers(source, positions, fused);
  std::vector<HloInstruction*> members(fused.begin(), fused.end());
  absl::c_sort(members, by_position);
  if (absl::c_any_of(members, [](const HloInstruction* instr) {
        return instr->HasControlDependencies();
      })) {
    VLOG(2) << "Not fusing the quantization of " << source->name()
            << ": an instruction has control dependencies";
    return false;
  }

  std::vector<HloInstruction*> roots;
  for (const BlockQuantization& quantization : quantizations) {
    std::vector<HloInstruction*> data_outputs = quantization.data_outputs;
    absl::c_sort(data_outputs, by_position);
    roots.insert(roots.end(), data_outputs.begin(), data_outputs.end());
  }
  std::vector<HloInstruction*> predicates;
  for (const BlockQuantization& quantization : quantizations) {
    std::vector<HloInstruction*> scale_region = quantization.scale_region;
    absl::c_sort(scale_region, by_position);
    for (HloInstruction* instr : scale_region) {
      if (instr->IsRoot() ||
          absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
            return !fused.contains(user);
          })) {
        (instr->shape().element_type() == PRED ? predicates : roots)
            .push_back(instr);
      }
    }
  }
  roots.insert(roots.end(), predicates.begin(), predicates.end());

  HloComputation::Builder builder("block_scaled_quantize_computation");
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> clones;
  std::vector<HloInstruction*> parameters;
  auto clone_constant = [&](HloInstruction* constant) {
    HloInstruction*& clone = clones[constant];
    if (clone == nullptr) {
      clone = builder.AddInstruction(constant->Clone());
    }
    return clone;
  };
  auto clone_operand = [&](HloInstruction* operand) {
    if (hlo_query::IsScalarConstant(operand)) {
      return clone_constant(operand);
    }
    if (auto it = clones.find(operand); it != clones.end()) {
      return it->second;
    }
    HloInstruction* clone;
    if (hlo_query::IsBroadcastOfScalarConstant(*operand)) {
      clone = builder.AddInstruction(operand->CloneWithNewOperands(
          operand->shape(), {clone_constant(operand->mutable_operand(0))}));
    } else if (HloInstruction* scalar = LoopInvariantScalar(operand)) {
      clone = builder.AddInstruction(HloInstruction::CreateBroadcast(
          operand->shape(), clone_constant(scalar), {}));
    } else {
      clone = builder.AddInstruction(HloInstruction::CreateParameter(
          parameters.size(), operand->shape(),
          absl::StrCat("p", parameters.size())));
      parameters.push_back(operand);
    }
    clones[operand] = clone;
    return clone;
  };
  for (HloInstruction* instr : members) {
    std::vector<HloInstruction*> operands;
    for (HloInstruction* operand : instr->mutable_operands()) {
      operands.push_back(fused.contains(operand) ? clones.at(operand)
                                                 : clone_operand(operand));
    }
    clones[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), operands));
  }
  std::vector<HloInstruction*> fused_roots;
  for (const HloInstruction* root : roots) {
    fused_roots.push_back(clones.at(root));
  }
  builder.AddInstruction(HloInstruction::CreateTuple(fused_roots));
  HloComputation* fused_computation = module->AddComputationAndUnifyNamesAndIds(
      builder.Build(), /*is_entry=*/false);
  HloInstruction* fusion =
      computation->AddInstruction(HloInstruction::CreateFusion(
          fused_computation->root_instruction()->shape(),
          HloInstruction::FusionKind::kCustom, parameters, fused_computation));
  module->SetAndUniquifyInstrName(fusion, "block_scaled_quantize");
  fusion->set_metadata(quantizations.front().reduce->metadata());
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        fusion->backend_config<GpuBackendConfig>());
  gpu_config.mutable_fusion_backend_config()->set_kind(
      kBlockScaledQuantizeFusionKind);
  ABSL_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  Decision decision = can_emit(*Cast<HloFusionInstruction>(fusion));
  if (decision.IsForbidden()) {
    VLOG(2) << "Not fusing the quantization of " << source->name() << ": "
            << decision.Explain();
    ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(fusion));
    ABSL_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(fused_computation));
    return false;
  }

  for (int64_t i = 0; i < roots.size(); ++i) {
    HloInstruction* root = roots[i];
    HloInstruction* output = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(fusion, i));
    // Never absorbed, so any position keeps the order of the others valid.
    positions[output] = positions.at(root);
    std::vector<HloInstruction*> outside_users;
    for (HloInstruction* user : root->users()) {
      if (!fused.contains(user)) {
        outside_users.push_back(user);
      }
    }
    for (HloInstruction* user : outside_users) {
      ABSL_RETURN_IF_ERROR(root->ReplaceUseWith(user, output));
    }
    if (root->IsRoot()) {
      computation->set_root_instruction(output);
    }
  }
  positions[fusion] = positions.at(roots.front());
  InstructionSet removed;
  for (HloInstruction* root : roots) {
    if (!removed.contains(root) && root->IsDead()) {
      ABSL_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(
          root, [&](HloInstruction* instr) { removed.insert(instr); }));
    }
  }
  VLOG(2) << "Fused " << fusion->ToString();
  return true;
}

// Fuses the first row and column quantization together, the others alone.
absl::StatusOr<bool> FuseQuantizationsOfSource(
    absl::Span<HloInstruction* const> reduces,
    const BlockScaledQuantizeFusion::CanEmit& can_emit,
    PostOrderPositions& positions) {
  std::optional<BlockQuantization> row;
  std::optional<BlockQuantization> column;
  for (HloInstruction* reduce : reduces) {
    BlockQuantization quantization;
    if (MatchBlockQuantization(reduce, &quantization).IsForbidden()) {
      continue;
    }
    std::optional<BlockQuantization>& first =
        quantization.column ? column : row;
    if (!first.has_value()) {
      first = std::move(quantization);
    }
  }
  bool changed = false;
  InstructionSet fused_reduces;
  if (row.has_value() && column.has_value() && row->source == column->source) {
    ABSL_ASSIGN_OR_RETURN(
        changed, FuseQuantizations({*row, *column}, can_emit, positions));
    if (changed) {
      fused_reduces = {row->reduce, column->reduce};
    }
  }
  for (HloInstruction* reduce : reduces) {
    BlockQuantization quantization;
    if (fused_reduces.contains(reduce) ||
        MatchBlockQuantization(reduce, &quantization).IsForbidden()) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(
        bool fused, FuseQuantizations({quantization}, can_emit, positions));
    changed |= fused;
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> BlockScaledQuantizeFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<std::vector<HloInstruction*>> reduces_by_source;
    absl::flat_hash_map<const HloInstruction*, int64_t> source_index;
    PostOrderPositions positions;
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      const int64_t position = positions.size();
      positions[instr] = position;
      if (instr->opcode() != HloOpcode::kReduce) {
        continue;
      }
      BlockQuantization quantization;
      Decision decision = MatchBlockQuantization(instr, &quantization);
      if (decision.IsForbidden()) {
        VLOG(3) << "Not a block quantization: " << instr->name() << ": "
                << decision.Explain();
        continue;
      }
      auto [it, inserted] = source_index.try_emplace(quantization.source,
                                                     reduces_by_source.size());
      if (inserted) {
        reduces_by_source.emplace_back();
      }
      reduces_by_source[it->second].push_back(instr);
    }
    for (const std::vector<HloInstruction*>& reduces : reduces_by_source) {
      ABSL_ASSIGN_OR_RETURN(
          bool fused, FuseQuantizationsOfSource(reduces, can_emit_, positions));
      changed |= fused;
    }
  }
  return changed;
}

}  // namespace xla::gpu
