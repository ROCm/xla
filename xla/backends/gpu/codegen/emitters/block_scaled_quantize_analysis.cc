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

#include "xla/backends/gpu/codegen/emitters/block_scaled_quantize_analysis.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/index_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using Orientation = BlockScaledQuantization::Orientation;
using ::mlir::MLIRContext;

// Where an instruction of the fused computation belongs.
enum class Region {
  // Constants and parameters outside the prologue and their broadcasts, read
  // at the index of their user.
  kLeaf,
  // The producers of the source.
  kPrologue,
  // Values computed per element of the source: the source itself, the amax
  // operands and the data region.
  kElement,
  // An amax reduce and the values computed per block from it.
  kScale,
  // A broadcast of a scale-region value to the data shape.
  kBroadcastScale,
};

struct Classification {
  Region region;
  // kElement: (row, col) of the source -> index of the instruction.
  // kScale: index of the owning amax -> index of the instruction.
  IndexingMap indexing = IndexingMap::GetUndefined();
  // The amax whose scale the value depends on.
  const HloInstruction* owner = nullptr;
  // kBroadcastScale: the broadcast of the scale value.
  const HloInstruction* broadcast = nullptr;
};

bool IsScalarLike(const HloInstruction& instr) {
  auto is_scalar_leaf = [](const HloInstruction& leaf) {
    return (leaf.opcode() == HloOpcode::kConstant ||
            leaf.opcode() == HloOpcode::kParameter) &&
           ShapeUtil::IsEffectiveScalar(leaf.shape());
  };
  return is_scalar_leaf(instr) || (instr.opcode() == HloOpcode::kBroadcast &&
                                   is_scalar_leaf(*instr.operand(0)));
}

bool IsParameterOrItsBroadcast(const HloInstruction& instr) {
  const HloInstruction* leaf =
      instr.opcode() == HloOpcode::kBroadcast ? instr.operand(0) : &instr;
  return leaf->opcode() == HloOpcode::kParameter ||
         leaf->opcode() == HloOpcode::kConstant;
}

absl::Status Refuse(absl::string_view reason) {
  return absl::InvalidArgumentError(
      absl::StrCat("Not a block-scaled quantization: ", reason));
}

// Strips the converts above `instr`, recording their types from the top.
const HloInstruction* StripConverts(const HloInstruction* instr,
                                    std::vector<PrimitiveType>& types) {
  while (instr->opcode() == HloOpcode::kConvert) {
    types.push_back(instr->shape().element_type());
    instr = instr->operand(0);
  }
  return instr;
}

// max(a, b); with float normalization, max(a, b) in a wider float type and
// converted back, or, without excess precision, the max of a and b rounded to a
// narrower float type and rounded again. Each is a max of the rounded values.
bool IsMaxReducer(const HloComputation& reducer) {
  std::vector<PrimitiveType> root_types;
  std::vector<PrimitiveType> lhs_types;
  std::vector<PrimitiveType> rhs_types;
  const HloInstruction* max =
      StripConverts(reducer.root_instruction(), root_types);
  if (max->opcode() != HloOpcode::kMaximum) {
    return false;
  }
  const HloInstruction* lhs = StripConverts(max->operand(0), lhs_types);
  const HloInstruction* rhs = StripConverts(max->operand(1), rhs_types);
  if (lhs->opcode() != HloOpcode::kParameter ||
      rhs->opcode() != HloOpcode::kParameter || lhs == rhs ||
      lhs_types != rhs_types) {
    return false;
  }
  const PrimitiveType type = lhs->shape().element_type();
  if (!primitive_util::IsFloatingPointType(type)) {
    return lhs_types.empty() && root_types.empty();
  }
  switch (lhs_types.size()) {
    case 0:
      return root_types.empty();
    case 1:
      return root_types == std::vector<PrimitiveType>{type} &&
             primitive_util::IsFloatingPointType(lhs_types[0]) &&
             primitive_util::CastPreservesValues(type, lhs_types[0]);
    case 2:
      return lhs_types[0] == type && root_types == lhs_types &&
             primitive_util::IsFloatingPointType(lhs_types[1]) &&
             primitive_util::CastPreservesValues(lhs_types[1], type);
    default:
      return false;
  }
}

absl::Status CheckAmax(const HloInstruction& reduce) {
  if (reduce.operand_count() != 2) {
    return Refuse(absl::StrCat(reduce.name(), " is a variadic reduce"));
  }
  if (reduce.dimensions().size() != 1) {
    return Refuse(
        absl::StrCat(reduce.name(), " does not reduce exactly one dimension"));
  }
  const int64_t block_size =
      reduce.operand(0)->shape().dimensions(reduce.dimensions(0));
  if (block_size != 32) {
    return Refuse(absl::StrCat(reduce.name(), " has block size ", block_size));
  }
  if (!IsMaxReducer(*reduce.to_apply())) {
    return Refuse(absl::StrCat(reduce.name(), " is not a max reduce"));
  }
  // Float normalization widens a bf16 init behind a convert.
  const HloInstruction* init = reduce.operand(1);
  while (init->opcode() == HloOpcode::kConvert) {
    init = init->operand(0);
  }
  if (init->opcode() != HloOpcode::kConstant ||
      !(init->literal().IsAllFloat(0) ||
        init->literal() ==
            LiteralUtil::MinValue(init->shape().element_type()))) {
    return Refuse(
        absl::StrCat(reduce.name(), " has an init other than 0 or -inf"));
  }
  return absl::OkStatus();
}

// Follows the amax operand upward through converts, bitcasts, the abs and at
// most one transpose. Returns where the walk ends: the source.
absl::StatusOr<const HloInstruction*> WalkToSource(
    const HloInstruction& reduce) {
  const HloInstruction* instr = reduce.operand(0);
  int num_abs = 0;
  int num_transposes = 0;
  while (instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kConvert ||
         instr->opcode() == HloOpcode::kAbs ||
         instr->opcode() == HloOpcode::kTranspose) {
    num_abs += instr->opcode() == HloOpcode::kAbs;
    num_transposes += instr->opcode() == HloOpcode::kTranspose;
    instr = instr->operand(0);
  }
  if (num_abs != 1) {
    return Refuse(absl::StrCat("the operand of ", reduce.name(), " has ",
                               num_abs, " abs instead of one"));
  }
  if (num_transposes > 1) {
    return Refuse(absl::StrCat("the operand of ", reduce.name(), " has ",
                               num_transposes, " transposes"));
  }
  return instr;
}

IndexingMap Compose(const IndexingMap& first, const IndexingMap& second) {
  IndexingMap result = ComposeIndexingMaps(first, second);
  result.Simplify();
  result.RemoveUnusedSymbols();
  return result;
}

bool SameFunction(const IndexingMap& lhs, const IndexingMap& rhs) {
  return lhs.GetSymbolicMap() == rhs.GetSymbolicMap() &&
         lhs.GetRangeVarsCount() == 0 && rhs.GetRangeVarsCount() == 0;
}

// The map from the index of `instr`'s operand `operand_index` to the index of
// `instr`, or an undefined map.
IndexingMap InputToOutput(const HloInstruction& instr, int64_t operand_index,
                          MLIRContext* mlir_context) {
  HloInstructionIndexing indexing =
      ComputeInputToOutputIndexing(&instr, operand_index, mlir_context);
  if (indexing.indexing_maps.size() != 1 ||
      indexing.indexing_maps[0].size() != 1) {
    return IndexingMap::GetUndefined();
  }
  return indexing.indexing_maps[0].begin()->map();
}

// Composes the map of every classified operand of `instr` with `instr`'s own
// input-to-output map and checks that all operands agree.
absl::StatusOr<IndexingMap> PropagateIndexing(
    const HloInstruction& instr, absl::Span<const int64_t> operand_indices,
    const absl::flat_hash_map<const HloInstruction*, Classification>& regions,
    MLIRContext* mlir_context) {
  std::optional<IndexingMap> result;
  for (int64_t operand_index : operand_indices) {
    IndexingMap step = InputToOutput(instr, operand_index, mlir_context);
    if (step.IsUndefined()) {
      return Refuse(absl::StrCat("no indexing through ", instr.name()));
    }
    IndexingMap candidate = Compose(
        regions.at(instr.operand(operand_index)).indexing, std::move(step));
    if (result.has_value() && !SameFunction(*result, candidate)) {
      return Refuse(absl::StrCat("the operands of ", instr.name(),
                                 " are read at different indices"));
    }
    result = std::move(candidate);
  }
  return *result;
}

bool IsAllowedRegionOp(const HloInstruction& instr) {
  return HloInstruction::IsOpElementwise(instr.opcode()) ||
         instr.opcode() == HloOpcode::kBitcast ||
         instr.opcode() == HloOpcode::kTranspose;
}

bool IsDataType(PrimitiveType type) {
  return type == F8E4M3FN || type == F4E2M1FN;
}

// Whether each dimension of `shape` is the product of consecutive dimensions of
// `split`.
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

// The reshape, transpose and reshape of a result `merge` that store its bytes
// in another order (the hipBLASLt 32x8 scale swizzle): the first reshape only
// splits dimensions and the last one restores the shape.
std::vector<const HloInstruction*> MatchResultPermutation(
    const HloInstruction& merge) {
  if (merge.opcode() != HloOpcode::kReshape || merge.user_count() != 1) {
    return {};
  }
  const HloInstruction* transpose = merge.operand(0);
  if (transpose->opcode() != HloOpcode::kTranspose ||
      transpose->user_count() != 1) {
    return {};
  }
  const HloInstruction* split = transpose->operand(0);
  if (split->opcode() != HloOpcode::kReshape || split->user_count() != 1 ||
      !ShapeUtil::SameDimensions(merge.shape(), split->operand(0)->shape()) ||
      !OnlySplitsDimensions(split->operand(0)->shape(), split->shape())) {
    return {};
  }
  return {split, transpose, &merge};
}

// (d0, d1) of `sizes` -> the source (row, col) of the block's first element.
IndexingMap BlockToElement(Orientation orientation,
                           absl::Span<const int64_t> sizes, int64_t block_size,
                           MLIRContext* mlir_context) {
  SymbolicExpr d0 = CreateDimExpr(0, mlir_context);
  SymbolicExpr d1 = CreateDimExpr(1, mlir_context);
  llvm::SmallVector<SymbolicExpr> results =
      orientation == Orientation::kRow
          ? llvm::SmallVector<SymbolicExpr>{d0, d1 * block_size}
          : llvm::SmallVector<SymbolicExpr>{d0 * block_size, d1};
  return IndexingMap::FromTensorSizes(
      SymbolicMap::Get(mlir_context, 2, 0, std::move(results)), sizes, {});
}

// Whether the amax of `element_to_amax` reduces blocks of `orientation`: the
// amax index does not depend on the position inside the block.
bool ReducesBlocksOf(Orientation orientation,
                     const IndexingMap& element_to_amax, int64_t rows,
                     int64_t cols, int64_t block_size,
                     MLIRContext* mlir_context) {
  SymbolicExpr d0 = CreateDimExpr(0, mlir_context);
  SymbolicExpr d1 = CreateDimExpr(1, mlir_context);
  SymbolicExpr d2 = CreateDimExpr(2, mlir_context);
  // The position inside the block is d2 for rows and d1 for columns.
  IndexingMap block_view =
      orientation == Orientation::kRow
          ? IndexingMap::FromTensorSizes(
                SymbolicMap::Get(mlir_context, 3, 0,
                                 {d0, d1 * block_size + d2}),
                {rows, cols / block_size, block_size}, {})
          : IndexingMap::FromTensorSizes(
                SymbolicMap::Get(mlir_context, 3, 0,
                                 {d0 * block_size + d1, d2}),
                {rows / block_size, block_size, cols}, {});
  IndexingMap composed = Compose(block_view, element_to_amax);
  const int64_t position = orientation == Orientation::kRow ? 2 : 1;
  return composed.GetRangeVarsCount() == 0 &&
         absl::c_none_of(composed.GetSymbolicMap().GetResults(),
                         [&](SymbolicExpr result) {
                           return result.IsFunctionOfVariable(position);
                         });
}

}  // namespace

std::vector<int64_t> BlockScaledQuantization::BlockCounts(
    int64_t rows, int64_t cols, int64_t block_size) const {
  if (orientation == Orientation::kRow) {
    return {rows, cols / block_size};
  }
  return {rows / block_size, cols};
}

const BlockScaledQuantization* BlockScaledQuantizeAnalysis::row() const {
  for (const BlockScaledQuantization& quantization : quantizations) {
    if (quantization.orientation == Orientation::kRow) {
      return &quantization;
    }
  }
  return nullptr;
}

const BlockScaledQuantization* BlockScaledQuantizeAnalysis::column() const {
  for (const BlockScaledQuantization& quantization : quantizations) {
    if (quantization.orientation == Orientation::kColumn) {
      return &quantization;
    }
  }
  return nullptr;
}

// Whether the data outputs of the column quantization `quantization` store the
// rows of S contiguously rather than its blocks.
absl::StatusOr<bool> IsDataInSourceOrder(
    const BlockScaledQuantization& quantization, const HloInstruction& root,
    MLIRContext* mlir_context) {
  std::optional<bool> source_order;
  for (int64_t i = 0; i < quantization.data_root_indices.size(); ++i) {
    const int64_t index = quantization.data_root_indices[i];
    const IndexingMap& indexing = quantization.data_root_indexing[i];
    const Shape& shape = root.operand(index)->shape();
    auto offset = [&](int64_t row, int64_t col) {
      llvm::SmallVector<int64_t> element = indexing.Evaluate(
          CreateSymbolicConstantExprs({row, col}, mlir_context), {});
      return IndexUtil::MultidimensionalIndexToLinearIndex(shape, element);
    };
    bool along_row;
    if (offset(1, 0) == offset(0, 0) + 1) {
      along_row = false;
    } else if (offset(0, 1) == offset(0, 0) + 1) {
      along_row = true;
    } else {
      return Refuse(absl::StrCat("the data output ",
                                 root.operand(index)->name(),
                                 " stores neither a block nor a row of the "
                                 "source contiguously"));
    }
    if (source_order.has_value() && *source_order != along_row) {
      return Refuse("the data outputs of one amax have different orders");
    }
    source_order = along_row;
  }
  return *source_order;
}

absl::StatusOr<BlockScaledQuantizeAnalysis> AnalyzeBlockScaledQuantize(
    const HloComputation& computation, MLIRContext* mlir_context) {
  const HloInstruction* root = computation.root_instruction();
  if (root->opcode() != HloOpcode::kTuple) {
    return Refuse("the root is not a tuple");
  }
  std::vector<HloInstruction*> post_order =
      computation.MakeInstructionPostOrder();

  std::vector<const HloInstruction*> amaxes;
  for (const HloInstruction* instr : post_order) {
    if (!instr->shape().IsTuple() && !instr->shape().is_static()) {
      return Refuse(absl::StrCat(instr->name(), " has a dynamic shape"));
    }
    if (instr->opcode() == HloOpcode::kReduce) {
      ABSL_RETURN_IF_ERROR(CheckAmax(*instr));
      amaxes.push_back(instr);
    }
  }
  if (amaxes.empty() || amaxes.size() > 2) {
    return Refuse(absl::StrCat(amaxes.size(), " amax reduces"));
  }

  BlockScaledQuantizeAnalysis analysis;
  for (const HloInstruction* amax : amaxes) {
    ABSL_ASSIGN_OR_RETURN(const HloInstruction* source, WalkToSource(*amax));
    if (analysis.source != nullptr && analysis.source != source) {
      return Refuse("the amax reduces read different sources");
    }
    analysis.source = source;
  }
  const HloInstruction* source = analysis.source;
  const Shape& source_shape = source->shape();
  if (IsScalarLike(*source) || !source_shape.IsArray() ||
      !(source_shape.element_type() == BF16 ||
        source_shape.element_type() == F32)) {
    return Refuse(absl::StrCat("unsupported source ", source->ToString()));
  }
  analysis.block_size = amaxes.front()->operand(0)->shape().dimensions(
      amaxes.front()->dimensions(0));
  // The source's producers form the prologue.
  absl::flat_hash_set<const HloInstruction*> prologue;
  std::vector<const HloInstruction*> stack(source->operands().begin(),
                                           source->operands().end());
  while (!stack.empty()) {
    const HloInstruction* instr = stack.back();
    stack.pop_back();
    if (IsScalarLike(*instr) || !prologue.insert(instr).second) {
      continue;
    }
    absl::c_copy(instr->operands(), std::back_inserter(stack));
  }
  for (const HloInstruction* parameter : computation.parameter_instructions()) {
    if (parameter == source || prologue.contains(parameter)) {
      analysis.prologue_parameters.push_back(parameter);
    }
  }

  absl::flat_hash_set<const HloInstruction*> result_permutations;
  for (const HloInstruction* output : root->operands()) {
    for (const HloInstruction* instr : MatchResultPermutation(*output)) {
      result_permutations.insert(instr);
    }
  }

  absl::flat_hash_map<const HloInstruction*, Classification> regions;
  absl::flat_hash_map<const HloInstruction*, IndexingMap> element_to_amax;
  for (const HloInstruction* instr : post_order) {
    if (instr == root) {
      continue;
    }
    if (instr == source) {
      regions[instr] = {Region::kElement,
                        CreateIdentityMap(source_shape, mlir_context)};
      continue;
    }
    if (IsScalarLike(*instr) ||
        (!prologue.contains(instr) && IsParameterOrItsBroadcast(*instr))) {
      regions[instr] = {Region::kLeaf};
      continue;
    }
    if (prologue.contains(instr)) {
      if (instr->opcode() == HloOpcode::kReduce) {
        return Refuse(
            absl::StrCat("the prologue has a reduce ", instr->name()));
      }
      regions[instr] = {Region::kPrologue};
      continue;
    }
    if (instr->opcode() == HloOpcode::kReduce) {
      const Classification& operand = regions.at(instr->operand(0));
      if (operand.region != Region::kElement || operand.owner != nullptr) {
        return Refuse(
            absl::StrCat(instr->name(), " does not reduce the source"));
      }
      ABSL_ASSIGN_OR_RETURN(
          IndexingMap indexing,
          PropagateIndexing(*instr, {0}, regions, mlir_context));
      element_to_amax.insert_or_assign(instr, std::move(indexing));
      regions[instr] = {Region::kScale,
                        CreateIdentityMap(instr->shape(), mlir_context), instr};
      continue;
    }
    if (instr->opcode() == HloOpcode::kBroadcast) {
      const Classification& operand = regions.at(instr->operand(0));
      if (operand.region == Region::kLeaf) {
        regions[instr] = {Region::kLeaf};
        continue;
      }
      if (operand.region != Region::kScale) {
        return Refuse(absl::StrCat(instr->name(),
                                   " broadcasts a value other than a scale"));
      }
      regions[instr] = {Region::kBroadcastScale, IndexingMap::GetUndefined(),
                        operand.owner, instr};
      continue;
    }
    if (!IsAllowedRegionOp(*instr) && !result_permutations.contains(instr)) {
      return Refuse(absl::StrCat("unsupported instruction ", instr->name()));
    }
    if (result_permutations.contains(instr) &&
        regions.at(instr->operand(0)).region != Region::kScale) {
      return Refuse(absl::StrCat(instr->name(),
                                 " permutes a result other than a scale"));
    }
    std::vector<int64_t> scale_operands;
    std::vector<int64_t> element_operands;
    std::vector<int64_t> broadcast_operands;
    const HloInstruction* owner = nullptr;
    for (int64_t i = 0; i < instr->operand_count(); ++i) {
      const Classification& operand = regions.at(instr->operand(i));
      switch (operand.region) {
        case Region::kLeaf:
          continue;
        case Region::kPrologue:
          return Refuse(absl::StrCat(instr->name(),
                                     " reads the prologue value ",
                                     instr->operand(i)->name()));
        case Region::kElement:
          element_operands.push_back(i);
          break;
        case Region::kScale:
          scale_operands.push_back(i);
          break;
        case Region::kBroadcastScale:
          broadcast_operands.push_back(i);
          break;
      }
      if (operand.owner != nullptr) {
        if (owner != nullptr && owner != operand.owner) {
          return Refuse(
              absl::StrCat(instr->name(), " reads the scales of two amaxes"));
        }
        owner = operand.owner;
      }
    }
    if (!scale_operands.empty()) {
      if (!element_operands.empty() || !broadcast_operands.empty()) {
        return Refuse(absl::StrCat(
            instr->name(), " mixes the scale region with the data region"));
      }
      ABSL_ASSIGN_OR_RETURN(
          IndexingMap indexing,
          PropagateIndexing(*instr, scale_operands, regions, mlir_context));
      regions[instr] = {Region::kScale, std::move(indexing), owner};
      continue;
    }
    if (element_operands.empty() && broadcast_operands.empty()) {
      // A function of leaves only, such as a widened constant.
      regions[instr] = {Region::kLeaf};
      continue;
    }
    if (element_operands.empty() && broadcast_operands.size() == 1 &&
        HloInstruction::IsOpElementwise(instr->opcode())) {
      // A function of one broadcast scale, such as its widening convert after
      // float normalization, is read like the broadcast.
      regions[instr] = {
          Region::kBroadcastScale, IndexingMap::GetUndefined(), owner,
          regions.at(instr->operand(broadcast_operands.front())).broadcast};
      continue;
    }
    if (element_operands.empty()) {
      return Refuse(absl::StrCat(instr->name(), " reads no source value"));
    }
    ABSL_ASSIGN_OR_RETURN(
        IndexingMap indexing,
        PropagateIndexing(*instr, element_operands, regions, mlir_context));
    for (int64_t i : broadcast_operands) {
      // The scale read through the broadcast must be the one of the block that
      // the element belongs to.
      const HloInstruction* broadcast = regions.at(instr->operand(i)).broadcast;
      const HloInstruction* scale = broadcast->operand(0);
      IndexingMap through_data = Compose(
          indexing, ComputeOutputToInputIndexing(broadcast, 0, mlir_context)
                        .indexing_maps[0]
                        .begin()
                        ->map());
      IndexingMap through_amax =
          Compose(element_to_amax.at(owner), regions.at(scale).indexing);
      if (!SameFunction(through_data, through_amax)) {
        return Refuse(absl::StrCat(instr->name(), " reads the scale ",
                                   scale->name(), " of another block"));
      }
    }
    regions[instr] = {Region::kElement, std::move(indexing), owner};
  }

  // S viewed as [rows, cols], cols the product of its k minor physical
  // dimensions for the smallest k that makes every amax a row or a column one.
  const int64_t num_elements = ShapeUtil::ElementsIn(source_shape);
  std::vector<Orientation> orientations;
  for (int64_t k = 0, cols = 1; k < source_shape.dimensions().size(); ++k) {
    cols *= source_shape.dimensions(source_shape.layout().minor_to_major(k));
    const int64_t rows = num_elements / cols;
    IndexingMap view =
        GetBitcastMap(ShapeUtil::MakeShapeWithDescendingLayout(
                          source_shape.element_type(), {rows, cols}),
                      source_shape, mlir_context);
    orientations.clear();
    for (const HloInstruction* amax : amaxes) {
      IndexingMap element_to_amax_in_view =
          Compose(view, element_to_amax.at(amax));
      for (Orientation orientation :
           {Orientation::kRow, Orientation::kColumn}) {
        const int64_t blocked = orientation == Orientation::kRow ? cols : rows;
        if (blocked % analysis.block_size == 0 &&
            ReducesBlocksOf(orientation, element_to_amax_in_view, rows, cols,
                            analysis.block_size, mlir_context)) {
          orientations.push_back(orientation);
          break;
        }
      }
    }
    if (orientations.size() == amaxes.size() &&
        (orientations.size() == 1 || orientations[0] != orientations[1])) {
      analysis.rows = rows;
      analysis.cols = cols;
      analysis.source_indexing = std::move(view);
      break;
    }
  }
  if (analysis.rows == 0) {
    return Refuse(absl::StrCat(
        "the amax reduces block neither rows nor columns of the source; the "
        "first maps from the source to the amax: ",
        ToString(element_to_amax.at(amaxes.front()))));
  }
  // The column emitter's tile is 64 columns wide.
  const int64_t row_tile = amaxes.size() == 2 ? 64 : 32;
  const int64_t col_tile =
      absl::c_linear_search(orientations, Orientation::kColumn) ? 64 : 32;
  if (analysis.rows % row_tile != 0 || analysis.cols % col_tile != 0) {
    return Refuse(absl::StrCat("the source is [", analysis.rows, ", ",
                               analysis.cols, "], not a multiple of [",
                               row_tile, ", ", col_tile, "]"));
  }
  for (auto [amax, orientation] : llvm::zip(amaxes, orientations)) {
    BlockScaledQuantization quantization;
    quantization.orientation = orientation;
    quantization.amax = amax;
    quantization.block_to_amax = Compose(
        BlockToElement(orientation,
                       quantization.BlockCounts(analysis.rows, analysis.cols,
                                                analysis.block_size),
                       analysis.block_size, mlir_context),
        Compose(analysis.source_indexing, element_to_amax.at(amax)));
    analysis.quantizations.push_back(std::move(quantization));
  }
  absl::c_stable_sort(analysis.quantizations,
                      [](const BlockScaledQuantization& lhs,
                         const BlockScaledQuantization& rhs) {
                        return lhs.orientation == Orientation::kRow &&
                               rhs.orientation == Orientation::kColumn;
                      });
  auto quantization_of = [&](const HloInstruction* amax) {
    return absl::c_find_if(analysis.quantizations,
                           [&](const BlockScaledQuantization& quantization) {
                             return quantization.amax == amax;
                           });
  };

  for (int64_t i = 0; i < root->operand_count(); ++i) {
    const HloInstruction* output = root->operand(i);
    const Classification& classification = regions.at(output);
    if (classification.owner == nullptr ||
        (classification.region != Region::kElement &&
         classification.region != Region::kScale)) {
      return Refuse(absl::StrCat("the output ", output->name(),
                                 " is outside the scale and data regions"));
    }
    BlockScaledQuantization& quantization =
        *quantization_of(classification.owner);
    if (classification.region == Region::kScale) {
      quantization.scale_root_indices.push_back(i);
      quantization.scale_root_indexing.push_back(classification.indexing);
      continue;
    }
    const PrimitiveType type = output->shape().element_type();
    if (output->opcode() != HloOpcode::kConvert || !IsDataType(type)) {
      return Refuse(absl::StrCat("the data output ", output->name(),
                                 " is not a convert to F8E4M3FN or F4E2M1FN"));
    }
    if (quantization.data_type != PRIMITIVE_TYPE_INVALID &&
        quantization.data_type != type) {
      return Refuse("the data outputs of one amax have different types");
    }
    quantization.data_type = type;
    quantization.data_root_indices.push_back(i);
    quantization.data_root_indexing.push_back(
        Compose(analysis.source_indexing, classification.indexing));
  }

  for (const HloInstruction* instr : post_order) {
    auto it = regions.find(instr);
    if (it == regions.end() || it->second.broadcast != instr) {
      continue;
    }
    BlockScaledQuantization& quantization = *quantization_of(it->second.owner);
    const HloInstruction* scale = instr->operand(0);
    if (!absl::c_linear_search(quantization.broadcast_scales, scale)) {
      quantization.broadcast_scales.push_back(scale);
      quantization.broadcast_scale_indexing.push_back(
          regions.at(scale).indexing);
    }
  }
  for (BlockScaledQuantization& quantization : analysis.quantizations) {
    if (quantization.data_root_indices.empty() ||
        quantization.scale_root_indices.empty()) {
      return Refuse(absl::StrCat(quantization.amax->name(),
                                 " lacks a data or a scale output"));
    }
    quantization.amax_operand_indexing =
        Compose(analysis.source_indexing,
                regions.at(quantization.amax->operand(0)).indexing);
    if (quantization.orientation == Orientation::kColumn) {
      ABSL_ASSIGN_OR_RETURN(
          quantization.data_in_source_order,
          IsDataInSourceOrder(quantization, *root, mlir_context));
    }
  }
  return analysis;
}

}  // namespace xla::gpu
