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

#include "xla/backends/gpu/codegen/emitters/block_scaled_quantize.h"

#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/gpu/codegen/emitters/block_scaled_quantize_analysis.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/emitters/utils.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::MLIRContext;
using ::mlir::Value;
using ::mlir::ValueRange;
using Orientation = BlockScaledQuantization::Orientation;
using HloValueMap =
    absl::flat_hash_map<const HloInstruction*, SmallVector<Value>>;

constexpr int64_t kThreadsPerBlock = 256;
constexpr int64_t kBytesPerLoad = 16;

// Epilogues per quantization: the amax operand, the scale region and the data
// region.
enum EpilogueKind { kAmaxOperand = 0, kScale = 1, kData = 2, kNumEpilogues };

int EpilogueIndex(int quantization_index, EpilogueKind kind) {
  return quantization_index * kNumEpilogues + kind;
}

// Broadcasts `value` from lane `source_lane` (an i32) of the warp.
Value ShuffleFromLane(Value value, Value source_lane, int64_t warp_size,
                      ImplicitLocOpBuilder& b) {
  mlir::Type type = value.getType();
  const int bit_width = type.getIntOrFloatBitWidth();
  Value as_i32 = value;
  if (bit_width < 32) {
    as_i32 =
        mlir::arith::BitcastOp::create(b, b.getIntegerType(bit_width), value);
    as_i32 = mlir::arith::ExtUIOp::create(b, b.getI32Type(), as_i32);
  }
  Value width = mlir::arith::ConstantIntOp::create(b, warp_size, 32);
  Value shuffled = mlir::gpu::ShuffleOp::create(b, as_i32, source_lane, width,
                                                mlir::gpu::ShuffleMode::IDX)
                       .getShuffleResult();
  if (bit_width < 32) {
    shuffled =
        mlir::arith::TruncIOp::create(b, b.getIntegerType(bit_width), shuffled);
    shuffled = mlir::arith::BitcastOp::create(b, type, shuffled);
  }
  return shuffled;
}

// Gives each call of the epilogue `name` in `entry_function` its own copy. The
// inliner keeps a GPU callee with several callers out of line, and the FP8 and
// FP4 conversions inside it then cannot pair at the vector stores.
void CopyEpiloguePerCall(mlir::func::FuncOp entry_function,
                         absl::string_view name) {
  auto module = entry_function->getParentOfType<mlir::ModuleOp>();
  auto epilogue = module.lookupSymbol<mlir::func::FuncOp>(name);
  SmallVector<PureCallOp> calls;
  entry_function.walk([&](PureCallOp call) {
    if (call.getCallee() == llvm::StringRef(name)) {
      calls.push_back(call);
    }
  });
  for (int64_t i = 1; i < calls.size(); ++i) {
    mlir::func::FuncOp copy = epilogue.clone();
    copy.setSymName(absl::StrCat(name, "_copy_", i));
    module.push_back(copy);
    calls[i].setCalleeAttr(mlir::FlatSymbolRefAttr::get(copy.getSymNameAttr()));
  }
}

IndexingMap Compose(const IndexingMap& first, const IndexingMap& second) {
  IndexingMap result = ComposeIndexingMaps(first, second);
  result.Simplify();
  return result;
}

std::vector<IndexingMap::Variable> ThreadDims(
    const LaunchDimensions& launch_dims) {
  return DimVarsFromGPUGrid(
      {static_cast<int64_t>(launch_dims.thread_counts_per_block().x), 1, 1,
       static_cast<int64_t>(launch_dims.block_counts().x),
       static_cast<int64_t>(launch_dims.block_counts().y), 1});
}

// Thread and block ids (and `symbol_sizes`) -> `results`, for the threads
// whose `work_item` is below `num_work_items`.
IndexingMap ThreadMap(const LaunchDimensions& launch_dims,
                      llvm::SmallVector<SymbolicExpr> results,
                      absl::Span<const int64_t> symbol_sizes,
                      SymbolicExpr work_item, int64_t num_work_items,
                      MLIRContext* mlir_context) {
  IndexingMap map(SymbolicMap::Get(mlir_context, 6, symbol_sizes.size(),
                                   std::move(results)),
                  ThreadDims(launch_dims),
                  RangeVarsFromTensorSizes(symbol_sizes), /*rt_vars=*/{});
  map.AddConstraint(work_item, {0, num_work_items - 1});
  map.Simplify();
  return map;
}

// The global index of the thread.
SymbolicExpr GlobalThread(const LaunchDimensions& launch_dims,
                          MLIRContext* mlir_context) {
  return CreateDimExpr(0, mlir_context) +
         (CreateDimExpr(3, mlir_context) +
          CreateDimExpr(4, mlir_context) *
              static_cast<int64_t>(launch_dims.block_counts().x)) *
             kThreadsPerBlock;
}

// Row blocks: `block_size / vector_size` consecutive lanes own one block.
struct RowMapping {
  // Ids[element of the lane's slice] -> the source's (row, col).
  IndexingMap element;
  // Ids -> the block coordinates (row, col / block_size).
  IndexingMap block;
  // Ids -> the lane of the warp that owns the start of the block.
  IndexingMap block_lane;
  // Ids -> the position of the lane among the lanes of its block.
  IndexingMap lane;
};

RowMapping GetRowMapping(const LaunchDimensions& launch_dims, int64_t rows,
                         int64_t cols, int64_t block_size, int64_t vector_size,
                         int64_t warp_size, MLIRContext* mlir_context) {
  const int64_t lanes = block_size / vector_size;
  const int64_t blocks_per_row = cols / block_size;
  SymbolicExpr thread = CreateDimExpr(0, mlir_context);
  SymbolicExpr element = CreateSymbolExpr(0, 6, mlir_context);
  SymbolicExpr block = GlobalThread(launch_dims, mlir_context).floorDiv(lanes);
  SymbolicExpr row = block.floorDiv(blocks_per_row);
  SymbolicExpr block_col = block % blocks_per_row;
  SymbolicExpr lane = thread % lanes;
  const int64_t num_blocks = rows * blocks_per_row;
  auto make_map = [&](llvm::SmallVector<SymbolicExpr> results,
                      absl::Span<const int64_t> symbols) {
    return ThreadMap(launch_dims, std::move(results), symbols, block,
                     num_blocks, mlir_context);
  };
  return RowMapping{
      make_map({row, block_col * block_size + lane * vector_size + element},
               {vector_size}),
      make_map({row, block_col}, {}),
      make_map({(thread.floorDiv(lanes) * lanes) % warp_size}, {}),
      make_map({lane}, {})};
}

// Column blocks, alone or with row blocks: a workgroup owns a tile of
// kTileRows rows and `strips` strips of kStripCols columns of the source; each
// wave loads a band of kBandRows rows of one strip.
constexpr int64_t kTileRows = 128;
constexpr int64_t kStripCols = 64;
constexpr int64_t kBandRows = 32;
constexpr int64_t kBandsPerStrip = kTileRows / kBandRows;
constexpr int64_t kBandThreads = 64;
// In the column phase, kLanesPerColumn lanes own one column of the tile, each
// reading kRowsPerLane rows; a wave owns kStripCols / kBandsPerStrip columns.
constexpr int64_t kLanesPerColumn = 8;
constexpr int64_t kRowsPerLane = kTileRows / kLanesPerColumn;
constexpr int64_t kColumnPasses =
    kStripCols * kLanesPerColumn / (kBandsPerStrip * kBandThreads);

class TileMapping {
 public:
  TileMapping(const LaunchDimensions& launch_dims, int64_t rows, int64_t cols,
              int64_t block_size, int64_t vector_size, int64_t strips,
              int64_t warp_size, MLIRContext* mlir_context)
      : launch_dims_(launch_dims),
        rows_(rows),
        cols_(cols),
        block_size_(block_size),
        vector_size_(vector_size),
        tile_cols_(strips * kStripCols),
        warp_size_(warp_size),
        lanes_per_tile_row_(kStripCols / vector_size),
        rows_per_load_(kBandThreads / lanes_per_tile_row_),
        mlir_context_(mlir_context) {
    SymbolicExpr thread = CreateDimExpr(0, mlir_context);
    SymbolicExpr workgroup =
        CreateDimExpr(3, mlir_context) +
        CreateDimExpr(4, mlir_context) *
            static_cast<int64_t>(launch_dims.block_counts().x);
    thread_ = thread;
    workgroup_ = workgroup;
    tile_row_ = workgroup.floorDiv(cols / tile_cols_);
    tile_col_ = workgroup % (cols / tile_cols_);
    wave_ = thread.floorDiv(kBandThreads);
    band_ = wave_ % kBandsPerStrip;
    strip_ = wave_.floorDiv(kBandsPerStrip);
    lane_ = thread % kBandThreads;
  }

  int64_t tile_cols() const { return tile_cols_; }
  int64_t num_loads() const { return kBandRows / rows_per_load_; }
  int64_t lanes_per_row_block() const { return block_size_ / vector_size_; }
  int64_t lanes_per_column_block() const { return block_size_ / kRowsPerLane; }
  int64_t num_tiles() const {
    return CeilOfRatio(rows_, kTileRows) * (cols_ / tile_cols_);
  }

  // Ids[load, element] -> the source's (row, col).
  IndexingMap Load(SymbolicExpr load, SymbolicExpr element,
                   absl::Span<const int64_t> symbols) const {
    return Map({LoadRow(load), LoadCol() + element}, symbols, BandValid());
  }
  // Ids[load, element] -> the shared tile's (row, col).
  IndexingMap LoadShared(SymbolicExpr load, SymbolicExpr element,
                         absl::Span<const int64_t> symbols) const {
    return Map({band_ * kBandRows + lane_.floorDiv(lanes_per_tile_row_) +
                    load * rows_per_load_,
                strip_ * kStripCols +
                    (lane_ % lanes_per_tile_row_) * vector_size_ + element},
               symbols, BandValid());
  }
  // Ids[load] -> the row block coordinates (row, col / block_size).
  IndexingMap RowBlock(SymbolicExpr load,
                       absl::Span<const int64_t> symbols) const {
    return Map({LoadRow(load), LoadCol().floorDiv(block_size_)}, symbols,
               BandValid());
  }
  // Ids -> the lane's position in its row block.
  IndexingMap RowLane() const {
    return Map({lane_ % lanes_per_row_block()}, {}, BandValid());
  }
  // Ids -> the warp lane that owns the start of the lane's row block.
  IndexingMap RowBlockLane() const {
    return Map(
        {(thread_.floorDiv(lanes_per_row_block()) * lanes_per_row_block()) %
         warp_size_},
        {}, BandValid());
  }
  // Ids[pass, row] -> the source's (row, col) in the column phase.
  IndexingMap ColumnElement(SymbolicExpr pass, SymbolicExpr row,
                            absl::Span<const int64_t> symbols) const {
    return Map({tile_row_ * kTileRows + Sub() * kRowsPerLane + row,
                tile_col_ * tile_cols_ + Column(pass)},
               symbols, ColumnValid());
  }
  // Ids[pass, row] -> the shared tile's (row, col) in the column phase.
  IndexingMap ColumnShared(SymbolicExpr pass, SymbolicExpr row,
                           absl::Span<const int64_t> symbols) const {
    return Map({Sub() * kRowsPerLane + row, Column(pass)}, symbols,
               ColumnValid());
  }
  // Ids[pass] -> the column block coordinates (row / block_size, col).
  IndexingMap ColumnBlock(SymbolicExpr pass,
                          absl::Span<const int64_t> symbols) const {
    return Map({tile_row_ * (kTileRows / block_size_) +
                    Sub().floorDiv(lanes_per_column_block()),
                tile_col_ * tile_cols_ + Column(pass)},
               symbols, ColumnValid());
  }
  // Ids[pass] -> the (block, col) of the column phase in the shared scales.
  IndexingMap ColumnScaleShared(SymbolicExpr pass) const {
    return Map({Sub().floorDiv(lanes_per_column_block()), Column(pass)}, {},
               ColumnValid());
  }
  // Ids[element] -> the (block, col) of a loaded element in the shared scales.
  IndexingMap LoadScaleShared(SymbolicExpr element,
                              absl::Span<const int64_t> symbols) const {
    return Map(
        {band_, strip_ * kStripCols +
                    (lane_ % lanes_per_tile_row_) * vector_size_ + element},
        symbols, BandValid());
  }
  // Ids -> the lane's position in its column block.
  IndexingMap ColumnLane() const {
    return Map({lane_ % lanes_per_column_block()}, {}, ColumnValid());
  }
  // Ids -> the warp lane that owns the start of the lane's column block.
  IndexingMap ColumnBlockLane() const {
    return Map({(thread_.floorDiv(lanes_per_column_block()) *
                 lanes_per_column_block()) %
                warp_size_},
               {}, ColumnValid());
  }

  SymbolicExpr Constant(int64_t value) const {
    return CreateSymbolicConstant(value, mlir_context_);
  }
  SymbolicExpr Symbol(int64_t index) const {
    return CreateSymbolExpr(index, 6, mlir_context_);
  }

 private:
  SymbolicExpr LoadRow(SymbolicExpr load) const {
    return tile_row_ * kTileRows + band_ * kBandRows +
           lane_.floorDiv(lanes_per_tile_row_) + load * rows_per_load_;
  }
  SymbolicExpr LoadCol() const {
    return tile_col_ * tile_cols_ + strip_ * kStripCols +
           (lane_ % lanes_per_tile_row_) * vector_size_;
  }
  SymbolicExpr Sub() const { return lane_ % kLanesPerColumn; }
  SymbolicExpr Column(SymbolicExpr pass) const {
    return wave_ * (kStripCols / kBandsPerStrip) +
           pass * (kBandThreads / kLanesPerColumn) +
           lane_.floorDiv(kLanesPerColumn);
  }
  // The band of 32 rows is inside the source.
  std::pair<SymbolicExpr, int64_t> BandValid() const {
    return {tile_row_ * (kTileRows / kBandRows) + band_, rows_ / kBandRows};
  }
  // The column block is inside the source.
  std::pair<SymbolicExpr, int64_t> ColumnValid() const {
    return {tile_row_ * (kTileRows / block_size_) +
                Sub().floorDiv(lanes_per_column_block()),
            rows_ / block_size_};
  }

  IndexingMap Map(llvm::SmallVector<SymbolicExpr> results,
                  absl::Span<const int64_t> symbols,
                  std::pair<SymbolicExpr, int64_t> valid) const {
    IndexingMap map(
        SymbolicMap::Get(mlir_context_, 6, symbols.size(), std::move(results)),
        ThreadDims(launch_dims_), RangeVarsFromTensorSizes(symbols),
        /*rt_vars=*/{});
    map.AddConstraint(workgroup_, {0, num_tiles() - 1});
    map.AddConstraint(valid.first, {0, valid.second - 1});
    map.Simplify();
    return map;
  }

  LaunchDimensions launch_dims_;
  int64_t rows_;
  int64_t cols_;
  int64_t block_size_;
  int64_t vector_size_;
  int64_t tile_cols_;
  int64_t warp_size_;
  int64_t lanes_per_tile_row_;
  int64_t rows_per_load_;
  MLIRContext* mlir_context_;
  SymbolicExpr thread_;
  SymbolicExpr workgroup_;
  SymbolicExpr tile_row_;
  SymbolicExpr tile_col_;
  SymbolicExpr wave_;
  SymbolicExpr band_;
  SymbolicExpr strip_;
  SymbolicExpr lane_;
};

// Emits the pieces of one quantization that every thread mapping shares.
class QuantizationEmitter {
 public:
  QuantizationEmitter(const BlockScaledQuantizeAnalysis& facts,
                      int quantization_index,
                      const emitters::PartitionedComputations& computations,
                      const emitters::CallTargetProvider& call_targets,
                      mlir::func::FuncOp entry_function,
                      const HloFusionInstruction& fusion)
      : facts_(facts),
        quantization_(facts.quantizations[quantization_index]),
        quantization_index_(quantization_index),
        computations_(computations),
        computation_(computations.FindPartitionedComputation(
            fusion.fused_instructions_computation())),
        call_targets_(call_targets),
        entry_function_(entry_function),
        root_(fusion.fused_expression_root()) {}

  const BlockScaledQuantization& quantization() const { return quantization_; }

  mlir::func::FuncOp Reducer() const {
    return call_targets_(quantization_.amax->to_apply()->root_instruction());
  }

  Value AmaxInit(ImplicitLocOpBuilder& b) const {
    return emitters::ProvideParameter(computation_, quantization_.amax,
                                      /*operand_index=*/1, {}, call_targets_,
                                      entry_function_, b)
        .front();
  }

  // The source at (row, col).
  Value Source(ValueRange element, ImplicitLocOpBuilder& b) const {
    SmallVector<Value> operands(entry_function_.getArguments().take_front(
        computation_.computation().num_parameters()));
    absl::c_copy(
        emitters::ApplyIndexing(facts_.source_indexing, element, {}, b),
        std::back_inserter(operands));
    ValueRange results =
        PureCallOp::create(b, call_targets_(facts_.source), operands)
            .getResults();
    int64_t offset = 0;
    for (const HloInstruction* root :
         computation_.FindSubgraph(facts_.source).roots) {
      if (root == facts_.source) {
        break;
      }
      offset +=
          root->shape().IsTuple() ? root->shape().tuple_shapes().size() : 1;
    }
    return results[offset];
  }

  // Folds the amax operand of `x`, the source at `element`, into `amax`.
  Value ReduceInto(Value amax, Value x, ValueRange element,
                   ImplicitLocOpBuilder& b) const {
    Value operand =
        emitters::EmitEpilogue(EpilogueIndex(quantization_index_, kAmaxOperand),
                               computations_, entry_function_,
                               {{facts_.source, {x}}}, element, b)
            .at(quantization_.amax->operand(0))
            .front();
    return PureCallOp::create(b, Reducer(), {amax, operand}).getResult(0);
  }

  // Evaluates the scale region of the block with amax `amax`. Returns the
  // values the data region reads.
  HloValueMap Scales(Value amax, ValueRange block, ImplicitLocOpBuilder& b) {
    // Lanes outside the domain evaluate it too; clamped, their reads of the
    // region's tensor operands stay in bounds.
    SmallVector<Value> clamped;
    for (auto [coordinate, count] :
         llvm::zip(block, quantization_.BlockCounts(facts_.rows, facts_.cols,
                                                    facts_.block_size))) {
      clamped.push_back(mlir::arith::MinUIOp::create(
          b, coordinate, mlir::arith::ConstantIndexOp::create(b, count - 1)));
    }
    scale_values_ = emitters::EmitEpilogue(
        EpilogueIndex(quantization_index_, kScale), computations_,
        entry_function_, {{quantization_.amax, {amax}}}, clamped, b);
    HloValueMap broadcast_scales;
    for (const HloInstruction* scale : quantization_.broadcast_scales) {
      broadcast_scales[scale] = {scale_values_.at(scale).front()};
    }
    return broadcast_scales;
  }

  // Writes the scale outputs of the block last passed to `Scales`, if
  // `condition` holds (or unconditionally if it is null).
  void WriteScales(ValueRange block, Value condition,
                   SmallVector<Value>& outputs, ImplicitLocOpBuilder& b) const {
    const emitters::PartitionedComputation::Subgraph& epilogue =
        computations_.epilogues()[EpilogueIndex(quantization_index_, kScale)];
    for (int64_t index : quantization_.scale_root_indices) {
      const HloInstruction* scale_root = root_->operand(index);
      int64_t position =
          absl::c_find(epilogue.roots, scale_root) - epilogue.roots.begin();
      SmallVector<Value> indices = emitters::ApplyIndexing(
          epilogue.root_indexing[position], block, {}, b);
      Value value = scale_values_.at(scale_root).front();
      outputs[index] = condition
                           ? PredicatedInsertOp::create(b, condition, value,
                                                        outputs[index], indices)
                                 .getResult()
                           : mlir::tensor::InsertOp::create(
                                 b, value, outputs[index], indices)
                                 .getResult();
    }
  }

  // Quantizes `x`, the source at `element`, into the data outputs among
  // `data_outputs` (in the order of `data_root_indices`).
  SmallVector<Value> WriteData(Value x, ValueRange element,
                               const HloValueMap& broadcast_scales,
                               ValueRange data_outputs,
                               ImplicitLocOpBuilder& b) const {
    HloValueMap injected = broadcast_scales;
    injected[facts_.source] = {x};
    absl::flat_hash_map<const HloInstruction*, ValueRange> data =
        emitters::EmitEpilogue(EpilogueIndex(quantization_index_, kData),
                               computations_, entry_function_, injected,
                               element, b);
    SmallVector<Value> results;
    for (auto [index, indexing, output] :
         llvm::zip(quantization_.data_root_indices,
                   quantization_.data_root_indexing, data_outputs)) {
      SmallVector<Value> indices =
          emitters::ApplyIndexing(indexing, element, {}, b);
      results.push_back(mlir::tensor::InsertOp::create(
          b, data.at(root_->operand(index)).front(), output, indices));
    }
    return results;
  }

  SmallVector<Value> DataOutputs(ValueRange outputs) const {
    SmallVector<Value> result;
    for (int64_t index : quantization_.data_root_indices) {
      result.push_back(outputs[index]);
    }
    return result;
  }

  void SetDataOutputs(ValueRange data, SmallVector<Value>& outputs) const {
    for (auto [index, value] :
         llvm::zip(quantization_.data_root_indices, data)) {
      outputs[index] = value;
    }
  }

  mlir::VectorType SourceVectorType(int64_t size,
                                    ImplicitLocOpBuilder& b) const {
    return mlir::VectorType::get({size},
                                 emitters::PrimitiveTypeToMlirType(
                                     facts_.source->shape().element_type(), b));
  }

 private:
  const BlockScaledQuantizeAnalysis& facts_;
  const BlockScaledQuantization& quantization_;
  int quantization_index_;
  const emitters::PartitionedComputations& computations_;
  const emitters::PartitionedComputation& computation_;
  const emitters::CallTargetProvider& call_targets_;
  mutable mlir::func::FuncOp entry_function_;
  const HloInstruction* root_;
  absl::flat_hash_map<const HloInstruction*, ValueRange> scale_values_;
};

Value ZeroVector(mlir::VectorType type, ImplicitLocOpBuilder& b) {
  return mlir::arith::ConstantOp::create(
      b, type, mlir::cast<mlir::TypedAttr>(b.getZeroAttr(type)));
}

void EmitRow(QuantizationEmitter& q, const RowMapping& mapping,
             int64_t vector_size, int64_t lanes_per_block, int64_t warp_size,
             ValueRange ids, SmallVector<Value>& outputs,
             ImplicitLocOpBuilder& b) {
  // Each lane loads its slice of the source and reduces it.
  ValueRange loaded = emitters::EmitXlaLoopOp(
      b, ids,
      {q.AmaxInit(b), ZeroVector(q.SourceVectorType(vector_size, b), b)},
      mapping.element,
      [&](ImplicitLocOpBuilder& nested_b, ValueRange symbols,
          ValueRange element, ValueRange iter_args) -> SmallVector<Value> {
        Value x = q.Source(element, nested_b);
        return {q.ReduceInto(iter_args[0], x, element, nested_b),
                mlir::vector::InsertOp::create(nested_b, x, iter_args[1],
                                               symbols.front())};
      });
  Value source_vector = loaded[1];

  // The block amax, in every lane of the block.
  Value block_amax =
      ShuffleReduceOp::create(b, q.Reducer(), {loaded[0]}, lanes_per_block / 2)
          .getResult(0);
  Value block_lane = mlir::arith::IndexCastUIOp::create(
      b, b.getI32Type(),
      emitters::ApplyIndexing(mapping.block_lane, ids, {}, b).front());
  block_amax = ShuffleFromLane(block_amax, block_lane, warp_size, b);

  // Every lane evaluates the scale region; the first lane of the block writes
  // the scale outputs.
  SmallVector<Value> block = emitters::ApplyIndexing(mapping.block, ids, {}, b);
  HloValueMap scales = q.Scales(block_amax, block, b);
  Value first_lane = mlir::arith::CmpIOp::create(
      b, mlir::arith::CmpIPredicate::eq,
      emitters::ApplyIndexing(mapping.lane, ids, {}, b).front(),
      mlir::arith::ConstantIndexOp::create(b, 0));
  q.WriteScales(
      block,
      mlir::arith::AndIOp::create(
          b, emitters::CheckConstraints(mapping.block, ids, {}, b), first_lane),
      outputs, b);

  // Every lane quantizes its slice.
  q.SetDataOutputs(
      emitters::EmitXlaLoopOp(
          b, ids, q.DataOutputs(outputs), mapping.element,
          [&](ImplicitLocOpBuilder& nested_b, ValueRange symbols,
              ValueRange element, ValueRange iter_args) -> SmallVector<Value> {
            Value x = mlir::vector::ExtractOp::create(nested_b, source_vector,
                                                      symbols.front());
            return q.WriteData(x, element, scales, iter_args, nested_b);
          }),
      outputs);
}

// The amax of the block, in every lane of it: `lane_amax` reduced over the
// `lanes` lanes of the block, which start at the lane `block_lane` gives.
Value BlockAmax(const QuantizationEmitter& q, Value lane_amax, int64_t lanes,
                const IndexingMap& block_lane, ValueRange ids,
                int64_t warp_size, ImplicitLocOpBuilder& b) {
  if (lanes == 1) {
    return lane_amax;
  }
  Value reduced =
      ShuffleReduceOp::create(b, q.Reducer(), {lane_amax}, lanes / 2)
          .getResult(0);
  Value source_lane = mlir::arith::IndexCastUIOp::create(
      b, b.getI32Type(),
      emitters::ApplyIndexing(block_lane, ids, {}, b).front());
  return ShuffleFromLane(reduced, source_lane, warp_size, b);
}

// Whether the lane `lane` gives is the first of its block and `block` holds.
Value IsBlockWriter(const IndexingMap& block, const IndexingMap& lane,
                    ValueRange ids, ImplicitLocOpBuilder& b) {
  return mlir::arith::AndIOp::create(
      b, emitters::CheckConstraints(block, ids, {}, b),
      mlir::arith::CmpIOp::create(
          b, mlir::arith::CmpIPredicate::eq,
          emitters::ApplyIndexing(lane, ids, {}, b).front(),
          mlir::arith::ConstantIndexOp::create(b, 0)));
}

// Quantizes `values` (one per element of `element`'s symbol) into the data
// outputs of `q`.
void WriteDataLoop(const QuantizationEmitter& q, Value values,
                   const IndexingMap& element, const HloValueMap& scales,
                   ValueRange ids, SmallVector<Value>& outputs,
                   ImplicitLocOpBuilder& b) {
  q.SetDataOutputs(
      emitters::EmitXlaLoopOp(b, ids, q.DataOutputs(outputs), element,
                              [&](ImplicitLocOpBuilder& nested_b,
                                  ValueRange symbols, ValueRange coordinates,
                                  ValueRange iter_args) -> SmallVector<Value> {
                                Value x = mlir::vector::ExtractOp::create(
                                    nested_b, values, symbols.back());
                                return q.WriteData(x, coordinates, scales,
                                                   iter_args, nested_b);
                              }),
      outputs);
}

void EmitTile(QuantizationEmitter* row, QuantizationEmitter& column,
              const TileMapping& mapping, int64_t vector_size,
              int64_t shared_padding, int64_t warp_size, ValueRange ids,
              SmallVector<Value>& outputs, ImplicitLocOpBuilder& b) {
  mlir::VectorType slice_type = column.SourceVectorType(vector_size, b);
  Value shared = AllocateSharedOp::create(
      b, mlir::RankedTensorType::get(
             {kTileRows, mapping.tile_cols() + shared_padding},
             slice_type.getElementType()));

  // Column data in the source's order is written along the rows, from the
  // registers, with the column scales passed through shared memory.
  const bool column_data_along_rows =
      column.quantization().data_in_source_order;
  const bool keep_slices = row != nullptr || column_data_along_rows;

  // Each band of threads loads its 32 rows into the shared tile and, if needed,
  // into registers, and reduces its row slices.
  SmallVector<Value> slices;
  SmallVector<Value> slice_amaxes;
  for (int64_t load = 0; load < mapping.num_loads(); ++load) {
    IndexingMap element =
        mapping.Load(mapping.Constant(load), mapping.Symbol(0), {vector_size});
    IndexingMap shared_element = mapping.LoadShared(
        mapping.Constant(load), mapping.Symbol(0), {vector_size});
    SmallVector<Value> inits = {shared};
    if (keep_slices) {
      inits.push_back(ZeroVector(slice_type, b));
    }
    if (row != nullptr) {
      inits.push_back(row->AmaxInit(b));
    }
    ValueRange loaded = emitters::EmitXlaLoopOp(
        b, ids, inits, element,
        [&](ImplicitLocOpBuilder& nested_b, ValueRange symbols,
            ValueRange coordinates,
            ValueRange iter_args) -> SmallVector<Value> {
          Value x = column.Source(coordinates, nested_b);
          SmallVector<Value> results = {mlir::tensor::InsertOp::create(
              nested_b, x, iter_args[0],
              emitters::ApplyIndexing(shared_element, ids, symbols, nested_b))};
          if (keep_slices) {
            results.push_back(mlir::vector::InsertOp::create(
                nested_b, x, iter_args[1], symbols.front()));
          }
          if (row != nullptr) {
            results.push_back(
                row->ReduceInto(iter_args[2], x, coordinates, nested_b));
          }
          return results;
        });
    shared = loaded[0];
    if (keep_slices) {
      slices.push_back(loaded[1]);
    }
    if (row != nullptr) {
      slice_amaxes.push_back(loaded[2]);
    }
  }

  // The row blocks, from the registers.
  if (row != nullptr) {
    for (int64_t load = 0; load < mapping.num_loads(); ++load) {
      Value block_amax =
          BlockAmax(*row, slice_amaxes[load], mapping.lanes_per_row_block(),
                    mapping.RowBlockLane(), ids, warp_size, b);
      IndexingMap block_map = mapping.RowBlock(mapping.Constant(load), {});
      SmallVector<Value> block = emitters::ApplyIndexing(block_map, ids, {}, b);
      HloValueMap scales = row->Scales(block_amax, block, b);
      row->WriteScales(block,
                       IsBlockWriter(block_map, mapping.RowLane(), ids, b),
                       outputs, b);
      WriteDataLoop(*row, slices[load],
                    mapping.Load(mapping.Constant(load), mapping.Symbol(0),
                                 {vector_size}),
                    scales, ids, outputs, b);
    }
  }

  shared = SyncThreadsOp::create(b, mlir::TypeRange{shared.getType()},
                                 ValueRange{shared})
               .getResult(0);

  // The column blocks, from the shared tile.
  const std::vector<const HloInstruction*>& broadcast_scales =
      column.quantization().broadcast_scales;
  SmallVector<Value> shared_scales;
  if (column_data_along_rows) {
    for (const HloInstruction* scale : broadcast_scales) {
      shared_scales.push_back(AllocateSharedOp::create(
          b,
          mlir::RankedTensorType::get({kBandsPerStrip, mapping.tile_cols()},
                                      emitters::PrimitiveTypeToMlirType(
                                          scale->shape().element_type(), b))));
    }
  }
  mlir::VectorType column_type = column.SourceVectorType(kRowsPerLane, b);
  for (int64_t pass = 0; pass < kColumnPasses; ++pass) {
    IndexingMap element = mapping.ColumnElement(
        mapping.Constant(pass), mapping.Symbol(0), {kRowsPerLane});
    IndexingMap shared_element = mapping.ColumnShared(
        mapping.Constant(pass), mapping.Symbol(0), {kRowsPerLane});
    ValueRange read = emitters::EmitXlaLoopOp(
        b, ids, {column.AmaxInit(b), ZeroVector(column_type, b)}, element,
        [&](ImplicitLocOpBuilder& nested_b, ValueRange symbols,
            ValueRange coordinates,
            ValueRange iter_args) -> SmallVector<Value> {
          Value x = mlir::tensor::ExtractOp::create(
              nested_b, shared,
              emitters::ApplyIndexing(shared_element, ids, symbols, nested_b));
          return {column.ReduceInto(iter_args[0], x, coordinates, nested_b),
                  mlir::vector::InsertOp::create(nested_b, x, iter_args[1],
                                                 symbols.front())};
        });
    Value block_amax =
        BlockAmax(column, read[0], mapping.lanes_per_column_block(),
                  mapping.ColumnBlockLane(), ids, warp_size, b);
    IndexingMap block_map = mapping.ColumnBlock(mapping.Constant(pass), {});
    SmallVector<Value> block = emitters::ApplyIndexing(block_map, ids, {}, b);
    HloValueMap scales = column.Scales(block_amax, block, b);
    column.WriteScales(block,
                       IsBlockWriter(block_map, mapping.ColumnLane(), ids, b),
                       outputs, b);
    if (!column_data_along_rows) {
      WriteDataLoop(column, read[1], element, scales, ids, outputs, b);
      continue;
    }
    SmallVector<Value> at = emitters::ApplyIndexing(
        mapping.ColumnScaleShared(mapping.Constant(pass)), ids, {}, b);
    for (auto [scale, shared_scale] :
         llvm::zip(broadcast_scales, shared_scales)) {
      shared_scale = mlir::tensor::InsertOp::create(b, scales.at(scale).front(),
                                                    shared_scale, at);
    }
  }
  if (!column_data_along_rows) {
    return;
  }

  // The column data along the rows, each element with its column's scale.
  ValueRange synced =
      SyncThreadsOp::create(b, ValueRange(shared_scales).getTypes(),
                            shared_scales)
          .getResults();
  IndexingMap scale_element =
      mapping.LoadScaleShared(mapping.Symbol(0), {vector_size});
  for (int64_t load = 0; load < mapping.num_loads(); ++load) {
    column.SetDataOutputs(
        emitters::EmitXlaLoopOp(
            b, ids, column.DataOutputs(outputs),
            mapping.Load(mapping.Constant(load), mapping.Symbol(0),
                         {vector_size}),
            [&](ImplicitLocOpBuilder& nested_b, ValueRange symbols,
                ValueRange coordinates,
                ValueRange iter_args) -> SmallVector<Value> {
              Value x = mlir::vector::ExtractOp::create(nested_b, slices[load],
                                                        symbols.back());
              SmallVector<Value> at = emitters::ApplyIndexing(
                  scale_element, ids, symbols, nested_b);
              HloValueMap scales;
              for (auto [scale, shared_scale] :
                   llvm::zip(broadcast_scales, synced)) {
                scales[scale] = {mlir::tensor::ExtractOp::create(
                    nested_b, shared_scale, at)};
              }
              return column.WriteData(x, coordinates, scales, iter_args,
                                      nested_b);
            }),
        outputs);
  }
}

}  // namespace

BlockScaledQuantizeEmitter::BlockScaledQuantizeEmitter(
    const HloFusionAnalysis& analysis)
    : analysis_(analysis) {
  MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);
  absl::StatusOr<BlockScaledQuantizeAnalysis> facts = Analyze(&mlir_context);
  if (!facts.ok()) {
    return;
  }
  mapping_ = facts->column() != nullptr ? Mapping::kTile : Mapping::kRow;
  has_rows_ = facts->row() != nullptr;
  rows_ = facts->rows;
  cols_ = facts->cols;
  block_size_ = facts->block_size;
  vector_size_ = kBytesPerLoad / primitive_util::ByteWidth(
                                     facts->source->shape().element_type());
  lanes_per_block_ = block_size_ / vector_size_;
  // Two strips where the columns and the shared memory allow them; the row
  // padding of the shared tile is 4 bytes.
  const int64_t two_strip_tile_bytes =
      kTileRows * (2 * kStripCols * kBytesPerLoad / vector_size_ + 4);
  if (mapping_ == Mapping::kTile && cols_ % (2 * kStripCols) == 0 &&
      two_strip_tile_bytes <=
          analysis.device_info().shared_memory_per_block()) {
    strips_ = 2;
  }
}

absl::StatusOr<BlockScaledQuantizeAnalysis> BlockScaledQuantizeEmitter::Analyze(
    MLIRContext* mlir_context) const {
  return AnalyzeBlockScaledQuantize(
      *analysis_.fusion_root(0).instruction().parent(), mlir_context);
}

LaunchDimensions BlockScaledQuantizeEmitter::launch_dimensions() const {
  int64_t num_workgroups = 1;
  int64_t threads = kThreadsPerBlock;
  switch (mapping_) {
    case Mapping::kNone:
      break;
    case Mapping::kRow:
      num_workgroups =
          CeilOfRatio(rows_ * cols_ / vector_size_, kThreadsPerBlock);
      break;
    case Mapping::kTile:
      num_workgroups =
          CeilOfRatio(rows_, kTileRows) * (cols_ / (strips_ * kStripCols));
      threads = strips_ * kBandsPerStrip * kBandThreads;
      break;
  }
  std::array<uint64_t, 2> grid = MaybeSplitGridDimensionX(
      threads, num_workgroups, analysis_.device_info());
  return LaunchDimensions(se::BlockDim(grid[0], grid[1], 1),
                          se::ThreadDim(threads, 1, 1));
}

std::optional<IndexingMap>
BlockScaledQuantizeEmitter::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  absl::StatusOr<BlockScaledQuantizeAnalysis> facts = Analyze(mlir_context);
  if (!facts.ok()) {
    return std::nullopt;
  }
  const int64_t warp_size = WarpSize(analysis_.device_info());
  for (const BlockScaledQuantization& quantization : facts->quantizations) {
    // Ids[symbols] -> the source's (row, col) and the block coordinates, and
    // the lane that writes the block's scales.
    IndexingMap element = IndexingMap::GetUndefined();
    IndexingMap block = IndexingMap::GetUndefined();
    IndexingMap writer = IndexingMap::GetUndefined();
    if (mapping_ == Mapping::kRow) {
      RowMapping mapping =
          GetRowMapping(launch_dimensions(), rows_, cols_, block_size_,
                        vector_size_, warp_size, mlir_context);
      element = mapping.element;
      block = mapping.block;
      writer = mapping.lane;
    } else {
      TileMapping mapping(launch_dimensions(), rows_, cols_, block_size_,
                          vector_size_, strips_, warp_size, mlir_context);
      if (quantization.orientation == Orientation::kRow) {
        element = mapping.Load(mapping.Symbol(0), mapping.Symbol(1),
                               {mapping.num_loads(), vector_size_});
        block = mapping.RowBlock(mapping.Symbol(0), {mapping.num_loads()});
        writer = mapping.RowLane();
      } else if (quantization.data_in_source_order) {
        element = mapping.Load(mapping.Symbol(0), mapping.Symbol(1),
                               {mapping.num_loads(), vector_size_});
        block = mapping.ColumnBlock(mapping.Symbol(0), {kColumnPasses});
        writer = mapping.ColumnLane();
      } else {
        element = mapping.ColumnElement(mapping.Symbol(0), mapping.Symbol(1),
                                        {kColumnPasses, kRowsPerLane});
        block = mapping.ColumnBlock(mapping.Symbol(0), {kColumnPasses});
        writer = mapping.ColumnLane();
      }
    }
    for (auto [index, root_indexing] : llvm::zip(
             quantization.data_root_indices, quantization.data_root_indexing)) {
      if (index == root_index) {
        return Compose(element, root_indexing);
      }
    }
    for (auto [index, root_indexing] :
         llvm::zip(quantization.scale_root_indices,
                   quantization.scale_root_indexing)) {
      if (index == root_index) {
        IndexingMap map = ComposeIndexingMaps(
            block,
            ComposeIndexingMaps(quantization.block_to_amax, root_indexing));
        map.AddConstraint(writer.GetSymbolicMap().GetResult(0), {0, 0});
        map.Simplify();
        return map;
      }
    }
  }
  return std::nullopt;
}

std::optional<std::vector<IndexingMap>>
BlockScaledQuantizeEmitter::ComputeThreadIdToInputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  return std::nullopt;
}

std::vector<emitters::EpilogueSpecification>
BlockScaledQuantizeEmitter::GetEpilogues(const HloFusionInstruction& fusion,
                                         MLIRContext* mlir_context) const {
  absl::StatusOr<BlockScaledQuantizeAnalysis> facts = Analyze(mlir_context);
  if (!facts.ok()) {
    return {};
  }
  const HloInstruction* root = fusion.fused_expression_root();
  std::vector<emitters::EpilogueSpecification> epilogues;
  for (const BlockScaledQuantization& quantization : facts->quantizations) {
    emitters::EpilogueSpecification amax_operand;
    amax_operand.heroes = {facts->source};
    amax_operand.roots = {quantization.amax->operand(0)};
    amax_operand.index_ranges = {facts->rows, facts->cols};
    amax_operand.root_indexing = {quantization.amax_operand_indexing};
    epilogues.push_back(std::move(amax_operand));

    emitters::EpilogueSpecification scale;
    scale.heroes = {quantization.amax};
    scale.index_ranges =
        quantization.BlockCounts(facts->rows, facts->cols, facts->block_size);
    auto add_scale_root = [&](const HloInstruction* instr,
                              const IndexingMap& indexing) {
      if (absl::c_linear_search(scale.roots, instr)) {
        return;
      }
      scale.roots.push_back(instr);
      scale.root_indexing.push_back(
          Compose(quantization.block_to_amax, indexing));
    };
    for (auto [index, indexing] : llvm::zip(quantization.scale_root_indices,
                                            quantization.scale_root_indexing)) {
      add_scale_root(root->operand(index), indexing);
    }
    for (auto [instr, indexing] :
         llvm::zip(quantization.broadcast_scales,
                   quantization.broadcast_scale_indexing)) {
      add_scale_root(instr, indexing);
    }
    epilogues.push_back(std::move(scale));

    emitters::EpilogueSpecification data;
    data.heroes = {facts->source};
    absl::c_copy(quantization.broadcast_scales,
                 std::back_inserter(data.heroes));
    for (int64_t index : quantization.data_root_indices) {
      data.roots.push_back(root->operand(index));
    }
    data.index_ranges = {facts->rows, facts->cols};
    data.root_indexing = quantization.data_root_indexing;
    epilogues.push_back(std::move(data));
  }
  return epilogues;
}

absl::Status BlockScaledQuantizeEmitter::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  MLIRContext* mlir_context = entry_function.getContext();
  ABSL_ASSIGN_OR_RETURN(BlockScaledQuantizeAnalysis facts,
                        Analyze(mlir_context));
  ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
  b.setInsertionPointToStart(entry_function.addEntryBlock());
  SmallVector<Value> ids = EmitThreadAndBlockIds(b);
  SmallVector<Value> outputs(entry_function.getArguments().drop_front(
      fusion.fused_parameters().size()));
  const int64_t warp_size = WarpSize(analysis_.device_info());
  std::vector<QuantizationEmitter> quantizations;
  for (int index = 0; index < static_cast<int>(facts.quantizations.size());
       ++index) {
    quantizations.emplace_back(facts, index, computations, call_targets,
                               entry_function, fusion);
  }
  if (mapping_ == Mapping::kRow) {
    EmitRow(quantizations.front(),
            GetRowMapping(launch_dimensions(), rows_, cols_, block_size_,
                          vector_size_, warp_size, mlir_context),
            vector_size_, lanes_per_block_, warp_size, ids, outputs, b);
  } else {
    const int64_t element_bytes =
        primitive_util::ByteWidth(facts.source->shape().element_type());
    EmitTile(has_rows_ ? &quantizations.front() : nullptr, quantizations.back(),
             TileMapping(launch_dimensions(), rows_, cols_, block_size_,
                         vector_size_, strips_, warp_size, mlir_context),
             vector_size_, /*shared_padding=*/4 / element_bytes, warp_size, ids,
             outputs, b);
  }
  mlir::func::ReturnOp::create(b, outputs);
  for (int index = 0; index < static_cast<int>(facts.quantizations.size());
       ++index) {
    CopyEpiloguePerCall(
        entry_function,
        computations.epilogues()[EpilogueIndex(index, kData)].name);
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
