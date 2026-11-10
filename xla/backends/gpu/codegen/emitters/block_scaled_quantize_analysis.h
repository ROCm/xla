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

#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_BLOCK_SCALED_QUANTIZE_ANALYSIS_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_BLOCK_SCALED_QUANTIZE_ANALYSIS_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// One orientation of a block-scaled quantization of the source value S, viewed
// as a row-major [rows, cols] matrix: a row block is `block_size` consecutive
// elements of one row, a column block `block_size` consecutive elements of one
// column.
struct BlockScaledQuantization {
  enum class Orientation { kRow, kColumn };

  Orientation orientation = Orientation::kRow;
  // The reduce computing the block amax.
  const HloInstruction* amax = nullptr;
  // Operand indices in the root tuple of the quantized data (converts to an
  // FP8 or FP4 type) and of the scale-region outputs.
  std::vector<int64_t> data_root_indices;
  std::vector<int64_t> scale_root_indices;
  // Scale-region values that the data region reads through a broadcast.
  std::vector<const HloInstruction*> broadcast_scales;
  PrimitiveType data_type = PRIMITIVE_TYPE_INVALID;
  // A column quantization whose data outputs keep the order of S, so that a
  // row of S, not a block, is contiguous in them.
  bool data_in_source_order = false;

  // (row, col) of S -> the index of the amax operand and of each data root.
  IndexingMap amax_operand_indexing = IndexingMap::GetUndefined();
  std::vector<IndexingMap> data_root_indexing;
  // Block coordinates -> the index of `amax`. The block coordinates are
  // (row, col / block_size) for a row quantization and (row / block_size, col)
  // for a column quantization.
  IndexingMap block_to_amax = IndexingMap::GetUndefined();
  // Index of `amax` -> the index of each scale root and broadcast scale.
  std::vector<IndexingMap> scale_root_indexing;
  std::vector<IndexingMap> broadcast_scale_indexing;

  // The block coordinates' sizes, [rows, cols / block_size] or
  // [rows / block_size, cols].
  std::vector<int64_t> BlockCounts(int64_t rows, int64_t cols,
                                   int64_t block_size) const;
};

// The facts the block-scaled quantize emitter needs about a fused computation
// of the `__block_scaled_quantize` kind.
struct BlockScaledQuantizeAnalysis {
  // The value whose blocks are quantized; its producers are the prologue.
  const HloInstruction* source = nullptr;
  // S viewed as [rows, cols] in its physical order, cols the product of its
  // minor dimensions that the blocks need.
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t block_size = 0;
  // (row, col) -> the index of S.
  IndexingMap source_indexing = IndexingMap::GetUndefined();
  // One or two quantizations, the row one first.
  std::vector<BlockScaledQuantization> quantizations;
  // The parameters the prologue reads.
  std::vector<const HloInstruction*> prologue_parameters;

  const BlockScaledQuantization* row() const;
  const BlockScaledQuantization* column() const;
};

// Analyzes `computation`, the fused computation of a block-scaled quantize
// fusion. Returns InvalidArgument naming the reason if the emitter cannot emit
// it. The indexing maps live in `mlir_context`.
absl::StatusOr<BlockScaledQuantizeAnalysis> AnalyzeBlockScaledQuantize(
    const HloComputation& computation, mlir::MLIRContext* mlir_context);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_BLOCK_SCALED_QUANTIZE_ANALYSIS_H_
