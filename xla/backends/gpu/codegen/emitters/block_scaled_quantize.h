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

#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_BLOCK_SCALED_QUANTIZE_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_BLOCK_SCALED_QUANTIZE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/block_scaled_quantize_analysis.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"

namespace xla::gpu {

// Rows only: the lanes of a block reduce their 16-byte slices by shuffles.
// With columns: 4 waves per 64 columns stage a 128 x 64 or 128 x 128 tile;
// rows from registers, columns from shared memory, 8 lanes per column.
class BlockScaledQuantizeEmitter : public MlirKernelEmitter {
 public:
  explicit BlockScaledQuantizeEmitter(const HloFusionAnalysis& analysis);

  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

 protected:
  std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const override;

  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

 private:
  absl::StatusOr<BlockScaledQuantizeAnalysis> Analyze(
      mlir::MLIRContext* mlir_context) const;

  enum class Mapping { kNone, kRow, kTile };

  const HloFusionAnalysis& analysis_;
  Mapping mapping_ = Mapping::kNone;
  bool has_rows_ = false;
  int64_t rows_ = 0;
  int64_t cols_ = 0;
  int64_t block_size_ = 0;
  // Source elements per lane and lanes per row block.
  int64_t vector_size_ = 0;
  int64_t lanes_per_block_ = 0;
  // 64-column strips of the tile.
  int64_t strips_ = 1;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_BLOCK_SCALED_QUANTIZE_H_
