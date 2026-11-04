/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_LOOP_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_LOOP_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/loop_kernel_emitter.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"

namespace xla {
namespace gpu {

// Generic loop fusion.
class LoopFusion final : public MlirKernelEmitter {
 public:
  explicit LoopFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis),
        unroll_factor_(ComputeLoopFusionConfig(analysis)),
        dynamic_row_extent_(FindDynamicRowExtent(analysis)) {}
  LaunchDimensions launch_dimensions() const override;
  int unroll_factor() const override { return unroll_factor_; }

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

 private:
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateMLIRModule(
      mlir::MLIRContext& context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const override;

  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  WorkDimensions GetWorkDimensions() const;

  // Recognizes the masked MoE epilogue pattern at the root of a single-output
  // loop fusion,
  //   select(compare(iota(dim=d), broadcast(n), LT), value, broadcast(c))
  // (or the mirrored compare), where `n` is a scalar integer fusion parameter
  // and `d` is the most-major dimension of the root. When the module sets
  // xla_gpu_experimental_dynamic_row_extent, returns the extent the kernel
  // is guarded on; otherwise std::nullopt.
  static std::optional<emitters::DynamicRowExtent> FindDynamicRowExtent(
      const HloFusionAnalysis& analysis);

 private:
  const HloFusionAnalysis& analysis_;
  int unroll_factor_;
  std::optional<emitters::DynamicRowExtent> dynamic_row_extent_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_LOOP_H_
