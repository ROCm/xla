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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_TRITON_ORIGAMI_COST_MODEL_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_TRITON_ORIGAMI_COST_MODEL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Analytic GEMM cost model backed by AMD's Origami, for pre-selecting Triton
// GEMM configs for the autotuner on ROCm. See docs/origami-xla-cost-model-plan.
class OrigamiDotCostModel {
 public:
  static absl::StatusOr<std::unique_ptr<OrigamiDotCostModel>> Create(
      const se::DeviceDescription& device_description);

  ~OrigamiDotCostModel();

  OrigamiDotCostModel(const OrigamiDotCostModel&) = delete;
  OrigamiDotCostModel& operator=(const OrigamiDotCostModel&) = delete;

  // Configs Origami cannot price are absent from the map, not infinite.
  //
  // Triton's num_warps and num_stages are deliberately not forwarded, so all
  // configs sharing a (block_m, block_n, block_k) get the same estimate.
  // Prefer SelectTopTileShapes over ranking this directly.
  absl::StatusOr<absl::flat_hash_map<TritonGemmConfig, absl::Duration>>
  EstimateRunTimes(const HloDotInstruction* dot,
                   absl::Span<const TritonGemmConfig> configs) const;

  // Returns every config belonging to the `num_tile_shapes` best-estimated
  // tile shapes, preserving input order.
  absl::StatusOr<std::vector<TritonGemmConfig>> SelectTopTileShapes(
      const HloDotInstruction* dot, absl::Span<const TritonGemmConfig> configs,
      int64_t num_tile_shapes) const;

 private:
  class Impl;
  explicit OrigamiDotCostModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

absl::Status IsSupportedByOrigami(
    const HloDotInstruction* dot,
    const se::DeviceDescription& device_description);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_TRITON_ORIGAMI_COST_MODEL_H_
