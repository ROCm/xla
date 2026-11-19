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

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/autotuner/triton/origami_cost_model.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {
absl::Status NotBuilt() {
  return absl::UnimplementedError("XLA was not built with Origami support");
}
}  // namespace

class OrigamiDotCostModel::Impl {};

OrigamiDotCostModel::OrigamiDotCostModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

OrigamiDotCostModel::~OrigamiDotCostModel() = default;

absl::StatusOr<std::unique_ptr<OrigamiDotCostModel>>
OrigamiDotCostModel::Create(const se::DeviceDescription&) {
  return NotBuilt();
}

absl::StatusOr<absl::flat_hash_map<TritonGemmConfig, absl::Duration>>
OrigamiDotCostModel::EstimateRunTimes(
    const HloDotInstruction*, absl::Span<const TritonGemmConfig>) const {
  return NotBuilt();
}

absl::StatusOr<std::vector<TritonGemmConfig>>
OrigamiDotCostModel::SelectTopTileShapes(const HloDotInstruction*,
                                         absl::Span<const TritonGemmConfig>,
                                         int64_t) const {
  return NotBuilt();
}

absl::Status IsSupportedByOrigami(const HloDotInstruction*,
                                  const se::DeviceDescription&) {
  return NotBuilt();
}

}  // namespace xla::gpu
