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

#include "xla/service/gpu/valid_prefix_extent.h"

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::StatusOr<ValidPrefixExtent> Validate(const HloInstruction& instruction,
                                           int64_t operand_index,
                                           int64_t dimension) {
  if (instruction.opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError(
        "valid-prefix extent requires a fusion instruction");
  }
  if (operand_index < 0 || operand_index >= instruction.operand_count()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "valid-prefix extent must name a valid fusion operand, got ",
        operand_index));
  }
  const Shape& extent_shape = instruction.operand(operand_index)->shape();
  if (extent_shape.element_type() != S32 ||
      !extent_shape.dimensions().empty()) {
    return absl::InvalidArgumentError(
        "valid-prefix extent must refer to an s32 scalar operand");
  }
  if (!instruction.shape().IsArray() ||
      instruction.shape().dimensions().empty()) {
    return absl::InvalidArgumentError(
        "valid-prefix extent requires an array output with rank at least one");
  }
  if (dimension != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "valid-prefix extent only supports output dimension 0, got ",
        dimension));
  }
  return ValidPrefixExtent{
      /*operand_index=*/operand_index,
      /*dimension=*/dimension,
      /*capacity=*/instruction.shape().dimensions(dimension)};
}

}  // namespace

absl::StatusOr<std::optional<ValidPrefixExtent>> GetValidPrefixExtent(
    const HloInstruction& instruction) {
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig config,
                        instruction.backend_config<GpuBackendConfig>());
  if (!config.has_valid_prefix_extent()) {
    return std::nullopt;
  }
  const ValidPrefixExtentConfig& extent = config.valid_prefix_extent();
  return Validate(instruction, extent.operand_index(), extent.dimension());
}

absl::Status SetValidPrefixExtent(HloInstruction* instruction,
                                  int64_t operand_index) {
  if (instruction == nullptr) {
    return absl::InvalidArgumentError(
        "cannot attach a valid-prefix extent to a null instruction");
  }
  ABSL_RETURN_IF_ERROR(Validate(*instruction, operand_index, /*dimension=*/0)
                           .status());
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig config,
                        instruction->backend_config<GpuBackendConfig>());
  ValidPrefixExtentConfig* extent = config.mutable_valid_prefix_extent();
  extent->set_operand_index(operand_index);
  extent->set_dimension(0);
  return instruction->set_backend_config(config);
}

}  // namespace xla::gpu
