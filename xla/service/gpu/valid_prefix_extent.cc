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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

bool PreservesValidPrefix(const HloInstruction& user, int64_t operand_index) {
  const Shape& input = user.operand(operand_index)->shape();
  const Shape& output = user.shape();
  if (!input.IsArray() || input.dimensions().empty() || !output.IsArray() ||
      output.dimensions().empty() ||
      input.dimensions(0) != output.dimensions(0)) {
    return false;
  }
  if (user.IsElementwise()) {
    return input.dimensions() == output.dimensions();
  }
  switch (user.opcode()) {
    case HloOpcode::kCopy:
    case HloOpcode::kReshape:
      return operand_index == 0;
    case HloOpcode::kBroadcast:
      return operand_index == 0 && !user.dimensions().empty() &&
             user.dimensions(0) == 0;
    default:
      return false;
  }
}

absl::Status ValidateMarker(const HloInstruction& marker) {
  if (!marker.IsCustomCall(kValidPrefixCallTarget)) {
    return absl::InvalidArgumentError("expected a valid-prefix marker");
  }
  if (marker.operand_count() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("valid-prefix marker requires two operands, got ",
                     marker.operand_count()));
  }
  const Shape& data_shape = marker.operand(0)->shape();
  if (!data_shape.IsArray() || data_shape.dimensions().empty() ||
      !marker.shape().IsArray() || marker.shape().dimensions().empty() ||
      data_shape.dimensions() != marker.shape().dimensions() ||
      data_shape.element_type() != marker.shape().element_type()) {
    return absl::InvalidArgumentError(
        "valid-prefix marker requires matching non-scalar array data");
  }
  const Shape& extent_shape = marker.operand(1)->shape();
  if (extent_shape.element_type() != S32 ||
      !extent_shape.dimensions().empty()) {
    return absl::InvalidArgumentError(
        "valid-prefix extent must be an s32 scalar");
  }
  if (marker.operand(1)->opcode() != HloOpcode::kParameter) {
    return absl::InvalidArgumentError(
        "valid-prefix extent must be a fusion parameter");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::optional<ValidPrefixExtent>> GetValidPrefixExtent(
    const HloInstruction& instruction) {
  if (instruction.opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError(
        "valid-prefix extent requires a fusion instruction");
  }
  if (!instruction.shape().IsArray() ||
      instruction.shape().dimensions().empty()) {
    return std::nullopt;
  }

  using ExtentParameters = absl::flat_hash_set<int64_t>;
  absl::flat_hash_map<const HloInstruction*, ExtentParameters> extents;
  std::vector<const HloInstruction*> worklist;
  for (const HloInstruction* fused :
       instruction.fused_instructions_computation()->instructions()) {
    if (!fused->IsCustomCall(kValidPrefixCallTarget)) {
      continue;
    }
    absl::Status status = ValidateMarker(*fused);
    if (!status.ok()) {
      return status;
    }
    extents[fused].insert(fused->operand(1)->parameter_number());
    worklist.push_back(fused);
  }

  while (!worklist.empty()) {
    const HloInstruction* current = worklist.back();
    worklist.pop_back();
    for (const HloInstruction* user : current->users()) {
      bool preserves = false;
      for (int64_t i = 0; i < user->operand_count(); ++i) {
        preserves |=
            user->operand(i) == current && PreservesValidPrefix(*user, i);
      }
      if (!preserves) {
        continue;
      }
      ExtentParameters& user_extents = extents[user];
      size_t old_size = user_extents.size();
      user_extents.insert(extents[current].begin(), extents[current].end());
      if (user_extents.size() != old_size) {
        worklist.push_back(user);
      }
    }
  }

  const ExtentParameters& root_extents =
      extents[instruction.fused_expression_root()];
  if (root_extents.size() != 1) {
    return std::nullopt;
  }
  int64_t operand_index = *root_extents.begin();
  if (operand_index < 0 || operand_index >= instruction.operand_count()) {
    return absl::InvalidArgumentError(
        absl::StrCat("valid-prefix extent parameter is not a fusion operand: ",
                     operand_index));
  }
  return ValidPrefixExtent{operand_index, instruction.shape().dimensions(0)};
}

}  // namespace xla::gpu
