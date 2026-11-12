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

#ifndef XLA_SERVICE_GPU_VALID_PREFIX_EXTENT_H_
#define XLA_SERVICE_GPU_VALID_PREFIX_EXTENT_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {

inline constexpr absl::string_view kValidPrefixCallTarget = "xla.valid_prefix";

// Describes a valid prefix derived from HLO dataflow at the point of use.
//
// Only rows [0, extent) are semantically observable. Code generation may omit
// loads, computation, and stores for the remaining rows. The association is
// represented by an xla.valid_prefix(data, extent) custom call inside the
// fusion, so HLO operand rewrites update it together with normal dataflow.
struct ValidPrefixExtent {
  int64_t operand_index;
  int64_t capacity;
};

// Returns the unique dimension-0 valid prefix reaching a fusion root.
absl::StatusOr<std::optional<ValidPrefixExtent>> GetValidPrefixExtent(
    const HloInstruction& instruction);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_VALID_PREFIX_EXTENT_H_
