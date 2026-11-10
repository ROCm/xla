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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_HIPBLASLT_STREAM_K_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_HIPBLASLT_STREAM_K_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace xla {
namespace gpu {

// Returns the StreamK parameter of a hipBLASLt (TensileLite) kernel or
// solution name: the value of its "SK<n>" token, e.g. 5 for
// "..._SVW8_SK5_SKFTR0_...". 0 is a data-parallel kernel, any other value a
// stream-K one. Returns nullopt if the name has no such token.
std::optional<int> HipblasLtStreamKMode(absl::string_view name);

// Returns the positions in `names` of the solutions to keep when stream-K
// solutions are excluded: every name that is not stream-K (including names
// without an SK token). If every solution is stream-K, returns all positions so
// that a GEMM always keeps a candidate.
std::vector<int> SelectNonStreamKSolutions(absl::Span<const std::string> names);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_HIPBLASLT_STREAM_K_H_
