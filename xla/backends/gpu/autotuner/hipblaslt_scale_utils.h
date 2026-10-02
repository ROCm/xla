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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_HIPBLASLT_SCALE_UTILS_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_HIPBLASLT_SCALE_UTILS_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/decision.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Whether a native scaled-dot can use the initial, unpadded gfx950 32x8 scale
// path. Other shapes and devices retain their existing linear-scale candidates.
// Data operands must both be E4M3FN or both be E2M1FN with packed E(4) layouts.
Decision CanUseHipblasLtScale32x8(
    const HloInstruction& scaled_dot,
    const stream_executor::GpuComputeCapability& gpu_version);

// Materializes hipBLASLt's 32x8 scale byte order in a loop fusion, leaving the
// original scale available to other users. Accepts row-major [MN, K/32] or
// [1, MN, K/32] E8M0 scales with MN divisible by 32 and K/32 divisible by 8.
// The returned buffer has the same shape, but must only be consumed using
// HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT.
absl::StatusOr<HloInstruction*> SwizzleHipblasLtScale32x8(
    HloInstruction* scale);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_HIPBLASLT_SCALE_UTILS_H_
