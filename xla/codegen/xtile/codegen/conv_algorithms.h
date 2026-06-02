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

#ifndef XLA_CODEGEN_XTILE_CODEGEN_CONV_ALGORITHMS_H_
#define XLA_CODEGEN_XTILE_CODEGEN_CONV_ALGORITHMS_H_

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {
namespace xtile {

// Returns the type to use for accumulation for the given `conv` instruction.
absl::StatusOr<::mlir::Type> GetConvAccumulatorType(
    mlir::ImplicitLocOpBuilder& b, const HloConvolutionInstruction& conv);

// Canonicalizes a rank-N conv kernel tile to a rank-2 `[K, N]` matrix, where
// K = product of (kernel spatial axes) * c_in, and N = c_out. 
absl::StatusOr<TensorValue> CanonicalizeConvKernelToKN(
    mlir::ImplicitLocOpBuilder& b, TensorValue kernel_tile,
    const HloConvolutionInstruction& conv);

// Canonicalizes a rank-N conv accumulator tile to a rank-2 `[M, N]` matrix,
// where M = batch * product(output spatial axes), and N = c_out. 
absl::StatusOr<TensorValue> CanonicalizeConvAccToMN(
    mlir::ImplicitLocOpBuilder& b, ::mlir::Value acc,
    const HloConvolutionInstruction& conv);

// Inverse of CanonicalizeConvAccToMN
absl::StatusOr<::mlir::Value> RestoreConvAccFromMN(
    mlir::ImplicitLocOpBuilder& b, ::mlir::Value acc_2d,
    const HloConvolutionInstruction& conv);

}  // namespace xtile
}  // namespace xla

#endif  // XLA_CODEGEN_XTILE_CODEGEN_CONV_ALGORITHMS_H_
