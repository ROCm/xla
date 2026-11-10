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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TOPK_KERNEL_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TOPK_KERNEL_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/triton/topk_config.h"

namespace xla::gpu {

// Builds the Triton IR of the TopK kernel `kind` for `spec` with `config`:
//  - kSingle: each program selects the top `spec.k` elements of
//    `config.rows_per_program` whole rows. Arguments (input, values, indices).
//  - kChunk: each program selects the top `spec.k` keys of one chunk of
//    `config.block_n` elements and writes them to the scratch buffer, `k` per
//    chunk in row and chunk order. Arguments (input, values, indices, scratch).
//  - kMerge: `spec` and `config` are those of the merge step
//    (GetTopKMergeSpec, GetTopKMergeConfig); each program selects the top
//    `spec.k` of the candidate keys of its rows in the scratch buffer.
//    Arguments as kChunk.
// The arguments are the custom call's buffers in their order.
//
// Each element gets the 64-bit priority key of the stable TopK adapter
// (openxla/xla#43913): an order-preserving form of the value in the high 32
// bits and a reversed index in the low 32 bits, bit for bit as upstream's
// pack in TopkSpecializer. All keys of a row differ, so `k` rounds of "take the
// row maximum and remove it" give the stable order. Positions past the end of
// a row get key 0, which is below every real key. Chunks use the index within
// the whole row, so the merge step decodes the original positions.
//
// `context` must have the Triton, arith and SCF dialects loaded.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> BuildTopKKernel(
    const TopKKernelSpec& spec, const TopKKernelConfig& config,
    TopKKernelKind kind, absl::string_view name, mlir::MLIRContext& context);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TOPK_KERNEL_H_
