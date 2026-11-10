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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TOPK_CONFIG_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TOPK_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

// Shapes, limits and launch configurations of the Triton TopK kernels.
//
// The routing rule in TopkSpecializer and the thunk emitter both use this
// library, so a TopK is only routed to Triton if the emitter can build a kernel
// for it.

namespace xla::gpu {

// What a Triton TopK kernel computes: the `k` largest elements of each of the
// `batch` rows of `n` elements, in the given comparison order.
struct TopKKernelSpec {
  PrimitiveType input_type;
  ComparisonOrder order;
  int64_t batch;
  int64_t n;
  int64_t k;

  std::string ToString() const;
};

// The Triton TopK kernels. A TopK whose rows fit one program runs kSingle;
// longer rows run kChunk, then kMerge.
enum class TopKKernelKind {
  // Each program selects the top `k` of `rows_per_program` whole rows.
  kSingle,
  // Each program selects the top `k` keys of one chunk of `block_n` elements
  // of a row and writes them to the scratch output.
  kChunk,
  // Each program selects the top `k` of the `num_chunks * k` candidate keys
  // of `rows_per_program` rows in the scratch output.
  kMerge,
};

// Launch configuration of one Triton TopK kernel, where each program owns
// `rows_per_program` rows of `block_n` elements.
struct TopKKernelConfig {
  // Row (or chunk) length rounded up to a power of two; positions past the end
  // are padding.
  int64_t block_n;
  int64_t rows_per_program;
  int64_t num_warps;
  int64_t num_programs;
  // Chunks of `block_n` elements per row: 1 for kSingle and kMerge.
  int64_t num_chunks = 1;

  std::string ToString() const;
};

// Reads the spec of a "TopK" or "__gpu$TopK" custom call of rank 1 or 2.
absl::StatusOr<TopKKernelSpec> GetTopKKernelSpec(
    const HloCustomCallInstruction& topk);

// Checks what the kernels support regardless of the device: f32 or bf16, TOTAL
// or PARTIAL order, 1 <= k <= 128, n < 2^31 and k/n < 0.85.
absl::Status CheckTopKKernelLimits(const TopKKernelSpec& spec);

// Picks the configuration of the first kernel for `spec` on `device`: kSingle
// if a row fits one program, otherwise kChunk (`num_chunks` > 1) with a merge
// step that fits one program per row. Returns nullopt if neither fits the
// device. Register and LDS use are estimates; the compiled kernels are checked
// again by the emitter.
std::optional<TopKKernelConfig> GetTopKKernelConfig(
    const TopKKernelSpec& spec, const se::DeviceDescription& device);

// The merge step of a chunked `config`: a TopK of the `num_chunks * k`
// candidate keys of each row, and its kMerge configuration (nullopt if it does
// not fit the device).
TopKKernelSpec GetTopKMergeSpec(const TopKKernelSpec& spec,
                                const TopKKernelConfig& config);
std::optional<TopKKernelConfig> GetTopKMergeConfig(
    const TopKKernelSpec& spec, const TopKKernelConfig& config,
    const se::DeviceDescription& device);

// Elements of the u64 scratch output (the candidate keys) of a chunked
// `config`; 0 if the TopK needs no scratch.
int64_t GetTopKScratchElements(const TopKKernelSpec& spec,
                               const TopKKernelConfig& config);

// Returns true if the measured speed table of `device` says that the Triton
// kernel is faster than the default TopK path for `spec`. Devices without a
// table return false.
bool IsTritonTopKFaster(const TopKKernelSpec& spec,
                        const se::DeviceDescription& device);

// The backend config of a __gpu$TopK custom call computed by the Triton TopK
// kernels carries `impl = "triton"`.
bool IsTritonTopK(const HloCustomCallInstruction& topk);
absl::Status MarkAsTritonTopK(HloCustomCallInstruction& topk);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TOPK_CONFIG_H_
