/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_H_

#include <array>
#include <cstdint>

#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {

// Strategy for performing an all-reduce.
enum class AllReduceStrategy : uint32_t {
  // With one-shot strategy all GPUs gathers and reduces data from all peer
  // GPUs.
  kOneShot,
  // With two-shot strategy each GPU gathers and reduces only a part of the
  // data in the first shot, as a second shot it gathers peer GPUs results to
  // construct a final result.
  kTwoShot,
  // With multimem strategy single GPU uses multimem instructions to perform
  // reduce+broadcast directly on source and destination buffers that were
  // pre-mapped to multimem addresses.
  kMultimem,
};

template <typename Sink>
void AbslStringify(Sink& sink, AllReduceStrategy strategy) {
  switch (strategy) {
    case AllReduceStrategy::kOneShot:
      sink.Append("kOneShot");
      break;
    case AllReduceStrategy::kTwoShot:
      sink.Append("kTwoShot");
      break;
    case AllReduceStrategy::kMultimem:
      sink.Append("kMultimem");
      break;
  }
}

// The maximum number of input pointers that can be passed to the all-reduce
// kernel.
inline constexpr int64_t kMaxNumAllReduceInputPtrs = 8;
inline constexpr int64_t kNumElementsPerThread = 4;

// A pointer to a buffer that does not alias with other buffers.
template <typename U>
using RestrictedPtr = U* __restrict__;

template <typename T>
struct AllReduceKernelParams {
  // Pointer to the input buffer which is symmetric around peer ranks.
  RestrictedPtr<T> symmetric_input_ptrs = nullptr;
  // Local buffer of the device.
  RestrictedPtr<T> input_buffer;
  // Output buffer of the device. Can be the same as the local input buffer in
  // case of aliasing.
  RestrictedPtr<T> output_buffer;
  // Local rank of the device.
  int64_t rank;
  // Number of participating ranks in the all-reduce.
  int64_t num_ranks;
  // Size of tensor on each device.
  int64_t num_elements;
  // Elements to be processed by each rank. This is equal to `num_elements` for
  // one-shot all-reduce and `num_elements / num_ranks` for two-shot
  // all-reduce.
  int64_t num_elements_per_rank;
  // Elements to be processed by each block.
  int64_t num_elements_per_block;
  // Start offset of the rank responsible for accumulating the elements.
  // This is equal to `rank * num_elements_per_rank`.
  int64_t rank_offset;
  // Ranks rotated by `rank` % `num_ranks` to circumvent all GPUs reading from
  // the same location simultaneously. Index 0 is the rank itself.
  std::array<int64_t, kMaxNumAllReduceInputPtrs> rotated_ranks;

  // Legacy: value to be written to the signal flags. Should be different for
  // different invocations of the kernel with the same signal buffer. Kept as
  // a fallback for non-HIP-graph paths and for tests; under HIP graph
  // capture this scalar would be baked into the captured kernel-launch
  // node and reused on every replay, breaking the per-launch uniqueness
  // requirement (see signal_counter below).
  uint32_t signal_value;

  // Device-side per-block monotonic counter, one uint32_t per block.
  // At the start of every kernel launch, the leader thread of each block
  // atomically advances its slot and broadcasts the new value to the rest
  // of the block via shared memory. The result is then used as the
  // *effective* signal_value passed to SyncRemoteBlocks in place of the
  // baked-in `signal_value` scalar above. This makes the kernel safe to
  // capture into a HIP graph: every replay of the captured launch
  // advances the counter fresh, so the in-kernel rendezvous gets a
  // strictly increasing per-launch signal as the kernel's contract
  // requires.
  //
  // Allocation rules: must hold at least one uint32_t per block, must be
  // zero-initialized before the first launch, and must be the same
  // allocation across launches of this thunk on this stream (so the
  // counter is preserved across launches). Each rank has its own
  // allocation in its own device memory; values are NOT shared across
  // ranks. Cross-rank consistency comes from the fact that all ranks
  // launch the kernel the same number of times in lockstep (XLA pmap),
  // so per-rank counter values match across ranks per launch index.
  RestrictedPtr<uint32_t> signal_counter = nullptr;

  // Pointer to the signal flags buffer which is symmetric around peer ranks.
  // TODO(446447767): Remove this once we have a single pointer to symmetric
  // memory.
  RestrictedPtr<uint32_t> symmetric_signal_ptrs = nullptr;

  RestrictedPtr<CollectiveKernelMetadata> metadata;
};

// Defines a trait for the AllReduce kernel that can be used to register
// and look up the kernel in the GPU kernel registry.
template <typename ElementT, xla::ReductionKind ReductionKindT,
          AllReduceStrategy kAllReduceStrategy>
struct AllReduceKernel {
  using KernelType =
      stream_executor::TypedKernel<AllReduceKernelParams<ElementT>>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_H_
