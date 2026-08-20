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

#ifndef XLA_STREAM_EXECUTOR_GPU_ICACHE_FLUSH_H_
#define XLA_STREAM_EXECUTOR_GPU_ICACHE_FLUSH_H_

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/icache_flush_kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// Invalidates the GPU instruction cache across the whole device.
//
// Successive runs of the same kernel - as done when profiling autotuning
// candidates - hit a warm instruction cache, which makes later measurements
// faster than earlier ones for reasons that have nothing to do with the
// candidate being measured. Invalidating the instruction cache before every
// timed run puts all candidates in the same cold-cache state and reduces the
// spread of the measurements.
//
// This is currently only implemented for ROCm; `Create` fails on every other
// platform because no flush kernel is registered there.
class IcacheFlusher {
 public:
  // Loads the flush kernel onto `executor`. Returns a `NotFound` error if the
  // platform of `executor` has no instruction cache flush kernel registered.
  static absl::StatusOr<IcacheFlusher> Create(StreamExecutor* executor);

  // Enqueues the flush kernel on `stream`. Asynchronous: the caller must
  // synchronize, or enqueue the work whose instruction cache misses it wants to
  // observe, on the same stream.
  absl::Status Flush(Stream* stream);

 private:
  IcacheFlusher(IcacheFlushKernel::KernelType kernel, BlockDim block_dim)
      : kernel_(std::move(kernel)), block_dim_(block_dim) {}

  IcacheFlushKernel::KernelType kernel_;
  BlockDim block_dim_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ICACHE_FLUSH_H_
