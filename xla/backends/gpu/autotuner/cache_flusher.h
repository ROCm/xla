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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_CACHE_FLUSHER_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_CACHE_FLUSHER_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/gpu/cache_flush_kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"

namespace xla {
namespace gpu {

// Evicts the GPU cache hierarchy by streaming a read over a scratch buffer
// larger than the last level cache.
//
// The autotuner profiles every candidate of an instruction against the same
// input buffers. On hardware with a large memory-side last level cache, such
// as the 256 MB Infinity Cache on MI300 and MI350, that data stays resident
// across candidates and every candidate is measured reading out of cache
// rather than out of HBM. No coherence operation reaches a memory-side cache,
// so the only way to evict it is capacity pressure, which is what this does.
//
// Unlike rotating input buffers, this also evicts the output and scratch
// buffers, which the autotuner does not rotate, and it costs no extra
// allocation per instruction.
class CacheFlusher {
 public:
  // Allocates the scratch buffer and loads the kernel. `flush_bytes` should
  // exceed the last level cache; see kDefaultFlushBytes. Returns an error if
  // either the allocation or the kernel load fails, in which case the caller
  // should proceed without flushing rather than fail compilation.
  static absl::StatusOr<std::unique_ptr<CacheFlusher>> Create(
      se::StreamExecutor* stream_executor, se::Stream* stream,
      se::DeviceAddressAllocator* allocator, int64_t flush_bytes);

  // Enqueues the flush on `stream`. Stream ordered, so whatever is enqueued
  // after this observes a cold cache. Does not synchronize.
  absl::Status Flush();

  int64_t flush_bytes() const { return buffer_->size(); }

 private:
  // Allocated untyped and viewed as uint32_t at launch. The allocator's typed
  // Allocate<T> overload does not actually convert its result, so it cannot be
  // used here.
  CacheFlusher(se::Stream* stream, se::ScopedDeviceAddress<uint8_t> buffer,
               se::ScopedDeviceAddress<uint8_t> sink,
               se::gpu::CacheFlushKernel::KernelType kernel,
               int64_t threads_per_block, int64_t block_count)
      : stream_(stream),
        buffer_(std::move(buffer)),
        sink_(std::move(sink)),
        kernel_(std::move(kernel)),
        threads_per_block_(threads_per_block),
        block_count_(block_count) {}

  se::Stream* stream_;
  se::ScopedDeviceAddress<uint8_t> buffer_;
  // The kernel is written to only under a condition that never holds. It
  // exists so the compiler cannot prove the reads are dead and delete them.
  se::ScopedDeviceAddress<uint8_t> sink_;
  se::gpu::CacheFlushKernel::KernelType kernel_;
  int64_t threads_per_block_;
  int64_t block_count_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_CACHE_FLUSHER_H_
