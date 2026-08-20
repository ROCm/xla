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

#include "xla/backends/gpu/autotuner/cache_flusher.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/cache_flush_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {
namespace {

// Enough blocks to saturate the memory system without making the launch itself
// expensive. The kernel is a grid stride loop, so this does not have to cover
// the buffer.
constexpr int64_t kBlocksPerCore = 8;

}  // namespace

absl::StatusOr<std::unique_ptr<CacheFlusher>> CacheFlusher::Create(
    se::StreamExecutor* stream_executor, se::Stream* stream,
    se::DeviceAddressAllocator* allocator, int64_t flush_bytes) {
  if (flush_bytes <= 0) {
    return absl::InvalidArgumentError("flush_bytes must be positive");
  }
  // Round down to a whole number of elements.
  constexpr int64_t kElementBytes = sizeof(uint32_t);
  const int64_t rounded_bytes = (flush_bytes / kElementBytes) * kElementBytes;

  ABSL_ASSIGN_OR_RETURN(
      se::ScopedDeviceAddress<uint8_t> buffer,
      allocator->Allocate(stream_executor->device_ordinal(), rounded_bytes,
                          /*retry_on_failure=*/false));
  ABSL_ASSIGN_OR_RETURN(
      se::ScopedDeviceAddress<uint8_t> sink,
      allocator->Allocate(stream_executor->device_ordinal(), sizeof(uint32_t),
                          /*retry_on_failure=*/false));

  // Zero once, so the accumulated sum is deterministically zero and never hits
  // the sentinel the kernel compares against. Uninitialized memory could in
  // principle sum to it, which would add a stray write but not a correctness
  // problem; zeroing removes the question.
  se::DeviceAddressBase raw_buffer = *buffer;
  ABSL_RETURN_IF_ERROR(stream->MemZero(&raw_buffer, raw_buffer.size()));
  se::DeviceAddressBase raw_sink = *sink;
  ABSL_RETURN_IF_ERROR(stream->MemZero(&raw_sink, raw_sink.size()));
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  ABSL_ASSIGN_OR_RETURN(
      se::gpu::CacheFlushKernel::KernelType kernel,
      se::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .LoadKernel<se::gpu::CacheFlushKernel>(stream_executor));

  const se::DeviceDescription& device = stream_executor->GetDeviceDescription();
  const int64_t threads_per_block =
      std::min<int64_t>(device.threads_per_block_limit(), 256);
  const int64_t block_count =
      std::max<int64_t>(1, device.core_count() * kBlocksPerCore);

  VLOG(1) << "Cache flusher: " << rounded_bytes << " bytes, " << block_count
          << " blocks of " << threads_per_block << " threads.";

  return absl::WrapUnique(new CacheFlusher(stream, std::move(buffer),
                                           std::move(sink), std::move(kernel),
                                           threads_per_block, block_count));
}

absl::Status CacheFlusher::Flush() {
  se::DeviceAddress<uint32_t> buffer(*buffer_);
  se::DeviceAddress<uint32_t> sink(*sink_);
  return kernel_.Launch(se::ThreadDim(threads_per_block_),
                        se::BlockDim(block_count_), stream_, buffer,
                        buffer.ElementCount(), sink);
}

}  // namespace gpu
}  // namespace xla
