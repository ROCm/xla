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

#ifndef XLA_STREAM_EXECUTOR_GPU_CACHE_FLUSH_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_CACHE_FLUSH_KERNEL_LIB_CU_H_

#include <cstdint>

namespace stream_executor::gpu {

// Streams a read over the whole scratch buffer so that the lines it pulls in
// displace whatever the previous kernel left cached.
//
// Ordinary cached loads are deliberate. Non-temporal or streaming loads would
// bypass the very caches this is trying to fill, which is the opposite of what
// is wanted.
//
// The accumulator is stored only when it matches a sentinel. The comparison is
// false in practice, since the buffer is zeroed once at allocation and never
// written, but the compiler cannot prove that and therefore cannot drop the
// loads. Without this the entire loop is dead code and the kernel becomes a
// very fast no-op.
__global__ void CacheFlushKernelImpl(uint32_t* buffer, uint64_t num_elements,
                                     uint32_t* sink) {
  const uint64_t block_dim_x = static_cast<uint64_t>(blockDim.x);
  const uint64_t stride = block_dim_x * gridDim.x;
  uint32_t acc = 0;
  for (uint64_t idx = threadIdx.x + blockIdx.x * block_dim_x;
       idx < num_elements; idx += stride) {
    acc += buffer[idx];
  }
  if (acc == 0xdeadbeefu) {
    *sink = acc;
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_CACHE_FLUSH_KERNEL_LIB_CU_H_
