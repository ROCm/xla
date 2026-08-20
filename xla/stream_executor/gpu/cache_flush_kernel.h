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

#ifndef XLA_STREAM_EXECUTOR_GPU_CACHE_FLUSH_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_CACHE_FLUSH_KERNEL_H_

#include <cstdint>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {

// Defines a trait for the CacheFlushKernel that can be used to register and
// look up the kernel in the GPU kernel registry.
//
// The kernel reads every element of a scratch buffer and discards the result.
// Reading a buffer larger than the last level cache displaces whatever was
// resident before it, which is the only way to evict a memory-side cache such
// as the Infinity Cache on CDNA3/CDNA4, since no coherence operation reaches
// it. Arguments are the scratch buffer, its element count, and a sink the
// kernel never actually writes to but which the compiler cannot prove is dead.
struct CacheFlushKernel {
  using KernelType = TypedKernel<DeviceAddress<uint32_t>, uint64_t,
                                 DeviceAddress<uint32_t>>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_CACHE_FLUSH_KERNEL_H_
