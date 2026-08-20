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

#ifndef XLA_STREAM_EXECUTOR_GPU_ICACHE_FLUSH_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_ICACHE_FLUSH_KERNEL_H_

#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {

// Kernel that invalidates the GPU instruction cache of the compute unit it runs
// on. It takes no arguments and produces no output; the invalidation is a pure
// side effect of the instructions it executes.
//
// Only registered for the ROCm platform - there is no equivalent instruction on
// other platforms, so `GpuKernelRegistry::LoadKernel` fails there.
struct IcacheFlushKernel {
  using KernelType = TypedKernel<>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ICACHE_FLUSH_KERNEL_H_
