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

#include "xla/stream_executor/gpu/icache_flush.h"

#include <algorithm>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/icache_flush_kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {
namespace {

// One wavefront per block, and enough blocks that every compute unit is
// oversubscribed and therefore guaranteed to run at least one of them. Taken
// from the equivalent flush in ROCm's Triton GEMM autotuner and in PyTorch
// (https://github.com/pytorch/pytorch/pull/124362).
constexpr int kThreadsPerBlock = 64;
constexpr int kBlocksPerCore = 60;

}  // namespace

absl::StatusOr<IcacheFlusher> IcacheFlusher::Create(StreamExecutor* executor) {
  ABSL_ASSIGN_OR_RETURN(
      IcacheFlushKernel::KernelType kernel,
      GpuKernelRegistry::GetGlobalRegistry().LoadKernel<IcacheFlushKernel>(
          executor));

  // `core_count` is not populated on every device description, so make sure we
  // still launch something if it is missing.
  int core_count = std::max(1, executor->GetDeviceDescription().core_count());
  return IcacheFlusher(std::move(kernel),
                       BlockDim(core_count * kBlocksPerCore));
}

absl::Status IcacheFlusher::Flush(Stream* stream) {
  return kernel_.Launch(ThreadDim(kThreadsPerBlock), block_dim_, stream);
}

}  // namespace stream_executor::gpu
