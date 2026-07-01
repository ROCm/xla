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

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/rocm/delay_kernel.h"
#include "xla/stream_executor/typed_kernel_factory.h"

namespace stream_executor::gpu {
namespace {
// Wait for the value pointed to by `semaphore` to have value `target`, timing
// out after a reasonable time if not reached
__global__ void DelayKernel(volatile GpuSemaphoreState* semaphore,
                            GpuSemaphoreState target) {
  constexpr int64_t WAIT_CYCLES{1024};
  constexpr int64_t TIMEOUT_CYCLES{200000000};
  const int64_t tstart{clock64()};
  bool target_not_reached;
  while ((target_not_reached = (*semaphore != target)) &&
         (clock64() - tstart) < TIMEOUT_CYCLES) {
    int64_t elapsed{};
    const int64_t t0{clock64()};
    do {
      elapsed = clock64() - t0;
    } while (elapsed < WAIT_CYCLES);
  }
  if (target_not_reached) {
    *semaphore = GpuSemaphoreState::kTimedOut;
  }
}
}  // namespace

absl::StatusOr<GpuSemaphore> LaunchDelayKernel(Stream* stream) {
  StreamExecutor* executor = stream->parent();

  // Semaphore value that will be used to signal to the delay
  // kernel that it may exit
  ASSIGN_OR_RETURN(auto semaphore, GpuSemaphore::Create(executor));
  *semaphore = GpuSemaphoreState::kHold;
  ASSIGN_OR_RETURN(
      auto kernel,
      (TypedKernelFactory<DeviceAddress<GpuSemaphoreState>,
                          GpuSemaphoreState>::Create(executor, "DelayKernel",
                                                     reinterpret_cast<void*>(
                                                         DelayKernel))));
  // Launch a delay kernel into the stream. Spin until GetElapsedDuration() is
  // called, the timer is destroyed, or the timeout in the kernel is reached.
  RETURN_IF_ERROR(kernel.Launch(ThreadDim(1, 1, 1), BlockDim(1, 1, 1), stream,
                                semaphore.device(),
                                GpuSemaphoreState::kRelease));

  return semaphore;
}
}  // namespace stream_executor::gpu
