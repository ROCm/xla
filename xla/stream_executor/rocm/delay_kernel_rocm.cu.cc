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

#include <algorithm>
#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/rocm/delay_kernel.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::gpu {
namespace {

// Allocates the semaphore in host memory the device can read and write while a
// kernel is running.
//
// `hipHostMalloc(hipHostMallocPortable)` - what StreamExecutor's generic host
// allocation uses - is coarse-grained, so the device may hold the value in an
// L2 that host writes do not invalidate. A `volatile` load bypasses L1 but is
// still answered from L2, so the spin below would never observe the host's
// release. Only CDNA2 through CDNA4 and RDNA4 can dislodge such a line from the
// kernel side (BUFFER_INVL2 / GLOBAL_INV, or a system-scope load); CDNA1 and
// RDNA1-3 have no L2 invalidate instruction at all.
//
// Requesting coherent - i.e. fine-grained - memory avoids the problem for all
// of them, because the value is never cached in L2 to begin with. Measured on
// gfx908, gfx90a, gfx950, gfx1102, gfx1151 and gfx1201: with this flag a plain
// `volatile` spin observes the release on every one, and without it the kernel
// runs its full timeout on the first four.
//
// Scoped to the semaphore rather than applied in RocmExecutor::HostAllocate so
// that bulk pinned allocations keep their existing behaviour. This mirrors
// rocm_device_address_vmm_allocator.cc, which allocates its timeline counter
// the same way for the same reason.
absl::StatusOr<GpuSemaphore> CreateCoherentSemaphore() {
  void* ptr = nullptr;
  RETURN_IF_ERROR(ToStatus(
      hipHostMalloc(&ptr, sizeof(GpuSemaphoreState),
                    hipHostMallocPortable | hipHostMallocCoherent),
      "Failed to allocate coherent host memory for the delay kernel "
      "semaphore"));
  return GpuSemaphore::Create(std::make_unique<GenericMemoryAllocation>(
      ptr, sizeof(GpuSemaphoreState), [](void* location, uint64_t size) {
        hipError_t result = hipHostFree(location);
        if (result != hipSuccess) {
          LOG(ERROR) << "Failed to free delay kernel semaphore: "
                     << ToString(result);
        }
      }));
}

// Wait for the value pointed to by `semaphore` to have value `target`, timing
// out after approximately `timeout_ticks` wall clock ticks if that value is
// not reached. This can happen if, for example, blocking launches are enabled
// via HIP_LAUNCH_BLOCKING=1. It can also happen if launching a kernel after
// this delay kernel causes synchronisation, e.g. because of lazy loading.
//
// Unlike the CUDA variant this spins on `wall_clock64()` rather than
// `clock64()`. The CUDA version's timeout is a hardcoded cycle count that
// assumes a 2GHz shader clock; no AMD GPU runs at that rate, so the realized
// timeout is wrong on all of them - measured at 118ms on gfx90a (1688MHz) and
// 63ms on gfx1201 against the intended 100ms. On gfx1201 it is worse than a
// scale factor: `clock64()` reads HW_REG_SHADER_CYCLES, which ticks at ~3190MHz
// against a reported 2400MHz maximum core clock, so it is not the shader clock
// at all. The wall clock ticks at the constant rate reported by
// `hipDeviceAttributeWallClockRate`, which is where the tick counts passed in
// here come from.
//
// (The shader clock itself is stable during the spin - it sits at boost on
// every arch measured - so this is about the reference rate being unknowable,
// not about the clock moving underneath us.)
__global__ void DelayKernel(volatile GpuSemaphoreState* semaphore,
                            GpuSemaphoreState target, int64_t timeout_ticks,
                            int64_t poll_interval_ticks) {
  const int64_t tstart{wall_clock64()};
  bool target_not_reached;
  while ((target_not_reached = (*semaphore != target)) &&
         (wall_clock64() - tstart) < timeout_ticks) {
    int64_t elapsed{};
    const int64_t t0{wall_clock64()};
    do {
      elapsed = wall_clock64() - t0;
    } while (elapsed < poll_interval_ticks);
  }
  if (target_not_reached) {
    // We are exiting due to the timeout. Signal this back to the host so that
    // we can emit a warning, as it probably indicates suboptimal usage.
    *semaphore = GpuSemaphoreState::kTimedOut;
  }
}

// Returns the frequency of the device's constant rate wall clock in Hz.
//
// The rate is not uniform across AMD GPUs - it is 100MHz on gfx950 and gfx1201
// but 25MHz on gfx90a - so the fallback below is a guess that is wrong by 4x on
// at least one supported device. It is only reached if the attribute query
// itself fails, which is not expected to happen.
int64_t WallClockHz(int device_ordinal) {
  int rate_khz = 0;
  hipError_t result = hipDeviceGetAttribute(
      &rate_khz, hipDeviceAttributeWallClockRate, device_ordinal);
  if (result != hipSuccess || rate_khz <= 0) {
    LOG_FIRST_N(WARNING, 1)
        << "Could not query the wall clock rate of device " << device_ordinal
        << "; assuming 100MHz. The delay kernel timeout will be wrong on any "
           "device whose wall clock does not run at that rate.";
    return 100'000'000;
  }
  return int64_t{rate_khz} * 1000;
}
}  // namespace

absl::StatusOr<GpuSemaphore> LaunchDelayKernel(Stream* stream) {
  StreamExecutor* executor = stream->parent();

  // Allocate a semaphore value that will be used to signal to the delay
  // kernel that it may exit. See CreateCoherentSemaphore for why this does not
  // go through StreamExecutor::HostMemoryAllocate.
  ASSIGN_OR_RETURN(auto semaphore, CreateCoherentSemaphore());
  *semaphore = GpuSemaphoreState::kHold;
  // In principle the kernel could be loaded lazily and shared across
  // multiple GpuTimer objects.
  ASSIGN_OR_RETURN(
      auto kernel,
      (TypedKernelFactory<DeviceAddress<GpuSemaphoreState>, GpuSemaphoreState,
                          int64_t,
                          int64_t>::Create(executor, "DelayKernel",
                                           reinterpret_cast<void*>(
                                               DelayKernel))));
  // This runs before the timer's start event is recorded, so the attribute
  // query is off the timed path.
  const int64_t wall_clock_hz = WallClockHz(executor->device_ordinal());
  // Launch a delay kernel into this stream, which will spin until
  // GetElapsedDuration() is called, the timer is destroyed, or the timeout
  // in the kernel is reached.
  RETURN_IF_ERROR(kernel.Launch(
      ThreadDim(1, 1, 1), BlockDim(1, 1, 1), stream, semaphore.device(),
      GpuSemaphoreState::kRelease,
      /*timeout_ticks=*/wall_clock_hz / 10,  // 100ms
      /*poll_interval_ticks=*/
      std::max<int64_t>(wall_clock_hz / 1'000'000, 1)));  // 1us

  return semaphore;
}

}  // namespace stream_executor::gpu
