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

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_semaphore.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/rocm/delay_kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::gpu {
namespace {

// The semaphore lives in pinned host memory, which HIP allocates
// coarse-grained, so the device may hold it in an L2 that host writes do not
// invalidate. Observing the host's release therefore needs either a
// system-scope load or an explicit L2 invalidate, and which of those exists is
// per-architecture:
//
//   CDNA3/CDNA4, RDNA4  `volatile` already lowers to a system-scope access
//                       (`sc0 sc1`, `scope:SCOPE_SYS`). Nothing more needed.
//   CDNA2 (gfx90a)      No memory scope in the ISA, but BUFFER_INVL2 exists and
//                       __threadfence_system() emits it.
//   CDNA1, RDNA1-3      Neither: no per-instruction memory scope, and no L2
//                       invalidate instruction of any kind. Nothing works,
//                       which is why RocmExecutor::DelayKernelIsSupported()
//                       does not run the delay kernel there at all.
//
// So gfx90a is the only architecture reaching this kernel that needs the fence.
#if defined(__gfx90a__)
#define XLA_DELAY_KERNEL_NEEDS_SYSTEM_FENCE 1
#endif

// Reads the semaphore so that a host write is observable.
__device__ __forceinline__ GpuSemaphoreState LoadSemaphore(
    GpuSemaphoreState* semaphore) {
#ifdef XLA_DELAY_KERNEL_NEEDS_SYSTEM_FENCE
  // Same shape as collective_signal_rocm.cu.h. The fence is what carries the
  // access past L2 where the ISA cannot express it per instruction; on gfx90a
  // it emits buffer_wbl2 + buffer_invl2 + buffer_wbinvl1_vol.
  __threadfence_system();
  return __hip_atomic_load(semaphore, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
#else
  return *static_cast<volatile GpuSemaphoreState*>(semaphore);
#endif
}

// Writes the semaphore so that the host can observe it.
__device__ __forceinline__ void StoreSemaphore(GpuSemaphoreState* semaphore,
                                               GpuSemaphoreState value) {
#ifdef XLA_DELAY_KERNEL_NEEDS_SYSTEM_FENCE
  __hip_atomic_store(semaphore, value, __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_SYSTEM);
  __threadfence_system();
#else
  *static_cast<volatile GpuSemaphoreState*>(semaphore) = value;
#endif
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
__global__ void DelayKernel(GpuSemaphoreState* semaphore,
                            GpuSemaphoreState target, int64_t timeout_ticks,
                            int64_t poll_interval_ticks) {
  const int64_t tstart{wall_clock64()};
  bool target_not_reached;
  while ((target_not_reached = (LoadSemaphore(semaphore) != target)) &&
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
    StoreSemaphore(semaphore, GpuSemaphoreState::kTimedOut);
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
  // kernel that it may exit.
  ASSIGN_OR_RETURN(auto semaphore, GpuSemaphore::Create(executor));
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
