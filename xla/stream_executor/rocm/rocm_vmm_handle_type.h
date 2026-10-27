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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_HANDLE_TYPE_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_HANDLE_TYPE_H_

#include "rocm/include/hip/hip_runtime.h"

namespace stream_executor::gpu {

// Options controlling how ROCm VMM allocations are created.
//
// Whoever builds an allocator picks these; the VMM implementation only applies
// them. This mirrors CudaDeviceAllocator::Options, which carries the same
// choice on the CUDA side.
struct RocmVmmOptions {
  // Whether to request hipMemHandleTypePosixFileDescriptor instead of
  // hipMemHandleTypeNone.
  //
  // An allocation created with hipMemHandleTypeNone permits no export
  // mechanism, so its handle can never be shared with another process. RCCL's
  // symmetric memory path exports a registered buffer's allocation handle as a
  // POSIX file descriptor and obtains it from the NCCL proxy over a Unix
  // domain socket, so memory the collective layer may register has to be
  // allocated with this set.
  //
  // Has no effect below ROCm 10.0: earlier releases cannot be relied on to
  // export and import these handles correctly, so non-exportable handles are
  // requested there whatever this says.
  bool enable_posix_fd_handle = true;
};

// Returns the allocation handle type that ROCm VMM allocations should request
// for `options`, after applying the ROCm version gate described above. The
// answer is the same for every device.
hipMemAllocationHandleType RocmVmmHandleType(const RocmVmmOptions& options);

// Returns allocation properties for a pinned device allocation on `device`,
// requesting the handle type returned by RocmVmmHandleType(options).
//
// Every ROCm VMM site must build its properties through this function, so that
// the handle type is applied in one place rather than restated at each
// hipMemCreate(). hipMemGetAllocationGranularity() takes the same properties,
// so a site that queries granularity with different properties than the site
// that allocates would pad and align inconsistently.
hipMemAllocationProp MakeRocmVmmAllocationProps(hipDevice_t device,
                                                const RocmVmmOptions& options);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_HANDLE_TYPE_H_
