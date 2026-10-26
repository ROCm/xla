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

// Returns the allocation handle type that ROCm VMM allocations should
// request. The answer is the same for every device.
//
// An allocation created with hipMemHandleTypeNone permits no export
// mechanism, so its handle can never be shared with another device or
// process. Requesting hipMemHandleTypePosixFileDescriptor keeps VMM
// allocations exportable, matching what the CUDA device allocator already
// does with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR.
//
// Exportable handles are only requested on ROCm 10.0 and newer; earlier
// releases cannot be relied on to export and import them correctly.
//
// XLA_ROCM_VMM_EXPORTABLE_HANDLES overrides the version gate in both
// directions: 0 or false requests non-exportable handles, 1 or true requests
// exportable ones whatever the ROCm version.
hipMemAllocationHandleType RocmVmmHandleType();

// Returns allocation properties for a pinned device allocation on `device`,
// requesting the handle type returned by RocmVmmHandleType().
//
// Every ROCm VMM site must build its properties through this function, so
// that the handle type is decided in one place rather than restated at each
// hipMemCreate().
hipMemAllocationProp MakeRocmVmmAllocationProps(hipDevice_t device);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_HANDLE_TYPE_H_
