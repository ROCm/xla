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

// Returns the allocation handle type that ROCm VMM allocations should request
// on `device`.
//
// RCCL's symmetric-memory path (ncclCommWindowRegister with
// NCCL_WIN_COLL_SYMMETRIC) exports the registered buffer's allocation handle
// as a POSIX file descriptor and obtains it from the NCCL proxy over a Unix
// domain socket. An allocation created with hipMemHandleTypeNone cannot be
// exported, so that request never completes and registration blocks forever
// inside ncclProxyClientGetFdBlocking. Requesting
// hipMemHandleTypePosixFileDescriptor keeps VMM allocations registerable,
// matching what the CUDA device allocator already does with
// CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR.
//
// The handle type is probed once per device. A device that cannot create an
// exportable allocation falls back to hipMemHandleTypeNone, which preserves
// the previous behavior for everything except symmetric memory. Set
// XLA_ROCM_VMM_EXPORTABLE_HANDLES to 0 or false to force that fallback.
//
// The caller must have activated `device`'s context.
hipMemAllocationHandleType RocmVmmHandleType(hipDevice_t device);

// Returns allocation properties for a pinned device allocation on `device`,
// requesting the handle type returned by RocmVmmHandleType().
//
// Every ROCm VMM site must build its properties through this function.
// hipMemGetAllocationGranularity() takes the same properties and the
// granularity it reports can depend on the requested handle type, so a site
// that queries granularity with different properties than the site that
// allocates would pad and align inconsistently.
//
// The caller must have activated `device`'s context.
hipMemAllocationProp MakeRocmVmmAllocationProps(hipDevice_t device);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_HANDLE_TYPE_H_
