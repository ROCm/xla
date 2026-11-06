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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// Device memory allocator using ROCm/HIP Virtual Memory Management (VMM) APIs.
// Returned memory allocations are always mapped to a valid device address range
// and accessible from the device and its peers.
class RocmVmmAllocator : public MemoryAllocator {
 public:
  // Options controlling how ROCm VMM allocations are created.
  //
  // Whoever builds an allocator picks these; the VMM implementation only
  // applies them. This mirrors CudaDeviceAllocator::Options, which carries the
  // same choice on the CUDA side. The other ROCm VMM sites
  // (RocmRawMemoryAllocation, RocmMemoryReservation,
  // RocmDeviceAddressVmmAllocator) take the same struct, so they all go
  // through GetRocmVmmAllocationParams() below.
  struct Options {
    // Whether to request hipMemHandleTypePosixFileDescriptor instead of
    // hipMemHandleTypeNone.
    //
    // An allocation created with hipMemHandleTypeNone permits no export
    // mechanism, so its handle can never be shared with another process.
    // RCCL's symmetric memory path exports a registered buffer's allocation
    // handle as a POSIX file descriptor and obtains it from the NCCL proxy
    // over a Unix domain socket, so memory the collective layer may register
    // has to be allocated with this set.
    //
    // Has no effect below ROCm 10.0: earlier releases cannot be relied on to
    // export and import these handles correctly, so non-exportable handles
    // are requested there whatever this says.
    bool enable_posix_fd_handle = true;
  };

  // Uses the default Options.
  //
  // Spelled as an overload rather than as `const Options& options = {}` on
  // the constructor below: Options is a member of this class, and its default
  // member initializer is not available in a default argument of an enclosing
  // class member declaration.
  explicit RocmVmmAllocator(StreamExecutor* executor);

  // `options` decides the handle type the allocations request; see Options.
  RocmVmmAllocator(StreamExecutor* executor, const Options& options);

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) final;

 private:
  StreamExecutor* executor_;
  Options options_;
};

// What a ROCm VMM site needs before it can allocate or reserve.
struct RocmVmmAllocationParams {
  // Properties for a pinned device allocation. Zero-initialized so that the
  // fields GetRocmVmmAllocationParams() does not set (allocFlags,
  // win32HandleMetaData) are not handed to the driver as stack garbage.
  hipMemAllocationProp prop = {};

  // Recommended allocation granularity for `prop`. Allocation sizes and
  // reservation alignments must be multiples of it.
  size_t granularity = 0;
};

// Returns the allocation properties for `executor`'s device under `options`,
// together with the granularity the driver recommends for them.
//
// hipMemGetAllocationGranularity() takes the same properties hipMemCreate()
// does, so a site that queried granularity with different properties than it
// allocated with would pad and align inconsistently. The handle type is
// likewise decided in one place rather than restated at each hipMemCreate().
//
// The caller must have activated `executor`'s context.
absl::StatusOr<RocmVmmAllocationParams> GetRocmVmmAllocationParams(
    StreamExecutor* executor, const RocmVmmAllocator::Options& options);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_VMM_ALLOCATOR_H_
