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

#include "xla/stream_executor/rocm/rocm_vmm_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace stream_executor::gpu {

absl::StatusOr<RocmVmmAllocationParams> GetRocmVmmAllocationParams(
    StreamExecutor* executor, const RocmVmmAllocator::Options& options) {
  hipDevice_t device;
  ABSL_RETURN_IF_ERROR(ToStatus(
      hipDeviceGet(&device, executor->device_ordinal()), "hipDeviceGet"));

  RocmVmmAllocationParams params;
  params.prop.type = hipMemAllocationTypePinned;
  params.prop.location.type = hipMemLocationTypeDevice;
  params.prop.location.id = device;
  params.prop.requestedHandleTypes = hipMemHandleTypeNone;

  if (options.enable_posix_fd_handle) {
// Exportable VMM handles are only usable from ROCm 10.0 onwards; earlier
// releases cannot be relied on to export and import them correctly.
// rocm_config.h encodes TF_ROCM_VERSION as 10000 * major + 100 * minor +
// patch, so ROCm 10.0.0 is 100000.
#if TF_ROCM_VERSION >= 100000
    params.prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
#else
    // Logged once rather than per allocation. A caller that asked for
    // exportable handles and silently did not get them has no way to tell, and
    // the consequence only surfaces much later, where the memory is
    // registered.
    LOG_FIRST_N(WARNING, 1)
        << "Exportable ROCm VMM handles require ROCm 10.0 or newer; VMM "
           "allocations will request no handles (non-exportable).";
#endif
  }

  ABSL_RETURN_IF_ERROR(ToStatus(
      hipMemGetAllocationGranularity(&params.granularity, &params.prop,
                                     hipMemAllocationGranularityRecommended),
      "hipMemGetAllocationGranularity"));
  return params;
}

static hipMemAccessDesc GetVmmAccessDescriptor(int device) {
  hipMemAccessDesc descriptor = {};
  descriptor.location.type = hipMemLocationTypeDevice;
  descriptor.location.id = device;
  descriptor.flags = hipMemAccessFlagsProtReadWrite;
  return descriptor;
}

// Allocates device memory using HIP VMM APIs. Returns the mapped pointer,
// the padded size, and the allocation handle.
static absl::StatusOr<
    std::tuple<void*, uint64_t, hipMemGenericAllocationHandle_t>>
VmmAllocate(StreamExecutor* executor, uint64_t size,
            const RocmVmmAllocator::Options& options) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  ABSL_ASSIGN_OR_RETURN(RocmVmmAllocationParams params,
                        GetRocmVmmAllocationParams(executor, options));
  size_t granularity = params.granularity;

  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, granularity);
  hipMemGenericAllocationHandle_t handle;

  ABSL_RETURN_IF_ERROR(
      ToStatus(hipMemCreate(&handle, padded_size, &params.prop, 0ULL)));

  hipDeviceptr_t ptr = nullptr;
  absl::Status status = ToStatus(
      hipMemAddressReserve(&ptr, padded_size, granularity, nullptr, 0ULL));
  if (!status.ok()) {
    ToStatus(hipMemRelease(handle)).IgnoreError();
    return status;
  }

  status = ToStatus(hipMemMap(ptr, padded_size, 0, handle, 0ULL));
  if (!status.ok()) {
    ToStatus(hipMemAddressFree(ptr, padded_size)).IgnoreError();
    ToStatus(hipMemRelease(handle)).IgnoreError();
    return status;
  }

  // Cleanup on error: unmap + free VA + release physical memory.
  absl::Cleanup cleanup = [&]() {
    ToStatus(hipMemUnmap(ptr, padded_size)).IgnoreError();
    ToStatus(hipMemAddressFree(ptr, padded_size)).IgnoreError();
    ToStatus(hipMemRelease(handle)).IgnoreError();
  };

  VLOG(3) << "VMM allocated " << ptr << " requested size: " << size
          << " padded size: " << padded_size << " granularity: " << granularity
          << " device: " << executor->device_ordinal();

  // Set access for this device and all P2P-capable peers, matching the CUDA
  // pattern that gates on CanEnablePeerAccessTo().
  int device_count = 0;
  ABSL_RETURN_IF_ERROR(ToStatus(hipGetDeviceCount(&device_count)));
  for (int peer = 0; peer < device_count; peer++) {
    if (peer != executor->device_ordinal()) {
      auto peer_executor_or = const_cast<Platform*>(executor->GetPlatform())
                                  ->ExecutorForDevice(peer);
      if (!peer_executor_or.ok() ||
          !executor->CanEnablePeerAccessTo(peer_executor_or.value())) {
        continue;
      }
    }
    hipMemAccessDesc access_desc = GetVmmAccessDescriptor(peer);
    ABSL_RETURN_IF_ERROR(
        ToStatus(hipMemSetAccess(ptr, padded_size, &access_desc, 1)));
  }

  std::move(cleanup).Cancel();
  return std::make_tuple(ptr, padded_size, handle);
}

static void VmmDeallocate(StreamExecutor* executor, void* ptr,
                          uint64_t padded_size,
                          hipMemGenericAllocationHandle_t handle) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  VLOG(3) << "VMM deallocating " << ptr << " padded size: " << padded_size
          << " device: " << executor->device_ordinal();

  absl::Status status = ToStatus(hipMemUnmap(ptr, padded_size));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to unmap VMM memory at " << ptr << ": " << status;
  }
  status = ToStatus(hipMemRelease(handle));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to release VMM handle for " << ptr << ": " << status;
  }
  status = ToStatus(hipMemAddressFree(ptr, padded_size));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to free VMM address at " << ptr << ": " << status;
  }
}

namespace {

class RocmVmmMemoryAllocation final : public MemoryAllocation {
 public:
  RocmVmmMemoryAllocation(StreamExecutor* executor, void* ptr,
                          uint64_t requested_size, uint64_t padded_size,
                          hipMemGenericAllocationHandle_t handle)
      : executor_(executor),
        ptr_(ptr),
        requested_size_(requested_size),
        padded_size_(padded_size),
        handle_(handle) {}

  ~RocmVmmMemoryAllocation() final {
    if (ptr_ != nullptr) {
      VmmDeallocate(executor_, ptr_, padded_size_, handle_);
    }
  }

  DeviceAddressBase address() const final {
    return DeviceAddressBase(ptr_, requested_size_);
  }

  std::string ToString() const final {
    return absl::StrFormat(
        "RocmVmmMemoryAllocation[device=%d, ptr=%p, size=%d, padded_size=%d]",
        executor_->device_ordinal(), ptr_, requested_size_, padded_size_);
  }

 private:
  StreamExecutor* executor_;
  void* ptr_;
  uint64_t requested_size_;
  uint64_t padded_size_;
  hipMemGenericAllocationHandle_t handle_;
};

}  // namespace

RocmVmmAllocator::RocmVmmAllocator(StreamExecutor* executor)
    : RocmVmmAllocator(executor, Options()) {}

RocmVmmAllocator::RocmVmmAllocator(StreamExecutor* executor,
                                   const Options& options)
    : executor_(executor), options_(options) {}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> RocmVmmAllocator::Allocate(
    uint64_t size) {
  if (size == 0) {
    return std::make_unique<RocmVmmMemoryAllocation>(executor_, nullptr, 0, 0,
                                                     nullptr);
  }

  ABSL_ASSIGN_OR_RETURN(auto result, VmmAllocate(executor_, size, options_));
  auto [ptr, padded_size, handle] = result;

  return std::make_unique<RocmVmmMemoryAllocation>(executor_, ptr, size,
                                                   padded_size, handle);
}

}  // namespace stream_executor::gpu
