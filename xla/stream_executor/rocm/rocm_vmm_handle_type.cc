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

#include "xla/stream_executor/rocm/rocm_vmm_handle_type.h"

#include <cstddef>
#include <string>

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/tsl/util/env_var.h"

namespace stream_executor::gpu {

namespace {

constexpr absl::string_view kExportableHandlesEnvVar =
    "XLA_ROCM_VMM_EXPORTABLE_HANDLES";

// Returns true if kExportableHandlesEnvVar is set to a false value ("0" or
// "false"), which forces the non-exportable handle type used before symmetric
// memory needed exportable allocations.
bool ExportableHandlesDisabled() {
  bool enabled = true;
  absl::Status status = tsl::ReadBoolFromEnvVar(kExportableHandlesEnvVar,
                                                /*default_val=*/true, &enabled);
  if (!status.ok()) {
    LOG(WARNING) << status.message()
                 << "; requesting exportable VMM handle types.";
  }
  return !enabled;
}

// Attempts a minimal exportable allocation on `device` to find out whether the
// driver supports POSIX file descriptor handles here, and returns the handle
// type to use from now on. A probe allocation is cheaper than discovering the
// answer on the first real allocation, and keeps the decision out of the hot
// path.
hipMemAllocationHandleType ProbeHandleType(hipDevice_t device) {
  if (ExportableHandlesDisabled()) {
    return hipMemHandleTypeNone;
  }

  hipMemAllocationProp props = {};
  props.type = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device;
  props.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;

  size_t granularity = 0;
  hipError_t result = hipMemGetAllocationGranularity(
      &granularity, &props, hipMemAllocationGranularityMinimum);
  if (result != hipSuccess || granularity == 0) {
    LOG(WARNING) << "Device " << device
                 << " reports no granularity for exportable VMM allocations ("
                 << ToString(result)
                 << "); falling back to non-exportable handles. RCCL "
                    "symmetric memory is unavailable on this device.";
    return hipMemHandleTypeNone;
  }

  hipMemGenericAllocationHandle_t handle = nullptr;
  result = hipMemCreate(&handle, granularity, &props, 0ULL);
  if (result != hipSuccess) {
    LOG(WARNING) << "Device " << device
                 << " cannot create exportable (POSIX file descriptor) VMM "
                    "allocations: "
                 << ToString(result)
                 << "; falling back to non-exportable handles. RCCL symmetric "
                    "memory is unavailable on this device.";
    return hipMemHandleTypeNone;
  }

  hipError_t release_result = hipMemRelease(handle);
  if (release_result != hipSuccess) {
    LOG(ERROR) << "Failed to release the VMM handle type probe allocation on "
                  "device "
               << device << ": " << ToString(release_result);
  }
  return hipMemHandleTypePosixFileDescriptor;
}

}  // namespace

hipMemAllocationHandleType RocmVmmHandleType(hipDevice_t device) {
  static absl::Mutex mu(absl::kConstInit);
  static auto& cache =
      *new absl::flat_hash_map<int, hipMemAllocationHandleType>();

  absl::MutexLock lock(&mu);
  if (auto it = cache.find(device); it != cache.end()) {
    return it->second;
  }

  hipMemAllocationHandleType handle_type = ProbeHandleType(device);
  LOG(INFO) << "ROCm VMM allocations on device " << device << " will request "
            << (handle_type == hipMemHandleTypePosixFileDescriptor
                    ? "POSIX file descriptor handles (exportable, required by "
                      "RCCL symmetric memory)"
                    : "no handles (non-exportable)")
            << ".";
  cache.emplace(device, handle_type);
  return handle_type;
}

hipMemAllocationProp MakeRocmVmmAllocationProps(hipDevice_t device) {
  hipMemAllocationProp props = {};
  props.type = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device;
  props.requestedHandleTypes = RocmVmmHandleType(device);
  return props;
}

}  // namespace stream_executor::gpu
