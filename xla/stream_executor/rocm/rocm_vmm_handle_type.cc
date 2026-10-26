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

#include <unistd.h>

#include <cstddef>
#include <string>

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/tsl/util/env_var.h"

namespace stream_executor::gpu {

namespace {

constexpr absl::string_view kExportableHandlesEnvVar =
    "XLA_ROCM_VMM_EXPORTABLE_HANDLES";

// Exportable VMM handles are only usable from ROCm 10.0 onwards; earlier
// releases cannot be relied on to export and import them correctly. Encoded
// the way rocm_config.h encodes TF_ROCM_VERSION (10000 * major + 100 * minor +
// patch).
constexpr int kMinRocmVersionForExportableHandles = 100000;  // ROCm 10.0.0

// Returns whether VMM allocations should request exportable handles.
//
// kExportableHandlesEnvVar unset: gated on the ROCm version.
// Set to false: request non-exportable handles.
// Set to true: request exportable handles whatever the ROCm version.
bool ExportableHandlesEnabled() {
  std::string env_value;
  absl::Status status = tsl::ReadStringFromEnvVar(
      kExportableHandlesEnvVar, /*default_val=*/"", &env_value);
  if (status.ok() && env_value.empty()) {
    if (TF_ROCM_VERSION < kMinRocmVersionForExportableHandles) {
      LOG(INFO) << "Exportable VMM handles require ROCm 10.0 or newer; "
                   "requesting non-exportable handles.";
      return false;
    }
    return true;
  }

  bool enabled = true;
  status = tsl::ReadBoolFromEnvVar(kExportableHandlesEnvVar,
                                   /*default_val=*/true, &enabled);
  if (!status.ok()) {
    LOG(WARNING) << status.message()
                 << "; requesting exportable VMM handle types.";
    enabled = true;
  }
  if (enabled && TF_ROCM_VERSION < kMinRocmVersionForExportableHandles) {
    LOG(WARNING) << kExportableHandlesEnvVar
                 << " requests exportable VMM handles on ROCm older than 10.0. "
                    "This is only safe on a runtime carrying a backport of the "
                    "fix; otherwise an exported handle may resolve to memory "
                    "that has already been freed.";
  }
  return enabled;
}

// Exports a minimal allocation on `device` to find out whether POSIX file
// descriptor handles actually work here, and returns the handle type to use
// from now on. A probe is cheaper than discovering the answer on the first
// real allocation, and keeps the decision out of the hot path.
hipMemAllocationHandleType ProbeHandleType(hipDevice_t device) {
  if (!ExportableHandlesEnabled()) {
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
                 << " reports no VMM allocation granularity ("
                 << ToString(result)
                 << "); falling back to non-exportable handles.";
    return hipMemHandleTypeNone;
  }

  hipMemGenericAllocationHandle_t handle = nullptr;
  result = hipMemCreate(&handle, granularity, &props, 0ULL);
  if (result != hipSuccess) {
    // Allocating says nothing about export support, so a failure here is about
    // memory, not capability. Keep the exportable handle type rather than
    // disabling exportable handles for the lifetime of the process because the
    // device happened to be full at startup.
    LOG(WARNING) << "Could not allocate the VMM handle type probe on device "
                 << device << ": " << ToString(result)
                 << "; assuming exportable handles are supported.";
    return hipMemHandleTypePosixFileDescriptor;
  }

  int fd = -1;
  hipError_t export_result = hipMemExportToShareableHandle(
      &fd, handle, hipMemHandleTypePosixFileDescriptor, 0ULL);
  if (export_result == hipSuccess) {
    close(fd);
  } else {
    LOG(WARNING) << "Device " << device
                 << " cannot export VMM allocations as POSIX file "
                    "descriptors: "
                 << ToString(export_result)
                 << "; falling back to non-exportable handles.";
  }

  hipError_t release_result = hipMemRelease(handle);
  if (release_result != hipSuccess) {
    LOG(ERROR) << "Failed to release the VMM handle type probe allocation on "
                  "device "
               << device << ": " << ToString(release_result);
  }
  return export_result == hipSuccess ? hipMemHandleTypePosixFileDescriptor
                                     : hipMemHandleTypeNone;
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
                    ? "POSIX file descriptor handles (exportable)"
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
