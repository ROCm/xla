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

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"
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

}  // namespace

hipMemAllocationHandleType RocmVmmHandleType() {
  // Resolved once: the answer depends only on the ROCm version this was built
  // against and on the environment, neither of which changes while the process
  // runs. MakeRocmVmmAllocationProps() is on the allocation path, so the
  // environment must not be re-read per allocation.
  static const hipMemAllocationHandleType handle_type = [] {
    hipMemAllocationHandleType type = ExportableHandlesEnabled()
                                          ? hipMemHandleTypePosixFileDescriptor
                                          : hipMemHandleTypeNone;
    LOG(INFO) << "ROCm VMM allocations will request "
              << (type == hipMemHandleTypePosixFileDescriptor
                      ? "POSIX file descriptor handles (exportable)"
                      : "no handles (non-exportable)")
              << ".";
    return type;
  }();
  return handle_type;
}

hipMemAllocationProp MakeRocmVmmAllocationProps(hipDevice_t device) {
  hipMemAllocationProp props = {};
  props.type = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device;
  props.requestedHandleTypes = RocmVmmHandleType();
  return props;
}

}  // namespace stream_executor::gpu
