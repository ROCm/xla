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

#include "absl/log/log.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"

namespace stream_executor::gpu {

hipMemAllocationHandleType RocmVmmHandleType(const RocmVmmOptions& options) {
// Exportable VMM handles are only usable from ROCm 10.0 onwards; earlier
// releases cannot be relied on to export and import them correctly.
// rocm_config.h encodes TF_ROCM_VERSION as 10000 * major + 100 * minor +
// patch, so ROCm 10.0.0 is 100000.
#if TF_ROCM_VERSION >= 100000
  if (options.enable_posix_fd_handle) {
    return hipMemHandleTypePosixFileDescriptor;
  }
#else
  if (options.enable_posix_fd_handle) {
    // Logged once rather than per allocation. A caller that asked for
    // exportable handles and silently did not get them has no way to tell,
    // and the consequence only surfaces much later, where the memory is
    // registered.
    LOG_FIRST_N(WARNING, 1)
        << "Exportable ROCm VMM handles require ROCm 10.0 or newer; VMM "
           "allocations will request no handles (non-exportable).";
  }
#endif
  return hipMemHandleTypeNone;
}

hipMemAllocationProp MakeRocmVmmAllocationProps(hipDevice_t device,
                                                const RocmVmmOptions& options) {
  hipMemAllocationProp props = {};
  props.type = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device;
  props.requestedHandleTypes = RocmVmmHandleType(options);
  return props;
}

}  // namespace stream_executor::gpu
