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

// Stub implementation of HostOffloadingNanoRtExecutable that removes the
// nanort_client -> cpu_compiler_pure -> oneDNN build dependency chain from
// hlo_runner_main_gpu. HostExecute thunks are not used in ROCm benchmark
// workloads.
//
// TODO: replace with a proper split once the upstream dependency is resolved.

#include "xla/core/host_offloading/host_offloading_nanort_executable_stub.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"

namespace xla {

// static
absl::StatusOr<std::unique_ptr<HostOffloadingNanoRtExecutable>>
HostOffloadingNanoRtExecutable::LoadFromProto(
    const HostOffloadingExecutableProto& proto) {
  return absl::UnimplementedError(
      "HostOffloadingNanoRtExecutable::LoadFromProto: NanoRt CPU execution "
      "not available in this build. HostExecute thunks are not supported.");
}

}  // namespace xla
