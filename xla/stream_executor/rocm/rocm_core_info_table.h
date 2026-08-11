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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_CORE_INFO_TABLE_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_CORE_INFO_TABLE_H_

#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor {
namespace gpu {

// Returns the number of FP32 operations a compute unit can issue in parallel,
// which is what DeviceDescription::fpus_per_core reports; the CUDA equivalent is
// GetFpusPerCore in cuda_core_info_table.h.
//
// Unrecognised architectures fall back to the CDNA 2 and later value, which the
// caller previously assumed unconditionally.
int GetRocmFpusPerCore(const RocmComputeCapability& cc);

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_CORE_INFO_TABLE_H_
