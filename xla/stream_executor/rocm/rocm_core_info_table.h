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

#include "xla/stream_executor/gpu/core_info_table_types.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor {
namespace gpu {

// Vector ALU and MFMA throughput per CU. Prefer the backend neutral
// gpu/core_info_table.h; this header exists so that the dispatch layer there
// can reach the ROCm data.

// Returns the rows recorded for `cc`, or nullptr if the gfx target is not in
// the table.
const CoreInfoRows* FindRocmCoreInfo(const RocmComputeCapability& cc);

// Number of FP32 FMA units per CU to assume when `FindRocmCoreInfo` returns no
// FP32 vector row for `cc`.
int RocmFpusPerCoreFallback(const RocmComputeCapability& cc);

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_CORE_INFO_TABLE_H_
