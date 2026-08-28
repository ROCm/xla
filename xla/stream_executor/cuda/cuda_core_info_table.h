/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_CORE_INFO_TABLE_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_CORE_INFO_TABLE_H_

#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/gpu/core_info_table_types.h"

namespace stream_executor {
namespace gpu {

// CUDA Core and Tensor Core throughput per SM. Prefer the backend neutral
// gpu/core_info_table.h; this header exists so that the dispatch layer there
// can reach the CUDA data.

// Returns the rows recorded for `cc`, or nullptr if the compute capability is
// not in the table.
const CoreInfoRows* FindCudaCoreInfo(CudaComputeCapability cc);

// Number of FP32 FMA units (CUDA Cores) per SM to assume when
// `FindCudaCoreInfo` returns no FP32 CUDA Core row for `cc`.
int CudaFpusPerCoreFallback(CudaComputeCapability cc);

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_CORE_INFO_TABLE_H_
