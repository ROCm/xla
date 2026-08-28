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

#ifndef XLA_STREAM_EXECUTOR_GPU_CORE_INFO_TABLE_H_
#define XLA_STREAM_EXECUTOR_GPU_CORE_INFO_TABLE_H_

#include "xla/stream_executor/device_description.h"

// Execution unit throughput tables. Each GPU backend records the per-core rates
// of its vector and systolic units for the architectures it knows about, in
// cuda/cuda_core_info_table.cc and rocm/rocm_core_info_table.cc. This header
// dispatches to the right one based on the compute capability, so that callers
// which hold only a DeviceDescription do not have to name a backend.
//
// The row types the backend tables are written in live in
// core_info_table_types.h.

namespace stream_executor {
namespace gpu {

// Fills the scalar and matrix unit fields in `desc` with the vector ALU and
// systolic unit throughput descriptions recorded for `cc`, if the backend has
// a table for it. Fields are left unset for architectures that are not in a
// table, and for backends that have no table at all, which makes consumers
// fall back to their own estimates. See
// GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs.
//
// `base_clock_rate_ghz` must be the device's clock rate; passing 0 produces
// unit descriptions with a peak throughput of zero.
void FillExecutionUnitDesc(const GpuComputeCapability& cc,
                           float base_clock_rate_ghz, DeviceDescription& desc);

// Returns the number of FP32 FMA units per core for `cc`. Used as the scalar
// fallback by the GPU performance model. The value matches the count semantics
// expected by HloCostAnalysis, which separately multiplies by 2 to convert FMA
// to FLOPs.
int GetFpusPerCore(const GpuComputeCapability& cc);

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_CORE_INFO_TABLE_H_
