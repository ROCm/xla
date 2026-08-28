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

#include "xla/stream_executor/gpu/core_info_table.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_core_info_table.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/core_info_table_types.h"
#include "xla/stream_executor/rocm/rocm_core_info_table.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

// Number of FP32 FMA units per core assumed for a backend that contributes no
// table at all. Only oneAPI is in that group today: sycl_device_description.cc
// computes fpus_per_core directly from Level Zero device properties and never
// calls in here.
constexpr int kUntabulatedFpusPerCore = 128;

absl::flat_hash_map<int, DTypeCoreInfo> MakeBitwidthToRowMap(
    const std::vector<DTypeCoreInfo>& rows, bool is_float) {
  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_row;
  for (const auto& row : rows) {
    if (row.dtype.is_float != is_float) {
      continue;
    }
    bitwidth_to_row[row.dtype.bitwidth] = row;
  }
  return bitwidth_to_row;
}

void AddDTypeInfoToDesc(
    xla::PrimitiveType dtype, float base_clock_rate_ghz,
    const absl::flat_hash_map<int, DTypeCoreInfo>& bitwidth_to_row,
    ExecutionUnitDescription& desc) {
  int bitwidth = xla::primitive_util::BitWidth(dtype);
  const auto bitwidth_it = bitwidth_to_row.find(bitwidth);
  if (bitwidth_it == bitwidth_to_row.end()) {
    return;
  }
  const DTypeCoreInfo& perf_info = bitwidth_it->second;
  float clock_rate_ghz = perf_info.clock_scale * base_clock_rate_ghz;
  desc.SetRateInfo(dtype, ExecutionUnitDescription::RateInfo{
                              /*units_per_core=*/perf_info.units_per_core,
                              /*clock_rate_ghz=*/clock_rate_ghz,
                              /*ops_per_clock=*/perf_info.ops_per_clock});
}

ExecutionUnitDescription CreateEuDescription(
    float base_clock_rate_ghz, const std::vector<DTypeCoreInfo>& perf_rows) {
  ExecutionUnitDescription desc;
  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_float_row =
      MakeBitwidthToRowMap(perf_rows, /*is_float=*/true);
  xla::primitive_util::FloatingPointTypeForEach([&](auto dtype) {
    AddDTypeInfoToDesc(dtype, base_clock_rate_ghz, bitwidth_to_float_row, desc);
  });

  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_int_row =
      MakeBitwidthToRowMap(perf_rows, /*is_float=*/false);
  xla::primitive_util::IntegralTypeForEach([&](auto dtype) {
    AddDTypeInfoToDesc(dtype, base_clock_rate_ghz, bitwidth_to_int_row, desc);
  });

  return desc;
}

// Returns the rows the owning backend records for `cc`, or nullptr if the
// architecture is not tabulated or its backend has no table.
const CoreInfoRows* FindCoreInfoForCapability(const GpuComputeCapability& cc) {
  if (const CudaComputeCapability* cuda_cc = cc.cuda_compute_capability()) {
    return FindCudaCoreInfo(*cuda_cc);
  }
  if (const RocmComputeCapability* rocm_cc = cc.rocm_compute_capability()) {
    return FindRocmCoreInfo(*rocm_cc);
  }
  return nullptr;
}

// Returns the 32-bit float vector row of `rows`, or nullptr if `rows` is null
// or has no such row. The returned pointer aliases the backend's table, which
// is a function local static and outlives every caller.
const DTypeCoreInfo* FindFp32VectorRow(const CoreInfoRows* rows) {
  if (rows == nullptr) {
    return nullptr;
  }
  for (const DTypeCoreInfo& row : rows->vector_infos) {
    if (row.dtype.is_float && row.dtype.bitwidth == 32) {
      return &row;
    }
  }
  return nullptr;
}

}  // namespace

void FillExecutionUnitDesc(const GpuComputeCapability& cc,
                           float base_clock_rate_ghz, DeviceDescription& desc) {
  const CoreInfoRows* rows = FindCoreInfoForCapability(cc);
  if (rows == nullptr) {
    return;
  }
  // An empty row list leaves the corresponding field unset, which makes
  // consumers fall back to their own estimates.
  if (!rows->vector_infos.empty()) {
    desc.set_scalar_unit_description(
        CreateEuDescription(base_clock_rate_ghz, rows->vector_infos));
  }
  if (!rows->matrix_infos.empty()) {
    desc.set_matrix_unit_description(
        CreateEuDescription(base_clock_rate_ghz, rows->matrix_infos));
  }
}

int GetFpusPerCore(const GpuComputeCapability& cc) {
  const DTypeCoreInfo* fp32_row =
      FindFp32VectorRow(FindCoreInfoForCapability(cc));
  if (fp32_row != nullptr) {
    return fp32_row->units_per_core;
  }
  // The architecture is not tabulated; ask its backend what to assume.
  if (const CudaComputeCapability* cuda_cc = cc.cuda_compute_capability()) {
    return CudaFpusPerCoreFallback(*cuda_cc);
  }
  if (const RocmComputeCapability* rocm_cc = cc.rocm_compute_capability()) {
    return RocmFpusPerCoreFallback(*rocm_cc);
  }
  return kUntabulatedFpusPerCore;
}

}  // namespace gpu
}  // namespace stream_executor
