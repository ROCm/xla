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

#ifndef XLA_STREAM_EXECUTOR_GPU_CORE_INFO_TABLE_TYPES_H_
#define XLA_STREAM_EXECUTOR_GPU_CORE_INFO_TABLE_TYPES_H_

#include <vector>

// Row types shared by the per-backend execution unit throughput tables. This
// header is deliberately dependency free so that a backend table can depend on
// it without depending on the dispatch layer in core_info_table.h, which in
// turn depends on every backend table.
//
// The two backend tables must agree on the conventions documented here, since
// consumers apply the same arithmetic to both. In particular: FMA counts as one
// op, and rows are keyed by bitwidth rather than by primitive type.

namespace stream_executor {
namespace gpu {

// Instead of using base primitive types we use a simple description that maps
// to several primitive types at once. This way we can keep the backend tables
// more abstract: one row covers every primitive type of a given bitwidth.
struct DTypeDescr {
  bool is_float;
  int bitwidth;
};

constexpr DTypeDescr kI8 = DTypeDescr{/*is_float=*/false, 8};
constexpr DTypeDescr kI32 = DTypeDescr{/*is_float=*/false, 32};

constexpr DTypeDescr kF4 = DTypeDescr{/*is_float=*/true, 4};
constexpr DTypeDescr kF6 = DTypeDescr{/*is_float=*/true, 6};
constexpr DTypeDescr kF8 = DTypeDescr{/*is_float=*/true, 8};
constexpr DTypeDescr kF16 = DTypeDescr{/*is_float=*/true, 16};
constexpr DTypeDescr kF32 = DTypeDescr{/*is_float=*/true, 32};
constexpr DTypeDescr kF64 = DTypeDescr{/*is_float=*/true, 64};

// Throughput of one execution unit for every primitive type matching `dtype`.
// The peak for a device is
//   units_per_core * ops_per_clock * 2 (FMA) * core_count * clock_rate_ghz
// where the last two factors come from the DeviceDescription rather than from
// the table, because they vary between SKUs of the same architecture.
struct DTypeCoreInfo {
  DTypeDescr dtype;
  int units_per_core;
  int ops_per_clock = 1;    // Note: FMA is considered 1 op.
  float clock_scale = 1.0;  // Ratio of clock rate of this unit vs base device.
};

// The rows a backend table holds for a single architecture. `vector_infos`
// describes the general purpose ALUs (CUDA cores, CDNA vector units) and
// `matrix_infos` the systolic units (tensor cores, MFMA units). Either may be
// empty if the architecture has no such unit or it is not modeled yet.
struct CoreInfoRows {
  std::vector<DTypeCoreInfo> vector_infos;
  std::vector<DTypeCoreInfo> matrix_infos;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_CORE_INFO_TABLE_TYPES_H_
