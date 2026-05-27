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

#include "xla/stream_executor/rocm/rocm_core_info_table.h"

#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

// Like the CUDA core info table, we group primitive types by (is_float,
// bitwidth) so the rows below stay arch-focused. All numbers are encoded as
// `units_per_cu * ops_per_clock * 2 (FMA) = FLOPS/clock/CU`, matching the AMD
// CDNA white papers (FMA = 2 ops).
struct DTypeDescr {
  bool is_float;
  int bitwidth;
};

constexpr DTypeDescr kI8 = DTypeDescr{/*is_float=*/false, 8};

constexpr DTypeDescr kF4 = DTypeDescr{/*is_float=*/true, 4};
constexpr DTypeDescr kF6 = DTypeDescr{/*is_float=*/true, 6};
constexpr DTypeDescr kF8 = DTypeDescr{/*is_float=*/true, 8};
constexpr DTypeDescr kF16 = DTypeDescr{/*is_float=*/true, 16};
constexpr DTypeDescr kF32 = DTypeDescr{/*is_float=*/true, 32};
constexpr DTypeDescr kF64 = DTypeDescr{/*is_float=*/true, 64};

struct DTypeCoreInfo {
  DTypeDescr dtype;
  int units_per_cu;
  int ops_per_clock = 1;    // FMA counted as 1 op (consumers multiply by 2).
  float clock_scale = 1.0;  // Ratio of unit clock vs base device clock.
};

const std::vector<DTypeCoreInfo>* FindCoreInfoForDType(
    const RocmComputeCapability& cc, bool is_matrix) {
  struct CoreInfoTableForArch {
    absl::string_view gfx_version;
    std::vector<DTypeCoreInfo> vector_infos;
    std::vector<DTypeCoreInfo> matrix_infos;
  };

  // =============== Sources ===============
  // [CDNA1] Introducing AMD CDNA Architecture, Table 1, p.7 (MI100).
  //   https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/
  //   white-papers/amd-cdna-white-paper.pdf
  // [CDNA2] AMD CDNA 2 White Paper, Table 1, p.10 (MI250X).
  //   https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/
  //   white-papers/amd-cdna2-white-paper.pdf
  // [CDNA3] AMD CDNA 3 White Paper, Table 1, p.7 (MI300X/MI325X).
  //   https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/
  //   white-papers/amd-cdna-3-white-paper.pdf
  // [CDNA4] Introducing AMD CDNA 4 Architecture, Table 1, p.8 (MI355X).
  //   https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/
  //   white-papers/amd-cdna-4-architecture-whitepaper.pdf
  //
  // All CDNA CUs contain 4 SIMD/Matrix engines; we encode that as
  // units_per_cu=4 on matrix rows. Vector rows use units_per_cu = the count of
  // FP32-FMA-equivalent lanes such that units * ops * 2 reproduces the
  // FLOPS/clock/CU values in the tables above.

  static const absl::NoDestructor<std::vector<CoreInfoTableForArch>> kTable(
      std::vector<CoreInfoTableForArch>{
          // ===== gfx908 / CDNA1 / MI100 =====
          {"gfx908",
           /*vector_infos=*/
           {
               // DType, Units/CU, Ops/Clk        => FLOPS/clock/CU
               {kF32, 64, 1},                    // 128 [CDNA1]
               {kF16, 64, 1},                    // 128 (packed; assumed equal)
               {kF64, 32, 1},                    //  64 [CDNA1]
           },
           /*matrix_infos=*/
           {
               // DType, Units/CU, Ops/Clk        => FLOPS/clock/CU
               {kF16, 4, 128},                   // 1024 [CDNA1]
               // bf16: 512 [CDNA1]. Encoded via bitwidth=16 fall-through is
               // ambiguous with FP16; we keep one entry per bitwidth and
               // accept the FP16 rate for both (consistent with the CUDA
               // table convention). See README in this file's header.
               {kF32, 4, 32},                    //  256 [CDNA1]
               {kI8, 4, 128},                    // 1024 [CDNA1]
           }},
          // ===== gfx90a / CDNA2 / MI210/MI250/MI250X =====
          {"gfx90a",
           /*vector_infos=*/
           {
               {kF32, 64, 1},                    // 128 [CDNA2]
               {kF16, 64, 1},                    // 128 (packed; assumed equal)
               {kF64, 64, 1},                    // 128 [CDNA2]
           },
           /*matrix_infos=*/
           {
               {kF16, 4, 128},                   // 1024 [CDNA2]
               {kF32, 4, 32},                    //  256 [CDNA2]
               {kF64, 4, 32},                    //  256 [CDNA2]
               {kI8, 4, 128},                    // 1024 [CDNA2]
           }},
          // ===== gfx942 / CDNA3 / MI300A/MI300X/MI325X =====
          {"gfx942",
           /*vector_infos=*/
           {
               {kF32, 128, 1},                   // 256 [CDNA3]
               {kF16, 128, 1},                   // 256 (packed; assumed equal)
               {kF64, 64, 1},                    // 128 [CDNA3]
           },
           /*matrix_infos=*/
           {
               {kF8, 4, 512},                    // 4096 [CDNA3]
               {kF16, 4, 256},                   // 2048 [CDNA3]
               {kF32, 4, 32},                    //  256 [CDNA3]
               {kF64, 4, 32},                    //  256 [CDNA3]
               {kI8, 4, 512},                    // 4096 [CDNA3]
           }},
          // ===== gfx950 / CDNA4 / MI350/MI355X =====
          {"gfx950",
           /*vector_infos=*/
           {
               {kF32, 128, 1},                   // 256 [CDNA4]
               {kF16, 128, 1},                   // 256 [CDNA4] (vector FP16)
               {kF64, 64, 1},                    // 128 [CDNA4]
           },
           /*matrix_infos=*/
           {
               {kF4, 4, 2048},                   // 16384 [CDNA4] MXFP4
               {kF6, 4, 2048},                   // 16384 [CDNA4] MXFP6
               {kF8, 4, 1024},                   //  8192 [CDNA4]
               {kF16, 4, 512},                   //  4096 [CDNA4]
               {kF32, 4, 32},                    //   256 [CDNA4]
               {kF64, 4, 16},                    //   128 [CDNA4] (halved vs CDNA3)
               {kI8, 4, 1024},                   //  8192 [CDNA4]
           }},
      });

  for (const auto& entry : *kTable) {
    if (cc.gfx_version() == entry.gfx_version) {
      return is_matrix ? &entry.matrix_infos : &entry.vector_infos;
    }
  }
  return nullptr;
}

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
  const auto it = bitwidth_to_row.find(bitwidth);
  if (it == bitwidth_to_row.end()) {
    return;
  }
  const DTypeCoreInfo& info = it->second;
  float clock_rate_ghz = info.clock_scale * base_clock_rate_ghz;
  desc.SetRateInfo(dtype, ExecutionUnitDescription::RateInfo{
                              /*units_per_core=*/info.units_per_cu,
                              /*clock_rate_ghz=*/clock_rate_ghz,
                              /*ops_per_clock=*/info.ops_per_clock});
}

ExecutionUnitDescription CreateEuDescription(
    float base_clock_rate_ghz, const std::vector<DTypeCoreInfo>& rows) {
  ExecutionUnitDescription desc;
  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_float_row =
      MakeBitwidthToRowMap(rows, /*is_float=*/true);
  xla::primitive_util::FloatingPointTypeForEach([&](auto dtype) {
    AddDTypeInfoToDesc(dtype, base_clock_rate_ghz, bitwidth_to_float_row, desc);
  });
  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_int_row =
      MakeBitwidthToRowMap(rows, /*is_float=*/false);
  xla::primitive_util::IntegralTypeForEach([&](auto dtype) {
    AddDTypeInfoToDesc(dtype, base_clock_rate_ghz, bitwidth_to_int_row, desc);
  });
  return desc;
}

}  // namespace

void FillExecutionUnitDesc(const RocmComputeCapability& cc,
                           float base_clock_rate_ghz, DeviceDescription& desc) {
  if (const std::vector<DTypeCoreInfo>* vec_rows =
          FindCoreInfoForDType(cc, /*is_matrix=*/false)) {
    desc.set_scalar_unit_description(
        CreateEuDescription(base_clock_rate_ghz, *vec_rows));
  }
  if (const std::vector<DTypeCoreInfo>* mat_rows =
          FindCoreInfoForDType(cc, /*is_matrix=*/true)) {
    desc.set_matrix_unit_description(
        CreateEuDescription(base_clock_rate_ghz, *mat_rows));
  }
}

int GetFpusPerCore(const RocmComputeCapability& cc) {
  if (const std::vector<DTypeCoreInfo>* vec_rows =
          FindCoreInfoForDType(cc, /*is_matrix=*/false)) {
    for (const auto& row : *vec_rows) {
      if (row.dtype.is_float && row.dtype.bitwidth == 32) {
        return row.units_per_cu;
      }
    }
  }
  // Fallback if the gfx target isn't in the table yet. Historically all ROCm
  // devices reported 128; preserve that for unknown arches.
  return 128;
}

}  // namespace gpu
}  // namespace stream_executor
