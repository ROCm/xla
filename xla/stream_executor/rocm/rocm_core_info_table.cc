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

#include "xla/stream_executor/rocm/rocm_core_info_table.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

// Instead of using base primitive types we use a simple description that maps
// to several primitive types at once. This way we can keep the types in the
// table below more abstract.
struct DTypeDescr {
  bool is_float;
  int bitwidth;
};

constexpr DTypeDescr kI8 = DTypeDescr{/*is_float=*/false, 8};
constexpr DTypeDescr kI32 = DTypeDescr{/*is_float=*/false, 32};

constexpr DTypeDescr kF8 = DTypeDescr{/*is_float=*/true, 8};
constexpr DTypeDescr kF16 = DTypeDescr{/*is_float=*/true, 16};
constexpr DTypeDescr kF32 = DTypeDescr{/*is_float=*/true, 32};
constexpr DTypeDescr kF64 = DTypeDescr{/*is_float=*/true, 64};

struct DTypeCoreInfo {
  DTypeDescr dtype;
  int units_per_cu;
  int ops_per_clock = 1;    // Note: FMA is considered 1 op.
  float clock_scale = 1.0;  // Ratio of clock rate of this unit vs base device.
};

const std::vector<DTypeCoreInfo>* FindCoreInfoForDType(
    const std::string& gfx_version, bool is_matrix) {
  struct CoreInfoTableForGfx {
    std::string gfx_version;
    std::vector<DTypeCoreInfo> scalar_infos;
    std::vector<DTypeCoreInfo> matrix_infos;
  };

  // =============== Sources ===============
  // When adding a new source make sure to include the version.
  //
  // [CDNA1WP] AMD CDNA Architecture Whitepaper
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf
  //
  // [CDNA1ISA] CDNA1 ISA Reference Guide
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf
  //
  // [CDNA2WP] AMD CDNA 2 Architecture Whitepaper
  // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
  //
  // [CDNA2ISA] CDNA2 ISA Reference Guide
  // https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf
  //
  // [CDNA3WP] AMD CDNA 3 Architecture Whitepaper
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
  //
  // [CDNA3ISA] CDNA3 ISA Reference Guide
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  //
  // [CDNA4WP] AMD CDNA 4 Architecture Whitepaper
  // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
  //
  // [ROCmSpecs] ROCm Hardware Specs Table
  // https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html

  // =============== Constants ===============
  // CDNA matrix units (MFMA) run at the base device clock rate.
  constexpr int kMfmaPerCu = 4;  // All CDNA generations have 4 MFMA per CU.

  // =============== Lookup table ===============
  static const absl::NoDestructor<std::vector<CoreInfoTableForGfx>> kTable(
      std::vector<CoreInfoTableForGfx>{
          {"gfx908",
           // [CDNA1WP] MI100: 120 CUs, 7680 SPs → 64 SPs/CU.
           // [CDNA1ISA] Ch.1: 4 SIMDs × 16 lanes = 64 SPs per CU.
           // FP16 packed: 2 FP16 ops per SP per clock via V_PK_FMA_F16.
           // FP64 at 1:2 rate vs FP32.
           /*scalar_infos=*/
           {
               // DType, Units/CU, Ops/Clk
               {kF16, 64, 2},  // [CDNA1ISA]: packed FP16
               {kF32, 64, 1},
               {kF64, 32, 1},  // [CDNA1WP]: 1:2 FP64:FP32
               {kI32, 64, 1},
           },
           // [CDNA1WP] 4 Matrix Cores per CU.
           // [CDNA1ISA] Ch.7: MFMA instructions.
           // MI100 official peak: FP16=184.6 TFLOPS, FP32=92.3 TFLOPS
           // (120 CUs, 1.502 GHz). Verified: 120*4*128*1.502*2 = 184.6T.
           // No FP64 matrix, no FP8 matrix on CDNA 1.
           /*matrix_infos=*/
           {
               // DType  Units/CU  Ops/Clk
               {kF16, kMfmaPerCu, 128},
               {kF32, kMfmaPerCu, 64},
           }},

          {"gfx90a",
           // [CDNA2WP] MI250X: 220 CUs (per GCD), 14080 SPs → 64 SPs/CU.
           // FP64 at 1:1 ratio via natively 64-bit ALUs.
           // FP32: 128 FLOPS/clk/CU via packed FP32 (V_PK_FMA_F32).
           // [CDNA2ISA]: Packed FP32 doubles effective throughput.
           /*scalar_infos=*/
           {
               {kF16, 64, 2},  // [CDNA2ISA]: packed FP16
               {kF32, 64, 1},  // Base rate (packed gives 2x but reported
                                // separately via fpus_per_core compat)
               {kF64, 64, 1},  // [CDNA2WP]: 1:1 FP64:FP32 (unique)
               {kI32, 64, 1},
           },
           // [CDNA2WP] 4 MFMA units per CU.
           // MI250X official peak (per GCD, 110 CUs, 1.7 GHz):
           // FP16=383.0T, FP64=95.7T.
           // Verified: 110*4*256*1.7*2 = 383.0T.
           // FP64 matrix: V_MFMA_F64_16X16X4F64.
           // BF16 matrix support added (same rate as FP16).
           /*matrix_infos=*/
           {
               {kF16, kMfmaPerCu, 256},
               {kF32, kMfmaPerCu, 128},
               {kF64, kMfmaPerCu, 64},
           }},

          {"gfx942",
           // [CDNA3WP] MI300X: 304 CUs, 64 SPs/CU physical.
           // Dual-issue FP32: 256 FP32 FLOPS/clk/CU → 128 FMA/clk/CU.
           // [CDNA3ISA]: Dual-issue vector path.
           // FP64 at 1:2 ratio vs effective FP32.
           /*scalar_infos=*/
           {
               {kF16, 128, 1},  // [CDNA3WP]: dual-issue, same rate as FP32
               {kF32, 128, 1},  // [CDNA3WP]: dual-issue 256 FLOPS/clk/CU
               {kF64, 64, 1},   // [CDNA3WP]: 1:2 FP64:FP32
               {kI32, 64, 1},
           },
           // [CDNA3WP] 4 MFMA units per CU.
           // MI300X official peak (304 CUs, 2.1 GHz):
           // FP8=2615T, FP16=1307T, FP32=654T, FP64=163T.
           // Verified: 304*4*512*2.1*2 = 2614.9T (FP8).
           // FP16/BF16: 256 FMA ops/clk per MFMA unit.
           // FP32 (TF32): 128 ops/clk. FP64: 32 ops/clk.
           /*matrix_infos=*/
           {
               {kI8, kMfmaPerCu, 512},
               {kF8, kMfmaPerCu, 512},
               {kF16, kMfmaPerCu, 256},
               {kF32, kMfmaPerCu, 128},
               {kF64, kMfmaPerCu, 32},
           }},

          {"gfx950",
           // [CDNA4WP] MI350X: 256 CUs (8 XCDs × 32 active CUs).
           // 128 SPs/CU (doubled from CDNA3's 64).
           // FP32: 256 FLOPS/clk/CU (128 SPs × 2 via FMA).
           // FP64: 1:2 ratio vs FP32.
           /*scalar_infos=*/
           {
               {kF16, 128, 1},
               {kF32, 128, 1},
               {kF64, 64, 1},
               {kI32, 64, 1},
           },
           // [CDNA4WP] 4 MFMA units per CU.
           // MI350X official peak (256 CUs, 2.4 GHz):
           // FP8 throughput doubled per CU vs CDNA3 (ISSCC 2026).
           // 8192 FP8 FLOPS/clk/CU = 4 * 1024 * 2(FMA).
           // FP16/BF16: same per-CU rate as CDNA3 = 256 ops/clk per unit.
           // FP32 (TF32): 128 ops/clk. FP64: 32 ops/clk.
           // New: hardware MXFP8/MXFP6/MXFP4 support.
           /*matrix_infos=*/
           {
               {kI8, kMfmaPerCu, 1024},
               {kF8, kMfmaPerCu, 1024},
               {kF16, kMfmaPerCu, 256},
               {kF32, kMfmaPerCu, 128},
               {kF64, kMfmaPerCu, 32},
           }},
      });

  for (const auto& config : *kTable) {
    if (config.gfx_version == gfx_version) {
      return is_matrix ? &config.matrix_infos : &config.scalar_infos;
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
  const auto& bitwidth_it = bitwidth_to_row.find(bitwidth);
  if (bitwidth_it == bitwidth_to_row.end()) {
    return;
  }
  const DTypeCoreInfo& perf_info = bitwidth_it->second;
  float clock_rate_ghz = perf_info.clock_scale * base_clock_rate_ghz;
  desc.SetRateInfo(dtype, stream_executor::ExecutionUnitDescription::RateInfo{
                              /*units_per_core=*/perf_info.units_per_cu,
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

}  // namespace

void FillExecutionUnitDesc(const RocmComputeCapability& cc,
                           float base_clock_rate_ghz,
                           DeviceDescription& desc) {
  std::string gfx_version = cc.gfx_version();

  const std::vector<DTypeCoreInfo>* scalar_rows =
      FindCoreInfoForDType(gfx_version, /*is_matrix=*/false);
  if (scalar_rows != nullptr) {
    ExecutionUnitDescription scalar_desc =
        CreateEuDescription(base_clock_rate_ghz, *scalar_rows);
    desc.set_scalar_unit_description(std::move(scalar_desc));
  }

  const std::vector<DTypeCoreInfo>* matrix_rows =
      FindCoreInfoForDType(gfx_version, /*is_matrix=*/true);
  if (matrix_rows != nullptr) {
    ExecutionUnitDescription matrix_desc =
        CreateEuDescription(base_clock_rate_ghz, *matrix_rows);
    desc.set_matrix_unit_description(std::move(matrix_desc));
  }
}

int GetFpusPerCore(const RocmComputeCapability& cc) {
  // Hardcoded values preserving backward-compatible behavior.
  // TODO(b/xxxx): Align with scalar_unit_description values (64 for gfx908,
  // gfx90a) once downstream consumers are validated.
  std::string gfx_version = cc.gfx_version();
  if (gfx_version == "gfx906") {
    return 64;
  }
  if (gfx_version == "gfx900") {
    return 64;
  }
  return 128;  // gfx908, gfx90a, gfx942, gfx950, and others.
}

}  // namespace gpu
}  // namespace stream_executor
