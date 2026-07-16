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

#include "xla/stream_executor/rocm/rocm_flops_table.h"

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace stream_executor {
namespace gpu {

// FLOPs per CU per cycle (FMA = 2 FLOPs) per gfx architecture.
//
// These rates are architectural constants (independent of CU count and clock).
// Each was derived by dividing AMD's published peak throughput by
// (compute_units * peak_engine_clock), then confirming the result is the exact
// hardware rate:   rate = peak_flops / (cu_count * clock_hz).
// This is the same idea as xla/stream_executor/cuda/cuda_core_info_table.cc,
// which encodes per-SM-per-cycle op rates for NVIDIA.
//
// Matrix rates are the dense (non-sparse) MFMA rates. Vector rates are the VALU
// (non-matrix) rates. bf16/fp16 are equal on CDNA2+ but differ on CDNA1, where
// bf16 MFMA runs at half fp16.
static const absl::flat_hash_map<absl::string_view, RocmFlopsPerCuPerCycle>&
GflopsTable() {
  static const auto* const kTable =
      new absl::flat_hash_map<absl::string_view, RocmFlopsPerCuPerCycle>{
          // gfx908 -- MI100 (CDNA1), 120 CU @ 1.502 GHz. MFMA introduced.
          // Source: ROCm docs "AMD Instinct MI100 microarchitecture":
          //   FP64 vec 11.5, FP32 vec 23.1, FP16 matrix 184.6, BF16 matrix 92.3
          //   TFLOPS -> /(120*1.502e9) -> 64/128/1024/512. bf16 = fp16/2 (CDNA1).
          {"gfx908",
           {/*f64_vector=*/64.0,
            /*f32_vector=*/128.0,
            /*bf16_matrix=*/512.0,
            /*f16_matrix=*/1024.0,
            /*f8_matrix=*/0.0,  // no fp8 MFMA on CDNA1
            /*has_matrix_unit=*/true}},
          // gfx90a -- MI200/MI250(X) (CDNA2), 104 CU/GCD @ 1.7 GHz. Full-rate
          // FP64. Source: ROCm docs "AMD Instinct MI250 microarchitecture"
          // (per-GCD): FP64 vec 45.3, FP32 vec 45.3, FP16/BF16 matrix 362.1
          //   TFLOPS -> /(104*1.7e9) -> 256/256/2048/2048.
          {"gfx90a",
           {/*f64_vector=*/256.0,
            /*f32_vector=*/256.0,
            /*bf16_matrix=*/2048.0,
            /*f16_matrix=*/2048.0,
            /*f8_matrix=*/0.0,  // no fp8 MFMA on CDNA2
            /*has_matrix_unit=*/true}},
          // gfx942 -- MI300A/MI300X (CDNA3), 304 CU (MI300X) @ 2.1 GHz. fp8 MFMA
          // added. Source: AMD Instinct MI300X data sheet:
          //   FP64 vec 81.7, FP32 vec 163.4, FP16/BF16 matrix 1307.4,
          //   FP8 matrix 2614.9 TFLOPS -> /(304*2.1e9) -> 128/256/2048/2048/4096.
          {"gfx942",
           {/*f64_vector=*/128.0,
            /*f32_vector=*/256.0,
            /*bf16_matrix=*/2048.0,
            /*f16_matrix=*/2048.0,
            /*f8_matrix=*/4096.0,
            /*has_matrix_unit=*/true}},
          // gfx950 -- MI350X/MI355X (CDNA4), 256 CU @ up to 2.4 GHz. ~2x CDNA3
          // matrix throughput. Source: AMD Instinct MI355X brief:
          //   FP64 vec 78.6, FP32 vec 157.3 TFLOPS; dense FP16/BF16 matrix 2.5
          //   PFLOPS, FP8 matrix 5.0 PFLOPS -> /(256*2.4e9) ->
          //   128/256/4096/4096/8192.
          {"gfx950",
           {/*f64_vector=*/128.0,
            /*f32_vector=*/256.0,
            /*bf16_matrix=*/4096.0,
            /*f16_matrix=*/4096.0,
            /*f8_matrix=*/8192.0,
            /*has_matrix_unit=*/true}},
          // RDNA has no MFMA matrix unit; report the VALU fp32 vector rate.
          // gfx1030 -- RDNA2 (RX 6800/6900): 2 SIMD32 * FMA(x2) = 128/CU/cycle.
          //   Check: RX 6900 XT 23.0 TFLOPS / (80 CU * 2.25 GHz) = 128.
          {"gfx1030",
           {/*f64_vector=*/8.0,  // 1/16 rate consumer fp64
            /*f32_vector=*/128.0,
            /*bf16_matrix=*/0.0,
            /*f16_matrix=*/0.0,
            /*f8_matrix=*/0.0,
            /*has_matrix_unit=*/false}},
          // gfx1100 -- RDNA3 (RX 7900): dual-issue VALU peak = 256/CU/cycle.
          //   Check: RX 7900 XTX 61.4 TFLOPS / (96 CU * 2.5 GHz) = 256.
          // (WMMA exists on RDNA3 but is not modeled as a matrix unit here.)
          {"gfx1100",
           {/*f64_vector=*/8.0,
            /*f32_vector=*/256.0,
            /*bf16_matrix=*/0.0,
            /*f16_matrix=*/0.0,
            /*f8_matrix=*/0.0,
            /*has_matrix_unit=*/false}},
      };
  return *kTable;
}

std::optional<RocmFlopsPerCuPerCycle> GetRocmFlopsPerCuPerCycle(
    absl::string_view gfx_version) {
  const auto& table = GflopsTable();
  auto it = table.find(gfx_version);
  if (it == table.end()) {
    return std::nullopt;
  }
  return it->second;
}

double GetRocmPeakTeraflopsPerSecond(absl::string_view gfx_version,
                                     uint32_t cu_count, double clock_hz) {
  if (cu_count == 0 || clock_hz <= 0.0) {
    return 0.0;
  }
  std::optional<RocmFlopsPerCuPerCycle> rates =
      GetRocmFlopsPerCuPerCycle(gfx_version);
  if (!rates.has_value()) {
    return 0.0;
  }
  // Headline peak: matrix bf16 if the arch has a matrix unit, else vector fp32.
  double flops_per_cu_per_cycle =
      rates->has_matrix_unit ? rates->bf16_matrix : rates->f32_vector;
  if (flops_per_cu_per_cycle <= 0.0) {
    return 0.0;
  }
  double flops_per_second =
      static_cast<double>(cu_count) * flops_per_cu_per_cycle * clock_hz;
  return flops_per_second / 1e12;
}

}  // namespace gpu
}  // namespace stream_executor
