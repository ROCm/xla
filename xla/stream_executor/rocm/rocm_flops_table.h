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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_FLOPS_TABLE_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_FLOPS_TABLE_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"

namespace stream_executor {
namespace gpu {

// Per-arch peak throughput expressed as FLOPs per compute unit (CU) per cycle.
// FMA is counted as 2 FLOPs. Matrix (MFMA) rates apply to the arch's matrix
// units; vector rates apply to the standard VALU pipeline.
//
// These are architectural constants (a function of the gfx generation), not
// values queryable from rocprofiler/HIP at runtime -- hence the static table.
struct RocmFlopsPerCuPerCycle {
  double f64_vector = 0.0;
  double f32_vector = 0.0;
  double bf16_matrix = 0.0;
  double f16_matrix = 0.0;
  double f8_matrix = 0.0;
  bool has_matrix_unit = false;
};

// Returns the per-CU-per-cycle FLOP rates for a gfx architecture, keyed on the
// canonical gfx version string (e.g. "gfx942"), as produced by
// RocmComputeCapability::gfx_version(). Returns nullopt for unknown archs so
// callers can skip emitting a (potentially wrong) peak.
std::optional<RocmFlopsPerCuPerCycle> GetRocmFlopsPerCuPerCycle(
    absl::string_view gfx_version);

// Computes peak TFLOP/s for a device:
//   peak_tflops = cu_count * flops_per_cu_per_cycle * clock_hz / 1e12
// Uses the matrix bf16 rate when the arch has a matrix unit, otherwise the
// vector fp32 rate -- this matches what a training roofline expects as the
// headline compute peak. Returns 0.0 when the arch is unknown or inputs are
// zero, signaling the caller to emit nothing (preserving downstream fallback).
double GetRocmPeakTeraflopsPerSecond(absl::string_view gfx_version,
                                     uint32_t cu_count, double clock_hz);

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_FLOPS_TABLE_H_
