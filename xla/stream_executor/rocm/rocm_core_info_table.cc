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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor {
namespace gpu {
namespace {

// FP32 operations a compute unit can issue in parallel, per GCN architecture.
//
// Each entry was recovered from AMD's published FP32 vector peak as
//   fpus = peak_fp32_flops / (cu_count * clock_hz * 2)
// where the factor of two is the FMA each lane retires per cycle.
//
// The count exceeds the 64 physical lanes on CDNA 2 and later, and on RDNA 3,
// because those architectures issue packed FP32 (two results per lane per
// cycle). That is the intent of the field, which counts operations in flight
// rather than physical units -- the same reason Nvidia reports 128 for Hopper.
constexpr int kFpusPerCuDefault = 128;

const absl::flat_hash_map<absl::string_view, int>& FpusPerCuTable() {
  static const auto* const kTable = new absl::flat_hash_map<absl::string_view,
                                                            int>{
      // MI100 (CDNA 1): 23.1 TFLOPS at 120 CU, 1.502 GHz. No packed FP32.
      {"gfx908", 64},
      // MI250X (CDNA 2): 45.3 TFLOPS per GCD at 104 CU, 1.7 GHz.
      {"gfx90a", 128},
      // MI300X (CDNA 3): 163.4 TFLOPS at 304 CU, 2.1 GHz.
      {"gfx942", 128},
      // MI355X (CDNA 4): 157.3 TFLOPS at 256 CU, 2.4 GHz.
      {"gfx950", 128},
      // RX 6900 XT (RDNA 2): 23.0 TFLOPS at 80 CU, 2.25 GHz.
      {"gfx1030", 64},
      // RX 7900 XTX (RDNA 3): 61.4 TFLOPS at 96 CU, 2.5 GHz. Dual-issue VALU.
      {"gfx1100", 128},
  };
  return *kTable;
}

}  // namespace

int GetRocmFpusPerCore(const RocmComputeCapability& cc) {
  const auto& table = FpusPerCuTable();
  auto it = table.find(cc.gfx_version());
  if (it == table.end()) {
    return kFpusPerCuDefault;
  }
  return it->second;
}

}  // namespace gpu
}  // namespace stream_executor
