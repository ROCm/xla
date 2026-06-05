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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_BANDWIDTH_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_BANDWIDTH_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor::gpu {

// Returns the device memory (HBM/GDDR) bandwidth in bytes/second.
//
// HIP reports the memory *controller* clock (UCLK), not the data-rate clock, so
// the legacy `2 * bus_width * clock` formula undercounts on HBM3/HBM3e (~2x) and
// GDDR6 (~8x), where the data PHY runs faster than UCLK. The value is resolved
// in three tiers, first hit wins:
//   1. firmware peak from amd_smi gpu_metrics (vram_max_bandwidth), accurate
//      and board-specific, but only populated on newer datacenter parts;
//   2. a per-gfx peak for known architectures;
//   3. the legacy formula for unmodeled arches (still correct on HBM2/HBM2e,
//      where the reported clock happens to equal the data-rate clock).
//
// `pci_bus_id` is the HIP PCI bus ID string (e.g. "0000:41:00.0"), used for the
// firmware lookup. `mem_bus_width_bits` and `mem_clock_khz` come from
// hipDeviceProp_t (memoryBusWidth, memoryClockRate) and feed the formula.
int64_t GetRocmMemoryBandwidth(absl::string_view pci_bus_id,
                               const RocmComputeCapability& cc,
                               int64_t mem_bus_width_bits,
                               int64_t mem_clock_khz);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_BANDWIDTH_H_
