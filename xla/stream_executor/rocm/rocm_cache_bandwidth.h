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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_CACHE_BANDWIDTH_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_CACHE_BANDWIDTH_H_

#include <cstdint>

#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor::gpu {

// Peak cache read bandwidths, in bytes/second.
//
// No API reports these, so they are computed as `width * clock`. The width is
// an architectural constant taken from the AMD whitepapers and is identical on
// CDNA3 and CDNA4; the clock is queried per device, which is what makes the
// result correct per SKU (MI350X and MI355X share gfx950 but run different
// engine clocks, 2200 vs 2400 MHz).
//
// Both functions return 0 for architectures whose cache geometry is not
// modeled. Callers must leave the corresponding DeviceDescription field unset
// in that case, so the cost model falls back to scaling memory bandwidth.

// Aggregate L2 read bandwidth across all XCDs. `num_xcd` is the number of L2
// instances (from the SMI cache topology) and `gfx_clock_ghz` is the peak
// engine clock, which the L2 runs at because it sits on the XCD.
int64_t GetRocmL2CacheBandwidth(const RocmComputeCapability& cc,
                                int64_t num_xcd, double gfx_clock_ghz);

// Last level (Infinity Cache) read bandwidth. It sits on the IODs rather than
// the XCDs, so it runs at the Infinity Fabric clock, not the engine clock.
//
// Pass 0 for `fabric_clock_ghz` when the clock could not be queried:
// AMDSMI_CLK_TYPE_DF was measured reporting a zero peak on MI350X. The
// documented per-generation clock is then used instead, which is why this can
// still return a value with no clock available.
int64_t GetRocmLastLevelCacheBandwidth(const RocmComputeCapability& cc,
                                       double fabric_clock_ghz);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_CACHE_BANDWIDTH_H_
