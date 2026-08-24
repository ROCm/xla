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

#include "xla/stream_executor/rocm/rocm_cache_bandwidth.h"

#include <cstdint>

#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor::gpu {
namespace {

// Bytes read per clock by one XCD's L2.
//
// CDNA3 whitepaper p.9: "On the read side each channel can read out a 128-byte
// cache line and the L2 cache can sustain four requests from different CUs per
// cycle for a combined throughput of 2KBytes/clock for each XCD."
// CDNA4 whitepaper p.9 describes the same 16 channels "each capable of a full
// 128B cache line read and a 64B write per cycle", so the width is unchanged.
constexpr int64_t kL2ReadBytesPerClockPerXcd = 2048;

// Bytes read per clock by the whole Infinity Cache.
//
// CDNA3 whitepaper p.10: "Each stack of HBM memory is associated with 16
// parallel channels. A channel is 64-bytes wide ... In total, there are eight
// stacks of HBM across the four IODs, for 128 channels or 256MB of data."
// 128 channels x 64 B = 8192 B/clk. CDNA4 p.10 says the Infinity Cache is
// "largely unchanged in organization" with the same 16 channels x 64 B per
// stack across 8 stacks.
constexpr int64_t kInfinityCacheReadBytesPerClock = 8192;

// Sanity bound. A clock outside this range means the query returned garbage
// (or a units mismatch), and a bogus bandwidth is worse than none.
constexpr double kMinPlausibleClockGhz = 0.1;
constexpr double kMaxPlausibleClockGhz = 10.0;

bool IsPlausibleClock(double clock_ghz) {
  return clock_ghz >= kMinPlausibleClockGhz &&
         clock_ghz <= kMaxPlausibleClockGhz;
}

// The two generations whose cache geometry is modeled. Both have a per-XCD L2
// behind a device-wide Infinity Cache, with identical channel widths.
bool HasModeledCacheGeometry(const RocmComputeCapability& cc) {
  return cc.gfx9_mi300_series();
}

}  // namespace

int64_t GetRocmL2CacheBandwidth(const RocmComputeCapability& cc,
                                int64_t num_xcd, double gfx_clock_ghz) {
  if (!HasModeledCacheGeometry(cc)) return 0;
  if (num_xcd <= 0 || !IsPlausibleClock(gfx_clock_ghz)) return 0;

  return static_cast<int64_t>(kL2ReadBytesPerClockPerXcd * num_xcd *
                              gfx_clock_ghz * 1e9);
}

int64_t GetRocmLastLevelCacheBandwidth(const RocmComputeCapability& cc,
                                       double fabric_clock_ghz) {
  if (!HasModeledCacheGeometry(cc)) return 0;
  if (!IsPlausibleClock(fabric_clock_ghz)) return 0;

  return static_cast<int64_t>(kInfinityCacheReadBytesPerClock *
                              fabric_clock_ghz * 1e9);
}

}  // namespace stream_executor::gpu
