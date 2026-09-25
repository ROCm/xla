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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_SIDE_CACHE_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_SIDE_CACHE_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/rocm/smi_util.h"

namespace stream_executor::gpu {

// Returns the size in bytes of the memory-side cache (Infinity Cache, MALL) in
// front of device memory, or 0 if the device has none or it cannot be
// determined. See DeviceDescription::memory_side_cache_size().
//
// HIP reports only L2, and on CDNA3/CDNA4 only one XCD's share of it, so the
// value is resolved in two tiers and the first hit wins. Tier 1 is the
// driver's cache topology, read over SMI. Tier 2 is a per-gfx size for data
// center parts, whose size is fixed per gfx. RDNA sizes vary by SKU within one
// gfx (96, 80 or 64 MiB on gfx1100), so those rely on SMI alone.
//
// `pci_bus_id` identifies the device to SMI.
int64_t GetRocmMemorySideCacheSize(absl::string_view pci_bus_id,
                                   const RocmComputeCapability& cc);

// Tier 1 given the data caches SMI reported. The driver's cache topology lists
// the memory-side cache as level 3, since it is the only AMD GPU cache behind
// L2. Returns the largest such cache, or 0 if none is listed. Exposed for
// testing.
int64_t MemorySideCacheSizeFromDataCaches(
    absl::Span<const SmiDataCache> caches);

// Tier 2, or 0 for a gfx without a fixed size. Exposed for testing.
int64_t MemorySideCacheSizeForArch(const RocmComputeCapability& cc);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_SIDE_CACHE_H_
