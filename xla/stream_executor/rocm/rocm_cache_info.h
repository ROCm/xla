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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_CACHE_INFO_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_CACHE_INFO_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

// What SMI can tell us about a device's cache hierarchy that HIP cannot.
// A zero field means the value was not available; callers must leave the
// corresponding DeviceDescription field unset rather than substituting a
// guess.
struct RocmCacheInfo {
  // Size of the deepest data cache level. On CDNA3/CDNA4 this is the L3
  // (Infinity Cache), already divided by the memory partition mode by the
  // kernel. HIP does not report this at all.
  int64_t last_level_cache_size_bytes = 0;

  // Deliberately no L2 instance count here. amd_smi's num_cache_instance was
  // measured reporting 1 on MI350X, which has 8 XCDs, so it counts distinct
  // cache descriptions rather than physical instances. Use HIP's
  // hipDeviceAttributeNumberOfXccs instead.

  // Peak Infinity Fabric clock in MHz, which the last level cache runs at.
  // Zero on ASICs that do not expose the domain.
  int64_t fabric_clock_mhz = 0;
};

// Queries the cache hierarchy of the device with the given PCI bus ID string
// from HIP (e.g. "0000:41:00.0"). Does not log an error; the caller decides
// what a failure means. Returns Unimplemented when XLA is built against
// rocm_smi rather than amd_smi.
//
// Individually optional fields (the fabric clock) are left at zero rather than
// failing the whole query.
absl::StatusOr<RocmCacheInfo> GetRocmCacheInfo(absl::string_view pci_bus_id);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_CACHE_INFO_H_
