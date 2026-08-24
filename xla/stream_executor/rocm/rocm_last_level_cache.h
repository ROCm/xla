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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_LAST_LEVEL_CACHE_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_LAST_LEVEL_CACHE_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

// Returns the size in bytes of the device's last level data cache.
//
// HIP only reports L2 (hipDeviceProp_t::l2CacheSize), which on CDNA3/CDNA4 is
// private to a single XCD and sits in front of a much larger L3. SMI reads the
// full KFD cache topology, including the L3 entry, already divided by the
// memory partition mode.
//
// pci_bus_id is the PCI bus ID string from HIP (e.g. "0000:41:00.0"). Does not
// log an error; the caller decides what a failure means. Returns Unimplemented
// when XLA is built against rocm_smi rather than amd_smi.
absl::StatusOr<int64_t> GetRocmLastLevelCacheSize(absl::string_view pci_bus_id);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_LAST_LEVEL_CACHE_H_
