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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_DEVICE_MAP_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_DEVICE_MAP_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "rocm/include/rocprofiler-sdk/agent.h"

// Maps rocprofiler-sdk agents to HIP device numbers. The SDK lists every GPU
// in the machine, while HIP numbers only the GPUs that HIP_VISIBLE_DEVICES and
// ROCR_VISIBLE_DEVICES leave visible, in their order. The two agree on each
// GPU's PCI address.

namespace xla {
namespace profiler {

struct PciAddress {
  uint32_t domain = 0;
  uint32_t bus = 0;
  uint32_t device = 0;
  uint32_t function = 0;

  bool operator==(const PciAddress& other) const {
    return domain == other.domain && bus == other.bus &&
           device == other.device && function == other.function;
  }
};

// Parses "dddd:bb:dd.f" in hex, as printed by hipDeviceGetPCIBusId. The domain
// can have more than 4 digits.
std::optional<PciAddress> ParsePciBusId(absl::string_view bus_id);

// KFD packs the agent's address into location_id as bus << 8 | device << 3 |
// function, and HIP decodes the same value. On a partitioned GPU the function
// bits carry the partition number (Linux 6.12 and later; 6.5 for MI300), so
// each partition has its own address.
PciAddress PciAddressOf(const rocprofiler_agent_v0_t& agent);

struct HipDeviceMap {
  // Key: agent handle. Only GPU agents that matched a HIP device appear.
  absl::flat_hash_map<uint64_t, uint32_t> hip_device_of_agent;
  // Index: HIP device number. A device that matched no agent holds a
  // placeholder of type ROCPROFILER_AGENT_TYPE_NONE.
  std::vector<rocprofiler_agent_v0_t> gpu_agents;
};

// hip[k] is the PCI address HIP reported for device k, or nullopt if it
// reported none. Each HIP device is matched to the only GPU agent at its
// address. A device is left unmatched if another HIP device has its address,
// as partitions do on older kernels, or if no GPU agent or several have it.
HipDeviceMap BuildHipDeviceMap(absl::Span<const rocprofiler_agent_v0_t> agents,
                               absl::Span<const std::optional<PciAddress>> hip);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_DEVICE_MAP_H_
