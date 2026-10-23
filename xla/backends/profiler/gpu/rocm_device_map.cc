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

#include "xla/backends/profiler/gpu/rocm_device_map.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "rocm/include/rocprofiler-sdk/agent.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"

namespace xla {
namespace profiler {
namespace {

// SimpleHexAtoi alone also accepts whitespace, a sign and a "0x" prefix.
bool ParseHexField(absl::string_view field, uint32_t* value) {
  return absl::c_all_of(field, absl::ascii_isxdigit) &&
         absl::SimpleHexAtoi(field, value);
}

// The GPU agent that is the HIP device at `pci`, or null if none or ambiguous.
const rocprofiler_agent_v0_t* FindGpuAgent(
    absl::Span<const rocprofiler_agent_v0_t> agents,
    absl::Span<const std::optional<PciAddress>> hip,
    const std::optional<PciAddress>& pci) {
  if (!pci.has_value() || absl::c_count(hip, pci) != 1) return nullptr;
  const rocprofiler_agent_v0_t* found = nullptr;
  for (const rocprofiler_agent_v0_t& agent : agents) {
    if (agent.type == ROCPROFILER_AGENT_TYPE_GPU &&
        PciAddressOf(agent) == *pci) {
      if (found != nullptr) return nullptr;
      found = &agent;
    }
  }
  return found;
}

}  // namespace

std::optional<PciAddress> ParsePciBusId(absl::string_view bus_id) {
  std::vector<absl::string_view> fields = absl::StrSplit(bus_id, ':');
  if (fields.size() != 3) return std::nullopt;
  std::vector<absl::string_view> device_function =
      absl::StrSplit(fields[2], '.');
  PciAddress pci;
  if (device_function.size() != 2 || !ParseHexField(fields[0], &pci.domain) ||
      !ParseHexField(fields[1], &pci.bus) ||
      !ParseHexField(device_function[0], &pci.device) ||
      !ParseHexField(device_function[1], &pci.function)) {
    return std::nullopt;
  }
  return pci;
}

PciAddress PciAddressOf(const rocprofiler_agent_v0_t& agent) {
  return {agent.domain, (agent.location_id >> 8) & 0xff,
          (agent.location_id >> 3) & 0x1f, agent.location_id & 0x7};
}

HipDeviceMap BuildHipDeviceMap(
    absl::Span<const rocprofiler_agent_v0_t> agents,
    absl::Span<const std::optional<PciAddress>> hip) {
  rocprofiler_agent_v0_t placeholder{};
  placeholder.type = ROCPROFILER_AGENT_TYPE_NONE;
  HipDeviceMap map;
  map.gpu_agents.assign(hip.size(), placeholder);
  std::vector<uint32_t> unmatched;
  for (uint32_t k = 0; k < hip.size(); ++k) {
    const rocprofiler_agent_v0_t* agent = FindGpuAgent(agents, hip, hip[k]);
    if (agent == nullptr) {
      unmatched.push_back(k);
      continue;
    }
    map.hip_device_of_agent[agent->id.handle] = k;
    map.gpu_agents[k] = *agent;
  }
  if (!unmatched.empty()) {
    LOG(WARNING) << "HIP device(s) " << absl::StrJoin(unmatched, ", ")
                 << " match no single GPU agent by PCI address.";
  }
  return map;
}

}  // namespace profiler
}  // namespace xla
