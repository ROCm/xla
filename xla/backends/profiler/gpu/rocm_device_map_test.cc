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
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "rocm/include/rocprofiler-sdk/agent.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"

namespace xla {
namespace profiler {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

// SDK agent handles are opaque and start at an arbitrary per-process offset.
constexpr uint64_t kHandle0 = 41237;
// PCI buses of the 8 GPUs of an MI355X machine, in SDK order.
constexpr uint32_t kGpuBus[] = {0x05, 0x15, 0x65, 0x75, 0x85, 0x95, 0xe5, 0xf5};
// Stands for a placeholder in SlotHandles.
constexpr uint64_t kPlaceholder = ~uint64_t{0};

rocprofiler_agent_v0_t CpuAgent(uint64_t handle) {
  rocprofiler_agent_v0_t agent{};
  agent.id.handle = handle;
  agent.type = ROCPROFILER_AGENT_TYPE_CPU;
  return agent;
}

rocprofiler_agent_v0_t GpuAgent(uint64_t handle, uint32_t bus) {
  rocprofiler_agent_v0_t agent{};
  agent.id.handle = handle;
  agent.type = ROCPROFILER_AGENT_TYPE_GPU;
  agent.location_id = bus << 8;
  return agent;
}

uint64_t GpuHandle(uint32_t k) { return kHandle0 + 2 + k; }

// As the SDK lists them: 2 CPU agents, then GPU k with handle GpuHandle(k).
std::vector<rocprofiler_agent_v0_t> EightGpuAgents() {
  std::vector<rocprofiler_agent_v0_t> agents = {CpuAgent(kHandle0),
                                                CpuAgent(kHandle0 + 1)};
  for (uint32_t k = 0; k < 8; ++k) {
    agents.push_back(GpuAgent(GpuHandle(k), kGpuBus[k]));
  }
  return agents;
}

std::optional<PciAddress> HipDevice(uint32_t bus, uint32_t domain = 0) {
  return PciAddress{domain, bus, 0, 0};
}

// The agent handle in each HIP device slot.
std::vector<uint64_t> SlotHandles(const HipDeviceMap& map) {
  std::vector<uint64_t> handles;
  for (const rocprofiler_agent_v0_t& agent : map.gpu_agents) {
    handles.push_back(agent.type == ROCPROFILER_AGENT_TYPE_NONE
                          ? kPlaceholder
                          : agent.id.handle);
  }
  return handles;
}

TEST(ParsePciBusIdTest, ParsesHipFormat) {
  EXPECT_THAT(ParsePciBusId("0000:05:00.0"),
              Optional(PciAddress{0, 0x05, 0, 0}));
  EXPECT_THAT(ParsePciBusId("0000:E5:1F.7"),
              Optional(PciAddress{0, 0xe5, 0x1f, 7}));
  EXPECT_THAT(ParsePciBusId("10000:01:00.0"),
              Optional(PciAddress{0x10000, 0x01, 0, 0}));
}

TEST(ParsePciBusIdTest, RejectsMalformedStrings) {
  for (absl::string_view bus_id :
       {"", "0000:05:00", "0000:05.00.0", "0000:05:00.0.0", "0000:05:00:0",
        "0000:05:zz.0", "0000::00.0", "0x0:05:00.0", " 0000:05:00.0",
        "+000:05:00.0"}) {
    EXPECT_EQ(ParsePciBusId(bus_id), std::nullopt) << bus_id;
  }
}

TEST(PciAddressOfTest, DecodesKfdLocationId) {
  rocprofiler_agent_v0_t agent{};
  agent.domain = 1;
  agent.location_id = 0x65 << 8 | 0x02 << 3 | 0x1;

  EXPECT_EQ(PciAddressOf(agent), (PciAddress{1, 0x65, 0x02, 0x1}));
}

TEST(BuildHipDeviceMapTest, NumbersGpusInSdkOrderWithoutMasking) {
  std::vector<std::optional<PciAddress>> hip;
  absl::flat_hash_map<uint64_t, uint32_t> expected_map;
  std::vector<uint64_t> expected_slots;
  for (uint32_t k = 0; k < 8; ++k) {
    hip.push_back(HipDevice(kGpuBus[k]));
    expected_map[GpuHandle(k)] = k;
    expected_slots.push_back(GpuHandle(k));
  }

  HipDeviceMap map = BuildHipDeviceMap(EightGpuAgents(), hip);

  EXPECT_EQ(map.hip_device_of_agent, expected_map);
  EXPECT_EQ(SlotHandles(map), expected_slots);
}

TEST(BuildHipDeviceMapTest, FollowsHipOrderUnderVisibilityMasks) {
  // HIP_VISIBLE_DEVICES=2,0 or ROCR_VISIBLE_DEVICES=2,0: the SDK still lists
  // all 8 GPUs, and HIP numbers the visible ones in mask order.
  HipDeviceMap map = BuildHipDeviceMap(
      EightGpuAgents(), {HipDevice(kGpuBus[2]), HipDevice(kGpuBus[0])});

  EXPECT_THAT(
      map.hip_device_of_agent,
      UnorderedElementsAre(Pair(GpuHandle(2), 0), Pair(GpuHandle(0), 1)));
  EXPECT_THAT(SlotHandles(map), ElementsAre(GpuHandle(2), GpuHandle(0)));
}

TEST(BuildHipDeviceMapTest, NoHipDevicesGiveAnEmptyMap) {
  HipDeviceMap map = BuildHipDeviceMap(EightGpuAgents(), {});

  EXPECT_THAT(map.hip_device_of_agent, IsEmpty());
  EXPECT_THAT(map.gpu_agents, IsEmpty());
}

TEST(BuildHipDeviceMapTest, GivesHipDeviceWithoutAgentAPlaceholder) {
  HipDeviceMap map = BuildHipDeviceMap(
      EightGpuAgents(),
      {HipDevice(kGpuBus[0]), HipDevice(0x42), HipDevice(kGpuBus[1])});

  EXPECT_THAT(
      map.hip_device_of_agent,
      UnorderedElementsAre(Pair(GpuHandle(0), 0), Pair(GpuHandle(1), 2)));
  EXPECT_THAT(SlotHandles(map),
              ElementsAre(GpuHandle(0), kPlaceholder, GpuHandle(1)));
}

TEST(BuildHipDeviceMapTest, GivesHipDeviceWithoutAddressAPlaceholder) {
  // hipDeviceGetPCIBusId failed for device 0.
  HipDeviceMap map = BuildHipDeviceMap(EightGpuAgents(),
                                       {std::nullopt, HipDevice(kGpuBus[1])});

  EXPECT_THAT(map.hip_device_of_agent,
              UnorderedElementsAre(Pair(GpuHandle(1), 1)));
  EXPECT_THAT(SlotHandles(map), ElementsAre(kPlaceholder, GpuHandle(1)));
}

TEST(BuildHipDeviceMapTest, NeverMapsCpuAgents) {
  // A CPU agent's domain and location_id are 0, as in address 0000:00:00.0.
  HipDeviceMap map = BuildHipDeviceMap(
      {CpuAgent(kHandle0), GpuAgent(kHandle0 + 1, 0x05)}, {HipDevice(0x00)});

  EXPECT_THAT(map.hip_device_of_agent, IsEmpty());
  EXPECT_THAT(SlotHandles(map), ElementsAre(kPlaceholder));
}

TEST(BuildHipDeviceMapTest, NeverMatchesAcrossPciDomains) {
  HipDeviceMap map = BuildHipDeviceMap({GpuAgent(kHandle0, 0x05)},
                                       {HipDevice(0x05, /*domain=*/1)});

  EXPECT_THAT(map.hip_device_of_agent, IsEmpty());
  EXPECT_THAT(SlotHandles(map), ElementsAre(kPlaceholder));
}

TEST(BuildHipDeviceMapTest, LeavesTwoAgentsAtOneAddressUnmapped) {
  // Nothing tells the two agents at bus 0x05 apart.
  HipDeviceMap map =
      BuildHipDeviceMap({GpuAgent(kHandle0, 0x05), GpuAgent(kHandle0 + 1, 0x05),
                         GpuAgent(kHandle0 + 2, 0x15)},
                        {HipDevice(0x05), HipDevice(0x15)});

  EXPECT_THAT(map.hip_device_of_agent,
              UnorderedElementsAre(Pair(kHandle0 + 2, 1)));
  EXPECT_THAT(SlotHandles(map), ElementsAre(kPlaceholder, kHandle0 + 2));
}

TEST(BuildHipDeviceMapTest, LeavesTwoHipDevicesAtOneAddressUnmapped) {
  // Nothing tells the two HIP devices apart.
  HipDeviceMap map = BuildHipDeviceMap({GpuAgent(kHandle0, 0x05)},
                                       {HipDevice(0x05), HipDevice(0x05)});

  EXPECT_THAT(map.hip_device_of_agent, IsEmpty());
  EXPECT_THAT(SlotHandles(map), ElementsAre(kPlaceholder, kPlaceholder));
}

TEST(PciAddressFromLocationTest, DecodesKfdLocationId) {
  EXPECT_EQ(PciAddressFromLocation(1, 0x65 << 8 | 0x02 << 3 | 0x1),
            (PciAddress{1, 0x65, 0x02, 0x1}));
}

TEST(HasGpuPartitionsTest, GpusOnDistinctBusesAreNotPartitions) {
  EXPECT_FALSE(HasGpuPartitions(EightGpuAgents()));
}

TEST(HasGpuPartitionsTest, GpusThatDifferOnlyInFunctionArePartitions) {
  rocprofiler_agent_v0_t partition1 = GpuAgent(kHandle0 + 1, 0x05);
  partition1.location_id |= 1;

  EXPECT_TRUE(HasGpuPartitions({GpuAgent(kHandle0, 0x05), partition1}));
}

TEST(HasGpuPartitionsTest, GpusAtOneAddressArePartitions) {
  // Partitions on kernels before Linux 6.12 share the whole address.
  EXPECT_TRUE(HasGpuPartitions(
      {GpuAgent(kHandle0, 0x05), GpuAgent(kHandle0 + 1, 0x05)}));
}

TEST(HasGpuPartitionsTest, GpusInDifferentDomainsAreNotPartitions) {
  rocprofiler_agent_v0_t other_domain = GpuAgent(kHandle0 + 1, 0x05);
  other_domain.domain = 1;

  EXPECT_FALSE(HasGpuPartitions({GpuAgent(kHandle0, 0x05), other_domain}));
}

TEST(HasGpuPartitionsTest, IgnoresCpuAgents) {
  // Both CPU agents report 0000:00:00.0.
  EXPECT_FALSE(HasGpuPartitions({CpuAgent(kHandle0), CpuAgent(kHandle0 + 1),
                                 GpuAgent(kHandle0 + 2, 0x05)}));
}

// A peer-copy record is (issuing GPU, other GPU); see ResolvePeerCopyGpus.
constexpr uint64_t kG0 = kHandle0 + 2;
constexpr uint64_t kG1 = kHandle0 + 3;
constexpr uint64_t kG2 = kHandle0 + 4;

// (source_agent, destination_agent, from_pointer_owners)
using Gpus = std::tuple<uint64_t, std::optional<uint64_t>, bool>;

Gpus Fields(const PeerCopyGpus& gpus) {
  return {gpus.source_agent, gpus.destination_agent, gpus.from_pointer_owners};
}

TEST(ResolvePeerCopyGpusTest, TrustsOwnersWhenTheSourceIssued) {
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG0, kG1, kG0, kG1)),
            Gpus(kG0, kG1, true));
}

TEST(ResolvePeerCopyGpusTest, TrustsOwnersWhenTheDestinationIssued) {
  // Data G0 -> G1, issued on G1.
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG1, kG0, kG0, kG1)),
            Gpus(kG0, kG1, true));
}

TEST(ResolvePeerCopyGpusTest, TrustsOwnersWhenAThirdGpuIssued) {
  // Data G1 -> G2, issued on G0. G2 is not in the record.
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG0, kG1, kG1, kG2)),
            Gpus(kG1, kG2, true));
}

TEST(ResolvePeerCopyGpusTest, DistrustsADestinationOwnerThatDisagrees) {
  // The source issued, so the record's other GPU must be the destination.
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG0, kG1, kG0, kG2)),
            Gpus(kG1, std::nullopt, false));
}

TEST(ResolvePeerCopyGpusTest, DistrustsAnUnknownDestinationOfTheIssuer) {
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG0, kG1, kG0, std::nullopt)),
            Gpus(kG1, std::nullopt, false));
}

TEST(ResolvePeerCopyGpusTest, KeepsAKnownSourceWhenTheDestinationIsUnknown) {
  // Data G0 -> ?, issued on G1.
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG1, kG0, kG0, std::nullopt)),
            Gpus(kG0, std::nullopt, false));
}

TEST(ResolvePeerCopyGpusTest, DistrustsASourceOwnerOutsideTheRecord) {
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG0, kG1, kG2, kG1)),
            Gpus(kG1, std::nullopt, false));
}

TEST(ResolvePeerCopyGpusTest, DistrustsAnUnknownSource) {
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG0, kG1, std::nullopt, kG1)),
            Gpus(kG1, std::nullopt, false));
}

TEST(ResolvePeerCopyGpusTest, PutsACopyWithNoOwnersOnTheRecordsOtherGpu) {
  // The other GPU is always one of the copy's two, even when a third issued.
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG0, kG1, std::nullopt, std::nullopt)),
            Gpus(kG1, std::nullopt, false));
}

TEST(ResolvePeerCopyGpusTest, DistrustsEqualOwners) {
  EXPECT_EQ(Fields(ResolvePeerCopyGpus(kG0, kG1, kG1, kG1)),
            Gpus(kG1, std::nullopt, false));
}

}  // namespace
}  // namespace profiler
}  // namespace xla
