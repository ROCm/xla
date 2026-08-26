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

// Grouped AllGather under WarpSpeed.
//
// See README.md in this directory for the mechanism these cases target and for
// how to run them. In short: RCCL assigns each task in a kernel plan its own
// range of channels, while WarpSpeed maps several channels onto one block. A
// plan holding more than one task therefore has blocks whose channel range and
// whose work-descriptor index disagree. One AllGather never exposes it; two in
// the same group do.
//
// XLA reaches that shape by construction. AllGatherThunk submits every buffer
// it owns inside a single GroupExecute, and the collective combiner packs
// buffers into one AllGather up to a byte threshold, so a combined collective
// is a multi-task plan.
//
// Each case declares the shape; the runner supplies the size and states whether
// WarpSpeed is expected to activate. Both are checked, because a correct result
// from a build where the feature never engaged proves nothing.

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/rccl_benchmark/common/case_config.h"
#include "xla/backends/gpu/rccl_benchmark/common/data_pattern.h"
#include "xla/backends/gpu/rccl_benchmark/common/guarded_buffer.h"
#include "xla/backends/gpu/rccl_benchmark/common/path_assert.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::rccl_benchmark {
namespace {

// Fixed at eight because that is the FSDP width the failure was found at, and
// because the number of ranks decides how many channels the plan spans, which
// is the quantity the defect is sensitive to.
constexpr int kNumDevices = 8;
constexpr PrimitiveType kElementType = BF16;
constexpr int64_t kElementBytes = 2;

// Source and destination extents of one logical buffer, each padded with poison
// on both sides.
struct BufferLayout {
  GuardedRegion source;
  GuardedRegion destination;

  int64_t elements_per_rank() const {
    return source.payload_bytes / kElementBytes;
  }
};

std::vector<BufferLayout> MakeLayouts(const CaseConfig& config,
                                      int num_buffers) {
  std::vector<BufferLayout> layouts;
  layouts.reserve(num_buffers);
  for (int i = 0; i < num_buffers; ++i) {
    BufferLayout layout;
    layout.source = GuardedRegion{config.guard_bytes, config.per_rank_bytes};
    layout.destination =
        GuardedRegion{config.guard_bytes, config.per_rank_bytes * kNumDevices};
    layouts.push_back(layout);
  }
  return layouts;
}

// Allocation 2i holds the source of buffer i, allocation 2i+1 its destination.
int SourceAllocationIndex(int buffer_id) { return 2 * buffer_id; }
int DestinationAllocationIndex(int buffer_id) { return 2 * buffer_id + 1; }

std::vector<int64_t> AllocationSizes(absl::Span<const BufferLayout> layouts) {
  std::vector<int64_t> sizes;
  sizes.reserve(2 * layouts.size());
  for (const BufferLayout& layout : layouts) {
    sizes.push_back(layout.source.total_bytes());
    sizes.push_back(layout.destination.total_bytes());
  }
  return sizes;
}

// BufferAllocation::Slice stores a pointer to its allocation, so the
// allocations have to outlive every thunk built from them and must never be
// reallocated. Holding them in one reserved vector makes both guarantees local
// and obvious.
class AllocationTable {
 public:
  explicit AllocationTable(absl::Span<const int64_t> sizes) {
    allocations_.reserve(sizes.size());
    for (int i = 0; i < sizes.size(); ++i) {
      allocations_.emplace_back(/*index=*/i, sizes[i], /*color=*/0);
    }
  }

  const BufferAllocation& at(int index) const { return allocations_[index]; }

 private:
  std::vector<BufferAllocation> allocations_;
};

CollectiveConfig MakeAllGatherConfig(int num_buffers) {
  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }

  CollectiveConfig config;
  // One element type per buffer. Uniform here; the same field is where a
  // combined collective with mixed types would be expressed.
  config.operand_element_type.assign(num_buffers, kElementType);
  config.replica_groups = {replica_group};
  config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  return config;
}

// Builds one thunk covering `buffer_ids`. Every buffer handed to a single thunk
// is submitted inside one RCCL group, so the length of `buffer_ids` is exactly
// the number of tasks the resulting kernel plan will carry.
AllGatherThunk MakeThunk(const AllocationTable& table,
                         absl::Span<const BufferLayout> layouts,
                         absl::Span<const int> buffer_ids) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(buffer_ids.size());
  for (int buffer_id : buffer_ids) {
    const BufferLayout& layout = layouts[buffer_id];
    const int64_t elements_per_rank = layout.elements_per_rank();

    ShapedSlice source{
        BufferAllocation::Slice(&table.at(SourceAllocationIndex(buffer_id)),
                                layout.source.payload_offset(),
                                layout.source.payload_bytes),
        ShapeUtil::MakeShape(kElementType, {elements_per_rank})};
    ShapedSlice destination{
        BufferAllocation::Slice(&table.at(DestinationAllocationIndex(buffer_id)),
                                layout.destination.payload_offset(),
                                layout.destination.payload_bytes),
        ShapeUtil::MakeShape(kElementType,
                             {elements_per_rank * kNumDevices})};

    buffers.push_back(CollectiveThunk::Buffer{
        .element_count = elements_per_rank,
        .source_buffer = source,
        .destination_buffer = destination,
        .source_memory_space = 0,
        .destination_memory_space = 0});
  }

  const int num_buffers = static_cast<int>(buffers.size());
  return AllGatherThunk(Thunk::ThunkInfo(), MakeAllGatherConfig(num_buffers),
                        std::move(buffers));
}

absl::Status PrepareDeviceBuffers(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> allocations,
    absl::Span<const BufferLayout> layouts, int device_ordinal) {
  for (int buffer_id = 0; buffer_id < layouts.size(); ++buffer_id) {
    const BufferLayout& layout = layouts[buffer_id];
    const std::vector<uint16_t> source = MakeSourcePattern(
        buffer_id, device_ordinal, layout.elements_per_rank());
    RETURN_IF_ERROR(WriteGuardedBuffer(
        stream, allocations[SourceAllocationIndex(buffer_id)], layout.source,
        source));
    // Poison the destination before every execution so that "the collective
    // did not run" is distinguishable from "the collective wrote the right
    // answer", including on repeat iterations.
    RETURN_IF_ERROR(WriteGuardedBufferFilled(
        stream, allocations[DestinationAllocationIndex(buffer_id)],
        layout.destination, UnwrittenPayloadWord()));
  }
  return absl::OkStatus();
}

absl::Status VerifyDeviceBuffers(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> allocations,
    absl::Span<const BufferLayout> layouts, int device_ordinal,
    int iteration) {
  for (int buffer_id = 0; buffer_id < layouts.size(); ++buffer_id) {
    const BufferLayout& layout = layouts[buffer_id];
    const int64_t elements_per_rank = layout.elements_per_rank();

    const std::string destination_label =
        absl::StrFormat("device %d, buffer %d destination, iteration %d",
                        device_ordinal, buffer_id, iteration);
    ASSIGN_OR_RETURN(
        const std::vector<uint8_t> destination_image,
        ReadGuardedBuffer(stream,
                          allocations[DestinationAllocationIndex(buffer_id)],
                          layout.destination));
    RETURN_IF_ERROR(CheckGuards(destination_image, layout.destination,
                                destination_label));
    // An AllGather is pure data movement, so the answer is exact: rank r's
    // contribution occupies slot r of the output, bit for bit.
    RETURN_IF_ERROR(CheckPayloadGenerated(
        destination_image, layout.destination,
        [&](int64_t index) {
          const int rank = static_cast<int>(index / elements_per_rank);
          return PatternWord(buffer_id, rank, index % elements_per_rank);
        },
        destination_label));

    const std::string source_label =
        absl::StrFormat("device %d, buffer %d source, iteration %d",
                        device_ordinal, buffer_id, iteration);
    ASSIGN_OR_RETURN(
        const std::vector<uint8_t> source_image,
        ReadGuardedBuffer(stream, allocations[SourceAllocationIndex(buffer_id)],
                          layout.source));
    RETURN_IF_ERROR(CheckGuards(source_image, layout.source, source_label));
    // The input must come back unchanged. A task that reads or writes through
    // another task's descriptor can corrupt an input as easily as an output.
    RETURN_IF_ERROR(CheckPayloadGenerated(
        source_image, layout.source,
        [&](int64_t index) {
          return PatternWord(buffer_id, device_ordinal, index);
        },
        source_label));
  }
  return absl::OkStatus();
}

// Runs `num_buffers` AllGathers submitted as one group: a single thunk owning
// every buffer, which becomes a single GroupExecute and a single multi-task
// kernel plan.
absl::Status RunOneGroup(const CaseConfig& config, int num_buffers) {
  const std::vector<BufferLayout> layouts = MakeLayouts(config, num_buffers);
  const std::vector<int64_t> sizes = AllocationSizes(layouts);
  const AllocationTable table(sizes);

  std::vector<int> buffer_ids(num_buffers);
  std::iota(buffer_ids.begin(), buffer_ids.end(), 0);
  AllGatherThunk thunk = MakeThunk(table, layouts, buffer_ids);

  const DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<CollectiveThunkMultiGpuTestState> states(kNumDevices);

  return RunOnDevices(
      kNumDevices, "rccl_bench_one_group", [&](int device) -> absl::Status {
        RETURN_IF_ERROR(SetupCollectiveThunkDevice(device, kNumDevices, sizes,
                                                   thunk, device_assignment,
                                                   states[device]));
        for (int iteration = 0; iteration < config.repeats; ++iteration) {
          RETURN_IF_ERROR(PrepareDeviceBuffers(*states[device].stream,
                                               states[device].create_buffers,
                                               layouts, device));
          const BufferAllocations allocations =
              MakeBufferAllocations(states[device], states[device].create_buffers);
          const Thunk::ExecuteParams params =
              MakeExecuteParams(states[device], allocations);
          RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, params));
          RETURN_IF_ERROR(VerifyDeviceBuffers(*states[device].stream,
                                              states[device].create_buffers,
                                              layouts, device, iteration));
        }
        return absl::OkStatus();
      });
}

// Runs two AllGathers over the same buffers as two independent groups. This is
// the control that separates "two collectives" from "two collectives in one
// plan": the data, the sizes and the feature state are identical, only the
// submission differs.
absl::Status RunTwoSeparateGroups(const CaseConfig& config) {
  constexpr int kNumBuffers = 2;
  const std::vector<BufferLayout> layouts = MakeLayouts(config, kNumBuffers);
  const std::vector<int64_t> sizes = AllocationSizes(layouts);
  const AllocationTable table(sizes);

  const int first_id[] = {0};
  const int second_id[] = {1};
  AllGatherThunk first = MakeThunk(table, layouts, first_id);
  AllGatherThunk second = MakeThunk(table, layouts, second_id);

  const DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<CollectiveThunkMultiGpuTestState> states(kNumDevices);

  return RunOnDevices(
      kNumDevices, "rccl_bench_separate", [&](int device) -> absl::Status {
        RETURN_IF_ERROR(SetupCollectiveThunksDevice(
            device, kNumDevices, sizes, {&first, &second}, device_assignment,
            states[device]));
        for (int iteration = 0; iteration < config.repeats; ++iteration) {
          RETURN_IF_ERROR(PrepareDeviceBuffers(*states[device].stream,
                                               states[device].create_buffers,
                                               layouts, device));
          const BufferAllocations allocations =
              MakeBufferAllocations(states[device], states[device].create_buffers);
          const Thunk::ExecuteParams params =
              MakeExecuteParams(states[device], allocations);
          RETURN_IF_ERROR(ExecuteOnStreamAndBlock(first, params));
          RETURN_IF_ERROR(ExecuteOnStreamAndBlock(second, params));
          RETURN_IF_ERROR(VerifyDeviceBuffers(*states[device].stream,
                                              states[device].create_buffers,
                                              layouts, device, iteration));
        }
        return absl::OkStatus();
      });
}

// Reports the path assertion before the correctness result.
//
// The order is deliberate. If the intended path did not run, the correctness
// outcome carries no information either way, and saying so first keeps a
// vacuous green from being read as evidence.
void ReportCase(const CaseConfig& config, const absl::Status& run_status) {
  const absl::StatusOr<RcclPathObservation> observation = ObserveRcclPath();
  ASSERT_OK(observation.status())
      << "could not read the RCCL debug log; the case is inconclusive";

  EXPECT_OK(ExpectWarpSpeed(*observation, config.expect_warp_speed))
      << "case configuration: " << config.Describe();

  EXPECT_OK(run_status) << "case configuration: " << config.Describe() << "\n"
                        << observation->DebugString();
}

class WarpSpeedGroupedAllGatherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!HasEnoughGpus(kNumDevices)) {
      // Not a skip. A lane that silently passes when the hardware cannot host
      // the case reports coverage it does not have.
      FAIL() << "This case requires " << kNumDevices
             << " GPUs. Running it on fewer would exercise a different channel "
                "layout, so it fails rather than reporting a pass.";
    }

    // Create every executor from one thread before the device threads start.
    // Eight threads racing to populate the executor cache is not what these
    // cases are trying to test, and it crashes before any collective runs.
    for (int device = 0; device < kNumDevices; ++device) {
      ASSERT_NE(GetGpuExecutor(device), nullptr)
          << "no executor for device " << device;
    }

    config_ = CaseConfigFromEnv();
  }

  CaseConfig config_;
};

// The case under investigation: two AllGathers, one group, one plan.
TEST_F(WarpSpeedGroupedAllGatherTest, TwoBuffersInOneGroup) {
  ReportCase(config_, RunOneGroup(config_, /*num_buffers=*/2));
}

// Control: same size, same feature state, one task in the plan.
TEST_F(WarpSpeedGroupedAllGatherTest, SingleBuffer) {
  ReportCase(config_, RunOneGroup(config_, /*num_buffers=*/1));
}

// Control: same two buffers, submitted as two plans instead of one.
TEST_F(WarpSpeedGroupedAllGatherTest, TwoBuffersInSeparateGroups) {
  ReportCase(config_, RunTwoSeparateGroups(config_));
}

// Severity probe: more tasks means more channel ranges, and the combiner
// routinely produces far more than two buffers per collective.
TEST_F(WarpSpeedGroupedAllGatherTest, FourBuffersInOneGroup) {
  ReportCase(config_, RunOneGroup(config_, /*num_buffers=*/4));
}

}  // namespace
}  // namespace xla::gpu::rccl_benchmark
