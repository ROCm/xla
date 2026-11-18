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

// Routing of TopK to the Triton TopK kernels on ROCm. Runs on the CPU with the
// device descriptions of TestGpuDeviceInfo.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/codegen/triton/topk_config.h"
#include "xla/backends/gpu/transforms/topk_specializer.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/topk_rewriter.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

enum class TritonTopKFlags { kOff, kOn, kForced };

class TopkSpecializerTritonTest : public HloHardwareIndependentTestBase {
 protected:
  // Parses a module whose root is `ROOT t = (out, idx) custom-call(p)` with
  // target "TopK" and the HLO attributes `extra_attrs`, if any.
  std::unique_ptr<HloModule> ParseTopK(absl::string_view in,
                                       absl::string_view out,
                                       absl::string_view idx,
                                       absl::string_view backend_config,
                                       TritonTopKFlags flags,
                                       absl::string_view extra_attrs = "") {
    std::string config_attr =
        backend_config.empty()
            ? ""
            : absl::Substitute(", backend_config=$0", backend_config);
    if (!extra_attrs.empty()) {
      absl::StrAppend(&config_attr, ", ", extra_attrs);
    }
    auto module = ParseAndReturnVerifiedModule(absl::Substitute(
        R"(
      HloModule m
      ENTRY e {
        p = $0 parameter(0)
        ROOT t = ($1, $2) custom-call(p), custom_call_target="TopK"$3
      })",
        in, out, idx, config_attr));
    CHECK_OK(module.status());
    auto& debug_options = (*module)->mutable_config().mutable_debug_options();
    debug_options.set_xla_gpu_experimental_enable_triton_topk(
        flags != TritonTopKFlags::kOff);
    debug_options.set_xla_gpu_experimental_force_triton_topk(
        flags == TritonTopKFlags::kForced);
    return *std::move(module);
  }

  static bool HasTritonTopK(const HloModule& module) {
    for (const HloComputation* comp : module.computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        auto* cc = DynCast<HloCustomCallInstruction>(instr);
        if (cc != nullptr && IsTritonTopK(*cc)) {
          return true;
        }
      }
    }
    return false;
  }

  // Expects the root to be the original TopK rewritten to a marked __gpu$TopK
  // on the same operand, with the same shape and backend config fields.
  static void ExpectMarkedTopK(const HloModule& module, bool stable,
                               ComparisonOrder order) {
    const HloInstruction* root = module.entry_computation()->root_instruction();
    auto* topk = DynCast<HloCustomCallInstruction>(root);
    ASSERT_NE(topk, nullptr) << root->ToString();
    EXPECT_EQ(topk->custom_call_target(), "__gpu$TopK");
    EXPECT_TRUE(IsTritonTopK(*topk));
    EXPECT_EQ(topk->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(topk->shape().tuple_shapes().size(), 2);
    EXPECT_EQ(hlo_instruction_utils::IsTopKStable(topk), stable);
    EXPECT_EQ(hlo_instruction_utils::GetTopKOrder(topk), order);
  }
};

TEST_F(TopkSpecializerTritonTest, FlagOffKeepsTopK) {
  auto module = ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                          TritonTopKFlags::kOff);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(TopkSpecializerTritonTest, FlagOnKeepsShapesWhereTodaysPathIsFaster) {
  auto module = ParseTopK("f32[8,16384]", "f32[8,128]", "s32[8,128]", "",
                          TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(TopkSpecializerTritonTest, FlagOnRoutesShapesWhereTritonIsFaster) {
  auto module = ParseTopK("bf16[24576,64]", "bf16[24576,6]", "s32[24576,6]", "",
                          TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  ExpectMarkedTopK(*module, /*stable=*/true, ComparisonOrder::kTotal);
}

TEST_F(TopkSpecializerTritonTest, FlagOnDoesNotRouteOnUnmeasuredGpus) {
  auto module = ParseTopK("bf16[24576,64]", "bf16[24576,6]", "s32[24576,6]", "",
                          TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI300DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(HasTritonTopK(*module));
}

struct RoutedCase {
  std::string name;
  std::string in, out, idx, backend_config;
  bool stable;
  ComparisonOrder order;
};

class TopkSpecializerTritonRoutedTest
    : public TopkSpecializerTritonTest,
      public ::testing::WithParamInterface<RoutedCase> {};

TEST_P(TopkSpecializerTritonRoutedTest, ForcedTopKIsMarked) {
  const RoutedCase& c = GetParam();
  auto module =
      ParseTopK(c.in, c.out, c.idx, c.backend_config, TritonTopKFlags::kForced);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  ExpectMarkedTopK(*module, c.stable, c.order);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, TopkSpecializerTritonRoutedTest,
    ::testing::Values(
        RoutedCase{"StableTotalF32", "f32[8,1000]", "f32[8,32]", "s32[8,32]",
                   "", true, ComparisonOrder::kTotal},
        RoutedCase{"UnstablePartialBf16", "bf16[24576,64]", "bf16[24576,6]",
                   "s32[24576,6]", R"({is_stable = false, order = "PARTIAL"})",
                   false, ComparisonOrder::kPartial},
        RoutedCase{"StablePartialF32", "f32[16,4096]", "f32[16,8]", "s32[16,8]",
                   R"({order = "PARTIAL"})", true, ComparisonOrder::kPartial},
        RoutedCase{"UnstableTotalBf16", "bf16[32,2048]", "bf16[32,64]",
                   "s32[32,64]", R"({is_stable = false})", false,
                   ComparisonOrder::kTotal},
        RoutedCase{"RankOne", "f32[4096]", "f32[16]", "s32[16]", "", true,
                   ComparisonOrder::kTotal},
        RoutedCase{"ShortRow", "f32[3,16]", "f32[3,2]", "s32[3,2]", "", true,
                   ComparisonOrder::kTotal},
        RoutedCase{"MaxK", "f32[7,1000]", "f32[7,128]", "s32[7,128]", "", true,
                   ComparisonOrder::kTotal}),
    [](const ::testing::TestParamInfo<RoutedCase>& info) {
      return info.param.name;
    });

struct RejectedCase {
  std::string name;
  std::string in, out, idx;
};

class TopkSpecializerTritonRejectedTest
    : public TopkSpecializerTritonTest,
      public ::testing::WithParamInterface<RejectedCase> {};

TEST_P(TopkSpecializerTritonRejectedTest, ForcedTopKIsNotMarked) {
  const RejectedCase& c = GetParam();
  auto module = ParseTopK(c.in, c.out, c.idx, "", TritonTopKFlags::kForced);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(HasTritonTopK(*module));
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, TopkSpecializerTritonRejectedTest,
    ::testing::Values(
        RejectedCase{"F16", "f16[8,1000]", "f16[8,32]", "s32[8,32]"},
        RejectedCase{"KAboveLimit", "f32[8,1000]", "f32[8,129]", "s32[8,129]"},
        RejectedCase{"KCloseToN", "f32[8,16]", "f32[8,14]", "s32[8,14]"},
        // 1024 chunks of 16384 leave 131072 candidates to merge.
        RejectedCase{"TooManyCandidates", "f32[1,16777216]", "f32[1,128]",
                     "s32[1,128]"}),
    [](const ::testing::TestParamInfo<RejectedCase>& info) {
      return info.param.name;
    });

TEST_F(TopkSpecializerTritonTest, ForcedLongRowGetsAScratchOutput) {
  // 16 chunks of 4096 per row, 8 candidates each.
  auto module = ParseTopK("f32[8,65536]", "f32[8,8]", "s32[8,8]",
                          R"({order = "PARTIAL"})", TritonTopKFlags::kForced);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  ASSERT_EQ(root->operand_count(), 2);
  auto* topk = DynCast<HloCustomCallInstruction>(root->operand(0)->operand(0));
  ASSERT_NE(topk, nullptr) << root->ToString();
  EXPECT_EQ(topk->custom_call_target(), "__gpu$TopK");
  EXPECT_TRUE(IsTritonTopK(*topk));
  EXPECT_EQ(topk->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(topk->shape().ToString(), "(f32[8,8], s32[8,8], u64[1024])");
  EXPECT_EQ(hlo_instruction_utils::GetTopKOrder(topk),
            ComparisonOrder::kPartial);
  EXPECT_EQ(root->operand(1)->operand(0), topk);
}

TEST_F(TopkSpecializerTritonTest, FlagOnRoutesLongRows) {
  auto module = ParseTopK("f32[8,65536]", "f32[8,8]", "s32[8,8]", "",
                          TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kTuple);
}

TEST_F(TopkSpecializerTritonTest, OneLongRowWithLargeKKeepsTodaysPath) {
  // Merging the candidates of the chunks costs more than today's sort.
  auto module = ParseTopK("f32[1,1048576]", "f32[1,128]", "s32[1,128]", "",
                          TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(HasTritonTopK(*module));
  EXPECT_TRUE(
      module->entry_computation()->root_instruction()->IsCustomCall("TopK"));
}

constexpr absl::string_view kDeferredPackedBf16 =
    R"(frontend_attributes={_xla_topk_deferred_packed_bf16="true"})";

TEST_F(TopkSpecializerTritonTest, DeferredPackedBf16KeepsTheTopKWhenNotRouted) {
  // Not routed on a GPU without measurements. Without the mark, it takes the
  // prebuilt kernel.
  auto unmarked = ParseTopK("bf16[8,1024]", "bf16[8,16]", "s32[8,16]", "",
                            TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI300DeviceInfo()),
                 unmarked.get()));
  EXPECT_TRUE(changed);

  // With the mark, TopkDecomposer later gives it the packed BF16 sort.
  auto marked = ParseTopK("bf16[8,1024]", "bf16[8,16]", "s32[8,16]", "",
                          TritonTopKFlags::kOn, kDeferredPackedBf16);
  ASSERT_OK_AND_ASSIGN(
      changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI300DeviceInfo()),
                 marked.get()));
  EXPECT_FALSE(changed);
  const HloInstruction* root = marked->entry_computation()->root_instruction();
  EXPECT_TRUE(root->IsCustomCall("TopK"));
  EXPECT_TRUE(IsTopKWithDeferredPackedBf16(root));
}

TEST_F(TopkSpecializerTritonTest, DeferredPackedBf16IsRoutedWhenForced) {
  auto module = ParseTopK("bf16[24576,64]", "bf16[24576,6]", "s32[24576,6]",
                          R"({is_stable = true, order = "TOTAL"})",
                          TritonTopKFlags::kForced, kDeferredPackedBf16);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  ExpectMarkedTopK(*module, /*stable=*/true, ComparisonOrder::kTotal);
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    EXPECT_FALSE(IsTopKWithDeferredPackedBf16(inst)) << inst->ToString();
  }
}

TEST_F(TopkSpecializerTritonTest, CudaIsNotRouted) {
  auto module = ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                          TritonTopKFlags::kForced);
  ASSERT_THAT(
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::H100SXMDeviceInfo()),
                 module.get()),
      absl_testing::IsOk());
  EXPECT_FALSE(HasTritonTopK(*module));
}

TEST_F(TopkSpecializerTritonTest, OtherRocmArchitecturesAreRoutedWhenForced) {
  for (const se::DeviceDescription& device :
       {TestGpuDeviceInfo::AMDMI300DeviceInfo(),
        TestGpuDeviceInfo::AMDRX7900DeviceInfo()}) {
    SCOPED_TRACE(device.name());
    auto module = ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                            TritonTopKFlags::kForced);
    ASSERT_OK_AND_ASSIGN(bool changed,
                         RunHloPass(TopkSpecializer(device), module.get()));
    EXPECT_TRUE(changed);
    ExpectMarkedTopK(*module, /*stable=*/true, ComparisonOrder::kTotal);
  }
}

TEST_F(TopkSpecializerTritonTest, RowLimitFollowsTheWaveSize) {
  // 16384 keys fit a wave-64 program of 8 warps; a wave-32 device splits them
  // into chunks.
  for (const auto& [device, chunked] :
       {std::pair{TestGpuDeviceInfo::AMDMI350DeviceInfo(), false},
        std::pair{TestGpuDeviceInfo::AMDRX7900DeviceInfo(), true}}) {
    SCOPED_TRACE(device.name());
    auto module = ParseTopK("f32[8,16384]", "f32[8,32]", "s32[8,32]", "",
                            TritonTopKFlags::kForced);
    ASSERT_OK_AND_ASSIGN(bool changed,
                         RunHloPass(TopkSpecializer(device), module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(HasTritonTopK(*module));
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    EXPECT_EQ(root->opcode(),
              chunked ? HloOpcode::kTuple : HloOpcode::kCustomCall);
  }
}

// The comparator SPMD partitioning gives the merge sort after a TopK: values in
// descending total order, ties by ascending index.
constexpr absl::string_view kSpmdComparator = R"(
        eq = pred[] compare(v0, v1), direction=EQ, type=TOTALORDER
        lt = pred[] compare(i0, i1), direction=LT
        gt = pred[] compare(v0, v1), direction=GT, type=TOTALORDER
        ROOT r = pred[] select(eq, lt, gt))";

class TopkSpecializerTritonSortTest : public TopkSpecializerTritonTest {
 protected:
  // Parses a TopK of `p` = `type[in]` into `type[out]` followed by a sort of
  // its values `v` and of `indices` along `dimension`, compared by
  // `comparator`. `indices` names one of the TopK indices `i`, `i` plus SPMD's
  // offset, the number `o` of the partition in its group times the row length,
  // on either side (`offset`, `offset_lhs`), `i` plus the partition id times
  // the row length (`partition_offset`), offsets that can reach 2^31: a
  // parameter times the row length (`unbounded_offset`), 2^25 times the row
  // length (`far_offset`), 2^31 - 63 (`near_offset`), or `i + i` (`doubled`).
  std::unique_ptr<HloModule> ParseTopKThenSort(
      absl::string_view type, absl::string_view in, absl::string_view out,
      absl::string_view backend_config, TritonTopKFlags flags,
      absl::string_view comparator = kSpmdComparator,
      absl::string_view indices = "offset", int64_t dimension = 1) {
    std::string config_attr =
        backend_config.empty()
            ? ""
            : absl::Substitute(", backend_config=$0", backend_config);
    absl::string_view row_length = in.substr(in.rfind(',') + 1);
    auto module = ParseAndReturnVerifiedModule(absl::Substitute(
        R"(
      HloModule m
      cmp {
        v0 = $0[] parameter(0)
        v1 = $0[] parameter(1)
        i0 = s32[] parameter(2)
        i1 = s32[] parameter(3)$4
      }
      ENTRY e {
        p = $0[$1] parameter(0)
        x = s32[] parameter(1)
        t = ($0[$2], s32[$2]) custom-call(p), custom_call_target="TopK"$3
        v = $0[$2] get-tuple-element(t), index=0
        i = s32[$2] get-tuple-element(t), index=1
        n = s32[] constant($7)
        id = u32[] partition-id()
        groups = u32[8] constant({0, 0, 1, 1, 2, 2, 3, 3})
        group = u32[1] dynamic-slice(groups, id), dynamic_slice_sizes={1}
        group_scalar = u32[] reshape(group)
        o = s32[] convert(group_scalar)
        on = s32[] multiply(o, n)
        b = s32[$2] broadcast(on), dimensions={}
        offset = s32[$2] add(i, b)
        offset_lhs = s32[$2] add(b, i)
        id_s32 = s32[] convert(id)
        idn = s32[] multiply(id_s32, n)
        bid = s32[$2] broadcast(idn), dimensions={}
        partition_offset = s32[$2] add(i, bid)
        xn = s32[] multiply(x, n)
        bx = s32[$2] broadcast(xn), dimensions={}
        unbounded_offset = s32[$2] add(i, bx)
        far = s32[2] constant({0, 33554432})
        far_slot = s32[1] dynamic-slice(far, id), dynamic_slice_sizes={1}
        far_scalar = s32[] reshape(far_slot)
        far_n = s32[] multiply(far_scalar, n)
        bfar = s32[$2] broadcast(far_n), dimensions={}
        far_offset = s32[$2] add(i, bfar)
        near = s32[2] constant({0, 2147483585})
        near_slot = s32[1] dynamic-slice(near, id), dynamic_slice_sizes={1}
        near_scalar = s32[] reshape(near_slot)
        bnear = s32[$2] broadcast(near_scalar), dimensions={}
        near_offset = s32[$2] add(i, bnear)
        doubled = s32[$2] add(i, i)
        ROOT s = ($0[$2], s32[$2]) sort(v, $5), dimensions={$6}, to_apply=cmp
      })",
        type, in, out, config_attr, comparator, indices, dimension,
        row_length));
    CHECK_OK(module.status());
    auto& debug_options = (*module)->mutable_config().mutable_debug_options();
    debug_options.set_xla_gpu_experimental_enable_triton_topk(
        flags != TritonTopKFlags::kOff);
    debug_options.set_xla_gpu_experimental_force_triton_topk(
        flags == TritonTopKFlags::kForced);
    return *std::move(module);
  }

  // Parses an entry computation of `body` with the bf16 SPMD comparator `cmp`,
  // with the Triton TopK kernels enabled.
  std::unique_ptr<HloModule> ParseEntryWithSpmdComparator(
      absl::string_view body) {
    auto module = ParseAndReturnVerifiedModule(absl::Substitute(
        R"(
      HloModule m
      cmp {
        v0 = bf16[] parameter(0)
        v1 = bf16[] parameter(1)
        i0 = s32[] parameter(2)
        i1 = s32[] parameter(3)$0
      }
      ENTRY e {$1
      })",
        kSpmdComparator, body));
    CHECK_OK(module.status());
    (*module)
        ->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_enable_triton_topk(true);
    return *std::move(module);
  }

  static bool HasSort(const HloModule& module) {
    return absl::c_any_of(module.entry_computation()->instructions(),
                          [](const HloInstruction* inst) {
                            return inst->opcode() == HloOpcode::kSort;
                          });
  }

  absl::StatusOr<bool> RunOnMi350(HloModule* module) {
    return RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                      module);
  }
};

TEST_F(TopkSpecializerTritonSortTest, RemovesTheSpmdMergeSortAfterTheTopK) {
  auto module = ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                                  TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_FALSE(HasSort(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(root->operand(1)->name(), "offset");
}

TEST_F(TopkSpecializerTritonSortTest, RemovesTheSortOfOtherIndexForms) {
  for (absl::string_view indices : {"i", "offset_lhs", "partition_offset"}) {
    SCOPED_TRACE(indices);
    auto module =
        ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                          TritonTopKFlags::kOn, kSpmdComparator, indices);
    ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_FALSE(HasSort(*module));
  }
}

TEST_F(TopkSpecializerTritonSortTest, PartitionCountBoundsThePartitionOffset) {
  // The largest partition id times 64, plus 63, must stay below 2^31.
  for (auto [num_partitions, removed] :
       {std::pair{int64_t{8}, true}, std::pair{int64_t{1} << 25, true},
        std::pair{(int64_t{1} << 25) + 1, false}}) {
    SCOPED_TRACE(num_partitions);
    auto module = ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                                    TritonTopKFlags::kOn, kSpmdComparator,
                                    "partition_offset");
    module->mutable_config().set_num_partitions(num_partitions);
    ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_EQ(HasSort(*module), !removed);
  }
}

TEST_F(TopkSpecializerTritonSortTest, RemovesTheSortAfterAChunkedTopK) {
  // 16 chunks of 4096 per row: the TopK results come through the tuple that
  // drops the scratch output.
  auto module =
      ParseTopKThenSort("f32", "8,65536", "8,8", "", TritonTopKFlags::kForced);
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_FALSE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, RemovesTheSortOfValuesOnly) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule m
    cmp {
      v0 = bf16[] parameter(0)
      v1 = bf16[] parameter(1)
      ROOT gt = pred[] compare(v0, v1), direction=GT, type=TOTALORDER
    }
    ENTRY e {
      p = bf16[24576,64] parameter(0)
      t = (bf16[24576,6], s32[24576,6]) custom-call(p), custom_call_target="TopK"
      v = bf16[24576,6] get-tuple-element(t), index=0
      ROOT s = bf16[24576,6] sort(v), dimensions={1}, to_apply=cmp
    })"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_triton_topk(true);
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(HasSort(*module));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kGetTupleElement);
}

TEST_F(TopkSpecializerTritonSortTest, KeepsTheSortAfterAPartialOrderTopK) {
  // A PARTIAL order TopK treats -0 and +0 as equal; the sort does not.
  auto module = ParseTopKThenSort("bf16", "24576,64", "24576,6",
                                  R"({is_stable = false, order = "PARTIAL"})",
                                  TritonTopKFlags::kForced);
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, KeepsTheSortAfterTodaysTopK) {
  // Today's path is faster, so the TopK keeps it.
  auto module =
      ParseTopKThenSort("f32", "8,16384", "8,128", "", TritonTopKFlags::kOn);
  ASSERT_OK(RunOnMi350(module.get()));
  EXPECT_FALSE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, FlagOffKeepsTheSort) {
  auto module = ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                                  TritonTopKFlags::kOff);
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_TRUE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, KeepsTheSortOfIndicesOfAnotherTopK) {
  auto module = ParseEntryWithSpmdComparator(R"(
      p = bf16[24576,64] parameter(0)
      q = bf16[24576,64] parameter(1)
      t = (bf16[24576,6], s32[24576,6]) custom-call(p), custom_call_target="TopK"
      u = (bf16[24576,6], s32[24576,6]) custom-call(q), custom_call_target="TopK"
      v = bf16[24576,6] get-tuple-element(t), index=0
      j = s32[24576,6] get-tuple-element(u), index=1
      ROOT s = (bf16[24576,6], s32[24576,6]) sort(v, j), dimensions={1}, to_apply=cmp)");
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, KeepsASortWithControlDependencies) {
  auto module = ParseEntryWithSpmdComparator(R"(
      p = bf16[24576,64] parameter(0)
      o = s32[] parameter(1)
      n = s32[] negate(o)
      t = (bf16[24576,6], s32[24576,6]) custom-call(p), custom_call_target="TopK"
      v = bf16[24576,6] get-tuple-element(t), index=0
      i = s32[24576,6] get-tuple-element(t), index=1
      ROOT s = (bf16[24576,6], s32[24576,6]) sort(v, i), dimensions={1}, to_apply=cmp, control-predecessors={n})");
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
  // No replacement tuple is left behind.
  EXPECT_FALSE(absl::c_any_of(module->entry_computation()->instructions(),
                              [](const HloInstruction* inst) {
                                return inst->opcode() == HloOpcode::kTuple;
                              }));
}

struct KeptSortCase {
  std::string name;
  std::string comparator;
  std::string indices;
  int64_t dimension;
};

class TopkSpecializerTritonKeptSortTest
    : public TopkSpecializerTritonSortTest,
      public ::testing::WithParamInterface<KeptSortCase> {};

TEST_P(TopkSpecializerTritonKeptSortTest, SortIsKept) {
  const KeptSortCase& c = GetParam();
  auto module =
      ParseTopKThenSort("bf16", "24576,64", "24576,6", "", TritonTopKFlags::kOn,
                        c.comparator, c.indices, c.dimension);
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
}

INSTANTIATE_TEST_SUITE_P(
    Sorts, TopkSpecializerTritonKeptSortTest,
    ::testing::Values(
        KeptSortCase{"HigherIndexFirst", R"(
        eq = pred[] compare(v0, v1), direction=EQ, type=TOTALORDER
        gt_i = pred[] compare(i0, i1), direction=GT
        gt = pred[] compare(v0, v1), direction=GT, type=TOTALORDER
        ROOT r = pred[] select(eq, gt_i, gt))",
                     "offset", 1},
        KeptSortCase{"Ascending", R"(
        eq = pred[] compare(v0, v1), direction=EQ, type=TOTALORDER
        lt_i = pred[] compare(i0, i1), direction=LT
        lt = pred[] compare(v0, v1), direction=LT, type=TOTALORDER
        ROOT r = pred[] select(eq, lt_i, lt))",
                     "offset", 1},
        KeptSortCase{"PartialOrderCompare", R"(
        eq = pred[] compare(v0, v1), direction=EQ
        lt_i = pred[] compare(i0, i1), direction=LT
        gt = pred[] compare(v0, v1), direction=GT
        ROOT r = pred[] select(eq, lt_i, gt))",
                     "offset", 1},
        KeptSortCase{"SwappedParameters", R"(
        eq = pred[] compare(v0, v1), direction=EQ, type=TOTALORDER
        lt_i = pred[] compare(i1, i0), direction=LT
        gt = pred[] compare(v0, v1), direction=GT, type=TOTALORDER
        ROOT r = pred[] select(eq, lt_i, gt))",
                     "offset", 1},
        KeptSortCase{"PerElementOffset", std::string(kSpmdComparator),
                     "doubled", 1},
        // Could wrap past 2^31 and reorder the indices.
        KeptSortCase{"UnboundedOffset", std::string(kSpmdComparator),
                     "unbounded_offset", 1},
        KeptSortCase{"OffsetPast2To31", std::string(kSpmdComparator),
                     "far_offset", 1},
        KeptSortCase{"IndexPlusOffsetPast2To31", std::string(kSpmdComparator),
                     "near_offset", 1},
        KeptSortCase{"OtherDimension", std::string(kSpmdComparator), "offset",
                     0}),
    [](const ::testing::TestParamInfo<KeptSortCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla::gpu
