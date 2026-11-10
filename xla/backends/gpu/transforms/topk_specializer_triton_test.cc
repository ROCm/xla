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

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
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

TEST_F(TopkSpecializerTritonTest, FlagOnKeepsShapesOutsideTheSpeedTable) {
  auto module = ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                          TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(TopkSpecializerTritonTest, FlagOnRoutesShapesInTheSpeedTable) {
  auto module = ParseTopK("bf16[24576,64]", "bf16[24576,6]", "s32[24576,6]", "",
                          TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  ExpectMarkedTopK(*module, /*stable=*/true, ComparisonOrder::kTotal);
}

TEST_F(TopkSpecializerTritonTest, FlagOnDoesNotRouteWithoutATable) {
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

TEST_F(TopkSpecializerTritonTest, FlagOnRoutesLongRowsInTheSpeedTable) {
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

TEST_F(TopkSpecializerTritonTest, LongRowOutsideTheSpeedTableKeepsTodaysPath) {
  // Many rows of 32769..65535 with k = 1: today's prebuilt kernel was faster.
  auto module = ParseTopK("f32[1024,40000]", "f32[1024,1]", "s32[1024,1]", "",
                          TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(HasTritonTopK(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(Cast<HloCustomCallInstruction>(root->operand(0)->operand(0))
                ->custom_call_target(),
            "__gpu$TopK");
}

constexpr absl::string_view kDeferredPackedBf16 =
    R"(frontend_attributes={_xla_topk_deferred_packed_bf16="true"})";

TEST_F(TopkSpecializerTritonTest, DeferredPackedBf16KeepsTheTopKWhenNotRouted) {
  // Not in the speed table. Without the mark, it takes the prebuilt kernel.
  auto unmarked = ParseTopK("bf16[8,1024]", "bf16[8,16]", "s32[8,16]", "",
                            TritonTopKFlags::kOn);
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 unmarked.get()));
  EXPECT_TRUE(changed);

  // With the mark, TopkDecomposer later gives it the packed BF16 sort.
  auto marked = ParseTopK("bf16[8,1024]", "bf16[8,16]", "s32[8,16]", "",
                          TritonTopKFlags::kOn, kDeferredPackedBf16);
  ASSERT_OK_AND_ASSIGN(
      changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
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

}  // namespace
}  // namespace xla::gpu
