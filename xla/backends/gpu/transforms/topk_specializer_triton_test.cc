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
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

enum class TritonTopKFlags { kOff, kOn, kForced };

class TopkSpecializerTritonTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_triton_topk(
        flags_ != TritonTopKFlags::kOff);
    debug_options.set_xla_gpu_experimental_force_triton_topk(
        flags_ == TritonTopKFlags::kForced);
    return debug_options;
  }

  // Parses a module whose root is `ROOT t = (out, idx) custom-call(p)` with
  // target "TopK" and the HLO attributes `extra_attrs`, if any.
  absl::StatusOr<std::unique_ptr<HloModule>> ParseTopK(
      absl::string_view in, absl::string_view out, absl::string_view idx,
      absl::string_view backend_config, TritonTopKFlags flags,
      absl::string_view extra_attrs = "") {
    std::string config_attr =
        backend_config.empty()
            ? ""
            : absl::Substitute(", backend_config=$0", backend_config);
    if (!extra_attrs.empty()) {
      absl::StrAppend(&config_attr, ", ", extra_attrs);
    }
    flags_ = flags;
    return ParseAndReturnVerifiedModule(absl::Substitute(
        R"hlo(
      HloModule m
      ENTRY e {
        p = $0 parameter(0)
        ROOT t = ($1, $2) custom-call(p), custom_call_target="TopK"$3
      })hlo",
        in, out, idx, config_attr));
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

  // The flags of the modules parsed next.
  TritonTopKFlags flags_ = TritonTopKFlags::kOff;
};

TEST_F(TopkSpecializerTritonTest, FlagOffKeepsTopK) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                                 TritonTopKFlags::kOff));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(TopkSpecializerTritonTest, FlagOnWithoutForceKeepsTopK) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("f32[8,16384]", "f32[8,128]", "s32[8,128]", "",
                                 TritonTopKFlags::kOn));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK(c.in, c.out, c.idx, c.backend_config,
                                 TritonTopKFlags::kForced));
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
        RoutedCase{"StablePartialF32", "f32[16,4096]", "f32[16,8]", "s32[16,8]",
                   R"({order = "PARTIAL"})", true, ComparisonOrder::kPartial},
        RoutedCase{"UnstableTotalF32", "f32[32,2048]", "f32[32,64]",
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
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopK(c.in, c.out, c.idx, "", TritonTopKFlags::kForced));
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
        RejectedCase{"Bf16", "bf16[8,1000]", "bf16[8,32]", "s32[8,32]"},
        RejectedCase{"KAboveLimit", "f32[8,1000]", "f32[8,129]", "s32[8,129]"},
        RejectedCase{"KCloseToN", "f32[8,16]", "f32[8,14]", "s32[8,14]"},
        RejectedCase{"EmptyBatch", "f32[0,1000]", "f32[0,32]", "s32[0,32]"},
        // A row of 65536 keys does not fit one program; k > 16 also keeps
        // the TopK off the prebuilt kernel.
        RejectedCase{"LongRow", "f32[8,65536]", "f32[8,32]", "s32[8,32]"}),
    [](const ::testing::TestParamInfo<RejectedCase>& info) {
      return info.param.name;
    });

TEST_F(TopkSpecializerTritonTest, ForcedTopKKeepsTheMetadata) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopK("f32[8,1000]", "f32[8,8]", "s32[8,8]", "",
                TritonTopKFlags::kForced, R"(metadata={op_name="top_k"})"));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_EQ(
      module->entry_computation()->root_instruction()->metadata().op_name(),
      "top_k");
}

TEST_F(TopkSpecializerTritonTest, CudaIsNotRouted) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                                 TritonTopKFlags::kForced));
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
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                                   TritonTopKFlags::kForced));
    ASSERT_OK_AND_ASSIGN(bool changed,
                         RunHloPass(TopkSpecializer(device), module.get()));
    EXPECT_TRUE(changed);
    ExpectMarkedTopK(*module, /*stable=*/true, ComparisonOrder::kTotal);
  }
}

TEST_F(TopkSpecializerTritonTest, RowLimitFollowsTheWaveSize) {
  // 16384 keys fit a wave-64 program of 8 warps, but not a wave-32 one.
  for (const auto& [device, fits] :
       {std::pair{TestGpuDeviceInfo::AMDMI350DeviceInfo(), true},
        std::pair{TestGpuDeviceInfo::AMDRX7900DeviceInfo(), false}}) {
    SCOPED_TRACE(device.name());
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseTopK("f32[8,16384]", "f32[8,32]", "s32[8,32]", "",
                                   TritonTopKFlags::kForced));
    ASSERT_THAT(RunHloPass(TopkSpecializer(device), module.get()),
                absl_testing::IsOk());
    EXPECT_EQ(HasTritonTopK(*module), fits);
  }
}

}  // namespace
}  // namespace xla::gpu
