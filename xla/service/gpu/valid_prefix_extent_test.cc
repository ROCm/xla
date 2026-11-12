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

#include "xla/service/gpu/valid_prefix_extent.h"

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"

namespace xla::gpu {
namespace {

using ValidPrefixExtentTest = HloHardwareIndependentTestBase;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr absl::string_view kValidFusionHlo = R"hlo(
HloModule ValidPrefixExtentContract

fused_computation {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  bounded = f32[8,4] custom-call(x, n),
    custom_call_target="xla.valid_prefix"
  ROOT result = f32[8,4] exponential(bounded)
}

ENTRY main {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  ROOT fusion = f32[8,4] fusion(x, n), kind=kLoop,
    calls=fused_computation
}
)hlo";

TEST_F(ValidPrefixExtentTest, DataflowContractSurvivesCloneAndSerialization) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kValidFusionHlo));

  auto verify_contract = [](const HloModule& candidate) {
    const HloInstruction* fusion =
        candidate.entry_computation()->root_instruction();
    ASSERT_OK_AND_ASSIGN(std::optional<ValidPrefixExtent> extent,
                         GetValidPrefixExtent(*fusion));
    ASSERT_TRUE(extent.has_value());
    EXPECT_EQ(extent->operand_index, 1);
    EXPECT_EQ(extent->capacity, 8);
  };

  verify_contract(*module);
  verify_contract(*module->Clone("clone"));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> roundtrip,
      HloModule::CreateFromProto(module->ToProto(), module->config()));
  verify_contract(*roundtrip);
}

TEST_F(ValidPrefixExtentTest, ScalarFusionHasNoContract) {
  constexpr absl::string_view kScalarFusionHlo = R"hlo(
HloModule ScalarFusion

fused_computation {
  x = s32[] parameter(0)
  ROOT result = s32[] negate(x)
}

ENTRY main {
  x = s32[] parameter(0)
  ROOT fusion = s32[] fusion(x), kind=kLoop, calls=fused_computation
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kScalarFusionHlo));

  ASSERT_OK_AND_ASSIGN(
      std::optional<ValidPrefixExtent> extent,
      GetValidPrefixExtent(*module->entry_computation()->root_instruction()));
  EXPECT_FALSE(extent.has_value());
}

TEST_F(ValidPrefixExtentTest, FollowsOperandDeduplication) {
  constexpr absl::string_view kDuplicateOperandHlo = R"hlo(
HloModule DeduplicatedValidPrefixExtent

fused_computation {
  x = f32[8,4] parameter(0)
  duplicate_x = f32[8,4] parameter(1)
  n = s32[] parameter(2)
  bounded = f32[8,4] custom-call(duplicate_x, n),
    custom_call_target="xla.valid_prefix"
  ROOT result = f32[8,4] add(x, bounded)
}

ENTRY main {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  ROOT fusion = f32[8,4] fusion(x, x, n), kind=kLoop,
    calls=fused_computation
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kDuplicateOperandHlo));
  auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());

  ASSERT_OK_AND_ASSIGN(std::optional<ValidPrefixExtent> before,
                       GetValidPrefixExtent(*fusion));
  ASSERT_TRUE(before.has_value());
  EXPECT_EQ(before->operand_index, 2);

  ASSERT_OK(fusion->DeduplicateFusionOperands());
  ASSERT_EQ(fusion->operand_count(), 2);
  ASSERT_OK_AND_ASSIGN(std::optional<ValidPrefixExtent> after,
                       GetValidPrefixExtent(*fusion));
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(after->operand_index, 1);
  EXPECT_EQ(fusion->operand(after->operand_index),
            module->entry_computation()->parameter_instruction(1));
}

TEST_F(ValidPrefixExtentTest, RejectsComputedExtentInsideFusion) {
  constexpr absl::string_view kComputedExtentHlo = R"hlo(
HloModule ComputedValidPrefixExtent

fused_computation {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  zero = s32[] constant(0)
  computed_n = s32[] add(n, zero)
  ROOT result = f32[8,4] custom-call(x, computed_n),
    custom_call_target="xla.valid_prefix"
}

ENTRY main {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  ROOT fusion = f32[8,4] fusion(x, n), kind=kLoop,
    calls=fused_computation
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kComputedExtentHlo));

  EXPECT_THAT(
      GetValidPrefixExtent(*module->entry_computation()->root_instruction()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("must be a fusion parameter")));
}

TEST_F(ValidPrefixExtentTest, ConflictingExtentsDoNotFormAContract) {
  constexpr absl::string_view kConflictingExtentsHlo = R"hlo(
HloModule ConflictingValidPrefixExtents

fused_computation {
  x = f32[8,4] parameter(0)
  lhs_n = s32[] parameter(1)
  rhs_n = s32[] parameter(2)
  lhs = f32[8,4] custom-call(x, lhs_n),
    custom_call_target="xla.valid_prefix"
  rhs = f32[8,4] custom-call(x, rhs_n),
    custom_call_target="xla.valid_prefix"
  ROOT result = f32[8,4] add(lhs, rhs)
}

ENTRY main {
  x = f32[8,4] parameter(0)
  lhs_n = s32[] parameter(1)
  rhs_n = s32[] parameter(2)
  ROOT fusion = f32[8,4] fusion(x, lhs_n, rhs_n), kind=kLoop,
    calls=fused_computation
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kConflictingExtentsHlo));

  ASSERT_OK_AND_ASSIGN(
      std::optional<ValidPrefixExtent> extent,
      GetValidPrefixExtent(*module->entry_computation()->root_instruction()));
  EXPECT_FALSE(extent.has_value());
}

}  // namespace
}  // namespace xla::gpu
