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
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
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
  predicate = pred[] compare(n, n), direction=EQ
  broadcast = pred[8,4] broadcast(predicate), dimensions={}
  ROOT result = f32[8,4] select(broadcast, x, x)
}

ENTRY main {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  ROOT fusion = f32[8,4] fusion(x, n), kind=kLoop,
    calls=fused_computation
}
)hlo";

TEST_F(ValidPrefixExtentTest,
       ContractSurvivesCloneSerializationAndFingerprinting) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kValidFusionHlo));
  std::unique_ptr<HloModule> without_extent = module->Clone("without_extent");
  ASSERT_OK(SetValidPrefixExtent(
      module->entry_computation()->root_instruction(), /*operand_index=*/1));

  auto verify_contract = [](const HloModule& candidate) {
    const HloInstruction* fusion =
        candidate.entry_computation()->root_instruction();
    ASSERT_OK_AND_ASSIGN(std::optional<ValidPrefixExtent> extent,
                         GetValidPrefixExtent(*fusion));
    ASSERT_TRUE(extent.has_value());
    EXPECT_EQ(extent->operand_index, 1);
    EXPECT_EQ(extent->dimension, 0);
    EXPECT_EQ(extent->capacity, 8);
  };

  verify_contract(*module);
  verify_contract(*module->Clone("clone"));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> roundtrip,
      HloModule::CreateFromProto(module->ToProto(), module->config()));
  verify_contract(*roundtrip);

  HloPrintOptions fingerprint_options =
      HloPrintOptions::Fingerprint().set_print_backend_config(true);
  EXPECT_NE(module->GetFingerprint128(fingerprint_options),
            without_extent->GetFingerprint128(fingerprint_options));
  EXPECT_FALSE(module->entry_computation()->root_instruction()->Identical(
      *without_extent->entry_computation()->root_instruction(),
      [](const HloInstruction*, const HloInstruction*) { return true; },
      [](const HloComputation*, const HloComputation*) { return true; }));
}

TEST_F(ValidPrefixExtentTest, RejectsInvalidSetterInputs) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kValidFusionHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  EXPECT_THAT(SetValidPrefixExtent(
                  module->entry_computation()->parameter_instruction(0),
                  /*operand_index=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("requires a fusion instruction")));
  EXPECT_THAT(SetValidPrefixExtent(fusion, /*operand_index=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("valid fusion operand")));
  EXPECT_THAT(SetValidPrefixExtent(fusion, /*operand_index=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("s32 scalar operand")));
  EXPECT_THAT(SetValidPrefixExtent(nullptr, /*operand_index=*/0),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("instruction != nullptr")));
}

TEST_F(ValidPrefixExtentTest, RejectsScalarFusionOutput) {
  constexpr absl::string_view kScalarOutputHlo = R"hlo(
HloModule ScalarFusionOutput

fused_computation {
  n = s32[] parameter(0)
  ROOT result = s32[] copy(n)
}

ENTRY main {
  n = s32[] parameter(0)
  ROOT fusion = s32[] fusion(n), kind=kLoop, calls=fused_computation
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kScalarOutputHlo));

  EXPECT_THAT(
      SetValidPrefixExtent(module->entry_computation()->root_instruction(),
                           /*operand_index=*/0),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("array output with rank at least one")));
}

TEST_F(ValidPrefixExtentTest, RejectsInvalidSerializedDimension) {
  constexpr absl::string_view kInvalidDimensionHlo = R"hlo(
HloModule InvalidValidPrefixDimension

fused_computation {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  predicate = pred[] compare(n, n), direction=EQ
  broadcast = pred[8,4] broadcast(predicate), dimensions={}
  ROOT result = f32[8,4] select(broadcast, x, x)
}

ENTRY main {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  ROOT fusion = f32[8,4] fusion(x, n), kind=kLoop,
    calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"kLoop"},"valid_prefix_extent":{"operand_index":"1","dimension":"1"}}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kInvalidDimensionHlo));

  EXPECT_THAT(
      GetValidPrefixExtent(*module->entry_computation()->root_instruction()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("only supports output dimension 0")));
}

}  // namespace
}  // namespace xla::gpu
