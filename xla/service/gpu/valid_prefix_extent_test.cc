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

#include <gtest/gtest.h>
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

TEST_F(ValidPrefixExtentTest,
       ContractSurvivesCloneSerializationAndFingerprinting) {
  constexpr absl::string_view kHlo = R"(
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
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
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

}  // namespace
}  // namespace xla::gpu
