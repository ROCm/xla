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

#include "xla/backends/gpu/transforms/triton_conv_fusion_rewriter.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/conv_fusion_rewriter.h"
#include "xla/backends/gpu/transforms/triton_conv_eligibility.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class TritonConvFusionRewriterTest : public HloHardwareIndependentTestBase {
 protected:
  const se::DeviceDescription device_description_{
      TestGpuDeviceInfo::AMDMI300DeviceInfo()};

  std::unique_ptr<VerifiedHloModule> ClaimAndWrap(absl::string_view hlo,
                                                  bool expect_wrapped) {
    std::unique_ptr<VerifiedHloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_enable_triton_conv_fusion(true);

    EXPECT_OK(
        TritonConvEligibility(device_description_.gpu_compute_capability())
            .Run(module.get())
            .status());
    EXPECT_THAT(TritonConvFusionRewriter(device_description_).Run(module.get()),
                IsOkAndHolds(expect_wrapped));
    EXPECT_OK(verifier().Run(module.get()).status());
    return module;
  }
};

constexpr absl::string_view kPlainConv = R"(
ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(p0, p1),
    window={size=3x3}, dim_labels=b01f_01io->b01f
})";

TEST_F(TritonConvFusionRewriterTest, WrapsAClaimedConvolution) {
  std::unique_ptr<VerifiedHloModule> module =
      ClaimAndWrap(kPlainConv, /*expect_wrapped=*/true);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK: ROOT{{.*}}fusion(
    CHECK-SAME: kind=kCustom
    CHECK-SAME: "kind":"__triton_conv"
  )"));
}

TEST_F(TritonConvFusionRewriterTest, WrapsTheConvolutionAlone) {
  // No prologue or epilogue fusion at this point.
  constexpr absl::string_view kConvThenNegate = R"(
ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  conv = f32[1,6,6,16] convolution(p0, p1),
    window={size=3x3}, dim_labels=b01f_01io->b01f
  ROOT neg = f32[1,6,6,16] negate(conv)
})";
  std::unique_ptr<VerifiedHloModule> module =
      ClaimAndWrap(kConvThenNegate, /*expect_wrapped=*/true);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kNegate);
  EXPECT_EQ(
      module->entry_computation()->root_instruction()->operand(0)->opcode(),
      HloOpcode::kFusion);
  const HloComputation* fused = module->entry_computation()
                                    ->root_instruction()
                                    ->operand(0)
                                    ->fused_instructions_computation();
  EXPECT_EQ(fused->instruction_count(), 3);
  EXPECT_EQ(fused->root_instruction()->opcode(), HloOpcode::kConvolution);
}

TEST_F(TritonConvFusionRewriterTest, LeavesUnclaimedConvolutionsAlone) {
  constexpr absl::string_view kGrouped = R"(
ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,16,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(p0, p1),
    window={size=3x3}, dim_labels=b01f_01io->b01f, feature_group_count=2
})";
  std::unique_ptr<VerifiedHloModule> module =
      ClaimAndWrap(kGrouped, /*expect_wrapped=*/false);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kConvolution);
}

TEST_F(TritonConvFusionRewriterTest, WrapsOnlyTheEligibleOfTwoConvolutions) {
  constexpr absl::string_view kTwoConvs = R"(
ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  p2 = f32[3,3,16,16] parameter(2)
  good = f32[1,6,6,16] convolution(p0, p1),
    window={size=3x3}, dim_labels=b01f_01io->b01f
  bad = f32[1,6,6,16] convolution(p0, p2),
    window={size=3x3}, dim_labels=b01f_01io->b01f, feature_group_count=2
  ROOT sum = f32[1,6,6,16] add(good, bad)
})";
  std::unique_ptr<VerifiedHloModule> module =
      ClaimAndWrap(kTwoConvs, /*expect_wrapped=*/true);

  const HloInstruction* sum = module->entry_computation()->root_instruction();
  ASSERT_EQ(sum->opcode(), HloOpcode::kAdd);

  int fusion_count = 0;
  int conv_count = 0;
  for (const HloInstruction* operand : sum->operands()) {
    if (operand->opcode() == HloOpcode::kFusion) {
      ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                           operand->backend_config<GpuBackendConfig>());
      EXPECT_EQ(gpu_config.fusion_backend_config().kind(),
                kTritonConvFusionKind);
      ++fusion_count;
    } else if (operand->opcode() == HloOpcode::kConvolution) {
      ++conv_count;
    }
  }
  EXPECT_EQ(fusion_count, 1);
  EXPECT_EQ(conv_count, 1);
}

TEST_F(TritonConvFusionRewriterTest, IgnoresConvolutionsInsideFusions) {
  constexpr absl::string_view kFusedConv = R"(
fused_computation {
  a = f32[1,8,8,32] parameter(0)
  b = f32[3,3,32,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(a, b),
    window={size=3x3}, dim_labels=b01f_01io->b01f
}

ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  ROOT fusion = f32[1,6,6,16] fusion(p0, p1), kind=kLoop,
    calls=fused_computation
})";
  ClaimAndWrap(kFusedConv, /*expect_wrapped=*/false);
}

TEST_F(TritonConvFusionRewriterTest,
       ConvFusionRewriterDoesNotAlsoClaimAWrappedConvolution) {
  // Sanity check; convs wrapped by TritonConvFusionRewriter are ignored by
  // ConvFusionRewriter
  std::unique_ptr<VerifiedHloModule> module =
      ClaimAndWrap(kPlainConv, /*expect_wrapped=*/true);

  EXPECT_THAT(ConvFusionRewriter(device_description_).Run(module.get()),
              IsOkAndHolds(false));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), kTritonConvFusionKind);
}

}  // namespace
}  // namespace xla::gpu
