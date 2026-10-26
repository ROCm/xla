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

#include "xla/backends/gpu/transforms/triton_conv_eligibility.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class TritonConvEligibilityTest : public HloHardwareIndependentTestBase {
 protected:
  const se::DeviceDescription device_description_{
      TestGpuDeviceInfo::AMDMI300DeviceInfo()};

  Decision Eligible(absl::string_view hlo) {
    module_ = ParseAndReturnVerifiedModule(hlo).value();
    HloInstruction* conv = hlo_query::GetFirstInstructionWithOpcode(
        *module_->entry_computation(), HloOpcode::kConvolution);
    return IsEligibleForTritonConv(
        *Cast<HloConvolutionInstruction>(conv),
        device_description_.gpu_compute_capability());
  }

  std::unique_ptr<VerifiedHloModule> module_;
};

constexpr absl::string_view kPlainConv = R"(
ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(p0, p1),
    window={size=3x3}, dim_labels=b01f_01io->b01f
})";

TEST_F(TritonConvEligibilityTest, AllowsPlainForwardConvolution) {
  EXPECT_TRUE(Eligible(kPlainConv).IsAllowed());
}

TEST_F(TritonConvEligibilityTest, AllowsPaddedStridedConvolution) {
  EXPECT_TRUE(Eligible(R"(
ENTRY entry {
  p0 = f32[1,16,16,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  ROOT conv = f32[1,8,8,16] convolution(p0, p1),
    window={size=3x3 stride=2x2 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
})")
                  .IsAllowed());
}

TEST_F(TritonConvEligibilityTest, ForbidsGroupedConvolution) {
  EXPECT_TRUE(Eligible(R"(
ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,16,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(p0, p1),
    window={size=3x3}, dim_labels=b01f_01io->b01f, feature_group_count=2
})")
                  .IsForbidden());
}

TEST_F(TritonConvEligibilityTest, ForbidsBaseDilatedConvolution) {
  EXPECT_TRUE(Eligible(R"(
ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  ROOT conv = f32[1,13,13,16] convolution(p0, p1),
    window={size=3x3 lhs_dilate=2x2}, dim_labels=b01f_01io->b01f
})")
                  .IsForbidden());
}

TEST_F(TritonConvEligibilityTest, ForbidsWindowReversal) {
  EXPECT_TRUE(Eligible(R"(
ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(p0, p1),
    window={size=3x3 rhs_reversal=1x1}, dim_labels=b01f_01io->b01f
})")
                  .IsForbidden());
}

TEST_F(TritonConvEligibilityTest, ForbidsBackwardFilterShape) {
  EXPECT_TRUE(Eligible(R"(
ENTRY entry {
  input = f32[8,120,256,256] parameter(0)
  filter = f32[8,120,256,256] parameter(1)
  ROOT conv = f32[120,120,3,3] convolution(input, filter),
    window={size=256x256 pad=1_1x1_1}, dim_labels=fb01_io01->fb01
})")
                  .IsForbidden());
}

TEST_F(TritonConvEligibilityTest, ForbidsAlreadyClaimedConvolution) {
  module_ = ParseAndReturnVerifiedModule(kPlainConv).value();
  HloConvolutionInstruction* conv = Cast<HloConvolutionInstruction>(
      module_->entry_computation()->root_instruction());
  conv->set_convolution_kind(CONVOLUTION_KIND_FPROP);
  EXPECT_TRUE(IsEligibleForTritonConv(
                  *conv, device_description_.gpu_compute_capability())
                  .IsForbidden());
}

class TritonConvEligibilityPassTest : public TritonConvEligibilityTest {
 protected:
  std::unique_ptr<VerifiedHloModule> Run(absl::string_view hlo,
                                         bool expect_change) {
    std::unique_ptr<VerifiedHloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_enable_triton_conv_fusion(true);
    EXPECT_THAT(
        TritonConvEligibility(device_description_.gpu_compute_capability())
            .Run(module.get()),
        IsOkAndHolds(expect_change));
    return module;
  }

  ConvolutionKind KindOf(const VerifiedHloModule& module,
                         absl::string_view name) {
    return Cast<HloConvolutionInstruction>(FindInstruction(&module, name))
        ->convolution_kind();
  }
};

TEST_F(TritonConvEligibilityPassTest, ClaimsSupportedConvolution) {
  std::unique_ptr<VerifiedHloModule> module =
      Run(kPlainConv, /*expect_change=*/true);
  EXPECT_EQ(KindOf(*module, "conv"), CONVOLUTION_KIND_FPROP);
}

TEST_F(TritonConvEligibilityPassTest, LeavesUnsupportedConvolutionUnclaimed) {
  constexpr absl::string_view kGrouped = R"(
ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,16,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(p0, p1),
    window={size=3x3}, dim_labels=b01f_01io->b01f, feature_group_count=2
})";
  std::unique_ptr<VerifiedHloModule> module =
      Run(kGrouped, /*expect_change=*/false);
  EXPECT_EQ(KindOf(*module, "conv"), CONVOLUTION_KIND_UNSET);
}

TEST_F(TritonConvEligibilityPassTest, ClaimsOnlyTheEligibleOfTwoConvolutions) {
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
      Run(kTwoConvs, /*expect_change=*/true);
  EXPECT_EQ(KindOf(*module, "good"), CONVOLUTION_KIND_FPROP);
  EXPECT_EQ(KindOf(*module, "bad"), CONVOLUTION_KIND_UNSET);
}

TEST_F(TritonConvEligibilityPassTest, IgnoresConvolutionsInsideFusions) {
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
  Run(kFusedConv, /*expect_change=*/false);
}

}  // namespace
}  // namespace xla::gpu
