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

#include "xla/backends/gpu/autotuner/hipblaslt_scale_utils.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/substitute.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace xla::gpu {
namespace {

using HipblasLtScaleUtilsTest = HloHardwareIndependentTestBase;

std::string ScaledDotHlo(int64_t m = 64, int64_t n = 96, int64_t k = 512,
                         const char* dtype = "f8e4m3fn",
                         const char* input_layout = "1,0",
                         const char* scale_layout = "1,0") {
  return absl::Substitute(R"(
HloModule scales
ENTRY main {
  a = $4[$0,$2]{$5} parameter(0)
  b = $4[$1,$2]{$5} parameter(1)
  sa = f8e8m0fnu[$0,$3]{$6} parameter(2)
  sb = f8e8m0fnu[$1,$3]{$6} parameter(3)
  ROOT dot = f32[$0,$1] scaled-dot(a, b, sa, sb),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})",
                          m, n, k, k / 32, dtype, input_layout, scale_layout);
}

TEST_F(HipblasLtScaleUtilsTest, SupportsAlignedGfx950) {
  auto module_or = ParseAndReturnVerifiedModule(ScaledDotHlo());
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  EXPECT_TRUE(CanUseHipblasLtScale32x8(
                  *module->entry_computation()->root_instruction(),
                  stream_executor::GpuComputeCapability(
                      stream_executor::RocmComputeCapability("gfx950")))
                  .IsAllowed());
}

TEST_F(HipblasLtScaleUtilsTest, SupportsPackedFp4Gfx950) {
  auto module_or = ParseAndReturnVerifiedModule(
      ScaledDotHlo(64, 96, 512, "f4e2m1fn", "1,0:E(4)"));
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  EXPECT_TRUE(CanUseHipblasLtScale32x8(
                  *module->entry_computation()->root_instruction(),
                  stream_executor::GpuComputeCapability(
                      stream_executor::RocmComputeCapability("gfx950")))
                  .IsAllowed());
}

TEST_F(HipblasLtScaleUtilsTest, RejectsUnpackedFp4Storage) {
  for (int operand : {0, 1}) {
    for (int element_size : {0, 8}) {
      auto module_or = ParseAndReturnVerifiedModule(
          ScaledDotHlo(64, 96, 512, "f4e2m1fn", "1,0:E(4)"));
      ASSERT_THAT(module_or, absl_testing::IsOk());
      auto module = std::move(module_or).value();
      HloInstruction* dot = module->entry_computation()->root_instruction();
      dot->mutable_operand(operand)
          ->mutable_shape()
          ->mutable_layout()
          ->set_element_size_in_bits(element_size);
      EXPECT_FALSE(CanUseHipblasLtScale32x8(
                       *dot, stream_executor::GpuComputeCapability(
                                 stream_executor::RocmComputeCapability("gfx950")))
                       .IsAllowed())
          << "operand=" << operand << " element_size=" << element_size;
    }
  }
}

TEST_F(HipblasLtScaleUtilsTest, RejectsMixedFp4Fp8Inputs) {
  for (bool lhs_is_fp4 : {false, true}) {
    std::string hlo = absl::Substitute(R"(
HloModule mixed_types
ENTRY main {
  a = $0[64,512]{1,0$2} parameter(0)
  b = $1[96,512]{1,0$3} parameter(1)
  sa = f8e8m0fnu[64,16]{1,0} parameter(2)
  sb = f8e8m0fnu[96,16]{1,0} parameter(3)
  ROOT dot = f32[64,96] scaled-dot(a, b, sa, sb),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})",
                                       lhs_is_fp4 ? "f4e2m1fn" : "f8e4m3fn",
                                       lhs_is_fp4 ? "f8e4m3fn" : "f4e2m1fn",
                                       lhs_is_fp4 ? ":E(4)" : "",
                                       lhs_is_fp4 ? "" : ":E(4)");
    auto module_or = ParseAndReturnVerifiedModule(hlo);
    ASSERT_THAT(module_or, absl_testing::IsOk());
    auto module = std::move(module_or).value();
    HloInstruction* dot = module->entry_computation()->root_instruction();
    EXPECT_FALSE(CanUseHipblasLtScale32x8(
                     *dot, stream_executor::GpuComputeCapability(
                               stream_executor::RocmComputeCapability("gfx950")))
                     .IsAllowed());
  }
}

TEST_F(HipblasLtScaleUtilsTest, Fp4RequiresByteScales) {
  for (int element_size : {0, 4, 8}) {
    auto module_or = ParseAndReturnVerifiedModule(
        ScaledDotHlo(64, 96, 512, "f4e2m1fn", "1,0:E(4)"));
    ASSERT_THAT(module_or, absl_testing::IsOk());
    auto module = std::move(module_or).value();
    HloInstruction* dot = module->entry_computation()->root_instruction();
    for (int operand : {2, 3}) {
      dot->mutable_operand(operand)
          ->mutable_shape()
          ->mutable_layout()
          ->set_element_size_in_bits(element_size);
    }
    EXPECT_EQ(CanUseHipblasLtScale32x8(
                  *dot, stream_executor::GpuComputeCapability(
                            stream_executor::RocmComputeCapability("gfx950")))
                  .IsAllowed(),
              element_size != 4);
  }
}

TEST_F(HipblasLtScaleUtilsTest, ScaleLayoutSurvivesAutotuneSerialization) {
  AutotuneResult::GemmKey legacy;
  EXPECT_EQ(legacy.mx_scale_layout(),
            AutotuneResult::GemmKey::MX_SCALE_LAYOUT_LINEAR);
  AutotuneResult::GemmKey packed;
  packed.set_algorithm(7);
  packed.set_autotune_workspace_size(64 * 1024 * 1024);
  packed.set_mx_scale_layout(
      AutotuneResult::GemmKey::MX_SCALE_LAYOUT_HIPBLASLT_32X8);
  AutotuneResult::GemmKey reloaded;
  ASSERT_TRUE(reloaded.ParseFromString(packed.SerializeAsString()));
  EXPECT_EQ(reloaded.algorithm(), 7);
  EXPECT_EQ(reloaded.autotune_workspace_size(), 64 * 1024 * 1024);
  EXPECT_EQ(reloaded.mx_scale_layout(), packed.mx_scale_layout());
}

TEST_F(HipblasLtScaleUtilsTest, RejectsOtherMxDevices) {
  auto module_or = ParseAndReturnVerifiedModule(ScaledDotHlo());
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  for (const char* arch : {"gfx942", "gfx1250"}) {
    EXPECT_FALSE(CanUseHipblasLtScale32x8(
                     *module->entry_computation()->root_instruction(),
                     stream_executor::GpuComputeCapability(
                         stream_executor::RocmComputeCapability(arch)))
                     .IsAllowed());
  }
}

TEST_F(HipblasLtScaleUtilsTest, RejectsPaddingTypesAndNoncanonicalLayouts) {
  for (const std::string& hlo :
       {ScaledDotHlo(16, 96, 512), ScaledDotHlo(64, 16, 512),
        ScaledDotHlo(64, 96, 288), ScaledDotHlo(64, 96, 512, "f8e5m2"),
        ScaledDotHlo(64, 96, 512, "f8e4m3fn", "0,1"),
        ScaledDotHlo(64, 96, 512, "f8e4m3fn", "1,0", "0,1"),
        ScaledDotHlo(16, 96, 512, "f4e2m1fn", "1,0:E(4)"),
        ScaledDotHlo(64, 16, 512, "f4e2m1fn", "1,0:E(4)"),
        ScaledDotHlo(64, 96, 288, "f4e2m1fn", "1,0:E(4)"),
        ScaledDotHlo(64, 96, 512, "f4e2m1fn", "0,1:E(4)"),
        ScaledDotHlo(64, 96, 512, "f4e2m1fn", "1,0:E(4)", "0,1")}) {
    auto module_or = ParseAndReturnVerifiedModule(hlo);
    ASSERT_THAT(module_or, absl_testing::IsOk());
    auto module = std::move(module_or).value();
    EXPECT_FALSE(CanUseHipblasLtScale32x8(
                     *module->entry_computation()->root_instruction(),
                     stream_executor::GpuComputeCapability(
                         stream_executor::RocmComputeCapability("gfx950")))
                     .IsAllowed())
        << hlo;
  }
}

TEST_F(HipblasLtScaleUtilsTest, OnlySupportsUnitBatch) {
  for (int batch : {1, 2}) {
    std::string hlo = absl::Substitute(R"(
HloModule batch
ENTRY main {
  a = f8e4m3fn[$0,64,512]{2,1,0} parameter(0)
  b = f8e4m3fn[$0,96,512]{2,1,0} parameter(1)
  sa = f8e8m0fnu[$0,64,16]{2,1,0} parameter(2)
  sb = f8e8m0fnu[$0,96,16]{2,1,0} parameter(3)
  ROOT dot = f32[$0,64,96] scaled-dot(a, b, sa, sb),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})",
                                       batch);
    auto module_or = ParseAndReturnVerifiedModule(hlo);
    ASSERT_THAT(module_or, absl_testing::IsOk());
    auto module = std::move(module_or).value();
    EXPECT_EQ(CanUseHipblasLtScale32x8(
                  *module->entry_computation()->root_instruction(),
                  stream_executor::GpuComputeCapability(
                      stream_executor::RocmComputeCapability("gfx950")))
                  .IsAllowed(),
              batch == 1);
  }
}

TEST_F(HipblasLtScaleUtilsTest, Fp4OnlySupportsUnitBatch) {
  for (int batch : {1, 2}) {
    std::string hlo = absl::Substitute(R"(
HloModule batch
ENTRY main {
  a = f4e2m1fn[$0,64,512]{2,1,0:E(4)} parameter(0)
  b = f4e2m1fn[$0,96,512]{2,1,0:E(4)} parameter(1)
  sa = f8e8m0fnu[$0,64,16]{2,1,0} parameter(2)
  sb = f8e8m0fnu[$0,96,16]{2,1,0} parameter(3)
  ROOT dot = f32[$0,64,96] scaled-dot(a, b, sa, sb),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})",
                                       batch);
    auto module_or = ParseAndReturnVerifiedModule(hlo);
    ASSERT_THAT(module_or, absl_testing::IsOk());
    auto module = std::move(module_or).value();
    EXPECT_EQ(CanUseHipblasLtScale32x8(
                  *module->entry_computation()->root_instruction(),
                  stream_executor::GpuComputeCapability(
                      stream_executor::RocmComputeCapability("gfx950")))
                  .IsAllowed(),
              batch == 1);
  }
}

TEST_F(HipblasLtScaleUtilsTest, RejectsTiledByteStorage) {
  auto module_or = ParseAndReturnVerifiedModule(ScaledDotHlo());
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  HloInstruction* dot = module->entry_computation()->root_instruction();
  // A row-major minor_to_major order alone does not imply the required byte
  // layout when the shape also has a physical tile.
  Tile* tile =
      dot->mutable_operand(2)->mutable_shape()->mutable_layout()->add_tiles();
  tile->add_dimensions(32).add_dimensions(8);
  EXPECT_FALSE(CanUseHipblasLtScale32x8(
                   *dot, stream_executor::GpuComputeCapability(
                             stream_executor::RocmComputeCapability("gfx950")))
                   .IsAllowed());
}

TEST_F(HipblasLtScaleUtilsTest, RejectsTiledFp4Storage) {
  auto module_or = ParseAndReturnVerifiedModule(
      ScaledDotHlo(64, 96, 512, "f4e2m1fn", "1,0:E(4)"));
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  HloInstruction* dot = module->entry_computation()->root_instruction();
  Tile* tile =
      dot->mutable_operand(0)->mutable_shape()->mutable_layout()->add_tiles();
  tile->add_dimensions(32).add_dimensions(32);
  EXPECT_FALSE(CanUseHipblasLtScale32x8(
                   *dot, stream_executor::GpuComputeCapability(
                             stream_executor::RocmComputeCapability("gfx950")))
                   .IsAllowed());
}

TEST_F(HipblasLtScaleUtilsTest, PreservesBytesAtHardwareScaleOffsets) {
  // Check multiple tiles in both dimensions, with and without a unit batch.
  // The expected address formula is independent of the HLO reshape/transpose
  // implementation and exercises all bytes, rather than using all-one scales.
  for (auto [rows, cols] :
       std::vector<std::pair<int64_t, int64_t>>{{32, 8}, {64, 16}, {96, 24}}) {
    for (bool batched : {false, true}) {
      std::string hlo = absl::Substitute(R"(
HloModule shuffle
ENTRY main {
  ROOT scale = f8e8m0fnu[$0$1,$2]{$3} parameter(0)
})",
                                         batched ? "1," : "", rows, cols,
                                         batched ? "2,1,0" : "1,0");
      auto module_or = ParseAndReturnVerifiedModule(hlo);
      ASSERT_THAT(module_or, absl_testing::IsOk());
      auto module = std::move(module_or).value();
      HloInstruction* scale = module->entry_computation()->root_instruction();
      Literal input(scale->shape());
      auto* src = static_cast<uint8_t*>(input.untyped_data());
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          src[r * cols + c] = (r * 19 + c * 73) % 255;
        }
      }
      auto fusion_or = SwizzleHipblasLtScale32x8(scale);
      ASSERT_THAT(fusion_or, absl_testing::IsOk());
      HloInstruction* fusion = std::move(fusion_or).value();
      EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kLoop);
      EXPECT_EQ(fusion->operand(0), scale);
      auto result_or = HloEvaluator().Evaluate(
          *fusion->fused_instructions_computation(), {&input});
      ASSERT_THAT(result_or, absl_testing::IsOk());
      Literal result = std::move(result_or).value();
      const auto* dst = static_cast<const uint8_t*>(result.untyped_data());
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          int64_t tile = (r / 32) * (cols / 8) + c / 8;
          int64_t offset = tile * 256 + (c % 4) * 64 + (r % 16) * 4 +
                           ((c % 8) / 4) * 2 + (r % 32) / 16;
          EXPECT_EQ(dst[offset], src[r * cols + c])
              << "r=" << r << " c=" << c << " rows=" << rows
              << " cols=" << cols;
        }
      }
    }
  }
}

TEST_F(HipblasLtScaleUtilsTest, MultipleScaleFusionsHaveUniqueNames) {
  auto module_or = ParseAndReturnVerifiedModule(R"(
HloModule two_scales
ENTRY main {
  a = f8e8m0fnu[64,16]{1,0} parameter(0)
  b = f8e8m0fnu[96,16]{1,0} parameter(1)
  ROOT scales = (f8e8m0fnu[64,16]{1,0}, f8e8m0fnu[96,16]{1,0}) tuple(a,b)
})");
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  HloInstruction* root = module->entry_computation()->root_instruction();
  auto a_or = SwizzleHipblasLtScale32x8(root->mutable_operand(0));
  ASSERT_THAT(a_or, absl_testing::IsOk());
  auto b_or = SwizzleHipblasLtScale32x8(root->mutable_operand(1));
  ASSERT_THAT(b_or, absl_testing::IsOk());
  EXPECT_NE((*a_or)->name(), (*b_or)->name());
  ASSERT_THAT(root->ReplaceOperandWith(0, *a_or), absl_testing::IsOk());
  ASSERT_THAT(root->ReplaceOperandWith(1, *b_or), absl_testing::IsOk());
  EXPECT_THAT(module->Verify(), absl_testing::IsOk());
}

}  // namespace
}  // namespace xla::gpu
