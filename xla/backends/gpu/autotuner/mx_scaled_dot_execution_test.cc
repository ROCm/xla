/* Copyright 2025 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/autotuner/hipblaslt_scale_utils.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class MxScaledDotExecutionTest : public HloPjRtGpuTestBase {
 protected:
  void SetUp() override {
    const auto& gpu_cc = device_description().gpu_compute_capability();
    const auto* rocm_cc = gpu_cc.rocm_compute_capability();
    if (rocm_cc == nullptr || !rocm_cc->has_mx_type_support()) {
      GTEST_SKIP() << "MX execution test requires MX type support (gfx950+).";
    }
  }

  // Runs numerical correctness and verifies that the optimized HLO uses the
  // expected custom call target.
  void RunMxCorrectnessTest(absl::string_view hlo_string,
                            const ErrorSpec& error_spec) {
    HloModuleConfig ref_config = GetModuleConfigForTest();
    ref_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(false);
    ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
    ASSERT_OK_AND_ASSIGN(auto ref_optimized,
                         GetOptimizedModule(hlo_string, ref_config));
    EXPECT_THAT(
        RunFileCheck(ref_optimized->ToString(),
                     R"(CHECK: {{__cublas\$lt\$matmul|__cublas\$gemm}})"),
        absl_testing::IsOkAndHolds(true));

    HloModuleConfig test_config = GetModuleConfigForTest();
    test_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(true);
    test_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(true);
    ASSERT_OK_AND_ASSIGN(auto test_optimized,
                         GetOptimizedModule(hlo_string, test_config));
    // The autotuner may pick any of the MX-aware ROCm backends:
    //   __triton_nested_gemm_fusion -> Triton
    //   __cublas$lt$matmul$mx       -> hipBLASLt
    //   __cublas$lt$matmul          -> HIPBLASLT_FISSION (should never win)
    EXPECT_THAT(
        RunFileCheck(
            test_optimized->ToString(),
            R"(CHECK: {{__triton_nested_gemm_fusion|__cublas\$lt\$matmul\$mx}})"),
        absl_testing::IsOkAndHolds(true));

    EXPECT_TRUE(RunAndCompareTwoModules(std::move(test_optimized),
                                        std::move(ref_optimized), error_spec,
                                        /*run_hlo_passes=*/false));
  }

  void RunPreswizzledCorrectnessTest(bool fp4);
};

constexpr absl::string_view kMxFp8Hlo = R"(
HloModule mx_test
ENTRY main {
  %lhs = f8e4m3fn[32,256] parameter(0)
  %rhs = f8e4m3fn[16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[16,8] parameter(3)
  ROOT %result = f32[32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

TEST_F(MxScaledDotExecutionTest, MxFp8Correctness) {
  RunMxCorrectnessTest(kMxFp8Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp8BatchedHlo = R"(
HloModule mx_batched_test
ENTRY main {
  %lhs = f8e4m3fn[1,32,256] parameter(0)
  %rhs = f8e4m3fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest, MxFp8BatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp8BatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp8MixedTypesBatchedHlo = R"(
HloModule mx_fp8_mixed_types_batched_test
ENTRY main {
  %lhs = f8e4m3fn[1,32,256] parameter(0)
  %rhs = f8e5m2[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest, MxFp8MixedTypesBatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp8MixedTypesBatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp4Hlo = R"(
HloModule mx_fp4_test
ENTRY main {
  %lhs = f4e2m1fn[32,256] parameter(0)
  %rhs = f4e2m1fn[16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[16,8] parameter(3)
  ROOT %result = f32[32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

TEST_F(MxScaledDotExecutionTest, MxFp4Correctness) {
  RunMxCorrectnessTest(kMxFp4Hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp4BatchedHlo = R"(
HloModule mx_fp4_batched_test
ENTRY main {
  %lhs = f4e2m1fn[1,32,256] parameter(0)
  %rhs = f4e2m1fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest, MxFp4BatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp4BatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

constexpr absl::string_view kMxFp4Fp8MixedBatchedHlo = R"(
HloModule mx_fp4_fp8_mixed_batched_test
ENTRY main {
  %lhs = f4e2m1fn[1,32,256] parameter(0)
  %rhs = f8e4m3fn[1,16,256] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest, MxFp4Fp8MixedBatchedCorrectness) {
  RunMxCorrectnessTest(kMxFp4Fp8MixedBatchedHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

// The scaled-dot operands are reshapes of higher-rank parameters, mimicking the
// in-graph block-scaled quantization case where the fp8 operand is produced in
// [batch, M, K/block, block] form and only flattened to [batch, M, K] by a
// reshape/bitcast that gets absorbed into the gemm fusion. This exercises the
// path where the fusion's external operands have a different rank than the
// scaled-dot operands; the MX rewrite must re-bitcast them back so the custom
// call operands stay consistent with the dot dimension numbers.
constexpr absl::string_view kMxFp8ReshapedOperandsHlo = R"(
HloModule mx_fp8_reshaped_operands_test
ENTRY main {
  %lhs_blocked = f8e4m3fn[1,32,8,32] parameter(0)
  %rhs_blocked = f8e4m3fn[1,16,8,32] parameter(1)
  %lhs_scale = f8e8m0fnu[1,32,8] parameter(2)
  %rhs_scale = f8e8m0fnu[1,16,8] parameter(3)
  %lhs = f8e4m3fn[1,32,256] reshape(%lhs_blocked)
  %rhs = f8e4m3fn[1,16,256] reshape(%rhs_blocked)
  ROOT %result = f32[1,32,16] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest, MxFp8ReshapedOperandsCorrectness) {
  RunMxCorrectnessTest(kMxFp8ReshapedOperandsHlo,
                       ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5));
}

void MxScaledDotExecutionTest::RunPreswizzledCorrectnessTest(bool fp4) {
  const auto* rocm =
      device_description().gpu_compute_capability().rocm_compute_capability();
  if (rocm == nullptr || !rocm->gfx9_mi350()) {
    GTEST_SKIP() << "Pre-swizzled scale tests require gfx950";
  }
  for (bool batched : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "batched=" << batched);
    std::string hlo = absl::Substitute(
        R"(
HloModule preswizzled_scales
ENTRY main {
  a = $4[$064,512]{$1$5} parameter(0)
  b = $4[$096,512]{$1$5} parameter(1)
  sa = f8e8m0fnu[$064,16]{$1} parameter(2)
  sb = f8e8m0fnu[$096,16]{$1} parameter(3)
  ROOT dot = f32[$064,96]{$1} scaled-dot(a,b,sa,sb),
      lhs_contracting_dims={$2}, rhs_contracting_dims={$2}$3
})",
        batched ? "1," : "", batched ? "2,1,0" : "1,0", batched ? 2 : 1,
        batched ? ", lhs_batch_dims={0}, rhs_batch_dims={0}" : "",
        fp4 ? "f4e2m1fn" : "f8e4m3fn", fp4 ? ":E(4)" : "");
    HloModuleConfig ref_config = GetModuleConfigForTest();
    ref_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(false);
    ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
    auto reference_or = GetOptimizedModule(hlo, ref_config);
    ASSERT_THAT(reference_or, absl_testing::IsOk());
    auto reference = std::move(reference_or).value();

    HloModuleConfig packed_config = GetModuleConfigForTest();
    DebugOptions& options = packed_config.mutable_debug_options();
    options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
    options.set_xla_gpu_enable_triton_gemm(true);
    options.set_xla_gpu_blas_max_algorithms(4);
    options.set_xla_autotuner_preferred_backend(autotuner::HIPBLASLT);
    options.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
        DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8);
    auto packed_or = GetOptimizedModule(hlo, packed_config);
    ASSERT_THAT(packed_or, absl_testing::IsOk());
    auto packed = std::move(packed_or).value();
    // Follow live operands from the result: a marker in an unused computation
    // is not evidence that the executable actually selected packed hipBLASLt.
    const HloInstruction* mx_call = nullptr;
    std::vector<const HloInstruction*> pending{
        packed->entry_computation()->root_instruction()};
    while (!pending.empty()) {
      const HloInstruction* instruction = pending.back();
      pending.pop_back();
      if (instruction->opcode() == HloOpcode::kCustomCall &&
          instruction->custom_call_target() == "__cublas$lt$matmul$mx") {
        mx_call = instruction;
        break;
      }
      for (const HloInstruction* operand : instruction->operands()) {
        pending.push_back(operand);
      }
    }
    ASSERT_NE(mx_call, nullptr) << packed->ToString();
    auto config_or = mx_call->backend_config<GpuBackendConfig>();
    ASSERT_THAT(config_or, absl_testing::IsOk());
    auto config = std::move(config_or).value();
    ASSERT_EQ(config.gemm_backend_config().scale_mode(), 3);
    if (fp4) {
      for (int operand : {0, 1}) {
        const Shape& data_shape = mx_call->operand(operand)->shape();
        ASSERT_EQ(data_shape.element_type(), F4E2M1FN);
        ASSERT_EQ(data_shape.layout().element_size_in_bits(), 4);
      }
    }

    // Nonconstant, exactly representable data and power-of-two scales. Change
    // scales along both MN and K/32 so an incorrect swizzle cannot pass as it
    // would with all-one scales. The inputs are parameters, not HLO constants.
    std::vector<Literal> inputs;
    inputs.reserve(4);
    for (int operand = 0; operand < 4; ++operand) {
      const Shape& shape =
          packed->entry_computation()->parameter_instruction(operand)->shape();
      inputs.emplace_back(shape);
      auto* bytes = static_cast<uint8_t*>(inputs.back().untyped_data());
      int64_t rank = shape.dimensions().size();
      int64_t rows = shape.dimensions(rank - 2);
      int64_t cols = shape.dimensions(rank - 1);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          if (operand < 2 && fp4) {
            // Host literals store one FP4 value per byte; the transfer manager
            // packs adjacent values into low/high nibbles on the GPU. Cover all
            // 16 encodings with distinct adjacent values and varying K blocks.
            bytes[r * cols + c] =
                (r * 3 + c * 5 + (c / 32) * 7 + operand * 11) % 16;
          } else if (operand < 2) {
            bytes[r * cols + c] = (0x28 + (r * 17 + c * 11 + operand) % 17) |
                                  (((r + c) & 1) << 7);
          } else {
            bytes[r * cols + c] =
                125 + (r * 7 + c * 11 + operand) % (fp4 ? 3 : 5);
          }
        }
      }
    }
    EXPECT_TRUE(RunAndCompareTwoModules(
        std::move(packed), std::move(reference),
        {&inputs[0], &inputs[1], &inputs[2], &inputs[3]},
        ErrorSpec(/*aabs=*/1e-3, /*arel=*/1e-5), /*run_hlo_passes=*/false));
  }
}

TEST_F(MxScaledDotExecutionTest, PreswizzledMxFp8MatchesDequantizedReference) {
  RunPreswizzledCorrectnessTest(/*fp4=*/false);
}

TEST_F(MxScaledDotExecutionTest, PreswizzledMxFp4MatchesDequantizedReference) {
  RunPreswizzledCorrectnessTest(/*fp4=*/true);
}

// A residual add after the scaled-dot, which GemmFusionV2 fuses into the
// scaled-dot fusion. hipBLASLt runs the GEMM and a loop fusion runs the add.
constexpr absl::string_view kMxFp8ResidualHlo = R"(
HloModule mx_fp8_residual_test
ENTRY main {
  a = f8e4m3fn[1,64,512]{2,1,0} parameter(0)
  b = f8e4m3fn[1,96,512]{2,1,0} parameter(1)
  sa = f8e8m0fnu[1,64,16]{2,1,0} parameter(2)
  sb = f8e8m0fnu[1,96,16]{2,1,0} parameter(3)
  residual = bf16[1,64,96]{2,1,0} parameter(4)
  dot = bf16[1,64,96]{2,1,0} scaled-dot(a, b, sa, sb),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
  ROOT sum = bf16[1,64,96]{2,1,0} add(dot, residual)
})";

TEST_F(MxScaledDotExecutionTest,
       MxFp8ResidualEpilogueMatchesDequantizedReference) {
  const auto* rocm =
      device_description().gpu_compute_capability().rocm_compute_capability();
  if (rocm == nullptr || !rocm->gfx9_mi350()) {
    GTEST_SKIP() << "Pre-swizzled scale tests require gfx950";
  }
  // {scale layout policy, expected scale_mode}
  for (auto [layout, scale_mode] :
       {std::pair{DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_LINEAR, 2},
        std::pair{DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8,
                  3}}) {
    SCOPED_TRACE(::testing::Message() << "layout=" << layout);
    HloModuleConfig ref_config = GetModuleConfigForTest();
    ref_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(false);
    ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
    auto reference_or = GetOptimizedModule(kMxFp8ResidualHlo, ref_config);
    ASSERT_THAT(reference_or, absl_testing::IsOk());
    auto reference = std::move(reference_or).value();

    HloModuleConfig hipblaslt_config = GetModuleConfigForTest();
    DebugOptions& options = hipblaslt_config.mutable_debug_options();
    options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
    options.set_xla_gpu_enable_triton_gemm(true);
    options.set_xla_gpu_blas_max_algorithms(4);
    options.set_xla_autotuner_preferred_backend(autotuner::HIPBLASLT);
    options.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(layout);
    auto hipblaslt_or = GetOptimizedModule(kMxFp8ResidualHlo, hipblaslt_config);
    ASSERT_THAT(hipblaslt_or, absl_testing::IsOk());
    auto hipblaslt = std::move(hipblaslt_or).value();
    // Follow live operands from the result to the MX call.
    const HloInstruction* mx_call = nullptr;
    std::vector<const HloInstruction*> pending{
        hipblaslt->entry_computation()->root_instruction()};
    while (!pending.empty()) {
      const HloInstruction* instruction = pending.back();
      pending.pop_back();
      if (instruction->opcode() == HloOpcode::kCustomCall &&
          instruction->custom_call_target() == "__cublas$lt$matmul$mx") {
        mx_call = instruction;
        break;
      }
      for (const HloInstruction* operand : instruction->operands()) {
        pending.push_back(operand);
      }
    }
    ASSERT_NE(mx_call, nullptr) << hipblaslt->ToString();
    auto config_or = mx_call->backend_config<GpuBackendConfig>();
    ASSERT_THAT(config_or, absl_testing::IsOk());
    auto config = std::move(config_or).value();
    ASSERT_EQ(config.gemm_backend_config().scale_mode(), scale_mode);
    // The add was fused into the scaled-dot fusion and split off by the
    // hipBLASLt lowering, so the GEMM result feeds its epilogue fusion.
    const HloInstruction* epilogue = nullptr;
    for (const HloInstruction* result : mx_call->users()) {
      for (const HloInstruction* user : result->users()) {
        if (user->opcode() == HloOpcode::kFusion &&
            absl::StrContains(user->name(), "_epilogue")) {
          epilogue = user;
        }
      }
    }
    ASSERT_NE(epilogue, nullptr) << hipblaslt->ToString();

    // Exactly representable data and power-of-two scales as above, and a
    // residual in [-1, 1].
    std::vector<Literal> inputs;
    inputs.reserve(5);
    for (int operand = 0; operand < 4; ++operand) {
      const Shape& shape = hipblaslt->entry_computation()
                               ->parameter_instruction(operand)
                               ->shape();
      inputs.emplace_back(shape);
      auto* bytes = static_cast<uint8_t*>(inputs.back().untyped_data());
      int64_t rank = shape.dimensions().size();
      int64_t rows = shape.dimensions(rank - 2);
      int64_t cols = shape.dimensions(rank - 1);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          if (operand < 2) {
            bytes[r * cols + c] = (0x28 + (r * 17 + c * 11 + operand) % 17) |
                                  (((r + c) & 1) << 7);
          } else {
            bytes[r * cols + c] = 125 + (r * 7 + c * 11 + operand) % 5;
          }
        }
      }
    }
    inputs.emplace_back(
        hipblaslt->entry_computation()->parameter_instruction(4)->shape());
    ASSERT_THAT(
        inputs.back().Populate<bfloat16>([](absl::Span<const int64_t> index) {
          return bfloat16(
              static_cast<float>((index[1] * 5 + index[2] * 3) % 9 - 4) / 4.0f);
        }),
        absl_testing::IsOk());
    // The GEMM result is rounded to BF16 before the add on both paths.
    EXPECT_TRUE(RunAndCompareTwoModules(
        std::move(hipblaslt), std::move(reference),
        {&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]},
        ErrorSpec(/*aabs=*/1e-2, /*arel=*/1e-2), /*run_hlo_passes=*/false));
  }
}

// An RHS that arrives with K outermost, as a weight all-gathered along its
// contracting dimension does under FSDP. GemmFusionV2 fuses the transpose to
// the contracting-minor operand into the scaled-dot fusion, and the hipBLASLt
// lowering moves it to a loop fusion before the GEMM.
constexpr absl::string_view kMxFp8TransposedRhsHlo = R"(
HloModule mx_fp8_transposed_rhs_test
ENTRY main {
  a = f8e4m3fn[1,64,512]{2,1,0} parameter(0)
  b = f8e4m3fn[512,1,96]{2,1,0} parameter(1)
  sa = f8e8m0fnu[1,64,16]{2,1,0} parameter(2)
  sb = f8e8m0fnu[1,96,16]{2,1,0} parameter(3)
  b_t = f8e4m3fn[1,96,512]{2,1,0} transpose(b), dimensions={1,2,0}
  ROOT dot = f32[1,64,96]{2,1,0} scaled-dot(a, b_t, sa, sb),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
})";

TEST_F(MxScaledDotExecutionTest,
       MxFp8TransposedRhsMatchesDequantizedReference) {
  const auto* rocm =
      device_description().gpu_compute_capability().rocm_compute_capability();
  if (rocm == nullptr || !rocm->gfx9_mi350()) {
    GTEST_SKIP() << "Pre-swizzled scale tests require gfx950";
  }
  // {scale layout policy, expected scale_mode}
  for (auto [layout, scale_mode] :
       {std::pair{DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_LINEAR, 2},
        std::pair{DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8,
                  3}}) {
    SCOPED_TRACE(::testing::Message() << "layout=" << layout);
    HloModuleConfig ref_config = GetModuleConfigForTest();
    ref_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(false);
    ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
    auto reference_or = GetOptimizedModule(kMxFp8TransposedRhsHlo, ref_config);
    ASSERT_THAT(reference_or, absl_testing::IsOk());
    auto reference = std::move(reference_or).value();

    HloModuleConfig hipblaslt_config = GetModuleConfigForTest();
    DebugOptions& options = hipblaslt_config.mutable_debug_options();
    options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
    options.set_xla_gpu_enable_triton_gemm(true);
    options.set_xla_gpu_blas_max_algorithms(4);
    options.set_xla_autotuner_preferred_backend(autotuner::HIPBLASLT);
    options.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(layout);
    auto hipblaslt_or =
        GetOptimizedModule(kMxFp8TransposedRhsHlo, hipblaslt_config);
    ASSERT_THAT(hipblaslt_or, absl_testing::IsOk());
    auto hipblaslt = std::move(hipblaslt_or).value();
    // Follow live operands from the result to the MX call.
    const HloInstruction* mx_call = nullptr;
    std::vector<const HloInstruction*> pending{
        hipblaslt->entry_computation()->root_instruction()};
    while (!pending.empty()) {
      const HloInstruction* instruction = pending.back();
      pending.pop_back();
      if (instruction->opcode() == HloOpcode::kCustomCall &&
          instruction->custom_call_target() == "__cublas$lt$matmul$mx") {
        mx_call = instruction;
        break;
      }
      for (const HloInstruction* operand : instruction->operands()) {
        pending.push_back(operand);
      }
    }
    ASSERT_NE(mx_call, nullptr) << hipblaslt->ToString();
    auto config_or = mx_call->backend_config<GpuBackendConfig>();
    ASSERT_THAT(config_or, absl_testing::IsOk());
    auto config = std::move(config_or).value();
    ASSERT_EQ(config.gemm_backend_config().scale_mode(), scale_mode);
    // The transpose was fused into the scaled-dot fusion and split off by the
    // hipBLASLt lowering, so a relayout fusion computes the RHS of the GEMM.
    const HloInstruction* rhs = mx_call->operand(1);
    ASSERT_EQ(rhs->opcode(), HloOpcode::kFusion) << hipblaslt->ToString();
    EXPECT_TRUE(absl::StrContains(rhs->name(), "_relayout"))
        << hipblaslt->ToString();
    int num_transposes = 0;
    for (const HloInstruction* instruction :
         rhs->fused_instructions_computation()->instructions()) {
      num_transposes += instruction->opcode() == HloOpcode::kTranspose;
    }
    EXPECT_EQ(num_transposes, 1) << hipblaslt->ToString();

    // Exactly representable data and power-of-two scales that change along
    // both MN and K/32, as in RunPreswizzledCorrectnessTest, indexed by the
    // logical [MN, K] position so that the product signs are constant along
    // K. `b` is stored as [K, 1, N].
    auto value = [](int operand, int64_t mn, int64_t k) -> uint8_t {
      if (operand < 2) {
        return static_cast<uint8_t>((0x28 + (mn * 17 + k * 11 + operand) % 17) |
                                    (((mn + k) & 1) << 7));
      }
      return static_cast<uint8_t>(125 + (mn * 7 + k * 11 + operand) % 5);
    };
    std::vector<Literal> inputs;
    inputs.reserve(4);
    for (int operand = 0; operand < 4; ++operand) {
      const Shape& shape = hipblaslt->entry_computation()
                               ->parameter_instruction(operand)
                               ->shape();
      inputs.emplace_back(shape);
      auto* bytes = static_cast<uint8_t*>(inputs.back().untyped_data());
      if (operand == 1) {
        for (int64_t k = 0; k < 512; ++k) {
          for (int64_t n = 0; n < 96; ++n) {
            bytes[k * 96 + n] = value(operand, n, k);
          }
        }
        continue;
      }
      int64_t rows = shape.dimensions(1);
      int64_t cols = shape.dimensions(2);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          bytes[r * cols + c] = value(operand, r, c);
        }
      }
    }
    EXPECT_TRUE(RunAndCompareTwoModules(
        std::move(hipblaslt), std::move(reference),
        {&inputs[0], &inputs[1], &inputs[2], &inputs[3]},
        ErrorSpec(/*aabs=*/1e-3, /*arel=*/1e-5), /*run_hlo_passes=*/false));
  }
}

// Two scaled-dots that share their LHS data and scale, as the q, k and v
// projections share their normalized input.
constexpr absl::string_view kMxFp8SharedScaleHlo = R"(
HloModule mx_fp8_shared_scale_test
ENTRY main {
  a = f8e4m3fn[64,512]{1,0} parameter(0)
  b = f8e4m3fn[96,512]{1,0} parameter(1)
  c = f8e4m3fn[128,512]{1,0} parameter(2)
  sa = f8e8m0fnu[64,16]{1,0} parameter(3)
  sb = f8e8m0fnu[96,16]{1,0} parameter(4)
  sc = f8e8m0fnu[128,16]{1,0} parameter(5)
  dot0 = f32[64,96]{1,0} scaled-dot(a, b, sa, sb),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  dot1 = f32[64,128]{1,0} scaled-dot(a, c, sa, sc),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  ROOT results = (f32[64,96]{1,0}, f32[64,128]{1,0}) tuple(dot0, dot1)
})";

TEST_F(MxScaledDotExecutionTest,
       SharedScaleSwizzledOnceMatchesDequantizedReference) {
  const auto* rocm =
      device_description().gpu_compute_capability().rocm_compute_capability();
  if (rocm == nullptr || !rocm->gfx9_mi350()) {
    GTEST_SKIP() << "Pre-swizzled scale tests require gfx950";
  }
  HloModuleConfig ref_config = GetModuleConfigForTest();
  ref_config.mutable_debug_options()
      .set_xla_gpu_experimental_scaled_dot_with_triton(false);
  ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
  auto reference_or = GetOptimizedModule(kMxFp8SharedScaleHlo, ref_config);
  ASSERT_THAT(reference_or, absl_testing::IsOk());
  auto reference = std::move(reference_or).value();

  HloModuleConfig packed_config = GetModuleConfigForTest();
  DebugOptions& options = packed_config.mutable_debug_options();
  options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
  options.set_xla_gpu_enable_triton_gemm(true);
  options.set_xla_gpu_blas_max_algorithms(4);
  options.set_xla_autotuner_preferred_backend(autotuner::HIPBLASLT);
  options.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8);
  auto packed_or = GetOptimizedModule(kMxFp8SharedScaleHlo, packed_config);
  ASSERT_THAT(packed_or, absl_testing::IsOk());
  auto packed = std::move(packed_or).value();

  // Both GEMMs run as packed hipBLASLt calls, and one swizzle fusion of `sa`
  // feeds both of them.
  const HloComputation* entry = packed->entry_computation();
  std::vector<const HloInstruction*> mx_calls;
  int swizzles = 0;
  for (const HloInstruction* instruction : entry->instructions()) {
    if (instruction->opcode() == HloOpcode::kCustomCall &&
        instruction->custom_call_target() == "__cublas$lt$matmul$mx") {
      mx_calls.push_back(instruction);
    }
    if (IsHipblasLtScale32x8Swizzle(*instruction)) {
      ++swizzles;
    }
  }
  ASSERT_THAT(mx_calls, ::testing::SizeIs(2)) << packed->ToString();
  for (const HloInstruction* mx_call : mx_calls) {
    auto config_or = mx_call->backend_config<GpuBackendConfig>();
    ASSERT_THAT(config_or, absl_testing::IsOk());
    ASSERT_EQ(config_or->gemm_backend_config().scale_mode(), 3);
  }
  const HloInstruction* shared_swizzle = mx_calls[0]->operand(2);
  EXPECT_EQ(mx_calls[1]->operand(2), shared_swizzle) << packed->ToString();
  EXPECT_TRUE(IsHipblasLtScale32x8Swizzle(*shared_swizzle))
      << packed->ToString();
  EXPECT_EQ(shared_swizzle->operand(0), entry->parameter_instruction(3))
      << packed->ToString();
  EXPECT_EQ(swizzles, 3) << packed->ToString();

  // Exactly representable data and power-of-two scales that change along both
  // MN and K/32, as in RunPreswizzledCorrectnessTest.
  std::vector<Literal> inputs;
  inputs.reserve(6);
  for (int operand = 0; operand < 6; ++operand) {
    const Shape& shape = entry->parameter_instruction(operand)->shape();
    inputs.emplace_back(shape);
    auto* bytes = static_cast<uint8_t*>(inputs.back().untyped_data());
    int64_t rows = shape.dimensions(0);
    int64_t cols = shape.dimensions(1);
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        if (operand < 3) {
          bytes[r * cols + c] =
              (0x28 + (r * 17 + c * 11 + operand) % 17) | (((r + c) & 1) << 7);
        } else {
          bytes[r * cols + c] = 125 + (r * 7 + c * 11 + operand) % 5;
        }
      }
    }
  }
  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(packed), std::move(reference),
      {&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5]},
      ErrorSpec(/*aabs=*/1e-3, /*arel=*/1e-5), /*run_hlo_passes=*/false));
}

// A unit-batch scaled-dot whose result is transposed, as the weight gradient of
// a scan layer is, or not. Layout assignment makes the transpose a bitcast by
// giving the scaled-dot a column-major output, which hipBLASLt computes without
// swapping its operands; without the transpose the output is row-major.
std::string MxUnitBatchHlo(bool fp4, bool transposed) {
  return absl::Substitute(
      R"(
HloModule mx_unit_batch
ENTRY main {
  a = $0[1,64,512]{2,1,0$1} parameter(0)
  b = $0[1,96,512]{2,1,0$1} parameter(1)
  sa = f8e8m0fnu[1,64,16]{2,1,0} parameter(2)
  sb = f8e8m0fnu[1,96,16]{2,1,0} parameter(3)
  dot = bf16[1,64,96]{2,1,0} scaled-dot(a, b, sa, sb),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
  $2
})",
      fp4 ? "f4e2m1fn" : "f8e4m3fn", fp4 ? ":E(4)" : "",
      transposed ? "r = bf16[64,96]{1,0} reshape(dot)\n"
                   "  ROOT t = bf16[96,64]{1,0} transpose(r), dimensions={1,0}"
                 : "ROOT r = bf16[64,96]{1,0} reshape(dot)");
}

// Returns the hipBLASLt MX call that the result of `module` depends on, or
// nullptr.
const HloInstruction* FindMxCall(const HloModule& module) {
  std::vector<const HloInstruction*> pending{
      module.entry_computation()->root_instruction()};
  while (!pending.empty()) {
    const HloInstruction* instruction = pending.back();
    pending.pop_back();
    if (instruction->opcode() == HloOpcode::kCustomCall &&
        instruction->custom_call_target() == "__cublas$lt$matmul$mx") {
      return instruction;
    }
    for (const HloInstruction* operand : instruction->operands()) {
      pending.push_back(operand);
    }
  }
  return nullptr;
}

// The position of a BF16 value in the order of all BF16 values, so that the
// difference of two positions is their distance in ulps.
int32_t Bf16Ordinal(bfloat16 value) {
  uint16_t bits = absl::bit_cast<uint16_t>(value);
  int32_t magnitude = bits & 0x7fff;
  return (bits & 0x8000) != 0 ? -magnitude : magnitude;
}

TEST_F(MxScaledDotExecutionTest,
       ColumnMajorOutputMatchesRowMajorAndDequantizedReference) {
  const auto* rocm =
      device_description().gpu_compute_capability().rocm_compute_capability();
  if (rocm == nullptr || !rocm->gfx9_mi350()) {
    GTEST_SKIP() << "Pre-swizzled scale tests require gfx950";
  }
  // {data type, FP8 data with exponents from 2^-5 to 2^6 and no sign}. Without
  // it, the data and the scales are those of RunPreswizzledCorrectnessTest,
  // whose products sum exactly in F32. The wide positive data rounds in F32,
  // and with no cancellation two orders of the sums round to BF16 values at
  // most one ulp apart.
  for (auto [fp4, wide] : {std::pair{false, false}, std::pair{false, true},
                           std::pair{true, false}}) {
    // {scale layout policy, expected scale_mode}
    for (auto [layout, scale_mode] :
         {std::pair{DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_LINEAR, 2},
          std::pair{DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8,
                    3}}) {
      SCOPED_TRACE(::testing::Message() << "fp4=" << fp4 << " wide=" << wide
                                        << " layout=" << layout);
      const std::string column_hlo = MxUnitBatchHlo(fp4, /*transposed=*/true);
      HloModuleConfig ref_config = GetModuleConfigForTest();
      ref_config.mutable_debug_options()
          .set_xla_gpu_experimental_scaled_dot_with_triton(false);
      ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
      auto reference_or = GetOptimizedModule(column_hlo, ref_config);
      ASSERT_THAT(reference_or, absl_testing::IsOk());
      auto reference = std::move(reference_or).value();

      HloModuleConfig hipblaslt_config = GetModuleConfigForTest();
      DebugOptions& options = hipblaslt_config.mutable_debug_options();
      options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
      options.set_xla_gpu_enable_triton_gemm(true);
      options.set_xla_gpu_blas_max_algorithms(4);
      options.set_xla_autotuner_preferred_backend(autotuner::HIPBLASLT);
      options.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(layout);
      auto column_or = GetOptimizedModule(column_hlo, hipblaslt_config);
      ASSERT_THAT(column_or, absl_testing::IsOk());
      auto column = std::move(column_or).value();
      auto row_or = GetOptimizedModule(MxUnitBatchHlo(fp4, /*transposed=*/false),
                                       hipblaslt_config);
      ASSERT_THAT(row_or, absl_testing::IsOk());
      auto row = std::move(row_or).value();
      // {module, expected minor_to_major of the GEMM result}
      for (auto [module, minor_to_major] :
           {std::pair{column.get(), std::vector<int64_t>{1, 0, 2}},
            std::pair{row.get(), std::vector<int64_t>{2, 1, 0}}}) {
        const HloInstruction* mx_call = FindMxCall(*module);
        ASSERT_NE(mx_call, nullptr) << module->ToString();
        auto config_or = mx_call->backend_config<GpuBackendConfig>();
        ASSERT_THAT(config_or, absl_testing::IsOk());
        ASSERT_EQ(config_or->gemm_backend_config().scale_mode(), scale_mode);
        EXPECT_THAT(mx_call->shape().tuple_shapes(0).layout().minor_to_major(),
                    ::testing::ElementsAreArray(minor_to_major))
            << module->ToString();
      }

      std::vector<Literal> inputs;
      inputs.reserve(4);
      for (int operand = 0; operand < 4; ++operand) {
        const Shape& shape =
            column->entry_computation()->parameter_instruction(operand)->shape();
        inputs.emplace_back(shape);
        auto* bytes = static_cast<uint8_t*>(inputs.back().untyped_data());
        int64_t rows = shape.dimensions(1);
        int64_t cols = shape.dimensions(2);
        for (int64_t r = 0; r < rows; ++r) {
          for (int64_t c = 0; c < cols; ++c) {
            uint8_t& byte = bytes[r * cols + c];
            if (operand < 2 && fp4) {
              // One FP4 value per byte of a host literal, as above.
              byte = (r * 3 + c * 5 + (c / 32) * 7 + operand * 11) % 16;
            } else if (operand < 2 && wide) {
              byte = 0x10 + (r * 29 + c * 13 + operand * 7) % 0x60;
            } else if (operand < 2) {
              byte = (0x28 + (r * 17 + c * 11 + operand) % 17) |
                     (((r + c) & 1) << 7);
            } else {
              byte = 125 + (r * 7 + c * 11 + operand) % (fp4 ? 3 : 5);
            }
          }
        }
      }
      std::vector<const Literal*> arguments;
      for (const Literal& input : inputs) {
        arguments.push_back(&input);
      }
      // The tolerance of the BF16 residual test above.
      EXPECT_TRUE(RunAndCompareTwoModules(column->Clone(), std::move(reference),
                                          arguments,
                                          ErrorSpec(/*aabs=*/1e-2,
                                                    /*arel=*/1e-2),
                                          /*run_hlo_passes=*/false));

      auto column_result_or =
          Execute(std::move(column), arguments, /*run_hlo_passes=*/false);
      ASSERT_THAT(column_result_or, absl_testing::IsOk());
      auto row_result_or =
          Execute(std::move(row), arguments, /*run_hlo_passes=*/false);
      ASSERT_THAT(row_result_or, absl_testing::IsOk());
      int64_t num_different = 0;
      int32_t max_ulps = 0;
      for (int64_t m = 0; m < 64; ++m) {
        for (int64_t n = 0; n < 96; ++n) {
          int32_t ulps =
              std::abs(Bf16Ordinal(column_result_or->Get<bfloat16>({n, m})) -
                       Bf16Ordinal(row_result_or->Get<bfloat16>({m, n})));
          num_different += ulps != 0;
          max_ulps = std::max(max_ulps, ulps);
        }
      }
      LOG(INFO) << "fp4=" << fp4 << " wide=" << wide << " layout=" << layout
                << ": " << num_different
                << " of 6144 elements differ between the column-major and the "
                   "row-major result, by at most "
                << max_ulps << " BF16 ulps";
      if (wide) {
        EXPECT_LE(max_ulps, 1);
      } else {
        EXPECT_EQ(num_different, 0);
      }
    }
  }
}

}  // namespace
}  // namespace xla::gpu
