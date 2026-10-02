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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
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
    TF_ASSERT_OK_AND_ASSIGN(auto ref_optimized,
                            GetOptimizedModule(hlo_string, ref_config));
    EXPECT_THAT(
        RunFileCheck(ref_optimized->ToString(),
                     R"(CHECK: {{__cublas\$lt\$matmul|__cublas\$gemm}})"),
        absl_testing::IsOkAndHolds(true));

    HloModuleConfig test_config = GetModuleConfigForTest();
    test_config.mutable_debug_options()
        .set_xla_gpu_experimental_scaled_dot_with_triton(true);
    test_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(true);
    TF_ASSERT_OK_AND_ASSIGN(auto test_optimized,
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

}  // namespace
}  // namespace xla::gpu
