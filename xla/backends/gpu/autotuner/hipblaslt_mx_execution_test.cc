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

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"
#include "xla/error_spec.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using HipblasLtMxExecutionTest = HloPjRtTestBase;

// Shapes from PR #33947 (known to work with hipBLASLt MX on gfx950).
// K=256 with block_size=32 gives scale dim K/32=8.
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

// Numerical correctness: kScaledDot goes through the full pipeline with
// xla_gpu_experimental_scaled_dot_with_triton=true, entering GemmFusion and
// the autotuner. The reference decomposes it via ScaledDotRewriter.
TEST_F(HipblasLtMxExecutionTest, MxFp8Correctness) {
  TF_ASSERT_OK_AND_ASSIGN(auto reference_module,
                          ParseAndReturnUnverifiedModule(kMxFp8Hlo));
  ScaledDotRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto changed, rewriter.Run(reference_module.get(), {}));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnUnverifiedModule(kMxFp8Hlo));
  test_module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_scaled_dot_with_triton(true);
  test_module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_triton_gemm(true);

  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(test_module), std::move(reference_module),
      ErrorSpec(/*aabs=*/0.1, /*arel=*/0.1),
      /*run_hlo_passes=*/true));
}

TEST_F(HipblasLtMxExecutionTest, MxFp8BatchedCorrectness) {
  TF_ASSERT_OK_AND_ASSIGN(auto reference_module,
                          ParseAndReturnUnverifiedModule(kMxFp8BatchedHlo));
  ScaledDotRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto changed, rewriter.Run(reference_module.get(), {}));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnUnverifiedModule(kMxFp8BatchedHlo));
  test_module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_scaled_dot_with_triton(true);
  test_module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_triton_gemm(true);

  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(test_module), std::move(reference_module),
      ErrorSpec(/*aabs=*/0.1, /*arel=*/0.1),
      /*run_hlo_passes=*/true));
}

// Fallback correctness: default flags decompose kScaledDot via
// ScaledDotRewriter in the pipeline, matching our reference.
TEST_F(HipblasLtMxExecutionTest, DecompositionFallbackCorrectness) {
  TF_ASSERT_OK_AND_ASSIGN(auto reference_module,
                          ParseAndReturnUnverifiedModule(kMxFp8Hlo));
  ScaledDotRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto changed, rewriter.Run(reference_module.get(), {}));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto test_module,
                          ParseAndReturnUnverifiedModule(kMxFp8Hlo));

  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(test_module), std::move(reference_module),
      ErrorSpec(/*aabs=*/0.01, /*arel=*/0.01),
      /*run_hlo_passes=*/true));
}

}  // namespace
}  // namespace xla::gpu
