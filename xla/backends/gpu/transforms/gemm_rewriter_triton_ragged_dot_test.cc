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

// End-to-end tests for the experimental Triton XTile backend for kRaggedDot.
//
// Each test verifies two things:
//   1. HLO transformation: GemmRewriter wraps the ragged-dot in a
//      kTritonFusionKind ("__triton") fusion (always checked).
//   2. Numerical correctness via RunAndCompare (skipped on pre-Ampere CUDA;
//      always runs on ROCm).
//
// Activation flags:
//   --xla_gpu_experimental_triton_ragged_dot=true
//   --xla_gpu_experimental_enable_tiling_propagation=true

#include <memory>

#include <gtest/gtest.h>
#include "xla/backends/gpu/transforms/gemm_rewriter_test_lib.h"
#include "xla/error_spec.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class TritonRaggedDotTest
    : public HloPjRtInterpreterReferenceMixin<GemmRewriteTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_triton_ragged_dot(true);
    debug_options.set_xla_gpu_experimental_enable_tiling_propagation(true);
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }

  // True if the device supports Triton: Ampere+ on CUDA, always on ROCm.
  bool SupportsTriton() const {
    if (IsCuda()) {
      auto* cc = Capability().cuda_compute_capability();
      return cc != nullptr && cc->IsAtLeastAmpere();
    }
    return true;  // ROCm always supports Triton.
  }

  // Checks the HLO transformation and, when the device supports Triton,
  // also validates numerical correctness.
  void CheckHloAndMaybeRun(const char* hlo_text,
                           const ErrorSpec& error_spec = ErrorSpec{1e-4,
                                                                   1e-4}) {
    // 1. Always check the IR transformation.
    MatchOptimizedHlo(hlo_text, R"(
      ; CHECK-NOT: groupedMatmul
      ; CHECK:     kind=kCustom
      ; CHECK-SAME: "__triton"
    )");

    // 2. Numerical check when Triton is available on the device.
    if (SupportsTriton()) {
      EXPECT_TRUE(RunAndCompare(hlo_text, error_spec));
    } else {
      GTEST_SKIP() << "Triton not available on this device (pre-Ampere CUDA).";
    }
  }
};

// ============================================================================
// Tests
// ============================================================================

// Large non-contracting dimensions — triggers M/N schedule swap (M_avg < N).
// G=8, M=256, K=128, N=64.  Group sizes sum to 256 and are all positive.
// (RunAndCompare generates random lhs/rhs; gs is constant so it stays valid.)
TEST_F(TritonRaggedDotTest, LargeNonContracting) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotLarge

ENTRY main {
  lhs = f32[256,128] parameter(0)
  rhs = f32[8,128,64] parameter(1)
  gs  = s32[8] constant({24, 40, 28, 36, 32, 38, 30, 28})
  ROOT rd = f32[256,64] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text);
}

// Constant group_sizes tensor.
TEST_F(TritonRaggedDotTest, ConstantGroupSizes) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotConstGS

ENTRY main {
  lhs = f32[64,9] parameter(0)
  rhs = f32[2,9,8] parameter(1)
  gs  = s32[2] constant({16, 48})
  ROOT rd = f32[64,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text);
}

// Balanced groups — all equal size.
TEST_F(TritonRaggedDotTest, BalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBalanced

ENTRY main {
  lhs = f32[128,32] parameter(0)
  rhs = f32[4,32,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = f32[128,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text);
}

// Unbalanced groups — exercises the G-loop with variable-size groups.
TEST_F(TritonRaggedDotTest, UnbalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotUnbalanced

ENTRY main {
  lhs = f32[64,16] parameter(0)
  rhs = f32[2,16,8] parameter(1)
  gs  = s32[2] constant({4, 60})
  ROOT rd = f32[64,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text);
}

// Group sizes not multiples of BLOCK_M=32 — exercises M-boundary masking.
TEST_F(TritonRaggedDotTest, NonMultipleBlockSize) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotNonMultiple

ENTRY main {
  lhs = f32[96,32] parameter(0)
  rhs = f32[3,32,8] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = f32[96,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// Batched kRaggedNonContracting: LHS (B,M,K), RHS (B,G,K,N), gs (B,G).
// B=2, M=96, K=32, G=3, N=32.  Balanced groups (all size 32=BLOCK_M).
// gs has shape [B,G]: each batch element has its own group_sizes row.
// M_avg=32 >= N=32 → no M/N swap, PriorityFusion keeps [1,32,32] tiles.
TEST_F(TritonRaggedDotTest, BatchedBalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchedBalanced

ENTRY main {
  lhs = f32[2,96,32] parameter(0)
  rhs = f32[2,3,32,32] parameter(1)
  gs  = s32[2,3] constant({{32, 32, 32}, {32, 32, 32}})
  ROOT rd = f32[2,96,32] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2},
      lhs_ragged_dims={1}, rhs_group_dims={1}
}
)";
  CheckHloAndMaybeRun(hlo_text);
}

// Batched kRaggedNonContracting with unbalanced groups.
// B=3, M=96, K=32, G=3, N=32.  Non-uniform groups (10+30+56=96).
// gs has shape [B,G]: same group_sizes for all batches.
// M_avg=32 >= N=32 → no M/N swap.
TEST_F(TritonRaggedDotTest, BatchedUnbalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchedUnbalanced

ENTRY main {
  lhs = f32[3,96,32] parameter(0)
  rhs = f32[3,3,32,32] parameter(1)
  gs  = s32[3,3] constant({{10, 30, 56}, {10, 30, 56}, {10, 30, 56}})
  ROOT rd = f32[3,96,32] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2},
      lhs_ragged_dims={1}, rhs_group_dims={1}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// Four groups with a larger K dimension.
TEST_F(TritonRaggedDotTest, FourGroupsLargerK) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotFourGroups

ENTRY main {
  lhs = f32[128,64] parameter(0)
  rhs = f32[4,64,32] parameter(1)
  gs  = s32[4] constant({8, 40, 48, 32})
  ROOT rd = f32[128,32] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// ============================================================================
// Dtype tests: fp16, bf16, fp8
// ============================================================================

// F16 balanced groups — same shapes as BalancedGroups but with f16 operands.
// f16 accumulates into f16 output; exercises mixed-precision path in Triton.
TEST_F(TritonRaggedDotTest, Fp16BalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotFp16Balanced

ENTRY main {
  lhs = f16[128,32] parameter(0)
  rhs = f16[4,32,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = f16[128,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// F16 unbalanced groups — exercises G-loop with variable-size groups in fp16.
TEST_F(TritonRaggedDotTest, Fp16UnbalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotFp16Unbalanced

ENTRY main {
  lhs = f16[64,16] parameter(0)
  rhs = f16[2,16,8] parameter(1)
  gs  = s32[2] constant({4, 60})
  ROOT rd = f16[64,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// BF16 balanced groups — same shapes as BalancedGroups but with bf16 operands.
// bf16 has ~3 significant decimal digits; error tolerance is wider than f32.
TEST_F(TritonRaggedDotTest, Bf16BalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBf16Balanced

ENTRY main {
  lhs = bf16[128,32] parameter(0)
  rhs = bf16[4,32,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = bf16[128,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// BF16 unbalanced groups.
TEST_F(TritonRaggedDotTest, Bf16UnbalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBf16Unbalanced

ENTRY main {
  lhs = bf16[96,32] parameter(0)
  rhs = bf16[3,32,8] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = bf16[96,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// FP8 (E4M3FNUZ) balanced groups with F32 output.
// fp8 inputs accumulate into f32 (the Triton emitter always uses f32
// accumulator, then casts to the declared output type if different).
// Note: fp8 ragged-dot support requires hardware fp8 capability; this test
// exercises the HLO rewrite path unconditionally and the numerical path
// only when Triton is available and the backend handles fp8 dot.
TEST_F(TritonRaggedDotTest, Fp8BalancedGroupsF32Output) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotFp8Balanced

ENTRY main {
  lhs = f8e4m3fnuz[64,32] parameter(0)
  rhs = f8e4m3fnuz[4,32,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f32[64,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-1, 1e-1});
}

// ============================================================================
// Layout tests (analogous to GroupedGemmRewriteTest layout tests)
// ============================================================================

// Column-major output layout.  Analogous to the GroupedGemmRewriteTest
// `CustomCallTargetGroupedGemmMulipleGroupsOutputColumnMajor` test.
// LHS f16[64,9]{1,0} (row-major), RHS f16[4,9,8]{2,1,0} (standard 3-D
// row-major: N fastest, then K, then G), output f16[64,8]{0,1} (column-major:
// M is physically innermost, N is outermost).
// s64 group_sizes: exercises the 64-bit group-size path.
TEST_F(TritonRaggedDotTest, OutputColumnMajor) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotColumnMajorOut

ENTRY main {
  lhs = f16[64,9]{1,0} parameter(0)
  rhs = f16[4,9,8]{2,1,0} parameter(1)
  gs  = s64[4] constant({16, 8, 24, 16})
  ROOT rd = f16[64,8]{0,1} ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// Non-standard RHS memory layout.  Analogous to the GroupedGemmRewriteTest
// `CustomCallTargetGroupedGemmRaggedInNonContractingGroupDimNoOuterDim` test.
// RHS f16[8,64,32]{2,0,1}: G=8, K=64, N=32 with physical memory order
// N (dim 2, fastest) → G (dim 0) → K (dim 1, slowest).
// This is a K-major RHS layout that differs from the standard {2,1,0}.
// The XTile emitter handles arbitrary strides via TileInfo, so the numerical
// result must still match the row-major reference.
TEST_F(TritonRaggedDotTest, RhsNonStandardLayout) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotRhsNonStdLayout

ENTRY main {
  lhs = f16[128,64]{1,0} parameter(0)
  rhs = f16[8,64,32]{2,0,1} parameter(1)
  gs  = s32[8] constant({16, 16, 16, 16, 16, 16, 16, 16})
  ROOT rd = f16[128,32]{1,0} ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// Column-major LHS layout.
// LHS f16[128,32]{0,1}: M=128, K=32 with K as the physically innermost
// dimension (column-major for the LHS).  Tests that TileInfo correctly
// computes row offsets with non-unit LHS row stride.
TEST_F(TritonRaggedDotTest, LhsColumnMajorLayout) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotLhsColMajor

ENTRY main {
  lhs = f16[128,32]{0,1} parameter(0)
  rhs = f16[4,32,16]{2,1,0} parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = f16[128,16]{1,0} ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

}  // namespace
}  // namespace gpu
}  // namespace xla
