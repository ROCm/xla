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

// ============================================================================
// kRaggedContracting tests
//
// HLO shape: LHS [M_total, K], RHS [M_total, N], gs [G] → output [G, K, N]
//   lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
//
// For each group g: output[g, :, :] = LHS[sum_m:sum_m+gs[g], :]^T
//                                     @ RHS[sum_m:sum_m+gs[g], :]
// where sum_m = sum(gs[0..g-1]).
// ============================================================================

// Basic kRaggedContracting — balanced groups.
// M_total=128, K=32, N=16, G=4, groups all equal (32 rows each).
TEST_F(TritonRaggedDotTest, ContractingBalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingBalanced

ENTRY main {
  lhs = f32[128,32] parameter(0)
  rhs = f32[128,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = f32[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// kRaggedContracting — unbalanced groups.
// M_total=96, K=32, N=8, G=3, groups {10, 30, 56}.
TEST_F(TritonRaggedDotTest, ContractingUnbalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingUnbalanced

ENTRY main {
  lhs = f32[96,32] parameter(0)
  rhs = f32[96,8] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = f32[3,32,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// kRaggedContracting — groups not multiples of BLOCK_M (exercises M-boundary
// masking in the accumulation loop).
// M_total=96, K=32, N=16, G=3, groups {10, 30, 56}.
TEST_F(TritonRaggedDotTest, ContractingNonMultipleBlock) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingNonMultiple

ENTRY main {
  lhs = f32[96,32] parameter(0)
  rhs = f32[96,16] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = f32[3,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// kRaggedContracting — transposed LHS [K, M_total] layout.
// Analogous to the GroupedGemmRewriteTest kRaggedContracting test where
// LHS has shape [K, M_total] (lhs_contracting_dims={1}).
// M_total=96, K=64, N=32, G=3.
TEST_F(TritonRaggedDotTest, ContractingLhsTransposed) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingTransposed

ENTRY main {
  lhs = f32[64,96] parameter(0)
  rhs = f32[96,32] parameter(1)
  gs  = s32[3] constant({32, 32, 32})
  ROOT rd = f32[3,64,32] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      lhs_ragged_dims={1}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// kRaggedContracting in f16.
TEST_F(TritonRaggedDotTest, ContractingFp16) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingFp16

ENTRY main {
  lhs = f16[64,32] parameter(0)
  rhs = f16[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f16[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedContracting in bf16 — balanced groups.
TEST_F(TritonRaggedDotTest, ContractingBf16Balanced) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingBf16Balanced

ENTRY main {
  lhs = bf16[128,32] parameter(0)
  rhs = bf16[128,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = bf16[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedContracting in bf16 — unbalanced groups exercising M-boundary masking.
TEST_F(TritonRaggedDotTest, ContractingBf16Unbalanced) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingBf16Unbalanced

ENTRY main {
  lhs = bf16[96,32] parameter(0)
  rhs = bf16[96,8] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = bf16[3,32,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedContracting with FP8 (E4M3FNUZ) inputs and F32 output.
// The Triton emitter always accumulates in f32 and casts the output afterward.
// fp8 ragged-dot requires hardware fp8 capability; numerical check runs only
// when Triton is available.
TEST_F(TritonRaggedDotTest, ContractingFp8F32Output) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingFp8

ENTRY main {
  lhs = f8e4m3fnuz[64,32] parameter(0)
  rhs = f8e4m3fnuz[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f32[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-1, 1e-1});
}

// ============================================================================
// kRaggedContracting layout tests
//
// These mirror the kRaggedNonContracting layout tests but for the contracting
// mode.  The emitter uses direct ExtractTileOp loads with explicit buffer
// offsets, so non-standard layouts still produce correct results.
// ============================================================================

// kRaggedContracting with column-major output [G, K, N]{1,2,0}.
// The output's physically fastest dim is N (dim 2), then G (dim 0),
// then K (dim 1) slowest.  The InsertTileOp handles arbitrary output strides.
TEST_F(TritonRaggedDotTest, ContractingOutputColumnMajor) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingColMajorOut

ENTRY main {
  lhs = f16[64,32] parameter(0)
  rhs = f16[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f16[4,32,16]{1,2,0} ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedContracting with column-major LHS [K, M_total]{0,1}.
// LHS is [K, M_total] with M at dim 1 (lhs_contracting_dims={1}).
// The column-major layout makes K the physically innermost dim.
TEST_F(TritonRaggedDotTest, ContractingLhsColMajorTransposed) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingLhsColMajorTransposed

ENTRY main {
  lhs = f16[32,64]{0,1} parameter(0)
  rhs = f16[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f16[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      lhs_ragged_dims={1}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedContracting with s64 group_sizes — exercises 64-bit group-size path.
// Same shapes as ContractingBalancedGroups but gs type is s64.
TEST_F(TritonRaggedDotTest, ContractingS64GroupSizes) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotContractingS64

ENTRY main {
  lhs = f32[128,32] parameter(0)
  rhs = f32[128,16] parameter(1)
  gs  = s64[4] constant({32, 32, 32, 32})
  ROOT rd = f32[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// ============================================================================
// Autotuning integration tests
//
// These tests enable the Triton autotuner (autotune_level is NOT set to 0)
// to verify that:
//   1. TritonBackend::IsSupported() returns true for kRaggedDot fusions.
//   2. TritonBackend::GetSupportedConfigs() generates at least one candidate.
//   3. TritonBackend::ApplyConfig() correctly updates both the fusion-level
//      BlockLevelFusionConfig and the inner ragged-dot Tile proto.
//   4. The compiled fusion produces numerically correct results.
// ============================================================================

class TritonRaggedDotAutotunedTest
    : public HloPjRtInterpreterReferenceMixin<GemmRewriteTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_triton_ragged_dot(true);
    debug_options.set_xla_gpu_experimental_enable_tiling_propagation(true);
    // Do NOT set autotune_level=0 — let the autotuner run and select tile
    // sizes from GetSupportedConfigsForRaggedDot.
    return debug_options;
  }

  bool SupportsTriton() const {
    if (IsCuda()) {
      auto* cc = Capability().cuda_compute_capability();
      return cc != nullptr && cc->IsAtLeastAmpere();
    }
    return true;  // ROCm always supports Triton.
  }
};

// Balanced groups — autotuner selects the best (BLOCK_M, BLOCK_N, BLOCK_K).
TEST_F(TritonRaggedDotAutotunedTest, BalancedGroups) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotAutotunedBalanced

ENTRY main {
  lhs = f32[128,32] parameter(0)
  rhs = f32[4,32,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = f32[128,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4, 1e-4}));
}

// Unbalanced groups — exercises M-boundary masking across candidate tile sizes.
TEST_F(TritonRaggedDotAutotunedTest, UnbalancedGroups) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotAutotunedUnbalanced

ENTRY main {
  lhs = f32[96,32] parameter(0)
  rhs = f32[3,32,8] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = f32[96,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// F16 — exercises dtype-aware tile selection.
TEST_F(TritonRaggedDotAutotunedTest, Fp16BalancedGroups) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotAutotunedFp16

ENTRY main {
  lhs = f16[128,32] parameter(0)
  rhs = f16[4,32,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = f16[128,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

// kRaggedContracting balanced — autotuner selects best (BLOCK_M, BLOCK_K,
// BLOCK_N) for the contracting-mode group-GEMM (weight-gradient kernel).
TEST_F(TritonRaggedDotAutotunedTest, ContractingBalancedGroups) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotAutotunedContractingBalanced

ENTRY main {
  lhs = f32[128,32] parameter(0)
  rhs = f32[128,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = f32[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// kRaggedContracting unbalanced — exercises M-boundary masking across
// autotuner candidate tile sizes.
TEST_F(TritonRaggedDotAutotunedTest, ContractingUnbalancedGroups) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotAutotunedContractingUnbalanced

ENTRY main {
  lhs = f32[96,32] parameter(0)
  rhs = f32[96,8] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = f32[3,32,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// kRaggedContracting f16 — autotuner dtype-aware tile selection for fp16.
TEST_F(TritonRaggedDotAutotunedTest, ContractingFp16Balanced) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotAutotunedContractingFp16

ENTRY main {
  lhs = f16[64,32] parameter(0)
  rhs = f16[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f16[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

// kRaggedContracting with transposed LHS — autotuner handles [K,M] layout.
TEST_F(TritonRaggedDotAutotunedTest, ContractingLhsTransposed) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotAutotunedContractingTransposed

ENTRY main {
  lhs = f32[64,96] parameter(0)
  rhs = f32[96,32] parameter(1)
  gs  = s32[3] constant({32, 32, 32})
  ROOT rd = f32[3,64,32] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      lhs_ragged_dims={1}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// ============================================================================
// Persistent kRaggedContracting tests
//
// These run with xla_gpu_experimental_triton_ragged_dot_persistent_contracting
// = true.  Grid = K_tiles × N_tiles (no G in the grid); each program runs a G
// sequential outer loop and writes one [K, N] output tile per group via an
// inline InsertTileOp — analogous to the aiter tgmm_persistent_kernel.
// ============================================================================

class TritonRaggedDotPersistentContractingTest
    : public HloPjRtInterpreterReferenceMixin<GemmRewriteTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_triton_ragged_dot(true);
    debug_options.set_xla_gpu_experimental_enable_tiling_propagation(true);
    debug_options
        .set_xla_gpu_experimental_triton_ragged_dot_persistent_contracting(
            true);
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }

  bool SupportsTriton() const {
    if (IsCuda()) {
      auto* cc = Capability().cuda_compute_capability();
      return cc != nullptr && cc->IsAtLeastAmpere();
    }
    return true;  // ROCm always supports Triton.
  }

  void CheckHloAndMaybeRun(const char* hlo_text,
                           const ErrorSpec& error_spec = ErrorSpec{1e-4,
                                                                   1e-4}) {
    MatchOptimizedHlo(hlo_text, R"(
      ; CHECK:     kind=kCustom
      ; CHECK-SAME: "__triton"
    )");
    if (SupportsTriton()) {
      EXPECT_TRUE(RunAndCompare(hlo_text, error_spec));
    } else {
      GTEST_SKIP() << "Triton not available on this device.";
    }
  }
};

// Persistent balanced groups — G sequential outer loop, K×N grid.
TEST_F(TritonRaggedDotPersistentContractingTest, BalancedGroups) {
  const char* hlo_text = R"(
HloModule PersistentContractingBalanced

ENTRY main {
  lhs = f32[128,32] parameter(0)
  rhs = f32[128,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = f32[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// Persistent unbalanced groups — exercises M-boundary masking.
TEST_F(TritonRaggedDotPersistentContractingTest, UnbalancedGroups) {
  const char* hlo_text = R"(
HloModule PersistentContractingUnbalanced

ENTRY main {
  lhs = f32[96,32] parameter(0)
  rhs = f32[96,8] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = f32[3,32,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// Persistent f16.
TEST_F(TritonRaggedDotPersistentContractingTest, Fp16) {
  const char* hlo_text = R"(
HloModule PersistentContractingFp16

ENTRY main {
  lhs = f16[64,32] parameter(0)
  rhs = f16[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f16[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// Persistent with transposed LHS [K, M_total] — contracting_dim=1.
TEST_F(TritonRaggedDotPersistentContractingTest, LhsTransposed) {
  const char* hlo_text = R"(
HloModule PersistentContractingLhsTransposed

ENTRY main {
  lhs = f32[64,96] parameter(0)
  rhs = f32[96,32] parameter(1)
  gs  = s32[3] constant({32, 32, 32})
  ROOT rd = f32[3,64,32] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}, lhs_ragged_dims={1}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// Persistent non-multiple-of-block-size groups.
TEST_F(TritonRaggedDotPersistentContractingTest, NonMultipleBlock) {
  const char* hlo_text = R"(
HloModule PersistentContractingNonMultiple

ENTRY main {
  lhs = f32[96,32] parameter(0)
  rhs = f32[96,16] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = f32[3,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// Persistent bf16 — balanced groups.
TEST_F(TritonRaggedDotPersistentContractingTest, Bf16Balanced) {
  const char* hlo_text = R"(
HloModule PersistentContractingBf16Balanced

ENTRY main {
  lhs = bf16[128,32] parameter(0)
  rhs = bf16[128,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = bf16[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// Persistent bf16 — unbalanced groups.
TEST_F(TritonRaggedDotPersistentContractingTest, Bf16Unbalanced) {
  const char* hlo_text = R"(
HloModule PersistentContractingBf16Unbalanced

ENTRY main {
  lhs = bf16[96,32] parameter(0)
  rhs = bf16[96,8] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = bf16[3,32,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// Persistent fp8 with f32 output.
TEST_F(TritonRaggedDotPersistentContractingTest, Fp8F32Output) {
  const char* hlo_text = R"(
HloModule PersistentContractingFp8

ENTRY main {
  lhs = f8e4m3fnuz[64,32] parameter(0)
  rhs = f8e4m3fnuz[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f32[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-1, 1e-1});
}

// Persistent with column-major output [G, K, N]{1,2,0}.
TEST_F(TritonRaggedDotPersistentContractingTest, OutputColumnMajor) {
  const char* hlo_text = R"(
HloModule PersistentContractingColMajorOut

ENTRY main {
  lhs = f16[64,32] parameter(0)
  rhs = f16[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f16[4,32,16]{1,2,0} ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// Persistent with column-major LHS [K, M]{0,1} and contracting_dim=1.
TEST_F(TritonRaggedDotPersistentContractingTest, LhsColMajorTransposed) {
  const char* hlo_text = R"(
HloModule PersistentContractingLhsColMajorTransposed

ENTRY main {
  lhs = f16[32,64]{0,1} parameter(0)
  rhs = f16[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f16[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}, lhs_ragged_dims={1}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

TEST_F(TritonRaggedDotPersistentContractingTest, ScheduleSwap_LargeN) {
  const char* hlo_text = R"(
HloModule PersistentContractingScheduleSwap_LargeN

ENTRY main {
  lhs = f16[256,192] parameter(0)
  rhs = f16[256,512] parameter(1)
  gs  = s32[8] constant({32, 32, 32, 32, 32, 32, 32, 32})
  ROOT rd = f16[8,192,512] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

TEST_F(TritonRaggedDotPersistentContractingTest, ScheduleSwap_LargeK) {
  const char* hlo_text = R"(
HloModule PersistentContractingScheduleSwap_LargeK

ENTRY main {
  lhs = f16[256,512] parameter(0)
  rhs = f16[256,192] parameter(1)
  gs  = s32[8] constant({32, 32, 32, 32, 32, 32, 32, 32})
  ROOT rd = f16[8,512,192] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

TEST_F(TritonRaggedDotPersistentContractingTest, ScheduleSwap_ModerateNK) {
  const char* hlo_text = R"(
HloModule PersistentContractingScheduleSwap_ModerateNK

ENTRY main {
  lhs = f16[256,64] parameter(0)
  rhs = f16[256,96] parameter(1)
  gs  = s32[8] constant({32, 32, 32, 32, 32, 32, 32, 32})
  ROOT rd = f16[8,64,96] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// Persistent with s64 group_sizes.
TEST_F(TritonRaggedDotPersistentContractingTest, S64GroupSizes) {
  const char* hlo_text = R"(
HloModule PersistentContractingS64GroupSizes

ENTRY main {
  lhs = f32[128,32] parameter(0)
  rhs = f32[128,16] parameter(1)
  gs  = s64[4] constant({32, 32, 32, 32})
  ROOT rd = f32[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// ============================================================================
// Persistent kRaggedContracting autotuner tests
// ============================================================================

class TritonRaggedDotPersistentContractingAutotunedTest
    : public HloPjRtInterpreterReferenceMixin<GemmRewriteTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_triton_ragged_dot(true);
    debug_options.set_xla_gpu_experimental_enable_tiling_propagation(true);
    debug_options
        .set_xla_gpu_experimental_triton_ragged_dot_persistent_contracting(
            true);
    // Do NOT set autotune_level=0 — let the autotuner select tile sizes.
    return debug_options;
  }

  bool SupportsTriton() const {
    if (IsCuda()) {
      auto* cc = Capability().cuda_compute_capability();
      return cc != nullptr && cc->IsAtLeastAmpere();
    }
    return true;
  }
};

// Persistent autotuner — balanced groups, f32.
TEST_F(TritonRaggedDotPersistentContractingAutotunedTest, BalancedGroupsF32) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule PersistentAutotunedContractingBalancedF32

ENTRY main {
  lhs = f32[128,32] parameter(0)
  rhs = f32[128,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = f32[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Persistent autotuner — unbalanced groups exercising M-boundary masking.
TEST_F(TritonRaggedDotPersistentContractingAutotunedTest, UnbalancedGroups) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule PersistentAutotunedContractingUnbalanced

ENTRY main {
  lhs = f32[96,32] parameter(0)
  rhs = f32[96,8] parameter(1)
  gs  = s32[3] constant({10, 30, 56})
  ROOT rd = f32[3,32,8] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// Persistent autotuner — f16, dtype-aware tile selection.
TEST_F(TritonRaggedDotPersistentContractingAutotunedTest, Fp16) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule PersistentAutotunedContractingFp16

ENTRY main {
  lhs = f16[64,32] parameter(0)
  rhs = f16[64,16] parameter(1)
  gs  = s32[4] constant({16, 16, 16, 16})
  ROOT rd = f16[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

// Persistent autotuner — bf16.
TEST_F(TritonRaggedDotPersistentContractingAutotunedTest, Bf16) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule PersistentAutotunedContractingBf16

ENTRY main {
  lhs = bf16[128,32] parameter(0)
  rhs = bf16[128,16] parameter(1)
  gs  = s32[4] constant({32, 32, 32, 32})
  ROOT rd = bf16[4,32,16] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

// Persistent autotuner — transposed LHS [K, M].
TEST_F(TritonRaggedDotPersistentContractingAutotunedTest, LhsTransposed) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule PersistentAutotunedContractingTransposed

ENTRY main {
  lhs = f32[64,96] parameter(0)
  rhs = f32[96,32] parameter(1)
  gs  = s32[3] constant({32, 32, 32})
  ROOT rd = f32[3,64,32] ragged-dot(lhs, rhs, gs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}, lhs_ragged_dims={1}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// ============================================================================
// kRaggedBatch tests
//
// HLO shape: LHS [B_total, M, K], RHS [B_total, K, N], gs [G] → [B_total,M,N]
//   lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_ragged_dims={0}
//
// kRaggedBatch = batched GEMM where B_total = sum(group_sizes) and
// group_sizes[g] batch elements belong to group g.
// ============================================================================

class TritonRaggedDotBatchTest
    : public HloPjRtInterpreterReferenceMixin<GemmRewriteTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_triton_ragged_dot(true);
    debug_options.set_xla_gpu_experimental_enable_tiling_propagation(true);
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }

  bool SupportsTriton() const {
    if (IsCuda()) {
      auto* cc = Capability().cuda_compute_capability();
      return cc != nullptr && cc->IsAtLeastAmpere();
    }
    return true;
  }

  void CheckHloAndMaybeRun(const char* hlo_text,
                           const ErrorSpec& error_spec = ErrorSpec{1e-4,
                                                                   1e-4}) {
    MatchOptimizedHlo(hlo_text, R"(
      ; CHECK:     kind=kCustom
      ; CHECK-SAME: "__triton"
    )");
    if (SupportsTriton()) {
      EXPECT_TRUE(RunAndCompare(hlo_text, error_spec));
    } else {
      GTEST_SKIP() << "Triton not available on this device.";
    }
  }
};

// Basic kRaggedBatch — balanced groups (2 groups, 4 batch elems each).
// B_total=8, M=16, K=32, N=8, G=2.
TEST_F(TritonRaggedDotBatchTest, BalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatch

ENTRY main {
  lhs = f32[8,16,32] parameter(0)
  rhs = f32[8,32,8] parameter(1)
  gs  = s32[2] constant({4, 4})
  ROOT rd = f32[8,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// kRaggedBatch — unbalanced groups.
TEST_F(TritonRaggedDotBatchTest, UnbalancedGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchUnbalanced

ENTRY main {
  lhs = f32[5,16,9] parameter(0)
  rhs = f32[5,9,8] parameter(1)
  gs  = s64[2] constant({3, 2})
  ROOT rd = f32[5,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// kRaggedBatch f16.
TEST_F(TritonRaggedDotBatchTest, Fp16) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchFp16

ENTRY main {
  lhs = f16[16,64,9] parameter(0)
  rhs = f16[16,9,8] parameter(1)
  gs  = s64[4] constant({4, 2, 6, 4})
  ROOT rd = f16[16,64,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedBatch bf16.
TEST_F(TritonRaggedDotBatchTest, Bf16) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchBf16

ENTRY main {
  lhs = bf16[8,16,32] parameter(0)
  rhs = bf16[8,32,8] parameter(1)
  gs  = s32[2] constant({4, 4})
  ROOT rd = bf16[8,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedBatch fp8 with f32 output.
TEST_F(TritonRaggedDotBatchTest, Fp8F32Output) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchFp8

ENTRY main {
  lhs = f8e4m3fnuz[8,16,32] parameter(0)
  rhs = f8e4m3fnuz[8,32,8] parameter(1)
  gs  = s32[2] constant({4, 4})
  ROOT rd = f32[8,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-1, 1e-1});
}

// kRaggedBatch column-major output [B, M, N]{1,2,0}.
TEST_F(TritonRaggedDotBatchTest, OutputColumnMajor) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchColMajorOut

ENTRY main {
  lhs = f16[8,16,32] parameter(0)
  rhs = f16[8,32,8] parameter(1)
  gs  = s32[2] constant({4, 4})
  ROOT rd = f16[8,16,8]{1,2,0} ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedBatch column-major LHS [B, K, M]{0,1,2} with lhs_contracting=1.
// LHS K is the inner dim — tests non-unit LHS K stride.
TEST_F(TritonRaggedDotBatchTest, LhsColumnMajor) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchLhsColMajor

ENTRY main {
  lhs = f16[8,32,16]{0,1,2} parameter(0)
  rhs = f16[8,32,8] parameter(1)
  gs  = s32[2] constant({4, 4})
  ROOT rd = f16[8,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedBatch column-major RHS [B, N, K]{0,1,2} with rhs_contracting=2.
TEST_F(TritonRaggedDotBatchTest, RhsColumnMajor) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchRhsColMajor

ENTRY main {
  lhs = f16[8,16,32] parameter(0)
  rhs = f16[8,8,32]{0,1,2} parameter(1)
  gs  = s32[2] constant({4, 4})
  ROOT rd = f16[8,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-2, 1e-2});
}

// kRaggedBatch with s64 group_sizes.
TEST_F(TritonRaggedDotBatchTest, S64GroupSizes) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchS64

ENTRY main {
  lhs = f32[8,16,32] parameter(0)
  rhs = f32[8,32,8] parameter(1)
  gs  = s64[2] constant({4, 4})
  ROOT rd = f32[8,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// kRaggedBatch with multiple groups (4 groups).
TEST_F(TritonRaggedDotBatchTest, FourGroups) {
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchFourGroups

ENTRY main {
  lhs = f32[16,64,9] parameter(0)
  rhs = f32[16,9,8] parameter(1)
  gs  = s64[4] constant({4, 2, 6, 4})
  ROOT rd = f32[16,64,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  CheckHloAndMaybeRun(hlo_text, ErrorSpec{1e-3, 1e-3});
}

// kRaggedBatch autotuner.
class TritonRaggedDotBatchAutotunedTest
    : public HloPjRtInterpreterReferenceMixin<GemmRewriteTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmRewriteTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_triton_ragged_dot(true);
    debug_options.set_xla_gpu_experimental_enable_tiling_propagation(true);
    return debug_options;
  }
  bool SupportsTriton() const {
    if (IsCuda()) {
      auto* cc = Capability().cuda_compute_capability();
      return cc != nullptr && cc->IsAtLeastAmpere();
    }
    return true;
  }
};

TEST_F(TritonRaggedDotBatchAutotunedTest, BalancedGroups) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchAutotuned

ENTRY main {
  lhs = f32[8,16,32] parameter(0)
  rhs = f32[8,32,8] parameter(1)
  gs  = s32[2] constant({4, 4})
  ROOT rd = f32[8,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// kRaggedBatch autotuner — f16.
TEST_F(TritonRaggedDotBatchAutotunedTest, Fp16) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchAutotunedFp16

ENTRY main {
  lhs = f16[16,64,9] parameter(0)
  rhs = f16[16,9,8] parameter(1)
  gs  = s64[4] constant({4, 2, 6, 4})
  ROOT rd = f16[16,64,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

// kRaggedBatch autotuner — bf16.
TEST_F(TritonRaggedDotBatchAutotunedTest, Bf16) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchAutotunedBf16

ENTRY main {
  lhs = bf16[8,16,32] parameter(0)
  rhs = bf16[8,32,8] parameter(1)
  gs  = s32[2] constant({4, 4})
  ROOT rd = bf16[8,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

// kRaggedBatch autotuner — unbalanced groups.
TEST_F(TritonRaggedDotBatchAutotunedTest, UnbalancedGroups) {
  if (!SupportsTriton()) GTEST_SKIP() << "Triton not available.";
  const char* hlo_text = R"(
HloModule TritonRaggedDotBatchAutotunedUnbalanced

ENTRY main {
  lhs = f32[5,16,9] parameter(0)
  rhs = f32[5,9,8] parameter(1)
  gs  = s64[2] constant({3, 2})
  ROOT rd = f32[5,16,8] ragged-dot(lhs, rhs, gs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_ragged_dims={0}
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
