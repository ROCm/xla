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

// Tests for the experimental dynamic row extent guard in the GPU loop fusion
// emitter (xla_gpu_experimental_dynamic_row_extent). The guard recognizes the
// masked MoE epilogue pattern `select(iota < broadcast(n), value, fill)` and
// makes work items whose rows are at or beyond `n` skip the fusion body.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/loop.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

constexpr int64_t kRows = 4096;
constexpr int64_t kCols = 256;

// The masked epilogue as it reaches the GPU compiler today: `n` only feeds the
// select predicate, the launch grid is derived from the static capacity.
constexpr absl::string_view kClampModule = R"(
HloModule masked_clamp, input_output_alias={ {}: (0, {}) }

ENTRY entry {
  recv = f32[4096,256] parameter(0)
  n = s32[] parameter(1)
  iota = s32[4096,256] iota(), iota_dimension=0
  n_b = s32[4096,256] broadcast(n), dimensions={}
  mask = pred[4096,256] compare(iota, n_b), direction=LT
  lo = f32[] constant(0)
  hi = f32[] constant(50)
  lo_b = f32[4096,256] broadcast(lo), dimensions={}
  hi_b = f32[4096,256] broadcast(hi), dimensions={}
  clamped = f32[4096,256] clamp(lo_b, recv, hi_b)
  act = f32[4096,256] multiply(clamped, recv)
  ROOT masked = f32[4096,256] select(mask, act, lo_b)
})";

// A realistic MoE epilogue: silu(x) * gate + bias, in the element type `T`,
// with the select in the output type `O` (`O` == `T` unless the epilogue casts
// its result).
std::string SiluModule(absl::string_view t, absl::string_view o) {
  const std::string cast =
      t == o ? absl::StrCat("  act = ", o, "[4096,256] copy(act_t)\n")
             : absl::StrCat("  act = ", o, "[4096,256] convert(act_t)\n");
  return absl::StrCat(R"(
HloModule masked_silu

ENTRY entry {
  x = )",
                      t, R"([4096,256] parameter(0)
  gate = )",
                      t, R"([4096,256] parameter(1)
  bias = )",
                      t, R"([256] parameter(2)
  n = s32[] parameter(3)
  iota = s32[4096,256] iota(), iota_dimension=0
  n_b = s32[4096,256] broadcast(n), dimensions={}
  mask = pred[4096,256] compare(iota, n_b), direction=LT
  sig = )",
                      t, R"([4096,256] logistic(x)
  silu = )",
                      t, R"([4096,256] multiply(x, sig)
  gated = )",
                      t, R"([4096,256] multiply(silu, gate)
  bias_b = )",
                      t, R"([4096,256] broadcast(bias), dimensions={1}
  act_t = )",
                      t, R"([4096,256] add(gated, bias_b)
)",
                      cast, R"(  zero = )", o, R"([] constant(0)
  zero_b = )",
                      o, R"([4096,256] broadcast(zero), dimensions={}
  ROOT masked = )",
                      o, R"([4096,256] select(mask, act, zero_b)
})");
}

template <typename NativeT>
Literal MakeRamp(const Shape& shape, float scale) {
  Literal literal(shape);
  int64_t counter = 0;
  CHECK_OK(literal.Populate<NativeT>([&](absl::Span<const int64_t> index) {
    // Values in [-2, 2) * scale, with a different offset in every row so a
    // wrong row mapping is visible.
    const float v = ((counter++ % 97) / 97.0f - 0.5f) * 4.0f * scale;
    return static_cast<NativeT>(v);
  }));
  return literal;
}

class DynamicRowExtentTest : public HloTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = HloTestBase::GetDebugOptionsForTest();
    options.set_xla_gpu_experimental_dynamic_row_extent(enable_guard_);
    return options;
  }

  // Emits the MLIR for the (single) fusion that the entry computation of
  // `hlo` is rooted at, without running any HLO passes.
  absl::StatusOr<std::string> EmitFusionMlir(absl::string_view hlo) {
    ABSL_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo));
    auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_);
    LoopFusion loop_fusion(analysis);
    const MlirKernelEmitter& emitter = loop_fusion;
    mlir::MLIRContext mlir_context(MlirKernelEmitter::GetDialectRegistry());
    mlir_context.loadAllAvailableDialects();
    RegisterSymbolicExprStorage(&mlir_context);
    ABSL_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
        emitter.CreateMLIRModule(mlir_context, *fusion, "main",
                                 /*buffer_assignment=*/nullptr));
    std::string text;
    llvm::raw_string_ostream os(text);
    mlir_module->print(os);
    return text;
  }

  // Runs `hlo` with the guard off (reference, today's masked kernel) and on,
  // and checks that the first `n` rows agree bit for bit.
  void RunAndCompareLivePrefix(absl::string_view hlo,
                               const std::vector<Literal>& inputs, int64_t n) {
    std::vector<const Literal*> args;
    for (const Literal& input : inputs) {
      args.push_back(&input);
    }
    Literal n_literal = LiteralUtil::CreateR0<int32_t>(n);
    args.push_back(&n_literal);

    enable_guard_ = false;
    ASSERT_OK_AND_ASSIGN(auto reference_module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(Literal reference,
                         Execute(std::move(reference_module), args));

    enable_guard_ = true;
    ASSERT_OK_AND_ASSIGN(auto guarded_module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(Literal guarded,
                         Execute(std::move(guarded_module), args));

    const int64_t live_rows = std::min<int64_t>(std::max<int64_t>(n, 0), kRows);
    EXPECT_TRUE(
        LiteralTestUtil::Equal(reference.Slice({0, 0}, {live_rows, kCols}),
                               guarded.Slice({0, 0}, {live_rows, kCols})))
        << "live rows differ for n=" << n;
  }

  bool enable_guard_ = false;
  stream_executor::DeviceDescription device_ =
      TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
};

constexpr absl::string_view kFusedClampModule = R"(
HloModule fused_masked_clamp

fused_computation {
  recv = f32[4096,256] parameter(0)
  n = s32[] parameter(1)
  iota = s32[4096,256] iota(), iota_dimension=0
  n_b = s32[4096,256] broadcast(n), dimensions={}
  mask = pred[4096,256] compare(iota, n_b), direction=LT
  lo = f32[] constant(0)
  hi = f32[] constant(50)
  lo_b = f32[4096,256] broadcast(lo), dimensions={}
  hi_b = f32[4096,256] broadcast(hi), dimensions={}
  clamped = f32[4096,256] clamp(lo_b, recv, hi_b)
  act = f32[4096,256] multiply(clamped, recv)
  ROOT masked = f32[4096,256] select(mask, act, lo_b)
}

ENTRY entry {
  recv = f32[4096,256] parameter(0)
  n = s32[] parameter(1)
  ROOT fusion = f32[4096,256] fusion(recv, n), kind=kLoop, calls=fused_computation
})";

TEST_F(DynamicRowExtentTest, GuardIsEmittedOnlyWhenEnabled) {
  enable_guard_ = false;
  ASSERT_OK_AND_ASSIGN(std::string without, EmitFusionMlir(kFusedClampModule));
  EXPECT_THAT(without, Not(HasSubstr("scf.if")));

  enable_guard_ = true;
  ASSERT_OK_AND_ASSIGN(std::string with, EmitFusionMlir(kFusedClampModule));
  // The extent is read from the fusion's second operand and compared against
  // the work item's first row; the loop over elements sits inside the guard.
  EXPECT_THAT(with, HasSubstr("tensor.extract %arg1[]"));
  EXPECT_THAT(with, HasSubstr("arith.cmpi slt"));
  EXPECT_THAT(with, HasSubstr("scf.if"));
  EXPECT_THAT(with, HasSubstr("xla.loop"));
}

TEST_F(DynamicRowExtentTest, PatternWithoutScalarParameterIsNotGuarded) {
  // The bound is a constant, not a runtime parameter: nothing to guard on.
  constexpr absl::string_view kConstantBound = R"(
HloModule fused_constant_bound

fused_computation {
  recv = f32[4096,256] parameter(0)
  iota = s32[4096,256] iota(), iota_dimension=0
  n = s32[] constant(100)
  n_b = s32[4096,256] broadcast(n), dimensions={}
  mask = pred[4096,256] compare(iota, n_b), direction=LT
  zero = f32[] constant(0)
  zero_b = f32[4096,256] broadcast(zero), dimensions={}
  ROOT masked = f32[4096,256] select(mask, recv, zero_b)
}

ENTRY entry {
  recv = f32[4096,256] parameter(0)
  ROOT fusion = f32[4096,256] fusion(recv), kind=kLoop, calls=fused_computation
})";
  enable_guard_ = true;
  ASSERT_OK_AND_ASSIGN(std::string text, EmitFusionMlir(kConstantBound));
  EXPECT_THAT(text, Not(HasSubstr("scf.if")));
}

TEST_F(DynamicRowExtentTest, ClampLivePrefixMatchesMaskedKernel) {
  std::vector<Literal> inputs;
  inputs.push_back(
      MakeRamp<float>(ShapeUtil::MakeShape(F32, {kRows, kCols}), 30.0f));
  for (int64_t n : {0, 1, 7, 1000, 2048, 4095, 4096}) {
    RunAndCompareLivePrefix(kClampModule, inputs, n);
  }
}

TEST_F(DynamicRowExtentTest, GuardLeavesRowsBeyondExtentUntouched) {
  // The output aliases the input buffer. With the guard, rows >= n are never
  // written, so they still hold the input; the masked kernel writes zeros.
  Literal input =
      MakeRamp<float>(ShapeUtil::MakeShape(F32, {kRows, kCols}), 30.0f);
  const Literal expected_tail = input.Slice({1024, 0}, {kRows, kCols});
  Literal n_literal = LiteralUtil::CreateR0<int32_t>(1024);
  std::vector<const Literal*> args = {&input, &n_literal};

  enable_guard_ = true;
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kClampModule));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), args));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_tail,
                                     result.Slice({1024, 0}, {kRows, kCols})));

  enable_guard_ = false;
  ASSERT_OK_AND_ASSIGN(module, ParseAndReturnVerifiedModule(kClampModule));
  ASSERT_OK_AND_ASSIGN(result, Execute(std::move(module), args));
  Literal zeros(ShapeUtil::MakeShape(F32, {kRows - 1024, kCols}));
  zeros.PopulateWithValue<float>(0.0f);
  EXPECT_TRUE(
      LiteralTestUtil::Equal(zeros, result.Slice({1024, 0}, {kRows, kCols})));
}

TEST_F(DynamicRowExtentTest, SiluF32LivePrefixMatchesMaskedKernel) {
  std::vector<Literal> inputs;
  inputs.push_back(
      MakeRamp<float>(ShapeUtil::MakeShape(F32, {kRows, kCols}), 1.0f));
  inputs.push_back(
      MakeRamp<float>(ShapeUtil::MakeShape(F32, {kRows, kCols}), 0.5f));
  inputs.push_back(MakeRamp<float>(ShapeUtil::MakeShape(F32, {kCols}), 0.1f));
  RunAndCompareLivePrefix(SiluModule("f32", "f32"), std::move(inputs), 1500);
}

TEST_F(DynamicRowExtentTest, SiluBf16LivePrefixMatchesMaskedKernel) {
  std::vector<Literal> inputs;
  inputs.push_back(
      MakeRamp<bfloat16>(ShapeUtil::MakeShape(BF16, {kRows, kCols}), 1.0f));
  inputs.push_back(
      MakeRamp<bfloat16>(ShapeUtil::MakeShape(BF16, {kRows, kCols}), 0.5f));
  inputs.push_back(
      MakeRamp<bfloat16>(ShapeUtil::MakeShape(BF16, {kCols}), 0.1f));
  RunAndCompareLivePrefix(SiluModule("bf16", "bf16"), std::move(inputs), 777);
}

TEST_F(DynamicRowExtentTest, SiluF32ToBf16LivePrefixMatchesMaskedKernel) {
  std::vector<Literal> inputs;
  inputs.push_back(
      MakeRamp<float>(ShapeUtil::MakeShape(F32, {kRows, kCols}), 1.0f));
  inputs.push_back(
      MakeRamp<float>(ShapeUtil::MakeShape(F32, {kRows, kCols}), 0.5f));
  inputs.push_back(MakeRamp<float>(ShapeUtil::MakeShape(F32, {kCols}), 0.1f));
  RunAndCompareLivePrefix(SiluModule("f32", "bf16"), std::move(inputs), 3000);
}

}  // namespace
}  // namespace xla::gpu
