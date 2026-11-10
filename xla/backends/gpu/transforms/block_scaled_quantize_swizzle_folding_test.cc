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

#include "xla/backends/gpu/transforms/block_scaled_quantize_swizzle_folding.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

using BlockScaledQuantizeSwizzleFoldingTest = HloHardwareIndependentTestBase;

// A quantize fusion with an FP8 data and a u8 scale result, and the hipBLASLt
// 32x8 swizzle of a [1, 64, 8] E8M0 scale.
constexpr absl::string_view kComputations = R"hlo(
HloModule m

max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] maximum(a, b)
}

quantize {
  p = f32[64,8,32] parameter(0)
  zero = f32[] constant(0)
  abs = f32[64,8,32] abs(p)
  amax = f32[64,8] reduce(abs, zero), dimensions={2}, to_apply=max
  scale = u8[64,8] convert(amax)
  data = f8e4m3fn[64,8,32] convert(p)
  ROOT out = (f8e4m3fn[64,8,32], u8[64,8]) tuple(data, scale)
}

swizzle {
  scale = f8e8m0fnu[1,64,8] parameter(0)
  split = f8e8m0fnu[2,2,16,1,2,4] reshape(scale)
  permuted = f8e8m0fnu[2,1,4,16,2,2] transpose(split), dimensions={0,3,5,2,4,1}
  ROOT merged = f8e8m0fnu[1,64,8] reshape(permuted)
}

swizzle_copy {
  scale = f8e8m0fnu[1,64,8] parameter(0)
  split = f8e8m0fnu[2,2,16,1,2,4] reshape(scale)
  permuted = f8e8m0fnu[2,1,4,16,2,2] transpose(split), dimensions={0,3,5,2,4,1}
  ROOT merged = f8e8m0fnu[1,64,8] reshape(permuted)
}

swizzle_32x16 {
  scale = f8e8m0fnu[32,16] parameter(0)
  split = f8e8m0fnu[1,2,16,2,2,4] reshape(scale)
  permuted = f8e8m0fnu[1,2,4,16,2,2] transpose(split), dimensions={0,3,5,2,4,1}
  ROOT merged = f8e8m0fnu[32,16] reshape(permuted)
}
)hlo";

constexpr absl::string_view kQuantize = R"hlo(
  x = f32[64,8,32] parameter(0)
  q = (f8e4m3fn[64,8,32], u8[64,8]) fusion(x), kind=kCustom, calls=quantize, backend_config={"fusion_backend_config":{"kind":"__block_scaled_quantize"}}
  data = f8e4m3fn[64,8,32] get-tuple-element(q), index=0
  scale = u8[64,8] get-tuple-element(q), index=1
  scale_e8m0 = f8e8m0fnu[1,64,8] bitcast(scale))hlo";

std::string Module(absl::string_view entry) {
  return absl::StrCat(kComputations, "\nENTRY e {", kQuantize, entry, "\n}\n");
}

int64_t CountSwizzles(const HloModule& module) {
  int64_t count = 0;
  for (const HloInstruction* instr :
       module.entry_computation()->instructions()) {
    count += instr->opcode() == HloOpcode::kFusion &&
             instr->fusion_kind() == HloInstruction::FusionKind::kLoop;
  }
  return count;
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, FoldsSwizzleIntoScaleResult) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  swizzled = f8e8m0fnu[1,64,8] fusion(scale_e8m0), kind=kLoop, calls=swizzle
  ROOT gemm = f32[] custom-call(data, swizzled), custom_call_target="gemm")hlo")));
  const int64_t computations = module->computation_count();
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall(
                  m::GetTupleElement(m::Fusion(), 0),
                  m::Bitcast(m::GetTupleElement(m::Fusion(), 1)))));
  EXPECT_EQ(CountSwizzles(*module), 0);
  EXPECT_EQ(module->computation_count(), computations - 1);
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  const HloInstruction* transpose = nullptr;
  EXPECT_THAT(fusion->fused_expression_root()->operand(1),
              GmockMatch(m::Reshape(
                  m::Transpose(&transpose, m::Reshape(m::Convert())))));
  EXPECT_THAT(transpose->dimensions(), ElementsAre(0, 3, 5, 2, 4, 1));
  EXPECT_EQ(transpose->shape().element_type(), U8);
  EXPECT_EQ(fusion->fused_expression_root()->operand(1)->shape().ToString(),
            "u8[64,8]");
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, FoldsSwizzleSharedByGemms) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  swizzled = f8e8m0fnu[1,64,8] fusion(scale_e8m0), kind=kLoop, calls=swizzle
  gemm0 = f32[] custom-call(data, swizzled), custom_call_target="gemm"
  gemm1 = f32[] custom-call(data, swizzled), custom_call_target="gemm"
  ROOT out = (f32[], f32[]) tuple(gemm0, gemm1))hlo")));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(true));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::CustomCall(m::GetTupleElement(), m::Bitcast()),
                          m::CustomCall(m::GetTupleElement(), m::Bitcast()))));
  EXPECT_EQ(CountSwizzles(*module), 0);
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, FoldsSwizzleBehindBarrier) {
  // The scaled-dot lowering puts the GEMM's operands behind a barrier.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  data_view = f8e4m3fn[1,64,256] bitcast(data)
  pair = (f8e4m3fn[1,64,256], f8e8m0fnu[1,64,8]) tuple(data_view, scale_e8m0)
  barrier = (f8e4m3fn[1,64,256], f8e8m0fnu[1,64,8]) opt-barrier(pair)
  barrier_data = f8e4m3fn[1,64,256] get-tuple-element(barrier), index=0
  barrier_scale = f8e8m0fnu[1,64,8] get-tuple-element(barrier), index=1
  swizzled = f8e8m0fnu[1,64,8] fusion(barrier_scale), kind=kLoop, calls=swizzle
  ROOT gemm = f32[] custom-call(barrier_data, swizzled), custom_call_target="gemm")hlo")));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall(
                  m::GetTupleElement(m::OptimizationBarrier(), 0),
                  m::GetTupleElement(m::OptimizationBarrier(), 1))));
  EXPECT_EQ(CountSwizzles(*module), 0);
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, FoldsScaleBehindSeveralBarriers) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  pair0 = (f8e4m3fn[64,8,32], f8e8m0fnu[1,64,8]) tuple(data, scale_e8m0)
  barrier0 = (f8e4m3fn[64,8,32], f8e8m0fnu[1,64,8]) opt-barrier(pair0)
  data0 = f8e4m3fn[64,8,32] get-tuple-element(barrier0), index=0
  scale0 = f8e8m0fnu[1,64,8] get-tuple-element(barrier0), index=1
  swizzled0 = f8e8m0fnu[1,64,8] fusion(scale0), kind=kLoop, calls=swizzle
  gemm0 = f32[] custom-call(data0, swizzled0), custom_call_target="gemm"
  pair1 = (f8e4m3fn[64,8,32], f8e8m0fnu[1,64,8]) tuple(data, scale_e8m0)
  barrier1 = (f8e4m3fn[64,8,32], f8e8m0fnu[1,64,8]) opt-barrier(pair1)
  data1 = f8e4m3fn[64,8,32] get-tuple-element(barrier1), index=0
  scale1 = f8e8m0fnu[1,64,8] get-tuple-element(barrier1), index=1
  swizzled1 = f8e8m0fnu[1,64,8] fusion(scale1), kind=kLoop, calls=swizzle_copy
  gemm1 = f32[] custom-call(data1, swizzled1), custom_call_target="gemm"
  ROOT out = (f32[], f32[]) tuple(gemm0, gemm1))hlo")));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(true));

  EXPECT_EQ(CountSwizzles(*module), 0);
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest,
       KeepsBarrierElementWithOtherUser) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  pair = (f8e4m3fn[64,8,32], f8e8m0fnu[1,64,8]) tuple(data, scale_e8m0)
  barrier = (f8e4m3fn[64,8,32], f8e8m0fnu[1,64,8]) opt-barrier(pair)
  barrier_data = f8e4m3fn[64,8,32] get-tuple-element(barrier), index=0
  barrier_scale = f8e8m0fnu[1,64,8] get-tuple-element(barrier), index=1
  swizzled = f8e8m0fnu[1,64,8] fusion(barrier_scale), kind=kLoop, calls=swizzle
  gemm0 = f32[] custom-call(barrier_data, swizzled), custom_call_target="gemm"
  gemm1 = f32[] custom-call(barrier_data, barrier_scale), custom_call_target="gemm"
  ROOT out = (f32[], f32[]) tuple(gemm0, gemm1))hlo")));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, KeepsSwizzlesThatDiffer) {
  // [1, 64, 8] and [32, 16] views of one scale permute its bytes differently.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  other_view = f8e8m0fnu[32,16] bitcast(scale)
  swizzled0 = f8e8m0fnu[1,64,8] fusion(scale_e8m0), kind=kLoop, calls=swizzle
  swizzled1 = f8e8m0fnu[32,16] fusion(other_view), kind=kLoop, calls=swizzle_32x16
  gemm0 = f32[] custom-call(data, swizzled0), custom_call_target="gemm"
  gemm1 = f32[] custom-call(data, swizzled1), custom_call_target="gemm"
  ROOT out = (f32[], f32[]) tuple(gemm0, gemm1))hlo")));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, KeepsScaleThatDoesNotSplit) {
  // A [2, 64, 8] scale read as [1, 128, 8]: the split [4, 2, 16, ...] merges
  // its first two dimensions.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
HloModule m

max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] maximum(a, b)
}

quantize {
  p = f32[2,64,8,32] parameter(0)
  zero = f32[] constant(0)
  abs = f32[2,64,8,32] abs(p)
  amax = f32[2,64,8] reduce(abs, zero), dimensions={3}, to_apply=max
  scale = u8[2,64,8] convert(amax)
  data = f8e4m3fn[2,64,8,32] convert(p)
  ROOT out = (f8e4m3fn[2,64,8,32], u8[2,64,8]) tuple(data, scale)
}

swizzle {
  scale = f8e8m0fnu[1,128,8] parameter(0)
  split = f8e8m0fnu[4,2,16,1,2,4] reshape(scale)
  permuted = f8e8m0fnu[4,1,4,16,2,2] transpose(split), dimensions={0,3,5,2,4,1}
  ROOT merged = f8e8m0fnu[1,128,8] reshape(permuted)
}

ENTRY e {
  x = f32[2,64,8,32] parameter(0)
  q = (f8e4m3fn[2,64,8,32], u8[2,64,8]) fusion(x), kind=kCustom, calls=quantize, backend_config={"fusion_backend_config":{"kind":"__block_scaled_quantize"}}
  data = f8e4m3fn[2,64,8,32] get-tuple-element(q), index=0
  scale = u8[2,64,8] get-tuple-element(q), index=1
  scale_e8m0 = f8e8m0fnu[1,128,8] bitcast(scale)
  swizzled = f8e8m0fnu[1,128,8] fusion(scale_e8m0), kind=kLoop, calls=swizzle
  ROOT gemm = f32[] custom-call(data, swizzled), custom_call_target="gemm"
}
)hlo"));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, KeepsScaleWithUnswizzledUser) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  swizzled = f8e8m0fnu[1,64,8] fusion(scale_e8m0), kind=kLoop, calls=swizzle
  gemm0 = f32[] custom-call(data, swizzled), custom_call_target="gemm"
  gemm1 = f32[] custom-call(data, scale_e8m0), custom_call_target="gemm"
  ROOT out = (f32[], f32[]) tuple(gemm0, gemm1))hlo")));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, KeepsSwizzleWithControlEdges) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  swizzled = f8e8m0fnu[1,64,8] fusion(scale_e8m0), kind=kLoop, calls=swizzle, control-predecessors={data}
  ROOT gemm = f32[] custom-call(data, swizzled), custom_call_target="gemm")hlo")));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, KeepsSwizzleOfOtherProducer) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  other = f8e8m0fnu[1,64,8] parameter(1)
  swizzled = f8e8m0fnu[1,64,8] fusion(other), kind=kLoop, calls=swizzle
  gemm = f32[] custom-call(data, swizzled), custom_call_target="gemm"
  ROOT out = (f32[], f8e8m0fnu[1,64,8]) tuple(gemm, scale_e8m0))hlo")));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeSwizzleFoldingTest, IsIdempotent) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(Module(R"hlo(
  swizzled = f8e8m0fnu[1,64,8] fusion(scale_e8m0), kind=kLoop, calls=swizzle
  ROOT gemm = f32[] custom-call(data, swizzled), custom_call_target="gemm")hlo")));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(true));
  EXPECT_THAT(RunHloPass(BlockScaledQuantizeSwizzleFolding(), module.get()),
              IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::gpu
