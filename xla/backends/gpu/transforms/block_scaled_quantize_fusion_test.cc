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

#include "xla/backends/gpu/transforms/block_scaled_quantize_fusion.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/layout_util.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::IsSupersetOf;
using ::testing::UnorderedElementsAre;

using BlockScaledQuantizeFusionTest = HloHardwareIndependentTestBase;

BlockScaledQuantizeFusion AcceptingPass() {
  return BlockScaledQuantizeFusion(
      [](const HloFusionInstruction&) { return Decision::Allow(); });
}

// A row quantization of `s` to FP8, as in the forward body.
constexpr absl::string_view kRowQuantization = R"hlo(
HloModule m

max_bf16 {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT r = bf16[] maximum(a, b)
}

ENTRY e {
  s = bf16[2,64,128] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[2,64,128] abs(s)
  abs_blocks = bf16[128,4,32] bitcast(abs)
  amax = bf16[128,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,4] convert(amax)
  q_k_rmax = f32[] constant(0.00223214296)
  q_b_rmax = f32[128,4] broadcast(q_k_rmax), dimensions={}
  q_ratio = f32[128,4] multiply(amax_f32, q_b_rmax)
  q_bits = u32[128,4] bitcast(q_ratio)
  q_k_mask = u32[] constant(8388607)
  q_b_mask = u32[128,4] broadcast(q_k_mask), dimensions={}
  q_mant = u32[128,4] and(q_bits, q_b_mask)
  q_k_0 = u32[] constant(0)
  q_b_0 = u32[128,4] broadcast(q_k_0), dimensions={}
  q_gt = pred[128,4] compare(q_mant, q_b_0), direction=GT
  q_k_23 = u32[] constant(23)
  q_b_23 = u32[128,4] broadcast(q_k_23), dimensions={}
  q_exp = u32[128,4] shift-right-logical(q_bits, q_b_23)
  q_k_254 = u32[] constant(254)
  q_b_254 = u32[128,4] broadcast(q_k_254), dimensions={}
  q_ne = pred[128,4] compare(q_exp, q_b_254), direction=NE
  q_and0 = pred[128,4] and(q_gt, q_ne)
  q_eq = pred[128,4] compare(q_exp, q_b_0), direction=EQ
  q_k_half = u32[] constant(4194304)
  q_b_half = u32[128,4] broadcast(q_k_half), dimensions={}
  q_le = pred[128,4] compare(q_mant, q_b_half), direction=LE
  q_and1 = pred[128,4] and(q_eq, q_le)
  q_not = pred[128,4] not(q_and1)
  q_round_up = pred[128,4] and(q_and0, q_not)
  q_k_1 = u32[] constant(1)
  q_b_1 = u32[128,4] broadcast(q_k_1), dimensions={}
  q_exp1 = u32[128,4] add(q_exp, q_b_1)
  q_rounded = u32[128,4] select(q_round_up, q_exp1, q_exp)
  q_scale = u8[128,4] convert(q_rounded)
  q_scale_u32 = u32[128,4] convert(q_scale)
  q_shifted = u32[128,4] shift-left(q_scale_u32, q_b_23)
  q_decoded = f32[128,4] bitcast(q_shifted)
  q_k_f0 = f32[] constant(0)
  q_b_f0 = f32[128,4] broadcast(q_k_f0), dimensions={}
  q_is_zero = pred[128,4] compare(q_decoded, q_b_f0), direction=EQ
  q_k_tiny = f32[] constant(5.87747175e-39)
  q_b_tiny = f32[128,4] broadcast(q_k_tiny), dimensions={}
  q_scale_f32 = f32[128,4] select(q_is_zero, q_b_tiny, q_decoded)
  s_blocks = bf16[128,4,32] bitcast(s)
  s_f32 = f32[128,4,32] convert(s_blocks)
  q_scale_b = f32[128,4,32] broadcast(q_scale_f32), dimensions={0,1}
  q_scaled = f32[128,4,32] divide(s_f32, q_scale_b)
  q_k_lo = f32[] constant(-448)
  q_b_lo = f32[128,4,32] broadcast(q_k_lo), dimensions={}
  q_k_hi = f32[] constant(448)
  q_b_hi = f32[128,4,32] broadcast(q_k_hi), dimensions={}
  q_clamped = f32[128,4,32] clamp(q_b_lo, q_scaled, q_b_hi)
  q_data = f8e4m3fn[128,4,32] convert(q_clamped)
  data_out = f8e4m3fn[1,128,128] bitcast(q_data)
  scale_out = f8e8m0fnu[1,128,4] bitcast(q_scale)
  ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) tuple(data_out, scale_out)
}
)hlo";

// A row and a normalized column quantization of `s`, as in the backward body.
constexpr absl::string_view kRowAndColumnQuantization = R"hlo(
HloModule m

max_bf16 {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT r = bf16[] maximum(a, b)
}

ENTRY e {
  s = bf16[2,64,128] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[2,64,128] abs(s)
  s_f32 = f32[2,64,128] convert(s)
  abs_blocks_r = bf16[128,4,32] bitcast(abs)
  amax_r = bf16[128,4] reduce(abs_blocks_r, ninf), dimensions={2}, to_apply=max_bf16
  amax_r_f32 = f32[128,4] convert(amax_r)
  r_k_rmax = f32[] constant(0.00223214296)
  r_b_rmax = f32[128,4] broadcast(r_k_rmax), dimensions={}
  r_ratio = f32[128,4] multiply(amax_r_f32, r_b_rmax)
  r_bits = u32[128,4] bitcast(r_ratio)
  r_k_mask = u32[] constant(8388607)
  r_b_mask = u32[128,4] broadcast(r_k_mask), dimensions={}
  r_mant = u32[128,4] and(r_bits, r_b_mask)
  r_k_0 = u32[] constant(0)
  r_b_0 = u32[128,4] broadcast(r_k_0), dimensions={}
  r_gt = pred[128,4] compare(r_mant, r_b_0), direction=GT
  r_k_23 = u32[] constant(23)
  r_b_23 = u32[128,4] broadcast(r_k_23), dimensions={}
  r_exp = u32[128,4] shift-right-logical(r_bits, r_b_23)
  r_k_254 = u32[] constant(254)
  r_b_254 = u32[128,4] broadcast(r_k_254), dimensions={}
  r_ne = pred[128,4] compare(r_exp, r_b_254), direction=NE
  r_and0 = pred[128,4] and(r_gt, r_ne)
  r_eq = pred[128,4] compare(r_exp, r_b_0), direction=EQ
  r_k_half = u32[] constant(4194304)
  r_b_half = u32[128,4] broadcast(r_k_half), dimensions={}
  r_le = pred[128,4] compare(r_mant, r_b_half), direction=LE
  r_and1 = pred[128,4] and(r_eq, r_le)
  r_not = pred[128,4] not(r_and1)
  r_round_up = pred[128,4] and(r_and0, r_not)
  r_k_1 = u32[] constant(1)
  r_b_1 = u32[128,4] broadcast(r_k_1), dimensions={}
  r_exp1 = u32[128,4] add(r_exp, r_b_1)
  r_rounded = u32[128,4] select(r_round_up, r_exp1, r_exp)
  r_scale = u8[128,4] convert(r_rounded)
  r_scale_u32 = u32[128,4] convert(r_scale)
  r_shifted = u32[128,4] shift-left(r_scale_u32, r_b_23)
  r_decoded = f32[128,4] bitcast(r_shifted)
  r_k_f0 = f32[] constant(0)
  r_b_f0 = f32[128,4] broadcast(r_k_f0), dimensions={}
  r_is_zero = pred[128,4] compare(r_decoded, r_b_f0), direction=EQ
  r_k_tiny = f32[] constant(5.87747175e-39)
  r_b_tiny = f32[128,4] broadcast(r_k_tiny), dimensions={}
  r_scale_f32 = f32[128,4] select(r_is_zero, r_b_tiny, r_decoded)
  s_blocks = f32[128,4,32] bitcast(s_f32)
  r_scale_b = f32[128,4,32] broadcast(r_scale_f32), dimensions={0,1}
  r_scaled = f32[128,4,32] divide(s_blocks, r_scale_b)
  r_k_lo = f32[] constant(-448)
  r_b_lo = f32[128,4,32] broadcast(r_k_lo), dimensions={}
  r_k_hi = f32[] constant(448)
  r_b_hi = f32[128,4,32] broadcast(r_k_hi), dimensions={}
  r_clamped = f32[128,4,32] clamp(r_b_lo, r_scaled, r_b_hi)
  r_data = f8e4m3fn[128,4,32] convert(r_clamped)
  abs_view = bf16[128,4,32]{0,2,1} bitcast(abs)
  abs_blocks_c = bf16[4,32,128] bitcast(abs_view)
  amax_c = bf16[4,128] reduce(abs_blocks_c, ninf), dimensions={1}, to_apply=max_bf16
  amax_view = bf16[128,4]{0,1} bitcast(amax_c)
  amax_rows = bf16[4,128] bitcast(amax_view)
  amax_rows_f32 = f32[4,128] convert(amax_rows)
  amax_t = f32[128,4] transpose(amax_rows_f32), dimensions={1,0}
  c_k_rmax = f32[] constant(0.00223214296)
  c_b_rmax = f32[128,4] broadcast(c_k_rmax), dimensions={}
  c_ratio = f32[128,4] multiply(amax_t, c_b_rmax)
  c_bits = u32[128,4] bitcast(c_ratio)
  c_k_mask = u32[] constant(8388607)
  c_b_mask = u32[128,4] broadcast(c_k_mask), dimensions={}
  c_mant = u32[128,4] and(c_bits, c_b_mask)
  c_k_0 = u32[] constant(0)
  c_b_0 = u32[128,4] broadcast(c_k_0), dimensions={}
  c_gt = pred[128,4] compare(c_mant, c_b_0), direction=GT
  c_k_23 = u32[] constant(23)
  c_b_23 = u32[128,4] broadcast(c_k_23), dimensions={}
  c_exp = u32[128,4] shift-right-logical(c_bits, c_b_23)
  c_k_254 = u32[] constant(254)
  c_b_254 = u32[128,4] broadcast(c_k_254), dimensions={}
  c_ne = pred[128,4] compare(c_exp, c_b_254), direction=NE
  c_and0 = pred[128,4] and(c_gt, c_ne)
  c_eq = pred[128,4] compare(c_exp, c_b_0), direction=EQ
  c_k_half = u32[] constant(4194304)
  c_b_half = u32[128,4] broadcast(c_k_half), dimensions={}
  c_le = pred[128,4] compare(c_mant, c_b_half), direction=LE
  c_and1 = pred[128,4] and(c_eq, c_le)
  c_not = pred[128,4] not(c_and1)
  c_round_up = pred[128,4] and(c_and0, c_not)
  c_k_1 = u32[] constant(1)
  c_b_1 = u32[128,4] broadcast(c_k_1), dimensions={}
  c_exp1 = u32[128,4] add(c_exp, c_b_1)
  c_rounded = u32[128,4] select(c_round_up, c_exp1, c_exp)
  c_scale = u8[128,4] convert(c_rounded)
  c_scale_u32 = u32[128,4] convert(c_scale)
  c_shifted = u32[128,4] shift-left(c_scale_u32, c_b_23)
  c_decoded = f32[128,4] bitcast(c_shifted)
  c_k_f0 = f32[] constant(0)
  c_b_f0 = f32[128,4] broadcast(c_k_f0), dimensions={}
  c_is_zero = pred[128,4] compare(c_decoded, c_b_f0), direction=EQ
  c_k_tiny = f32[] constant(5.87747175e-39)
  c_b_tiny = f32[128,4] broadcast(c_k_tiny), dimensions={}
  c_scale_f32 = f32[128,4] select(c_is_zero, c_b_tiny, c_decoded)
  s_rows = f32[4,32,128] bitcast(s_f32)
  s_t = f32[128,4,32] transpose(s_rows), dimensions={2,0,1}
  c_scale_b = f32[128,4,32] broadcast(c_scale_f32), dimensions={0,1}
  c_scaled = f32[128,4,32] divide(s_t, c_scale_b)
  c_k_lo = f32[] constant(-448)
  c_b_lo = f32[128,4,32] broadcast(c_k_lo), dimensions={}
  c_k_hi = f32[] constant(448)
  c_b_hi = f32[128,4,32] broadcast(c_k_hi), dimensions={}
  c_clamped = f32[128,4,32] clamp(c_b_lo, c_scaled, c_b_hi)
  c_data = f8e4m3fn[128,4,32] convert(c_clamped)
  data_r = f8e4m3fn[1,128,128] bitcast(r_data)
  scale_r = f8e8m0fnu[1,128,4] bitcast(r_scale)
  data_c = f8e4m3fn[1,128,128] bitcast(c_data)
  scale_c = f8e8m0fnu[1,128,4] bitcast(c_scale)
  ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4], f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) tuple(data_r, scale_r, data_c, scale_c)
}
)hlo";

constexpr absl::string_view kReducers = R"hlo(
HloModule m

max_bf16 {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT r = bf16[] maximum(a, b)
}

max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] maximum(a, b)
}

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] add(a, b)
}

add_bf16 {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT r = bf16[] add(a, b)
}
)hlo";

// The E8M0 encode and decode of `kRowQuantization` for the f32 `amax`.
std::string Scale(absl::string_view q, absl::string_view dims,
                  absl::string_view amax, absl::string_view reciprocal_max,
                  absl::string_view tiny = "") {
  std::string tiny_lines;
  std::string tiny_name(tiny);
  if (tiny.empty()) {
    tiny_lines = R"(
  $q_k_tiny = f32[] constant(5.87747175e-39)
  $q_b_tiny = f32[$d] broadcast($q_k_tiny), dimensions={})";
    tiny_name = absl::StrCat(q, "_b_tiny");
  }
  return absl::StrReplaceAll(absl::StrCat(R"(
  $q_k_rmax = f32[] constant($rmax)
  $q_b_rmax = f32[$d] broadcast($q_k_rmax), dimensions={}
  $q_ratio = f32[$d] multiply($amax, $q_b_rmax)
  $q_bits = u32[$d] bitcast($q_ratio)
  $q_k_mask = u32[] constant(8388607)
  $q_b_mask = u32[$d] broadcast($q_k_mask), dimensions={}
  $q_mant = u32[$d] and($q_bits, $q_b_mask)
  $q_k_0 = u32[] constant(0)
  $q_b_0 = u32[$d] broadcast($q_k_0), dimensions={}
  $q_gt = pred[$d] compare($q_mant, $q_b_0), direction=GT
  $q_k_23 = u32[] constant(23)
  $q_b_23 = u32[$d] broadcast($q_k_23), dimensions={}
  $q_exp = u32[$d] shift-right-logical($q_bits, $q_b_23)
  $q_k_254 = u32[] constant(254)
  $q_b_254 = u32[$d] broadcast($q_k_254), dimensions={}
  $q_ne = pred[$d] compare($q_exp, $q_b_254), direction=NE
  $q_and0 = pred[$d] and($q_gt, $q_ne)
  $q_eq = pred[$d] compare($q_exp, $q_b_0), direction=EQ
  $q_k_half = u32[] constant(4194304)
  $q_b_half = u32[$d] broadcast($q_k_half), dimensions={}
  $q_le = pred[$d] compare($q_mant, $q_b_half), direction=LE
  $q_and1 = pred[$d] and($q_eq, $q_le)
  $q_not = pred[$d] not($q_and1)
  $q_round_up = pred[$d] and($q_and0, $q_not)
  $q_k_1 = u32[] constant(1)
  $q_b_1 = u32[$d] broadcast($q_k_1), dimensions={}
  $q_exp1 = u32[$d] add($q_exp, $q_b_1)
  $q_rounded = u32[$d] select($q_round_up, $q_exp1, $q_exp)
  $q_scale = u8[$d] convert($q_rounded)
  $q_scale_u32 = u32[$d] convert($q_scale)
  $q_shifted = u32[$d] shift-left($q_scale_u32, $q_b_23)
  $q_decoded = f32[$d] bitcast($q_shifted)
  $q_k_f0 = f32[] constant(0)
  $q_b_f0 = f32[$d] broadcast($q_k_f0), dimensions={}
  $q_is_zero = pred[$d] compare($q_decoded, $q_b_f0), direction=EQ)",
                                          tiny_lines, R"(
  $q_scale_f32 = f32[$d] select($q_is_zero, $tiny, $q_decoded)
)"),
                             {{"$q", q},
                              {"$d", dims},
                              {"$amax", amax},
                              {"$rmax", reciprocal_max},
                              {"$tiny", tiny_name}});
}

// The divide, clamp and convert of `blocks` by the broadcast scale of `q`.
std::string Data(absl::string_view q, absl::string_view dims,
                 absl::string_view blocks, absl::string_view type = "f8e4m3fn",
                 absl::string_view max = "448") {
  return absl::StrReplaceAll(R"(
  $q_scale_b = f32[$d] broadcast($q_scale_f32), dimensions={0,1}
  $q_scaled = f32[$d] divide($blocks, $q_scale_b)
  $q_k_lo = f32[] constant(-$max)
  $q_b_lo = f32[$d] broadcast($q_k_lo), dimensions={}
  $q_k_hi = f32[] constant($max)
  $q_b_hi = f32[$d] broadcast($q_k_hi), dimensions={}
  $q_clamped = f32[$d] clamp($q_b_lo, $q_scaled, $q_b_hi)
  $q_data = $type[$d] convert($q_clamped)
)",
                             {{"$q", q},
                              {"$d", dims},
                              {"$blocks", blocks},
                              {"$type", type},
                              {"$max", max}});
}

constexpr absl::string_view kReciprocalOf448 = "0.00223214296";

constexpr absl::string_view kRowRoot =
    "ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) "
    "tuple(data_out, scale_out)";

// The quantization of `kRowQuantization` applied to bf16[2,64,128] `s`.
std::string RowQuantizationOf(absl::string_view s) {
  return absl::StrCat(absl::StrReplaceAll(R"(
  ninf = bf16[] constant(-inf)
  abs = bf16[2,64,128] abs($s)
  abs_blocks = bf16[128,4,32] bitcast(abs)
  amax = bf16[128,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,4] convert(amax))",
                                          {{"$s", s}}),
                      Scale("q", "128,4", "amax_f32", kReciprocalOf448),
                      absl::StrReplaceAll(R"(
  s_blocks = bf16[128,4,32] bitcast($s)
  s_f32 = f32[128,4,32] convert(s_blocks))",
                                          {{"$s", s}}),
                      Data("q", "128,4,32", "s_f32"), R"(
  data_out = f8e4m3fn[1,128,128] bitcast(q_data)
  scale_out = f8e8m0fnu[1,128,4] bitcast(q_scale)
  )",
                      kRowRoot, "\n");
}

// `kRowQuantization` with `replacements`.
std::string RowQuantizationWith(
    std::initializer_list<std::pair<absl::string_view, absl::string_view>>
        replacements) {
  return absl::StrReplaceAll(kRowQuantization, replacements);
}

HloInstruction* OnlyFusion(const HloModule& module) {
  std::vector<HloInstruction*> fusions;
  for (HloInstruction* instr : module.entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      fusions.push_back(instr);
    }
  }
  return fusions.size() == 1 ? fusions.front() : nullptr;
}

std::vector<HloOpcode> OperandOpcodes(const HloInstruction* instr) {
  std::vector<HloOpcode> opcodes;
  for (const HloInstruction* operand : instr->operands()) {
    opcodes.push_back(operand->opcode());
  }
  return opcodes;
}

int64_t CountOpcode(const HloComputation* computation, HloOpcode opcode) {
  int64_t count = 0;
  for (const HloInstruction* instr : computation->instructions()) {
    count += instr->opcode() == opcode;
  }
  return count;
}

std::vector<std::string> RootShapes(const HloInstruction* fusion) {
  std::vector<std::string> shapes;
  for (const Shape& shape : fusion->shape().tuple_shapes()) {
    shapes.push_back(ShapeUtil::HumanString(shape));
  }
  return shapes;
}

TEST_F(BlockScaledQuantizeFusionTest, FusesRowQuantization) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kRowQuantization));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Bitcast(m::GetTupleElement(
                              m::Fusion(&fusion, m::Parameter(0)), 0)),
                          m::Bitcast(m::GetTupleElement(m::Fusion(), 1)))));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 7);
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kCustom);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(config.fusion_backend_config().kind(), "__block_scaled_quantize");
  EXPECT_THAT(RootShapes(fusion),
              ElementsAre("f8e4m3fn[128,4,32]", "u8[128,4]"));
  EXPECT_EQ(
      CountOpcode(fusion->fused_instructions_computation(), HloOpcode::kReduce),
      1);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesColumnQuantizationTransposeFirst) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          absl::StrCat(kReducers, R"(
ENTRY e {
  w = f32[128,1,128] parameter(0)
  ninf = bf16[] constant(-inf)
  w_bf16 = bf16[128,1,128] convert(w)
  abs = bf16[128,1,128] abs(w_bf16)
  abs_t = bf16[1,128,128] transpose(abs), dimensions={1,2,0}
  abs_blocks = bf16[128,4,32] bitcast(abs_t)
  amax = bf16[128,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,4] convert(amax))",
                       Scale("q", "128,4", "amax_f32", kReciprocalOf448), R"(
  w_f32 = f32[128,1,128] convert(w_bf16)
  w_t = f32[1,128,128] transpose(w_f32), dimensions={1,2,0}
  w_blocks = f32[128,4,32] bitcast(w_t))",
                       Data("q", "128,4,32", "w_blocks"), R"(
  data_out = f8e4m3fn[1,128,128] bitcast(q_data)
  scale_out = f8e8m0fnu[1,128,4] bitcast(q_scale)
  ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) tuple(data_out, scale_out)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion, GmockMatch(m::Fusion(m::Parameter(0))));
  EXPECT_THAT(RootShapes(fusion),
              ElementsAre("f8e4m3fn[128,4,32]", "u8[128,4]"));
  EXPECT_EQ(CountOpcode(fusion->fused_instructions_computation(),
                        HloOpcode::kTranspose),
            2);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 7);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesColumnQuantizationReduceFirst) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(absl::StrCat(
                           kReducers, R"(
ENTRY e {
  s = bf16[2,64,128] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[2,64,128] abs(s)
  abs_view = bf16[128,4,32]{0,2,1} bitcast(abs)
  abs_blocks = bf16[4,32,128] bitcast(abs_view)
  amax = bf16[4,128] reduce(abs_blocks, ninf), dimensions={1}, to_apply=max_bf16
  amax_view = bf16[128,4]{0,1} bitcast(amax)
  amax_rows = bf16[4,128] bitcast(amax_view)
  amax_rows_f32 = f32[4,128] convert(amax_rows)
  amax_t = f32[128,4] transpose(amax_rows_f32), dimensions={1,0})",
                           Scale("c", "128,4", "amax_t", kReciprocalOf448), R"(
  s_f32 = f32[2,64,128] convert(s)
  s_rows = f32[4,32,128] bitcast(s_f32)
  s_t = f32[128,4,32] transpose(s_rows), dimensions={2,0,1})",
                           Data("c", "128,4,32", "s_t"), R"(
  data_out = f8e4m3fn[1,128,128] bitcast(c_data)
  scale_out = f8e8m0fnu[1,128,4] bitcast(c_scale)
  ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) tuple(data_out, scale_out)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion, GmockMatch(m::Fusion(m::Parameter(0))));
  EXPECT_THAT(RootShapes(fusion),
              ElementsAre("f8e4m3fn[128,4,32]", "u8[128,4]"));
  std::vector<std::string> bitcast_layouts;
  for (const HloInstruction* instr : fusion->fused_instructions()) {
    if (instr->opcode() == HloOpcode::kBitcast) {
      bitcast_layouts.push_back(
          LayoutUtil::HumanString(instr->shape().layout()));
    }
  }
  EXPECT_THAT(bitcast_layouts, IsSupersetOf({"{0,2,1}", "{0,1}"}));
}

TEST_F(BlockScaledQuantizeFusionTest, FusesBothOrientationsOfOneSource) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kRowAndColumnQuantization));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Bitcast(m::GetTupleElement(
                              m::Fusion(&fusion, m::Parameter(0)), 0)),
                          m::Bitcast(m::GetTupleElement(m::Fusion(), 2)),
                          m::Bitcast(m::GetTupleElement(m::Fusion(), 1)),
                          m::Bitcast(m::GetTupleElement(m::Fusion(), 3)))));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 11);
  EXPECT_THAT(RootShapes(fusion),
              ElementsAre("f8e4m3fn[128,4,32]", "f8e4m3fn[128,4,32]",
                          "u8[128,4]", "u8[128,4]"));
  const HloInstruction* fused_root = fusion->fused_expression_root();
  // The column data reads the source through a transpose, the row data not.
  EXPECT_THAT(
      fused_root->operand(0),
      GmockMatch(m::Convert(m::Clamp(
          m::Broadcast(),
          m::Divide(m::Bitcast(m::Convert(m::Parameter(0))), m::Broadcast()),
          m::Broadcast()))));
  EXPECT_THAT(
      fused_root->operand(1),
      GmockMatch(m::Convert(m::Clamp(
          m::Broadcast(),
          m::Divide(m::Transpose(m::Bitcast(m::Convert())), m::Broadcast()),
          m::Broadcast()))));
  EXPECT_EQ(
      CountOpcode(fusion->fused_instructions_computation(), HloOpcode::kReduce),
      2);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesBothOrientationsOfWeight) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          absl::StrCat(kReducers, R"(
ENTRY e {
  w = f32[128,1,128] parameter(0)
  ninf = bf16[] constant(-inf)
  w_bf16 = bf16[128,1,128] convert(w)
  abs = bf16[128,1,128] abs(w_bf16)
  abs_blocks_r = bf16[128,4,32] bitcast(abs)
  amax_r = bf16[128,4] reduce(abs_blocks_r, ninf), dimensions={2}, to_apply=max_bf16
  amax_r_f32 = f32[128,4] convert(amax_r)
  abs_t = bf16[1,128,128] transpose(abs), dimensions={1,2,0}
  abs_blocks_c = bf16[128,4,32] bitcast(abs_t)
  amax_c = bf16[128,4] reduce(abs_blocks_c, ninf), dimensions={2}, to_apply=max_bf16
  amax_c_f32 = f32[128,4] convert(amax_c))",
                       Scale("r", "128,4", "amax_r_f32", kReciprocalOf448),
                       Scale("c", "128,4", "amax_c_f32", kReciprocalOf448), R"(
  w_f32 = f32[128,1,128] convert(w_bf16)
  w_blocks = f32[128,4,32] bitcast(w_f32)
  w_t = f32[1,128,128] transpose(w_f32), dimensions={1,2,0}
  w_t_blocks = f32[128,4,32] bitcast(w_t))",
                       Data("r", "128,4,32", "w_blocks"),
                       Data("c", "128,4,32", "w_t_blocks"),
                       R"(
  ROOT out = (u8[128,4], f8e4m3fn[128,4,32], u8[128,4], f8e4m3fn[128,4,32]) tuple(c_scale, c_data, r_scale, r_data)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(m::Parameter(0)), 3),
                          m::GetTupleElement(m::Fusion(), 1),
                          m::GetTupleElement(m::Fusion(), 2),
                          m::GetTupleElement(m::Fusion(), 0))));
  EXPECT_EQ(CountOpcode(module->entry_computation(), HloOpcode::kFusion), 1);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesBothOrientationsThroughTransposes) {
  // dk in the backward body: [b, h, s, d], blocked along d and along s.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          absl::StrCat(kReducers, R"(
ENTRY e {
  s = bf16[2,2,64,128] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[2,2,64,128] abs(s)
  abs_t = bf16[2,64,2,128] transpose(abs), dimensions={0,2,1,3}
  abs_blocks_r = bf16[256,4,32] bitcast(abs_t)
  amax_r = bf16[256,4] reduce(abs_blocks_r, ninf), dimensions={2}, to_apply=max_bf16
  amax_r_f32 = f32[256,4] convert(amax_r)
  abs_5 = bf16[1,2,2,64,128] bitcast(abs)
  abs_c = bf16[1,2,128,2,64] transpose(abs_5), dimensions={0,2,4,1,3}
  abs_blocks_c = bf16[512,2,32] bitcast(abs_c)
  amax_c = bf16[512,2] reduce(abs_blocks_c, ninf), dimensions={2}, to_apply=max_bf16
  amax_c_f32 = f32[512,2] convert(amax_c))",
                       Scale("r", "256,4", "amax_r_f32", kReciprocalOf448),
                       Scale("c", "512,2", "amax_c_f32", kReciprocalOf448), R"(
  s_f32 = f32[2,2,64,128] convert(s)
  s_t = f32[2,64,2,128] transpose(s_f32), dimensions={0,2,1,3}
  s_blocks_r = f32[256,4,32] bitcast(s_t)
  s_5 = f32[1,2,2,64,128] bitcast(s_f32)
  s_c = f32[1,2,128,2,64] transpose(s_5), dimensions={0,2,4,1,3}
  s_blocks_c = f32[512,2,32] bitcast(s_c))",
                       Data("r", "256,4,32", "s_blocks_r"),
                       Data("c", "512,2,32", "s_blocks_c"), R"(
  ROOT out = (f8e4m3fn[256,4,32], u8[256,4], f8e4m3fn[512,2,32], u8[512,2]) tuple(r_data, r_scale, c_data, c_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion, GmockMatch(m::Fusion(m::Parameter(0))));
  EXPECT_THAT(RootShapes(fusion),
              ElementsAre("f8e4m3fn[256,4,32]", "f8e4m3fn[512,2,32]",
                          "u8[256,4]", "u8[512,2]"));
  EXPECT_EQ(CountOpcode(fusion->fused_instructions_computation(),
                        HloOpcode::kTranspose),
            4);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesOrientationsAloneIfPairRefused) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kRowAndColumnQuantization));
  BlockScaledQuantizeFusion pass([](const HloFusionInstruction& fusion) {
    return Decision(fusion.shape().tuple_shapes().size() == 2,
                    "one orientation per fusion");
  });
  EXPECT_THAT(RunHloPass(pass, module.get()), IsOkAndHolds(true));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Bitcast(m::GetTupleElement(m::Fusion(), 0)),
                          m::Bitcast(m::GetTupleElement(m::Fusion(), 1)),
                          m::Bitcast(m::GetTupleElement(m::Fusion(), 0)),
                          m::Bitcast(m::GetTupleElement(m::Fusion(), 1)))));
  EXPECT_EQ(CountOpcode(module->entry_computation(), HloOpcode::kFusion), 2);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesOrientationsAloneIfRowDoesNotTile) {
  // [b, s, h, d]: the row reads runs of 64 along d, the column's view is
  // [64, 128] with blocks along s.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          absl::StrCat(kReducers, R"(
ENTRY e {
  s = bf16[2,64,2,64] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[2,64,2,64] abs(s)
  abs_r = bf16[2,2,64,64] transpose(abs), dimensions={0,2,1,3}
  abs_blocks_r = bf16[256,2,32] bitcast(abs_r)
  amax_r = bf16[256,2] reduce(abs_blocks_r, ninf), dimensions={2}, to_apply=max_bf16
  amax_r_f32 = f32[256,2] convert(amax_r)
  abs_c = bf16[2,2,64,64] transpose(abs), dimensions={0,2,3,1}
  abs_blocks_c = bf16[256,2,32] bitcast(abs_c)
  amax_c = bf16[256,2] reduce(abs_blocks_c, ninf), dimensions={2}, to_apply=max_bf16
  amax_c_f32 = f32[256,2] convert(amax_c))",
                       Scale("r", "256,2", "amax_r_f32", kReciprocalOf448),
                       Scale("c", "256,2", "amax_c_f32", kReciprocalOf448), R"(
  s_f32 = f32[2,64,2,64] convert(s)
  s_r = f32[2,2,64,64] transpose(s_f32), dimensions={0,2,1,3}
  s_blocks_r = f32[256,2,32] bitcast(s_r)
  s_c = f32[2,2,64,64] transpose(s_f32), dimensions={0,2,3,1}
  s_blocks_c = f32[256,2,32] bitcast(s_c))",
                       Data("r", "256,2,32", "s_blocks_r"),
                       Data("c", "256,2,32", "s_blocks_c"), R"(
  ROOT out = (f8e4m3fn[256,2,32], u8[256,2], f8e4m3fn[256,2,32], u8[256,2]) tuple(r_data, r_scale, c_data, c_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  EXPECT_EQ(CountOpcode(module->entry_computation(), HloOpcode::kFusion), 2);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesBothOrientationsOfHeadsWeight) {
  // Wq: [embed, heads, head_dim], blocked along heads x head_dim and along
  // embed, whose stride spans two dimensions.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          absl::StrCat(kReducers, R"(
ENTRY e {
  w = f32[1,128,4,32] parameter(0)
  ninf = bf16[] constant(-inf)
  w_bf16 = bf16[1,128,4,32] convert(w)
  abs = bf16[1,128,4,32] abs(w_bf16)
  abs_blocks_r = bf16[128,4,32] bitcast(abs)
  amax_r = bf16[128,4] reduce(abs_blocks_r, ninf), dimensions={2}, to_apply=max_bf16
  amax_r_f32 = f32[128,4] convert(amax_r)
  abs_view = bf16[128,4,32,1] bitcast(abs)
  abs_t = bf16[4,32,128,1] transpose(abs_view), dimensions={1,2,0,3}
  abs_blocks_c = bf16[128,4,32] bitcast(abs_t)
  amax_c = bf16[128,4] reduce(abs_blocks_c, ninf), dimensions={2}, to_apply=max_bf16
  amax_c_f32 = f32[128,4] convert(amax_c))",
                       Scale("r", "128,4", "amax_r_f32", kReciprocalOf448),
                       Scale("c", "128,4", "amax_c_f32", kReciprocalOf448), R"(
  w_f32 = f32[1,128,4,32] convert(w_bf16)
  w_blocks = f32[128,4,32] bitcast(w_f32)
  w_view = f32[128,4,32,1] bitcast(w_f32)
  w_t = f32[4,32,128,1] transpose(w_view), dimensions={1,2,0,3}
  w_t_blocks = f32[128,4,32] bitcast(w_t))",
                       Data("r", "128,4,32", "w_blocks"),
                       Data("c", "128,4,32", "w_t_blocks"),
                       R"(
  ROOT out = (f8e4m3fn[128,4,32], u8[128,4], f8e4m3fn[128,4,32], u8[128,4]) tuple(r_data, r_scale, c_data, c_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(m::Parameter(0)), 0),
                          m::GetTupleElement(m::Fusion(), 2),
                          m::GetTupleElement(m::Fusion(), 1),
                          m::GetTupleElement(m::Fusion(), 3))));
  EXPECT_EQ(CountOpcode(module->entry_computation(), HloOpcode::kFusion), 1);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesOrientationsAloneIfViewNotOf64) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(absl::StrCat(
          kReducers, R"(
ENTRY e {
  w = f32[96,1,128] parameter(0)
  ninf = bf16[] constant(-inf)
  w_bf16 = bf16[96,1,128] convert(w)
  abs = bf16[96,1,128] abs(w_bf16)
  abs_blocks_r = bf16[96,4,32] bitcast(abs)
  amax_r = bf16[96,4] reduce(abs_blocks_r, ninf), dimensions={2}, to_apply=max_bf16
  amax_r_f32 = f32[96,4] convert(amax_r)
  abs_t = bf16[1,128,96] transpose(abs), dimensions={1,2,0}
  abs_blocks_c = bf16[128,3,32] bitcast(abs_t)
  amax_c = bf16[128,3] reduce(abs_blocks_c, ninf), dimensions={2}, to_apply=max_bf16
  amax_c_f32 = f32[128,3] convert(amax_c))",
          Scale("r", "96,4", "amax_r_f32", kReciprocalOf448),
          Scale("c", "128,3", "amax_c_f32", kReciprocalOf448), R"(
  w_f32 = f32[96,1,128] convert(w_bf16)
  w_blocks = f32[96,4,32] bitcast(w_f32)
  w_t = f32[1,128,96] transpose(w_f32), dimensions={1,2,0}
  w_t_blocks = f32[128,3,32] bitcast(w_t))",
          Data("r", "96,4,32", "w_blocks"), Data("c", "128,3,32", "w_t_blocks"),
          R"(
  ROOT out = (f8e4m3fn[96,4,32], u8[96,4], f8e4m3fn[128,3,32], u8[128,3]) tuple(r_data, r_scale, c_data, c_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  EXPECT_EQ(CountOpcode(module->entry_computation(), HloOpcode::kFusion), 2);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesFp4Quantization) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(absl::StrCat(
          kReducers, R"(
ENTRY e {
  s = bf16[128,128] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[128,128] abs(s)
  abs_blocks = bf16[128,4,32] bitcast(abs)
  amax = bf16[128,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,4] convert(amax))",
          Scale("q", "128,4", "amax_f32", "0.166666672"), R"(
  s_blocks = bf16[128,4,32] bitcast(s)
  s_f32 = f32[128,4,32] convert(s_blocks))",
          absl::StrReplaceAll(
              Data("q", "128,4,32", "s_f32", "f4e2m1fn", "6"),
              {{"f4e2m1fn[128,4,32]", "f4e2m1fn[128,4,32]{2,1,0:E(4)}"}}),
          R"(
  data_out = f4e2m1fn[1,128,128]{2,1,0:E(4)} bitcast(q_data)
  scale_out = f8e8m0fnu[1,128,4] bitcast(q_scale)
  ROOT out = (f4e2m1fn[1,128,128]{2,1,0:E(4)}, f8e8m0fnu[1,128,4]) tuple(data_out, scale_out)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  ASSERT_OK_AND_ASSIGN(Shape packed_data,
                       ParseShape("f4e2m1fn[128,4,32]{2,1,0:E(4)}"));
  EXPECT_TRUE(ShapeUtil::Equal(fusion->shape().tuple_shapes(0), packed_data))
      << ShapeUtil::HumanStringWithLayout(fusion->shape().tuple_shapes(0));
  EXPECT_EQ(fusion->shape().tuple_shapes(1).element_type(), U8);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 7);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesF8e5m2Quantization) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(RowQuantizationWith(
                           {{"f8e4m3fn", "f8e5m2"},
                            {"constant(-448)", "constant(-57344)"},
                            {"constant(448)", "constant(57344)"},
                            {"0.00223214296", "1.74386169e-05"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_EQ(fusion->shape().tuple_shapes(0).element_type(), F8E5M2);
}

TEST_F(BlockScaledQuantizeFusionTest, FusesQuantizationOfF32Source) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(absl::StrCat(
                           kReducers, R"(
ENTRY e {
  s = f32[128,128] parameter(0)
  ninf = f32[] constant(-inf)
  abs = f32[128,128] abs(s)
  abs_blocks = f32[128,4,32] bitcast(abs)
  amax = f32[128,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_f32)",
                           Scale("q", "128,4", "amax", kReciprocalOf448), R"(
  s_blocks = f32[128,4,32] bitcast(s))",
                           Data("q", "128,4,32", "s_blocks"), R"(
  data_out = f8e4m3fn[128,128] bitcast(q_data)
  ROOT out = (f8e4m3fn[128,128], u8[128,4]) tuple(data_out, q_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion, GmockMatch(m::Fusion(m::Parameter(0))));
  EXPECT_THAT(RootShapes(fusion),
              ElementsAre("f8e4m3fn[128,4,32]", "u8[128,4]"));
}

TEST_F(BlockScaledQuantizeFusionTest, FusesBlockSize16) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          absl::StrCat(kReducers, R"(
ENTRY e {
  s = bf16[128,128] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[128,128] abs(s)
  abs_blocks = bf16[128,8,16] bitcast(abs)
  amax = bf16[128,8] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,8] convert(amax))",
                       Scale("q", "128,8", "amax_f32", kReciprocalOf448), R"(
  s_blocks = bf16[128,8,16] bitcast(s)
  s_f32 = f32[128,8,16] convert(s_blocks))",
                       Data("q", "128,8,16", "s_f32"), R"(
  data_out = f8e4m3fn[128,128] bitcast(q_data)
  ROOT out = (f8e4m3fn[128,128], u8[128,8]) tuple(data_out, q_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(RootShapes(fusion),
              ElementsAre("f8e4m3fn[128,8,16]", "u8[128,8]"));
}

TEST_F(BlockScaledQuantizeFusionTest, AbsorbsSiluTimesMul) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(
                           absl::StrCat(kReducers, R"(
ENTRY e {
  a = bf16[2,64,128] parameter(0)
  b = bf16[2,64,128] parameter(1)
  one = bf16[] constant(1)
  ones = bf16[2,64,128] broadcast(one), dimensions={}
  neg = bf16[2,64,128] negate(a)
  exp = bf16[2,64,128] exponential(neg)
  denominator = bf16[2,64,128] add(exp, ones)
  sigmoid = bf16[2,64,128] divide(ones, denominator)
  silu = bf16[2,64,128] multiply(a, sigmoid)
  h = bf16[2,64,128] multiply(silu, b))",
                                        RowQuantizationOf("h"), "}\n")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion, GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(
      CountOpcode(fusion->fused_instructions_computation(), HloOpcode::kExp),
      1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 8);
}

TEST_F(BlockScaledQuantizeFusionTest, AbsorbsSiluBackwardDuplicatingSigmoid) {
  // d_a in the backward body; silu(a) is also used outside.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(absl::StrReplaceAll(
          absl::StrCat(kReducers, R"(
ENTRY e {
  a = bf16[2,64,128] parameter(0)
  b = bf16[2,64,128] parameter(1)
  dh = bf16[2,64,128] parameter(2)
  one = bf16[] constant(1)
  ones = bf16[2,64,128] broadcast(one), dimensions={}
  neg = bf16[2,64,128] negate(a)
  exp = bf16[2,64,128] exponential(neg)
  denominator = bf16[2,64,128] add(exp, ones)
  sigmoid = bf16[2,64,128] divide(ones, denominator)
  silu = bf16[2,64,128] multiply(a, sigmoid)
  dh_b = bf16[2,64,128] multiply(b, dh)
  t3 = bf16[2,64,128] multiply(dh_b, sigmoid)
  t4 = bf16[2,64,128] multiply(a, dh_b)
  one_minus = bf16[2,64,128] subtract(ones, sigmoid)
  t2 = bf16[2,64,128] multiply(sigmoid, one_minus)
  t5 = bf16[2,64,128] multiply(t4, t2)
  s = bf16[2,64,128] add(t3, t5))",
                       RowQuantizationOf("s"), "}\n"),
          {{"ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) "
            "tuple(data_out, scale_out)",
            "ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4], "
            "bf16[2,64,128]) tuple(data_out, scale_out, silu)"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  const HloComputation* entry = module->entry_computation();
  EXPECT_THAT(fusion->operands(),
              UnorderedElementsAre(entry->parameter_instruction(0),
                                   entry->parameter_instruction(1),
                                   entry->parameter_instruction(2)));
  EXPECT_EQ(
      CountOpcode(fusion->fused_instructions_computation(), HloOpcode::kExp),
      1);
  EXPECT_EQ(CountOpcode(entry, HloOpcode::kExp), 1);
}

TEST_F(BlockScaledQuantizeFusionTest, DoesNotAbsorbBehindDuplicatedProducer) {
  // `p` is duplicated, so absorbing `q` would read x and y while `q` stays.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(absl::StrReplaceAll(
          absl::StrCat(kReducers, R"(
ENTRY e {
  x = bf16[2,64,128] parameter(0)
  y = bf16[2,64,128] parameter(1)
  q = bf16[2,64,128] add(x, y)
  p = f32[2,64,128] convert(q)
  n = f32[2,64,128] negate(p)
  s = bf16[2,64,128] convert(n))",
                       RowQuantizationOf("s"), "}\n"),
          {{"ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) "
            "tuple(data_out, scale_out)",
            "ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4], "
            "f32[2,64,128]) tuple(data_out, scale_out, p)"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion,
              GmockMatch(m::Fusion(m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(BlockScaledQuantizeFusionTest, KeepsSourceThatDuplicatingReadsMore) {
  // dX in the backward body: the residual sum is also used outside.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(absl::StrReplaceAll(
          absl::StrCat(kReducers, R"(
ENTRY e {
  x = bf16[2,64,128] parameter(0)
  g = f32[2,64,128] parameter(1)
  g_bf16 = bf16[2,64,128] convert(g)
  s = bf16[2,64,128] add(x, g_bf16))",
                       RowQuantizationOf("s"), "}\n"),
          {{"ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) "
            "tuple(data_out, scale_out)",
            "ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4], "
            "bf16[2,64,128]) tuple(data_out, scale_out, s)"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion, GmockMatch(m::Fusion(m::Add())));
}

TEST_F(BlockScaledQuantizeFusionTest, AbsorbsRmsNormApply) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(absl::StrCat(
          kReducers, R"(
ENTRY e {
  x = bf16[2,64,128] parameter(0)
  w = f32[128] parameter(1)
  tiny = f32[128,4] parameter(2)
  x_f32 = f32[2,64,128] convert(x)
  square = f32[2,64,128] multiply(x_f32, x_f32)
  zero = f32[] constant(0)
  sum = f32[2,64] reduce(square, zero), dimensions={2}, to_apply=add_f32
  inv_n = f32[] constant(0.0078125)
  inv_n_b = f32[2,64] broadcast(inv_n), dimensions={}
  mean = f32[2,64] multiply(sum, inv_n_b)
  eps = f32[] constant(1e-05)
  eps_b = f32[2,64] broadcast(eps), dimensions={}
  mean_eps = f32[2,64] add(mean, eps_b)
  rsqrt = f32[2,64] rsqrt(mean_eps)
  rsqrt_b = f32[2,64,128] broadcast(rsqrt), dimensions={0,1}
  normed = f32[2,64,128] multiply(x_f32, rsqrt_b)
  normed_bf16 = bf16[2,64,128] convert(normed)
  w_bf16 = bf16[128] convert(w)
  w_b = bf16[2,64,128] broadcast(w_bf16), dimensions={2}
  s = bf16[2,64,128] multiply(w_b, normed_bf16)
  ninf = bf16[] constant(-inf)
  abs = bf16[2,64,128] abs(s)
  abs_blocks = bf16[128,4,32] bitcast(abs)
  amax = bf16[128,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,4] convert(amax))",
          Scale("q", "128,4", "amax_f32", kReciprocalOf448, "tiny"), R"(
  s_blocks = bf16[128,4,32] bitcast(s)
  s_f32 = f32[128,4,32] convert(s_blocks))",
          Data("q", "128,4,32", "s_f32"), R"(
  data_out = f8e4m3fn[1,128,128] bitcast(q_data)
  scale_out = f8e8m0fnu[1,128,4] bitcast(q_scale)
  ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) tuple(data_out, scale_out)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(
      OperandOpcodes(fusion),
      UnorderedElementsAre(HloOpcode::kParameter, HloOpcode::kRsqrt,
                           HloOpcode::kParameter, HloOpcode::kParameter));
  // The per-row statistic stays outside: no rsqrt, mean or epsilon inside.
  const HloComputation* fused = fusion->fused_instructions_computation();
  EXPECT_EQ(CountOpcode(fused, HloOpcode::kRsqrt), 0);
  for (const HloInstruction* instr : fused->instructions()) {
    if (instr->opcode() != HloOpcode::kParameter) {
      EXPECT_NE(ShapeUtil::HumanString(instr->shape()), "f32[2,64]")
          << instr->ToString();
    }
  }
  // The f32 copy of x is duplicated: the sum of squares still reads it.
  EXPECT_EQ(module->entry_computation()->instruction_count(), 20);
}

TEST_F(BlockScaledQuantizeFusionTest, AbsorbsDataMovementBelowBroadcast) {
  // The norm weight: a dynamic-slice, convert and bitcast under a broadcast.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(
                           absl::StrCat(kReducers, R"(
ENTRY e {
  x = bf16[2,64,128] parameter(0)
  stacked = f32[4,128] parameter(1)
  i = s32[] parameter(2)
  zero = s32[] constant(0)
  w = f32[1,128] dynamic-slice(stacked, i, zero), dynamic_slice_sizes={1,128}
  w_bf16 = bf16[1,128] convert(w)
  w_row = bf16[128] bitcast(w_bf16)
  w_b = bf16[2,64,128] broadcast(w_row), dimensions={2}
  s = bf16[2,64,128] multiply(x, w_b))",
                                        RowQuantizationOf("s"), "}\n")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  const HloComputation* entry = module->entry_computation();
  EXPECT_THAT(fusion->operands(),
              UnorderedElementsAre(entry->parameter_instruction(0),
                                   entry->parameter_instruction(1),
                                   entry->parameter_instruction(2)));
  EXPECT_EQ(CountOpcode(fusion->fused_instructions_computation(),
                        HloOpcode::kDynamicSlice),
            1);
}

TEST_F(BlockScaledQuantizeFusionTest, LeavesStridedDynamicSliceOutside) {
  // The stacked norm weights with the layer most minor: a slice reads 1 of 4.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(
                           absl::StrCat(kReducers, R"(
ENTRY e {
  x = bf16[2,64,128] parameter(0)
  stacked = f32[4,128]{0,1} parameter(1)
  i = s32[] parameter(2)
  zero = s32[] constant(0)
  table = f32[128,4]{1,0} bitcast(stacked)
  w = f32[128,1] dynamic-slice(table, zero, i), dynamic_slice_sizes={128,1}
  w_bf16 = bf16[128,1] convert(w)
  w_row = bf16[128] bitcast(w_bf16)
  w_b = bf16[2,64,128] broadcast(w_row), dimensions={2}
  s = bf16[2,64,128] multiply(x, w_b))",
                                        RowQuantizationOf("s"), "}\n")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(
      OperandOpcodes(fusion),
      UnorderedElementsAre(HloOpcode::kParameter, HloOpcode::kDynamicSlice));
  EXPECT_EQ(CountOpcode(fusion->fused_instructions_computation(),
                        HloOpcode::kDynamicSlice),
            0);
}

TEST_F(BlockScaledQuantizeFusionTest, AbsorbsDynamicSliceAndConvert) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          absl::StrCat(kReducers, R"(
ENTRY e {
  stacked = f32[4,128,1,128] parameter(0)
  i = s32[] parameter(1)
  zero = s32[] constant(0)
  w = f32[1,128,1,128] dynamic-slice(stacked, i, zero, zero, zero), dynamic_slice_sizes={1,128,1,128}
  w_bf16 = bf16[1,128,1,128] convert(w)
  ninf = bf16[] constant(-inf)
  abs = bf16[1,128,1,128] abs(w_bf16)
  abs_blocks = bf16[128,4,32] bitcast(abs)
  amax = bf16[128,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,4] convert(amax))",
                       Scale("q", "128,4", "amax_f32", kReciprocalOf448), R"(
  w_blocks = bf16[128,4,32] bitcast(w_bf16)
  w_f32 = f32[128,4,32] convert(w_blocks))",
                       Data("q", "128,4,32", "w_f32"), R"(
  data_out = f8e4m3fn[1,128,128] bitcast(q_data)
  scale_out = f8e8m0fnu[1,128,4] bitcast(q_scale)
  ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) tuple(data_out, scale_out)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion, GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(CountOpcode(fusion->fused_instructions_computation(),
                        HloOpcode::kDynamicSlice),
            1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 8);
}

TEST_F(BlockScaledQuantizeFusionTest, ReadsTinyScaleFromLoopCarry) {
  // In the forward body the 2^-127 tensor is a loop-carried value.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(absl::StrCat(
          kReducers, R"(
ENTRY e {
  s = bf16[2,64,128] parameter(0)
  carry = (f32[1,128,4,1]) parameter(1)
  carry_tiny = f32[1,128,4,1] get-tuple-element(carry), index=0
  tiny = f32[128,4] bitcast(carry_tiny)
  ninf = bf16[] constant(-inf)
  abs = bf16[2,64,128] abs(s)
  abs_blocks = bf16[128,4,32] bitcast(abs)
  amax = bf16[128,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,4] convert(amax))",
          Scale("q", "128,4", "amax_f32", kReciprocalOf448, "tiny"), R"(
  s_blocks = bf16[128,4,32] bitcast(s)
  s_f32 = f32[128,4,32] convert(s_blocks))",
          Data("q", "128,4,32", "s_f32"), R"(
  ROOT out = (f8e4m3fn[128,4,32], u8[128,4]) tuple(q_data, q_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion, GmockMatch(m::Fusion(
                          m::Parameter(0),
                          m::Bitcast(m::GetTupleElement(m::Parameter(1))))));
}

// A row quantization in a while body whose 2^-127 operand is loop state.
std::string QuantizationInLoop(absl::string_view next_tiny) {
  constexpr absl::string_view kState =
      "(s32[], bf16[2,64,128], f32[1,128,4,1], f8e4m3fn[128,4,32], u8[128,4])";
  return absl::StrReplaceAll(
      absl::StrCat(kReducers, R"(
cond {
  p = $state parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(2)
  ROOT lt = pred[] compare(i, n), direction=LT
}

body {
  p = $state parameter(0)
  i = s32[] get-tuple-element(p), index=0
  one = s32[] constant(1)
  next_i = s32[] add(i, one)
  s = bf16[2,64,128] get-tuple-element(p), index=1
  carry_tiny = f32[1,128,4,1] get-tuple-element(p), index=2
  tiny = f32[128,4] bitcast(carry_tiny)
  ninf = bf16[] constant(-inf)
  abs = bf16[2,64,128] abs(s)
  abs_blocks = bf16[128,4,32] bitcast(abs)
  amax = bf16[128,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,4] convert(amax))",
                   Scale("q", "128,4", "amax_f32", kReciprocalOf448, "tiny"),
                   R"(
  s_blocks = bf16[128,4,32] bitcast(s)
  s_f32 = f32[128,4,32] convert(s_blocks))",
                   Data("q", "128,4,32", "s_f32"), R"(
  ROOT next = $state tuple(next_i, s, $next_tiny, q_data, q_scale)
}

ENTRY e {
  s = bf16[2,64,128] parameter(0)
  zero = s32[] constant(0)
  c = f32[] constant(5.87747175e-39)
  tiny = f32[1,128,4,1] broadcast(c), dimensions={}
  f8_zero = f8e4m3fn[] constant(0)
  data = f8e4m3fn[128,4,32] broadcast(f8_zero), dimensions={}
  u8_zero = u8[] constant(0)
  scale = u8[128,4] broadcast(u8_zero), dimensions={}
  init = $state tuple(zero, s, tiny, data, scale)
  ROOT loop = $state while(init), condition=cond, body=body
})"),
      {{"$state", kState}, {"$next_tiny", next_tiny}});
}

int64_t FusionOperandCount(const HloModule& module) {
  for (const HloInstruction* instr :
       module.GetComputationWithName("body")->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      return instr->operand_count();
    }
  }
  return -1;
}

TEST_F(BlockScaledQuantizeFusionTest, ClonesLoopInvariantTinyScale) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(QuantizationInLoop("carry_tiny")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  EXPECT_EQ(FusionOperandCount(*module), 1);
}

TEST_F(BlockScaledQuantizeFusionTest, ReadsLoopVariantTinyScale) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                           QuantizationInLoop("next_tiny"),
                           {{"  ROOT next =",
                             "  next_tiny = f32[1,128,4,1] negate(carry_tiny)\n"
                             "  ROOT next ="}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  EXPECT_EQ(FusionOperandCount(*module), 2);
}

TEST_F(BlockScaledQuantizeFusionTest, ExportsPredicateAfterScales) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(RowQuantizationWith(
          {{kRowRoot,
            "ROOT out = (pred[128,4], f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4]) "
            "tuple(q_is_zero, data_out, scale_out)"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(RootShapes(fusion),
              ElementsAre("f8e4m3fn[128,4,32]", "u8[128,4]", "pred[128,4]"));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(), 2),
                          m::Bitcast(m::GetTupleElement(m::Fusion(), 0)),
                          m::Bitcast(m::GetTupleElement(m::Fusion(), 1)))));
}

TEST_F(BlockScaledQuantizeFusionTest, LeavesRopeBackwardOutside) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(
                           absl::StrCat(kReducers, R"(
ENTRY e {
  g = bf16[2,4,64,32] parameter(0)
  y = bf16[2,4,64,32] parameter(1)
  g_t = bf16[2,64,4,32] transpose(g), dimensions={0,2,1,3}
  t = bf16[2,64,128] bitcast(g_t)
  hi = bf16[2,64,64] slice(t), slice={[0:2], [0:64], [64:128]}
  lo = bf16[2,64,64] slice(t), slice={[0:2], [0:64], [0:64]}
  neg = bf16[2,64,64] negate(lo)
  rotated = bf16[2,64,128] concatenate(hi, neg), dimensions={2}
  y_t = bf16[2,64,4,32] transpose(y), dimensions={0,2,1,3}
  u = bf16[2,64,128] bitcast(y_t)
  s = bf16[2,64,128] add(rotated, u))",
                                        RowQuantizationOf("s"), "}\n")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(true));

  const HloInstruction* fusion = OnlyFusion(*module);
  ASSERT_NE(fusion, nullptr);
  EXPECT_THAT(fusion, GmockMatch(m::Fusion(m::Add(
                          m::Concatenate(), m::Bitcast(m::Transpose())))));
  EXPECT_EQ(CountOpcode(module->entry_computation(), HloOpcode::kTranspose), 2);
  EXPECT_EQ(CountOpcode(fusion->fused_instructions_computation(),
                        HloOpcode::kTranspose),
            0);
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresAddReduction) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(absl::StrCat(absl::StrReplaceAll(
          kRowQuantization, {{"ENTRY", R"hlo(
add_bf16 {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT r = bf16[] add(a, b)
}

ENTRY)hlo"},
                             {"to_apply=max_bf16", "to_apply=add_bf16"}}))));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresNanInit) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(RowQuantizationWith(
                           {{"constant(-inf)", "constant(nan)"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresMaxWithoutAbs) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(RowQuantizationWith(
                           {{"abs = bf16[2,64,128] abs(s)",
                             "abs = bf16[2,64,128] negate(s)"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresBlockSize8) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          absl::StrCat(kReducers, R"(
ENTRY e {
  s = bf16[128,128] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[128,128] abs(s)
  abs_blocks = bf16[128,16,8] bitcast(abs)
  amax = bf16[128,16] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[128,16] convert(amax))",
                       Scale("q", "128,16", "amax_f32", kReciprocalOf448), R"(
  s_blocks = bf16[128,16,8] bitcast(s)
  s_f32 = f32[128,16,8] convert(s_blocks))",
                       Data("q", "128,16,8", "s_f32"), R"(
  data_out = f8e4m3fn[128,128] bitcast(q_data)
  ROOT out = (f8e4m3fn[128,128], u8[128,16]) tuple(data_out, q_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresBlockCrossingDimension) {
  // The second block of 32 along the transposed 48 wraps into the next column.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(absl::StrCat(
                           kReducers, R"(
ENTRY e {
  s = f32[2,48,32] parameter(0)
  ninf = f32[] constant(-inf)
  t = f32[2,32,48] transpose(s), dimensions={0,2,1}
  abs = f32[2,32,48] abs(t)
  abs_blocks = f32[96,32] bitcast(abs)
  amax = f32[96] reduce(abs_blocks, ninf), dimensions={1}, to_apply=max_f32)",
                           Scale("q", "96", "amax", kReciprocalOf448), R"(
  s_t = f32[2,32,48] transpose(s), dimensions={0,2,1}
  s_blocks = f32[96,32] bitcast(s_t)
  scale_b = f32[96,32] broadcast(q_scale_f32), dimensions={0}
  scaled = f32[96,32] divide(s_blocks, scale_b)
  lo = f32[] constant(-448)
  lo_b = f32[96,32] broadcast(lo), dimensions={}
  hi = f32[] constant(448)
  hi_b = f32[96,32] broadcast(hi), dimensions={}
  clamped = f32[96,32] clamp(lo_b, scaled, hi_b)
  data = f8e4m3fn[96,32] convert(clamped)
  ROOT out = (f8e4m3fn[96,32], u8[96]) tuple(data, q_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresDataValueUsedOutside) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(RowQuantizationWith(
          {{kRowRoot,
            "ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4], "
            "f32[128,4,32]) tuple(data_out, scale_out, q_clamped)"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresBroadcastScaleUsedOutside) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(RowQuantizationWith(
          {{kRowRoot,
            "ROOT out = (f8e4m3fn[1,128,128], f8e8m0fnu[1,128,4], "
            "f32[128,4,32]) tuple(data_out, scale_out, q_scale_b)"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresScaleReadingNonParameter) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(RowQuantizationWith(
          {{"q_b_tiny = f32[128,4] broadcast(q_k_tiny), dimensions={}",
            "q_b_tiny_half = f32[128,4] broadcast(q_k_tiny), dimensions={}\n"
            "  q_b_tiny = f32[128,4] add(q_b_tiny_half, q_b_tiny_half)"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresControlDependency) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(RowQuantizationWith(
                           {{"amax_f32 = f32[128,4] convert(amax)",
                             "amax_f32 = f32[128,4] convert(amax), "
                             "control-predecessors={abs}"}})));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresDynamicShape) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(
          absl::StrCat(kReducers, R"(
ENTRY e {
  s = bf16[<=128,4,32] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[<=128,4,32] abs(s)
  amax = bf16[<=128,4] reduce(abs, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[<=128,4] convert(amax))",
                       Scale("q", "<=128,4", "amax_f32", kReciprocalOf448), R"(
  s_f32 = f32[<=128,4,32] convert(s))",
                       Data("q", "<=128,4,32", "s_f32"), R"(
  ROOT out = (f8e4m3fn[<=128,4,32], u8[<=128,4]) tuple(q_data, q_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, IgnoresRowsNotMultipleOf32) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(absl::StrCat(
                           kReducers, R"(
ENTRY e {
  s = bf16[48,128] parameter(0)
  ninf = bf16[] constant(-inf)
  abs = bf16[48,128] abs(s)
  abs_blocks = bf16[48,4,32] bitcast(abs)
  amax = bf16[48,4] reduce(abs_blocks, ninf), dimensions={2}, to_apply=max_bf16
  amax_f32 = f32[48,4] convert(amax))",
                           Scale("q", "48,4", "amax_f32", kReciprocalOf448), R"(
  s_blocks = bf16[48,4,32] bitcast(s)
  s_f32 = f32[48,4,32] convert(s_blocks))",
                           Data("q", "48,4,32", "s_f32"), R"(
  data_out = f8e4m3fn[48,128] bitcast(q_data)
  ROOT out = (f8e4m3fn[48,128], u8[48,4]) tuple(data_out, q_scale)
})")));
  EXPECT_THAT(RunHloPass(AcceptingPass(), module.get()), IsOkAndHolds(false));
}

TEST_F(BlockScaledQuantizeFusionTest, UndoesFusionThatCannotBeEmitted) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kRowQuantization));
  BlockScaledQuantizeFusion pass([](const HloFusionInstruction&) {
    return Decision::Forbid("not supported");
  });
  EXPECT_THAT(RunHloPass(pass, module.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::gpu
