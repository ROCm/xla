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

#include "xla/backends/gpu/codegen/emitters/block_scaled_quantize_analysis.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using Orientation = BlockScaledQuantization::Orientation;

constexpr absl::string_view kMax = R"(
max_bf16 {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT m = bf16[] maximum(a, b)
}

add_bf16 {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT m = bf16[] add(a, b)
}

max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}

max_bf16_in_f32 {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  a_f32 = f32[] convert(a)
  b_f32 = f32[] convert(b)
  m = f32[] maximum(a_f32, b_f32)
  ROOT m_bf16 = bf16[] convert(m)
}

max_rounded_to_bf16 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  a_bf16 = bf16[] convert(a)
  a_rounded = f32[] convert(a_bf16)
  b_bf16 = bf16[] convert(b)
  b_rounded = f32[] convert(b_bf16)
  m = f32[] maximum(a_rounded, b_rounded)
  m_bf16 = bf16[] convert(m)
  ROOT m_rounded = f32[] convert(m_bf16)
}

max_rounding_one_operand {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  a_bf16 = bf16[] convert(a)
  a_rounded = f32[] convert(a_bf16)
  ROOT m = f32[] maximum(a_rounded, b)
}
)";

// JAX's E8M0 encode and decode of `$p_amax_f32`, as XLA leaves them after
// layout normalization: `$p_scale` is the u8 scale, `$p_decoded` the f32 power
// of two the data is divided by.
std::string ScaleRegion(absl::string_view prefix, absl::string_view shape,
                        absl::string_view inv_max) {
  return absl::StrReplaceAll(
      R"(
  $p_inv = f32[] constant($inv)
  $p_inv_b = f32[$s] broadcast($p_inv), dimensions={}
  $p_ratio = f32[$s] multiply($p_amax_f32, $p_inv_b)
  $p_bits = u32[$s] bitcast($p_ratio)
  $p_c_mant = u32[] constant(8388607)
  $p_mant_mask = u32[$s] broadcast($p_c_mant), dimensions={}
  $p_mant = u32[$s] and($p_bits, $p_mant_mask)
  $p_c0 = u32[] constant(0)
  $p_zero = u32[$s] broadcast($p_c0), dimensions={}
  $p_gt = pred[$s] compare($p_mant, $p_zero), direction=GT
  $p_c23 = u32[] constant(23)
  $p_s23 = u32[$s] broadcast($p_c23), dimensions={}
  $p_exp = u32[$s] shift-right-logical($p_bits, $p_s23)
  $p_c254 = u32[] constant(254)
  $p_e254 = u32[$s] broadcast($p_c254), dimensions={}
  $p_ne = pred[$s] compare($p_exp, $p_e254), direction=NE
  $p_and0 = pred[$s] and($p_gt, $p_ne)
  $p_eq0 = pred[$s] compare($p_exp, $p_zero), direction=EQ
  $p_c_half = u32[] constant(4194304)
  $p_half = u32[$s] broadcast($p_c_half), dimensions={}
  $p_le = pred[$s] compare($p_mant, $p_half), direction=LE
  $p_and1 = pred[$s] and($p_eq0, $p_le)
  $p_not = pred[$s] not($p_and1)
  $p_round_up = pred[$s] and($p_and0, $p_not)
  $p_c1 = u32[] constant(1)
  $p_one = u32[$s] broadcast($p_c1), dimensions={}
  $p_exp1 = u32[$s] add($p_exp, $p_one)
  $p_exp_sel = u32[$s] select($p_round_up, $p_exp1, $p_exp)
  $p_scale = u8[$s] convert($p_exp_sel)
  $p_scale32 = u32[$s] convert($p_scale)
  $p_shl = u32[$s] shift-left($p_scale32, $p_s23)
  $p_pow = f32[$s] bitcast($p_shl)
  $p_cf0 = f32[] constant(0)
  $p_fzero = f32[$s] broadcast($p_cf0), dimensions={}
  $p_is_zero = pred[$s] compare($p_pow, $p_fzero), direction=EQ
  $p_tiny = f32[] constant(5.87747175e-39)
  $p_tiny_b = f32[$s] broadcast($p_tiny), dimensions={}
  $p_decoded = f32[$s] select($p_is_zero, $p_tiny_b, $p_pow)
)",
      {{"$p", prefix}, {"$s", shape}, {"$inv", inv_max}});
}

// Divides `$p_x` by the broadcast `$p_decoded`, clamps and converts.
std::string DataRegion(absl::string_view prefix, absl::string_view shape,
                       absl::string_view type, absl::string_view max) {
  return absl::StrReplaceAll(
      R"(
  $p_bs = f32[$s] broadcast($p_decoded), dimensions={0,1}
  $p_q = f32[$s] divide($p_x, $p_bs)
  $p_lo = f32[] constant(-$max)
  $p_lo_b = f32[$s] broadcast($p_lo), dimensions={}
  $p_hi = f32[] constant($max)
  $p_hi_b = f32[$s] broadcast($p_hi), dimensions={}
  $p_cl = f32[$s] clamp($p_lo_b, $p_q, $p_hi_b)
  $p_data = $t[$s] convert($p_cl)
)",
      {{"$p", prefix}, {"$s", shape}, {"$t", type}, {"$max", max}});
}

std::string Module(absl::string_view body) {
  return absl::StrCat("HloModule m\n", kMax, "\nfused_quantize {\n", body,
                      "\n}\n");
}

// Row blocks of bf16[2,64,256], the form of the step 0 dump.
std::string RowBody(absl::string_view type, absl::string_view max,
                    absl::string_view inv_max) {
  return absl::StrCat(R"(
  p0 = bf16[2,64,256] parameter(0)
  xf = f32[2,64,256] convert(p0)
  r_x = f32[128,8,32] bitcast(xf)
  abs = bf16[2,64,256] abs(p0)
  ab = bf16[128,8,32] bitcast(abs)
  ninf = bf16[] constant(-inf)
  r_amax = bf16[128,8] reduce(ab, ninf), dimensions={2}, to_apply=max_bf16
  r_amax_f32 = f32[128,8] convert(r_amax)
)",
                      ScaleRegion("r", "128,8", inv_max),
                      DataRegion("r", "128,8,32", type, max));
}

// Column blocks of bf16[128,256] in the normalized form of the step 0 dump:
// the reduce runs over dimension 1 of [4,32,256] and its result is transposed.
constexpr absl::string_view kColumnNormalized = R"(
  xf = f32[128,256] convert(p0)
  c_xb = f32[4,32,256] bitcast(xf)
  c_x = f32[256,4,32] transpose(c_xb), dimensions={2,0,1}
  abs = bf16[128,256] abs(p0)
  c_ab0 = bf16[256,4,32]{0,2,1} bitcast(abs)
  c_ab = bf16[4,32,256] bitcast(c_ab0)
  ninf = bf16[] constant(-inf)
  c_amax = bf16[4,256] reduce(c_ab, ninf), dimensions={1}, to_apply=max_bf16
  c_at0 = bf16[256,4]{0,1} bitcast(c_amax)
  c_at1 = bf16[4,256] bitcast(c_at0)
  c_af = f32[4,256] convert(c_at1)
  c_amax_f32 = f32[256,4] transpose(c_af), dimensions={1,0}
)";

// Row blocks of bf16[m, n] `p0`, with `xf` and `abs` defined by the caller.
std::string RowOf(int64_t m, int64_t n) {
  return absl::StrReplaceAll(
             R"(
  r_x = f32[$M,$NB,32] bitcast(xf)
  r_ab = bf16[$M,$NB,32] bitcast(abs)
  r_amax = bf16[$M,$NB] reduce(r_ab, ninf), dimensions={2}, to_apply=max_bf16
  r_amax_f32 = f32[$M,$NB] convert(r_amax)
)",
             {{"$M", absl::StrCat(m)}, {"$NB", absl::StrCat(n / 32)}}) +
         ScaleRegion("r", absl::StrCat(m, ",", n / 32), "0.00223214296") +
         DataRegion("r", absl::StrCat(m, ",", n / 32, ",32"), "f8e4m3fn",
                    "448");
}

// Column blocks of bf16[m, n] `p0` in the normalized form, defining `xf`, `abs`
// and `ninf`.
std::string ColumnOf(int64_t m, int64_t n) {
  return absl::StrReplaceAll(R"(
  xf = f32[$M,$N] convert(p0)
  c_xb = f32[$MB,32,$N] bitcast(xf)
  c_x = f32[$N,$MB,32] transpose(c_xb), dimensions={2,0,1}
  abs = bf16[$M,$N] abs(p0)
  c_ab0 = bf16[$N,$MB,32]{0,2,1} bitcast(abs)
  c_ab = bf16[$MB,32,$N] bitcast(c_ab0)
  ninf = bf16[] constant(-inf)
  c_amax = bf16[$MB,$N] reduce(c_ab, ninf), dimensions={1}, to_apply=max_bf16
  c_at0 = bf16[$N,$MB]{0,1} bitcast(c_amax)
  c_at1 = bf16[$MB,$N] bitcast(c_at0)
  c_af = f32[$MB,$N] convert(c_at1)
  c_amax_f32 = f32[$N,$MB] transpose(c_af), dimensions={1,0}
)",
                             {{"$MB", absl::StrCat(m / 32)},
                              {"$M", absl::StrCat(m)},
                              {"$N", absl::StrCat(n)}}) +
         ScaleRegion("c", absl::StrCat(n, ",", m / 32), "0.00223214296") +
         DataRegion("c", absl::StrCat(n, ",", m / 32, ",32"), "f8e4m3fn",
                    "448");
}

std::string PairOf(int64_t m, int64_t n) {
  return absl::StrCat("  p0 = bf16[", m, ",", n, "] parameter(0)",
                      ColumnOf(m, n), RowOf(m, n), "  ROOT t = (f8e4m3fn[", m,
                      ",", n / 32, ",32], f8e4m3fn[", n, ",", m / 32,
                      ",32], u8[", m, ",", n / 32, "], u8[", n, ",", m / 32,
                      "]) tuple(r_data, c_data, r_scale, c_scale)");
}

class BlockScaledQuantizeAnalysisTest : public IndexingTestBase {
 protected:
  absl::StatusOr<BlockScaledQuantizeAnalysis> Analyze(absl::string_view hlo) {
    ABSL_ASSIGN_OR_RETURN(module_, ParseAndReturnVerifiedModule(hlo));
    return AnalyzeBlockScaledQuantize(
        *module_->GetComputationWithName("fused_quantize"), &mlir_context_);
  }

  std::vector<int64_t> Evaluate(const IndexingMap& map,
                                absl::Span<const int64_t> point) {
    llvm::SmallVector<int64_t, 4> result =
        map.Evaluate(CreateSymbolicConstantExprs(
                         llvm::ArrayRef<int64_t>(point.data(), point.size()),
                         &mlir_context_),
                     {});
    return {result.begin(), result.end()};
  }
};

TEST_F(BlockScaledQuantizeAnalysisTest, RowFp8) {
  ASSERT_OK_AND_ASSIGN(
      BlockScaledQuantizeAnalysis analysis,
      Analyze(Module(absl::StrCat(
          RowBody("f8e4m3fn", "448", "0.00223214296"),
          "  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, "
          "r_scale)"))));
  EXPECT_EQ(analysis.source->name(), "p0");
  EXPECT_EQ(analysis.rows, 128);
  EXPECT_EQ(analysis.cols, 256);
  EXPECT_EQ(analysis.block_size, 32);
  EXPECT_THAT(analysis.prologue_parameters, ElementsAre(analysis.source));
  ASSERT_EQ(analysis.quantizations.size(), 1);
  const BlockScaledQuantization& row = analysis.quantizations[0];
  EXPECT_EQ(row.orientation, Orientation::kRow);
  EXPECT_EQ(row.amax->name(), "r_amax");
  EXPECT_EQ(row.data_type, F8E4M3FN);
  EXPECT_THAT(row.data_root_indices, ElementsAre(0));
  EXPECT_THAT(row.scale_root_indices, ElementsAre(1));
  ASSERT_EQ(row.broadcast_scales.size(), 1);
  EXPECT_EQ(row.broadcast_scales[0]->name(), "r_decoded");
  // Row 70, column 100 is element 4 of block 3 of row 70.
  EXPECT_THAT(Evaluate(analysis.source_indexing, {70, 100}),
              ElementsAre(1, 6, 100));
  EXPECT_THAT(Evaluate(row.data_root_indexing[0], {70, 100}),
              ElementsAre(70, 3, 4));
  EXPECT_THAT(Evaluate(row.block_to_amax, {70, 3}), ElementsAre(70, 3));
  EXPECT_THAT(Evaluate(row.scale_root_indexing[0], {70, 3}),
              ElementsAre(70, 3));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RowFp4) {
  ASSERT_OK_AND_ASSIGN(
      BlockScaledQuantizeAnalysis analysis,
      Analyze(Module(
          absl::StrCat(RowBody("f4e2m1fn", "6", "0.166666672"),
                       "  ROOT t = (f4e2m1fn[128,8,32]{2,1,0:E(4)}, u8[128,8]) "
                       "tuple(r_data, r_scale)"))));
  ASSERT_EQ(analysis.quantizations.size(), 1);
  EXPECT_EQ(analysis.quantizations[0].orientation, Orientation::kRow);
  EXPECT_EQ(analysis.quantizations[0].data_type, F4E2M1FN);
}

// In the forward body the 2^-127 of the decode is a loop-carried tensor, so it
// reaches the scale region as a parameter.
TEST_F(BlockScaledQuantizeAnalysisTest, RowWithScaleOperand) {
  std::string body = RowBody("f8e4m3fn", "448", "0.00223214296");
  absl::StrReplaceAll(
      {{"r_tiny_b = f32[128,8] broadcast(r_tiny), dimensions={}",
        "r_tiny_b = f32[128,8] parameter(1)"}},
      &body);
  ASSERT_OK_AND_ASSIGN(
      BlockScaledQuantizeAnalysis analysis,
      Analyze(Module(absl::StrCat(
          body,
          "  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, "
          "r_scale)"))));
  EXPECT_THAT(analysis.prologue_parameters, ElementsAre(analysis.source));
  ASSERT_EQ(analysis.quantizations.size(), 1);
  EXPECT_EQ(analysis.quantizations[0].orientation, Orientation::kRow);
}

TEST_F(BlockScaledQuantizeAnalysisTest, RowWithPrologue) {
  ASSERT_OK_AND_ASSIGN(BlockScaledQuantizeAnalysis analysis,
                       Analyze(Module(absl::StrCat(
                           R"(
  x = bf16[128,256] parameter(0)
  rsqrt = f32[128] parameter(1)
  w = bf16[256] parameter(2)
  x_f32 = f32[128,256] convert(x)
  rsqrt_b = f32[128,256] broadcast(rsqrt), dimensions={0}
  normed = f32[128,256] multiply(x_f32, rsqrt_b)
  normed_bf16 = bf16[128,256] convert(normed)
  w_b = bf16[128,256] broadcast(w), dimensions={1}
  p0 = bf16[128,256] multiply(normed_bf16, w_b)
  xf = f32[128,256] convert(p0)
  r_x = f32[128,8,32] bitcast(xf)
  abs = bf16[128,256] abs(p0)
  ab = bf16[128,8,32] bitcast(abs)
  ninf = bf16[] constant(-inf)
  r_amax = bf16[128,8] reduce(ab, ninf), dimensions={2}, to_apply=max_bf16
  r_amax_f32 = f32[128,8] convert(r_amax)
)",
                           ScaleRegion("r", "128,8", "0.00223214296"),
                           DataRegion("r", "128,8,32", "f8e4m3fn", "448"),
                           R"(
  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, r_scale))"))));
  EXPECT_EQ(analysis.source->name(), "p0");
  ASSERT_EQ(analysis.prologue_parameters.size(), 3);
  EXPECT_EQ(analysis.prologue_parameters[0]->name(), "x");
  EXPECT_EQ(analysis.prologue_parameters[1]->name(), "rsqrt");
  EXPECT_EQ(analysis.prologue_parameters[2]->name(), "w");
  ASSERT_EQ(analysis.quantizations.size(), 1);
  EXPECT_EQ(analysis.quantizations[0].orientation, Orientation::kRow);
}

// The form the emitter sees in the full pipeline, after float normalization
// and SimplifyFPConversions have run on the fused computation (today's
// `fused_computation.1131` and `fused_reduce.7`): the bf16 multiply, abs and
// max reduce are f32 ops behind converts.
TEST_F(BlockScaledQuantizeAnalysisTest, RowAfterFloatNormalization) {
  ASSERT_OK_AND_ASSIGN(
      BlockScaledQuantizeAnalysis analysis,
      Analyze(absl::StrCat(R"(
HloModule m

max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}

fused_quantize {
  x = bf16[128,256] parameter(0)
  rsqrt = f32[128] parameter(1)
  w = bf16[256] parameter(2)
  x_f32 = f32[128,256] convert(x)
  rsqrt_b = f32[128,256] broadcast(rsqrt), dimensions={0}
  normed = f32[128,256] multiply(x_f32, rsqrt_b)
  w_f32 = f32[256] convert(w)
  w_b = f32[128,256] broadcast(w_f32), dimensions={1}
  p0 = f32[128,256] multiply(normed, w_b)
  abs = f32[128,256] abs(p0)
  abs_bf16 = bf16[128,256] convert(abs)
  ab = bf16[128,8,32] bitcast(abs_bf16)
  ab_f32 = f32[128,8,32] convert(ab)
  ninf = bf16[] constant(-inf)
  ninf_f32 = f32[] convert(ninf)
  r_amax_f32 = f32[128,8] reduce(ab_f32, ninf_f32), dimensions={2},
      to_apply=max_f32
  p0_bf16 = bf16[128,256] convert(p0)
  x_blocks = bf16[128,8,32] bitcast(p0_bf16)
  r_x = f32[128,8,32] convert(x_blocks)
)",
                           ScaleRegion("r", "128,8", "0.00223214296"),
                           DataRegion("r", "128,8,32", "f8e4m3fn", "448"),
                           R"(
  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, r_scale)
})")));
  EXPECT_EQ(analysis.source->name(), "p0");
  EXPECT_EQ(analysis.prologue_parameters.size(), 3);
  ASSERT_EQ(analysis.quantizations.size(), 1);
  EXPECT_EQ(analysis.quantizations[0].orientation, Orientation::kRow);
  EXPECT_EQ(analysis.quantizations[0].amax->name(), "r_amax_f32");
}

TEST_F(BlockScaledQuantizeAnalysisTest, ColumnNormalized) {
  ASSERT_OK_AND_ASSIGN(
      BlockScaledQuantizeAnalysis analysis,
      Analyze(Module(absl::StrCat(
          "  p0 = bf16[128,256] parameter(0)", kColumnNormalized,
          ScaleRegion("c", "256,4", "0.00223214296"),
          DataRegion("c", "256,4,32", "f8e4m3fn", "448"),
          "  ROOT t = (f8e4m3fn[256,4,32], u8[256,4]) tuple(c_data, "
          "c_scale)"))));
  EXPECT_EQ(analysis.rows, 128);
  EXPECT_EQ(analysis.cols, 256);
  ASSERT_EQ(analysis.quantizations.size(), 1);
  const BlockScaledQuantization& column = analysis.quantizations[0];
  EXPECT_EQ(column.orientation, Orientation::kColumn);
  // Row 70, column 100 is element 6 of block 2 of column 100.
  EXPECT_THAT(Evaluate(column.data_root_indexing[0], {70, 100}),
              ElementsAre(100, 2, 6));
  EXPECT_THAT(Evaluate(column.block_to_amax, {2, 100}), ElementsAre(2, 100));
  EXPECT_THAT(Evaluate(column.scale_root_indexing[0], {2, 100}),
              ElementsAre(100, 2));
}

TEST_F(BlockScaledQuantizeAnalysisTest, ColumnTransposedOperand) {
  ASSERT_OK_AND_ASSIGN(BlockScaledQuantizeAnalysis analysis,
                       Analyze(Module(absl::StrCat(
                           R"(
  p0 = bf16[128,256] parameter(0)
  xf = f32[128,256] convert(p0)
  c_xb = f32[4,32,256] bitcast(xf)
  c_x = f32[256,4,32] transpose(c_xb), dimensions={2,0,1}
  abs = bf16[128,256] abs(p0)
  ab = bf16[4,32,256] bitcast(abs)
  at = bf16[256,4,32] transpose(ab), dimensions={2,0,1}
  ninf = bf16[] constant(-inf)
  c_amax = bf16[256,4] reduce(at, ninf), dimensions={2}, to_apply=max_bf16
  c_amax_f32 = f32[256,4] convert(c_amax)
)",
                           ScaleRegion("c", "256,4", "0.00223214296"),
                           DataRegion("c", "256,4,32", "f8e4m3fn", "448"),
                           R"(
  ROOT t = (f8e4m3fn[256,4,32], u8[256,4]) tuple(c_data, c_scale))"))));
  ASSERT_EQ(analysis.quantizations.size(), 1);
  const BlockScaledQuantization& column = analysis.quantizations[0];
  EXPECT_EQ(column.orientation, Orientation::kColumn);
  EXPECT_THAT(Evaluate(column.block_to_amax, {2, 100}), ElementsAre(100, 2));
  EXPECT_THAT(Evaluate(column.scale_root_indexing[0], {100, 2}),
              ElementsAre(100, 2));
}

TEST_F(BlockScaledQuantizeAnalysisTest, BothOrientations) {
  ASSERT_OK_AND_ASSIGN(
      BlockScaledQuantizeAnalysis analysis,
      Analyze(Module(absl::StrCat(
          "  p0 = bf16[128,256] parameter(0)", kColumnNormalized, R"(
  r_x = f32[128,8,32] bitcast(xf)
  r_ab = bf16[128,8,32] bitcast(abs)
  r_amax = bf16[128,8] reduce(r_ab, ninf), dimensions={2}, to_apply=max_bf16
  r_amax_f32 = f32[128,8] convert(r_amax)
)",
          ScaleRegion("r", "128,8", "0.00223214296"),
          DataRegion("r", "128,8,32", "f8e4m3fn", "448"),
          ScaleRegion("c", "256,4", "0.00223214296"),
          DataRegion("c", "256,4,32", "f8e4m3fn", "448"),
          R"(
  ROOT t = (f8e4m3fn[128,8,32], f8e4m3fn[256,4,32], u8[128,8], u8[256,4])
      tuple(r_data, c_data, r_scale, c_scale))"))));
  ASSERT_EQ(analysis.quantizations.size(), 2);
  EXPECT_EQ(analysis.row(), &analysis.quantizations[0]);
  EXPECT_EQ(analysis.column(), &analysis.quantizations[1]);
  EXPECT_THAT(analysis.row()->data_root_indices, ElementsAre(0));
  EXPECT_THAT(analysis.row()->scale_root_indices, ElementsAre(2));
  EXPECT_THAT(analysis.column()->data_root_indices, ElementsAre(1));
  EXPECT_THAT(analysis.column()->scale_root_indices, ElementsAre(3));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesTwoTransposesOnTheWalk) {
  EXPECT_THAT(Analyze(Module(
                  absl::StrCat(R"(
  p0 = bf16[128,256] parameter(0)
  xf = f32[128,256] convert(p0)
  c_xb = f32[4,32,256] bitcast(xf)
  c_x = f32[256,4,32] transpose(c_xb), dimensions={2,0,1}
  abs = bf16[128,256] abs(p0)
  ab = bf16[4,32,256] bitcast(abs)
  at0 = bf16[4,256,32] transpose(ab), dimensions={0,2,1}
  at = bf16[256,4,32] transpose(at0), dimensions={1,0,2}
  ninf = bf16[] constant(-inf)
  c_amax = bf16[256,4] reduce(at, ninf), dimensions={2}, to_apply=max_bf16
  c_amax_f32 = f32[256,4] convert(c_amax)
)",
                               ScaleRegion("c", "256,4", "0.00223214296"),
                               DataRegion("c", "256,4,32", "f8e4m3fn", "448"),
                               R"(
  ROOT t = (f8e4m3fn[256,4,32], u8[256,4]) tuple(c_data, c_scale))"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("has 2 transposes")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesNonMaxReduce) {
  std::string body = RowBody("f8e4m3fn", "448", "0.00223214296");
  absl::StrReplaceAll({{"to_apply=max_bf16", "to_apply=add_bf16"}}, &body);
  EXPECT_THAT(Analyze(Module(absl::StrCat(
                  body,
                  "  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, "
                  "r_scale)"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is not a max reduce")));
}

// Float normalization without excess precision keeps the bf16 max in f32
// between converts.
TEST_F(BlockScaledQuantizeAnalysisTest, RowWithMaxInF32) {
  std::string body = RowBody("f8e4m3fn", "448", "0.00223214296");
  absl::StrReplaceAll({{"to_apply=max_bf16", "to_apply=max_bf16_in_f32"}},
                      &body);
  ASSERT_OK_AND_ASSIGN(
      BlockScaledQuantizeAnalysis analysis,
      Analyze(Module(absl::StrCat(
          body,
          "  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, "
          "r_scale)"))));
  ASSERT_EQ(analysis.quantizations.size(), 1);
  EXPECT_EQ(analysis.quantizations[0].amax->to_apply()->name(),
            "max_bf16_in_f32");
}

// Without excess precision float normalization widens the reduce to f32 and
// rounds the reducer's operands and result to bf16.
std::string RowWithF32Reduce(absl::string_view reducer) {
  std::string body = RowBody("f8e4m3fn", "448", "0.00223214296");
  absl::StrReplaceAll(
      {{"  r_amax = bf16[128,8] reduce(ab, ninf), dimensions={2}, "
        "to_apply=max_bf16",
        absl::StrCat("  ab_f32 = f32[128,8,32] convert(ab)\n"
                     "  ninf_f32 = f32[] convert(ninf)\n"
                     "  r_amax_wide = f32[128,8] reduce(ab_f32, ninf_f32), "
                     "dimensions={2}, to_apply=",
                     reducer,
                     "\n  r_amax = bf16[128,8] convert(r_amax_wide)")}},
      &body);
  return absl::StrCat(
      body,
      "  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, r_scale)");
}

TEST_F(BlockScaledQuantizeAnalysisTest, RowWithMaxRoundedToBf16) {
  ASSERT_OK_AND_ASSIGN(
      BlockScaledQuantizeAnalysis analysis,
      Analyze(Module(RowWithF32Reduce("max_rounded_to_bf16"))));
  ASSERT_EQ(analysis.quantizations.size(), 1);
  EXPECT_EQ(analysis.quantizations[0].amax->name(), "r_amax_wide");
}

// Rounding one operand only does not commute.
TEST_F(BlockScaledQuantizeAnalysisTest, RefusesMaxRoundingOneOperand) {
  EXPECT_THAT(Analyze(Module(RowWithF32Reduce("max_rounding_one_operand"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is not a max reduce")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesRootOutsideTheRegions) {
  EXPECT_THAT(Analyze(Module(absl::StrCat(
                  RowBody("f8e4m3fn", "448", "0.00223214296"),
                  "  ROOT t = (f8e4m3fn[128,8,32], u8[128,8], f32[2,64,256]) "
                  "tuple(r_data, r_scale, xf)"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("the output xf is outside the scale and data "
                                 "regions")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesBlockOfEight) {
  EXPECT_THAT(Analyze(Module(
                  absl::StrCat(R"(
  p0 = bf16[2,64,256] parameter(0)
  xf = f32[2,64,256] convert(p0)
  r_x = f32[128,32,8] bitcast(xf)
  abs = bf16[2,64,256] abs(p0)
  ab = bf16[128,32,8] bitcast(abs)
  ninf = bf16[] constant(-inf)
  r_amax = bf16[128,32] reduce(ab, ninf), dimensions={2}, to_apply=max_bf16
  r_amax_f32 = f32[128,32] convert(r_amax)
)",
                               ScaleRegion("r", "128,32", "0.00223214296"),
                               DataRegion("r", "128,32,8", "f8e4m3fn", "448"),
                               R"(
  ROOT t = (f8e4m3fn[128,32,8], u8[128,32]) tuple(r_data, r_scale))"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("has block size 8")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesBlockOf16) {
  std::string body = RowBody("f8e4m3fn", "448", "0.00223214296");
  absl::StrReplaceAll({{"[128,8,32]", "[128,16,16]"}, {"[128,8]", "[128,16]"}},
                      &body);
  EXPECT_THAT(Analyze(Module(absl::StrCat(
                  body,
                  "  ROOT t = (f8e4m3fn[128,16,16], u8[128,16]) tuple(r_data, "
                  "r_scale)"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("has block size 16")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesF8e5m2Data) {
  EXPECT_THAT(Analyze(Module(absl::StrCat(
                  RowBody("f8e5m2", "57344", "1.74386169e-05"),
                  "  ROOT t = (f8e5m2[128,8,32], u8[128,8]) tuple(r_data, "
                  "r_scale)"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is not a convert to F8E4M3FN or F4E2M1FN")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesF16Source) {
  std::string hlo = Module(absl::StrCat(
      RowBody("f8e4m3fn", "448", "0.00223214296"),
      "  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, r_scale)"));
  absl::StrReplaceAll({{"bf16", "f16"}}, &hlo);
  EXPECT_THAT(Analyze(hlo), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("unsupported source")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesScaleOfAnotherBlock) {
  std::string body = absl::StrCat(
      "  p0 = bf16[32,1024] parameter(0)\n  xf = f32[32,1024] convert(p0)\n"
      "  abs = bf16[32,1024] abs(p0)\n  ninf = bf16[] constant(-inf)",
      RowOf(32, 1024),
      "  ROOT t = (f8e4m3fn[32,32,32], u8[32,32]) tuple(r_data, r_scale)");
  absl::StrReplaceAll(
      {{"  r_bs = f32[32,32,32] broadcast(r_decoded),",
        "  r_decoded_t = f32[32,32] transpose(r_decoded), dimensions={1,0}\n"
        "  r_bs = f32[32,32,32] broadcast(r_decoded_t),"}},
      &body);
  EXPECT_THAT(Analyze(Module(body)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("reads the scale r_decoded_t of another "
                                 "block")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesScaleMixedWithBroadcast) {
  std::string body = absl::StrCat(
      RowBody("f8e4m3fn", "448", "0.00223214296"),
      "  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, r_scale)");
  absl::StrReplaceAll(
      {{"  r_ratio = f32[128,8] multiply(r_amax_f32, r_inv_b)",
        "  r_identity = f32[128,8] broadcast(r_amax_f32), dimensions={0,1}\n"
        "  r_mixed = f32[128,8] add(r_amax_f32, r_identity)\n"
        "  r_ratio = f32[128,8] multiply(r_mixed, r_inv_b)"}},
      &body);
  EXPECT_THAT(Analyze(Module(body)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("r_mixed mixes the scale region with the data "
                                 "region")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesScalesOfTwoAmaxes) {
  std::string body = PairOf(128, 128);
  absl::StrReplaceAll(
      {{"  r_cl = f32[128,4,32] clamp(r_lo_b, r_q, r_hi_b)",
        "  r_extra = f32[128,4,32] broadcast(c_decoded), dimensions={0,1}\n"
        "  r_both = f32[128,4,32] multiply(r_q, r_extra)\n"
        "  r_cl = f32[128,4,32] clamp(r_lo_b, r_both, r_hi_b)"}},
      &body);
  EXPECT_THAT(Analyze(Module(body)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("reads the scales of two amaxes")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesPairNotMultipleOf64) {
  EXPECT_THAT(Analyze(Module(PairOf(96, 256))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not a multiple of [64, 64]")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesDataReadingThePrologue) {
  std::string body = absl::StrCat(R"(
  x = bf16[128,256] parameter(0)
  rsqrt = f32[128] parameter(1)
  x_f32 = f32[128,256] convert(x)
  rsqrt_b = f32[128,256] broadcast(rsqrt), dimensions={0}
  normed = f32[128,256] multiply(x_f32, rsqrt_b)
  p0 = bf16[128,256] convert(normed)
  xf = f32[128,256] convert(p0)
  abs = bf16[128,256] abs(p0)
  ninf = bf16[] constant(-inf)
  x_blocks = f32[128,8,32] bitcast(x_f32)
)",
                                  RowOf(128, 256), R"(
  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, r_scale))");
  absl::StrReplaceAll(
      {{"  r_cl = f32[128,8,32] clamp(r_lo_b, r_q, r_hi_b)",
        "  r_both = f32[128,8,32] multiply(r_q, x_blocks)\n"
        "  r_cl = f32[128,8,32] clamp(r_lo_b, r_both, r_hi_b)"}},
      &body);
  EXPECT_THAT(Analyze(Module(body)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("reads the prologue value x_f32")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesMissingScaleOutput) {
  EXPECT_THAT(Analyze(Module(absl::StrCat(
                  RowBody("f8e4m3fn", "448", "0.00223214296"),
                  "  ROOT t = (f8e4m3fn[128,8,32]) tuple(r_data)"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("lacks a data or a scale output")));
}

// The pair of `PairOf` after float normalization: the bf16 abs and max reduce
// are f32 ops, and the converts around them are folded.
TEST_F(BlockScaledQuantizeAnalysisTest, BothAfterFloatNormalization) {
  std::string body = PairOf(128, 256);
  absl::StrReplaceAll(
      {{"abs = bf16[128,256] abs(p0)",
        "abs_f32 = f32[128,256] abs(xf)\n  abs = bf16[128,256] "
        "convert(abs_f32)"},
       {"r_amax = bf16[128,8] reduce(r_ab, ninf), dimensions={2}, "
        "to_apply=max_bf16\n  r_amax_f32 = f32[128,8] convert(r_amax)",
        "r_ab_f32 = f32[128,8,32] convert(r_ab)\n  ninf_f32 = f32[] "
        "convert(ninf)\n  r_amax_f32 = f32[128,8] reduce(r_ab_f32, "
        "ninf_f32), dimensions={2}, to_apply=max_f32"},
       {"c_amax = bf16[4,256] reduce(c_ab, ninf), dimensions={1}, "
        "to_apply=max_bf16\n  c_at0 = bf16[256,4]{0,1} bitcast(c_amax)\n  "
        "c_at1 = bf16[4,256] bitcast(c_at0)\n  c_af = f32[4,256] "
        "convert(c_at1)",
        "c_ab_f32 = f32[4,32,256] convert(c_ab)\n  c_ninf_f32 = f32[] "
        "convert(ninf)\n  c_amax = f32[4,256] reduce(c_ab_f32, c_ninf_f32), "
        "dimensions={1}, to_apply=max_f32\n  c_at0 = f32[256,4]{0,1} "
        "bitcast(c_amax)\n  c_af = f32[4,256] bitcast(c_at0)"}},
      &body);
  ASSERT_OK_AND_ASSIGN(BlockScaledQuantizeAnalysis analysis,
                       Analyze(Module(body)));
  ASSERT_EQ(analysis.quantizations.size(), 2);
  EXPECT_EQ(analysis.row()->amax->name(), "r_amax_f32");
  EXPECT_EQ(analysis.column()->amax->name(), "c_amax");
}

// A column alone after float normalization.
TEST_F(BlockScaledQuantizeAnalysisTest, ColumnAfterFloatNormalization) {
  std::string body = absl::StrCat(
      "  p0 = bf16[128,256] parameter(0)", ColumnOf(128, 256),
      "  ROOT t = (f8e4m3fn[256,4,32], u8[256,4]) tuple(c_data, c_scale)");
  absl::StrReplaceAll(
      {{"abs = bf16[128,256] abs(p0)",
        "abs_f32 = f32[128,256] abs(xf)\n  abs = bf16[128,256] "
        "convert(abs_f32)"},
       {"c_amax = bf16[4,256] reduce(c_ab, ninf), dimensions={1}, "
        "to_apply=max_bf16\n  c_at0 = bf16[256,4]{0,1} bitcast(c_amax)\n  "
        "c_at1 = bf16[4,256] bitcast(c_at0)\n  c_af = f32[4,256] "
        "convert(c_at1)",
        "c_ab_f32 = f32[4,32,256] convert(c_ab)\n  c_ninf_f32 = f32[] "
        "convert(ninf)\n  c_amax = f32[4,256] reduce(c_ab_f32, c_ninf_f32), "
        "dimensions={1}, to_apply=max_f32\n  c_at0 = f32[256,4]{0,1} "
        "bitcast(c_amax)\n  c_af = f32[4,256] bitcast(c_at0)"}},
      &body);
  ASSERT_OK_AND_ASSIGN(BlockScaledQuantizeAnalysis analysis,
                       Analyze(Module(body)));
  ASSERT_EQ(analysis.quantizations.size(), 1);
  EXPECT_EQ(analysis.quantizations[0].orientation, Orientation::kColumn);
}

// A bf16 data region after float normalization: the divide by the broadcast
// bf16 scale reads it through a widening convert.
TEST_F(BlockScaledQuantizeAnalysisTest, RowWithBf16DataRegion) {
  std::string body = absl::StrCat(
      "  p0 = bf16[128,256] parameter(0)\n  xf = f32[128,256] convert(p0)\n"
      "  abs = bf16[128,256] abs(p0)\n  ninf = bf16[] constant(-inf)",
      RowOf(128, 256), R"(
  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, r_scale))");
  absl::StrReplaceAll(
      {{"  r_x = f32[128,8,32] bitcast(xf)\n", ""},
       {"  xf = f32[128,256] convert(p0)\n", ""},
       {"  r_bs = f32[128,8,32] broadcast(r_decoded), dimensions={0,1}\n"
        "  r_q = f32[128,8,32] divide(r_x, r_bs)",
        "  r_dec_bf16 = bf16[128,8] convert(r_decoded)\n"
        "  r_bs_bf16 = bf16[128,8,32] broadcast(r_dec_bf16), dimensions={0,1}\n"
        "  r_bs_f32 = f32[128,8,32] convert(r_bs_bf16)\n"
        "  r_xs = bf16[128,8,32] bitcast(p0)\n"
        "  r_x32 = f32[128,8,32] convert(r_xs)\n"
        "  r_q = f32[128,8,32] divide(r_x32, r_bs_f32)"}},
      &body);
  ASSERT_OK_AND_ASSIGN(BlockScaledQuantizeAnalysis analysis,
                       Analyze(Module(body)));
  ASSERT_EQ(analysis.quantizations.size(), 1);
  EXPECT_THAT(
      analysis.quantizations[0].broadcast_scales,
      ElementsAre(::testing::Property(&HloInstruction::name, "r_dec_bf16")));
}

// The u8[rows, cols] scale `name` in the hipBLASLt 32x8 order, as the swizzle
// folding pass leaves it in the fused computation: `name`_sw.
std::string Swizzled(absl::string_view name, int64_t rows, int64_t cols) {
  return absl::StrReplaceAll(
      R"(
  $n_split = u8[$R32,2,16,$C8,2,4] reshape($n)
  $n_perm = u8[$R32,$C8,4,16,2,2] transpose($n_split), dimensions={0,3,5,2,4,1}
  $n_sw = u8[$R,$C] reshape($n_perm)
)",
      {{"$n", name},
       {"$R32", absl::StrCat(rows / 32)},
       {"$C8", absl::StrCat(cols / 8)},
       {"$R", absl::StrCat(rows)},
       {"$C", absl::StrCat(cols)}});
}

// Where element (row, col) of a u8[rows, cols] scale lands in the 32x8 order.
std::vector<int64_t> SwizzledPosition(int64_t rows, int64_t cols, int64_t row,
                                      int64_t col) {
  const int64_t split[] = {row / 32,     row % 32 / 16, row % 16,
                           col / 8,      col % 8 / 4,   col % 4};
  const int64_t permutation[] = {0, 3, 5, 2, 4, 1};
  const int64_t sizes[] = {rows / 32, cols / 8, 4, 16, 2, 2};
  int64_t linear = 0;
  for (int64_t i = 0; i < 6; ++i) {
    linear = linear * sizes[i] + split[permutation[i]];
  }
  return {linear / cols, linear % cols};
}

TEST_F(BlockScaledQuantizeAnalysisTest, RowScaleSwizzled) {
  for (auto [type, max, inv_max, layout] :
       {std::make_tuple("f8e4m3fn", "448", "0.00223214296", ""),
        std::make_tuple("f4e2m1fn", "6", "0.166666672", "{2,1,0:E(4)}")}) {
    SCOPED_TRACE(type);
    ASSERT_OK_AND_ASSIGN(
        BlockScaledQuantizeAnalysis analysis,
        Analyze(Module(absl::StrCat(
            RowBody(type, max, inv_max), Swizzled("r_scale", 128, 8),
            "  ROOT t = (", type, "[128,8,32]", layout,
            ", u8[128,8]) tuple(r_data, r_scale_sw)"))));
    ASSERT_EQ(analysis.quantizations.size(), 1);
    const BlockScaledQuantization& row = analysis.quantizations[0];
    EXPECT_THAT(row.scale_root_indices, ElementsAre(1));
    EXPECT_THAT(Evaluate(row.data_root_indexing[0], {70, 100}),
                ElementsAre(70, 3, 4));
    for (auto [r, b] : {std::pair<int64_t, int64_t>{70, 3}, {0, 0}, {127, 7},
                        {33, 5}}) {
      EXPECT_EQ(Evaluate(row.scale_root_indexing[0], {r, b}),
                SwizzledPosition(128, 8, r, b));
    }
  }
}

TEST_F(BlockScaledQuantizeAnalysisTest, ColumnScaleSwizzled) {
  ASSERT_OK_AND_ASSIGN(
      BlockScaledQuantizeAnalysis analysis,
      Analyze(Module(absl::StrCat(
          "  p0 = bf16[256,256] parameter(0)", ColumnOf(256, 256),
          Swizzled("c_scale", 256, 8),
          "  ROOT t = (f8e4m3fn[256,8,32], u8[256,8]) tuple(c_data, "
          "c_scale_sw)"))));
  ASSERT_EQ(analysis.quantizations.size(), 1);
  const BlockScaledQuantization& column = analysis.quantizations[0];
  EXPECT_EQ(column.orientation, Orientation::kColumn);
  // Block 2 of column 100 is element (100, 2) of the natural scale.
  EXPECT_EQ(Evaluate(column.scale_root_indexing[0], {2, 100}),
            SwizzledPosition(256, 8, 100, 2));
}

TEST_F(BlockScaledQuantizeAnalysisTest, BothScalesSwizzled) {
  std::string pair = PairOf(256, 256);
  pair.replace(pair.find("  ROOT t"), std::string::npos,
               absl::StrCat(Swizzled("r_scale", 256, 8),
                            Swizzled("c_scale", 256, 8),
                            "  ROOT t = (f8e4m3fn[256,8,32], "
                            "f8e4m3fn[256,8,32], u8[256,8], u8[256,8]) "
                            "tuple(r_data, c_data, r_scale_sw, c_scale_sw)"));
  ASSERT_OK_AND_ASSIGN(BlockScaledQuantizeAnalysis analysis,
                       Analyze(Module(pair)));
  ASSERT_EQ(analysis.quantizations.size(), 2);
  EXPECT_THAT(analysis.row()->scale_root_indices, ElementsAre(2));
  EXPECT_THAT(analysis.column()->scale_root_indices, ElementsAre(3));
  EXPECT_EQ(Evaluate(analysis.row()->scale_root_indexing[0], {200, 6}),
            SwizzledPosition(256, 8, 200, 6));
  EXPECT_EQ(Evaluate(analysis.column()->scale_root_indexing[0], {6, 200}),
            SwizzledPosition(256, 8, 200, 6));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesPermutedDataResult) {
  EXPECT_THAT(Analyze(Module(absl::StrCat(
                  RowBody("f8e4m3fn", "448", "0.00223214296"), R"(
  d_split = f8e4m3fn[4,32,8,32] reshape(r_data)
  d_perm = f8e4m3fn[32,4,8,32] transpose(d_split), dimensions={1,0,2,3}
  d_sw = f8e4m3fn[128,8,32] reshape(d_perm)
  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(d_sw, r_scale))"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("permutes a result other than a scale")));
}

TEST_F(BlockScaledQuantizeAnalysisTest, RefusesScaleReshapeThatMerges) {
  EXPECT_THAT(Analyze(Module(absl::StrCat(
                  RowBody("f8e4m3fn", "448", "0.00223214296"), R"(
  s_merged = u8[8,128] reshape(r_scale)
  s_perm = u8[128,8] transpose(s_merged), dimensions={1,0}
  s_sw = u8[128,8] reshape(s_perm)
  ROOT t = (f8e4m3fn[128,8,32], u8[128,8]) tuple(r_data, s_sw))"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported instruction s_merged")));
}

}  // namespace
}  // namespace xla::gpu
