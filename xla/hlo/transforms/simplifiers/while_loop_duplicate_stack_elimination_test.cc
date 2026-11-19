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

#include "xla/hlo/transforms/simplifiers/while_loop_duplicate_stack_elimination.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/while_loop_simplifier.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// A scan that writes `w`, a value computed in each iteration, both into the
// carried buffer `buf` (at layer_indices[i], with jnp's negative-index wrap)
// and into the residual stack `res` (at $RES_INDEX), as jax.lax.scan lowers
// buf.at[layer_indices[i]].set(w) with w also saved for the backward pass.
constexpr absl::string_view kScan = R"hlo(
HloModule scan

body {
  p = (s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  buf = f32[4,2,3] get-tuple-element(p), index=1
  res = f32[4,2,3] get-tuple-element(p), index=2
  layer_indices = s32[4] get-tuple-element(p), index=3
  shard = f32[2,3] get-tuple-element(p), index=4
  i_f = f32[] convert(i)
  i_b = f32[2,3] broadcast(i_f), dimensions={}
  w = f32[2,3] add(shard, i_b)
  zero = s32[] constant(0)
  n = s32[] constant(4)
  one = s32[] constant(1)
  two = s32[] constant(2)
  three = s32[] constant(3)
  seven = s32[] constant(7)
  half = s32[] divide(i, two)
  li_slice = s32[1] dynamic-slice(layer_indices, $LI_INDEX), dynamic_slice_sizes={1}
  li = s32[] reshape(li_slice)
  neg = pred[] compare(li, zero), direction=LT
  wrapped = s32[] add(li, n)
  buf_index = s32[] select(neg, wrapped, li)
  w_buf = f32[1,2,3] reshape(w)
  buf_new = f32[4,2,3] dynamic-update-slice(buf, w_buf, buf_index, zero, zero)
  w_copy = f32[2,3] copy(w)
  $EXTRA
  w_res = f32[1,2,3] reshape($RES_UPDATE)
  res_new = f32[4,2,3] dynamic-update-slice(res, w_res, $RES_INDEX, zero, zero)
  next = s32[] add(i, $STEP)
  ROOT t = (s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]) tuple(next, buf_new, res_new, $LI_NEXT, shard)
}

cond {
  p = (s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  limit = s32[] constant($TRIP)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

ENTRY main {
  buf0 = f32[4,2,3] parameter(0)
  shard = f32[2,3] parameter(1)
  c0 = f32[] constant(0)
  res0 = f32[4,2,3] broadcast(c0), dimensions={}
  $INDICES
  i0 = s32[] constant(0)
  init = (s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]) tuple(i0, $INITS, indices, shard)
  loop = (s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]) while(init), condition=cond, body=body$LOOP_ATTRS
  $BUF_OUT
  res_out = f32[4,2,3] get-tuple-element(loop), index=2$RES_OUT_ATTRS
  sum = f32[4,2,3] add(res_out, res_out)
  ROOT out = $ROOT
}
)hlo";

using Replacements = std::vector<std::pair<std::string, std::string>>;

// The scan with some placeholders replaced; the others take their defaults.
std::string Scan(const Replacements& replacements = {}) {
  absl::flat_hash_map<std::string, std::string> values = {
      {"$RES_INDEX", "i"},
      {"$RES_UPDATE", "w_copy"},
      {"$EXTRA", ""},  // More body instructions.
      {"$TRIP", "4"},
      {"$STEP", "one"},
      {"$LI_INDEX", "i"},
      {"$LI_NEXT", "layer_indices"},
      {"$INDICES", "indices = s32[4] iota(), iota_dimension=0"},
      {"$INITS", "buf0, res0"},
      {"$LOOP_ATTRS", ""},
      {"$BUF_OUT", "buf_out = f32[4,2,3] get-tuple-element(loop), index=1"},
      {"$RES_OUT_ATTRS", ""},
      {"$ROOT", "(f32[4,2,3], f32[4,2,3]) tuple(buf_out, sum)"}};
  for (const auto& [key, value] : replacements) {
    values[key] = value;
  }
  return absl::StrReplaceAll(kScan, values);
}

using WhileLoopDuplicateStackEliminationTest = HloHardwareIndependentTestBase;

TEST_F(WhileLoopDuplicateStackEliminationTest, ResidualReadsCarriedBuffer) {
  RunAndFilecheckHloRewrite(Scan(), WhileLoopDuplicateStackElimination(), R"(
    // CHECK:     ENTRY
    // CHECK:     %[[BUF:.*]] = f32[4,2,3]{2,1,0} get-tuple-element(%loop), index=1
    // CHECK-NOT: get-tuple-element(%loop), index=2
    // CHECK:     add(%[[BUF]], %[[BUF]])
  )");
}

// Only the dropped element is read: the pass adds a get-tuple-element of the
// kept one, with the metadata of the replaced one and the loop's sharding.
TEST_F(WhileLoopDuplicateStackEliminationTest, AddsGteOfKeptElement) {
  RunAndFilecheckHloRewrite(
      Scan({{"$BUF_OUT", ""},
            {"$LOOP_ATTRS",
             ", sharding={{replicated}, {maximal device=0}, {maximal "
             "device=1}, {replicated}, {replicated}}"},
            {"$RES_OUT_ATTRS", R"(, metadata={op_name="res"})"},
            {"$ROOT", "f32[4,2,3] copy(sum)"}}),
      WhileLoopDuplicateStackElimination(), R"(
    // CHECK:     ENTRY
    // CHECK-NOT: get-tuple-element(%loop), index=2
    // CHECK:     %[[BUF:.*]] = f32[4,2,3]{2,1,0} get-tuple-element(%loop), index=1, sharding={maximal device=0}, metadata={op_name="res"}
    // CHECK:     add(%[[BUF]], %[[BUF]])
  )");
}

// The two elements are equal up to layout: a copy in the body changes the
// layout of the residual stack, which the kept element would not have.
TEST_F(WhileLoopDuplicateStackEliminationTest, DifferentLayouts) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule layouts

body {
  p = (s32[], f32[4,3]{1,0}, f32[4,3]{0,1}) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  buf = f32[4,3]{1,0} get-tuple-element(p), index=1
  res = f32[4,3]{0,1} get-tuple-element(p), index=2
  i_f = f32[] convert(i)
  w = f32[1,3]{1,0} broadcast(i_f), dimensions={}
  zero = s32[] constant(0)
  one = s32[] constant(1)
  buf_new = f32[4,3]{1,0} dynamic-update-slice(buf, w, i, zero)
  res_row_major = f32[4,3]{1,0} copy(res)
  res_dus = f32[4,3]{1,0} dynamic-update-slice(res_row_major, w, i, zero)
  res_new = f32[4,3]{0,1} copy(res_dus)
  next = s32[] add(i, one)
  ROOT t = (s32[], f32[4,3]{1,0}, f32[4,3]{0,1}) tuple(next, buf_new, res_new)
}

cond {
  p = (s32[], f32[4,3]{1,0}, f32[4,3]{0,1}) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

ENTRY main {
  buf0 = f32[4,3]{1,0} parameter(0)
  c0 = f32[] constant(0)
  res0 = f32[4,3]{0,1} broadcast(c0), dimensions={}
  i0 = s32[] constant(0)
  init = (s32[], f32[4,3]{1,0}, f32[4,3]{0,1}) tuple(i0, buf0, res0)
  loop = (s32[], f32[4,3]{1,0}, f32[4,3]{0,1}) while(init), condition=cond, body=body
  buf_out = f32[4,3]{1,0} get-tuple-element(loop), index=1
  res_out = f32[4,3]{0,1} get-tuple-element(loop), index=2
  ROOT out = (f32[4,3]{1,0}, f32[4,3]{0,1}) tuple(buf_out, res_out)
}
)hlo";
  ASSERT_OK(RunAndCheckHloRewrite(kHlo, WhileLoopDuplicateStackElimination(),
                                  /*expect_change=*/false)
                .status());
}

// The kept element is the one whose initial value is not a fresh buffer, even
// when it comes second.
TEST_F(WhileLoopDuplicateStackEliminationTest, KeepsTheNonFreshBuffer) {
  RunAndFilecheckHloRewrite(Scan({{"$INITS", "res0, buf0"}}),
                            WhileLoopDuplicateStackElimination(), R"(
    // CHECK:     ENTRY
    // CHECK-NOT: get-tuple-element(%loop), index=1
    // CHECK:     %[[RES:.*]] = f32[4,2,3]{2,1,0} get-tuple-element(%loop), index=2
    // CHECK:     add(%[[RES]], %[[RES]])
    // CHECK:     tuple(%[[RES]], %sum)
  )");
}

struct Case {
  std::string name;
  Replacements replacements;
  bool expect_change;
};

class WhileLoopDuplicateStackEliminationCaseTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<Case> {};

// Runs the pass. When it changes the module, also runs WhileLoopSimplifier,
// which must leave one stack, and checks with HloEvaluator that the results
// are unchanged.
TEST_P(WhileLoopDuplicateStackEliminationCaseTest, Run) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(Scan(GetParam().replacements)));
  std::unique_ptr<HloModule> original = module->Clone();
  ASSERT_OK_AND_ASSIGN(bool changed,
                       WhileLoopDuplicateStackElimination().Run(module.get()));
  ASSERT_EQ(changed, GetParam().expect_change);
  if (!changed) {
    return;
  }
  ASSERT_OK_AND_ASSIGN(bool simplified,
                       WhileLoopSimplifier().Run(module.get()));
  EXPECT_TRUE(simplified);
  ASSERT_OK(HloDCE().Run(module.get()).status());
  const HloInstruction* loop = FindInstruction(module.get(), HloOpcode::kWhile);
  ASSERT_NE(loop, nullptr);
  EXPECT_EQ(absl::c_count(loop->shape().tuple_shapes(),
                          ShapeUtil::MakeShape(F32, {4, 2, 3})),
            1);

  Literal buf0 =
      LiteralUtil::CreateR3<float>({{{-1, -2, -3}, {-4, -5, -6}},
                                    {{-7, -8, -9}, {-10, -11, -12}},
                                    {{-13, -14, -15}, {-16, -17, -18}},
                                    {{-19, -20, -21}, {-22, -23, -24}}});
  Literal shard = LiteralUtil::CreateR2<float>({{1, 2, 3}, {4, 5, 6}});
  ASSERT_OK_AND_ASSIGN(Literal expected,
                       HloEvaluator().Evaluate(*original, {&buf0, &shard}));
  ASSERT_OK_AND_ASSIGN(Literal actual,
                       HloEvaluator().Evaluate(*module, {&buf0, &shard}));
  EXPECT_EQ(expected, actual);
}

INSTANTIATE_TEST_SUITE_P(
    Cases, WhileLoopDuplicateStackEliminationCaseTest,
    ::testing::Values(
        Case{"Scan", {}, true},
        // Only the dropped element is read.
        Case{"OnlyDroppedElementRead",
             {{"$BUF_OUT", ""}, {"$ROOT", "f32[4,2,3] copy(sum)"}},
             true},
        // Equal only after the dynamic-update-slice clamps 7 to 3.
        Case{"ClampedStarts",
             {{"$RES_INDEX", "big"},
              {"$EXTRA",
               "last = pred[] compare(i, three), direction=EQ\n"
               "  big = s32[] select(last, seven, i)"}},
             true},
        // Five iterations: both stacks write slot 3 twice (4 is clamped).
        Case{"OverlappingWrites", {{"$TRIP", "5"}}, true},
        // The same, but the residual's last write goes to slot 2.
        Case{"DifferentOverlap",
             {{"$TRIP", "5"},
              {"$RES_INDEX", "back"},
              {"$EXTRA",
               "four = pred[] compare(i, n), direction=EQ\n"
               "  back = s32[] select(four, two, i)"}},
             false},
        // i = 0, 2, 4, 6, written at i / 2.
        Case{"StepTwo",
             {{"$TRIP", "8"},
              {"$STEP", "two"},
              {"$RES_INDEX", "half"},
              {"$LI_INDEX", "half"}},
             true},
        // Step 2 with the buffer written at layer_indices[i]: slots 0, 2, 3.
        Case{"StepTwoPartialCoverage",
             {{"$TRIP", "8"}, {"$STEP", "two"}, {"$RES_INDEX", "half"}},
             false},
        Case{"ReversedOffsets",
             {{"$RES_INDEX", "rev"},
              {"$EXTRA", "rev = s32[] subtract(three, i)"}},
             false},
        // Three iterations leave a slot of each stack at its initial value.
        Case{"PartialCoverage", {{"$TRIP", "3"}}, false},
        Case{"DifferentValues", {{"$RES_UPDATE", "shard"}}, false},
        Case{"PermutedIndices",
             {{"$INDICES", "indices = s32[4] constant({1, 0, 2, 3})"}},
             false},
        // A use of the whole loop tuple.
        Case{"NonGteUser",
             {{"$ROOT",
               "((s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]), "
               "f32[4,2,3]) tuple(loop, sum)"}},
             false}),
    [](const ::testing::TestParamInfo<Case>& info) { return info.param.name; });

}  // namespace
}  // namespace xla
