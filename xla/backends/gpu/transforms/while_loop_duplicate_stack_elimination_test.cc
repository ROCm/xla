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

#include "xla/backends/gpu/transforms/while_loop_duplicate_stack_elimination.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

using WhileLoopDuplicateStackEliminationTest = HloHardwareIndependentTestBase;

// A scan that writes `w`, a value computed each iteration, both into the
// carried buffer `buf` (at layer_indices[i], with jnp's negative-index
// wrap-around) and into the residual stack `res` (at i), as
// jax.lax.scan(... buf.at[layer_indices[i]].set(w) ..., ys=w) lowers.
// Placeholders: $RES_INDEX (the residual's start index), $TRIP (loop bound),
// $RES_UPDATE (the value stored into res).
constexpr absl::string_view kScan = R"(
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
  li_slice = s32[1] dynamic-slice(layer_indices, i), dynamic_slice_sizes={1}
  li = s32[] reshape(li_slice)
  neg = pred[] compare(li, zero), direction=LT
  wrapped = s32[] add(li, n)
  buf_index = s32[] select(neg, wrapped, li)
  w_buf = f32[1,2,3] reshape(w)
  buf_new = f32[4,2,3] dynamic-update-slice(buf, w_buf, buf_index, zero, zero)
  w_copy = f32[2,3] copy(w)
  w_res = f32[1,2,3] reshape($RES_UPDATE)
  res_new = f32[4,2,3] dynamic-update-slice(res, w_res, $RES_INDEX, zero, zero)
  one = s32[] constant(1)
  next = s32[] add(i, one)
  ROOT t = (s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]) tuple(next, buf_new, res_new, layer_indices, shard)
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
  iota = s32[4] iota(), iota_dimension=0
  i0 = s32[] constant(0)
  init = (s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]) tuple(i0, buf0, res0, iota, shard)
  loop = (s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]) while(init), condition=cond, body=body
  buf_out = f32[4,2,3] get-tuple-element(loop), index=1
  res_out = f32[4,2,3] get-tuple-element(loop), index=2
  sum = f32[4,2,3] add(res_out, res_out)
  ROOT out = (f32[4,2,3], f32[4,2,3]) tuple(buf_out, sum)
}
)";

std::string Scan(absl::string_view res_index = "i",
                 absl::string_view trip = "4",
                 absl::string_view res_update = "w_copy") {
  return absl::StrReplaceAll(kScan, {{"$RES_INDEX", res_index},
                                     {"$TRIP", trip},
                                     {"$RES_UPDATE", res_update}});
}

// The users of the residual stack read the carried buffer, and the residual
// element then dies.
TEST_F(WhileLoopDuplicateStackEliminationTest, ResidualReadsCarriedBuffer) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(Scan()));
  ASSERT_OK_AND_ASSIGN(
      bool changed, WhileLoopDuplicateStackElimination().Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* sum = FindInstruction(module.get(), "sum");
  ASSERT_NE(sum, nullptr);
  for (const HloInstruction* operand : sum->operands()) {
    EXPECT_EQ(operand->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_EQ(operand->tuple_index(), 1);
  }

  ASSERT_OK_AND_ASSIGN(bool simplified,
                          WhileLoopSimplifier().Run(module.get()));
  EXPECT_TRUE(simplified);
  const HloInstruction* loop = nullptr;
  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      loop = instruction;
    }
  }
  ASSERT_NE(loop, nullptr);
  int64_t stacks = 0;
  for (const Shape& element : loop->shape().tuple_shapes()) {
    stacks += ShapeUtil::Equal(element,
                               ShapeUtil::MakeShape(F32, {4, 2, 3}));
  }
  EXPECT_EQ(stacks, 1);
}

// The kept element is the one whose initial value is not a fresh buffer, even
// when it comes second.
TEST_F(WhileLoopDuplicateStackEliminationTest, KeepsTheNonFreshBuffer) {
  std::string hlo = Scan();
  // Swap the initial values: the residual (index 2) starts from the parameter.
  hlo = absl::StrReplaceAll(
      hlo, {{"tuple(i0, buf0, res0, iota, shard)",
             "tuple(i0, res0, buf0, iota, shard)"}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, WhileLoopDuplicateStackElimination().Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* out = module->entry_computation()->root_instruction();
  EXPECT_EQ(out->operand(0)->tuple_index(), 2);
  EXPECT_EQ(FindInstruction(module.get(), "sum")->operand(0)->tuple_index(), 2);
}

// Writes at different offsets (reversed) hold different values.
TEST_F(WhileLoopDuplicateStackEliminationTest, DifferentOffsets) {
  std::string hlo = Scan("rev");
  hlo = absl::StrReplaceAll(
      hlo, {{"  w_copy = f32[2,3] copy(w)\n",
             "  w_copy = f32[2,3] copy(w)\n  three = s32[] constant(3)\n"
             "  rev = s32[] subtract(three, i)\n"}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, WhileLoopDuplicateStackElimination().Run(module.get()));
  EXPECT_FALSE(changed);
}

// Three iterations leave a slot of each stack at its initial value, and those
// differ.
TEST_F(WhileLoopDuplicateStackEliminationTest, PartialCoverage) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(Scan("i", "3")));
  ASSERT_OK_AND_ASSIGN(
      bool changed, WhileLoopDuplicateStackElimination().Run(module.get()));
  EXPECT_FALSE(changed);
}

// A different stored value.
TEST_F(WhileLoopDuplicateStackEliminationTest, DifferentValues) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(Scan("i", "4", "shard")));
  ASSERT_OK_AND_ASSIGN(
      bool changed, WhileLoopDuplicateStackElimination().Run(module.get()));
  EXPECT_FALSE(changed);
}

// The layer indices are not known (a loop element the body changes), so the
// offsets cannot be compared.
TEST_F(WhileLoopDuplicateStackEliminationTest, UnknownIndices) {
  std::string hlo = Scan();
  hlo = absl::StrReplaceAll(
      hlo, {{"tuple(next, buf_new, res_new, layer_indices, shard)",
             "tuple(next, buf_new, res_new, layer_indices_next, shard)"},
            {"  one = s32[] constant(1)\n",
             "  one = s32[] constant(1)\n  ones = s32[4] broadcast(one), "
             "dimensions={}\n  layer_indices_next = s32[4] "
             "add(layer_indices, ones)\n"}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, WhileLoopDuplicateStackElimination().Run(module.get()));
  EXPECT_FALSE(changed);
}

// Layer indices that are a permutation: the offsets differ from i.
TEST_F(WhileLoopDuplicateStackEliminationTest, PermutedIndices) {
  std::string hlo = Scan();
  hlo = absl::StrReplaceAll(
      hlo, {{"  iota = s32[4] iota(), iota_dimension=0\n",
             "  iota = s32[4] constant({1, 0, 2, 3})\n"}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, WhileLoopDuplicateStackElimination().Run(module.get()));
  EXPECT_FALSE(changed);
}

// The same indices as a constant array (not an iota) are recognized.
TEST_F(WhileLoopDuplicateStackEliminationTest, ConstantIndices) {
  std::string hlo = Scan();
  hlo = absl::StrReplaceAll(
      hlo, {{"  iota = s32[4] iota(), iota_dimension=0\n",
             "  iota = s32[4] constant({0, 1, 2, -1})\n"}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, WhileLoopDuplicateStackElimination().Run(module.get()));
  EXPECT_TRUE(changed);
}

// A use of the whole loop tuple is left alone.
TEST_F(WhileLoopDuplicateStackEliminationTest, NonGteUser) {
  std::string hlo = Scan();
  hlo = absl::StrReplaceAll(
      hlo, {{"ROOT out = (f32[4,2,3], f32[4,2,3]) tuple(buf_out, sum)",
             "ROOT out = ((s32[], f32[4,2,3], f32[4,2,3], s32[4], f32[2,3]), "
             "f32[4,2,3]) tuple(loop, sum)"}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, WhileLoopDuplicateStackElimination().Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla::gpu
