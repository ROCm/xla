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

// Sample HLO for the "dynamic buffer extents in MoE" note.
//
// An MoE layer's receive buffer is allocated at a static capacity N_recv and
// only a device-resident prefix of n rows is live. The two modules below
// express the same consumer (a clamp-and-multiply epilogue followed by a
// row reduction) in the two forms the note compares:
//
//   kMaskedModule   -- what JAX emits today: n folded into an iota < n
//                      predicate, every shape static at capacity.
//   kBoundedModule  -- the same computation on XLA's bounded shape
//                      f32[<=8,2] via set-dimension-size.
//
// The tests show (a) both forms compute the same values, and (b) what
// DynamicPadder does to the bounded form before the fusion emitter ever
// sees it: the bounded shapes become static, the row count survives only
// as the iota < n predicate (or a SliceToDynamic at the entry boundary),
// and set-dimension-size disappears from the module.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/dynamic_padder.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

// Today's form. The live count n only decides what the last N - n rows
// contribute; every instruction, and therefore the launch grid of the fusion
// XLA builds from this, is shaped by the capacity [8,2].
constexpr char kMaskedModule[] = R"(
HloModule moe_extent_masked

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  recv = f32[8,2] parameter(0)
  n = s32[] parameter(1)
  iota = s32[8,2] iota(), iota_dimension=0
  n_b = s32[8,2] broadcast(n), dimensions={}
  mask = pred[8,2] compare(iota, n_b), direction=LT
  lo = f32[] constant(0)
  hi = f32[] constant(50)
  lo_b = f32[8,2] broadcast(lo), dimensions={}
  hi_b = f32[8,2] broadcast(hi), dimensions={}
  clamped = f32[8,2] clamp(lo_b, recv, hi_b)
  act = f32[8,2] multiply(clamped, recv)
  zero_b = f32[8,2] broadcast(lo), dimensions={}
  masked = f32[8,2] select(mask, act, zero_b)
  ROOT combined = f32[2] reduce(masked, lo), dimensions={0}, to_apply=add_f32
}
)";

// Bounded form. n is attached to dimension 0 of the buffer by
// set-dimension-size, and the epilogue and the reduce are typed on the
// bounded shape. No iota, compare or select is written by hand. Every
// elementwise operand carries the same extent: DynamicPadder rejects a
// bounded operand that has no size source (such as a broadcast declared
// f32[<=8,2] directly) with "Dynamic input unexpectedly found".
constexpr char kBoundedModule[] = R"(
HloModule moe_extent_bounded

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  recv = f32[8,2] parameter(0)
  n = s32[] parameter(1)
  live = f32[<=8,2] set-dimension-size(recv, n), dimensions={0}
  lo = f32[] constant(0)
  hi = f32[] constant(50)
  lo_b = f32[8,2] broadcast(lo), dimensions={}
  hi_b = f32[8,2] broadcast(hi), dimensions={}
  lo_live = f32[<=8,2] set-dimension-size(lo_b, n), dimensions={0}
  hi_live = f32[<=8,2] set-dimension-size(hi_b, n), dimensions={0}
  clamped = f32[<=8,2] clamp(lo_live, live, hi_live)
  act = f32[<=8,2] multiply(clamped, live)
  ROOT combined = f32[2] reduce(act, lo), dimensions={0}, to_apply=add_f32
}
)";

// Bounded form without the reduce: the bounded value is the entry result.
constexpr char kBoundedElementwiseModule[] = R"(
HloModule moe_extent_bounded_elementwise

ENTRY entry {
  recv = f32[8,2] parameter(0)
  n = s32[] parameter(1)
  live = f32[<=8,2] set-dimension-size(recv, n), dimensions={0}
  lo = f32[] constant(0)
  hi = f32[] constant(50)
  lo_b = f32[8,2] broadcast(lo), dimensions={}
  hi_b = f32[8,2] broadcast(hi), dimensions={}
  lo_live = f32[<=8,2] set-dimension-size(lo_b, n), dimensions={0}
  hi_live = f32[<=8,2] set-dimension-size(hi_b, n), dimensions={0}
  clamped = f32[<=8,2] clamp(lo_live, live, hi_live)
  ROOT act = f32[<=8,2] multiply(clamped, live)
}
)";

// Test vector: the rank 0 buffer of the note's example, capacity 8, n = 5.
// Live rows hold 1.0 and contribute 1 * clamp(1) = 1 each; the three padding
// rows hold 100.0 and would contribute 100 * clamp(100) = 5000 each if any
// consumer read them. The correct result is therefore exactly [5, 5].
Literal ReceiveBuffer() {
  return LiteralUtil::CreateR2<float>({{1.0f, 1.0f},
                                       {1.0f, 1.0f},
                                       {1.0f, 1.0f},
                                       {1.0f, 1.0f},
                                       {1.0f, 1.0f},
                                       {100.0f, 100.0f},
                                       {100.0f, 100.0f},
                                       {100.0f, 100.0f}});
}

Literal LiveCount() { return LiteralUtil::CreateR0<int32_t>(5); }

Literal ExpectedCombined() {
  return LiteralUtil::CreateR1<float>({5.0f, 5.0f});
}

class MoeExtentPadderTest : public HloTestBase {
 protected:
  std::unique_ptr<HloModule> GetHloModule(const std::string& hlo_text) {
    return ParseAndReturnVerifiedModule(hlo_text).value();
  }

  // Mirrors DynamicPadderTest::RunPadder in dynamic_padder_test.cc.
  absl::StatusOr<bool> RunPadder(HloModule* module,
                                 bool slice_dynamic_output = false) {
    DynamicPadderOptions options;
    options.slice_dynamic_output = slice_dynamic_output;
    DynamicPadder padder(options);
    ABSL_ASSIGN_OR_RETURN(bool changed, RunHloPass(&padder, module));
    if (!changed) return false;
    TupleSimplifier tuple_simplifier;
    ABSL_RETURN_IF_ERROR(RunHloPass(&tuple_simplifier, module).status());
    AlgebraicSimplifier alg_simplifier(AlgebraicSimplifierOptions{});
    ABSL_RETURN_IF_ERROR(RunHloPass(&alg_simplifier, module).status());
    return true;
  }

  // Same matcher as DynamicPadderTest::ExpectPadded: the padder expresses
  // the live prefix as select(iota < broadcast(n), value, identity).
  void ExpectPadded(const HloInstruction* inst) {
    EXPECT_THAT(inst,
                op::Select(op::Lt(op::Iota(), op::Broadcast(op::Parameter())),
                           ::testing::_, op::Broadcast()));
  }

  // Mirrors ExecutionTest::PadAndExecute in dynamic_padder_test.cc.
  absl::StatusOr<Literal> PadAndExecute(std::unique_ptr<HloModule> module,
                                        absl::Span<Literal* const> arguments,
                                        bool slice_dynamic_output = true) {
    if (!slice_dynamic_output) {
      auto new_config = module->config();
      new_config.mutable_entry_computation_layout()
          ->mutable_result_layout()
          ->ClearDynamicShape();
      module->set_config(new_config);
    }
    DynamicPadderOptions options;
    options.slice_dynamic_output = slice_dynamic_output;
    DynamicPadder padder(options);
    ABSL_RETURN_IF_ERROR(padder.Run(module.get()).status());
    HloDCE dce;
    ABSL_RETURN_IF_ERROR(dce.Run(module.get()).status());
    return Execute(std::move(module), {arguments});
  }
};

// (a) Both forms compute the same values.

TEST_F(MoeExtentPadderTest, MaskedFormComputesLivePrefix) {
  auto module = GetHloModule(kMaskedModule);
  Literal recv = ReceiveBuffer();
  Literal n = LiveCount();
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&recv, &n}));
  EXPECT_EQ(result, ExpectedCombined());
}

TEST_F(MoeExtentPadderTest, BoundedFormComputesLivePrefix) {
  auto module = GetHloModule(kBoundedModule);
  Literal recv = ReceiveBuffer();
  Literal n = LiveCount();
  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&recv, &n}));
  EXPECT_EQ(result, ExpectedCombined());
}

// (b) What DynamicPadder leaves for the fusion emitter.

// After the padder, the bounded module is the masked module: the reduce
// operand is select(iota < n, act, 0), every shape is static at capacity
// [8,2], and no set-dimension-size remains. The emitter will size the launch
// grid from [8,2] and n is only a predicate inside the kernel body.
TEST_F(MoeExtentPadderTest, PadderRewritesBoundedFormToMaskedForm) {
  auto module = GetHloModule(kBoundedModule);
  ASSERT_OK_AND_ASSIGN(bool changed, RunPadder(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* entry = module->entry_computation();
  const HloInstruction* root = entry->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kReduce);
  ExpectPadded(root->operand(0));

  for (const HloInstruction* inst : entry->instructions()) {
    EXPECT_NE(inst->opcode(), HloOpcode::kSetDimensionSize) << inst->name();
    EXPECT_NE(inst->opcode(), HloOpcode::kGetDimensionSize) << inst->name();
    EXPECT_FALSE(inst->shape().is_dynamic())
        << inst->name() << " : " << inst->shape().ToString();
  }
}

// When the bounded value is the entry result, n survives only at the
// boundary as a SliceToDynamic custom call. The multiply feeding it is static
// [8,2]: a fusion built from it still launches over all 8 rows.
TEST_F(MoeExtentPadderTest, ExtentSurvivesOnlyAtEntryBoundary) {
  auto module = GetHloModule(kBoundedElementwiseModule);
  ASSERT_OK_AND_ASSIGN(bool changed,
                       RunPadder(module.get(), /*slice_dynamic_output=*/true));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CustomCall({"SliceToDynamic"}, op::Multiply(),
                                   ::testing::_, ::testing::_));
  EXPECT_TRUE(root->shape().is_dynamic_dimension(0));

  const HloInstruction* act = root->operand(0);
  EXPECT_EQ(act->opcode(), HloOpcode::kMultiply);
  EXPECT_FALSE(act->shape().is_dynamic());
  EXPECT_EQ(act->shape().dimensions(0), 8);
}

}  // namespace
}  // namespace xla
