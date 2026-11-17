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

#include "xla/backends/gpu/transforms/dus_accumulator_zero_init_elimination.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/hlo_module_config.h"

namespace xla::gpu {
namespace {

class DusAccumulatorZeroInitEliminationTest
    : public HloHardwareIndependentTestBase {
 protected:
  void EnableFlag(HloModule* module, bool value = true) {
    auto opts = module->config().debug_options();
    opts.set_xla_gpu_enable_dus_accumulator_zero_init_elimination(value);
    module->mutable_config().set_debug_options(opts);
  }

  HloModuleConfig ConfigWithFlag() const {
    HloModuleConfig config = GetModuleConfigForTest();
    config.mutable_debug_options()
        .set_xla_gpu_enable_dus_accumulator_zero_init_elimination(true);
    return config;
  }

  int64_t CountAllocateBuffers(HloModule* module) {
    int64_t n = 0;
    for (const HloComputation* c : module->computations()) {
      for (const HloInstruction* i : c->instructions()) {
        if (i->opcode() == HloOpcode::kCustomCall &&
            i->custom_call_target() == "AllocateBuffer") {
          ++n;
        }
      }
    }
    return n;
  }
};

// Scaffold A: single-slot raw-DUS scan with broadcast(0) init. Cases override
// only what differs from "BasicAscendingRawDus".
struct RawDusCase {
  absl::string_view name;
  bool expected_changed;
  int64_t expected_buffers;
  absl::string_view body_dus = R"(  z = s32[] constant(0)
  dus = bf16[4,8] dynamic-update-slice(acc, update, it, z))";
  absl::string_view init_expr = R"(  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] broadcast(c0), dimensions={})";
  absl::string_view while_extra =
      "";  // extra attribute on `w =`, e.g. sharding
  absl::string_view trip_limit = "4";
  absl::string_view trip_count = "";   // known_trip_count; empty: trip_limit
  absl::string_view iv_init = "z";     // ENTRY-level name feeding the IV's init
  absl::string_view known_init = "0";  // known_init_step.init
  absl::string_view iv_step = "1";     // known_init_step.step, body step
  absl::string_view entry_extra = "";  // entry pre-init declarations
  absl::string_view computations = "";  // extra computations, e.g. fusions
  bool check_first_slot = false;  // assert init->operand(1) is AllocateBuffer

  constexpr RawDusCase(absl::string_view name, bool expected_changed,
                       int64_t expected_buffers)
      : name(name),
        expected_changed(expected_changed),
        expected_buffers(expected_buffers) {}

  constexpr RawDusCase WithBodyDus(absl::string_view val) const {
    RawDusCase c = *this;
    c.body_dus = val;
    return c;
  }
  constexpr RawDusCase WithInitExpr(absl::string_view val) const {
    RawDusCase c = *this;
    c.init_expr = val;
    return c;
  }
  constexpr RawDusCase WithWhileExtra(absl::string_view val) const {
    RawDusCase c = *this;
    c.while_extra = val;
    return c;
  }
  constexpr RawDusCase WithTripLimit(absl::string_view val) const {
    RawDusCase c = *this;
    c.trip_limit = val;
    return c;
  }
  constexpr RawDusCase WithIvInit(absl::string_view val) const {
    RawDusCase c = *this;
    c.iv_init = val;
    return c;
  }
  // Loop with IV values init, init + step, ... (`trip` of them) below `limit`.
  constexpr RawDusCase WithIv(absl::string_view init, absl::string_view step,
                              absl::string_view trip,
                              absl::string_view limit) const {
    RawDusCase c = *this;
    c.known_init = init;
    c.iv_step = step;
    c.trip_count = trip;
    c.trip_limit = limit;
    return c;
  }
  constexpr RawDusCase WithEntryExtra(absl::string_view val) const {
    RawDusCase c = *this;
    c.entry_extra = val;
    return c;
  }
  constexpr RawDusCase WithComputations(absl::string_view val) const {
    RawDusCase c = *this;
    c.computations = val;
    return c;
  }
  // First iteration peeled out of the loop (see kPeeledFirstInit): the loop
  // runs iterations 1..3.
  constexpr RawDusCase Peeled(absl::string_view init_expr) const {
    return WithIv("1", "1", "3", "4")
        .WithIvInit("iv0")
        .WithEntryExtra("  iv0 = s32[] constant(1)\n")
        .WithInitExpr(init_expr);
  }
  constexpr RawDusCase WithCheckFirstSlot(bool val) const {
    RawDusCase c = *this;
    c.check_first_slot = val;
    return c;
  }
};

std::string MakeRawDusHlo(const RawDusCase& c) {
  return absl::StrReplaceAll(
      R"(
HloModule m
$COMPUTATIONS
body {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
$BODY_DUS
  step = s32[] constant($IV_STEP)
  next = s32[] add(it, step)
  ROOT out = (s32[], bf16[4,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant($TRIP_LIMIT)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
$ENTRY_EXTRA$INIT_EXPR
  init = (s32[], bf16[4,8]) tuple($IV_INIT, zero_init)
  w = (s32[], bf16[4,8]) while(init), condition=cond, body=body,$WHILE_EXTRA
      backend_config={"known_trip_count":{"n":"$TRIP_COUNT"},
                      "known_init_step":{"init":"$KNOWN_INIT",
                                         "step":"$IV_STEP"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)",
      {{"$BODY_DUS", c.body_dus},
       {"$INIT_EXPR", c.init_expr},
       {"$WHILE_EXTRA", c.while_extra},
       {"$TRIP_LIMIT", c.trip_limit},
       {"$TRIP_COUNT", c.trip_count.empty() ? c.trip_limit : c.trip_count},
       {"$KNOWN_INIT", c.known_init},
       {"$IV_STEP", c.iv_step},
       {"$IV_INIT", c.iv_init},
       {"$ENTRY_EXTRA", c.entry_extra},
       {"$COMPUTATIONS", c.computations}});
}

// CollectivePipeliner peeled iteration 0: the init DUS(broadcast(0), x0, 0)
// is simplified to pad(x0, 0).
constexpr absl::string_view kPeeledFirstInit = R"(  x0 = bf16[1,8] parameter(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] pad(x0, c0), padding=0_3x0_0)";
// Reverse scan: the peeled iteration 0 wrote the last slot.
constexpr absl::string_view kPeeledLastInit = R"(  x0 = bf16[1,8] parameter(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] pad(x0, c0), padding=3_0x0_0)";
constexpr absl::string_view kTwoPeeledInit = R"(  x0 = bf16[2,8] parameter(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] pad(x0, c0), padding=0_2x0_0)";
constexpr absl::string_view kPeeledPadTwoDimsInit =
    R"(  x0 = bf16[1,4] parameter(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] pad(x0, c0), padding=0_3x0_4)";

const RawDusCase kRawDusCases[] = {
    // Default scaffold; also asserts the init becomes AllocateBuffer.
    RawDusCase("BasicAscendingRawDus", true, 1).WithCheckFirstSlot(true),

    RawDusCase("DescendingDusSubIv", true, 1)
        .WithBodyDus(R"(  z = s32[] constant(0)
  three = s32[] constant(3)
  idx = s32[] subtract(three, it)
  dus = bf16[4,8] dynamic-update-slice(acc, update, idx, z))"),

    // sub(2, it) covers [-1, 2], not [0, 4) — must skip.
    RawDusCase("DescendingWithWrongConstant", false, 0)
        .WithBodyDus(R"(  z = s32[] constant(0)
  two = s32[] constant(2)
  idx = s32[] subtract(two, it)
  dus = bf16[4,8] dynamic-update-slice(acc, update, idx, z))"),

    // Second use of `acc` forces body-gte multi-user — must skip.
    RawDusCase("MultiUserBodyGteSkipped", false, 0)
        .WithBodyDus(R"(  z = s32[] constant(0)
  dus_inner = bf16[4,8] dynamic-update-slice(acc, update, it, z)
  dus = bf16[4,8] add(dus_inner, acc))"),

    RawDusCase("NonBroadcastInitSkipped", false, 0)
        .WithInitExpr("  zero_init = bf16[4,8] parameter(0)"),

    // IV runs 1..4 and writes slot `it`: slot 0 is never written (the start 4
    // is clamped to 3) — must skip.
    RawDusCase("NonZeroIvInitMissesFirstSlotSkipped", false, 0)
        .WithIv("1", "1", "4", "5")
        .WithIvInit("one_init")
        .WithEntryExtra("  one_init = s32[] constant(1)\n"),

    // IV runs 1..4 and writes slot it - 1, so slots 0..3.
    RawDusCase("NonZeroIvInitShiftedIndex", true, 1)
        .WithIv("1", "1", "4", "5")
        .WithIvInit("one_init")
        .WithEntryExtra("  one_init = s32[] constant(1)\n")
        .WithBodyDus(R"(  z = s32[] constant(0)
  m1 = s32[] constant(-1)
  idx = s32[] add(it, m1)
  dus = bf16[4,8] dynamic-update-slice(acc, update, idx, z))"),

    // IV takes 0, 2, 4, 6 and writes columns [it, it + 2).
    RawDusCase("IvStepTwo", true, 1)
        .WithIv("0", "2", "4", "8")
        .WithBodyDus(R"(  z = s32[] constant(0)
  update_tile = bf16[4,2] constant({ {1,1}, {1,1}, {1,1}, {1,1} })
  dus = bf16[4,8] dynamic-update-slice(acc, update_tile, z, it))"),

    // Same loop writing one column per iteration leaves odd columns — skip.
    RawDusCase("IvStepTwoGapSkipped", false, 0)
        .WithIv("0", "2", "4", "8")
        .WithBodyDus(R"(  z = s32[] constant(0)
  update_col = bf16[4,1] constant({ {1}, {1}, {1}, {1} })
  dus = bf16[4,8] dynamic-update-slice(acc, update_col, z, it))"),

    // Starts it - 1 = -1..2 with a 2-row slice: -1 is clamped to 0, and 2
    // reaches the last row, so rows 0..3 are written.
    RawDusCase("ClampedStartsCoverBuffer", true, 1)
        .WithBodyDus(R"(  z = s32[] constant(0)
  m1 = s32[] constant(-1)
  idx = s32[] add(it, m1)
  c1 = bf16[] constant(1)
  update2 = bf16[2,8] broadcast(c1), dimensions={}
  dus = bf16[4,8] dynamic-update-slice(acc, update2, idx, z))"),

    // Starts it - 2 = -2..1 clamp to 0, 0, 0, 1: rows 0..1 only — skip.
    RawDusCase("ClampedStartsMissRightEdgeSkipped", false, 0)
        .WithBodyDus(R"(  z = s32[] constant(0)
  m2 = s32[] constant(-2)
  idx = s32[] add(it, m2)
  dus = bf16[4,8] dynamic-update-slice(acc, update, idx, z))"),

    // A full-height slice at start it + 1 = 1..4 is clamped to row 0 and
    // overwrites the whole buffer every iteration.
    RawDusCase("FullSliceClampedStart", true, 1)
        .WithBodyDus(R"(  z = s32[] constant(0)
  p1 = s32[] constant(1)
  idx = s32[] add(it, p1)
  c1 = bf16[] constant(1)
  update4 = bf16[4,8] broadcast(c1), dimensions={}
  dus = bf16[4,8] dynamic-update-slice(acc, update4, idx, z))"),

    // init + (trip - 1) * step overflows int64: skip before computing the IV
    // range (signed overflow). The full-height slice covers the buffer for any
    // start, so only the overflow guard rejects this loop.
    RawDusCase("IvRangeOverflowSkipped", false, 0)
        .WithIv("9223372036854775806", "1", "4", "4")
        .WithBodyDus(R"(  z = s32[] constant(0)
  c1 = bf16[] constant(1)
  update4 = bf16[4,8] broadcast(c1), dimensions={}
  dus = bf16[4,8] dynamic-update-slice(acc, update4, it, z))"),

    RawDusCase("PeeledFirstSlice", true, 1).Peeled(kPeeledFirstInit),

    // Iteration i writes slot 3 - i: the loop (i = 1..3) writes slots 2..0.
    RawDusCase("ReversePeeledLastSlice", true, 1)
        .Peeled(kPeeledLastInit)
        .WithBodyDus(R"(  z = s32[] constant(0)
  three = s32[] constant(3)
  idx = s32[] subtract(three, it)
  dus = bf16[4,8] dynamic-update-slice(acc, update, idx, z))"),

    // Double-buffer style: two iterations peeled, loop covers slots 2..3.
    RawDusCase("TwoPeeledSlices", true, 1)
        .Peeled(kTwoPeeledInit)
        .WithIv("2", "1", "2", "4")
        .WithEntryExtra("  iv0 = s32[] constant(2)\n"),

    // Loop writes slots 2..3, the peeled slice is slot 0: slot 1 stays zero.
    RawDusCase("GapBetweenPeeledAndLoopSkipped", false, 0)
        .Peeled(kPeeledFirstInit)
        .WithIv("2", "1", "2", "4")
        .WithEntryExtra("  iv0 = s32[] constant(2)\n"),

    // Peeled slice and loop (slots 0..2) both miss slot 3.
    RawDusCase("PeeledRightEdgeUncoveredSkipped", false, 0)
        .Peeled(kPeeledFirstInit)
        .WithBodyDus(R"(  z = s32[] constant(0)
  m1 = s32[] constant(-1)
  idx = s32[] add(it, m1)
  dus = bf16[4,8] dynamic-update-slice(acc, update, idx, z))"),

    // The loop never writes the padded columns of the peeled row.
    RawDusCase("PeeledPadAlongTwoDimsSkipped", false, 0)
        .Peeled(kPeeledPadTwoDimsInit),

    // Interior or negative padding: the pad is not x0 plus zeros at the edges.
    RawDusCase("PeeledInteriorPaddingSkipped", false, 0)
        .Peeled(R"(  x0 = bf16[2,8] parameter(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] pad(x0, c0), padding=0_1_1x0_0)"),
    RawDusCase("PeeledNegativePaddingSkipped", false, 0)
        .Peeled(R"(  x0 = bf16[5,8] parameter(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] pad(x0, c0), padding=0_-1x0_0)"),

    // The body reads the stack, so the padding is observable.
    RawDusCase("PeeledInitReadInBodySkipped", false, 0)
        .Peeled(kPeeledFirstInit)
        .WithBodyDus(R"(  z = s32[] constant(0)
  dus_inner = bf16[4,8] dynamic-update-slice(acc, update, it, z)
  dus = bf16[4,8] add(dus_inner, acc))"),

    // Dropping a fused pad with a parameter padding value orphans the param.
    RawDusCase("PeeledPadFusionWithParameterPadValueSkipped", false, 0)
        .Peeled(R"(  x0 = bf16[1,8] parameter(0)
  pv = bf16[] parameter(1)
  zero_init = bf16[4,8] fusion(x0, pv), kind=kLoop, calls=fused_pad)")
        .WithComputations(R"(
fused_pad {
  p0 = bf16[1,8] parameter(0)
  pv = bf16[] parameter(1)
  ROOT pad = bf16[4,8] pad(p0, pv), padding=0_3x0_0
}
)"),

    RawDusCase("ShardedWhileSkipped", false, 0)
        .WithWhileExtra(
            "\n      sharding={{replicated}, {devices=[2,1]<=[2]}},"),

    // Init value is irrelevant when the slot is dead-input; -inf elides too.
    RawDusCase("BroadcastNegInfInitElidable", true, 1)
        .WithInitExpr(R"(  cneg = bf16[] constant(-inf)
  zero_init = bf16[4,8] broadcast(cneg), dimensions={})"),

    RawDusCase("TightTileDim1", true, 1).WithBodyDus(R"(  z = s32[] constant(0)
  two = s32[] constant(2)
  two_it = s32[] multiply(it, two)
  update_tile = bf16[4,2] constant({ {1,1}, {1,1}, {1,1}, {1,1} })
  dus = bf16[4,8] dynamic-update-slice(acc, update_tile, z, two_it))"),
};

class DusRawDusParam : public DusAccumulatorZeroInitEliminationTest,
                       public ::testing::WithParamInterface<RawDusCase> {};

TEST_P(DusRawDusParam, ParameterisedScan) {
  const RawDusCase& c = GetParam();
  SCOPED_TRACE(c.name);
  const std::string hlo = MakeRawDusHlo(c);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_EQ(changed, c.expected_changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), c.expected_buffers);

  if (c.check_first_slot) {
    const HloInstruction* w = hlo_query::GetFirstInstructionWithOpcode(
        *module->entry_computation(), HloOpcode::kWhile);
    ASSERT_NE(w, nullptr);
    const HloInstruction* init = w->operand(0);
    ASSERT_EQ(init->opcode(), HloOpcode::kTuple);
    const HloInstruction* slot1 = init->operand(1);
    EXPECT_EQ(slot1->opcode(), HloOpcode::kCustomCall);
    EXPECT_EQ(slot1->custom_call_target(), "AllocateBuffer");
  }
}

INSTANTIATE_TEST_SUITE_P(DusAccumulatorZeroInitEliminationTest, DusRawDusParam,
                         ::testing::ValuesIn(kRawDusCases),
                         [](const ::testing::TestParamInfo<RawDusCase>& info) {
                           return std::string(info.param.name);
                         });

// Scaffold B: single-slot fusion-DUS scan with broadcast(0) init. Each Case
// declares the fusions it calls and the body line invoking them.
struct FusionDusCase {
  absl::string_view name;
  bool expected_changed;
  int64_t expected_buffers;
  absl::string_view fusion_decls;  // required: pre-body fusion decls
  absl::string_view body_dus;      // required: body's `dus = ...` line
  absl::string_view update_shape = "bf16[1,8]";
  absl::string_view update_const = "{ {1,1,1,1,1,1,1,1} }";
  absl::string_view init_expr = R"(  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] broadcast(c0), dimensions={})";
  absl::string_view while_extra = "";

  constexpr FusionDusCase(absl::string_view name, bool expected_changed,
                          int64_t expected_buffers,
                          absl::string_view fusion_decls,
                          absl::string_view body_dus)
      : name(name),
        expected_changed(expected_changed),
        expected_buffers(expected_buffers),
        fusion_decls(fusion_decls),
        body_dus(body_dus) {}

  constexpr FusionDusCase WithUpdateShape(absl::string_view val) const {
    FusionDusCase c = *this;
    c.update_shape = val;
    return c;
  }
  constexpr FusionDusCase WithUpdateConst(absl::string_view val) const {
    FusionDusCase c = *this;
    c.update_const = val;
    return c;
  }
  constexpr FusionDusCase WithInitExpr(absl::string_view val) const {
    FusionDusCase c = *this;
    c.init_expr = val;
    return c;
  }
  constexpr FusionDusCase WithWhileExtra(absl::string_view val) const {
    FusionDusCase c = *this;
    c.while_extra = val;
    return c;
  }
};

std::string MakeFusionDusHlo(const FusionDusCase& c) {
  return absl::StrReplaceAll(
      R"(
HloModule m
$FUSION_DECLS
body {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  update = $UPDATE_SHAPE constant($UPDATE_CONST)
$BODY_DUS
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
$INIT_EXPR
  init = (s32[], bf16[4,8]) tuple(z, zero_init)
  w = (s32[], bf16[4,8]) while(init), condition=cond, body=body,$WHILE_EXTRA
      backend_config={"known_trip_count":{"n":"4"},
                      "known_init_step":{"init":"0","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)",
      {{"$FUSION_DECLS", c.fusion_decls},
       {"$BODY_DUS", c.body_dus},
       {"$UPDATE_SHAPE", c.update_shape},
       {"$UPDATE_CONST", c.update_const},
       {"$INIT_EXPR", c.init_expr},
       {"$WHILE_EXTRA", c.while_extra}});
}

const FusionDusCase kFusionDusCases[] = {
    FusionDusCase("FusionWrappedDus", true, 1,
                  R"(
%fused_dus (p0: bf16[4,8], p1: s32[], p2: bf16[1,8]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = s32[] parameter(1)
  %p2 = bf16[1,8] parameter(2)
  %z = s32[] constant(0)
  ROOT %dus = bf16[4,8] dynamic-update-slice(%p0, %p2, %p1, %z)
}
)",
                  "  dus = bf16[4,8] fusion(acc, it, update), kind=kLoop, "
                  "calls=%fused_dus"),

    FusionDusCase("PlainLoopFusionWithDescendingDus", true, 1,
                  R"(
%fused_dus (p0: bf16[4,8], p1: bf16[1,8], p2: s32[]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = bf16[1,8] parameter(1)
  %p2 = s32[] parameter(2)
  %three = s32[] constant(3)
  %idx = s32[] subtract(%three, %p2)
  %z = s32[] constant(0)
  ROOT %dus = bf16[4,8] dynamic-update-slice(%p0, %p1, %idx, %z)
}
)",
                  "  dus = bf16[4,8] fusion(acc, update, it), kind=kLoop, "
                  "calls=%fused_dus"),

    FusionDusCase("KInputFusionRootedAtDus", true, 1,
                  R"(
%reduce_add (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %r = f32[] add(%a, %b)
}

%fused_dus (p0: bf16[4,8], p1: f32[2,1,8], p2: s32[]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = f32[2,1,8] parameter(1)
  %zc = f32[] constant(0)
  %red = f32[1,8] reduce(%p1, %zc), dimensions={0}, to_apply=%reduce_add
  %cvt = bf16[1,8] convert(%red)
  %p2 = s32[] parameter(2)
  %z = s32[] constant(0)
  ROOT %dus = bf16[4,8] dynamic-update-slice(%p0, %cvt, %p2, %z)
}
)",
                  "  dus = bf16[4,8] fusion(acc, update, it), kind=kInput, "
                  "calls=%fused_dus")
        .WithUpdateShape("f32[2,1,8]")
        .WithUpdateConst("{ { {1,1,1,1,1,1,1,1} }, { {2,2,2,2,2,2,2,2} } }"),

    FusionDusCase("PlainLoopFusionRootNotDusSkipped", false, 0,
                  R"(
%fused_add (p0: bf16[4,8], p1: bf16[4,8]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = bf16[4,8] parameter(1)
  ROOT %a = bf16[4,8] add(%p0, %p1)
}
)",
                  R"(  c1 = bf16[] constant(1)
  ones = bf16[4,8] broadcast(c1), dimensions={}
  dus = bf16[4,8] fusion(acc, ones), kind=kLoop, calls=%fused_add)"),

    // The fusion runs DUS on a bitcast-permuted accumulator and the body
    // bitcasts the fusion output back; the walker must peel both bitcasts
    // and still recognise the inner DUS as covering.
    FusionDusCase(
        "LayoutPermutedBitcastDusFusion", true, 1,
        R"(
%fused_layout_dus (p0: bf16[4,8], p1: bf16[1,8], p2: s32[]) -> bf16[8,4] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = bf16[1,8] parameter(1)
  %p2 = s32[] parameter(2)
  %p0_perm = bf16[8,4] bitcast(%p0)
  %u_perm = bf16[8,1] bitcast(%p1)
  %z = s32[] constant(0)
  ROOT %dus = bf16[8,4] dynamic-update-slice(%p0_perm, %u_perm, %z, %p2)
}
)",
        R"(  fout = bf16[8,4] fusion(acc, update, it), kind=kLoop, calls=%fused_layout_dus
  dus = bf16[4,8] bitcast(fout))"),
};

class DusFusionDusParam : public DusAccumulatorZeroInitEliminationTest,
                          public ::testing::WithParamInterface<FusionDusCase> {
};

TEST_P(DusFusionDusParam, ParameterisedScan) {
  const FusionDusCase& c = GetParam();
  SCOPED_TRACE(c.name);
  const std::string hlo = MakeFusionDusHlo(c);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_EQ(changed, c.expected_changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), c.expected_buffers);
}

INSTANTIATE_TEST_SUITE_P(
    DusAccumulatorZeroInitEliminationTest, DusFusionDusParam,
    ::testing::ValuesIn(kFusionDusCases),
    [](const ::testing::TestParamInfo<FusionDusCase>& info) {
      return std::string(info.param.name);
    });

// Scaffold C: single-slot scan whose init is built by a fusion call (possibly
// via tuple/GTE). Body may be raw or fusion DUS.
struct FusionInitCase {
  absl::string_view name;
  bool expected_changed;
  int64_t expected_buffers;
  absl::string_view fusion_decls;  // pre-body fusion decls
  absl::string_view init_expr;     // declares `zero_init` (and any `tup = ...`)
  absl::string_view body_dus = R"(  z = s32[] constant(0)
  dus = bf16[4,8] dynamic-update-slice(acc, update, it, z))";

  constexpr FusionInitCase(absl::string_view name, bool expected_changed,
                           int64_t expected_buffers,
                           absl::string_view fusion_decls,
                           absl::string_view init_expr)
      : name(name),
        expected_changed(expected_changed),
        expected_buffers(expected_buffers),
        fusion_decls(fusion_decls),
        init_expr(init_expr) {}

  constexpr FusionInitCase WithBodyDus(absl::string_view val) const {
    FusionInitCase c = *this;
    c.body_dus = val;
    return c;
  }
};

std::string MakeFusionInitHlo(const FusionInitCase& c) {
  return absl::StrReplaceAll(
      R"(
HloModule m
$FUSION_DECLS
body {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
$BODY_DUS
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
$INIT_EXPR
  init = (s32[], bf16[4,8]) tuple(z, zero_init)
  w = (s32[], bf16[4,8]) while(init), condition=cond, body=body,
      backend_config={"known_trip_count":{"n":"4"},
                      "known_init_step":{"init":"0","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)",
      {{"$FUSION_DECLS", c.fusion_decls},
       {"$INIT_EXPR", c.init_expr},
       {"$BODY_DUS", c.body_dus}});
}

const FusionInitCase kFusionInitCases[] = {
    FusionInitCase("BroadcastZeroFusionInit", true, 1,
                   R"(
%fused_broadcast () -> bf16[4,8] {
  %c0 = bf16[] constant(0)
  ROOT %b = bf16[4,8] broadcast(%c0), dimensions={}
}
)",
                   "  zero_init = bf16[4,8] fusion(), kind=kLoop, "
                   "calls=%fused_broadcast"),

    FusionInitCase(
        "TupleOutputBroadcastFusionInit", true, 1,
        R"(
%fused_broadcast_tuple () -> (bf16[4,8], bf16[4,16]) {
  %c0 = bf16[] constant(0)
  %b0 = bf16[4,8] broadcast(%c0), dimensions={}
  %b1 = bf16[4,16] broadcast(%c0), dimensions={}
  ROOT %t = (bf16[4,8], bf16[4,16]) tuple(%b0, %b1)
}
)",
        R"(  tup = (bf16[4,8], bf16[4,16]) fusion(), kind=kLoop, calls=%fused_broadcast_tuple
  zero_init = bf16[4,8] get-tuple-element(tup), index=0)"),

    FusionInitCase(
        "PlainLoopFusionWithTupleBroadcastInit", true, 1,
        R"(
%fused_broadcast_tuple () -> (bf16[4,8], bf16[4,8]) {
  %c0 = bf16[] constant(0)
  %b = bf16[4,8] broadcast(%c0), dimensions={}
  %cp = bf16[4,8] copy(%b)
  ROOT %t = (bf16[4,8], bf16[4,8]) tuple(%b, %cp)
}

%fused_dus (p0: bf16[4,8], p1: bf16[1,8], p2: s32[]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = bf16[1,8] parameter(1)
  %p2 = s32[] parameter(2)
  %z = s32[] constant(0)
  ROOT %dus = bf16[4,8] dynamic-update-slice(%p0, %p1, %p2, %z)
}
)",
        R"(  tup = (bf16[4,8], bf16[4,8]) fusion(), kind=kLoop, calls=%fused_broadcast_tuple
  zero_init = bf16[4,8] get-tuple-element(tup), index=1)")
        .WithBodyDus("  dus = bf16[4,8] fusion(acc, update, it), kind=kLoop, "
                     "calls=%fused_dus"),
};

class DusFusionInitParam
    : public DusAccumulatorZeroInitEliminationTest,
      public ::testing::WithParamInterface<FusionInitCase> {};

TEST_P(DusFusionInitParam, ParameterisedScan) {
  const FusionInitCase& c = GetParam();
  SCOPED_TRACE(c.name);
  const std::string hlo = MakeFusionInitHlo(c);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_EQ(changed, c.expected_changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), c.expected_buffers);
}

INSTANTIATE_TEST_SUITE_P(
    DusAccumulatorZeroInitEliminationTest, DusFusionInitParam,
    ::testing::ValuesIn(kFusionInitCases),
    [](const ::testing::TestParamInfo<FusionInitCase>& info) {
      return std::string(info.param.name);
    });

// Cases that don't fit any scaffold — inline HLO for clarity.

TEST_F(DusAccumulatorZeroInitEliminationTest, MultiSlotScanEliminatesBoth) {
  constexpr absl::string_view kHlo = R"(
HloModule m

body {
  p = (s32[], bf16[4,8], bf16[4,16]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  acc1 = bf16[4,16] get-tuple-element(p), index=2
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
  update1 = bf16[1,16] constant({ {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2} })
  z = s32[] constant(0)
  dus = bf16[4,8] dynamic-update-slice(acc, update, it, z)
  dus1 = bf16[4,16] dynamic-update-slice(acc1, update1, it, z)
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8], bf16[4,16]) tuple(next, dus, dus1)
}

cond {
  p = (s32[], bf16[4,8], bf16[4,16]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] broadcast(c0), dimensions={}
  zi1 = bf16[4,16] broadcast(c0), dimensions={}
  init = (s32[], bf16[4,8], bf16[4,16]) tuple(z, zero_init, zi1)
  w = (s32[], bf16[4,8], bf16[4,16]) while(init), condition=cond, body=body,
      backend_config={"known_trip_count":{"n":"4"},
                      "known_init_step":{"init":"0","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), 2);
}

// Trip count is 4 but acc is bf16[8,8] — DUS covers only half, so skip.
TEST_F(DusAccumulatorZeroInitEliminationTest, TripCountMismatchSkipped) {
  constexpr absl::string_view kHlo = R"(
HloModule m

body {
  p = (s32[], bf16[8,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[8,8] get-tuple-element(p), index=1
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
  z = s32[] constant(0)
  dus = bf16[8,8] dynamic-update-slice(acc, update, it, z)
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[8,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[8,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[8,8] broadcast(c0), dimensions={}
  init = (s32[], bf16[8,8]) tuple(z, zero_init)
  w = (s32[], bf16[8,8]) while(init), condition=cond, body=body,
      backend_config={"known_trip_count":{"n":"4"},
                      "known_init_step":{"init":"0","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[8,8] get-tuple-element(w), index=1
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), 0);
}

// The raw peeled pad becomes a DUS of the peeled slice into an AllocateBuffer.
TEST_F(DusAccumulatorZeroInitEliminationTest, PeeledPadBecomesDusOfAlloc) {
  const HloModuleConfig config = ConfigWithFlag();
  RunAndFilecheckHloRewrite(
      MakeRawDusHlo(RawDusCase("Raw", true, 1).Peeled(kPeeledFirstInit)),
      DusAccumulatorZeroInitElimination(), R"(
// CHECK-DAG:  %[[X0:.*]] = bf16[1,8]{1,0} parameter(0)
// CHECK-DAG:  %[[ALLOC:.*]] = bf16[4,8]{1,0} custom-call(), custom_call_target="AllocateBuffer"
// CHECK:      %[[DUS:.*]] = bf16[4,8]{1,0} dynamic-update-slice(%[[ALLOC]], %[[X0]], %{{.*}}, %{{.*}})
// CHECK:      tuple(%{{.*}}, %[[DUS]])
)",
      [](HloModule* module) {
        EXPECT_EQ(hlo_query::GetFirstInstructionWithOpcode(
                      *module->entry_computation(), HloOpcode::kPad),
                  nullptr);
      },
      &config);
}

// Post-fusion form, as in the layer scans of a train step: the peeled
// iteration's pad is the root of a loop fusion that also computes the slice.
// The fusion's pad root becomes an in-place DUS into a new AllocateBuffer
// operand.
TEST_F(DusAccumulatorZeroInitEliminationTest, PeeledPadFusionBecomesDusFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule m

fused_pad {
  p0 = bf16[8] parameter(0)
  neg = bf16[8] negate(p0)
  b = bf16[1,8] bitcast(neg)
  c0 = bf16[] constant(0)
  ROOT pad = bf16[4,8] pad(b, c0), padding=0_3x0_0
}

fused_dus {
  acc = bf16[4,8] parameter(0)
  upd = bf16[1,8] parameter(1)
  idx = s32[] parameter(2)
  z = s32[] constant(0)
  ROOT dus = bf16[4,8] dynamic-update-slice(acc, upd, idx, z)
}

body {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
  dus = bf16[4,8] fusion(acc, update, it), kind=kLoop, calls=fused_dus
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  iv0 = s32[] constant(1)
  x0 = bf16[8] parameter(0)
  peeled_init = bf16[4,8] fusion(x0), kind=kLoop, calls=fused_pad
  init = (s32[], bf16[4,8]) tuple(iv0, peeled_init)
  w = (s32[], bf16[4,8]) while(init), condition=cond, body=body,
      backend_config={"known_trip_count":{"n":"3"},
                      "known_init_step":{"init":"1","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)";
  const HloModuleConfig config = ConfigWithFlag();
  RunAndFilecheckHloRewrite(kHlo, DusAccumulatorZeroInitElimination(), R"(
// CHECK:      %[[FUSED:.*]] ({{.*}}: bf16[8], {{.*}}: bf16[4,8]) -> bf16[4,8] {
// CHECK-DAG:    %[[B:.*]] = bf16[1,8]{1,0} bitcast(
// CHECK-DAG:    %[[BUF:.*]] = bf16[4,8]{1,0} parameter(1)
// CHECK:        ROOT %{{.*}} = bf16[4,8]{1,0} dynamic-update-slice(%[[BUF]], %[[B]], %{{.*}}, %{{.*}})
// CHECK:      ENTRY
// CHECK-DAG:    %[[X0:.*]] = bf16[8]{0} parameter(0)
// CHECK-DAG:    %[[ALLOC:.*]] = bf16[4,8]{1,0} custom-call(), custom_call_target="AllocateBuffer"
// CHECK:        %[[INIT:.*]] = bf16[4,8]{1,0} fusion(%[[X0]], %[[ALLOC]]), kind=kLoop, calls=%[[FUSED]]
// CHECK:        tuple(%{{.*}}, %[[INIT]])
)",
                            /*after_pass_checks=*/nullptr, &config);
}

// Reverse scan after fusion: the body computes its slot, 3 - i, in a
// multi-output fusion together with a dynamic-slice that also uses it, and the
// DUS fusion takes the index through a get-tuple-element of that fusion.
TEST_F(DusAccumulatorZeroInitEliminationTest,
       PeeledReverseScanIndexFromMultiOutputFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule m

fused_idx {
  table = f32[4,1] parameter(0)
  it = s32[] parameter(1)
  three = s32[] constant(3)
  idx = s32[] subtract(three, it)
  z = s32[] constant(0)
  ds = f32[1,1] dynamic-slice(table, idx, z), dynamic_slice_sizes={1,1}
  ROOT t = (f32[1,1], s32[]) tuple(ds, idx)
}

fused_dus {
  acc = bf16[4,8] parameter(0)
  idx = s32[] parameter(1)
  upd = bf16[1,8] parameter(2)
  z = s32[] constant(0)
  ROOT dus = bf16[4,8] dynamic-update-slice(acc, upd, idx, z)
}

body {
  p = (s32[], bf16[4,8], f32[4,1]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  table = f32[4,1] get-tuple-element(p), index=2
  mof = (f32[1,1], s32[]) fusion(table, it), kind=kLoop, calls=fused_idx
  idx = s32[] get-tuple-element(mof), index=1
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
  dus = bf16[4,8] fusion(acc, idx, update), kind=kLoop, calls=fused_dus
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8], f32[4,1]) tuple(next, dus, table)
}

cond {
  p = (s32[], bf16[4,8], f32[4,1]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  iv0 = s32[] constant(1)
  x0 = bf16[1,8] parameter(0)
  table = f32[4,1] parameter(1)
  c0 = bf16[] constant(0)
  peeled_init = bf16[4,8] pad(x0, c0), padding=3_0x0_0
  init = (s32[], bf16[4,8], f32[4,1]) tuple(iv0, peeled_init, table)
  w = (s32[], bf16[4,8], f32[4,1]) while(init), condition=cond, body=body,
      backend_config={"known_trip_count":{"n":"3"},
                      "known_init_step":{"init":"1","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), 1);
}

}  // namespace
}  // namespace xla::gpu
