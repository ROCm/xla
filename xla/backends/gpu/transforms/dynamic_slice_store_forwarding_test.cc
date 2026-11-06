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

#include "xla/backends/gpu/transforms/dynamic_slice_store_forwarding.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/dynamic_slice_annotator.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion_rewriter_v2.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform_id.h"  // IWYU pragma: keep

namespace xla::gpu {
namespace {

bool IsFakeHero(const HloInstruction* instr) {
  auto* custom_call = DynCast<HloCustomCallInstruction>(instr);
  return custom_call != nullptr &&
         absl::StartsWith(custom_call->custom_call_target(), "fake_");
}

// A scan body storing `value` (BODY_VALUE, which may define other users that
// feed `other`) into row `i` of the stacked loop state.
std::string Scan(absl::string_view body) {
  return absl::StrReplaceAll(R"(
    HloModule test

    body {
      p0 = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8]) parameter(0)
      i = s32[] get-tuple-element(p0), index=0
      input = f32[8,8] get-tuple-element(p0), index=1
      stack = f32[4,8,8] get-tuple-element(p0), index=2
      others = f32[4,8,8] get-tuple-element(p0), index=3
      c0 = s32[] constant(0)
      BODY
      stored = f32[1,8,8] bitcast(value)
      dus = f32[4,8,8] dynamic-update-slice(stack, stored, i, c0, c0)
      other_2d = f32[1,8,8] bitcast(other)
      dus_other = f32[4,8,8] dynamic-update-slice(others, other_2d, i, c0, c0)
      c1 = s32[] constant(1)
      next_i = s32[] add(i, c1)
      ROOT result = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8])
        tuple(next_i, input, dus, dus_other)
    }

    condition {
      p0 = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8]) parameter(0)
      i = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(i, c4), direction=LT
    }

    ENTRY main {
      input = f32[8,8] parameter(0)
      stack = f32[4,8,8] parameter(1)
      others = f32[4,8,8] parameter(2)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8])
        tuple(c0, input, stack, others)
      ROOT while = (s32[], f32[8,8], f32[4,8,8], f32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )",
                             {{"BODY", body}});
}

class DynamicSliceStoreForwardingTest : public HloHardwareIndependentTestBase {
  void SetUp() override {
    auto name = PlatformUtil::CanonicalPlatformName("gpu");
    CHECK_OK(name);
    auto platform_id = PlatformUtil::GetPlatformIdFromCanonicalName(*name);
    CHECK_OK(platform_id);
    platform_id_ = *platform_id;
  }

 protected:
  // Runs the annotator, this pass (if `forward`) and dynamic slice fusion.
  void Run(HloModule* module, bool forward) {
    HloPassPipeline pipeline("test");
    pipeline.AddPass<DynamicSliceAnnotator>();
    DynamicSliceFusionRewriterV2::Options options;
    options.predicate = IsFakeHero;
    if (forward) {
      pipeline.AddPass<DynamicSliceStoreForwarding>(IsFakeHero);
      options.allow_update_chains = true;
    }
    pipeline.AddPass<DynamicSliceFusionRewriterV2>(platform_id_,
                                                   std::move(options));
    ASSERT_OK(pipeline.Run(module).status());
  }

  static const HloComputation* Body(const HloModule* module) {
    return module->GetComputationWithName("body");
  }

  // The dynamic slice fusions in the loop body, and the opcodes of what is
  // left outside them.
  static std::vector<const HloInstruction*> Fusions(const HloModule* module) {
    std::vector<const HloInstruction*> fusions;
    for (const HloInstruction* instr : Body(module)->instructions()) {
      if (instr->opcode() == HloOpcode::kFusion) {
        fusions.push_back(instr);
      }
    }
    return fusions;
  }

  static int64_t Count(const HloModule* module, HloOpcode opcode) {
    return absl::c_count_if(
        Body(module)->instructions(),
        [&](const HloInstruction* instr) { return instr->opcode() == opcode; });
  }

  // The fusion's hero (its custom call).
  static const HloInstruction* Hero(const HloInstruction* fusion) {
    for (const HloInstruction* instr :
         fusion->fused_instructions_computation()->instructions()) {
      if (instr->opcode() == HloOpcode::kCustomCall) {
        return instr;
      }
    }
    return nullptr;
  }

  stream_executor::PlatformId platform_id_;
};

// A hero whose result is stored and also read by an elementwise user: the
// user reads the stored slice back and the hero writes into the stack.
TEST_F(DynamicSliceStoreForwardingTest, HeroWithOtherUser) {
  const std::string hlo = Scan(R"(
      value = f32[8,8] custom-call(input), custom_call_target="fake_hero"
      other = f32[8,8] negate(value))");
  for (bool forward : {false, true}) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    Run(module.get(), forward);
    std::vector<const HloInstruction*> fusions = Fusions(module.get());
    if (!forward) {
      EXPECT_THAT(fusions, ::testing::IsEmpty()) << module->ToString();
      continue;
    }
    ASSERT_EQ(fusions.size(), 1) << module->ToString();
    EXPECT_EQ(fusions[0]->fused_expression_root()->opcode(),
              HloOpcode::kDynamicUpdateSlice);
    // negate reads dynamic-slice(fusion, i).
    const HloInstruction* negate =
        *absl::c_find_if(Body(module.get())->instructions(),
                         [](const HloInstruction* instr) {
                           return instr->opcode() == HloOpcode::kNegate;
                         });
    const HloInstruction* read = negate->operand(0);
    while (read->opcode() == HloOpcode::kBitcast) {
      read = read->operand(0);
    }
    ASSERT_EQ(read->opcode(), HloOpcode::kDynamicSlice) << module->ToString();
    EXPECT_EQ(read->operand(0), fusions[0]);
    EXPECT_TRUE(read->backend_config<GpuBackendConfig>()
                    ->has_dynamic_slice_config());
  }
}

// The in-place pattern of a grouped GEMM followed by a kernel that zeroes the
// rows past the groups in place: the GEMM writes into the stack, the in-place
// kernel updates the stored slice, the other user reads it back.
TEST_F(DynamicSliceStoreForwardingTest, InPlaceHeroOnHeroResult) {
  const std::string hlo = Scan(R"(
      count = s32[] constant(5)
      gemm = f32[64] custom-call(input), custom_call_target="fake_gemm"
      gemm_2d = f32[8,8] bitcast(gemm)
      value = f32[8,8] custom-call(count, gemm_2d),
        custom_call_target="fake_zero_tail",
        output_to_operand_aliasing={{}: (1, {})}
      other = f32[8,8] negate(value))");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  Run(module.get(), /*forward=*/true);
  std::vector<const HloInstruction*> fusions = Fusions(module.get());
  ASSERT_EQ(fusions.size(), 2) << module->ToString();
  // fusion 0: the GEMM storing into the stack; fusion 1: the in-place kernel on
  // fusion 0's slice, storing into the same buffer.
  const HloInstruction* gemm_fusion =
      Hero(fusions[0])->custom_call_target() == "fake_gemm" ? fusions[0]
                                                             : fusions[1];
  const HloInstruction* zero_fusion =
      gemm_fusion == fusions[0] ? fusions[1] : fusions[0];
  EXPECT_EQ(Hero(zero_fusion)->custom_call_target(), "fake_zero_tail");
  EXPECT_EQ(zero_fusion->operand(0), gemm_fusion) << module->ToString();
  EXPECT_EQ(absl::c_count(zero_fusion->operands(), gemm_fusion), 1);
  // The in-place kernel reads the slice it stores: a dynamic-slice and a
  // dynamic-update-slice of the same fusion parameter.
  const HloInstruction* root = zero_fusion->fused_expression_root();
  ASSERT_EQ(root->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* hero = Hero(zero_fusion);
  const HloInstruction* slice = hero->operand(1);
  while (slice->opcode() == HloOpcode::kBitcast) {
    slice = slice->operand(0);
  }
  ASSERT_EQ(slice->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(slice->operand(0), root->operand(0));
  // The old kernel and store are gone; nothing copies the stack.
  EXPECT_EQ(Count(module.get(), HloOpcode::kCustomCall), 0)
      << module->ToString();
  EXPECT_EQ(Count(module.get(), HloOpcode::kDynamicUpdateSlice), 1);
  // negate reads the stored slice.
  EXPECT_EQ(Count(module.get(), HloOpcode::kDynamicSlice), 1);
}

// A fusible producer read by a hero as well: the hero reads the stored slice,
// so the producer has the store as its only user.
TEST_F(DynamicSliceStoreForwardingTest, ProducerReadByHero) {
  const std::string hlo = Scan(R"(
      value = f32[8,8] exponential(input)
      other = f32[8,8] custom-call(value), custom_call_target="fake_hero")");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  Run(module.get(), /*forward=*/true);
  const HloInstruction* exp =
      *absl::c_find_if(Body(module.get())->instructions(),
                       [](const HloInstruction* instr) {
                         return instr->opcode() == HloOpcode::kExponential;
                       });
  ASSERT_EQ(exp->user_count(), 1) << module->ToString();
  // The hero's fusion slices the first store's result and stores into others.
  std::vector<const HloInstruction*> fusions = Fusions(module.get());
  ASSERT_EQ(fusions.size(), 1) << module->ToString();
  const HloInstruction* store = exp->users()[0];
  while (store->opcode() == HloOpcode::kBitcast) {
    store = store->users()[0];
  }
  ASSERT_EQ(store->opcode(), HloOpcode::kDynamicUpdateSlice);
  EXPECT_TRUE(absl::c_linear_search(fusions[0]->operands(), store))
      << module->ToString();
}

// As above with the hero reading the producer through a bitcast, as a grouped
// GEMM reads flattened dispatch rows.
TEST_F(DynamicSliceStoreForwardingTest, ProducerReadByHeroThroughBitcast) {
  const std::string hlo = Scan(R"(
      value = f32[8,8] exponential(input)
      flat = f32[64] bitcast(value)
      hero = f32[64] custom-call(flat), custom_call_target="fake_hero"
      other = f32[8,8] bitcast(hero))");
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  Run(module.get(), /*forward=*/true);
  const HloInstruction* exp =
      *absl::c_find_if(Body(module.get())->instructions(),
                       [](const HloInstruction* instr) {
                         return instr->opcode() == HloOpcode::kExponential;
                       });
  EXPECT_EQ(exp->user_count(), 1) << module->ToString();
  EXPECT_EQ(Fusions(module.get()).size(), 1) << module->ToString();
}

// Users that cannot read a slice without a copy keep the stored value.
TEST_F(DynamicSliceStoreForwardingTest, KeepsValueForTupleUser) {
  const std::string hlo = absl::StrReplaceAll(
      Scan(R"(
      value = f32[8,8] custom-call(input), custom_call_target="fake_hero"
      other = f32[8,8] negate(value))"),
      {{"tuple(next_i, input, dus, dus_other)",
        "tuple(next_i, value, dus, dus_other)"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK(RunHloPass(DynamicSliceAnnotator(), module.get()).status());
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(DynamicSliceStoreForwarding(IsFakeHero), module.get()));
  EXPECT_FALSE(changed) << module->ToString();
}

}  // namespace
}  // namespace xla::gpu
