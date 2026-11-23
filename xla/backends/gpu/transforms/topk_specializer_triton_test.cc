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

// Routing of TopK to the Triton TopK kernels on ROCm. Runs on the CPU with the
// device descriptions of TestGpuDeviceInfo.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/codegen/triton/topk_config.h"
#include "xla/backends/gpu/transforms/topk_specializer.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/topk_rewriter.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

enum class TritonTopKFlags {
  kOff,
  kOn,
  kForced,
  kOnU64Adapter,
  kForcedU64Adapter
};

class TopkSpecializerTritonTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_triton_topk(
        flags_ != TritonTopKFlags::kOff);
    debug_options.set_xla_gpu_experimental_force_triton_topk(
        flags_ == TritonTopKFlags::kForced ||
        flags_ == TritonTopKFlags::kForcedU64Adapter);
    debug_options.set_xla_gpu_experimental_triton_topk_u64_adapter(
        flags_ == TritonTopKFlags::kOnU64Adapter ||
        flags_ == TritonTopKFlags::kForcedU64Adapter);
    return debug_options;
  }

  // Parses a module whose root is `ROOT t = (out, idx) custom-call(p)` with
  // target "TopK" and the HLO attributes `extra_attrs`, if any.
  absl::StatusOr<std::unique_ptr<HloModule>> ParseTopK(
      absl::string_view in, absl::string_view out, absl::string_view idx,
      absl::string_view backend_config, TritonTopKFlags flags,
      absl::string_view extra_attrs = "") {
    std::string config_attr =
        backend_config.empty()
            ? ""
            : absl::Substitute(", backend_config=$0", backend_config);
    if (!extra_attrs.empty()) {
      absl::StrAppend(&config_attr, ", ", extra_attrs);
    }
    flags_ = flags;
    return ParseAndReturnVerifiedModule(absl::Substitute(
        R"hlo(
      HloModule m
      ENTRY e {
        p = $0 parameter(0)
        ROOT t = ($1, $2) custom-call(p), custom_call_target="TopK"$3
      })hlo",
        in, out, idx, config_attr));
  }

  static bool HasTritonTopK(const HloModule& module) {
    for (const HloComputation* comp : module.computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        auto* cc = DynCast<HloCustomCallInstruction>(instr);
        if (cc != nullptr && IsTritonTopK(*cc)) {
          return true;
        }
      }
    }
    return false;
  }

  // Expects the root to be the original TopK rewritten to a marked __gpu$TopK
  // on the same operand, with the same shape and backend config fields.
  static void ExpectMarkedTopK(const HloModule& module, bool stable,
                               ComparisonOrder order) {
    const HloInstruction* root = module.entry_computation()->root_instruction();
    auto* topk = DynCast<HloCustomCallInstruction>(root);
    ASSERT_NE(topk, nullptr) << root->ToString();
    EXPECT_EQ(topk->custom_call_target(), "__gpu$TopK");
    EXPECT_TRUE(IsTritonTopK(*topk));
    EXPECT_EQ(topk->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(topk->shape().tuple_shapes().size(), 2);
    EXPECT_EQ(hlo_instruction_utils::IsTopKStable(topk), stable);
    EXPECT_EQ(hlo_instruction_utils::GetTopKOrder(topk), order);
  }

  // The flags of the modules parsed next.
  TritonTopKFlags flags_ = TritonTopKFlags::kOff;
};

TEST_F(TopkSpecializerTritonTest, FlagOffKeepsTopK) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                                 TritonTopKFlags::kOff));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(TopkSpecializerTritonTest,
       FlagOnKeepsShapesWhereTheDefaultPathIsFaster) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("f32[8,16384]", "f32[8,128]", "s32[8,128]", "",
                                 TritonTopKFlags::kOn));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(TopkSpecializerTritonTest, FlagOnRoutesShapesWhereTritonIsFaster) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("bf16[24576,64]", "bf16[24576,6]",
                                 "s32[24576,6]", "", TritonTopKFlags::kOn));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  ExpectMarkedTopK(*module, /*stable=*/true, ComparisonOrder::kTotal);
}

TEST_F(TopkSpecializerTritonTest, FlagOnDoesNotRouteOnUnmeasuredGpus) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("bf16[24576,64]", "bf16[24576,6]",
                                 "s32[24576,6]", "", TritonTopKFlags::kOn));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI300DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(HasTritonTopK(*module));
}

struct RoutedCase {
  std::string name;
  std::string in, out, idx, backend_config;
  bool stable;
  ComparisonOrder order;
};

class TopkSpecializerTritonRoutedTest
    : public TopkSpecializerTritonTest,
      public ::testing::WithParamInterface<RoutedCase> {};

TEST_P(TopkSpecializerTritonRoutedTest, ForcedTopKIsMarked) {
  const RoutedCase& c = GetParam();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK(c.in, c.out, c.idx, c.backend_config,
                                 TritonTopKFlags::kForced));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  ExpectMarkedTopK(*module, c.stable, c.order);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, TopkSpecializerTritonRoutedTest,
    ::testing::Values(
        RoutedCase{"StableTotalF32", "f32[8,1000]", "f32[8,32]", "s32[8,32]",
                   "", true, ComparisonOrder::kTotal},
        RoutedCase{"UnstablePartialBf16", "bf16[24576,64]", "bf16[24576,6]",
                   "s32[24576,6]", R"({is_stable = false, order = "PARTIAL"})",
                   false, ComparisonOrder::kPartial},
        RoutedCase{"StablePartialF32", "f32[16,4096]", "f32[16,8]", "s32[16,8]",
                   R"({order = "PARTIAL"})", true, ComparisonOrder::kPartial},
        RoutedCase{"UnstableTotalF32", "f32[32,2048]", "f32[32,64]",
                   "s32[32,64]", R"({is_stable = false})", false,
                   ComparisonOrder::kTotal},
        RoutedCase{"UnstableTotalBf16", "bf16[32,2048]", "bf16[32,64]",
                   "s32[32,64]", R"({is_stable = false})", false,
                   ComparisonOrder::kTotal},
        RoutedCase{"RankOne", "f32[4096]", "f32[16]", "s32[16]", "", true,
                   ComparisonOrder::kTotal},
        RoutedCase{"ShortRow", "f32[3,16]", "f32[3,2]", "s32[3,2]", "", true,
                   ComparisonOrder::kTotal},
        RoutedCase{"MaxK", "f32[7,1000]", "f32[7,128]", "s32[7,128]", "", true,
                   ComparisonOrder::kTotal}),
    [](const ::testing::TestParamInfo<RoutedCase>& info) {
      return info.param.name;
    });

struct RejectedCase {
  std::string name;
  std::string in, out, idx;
};

class TopkSpecializerTritonRejectedTest
    : public TopkSpecializerTritonTest,
      public ::testing::WithParamInterface<RejectedCase> {};

TEST_P(TopkSpecializerTritonRejectedTest, ForcedTopKIsNotMarked) {
  const RejectedCase& c = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopK(c.in, c.out, c.idx, "", TritonTopKFlags::kForced));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(HasTritonTopK(*module));
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, TopkSpecializerTritonRejectedTest,
    ::testing::Values(
        RejectedCase{"F16", "f16[8,1000]", "f16[8,32]", "s32[8,32]"},
        RejectedCase{"U64", "u64[8,1000]", "u64[8,32]", "s32[8,32]"},
        RejectedCase{"KAboveLimit", "f32[8,1000]", "f32[8,129]", "s32[8,129]"},
        RejectedCase{"KCloseToN", "f32[8,16]", "f32[8,14]", "s32[8,14]"},
        RejectedCase{"EmptyBatch", "f32[0,1000]", "f32[0,32]", "s32[0,32]"},
        // 1024 chunks of 16384 leave 131072 candidates to merge.
        RejectedCase{"TooManyCandidates", "f32[1,16777216]", "f32[1,128]",
                     "s32[1,128]"}),
    [](const ::testing::TestParamInfo<RejectedCase>& info) {
      return info.param.name;
    });

class TopkSpecializerTritonU64AdapterTest
    : public TopkSpecializerTritonTest,
      public ::testing::WithParamInterface<RoutedCase> {
 protected:
  // The only __gpu$TopK of `module`, or nullptr.
  static const HloCustomCallInstruction* FindGpuTopK(const HloModule& module) {
    const HloCustomCallInstruction* topk = nullptr;
    for (const HloInstruction* inst :
         module.entry_computation()->instructions()) {
      if (inst->IsCustomCall("__gpu$TopK")) {
        EXPECT_EQ(topk, nullptr) << inst->ToString();
        topk = Cast<HloCustomCallInstruction>(inst);
      }
    }
    return topk;
  }
};

TEST_P(TopkSpecializerTritonU64AdapterTest, ForcedTopKUsesTheU64Adapter) {
  const RoutedCase& c = GetParam();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK(c.in, c.out, c.idx, c.backend_config,
                                 TritonTopKFlags::kForcedU64Adapter));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);

  const HloCustomCallInstruction* topk = FindGpuTopK(*module);
  ASSERT_NE(topk, nullptr) << module->ToString();
  EXPECT_TRUE(IsTritonTopK(*topk));
  // The keys are distinct, so the adapter clears the stability.
  EXPECT_FALSE(hlo_instruction_utils::IsTopKStable(topk));
  EXPECT_EQ(hlo_instruction_utils::GetTopKOrder(topk), c.order);
  EXPECT_EQ(topk->operand(0)->shape().element_type(), U64);
  ASSERT_EQ(topk->shape().tuple_shapes().size(), 3);
  EXPECT_EQ(topk->shape().tuple_shapes(0).element_type(), U64);
  EXPECT_EQ(topk->shape().tuple_shapes(1).element_type(), S32);
  EXPECT_EQ(topk->shape().tuple_shapes(2).ToString(), "u8[0]");

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->shape().ToString(),
            absl::StrCat("(", c.out, ", ", c.idx, ")"));
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    EXPECT_FALSE(inst->IsCustomCall("TopK")) << inst->ToString();
  }
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, TopkSpecializerTritonU64AdapterTest,
    ::testing::Values(
        RoutedCase{"StableTotalF32", "f32[8,1000]", "f32[8,32]", "s32[8,32]",
                   "", true, ComparisonOrder::kTotal},
        RoutedCase{"UnstablePartialBf16", "bf16[24576,64]", "bf16[24576,6]",
                   "s32[24576,6]", R"({is_stable = false, order = "PARTIAL"})",
                   false, ComparisonOrder::kPartial},
        RoutedCase{"StableTotalBf16", "bf16[24576,64]", "bf16[24576,6]",
                   "s32[24576,6]", R"({is_stable = true, order = "TOTAL"})",
                   true, ComparisonOrder::kTotal},
        RoutedCase{"RankOne", "f32[4096]", "f32[16]", "s32[16]", "", true,
                   ComparisonOrder::kTotal}),
    [](const ::testing::TestParamInfo<RoutedCase>& info) {
      return info.param.name;
    });

TEST_F(TopkSpecializerTritonU64AdapterTest, LongRowGetsTheScratchInBytes) {
  // 16 chunks of 4096 per row, 8 candidates each: 1024 keys of 8 bytes.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopK("f32[8,65536]", "f32[8,8]", "s32[8,8]",
                R"({order = "PARTIAL"})", TritonTopKFlags::kForcedU64Adapter));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  const HloCustomCallInstruction* topk = FindGpuTopK(*module);
  ASSERT_NE(topk, nullptr) << module->ToString();
  EXPECT_TRUE(IsTritonTopK(*topk));
  EXPECT_EQ(topk->shape().ToString(), "(u64[8,8], s32[8,8], u8[8192])");
}

TEST_F(TopkSpecializerTritonU64AdapterTest, IsOnlyUsedWhenForced) {
  // A shape the speed rule routes without the adapter.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopK("bf16[24576,64]", "bf16[24576,6]", "s32[24576,6]", "",
                TritonTopKFlags::kOnU64Adapter));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(HasTritonTopK(*module));
}

TEST_F(TopkSpecializerTritonTest, ForcedLongRowGetsAScratchOutput) {
  // 16 chunks of 4096 per row, 8 candidates each.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopK("f32[8,65536]", "f32[8,8]", "s32[8,8]",
                R"({order = "PARTIAL"})", TritonTopKFlags::kForced));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  ASSERT_EQ(root->operand_count(), 2);
  auto* topk = DynCast<HloCustomCallInstruction>(root->operand(0)->operand(0));
  ASSERT_NE(topk, nullptr) << root->ToString();
  EXPECT_EQ(topk->custom_call_target(), "__gpu$TopK");
  EXPECT_TRUE(IsTritonTopK(*topk));
  EXPECT_EQ(topk->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(topk->shape().ToString(), "(f32[8,8], s32[8,8], u64[1024])");
  EXPECT_EQ(hlo_instruction_utils::GetTopKOrder(topk),
            ComparisonOrder::kPartial);
  EXPECT_EQ(root->operand(1)->operand(0), topk);
}

TEST_F(TopkSpecializerTritonTest, ForcedTopKKeepsTheMetadata) {
  for (absl::string_view n : {"1000", "65536"}) {
    SCOPED_TRACE(n);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseTopK(absl::StrCat("f32[8,", n, "]"), "f32[8,8]", "s32[8,8]", "",
                  TritonTopKFlags::kForced, R"(metadata={op_name="top_k"})"));
    ASSERT_OK_AND_ASSIGN(
        bool changed,
        RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                   module.get()));
    EXPECT_TRUE(changed);
    for (const HloInstruction* inst :
         module->entry_computation()->instructions()) {
      if (inst->IsCustomCall("__gpu$TopK")) {
        EXPECT_EQ(inst->metadata().op_name(), "top_k");
      }
    }
    EXPECT_TRUE(HasTritonTopK(*module));
  }
}

TEST_F(TopkSpecializerTritonTest, FlagOnRoutesLongRows) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("f32[8,65536]", "f32[8,8]", "s32[8,8]", "",
                                 TritonTopKFlags::kOn));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kTuple);
}

TEST_F(TopkSpecializerTritonTest, OneLongRowWithLargeKKeepsTheDefaultPath) {
  // Merging the candidates of the chunks costs more than the default sort.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("f32[1,1048576]", "f32[1,128]", "s32[1,128]",
                                 "", TritonTopKFlags::kOn));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_FALSE(changed);
  EXPECT_FALSE(HasTritonTopK(*module));
  EXPECT_TRUE(
      module->entry_computation()->root_instruction()->IsCustomCall("TopK"));
}

constexpr absl::string_view kDeferredPackedBf16 =
    R"(frontend_attributes={_xla_topk_deferred_packed_bf16="true"})";

TEST_F(TopkSpecializerTritonTest, DeferredPackedBf16KeepsTheTopKWhenNotRouted) {
  // Not forced, and MI300 has no measurements, so it is not routed. Without
  // the mark, it takes the prebuilt kernel.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> unmarked,
                       ParseTopK("bf16[8,1024]", "bf16[8,16]", "s32[8,16]", "",
                                 TritonTopKFlags::kOn));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI300DeviceInfo()),
                 unmarked.get()));
  EXPECT_TRUE(changed);

  // With the mark, TopkDecomposer later gives it the packed BF16 sort.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> marked,
                       ParseTopK("bf16[8,1024]", "bf16[8,16]", "s32[8,16]", "",
                                 TritonTopKFlags::kOn, kDeferredPackedBf16));
  ASSERT_OK_AND_ASSIGN(
      changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI300DeviceInfo()),
                 marked.get()));
  EXPECT_FALSE(changed);
  const HloInstruction* root = marked->entry_computation()->root_instruction();
  EXPECT_TRUE(root->IsCustomCall("TopK"));
  EXPECT_TRUE(IsTopKWithDeferredPackedBf16(root));
}

TEST_F(TopkSpecializerTritonTest, DeferredPackedBf16IsRoutedWhenForced) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopK("bf16[24576,64]", "bf16[24576,6]", "s32[24576,6]",
                R"({is_stable = true, order = "TOTAL"})",
                TritonTopKFlags::kForced, kDeferredPackedBf16));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                 module.get()));
  EXPECT_TRUE(changed);
  ExpectMarkedTopK(*module, /*stable=*/true, ComparisonOrder::kTotal);
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    EXPECT_FALSE(IsTopKWithDeferredPackedBf16(inst)) << inst->ToString();
  }
}

TEST_F(TopkSpecializerTritonTest, CudaIsNotRouted) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                                 TritonTopKFlags::kForced));
  ASSERT_THAT(
      RunHloPass(TopkSpecializer(TestGpuDeviceInfo::H100SXMDeviceInfo()),
                 module.get()),
      absl_testing::IsOk());
  EXPECT_FALSE(HasTritonTopK(*module));
}

TEST_F(TopkSpecializerTritonTest, OtherRocmArchitecturesAreRoutedWhenForced) {
  for (const se::DeviceDescription& device :
       {TestGpuDeviceInfo::AMDMI300DeviceInfo(),
        TestGpuDeviceInfo::AMDRX7900DeviceInfo()}) {
    SCOPED_TRACE(device.name());
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseTopK("f32[8,1000]", "f32[8,32]", "s32[8,32]", "",
                                   TritonTopKFlags::kForced));
    ASSERT_OK_AND_ASSIGN(bool changed,
                         RunHloPass(TopkSpecializer(device), module.get()));
    EXPECT_TRUE(changed);
    ExpectMarkedTopK(*module, /*stable=*/true, ComparisonOrder::kTotal);
  }
}

TEST_F(TopkSpecializerTritonTest, RowLimitFollowsTheWaveSize) {
  // 16384 keys fit a wave-64 program of 8 warps; a wave-32 device splits them
  // into chunks.
  for (const auto& [device, chunked] :
       {std::pair{TestGpuDeviceInfo::AMDMI350DeviceInfo(), false},
        std::pair{TestGpuDeviceInfo::AMDRX7900DeviceInfo(), true}}) {
    SCOPED_TRACE(device.name());
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseTopK("f32[8,16384]", "f32[8,32]", "s32[8,32]", "",
                                   TritonTopKFlags::kForced));
    ASSERT_OK_AND_ASSIGN(bool changed,
                         RunHloPass(TopkSpecializer(device), module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(HasTritonTopK(*module));
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    EXPECT_EQ(root->opcode(),
              chunked ? HloOpcode::kTuple : HloOpcode::kCustomCall);
  }
}

// The comparator SPMD partitioning gives the merge sort after a TopK: values in
// descending total order, ties by ascending index.
constexpr absl::string_view kSpmdComparator = R"hlo(
        eq = pred[] compare(v0, v1), direction=EQ, type=TOTALORDER
        lt = pred[] compare(i0, i1), direction=LT
        gt = pred[] compare(v0, v1), direction=GT, type=TOTALORDER
        ROOT r = pred[] select(eq, lt, gt))hlo";

class TopkSpecializerTritonSortTest : public TopkSpecializerTritonTest {
 protected:
  // Parses a TopK of `p` = `type[in]` into `type[out]` followed by a sort of
  // its values `v` and of `indices` along `dimension`, compared by
  // `comparator`. `indices` names one of the TopK indices `i`, `i` plus SPMD's
  // offset, the number `o` of the partition in its group times the row length,
  // on either side (`offset`, `offset_lhs`), `i` plus the partition id times
  // the row length (`partition_offset`), offsets that can reach 2^31: a
  // parameter times the row length (`unbounded_offset`), 2^25 times the row
  // length (`far_offset`), 2^31 - 63 (`near_offset`), or `i + i` (`doubled`).
  absl::StatusOr<std::unique_ptr<HloModule>> ParseTopKThenSort(
      absl::string_view type, absl::string_view in, absl::string_view out,
      absl::string_view backend_config, TritonTopKFlags flags,
      absl::string_view comparator = kSpmdComparator,
      absl::string_view indices = "offset", int64_t dimension = 1) {
    std::string config_attr =
        backend_config.empty()
            ? ""
            : absl::Substitute(", backend_config=$0", backend_config);
    absl::string_view row_length = in.substr(in.rfind(',') + 1);
    flags_ = flags;
    return ParseAndReturnVerifiedModule(absl::Substitute(
        R"hlo(
      HloModule m
      cmp {
        v0 = $0[] parameter(0)
        v1 = $0[] parameter(1)
        i0 = s32[] parameter(2)
        i1 = s32[] parameter(3)$4
      }
      ENTRY e {
        p = $0[$1] parameter(0)
        x = s32[] parameter(1)
        t = ($0[$2], s32[$2]) custom-call(p), custom_call_target="TopK"$3
        v = $0[$2] get-tuple-element(t), index=0
        i = s32[$2] get-tuple-element(t), index=1
        n = s32[] constant($7)
        id = u32[] partition-id()
        groups = u32[8] constant({0, 0, 1, 1, 2, 2, 3, 3})
        group = u32[1] dynamic-slice(groups, id), dynamic_slice_sizes={1}
        group_scalar = u32[] reshape(group)
        o = s32[] convert(group_scalar)
        on = s32[] multiply(o, n)
        b = s32[$2] broadcast(on), dimensions={}
        offset = s32[$2] add(i, b)
        offset_lhs = s32[$2] add(b, i)
        id_s32 = s32[] convert(id)
        idn = s32[] multiply(id_s32, n)
        bid = s32[$2] broadcast(idn), dimensions={}
        partition_offset = s32[$2] add(i, bid)
        xn = s32[] multiply(x, n)
        bx = s32[$2] broadcast(xn), dimensions={}
        unbounded_offset = s32[$2] add(i, bx)
        far = s32[2] constant({0, 33554432})
        far_slot = s32[1] dynamic-slice(far, id), dynamic_slice_sizes={1}
        far_scalar = s32[] reshape(far_slot)
        far_n = s32[] multiply(far_scalar, n)
        bfar = s32[$2] broadcast(far_n), dimensions={}
        far_offset = s32[$2] add(i, bfar)
        near = s32[2] constant({0, 2147483585})
        near_slot = s32[1] dynamic-slice(near, id), dynamic_slice_sizes={1}
        near_scalar = s32[] reshape(near_slot)
        bnear = s32[$2] broadcast(near_scalar), dimensions={}
        near_offset = s32[$2] add(i, bnear)
        doubled = s32[$2] add(i, i)
        ROOT s = ($0[$2], s32[$2]) sort(v, $5), dimensions={$6}, to_apply=cmp
      })hlo",
        type, in, out, config_attr, comparator, indices, dimension,
        row_length));
  }

  // Parses an entry computation of `body` with the bf16 SPMD comparator `cmp`,
  // with the Triton TopK kernels forced.
  absl::StatusOr<std::unique_ptr<HloModule>> ParseEntryWithSpmdComparator(
      absl::string_view body) {
    flags_ = TritonTopKFlags::kForced;
    return ParseAndReturnVerifiedModule(absl::Substitute(
        R"hlo(
      HloModule m
      cmp {
        v0 = bf16[] parameter(0)
        v1 = bf16[] parameter(1)
        i0 = s32[] parameter(2)
        i1 = s32[] parameter(3)$0
      }
      ENTRY e {$1
      })hlo",
        kSpmdComparator, body));
  }

  static bool HasSort(const HloModule& module) {
    return absl::c_any_of(module.entry_computation()->instructions(),
                          [](const HloInstruction* inst) {
                            return inst->opcode() == HloOpcode::kSort;
                          });
  }

  absl::StatusOr<bool> RunOnMi350(HloModule* module) {
    return RunHloPass(TopkSpecializer(TestGpuDeviceInfo::AMDMI350DeviceInfo()),
                      module);
  }
};

TEST_F(TopkSpecializerTritonSortTest, RemovesTheSpmdMergeSortAfterTheTopK) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                                         TritonTopKFlags::kForced));
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_FALSE(HasSort(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(root->operand(1)->name(), "offset");
}

// The u64 adapter's values come from its unpack, not from the Triton TopK.
TEST_F(TopkSpecializerTritonSortTest, KeepsTheSortAfterTheU64Adapter) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                                         TritonTopKFlags::kForcedU64Adapter));
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, RemovesTheSortOfOtherIndexForms) {
  for (absl::string_view indices : {"i", "offset_lhs", "partition_offset"}) {
    SCOPED_TRACE(indices);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                          TritonTopKFlags::kForced, kSpmdComparator, indices));
    ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_FALSE(HasSort(*module));
  }
}

TEST_F(TopkSpecializerTritonSortTest, PartitionCountBoundsThePartitionOffset) {
  // The largest partition id times 64, plus 63, must stay below 2^31.
  for (auto [num_partitions, removed] :
       {std::pair{int64_t{8}, true}, std::pair{int64_t{1} << 25, true},
        std::pair{(int64_t{1} << 25) + 1, false}}) {
    SCOPED_TRACE(num_partitions);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                          TritonTopKFlags::kForced, kSpmdComparator,
                          "partition_offset"));
    module->mutable_config().set_num_partitions(num_partitions);
    ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_EQ(HasSort(*module), !removed);
  }
}

TEST_F(TopkSpecializerTritonSortTest, RemovesTheSortAfterAChunkedTopK) {
  // 16 chunks of 4096 per row: the TopK results come through the tuple that
  // drops the scratch output.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopKThenSort("f32", "8,65536", "8,8", "", TritonTopKFlags::kForced));
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_FALSE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, RemovesTheSortOfValuesOnly) {
  flags_ = TritonTopKFlags::kForced;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
    HloModule m
    cmp {
      v0 = bf16[] parameter(0)
      v1 = bf16[] parameter(1)
      ROOT gt = pred[] compare(v0, v1), direction=GT, type=TOTALORDER
    }
    ENTRY e {
      p = bf16[24576,64] parameter(0)
      t = (bf16[24576,6], s32[24576,6]) custom-call(p), custom_call_target="TopK"
      v = bf16[24576,6] get-tuple-element(t), index=0
      ROOT s = bf16[24576,6] sort(v), dimensions={1}, to_apply=cmp
    })hlo"));
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(HasSort(*module));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kGetTupleElement);
}

TEST_F(TopkSpecializerTritonSortTest, KeepsTheSortAfterAPartialOrderTopK) {
  // A PARTIAL order TopK treats -0 and +0 as equal; the sort does not.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopKThenSort("bf16", "24576,64", "24576,6",
                        R"({is_stable = false, order = "PARTIAL"})",
                        TritonTopKFlags::kForced));
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, KeepsTheSortAfterADefaultPathTopK) {
  // The kernels do not take k/n >= 0.85, so the TopK keeps the default path.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseTopKThenSort("f32", "8,16", "8,14", "", TritonTopKFlags::kForced));
  ASSERT_OK(RunOnMi350(module.get()));
  EXPECT_FALSE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, FlagOffKeepsTheSort) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                                         TritonTopKFlags::kOff));
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_TRUE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, KeepsTheSortOfIndicesOfAnotherTopK) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseEntryWithSpmdComparator(R"hlo(
      p = bf16[24576,64] parameter(0)
      q = bf16[24576,64] parameter(1)
      t = (bf16[24576,6], s32[24576,6]) custom-call(p), custom_call_target="TopK"
      u = (bf16[24576,6], s32[24576,6]) custom-call(q), custom_call_target="TopK"
      v = bf16[24576,6] get-tuple-element(t), index=0
      j = s32[24576,6] get-tuple-element(u), index=1
      ROOT s = (bf16[24576,6], s32[24576,6]) sort(v, j), dimensions={1}, to_apply=cmp)hlo"));
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
}

TEST_F(TopkSpecializerTritonSortTest, KeepsASortWithControlDependencies) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseEntryWithSpmdComparator(R"hlo(
      p = bf16[24576,64] parameter(0)
      o = s32[] parameter(1)
      n = s32[] negate(o)
      t = (bf16[24576,6], s32[24576,6]) custom-call(p), custom_call_target="TopK"
      v = bf16[24576,6] get-tuple-element(t), index=0
      i = s32[24576,6] get-tuple-element(t), index=1
      ROOT s = (bf16[24576,6], s32[24576,6]) sort(v, i), dimensions={1}, to_apply=cmp, control-predecessors={n})hlo"));
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
  // No replacement tuple is left behind.
  EXPECT_FALSE(absl::c_any_of(module->entry_computation()->instructions(),
                              [](const HloInstruction* inst) {
                                return inst->opcode() == HloOpcode::kTuple;
                              }));
}

struct KeptSortCase {
  std::string name;
  std::string comparator;
  std::string indices;
  int64_t dimension;
};

class TopkSpecializerTritonKeptSortTest
    : public TopkSpecializerTritonSortTest,
      public ::testing::WithParamInterface<KeptSortCase> {};

TEST_P(TopkSpecializerTritonKeptSortTest, SortIsKept) {
  const KeptSortCase& c = GetParam();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseTopKThenSort("bf16", "24576,64", "24576,6", "",
                                         TritonTopKFlags::kForced, c.comparator,
                                         c.indices, c.dimension));
  ASSERT_OK_AND_ASSIGN(bool changed, RunOnMi350(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HasTritonTopK(*module));
  EXPECT_TRUE(HasSort(*module));
}

INSTANTIATE_TEST_SUITE_P(
    Sorts, TopkSpecializerTritonKeptSortTest,
    ::testing::Values(
        KeptSortCase{"HigherIndexFirst", R"hlo(
        eq = pred[] compare(v0, v1), direction=EQ, type=TOTALORDER
        gt_i = pred[] compare(i0, i1), direction=GT
        gt = pred[] compare(v0, v1), direction=GT, type=TOTALORDER
        ROOT r = pred[] select(eq, gt_i, gt))hlo",
                     "offset", 1},
        KeptSortCase{"Ascending", R"hlo(
        eq = pred[] compare(v0, v1), direction=EQ, type=TOTALORDER
        lt_i = pred[] compare(i0, i1), direction=LT
        lt = pred[] compare(v0, v1), direction=LT, type=TOTALORDER
        ROOT r = pred[] select(eq, lt_i, lt))hlo",
                     "offset", 1},
        KeptSortCase{"PartialOrderCompare", R"hlo(
        eq = pred[] compare(v0, v1), direction=EQ
        lt_i = pred[] compare(i0, i1), direction=LT
        gt = pred[] compare(v0, v1), direction=GT
        ROOT r = pred[] select(eq, lt_i, gt))hlo",
                     "offset", 1},
        KeptSortCase{"SwappedParameters", R"hlo(
        eq = pred[] compare(v0, v1), direction=EQ, type=TOTALORDER
        lt_i = pred[] compare(i1, i0), direction=LT
        gt = pred[] compare(v0, v1), direction=GT, type=TOTALORDER
        ROOT r = pred[] select(eq, lt_i, gt))hlo",
                     "offset", 1},
        KeptSortCase{"PerElementOffset", std::string(kSpmdComparator),
                     "doubled", 1},
        // Could wrap past 2^31 and reorder the indices.
        KeptSortCase{"UnboundedOffset", std::string(kSpmdComparator),
                     "unbounded_offset", 1},
        KeptSortCase{"OffsetPast2To31", std::string(kSpmdComparator),
                     "far_offset", 1},
        KeptSortCase{"IndexPlusOffsetPast2To31", std::string(kSpmdComparator),
                     "near_offset", 1},
        KeptSortCase{"OtherDimension", std::string(kSpmdComparator), "offset",
                     0}),
    [](const ::testing::TestParamInfo<KeptSortCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla::gpu
