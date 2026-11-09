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

#include "xla/backends/gpu/transforms/valid_prefix_extent_rewriter.h"

#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/valid_prefix_extent.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

class ValidPrefixExtentRewriterTest : public HloHardwareIndependentTestBase {
 public:
  ValidPrefixExtentRewriterTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

 protected:
  se::DeviceDescription device_info_ = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  GpuAliasInfo alias_info_{device_info_};
  mlir::MLIRContext mlir_context_;
};

TEST_F(ValidPrefixExtentRewriterTest,
       DynamicPadderAssociationSurvivesGpuFusionPipeline) {
  auto run_bridge =
      [&](absl::string_view hlo) -> absl::StatusOr<std::unique_ptr<HloModule>> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    DynamicPadderOptions options;
    options.preserve_valid_prefix = true;
    DynamicPadder dynamic_padder(options);
    ABSL_RETURN_IF_ERROR(dynamic_padder.Run(module.get()).status());
    bool found_marker = false;
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        found_marker |= instruction->IsCustomCall("xla.valid_prefix");
      }
    }
    if (!found_marker) {
      return absl::InternalError(absl::StrCat(
          "DynamicPadder lost valid-prefix dataflow:\n", module->ToString()));
    }

    auto shape_size = [](const Shape& shape) {
      return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
    };
    ABSL_RETURN_IF_ERROR(FusionPipeline(module->config().debug_options(),
                                        shape_size, &alias_info_,
                                        /*thread_pool=*/nullptr, device_info_,
                                        &mlir_context_)
                             .Run(module.get())
                             .status());

    ValidPrefixExtentRewriter rewriter;
    ABSL_RETURN_IF_ERROR(rewriter.Run(module.get()).status());
    HloDCE dce;
    ABSL_RETURN_IF_ERROR(dce.Run(module.get()).status());
    return module;
  };
  auto find_valid_prefix_fusion =
      [](HloModule* module) -> absl::StatusOr<HloInstruction*> {
    for (HloInstruction* instruction :
         module->entry_computation()->instructions()) {
      ABSL_ASSIGN_OR_RETURN(std::optional<ValidPrefixExtent> extent,
                            GetValidPrefixExtent(*instruction));
      if (extent.has_value()) {
        return instruction;
      }
    }
    return nullptr;
  };

  constexpr absl::string_view kSupported = R"(
HloModule ValidPrefixExtentBridge

ENTRY main {
  data = f32[8,4] parameter(0)
  n_input = s32[] parameter(1)
  zero = s32[] constant(0)
  n = s32[] add(n_input, zero)
  dynamic = f32[<=8,4] set-dimension-size(data, n), dimensions={0}
  negated = f32[<=8,4] negate(dynamic)
  ROOT result = f32[<=8,4] exponential(negated)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       run_bridge(kSupported));

  ASSERT_OK_AND_ASSIGN(HloInstruction * valid_prefix_fusion,
                       find_valid_prefix_fusion(module.get()));
  ASSERT_NE(valid_prefix_fusion, nullptr) << module->ToString();
  ASSERT_EQ(valid_prefix_fusion->operand_count(), 2);
  ASSERT_OK_AND_ASSIGN(std::optional<ValidPrefixExtent> extent,
                       GetValidPrefixExtent(*valid_prefix_fusion));
  ASSERT_TRUE(extent.has_value());
  EXPECT_EQ(extent->operand_index, 1);
  EXPECT_EQ(extent->dimension, 0);
  EXPECT_EQ(extent->capacity, 8);
  EXPECT_FALSE(valid_prefix_fusion->fused_parameter(1)->users().empty());
  const Shape& extent_shape = valid_prefix_fusion->operand(1)->shape();
  EXPECT_TRUE(extent_shape.IsArray());
  EXPECT_TRUE(extent_shape.dimensions().empty());
  EXPECT_EQ(extent_shape.element_type(), S32);

  constexpr absl::string_view kStructuralTransfers = R"(
HloModule StructuralValidPrefixTransfers

ENTRY main {
  data = f32[8] parameter(0)
  n = s32[] parameter(1)
  dynamic = f32[<=8] set-dimension-size(data, n), dimensions={0}
  copied = f32[<=8] copy(dynamic)
  broadcast = f32[<=8,4] broadcast(copied), dimensions={0}
  reshaped = f32[<=8,2,2] reshape(broadcast)
  ROOT result = f32[<=8,2,2] negate(reshaped)
}
)";
  ASSERT_OK_AND_ASSIGN(module, run_bridge(kStructuralTransfers));
  ASSERT_OK_AND_ASSIGN(valid_prefix_fusion,
                       find_valid_prefix_fusion(module.get()));
  ASSERT_NE(valid_prefix_fusion, nullptr) << module->ToString();

  constexpr absl::string_view kSharedExtent = R"(
HloModule SharedValidPrefixExtent

ENTRY main {
  lhs_data = f32[8,4] parameter(0)
  rhs_data = f32[8,4] parameter(1)
  n = s32[] parameter(2)
  lhs = f32[<=8,4] set-dimension-size(lhs_data, n), dimensions={0}
  rhs = f32[<=8,4] set-dimension-size(rhs_data, n), dimensions={0}
  sum = f32[<=8,4] add(lhs, rhs)
  ROOT result = f32[<=8,4] exponential(sum)
}
)";
  ASSERT_OK_AND_ASSIGN(module, run_bridge(kSharedExtent));
  ASSERT_OK_AND_ASSIGN(valid_prefix_fusion,
                       find_valid_prefix_fusion(module.get()));
  ASSERT_NE(valid_prefix_fusion, nullptr) << module->ToString();

  constexpr absl::string_view kUnequalExtents = R"(
HloModule UnequalValidPrefixExtents

ENTRY main {
  lhs_data = f32[8,4] parameter(0)
  rhs_data = f32[8,4] parameter(1)
  lhs_n = s32[] parameter(2)
  rhs_n = s32[] parameter(3)
  lhs = f32[<=8,4] set-dimension-size(lhs_data, lhs_n), dimensions={0}
  rhs = f32[<=8,4] set-dimension-size(rhs_data, rhs_n), dimensions={0}
  sum = f32[<=8,4] add(lhs, rhs)
  ROOT result = f32[<=8,4] exponential(sum)
}
)";
  ASSERT_OK_AND_ASSIGN(module, run_bridge(kUnequalExtents));
  ASSERT_OK_AND_ASSIGN(valid_prefix_fusion,
                       find_valid_prefix_fusion(module.get()));
  EXPECT_EQ(valid_prefix_fusion, nullptr);

  constexpr absl::string_view kUnsupported = R"(
HloModule UnsupportedValidPrefixExtent

ENTRY main {
  data = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  dynamic = f32[<=8,4] set-dimension-size(data, n), dimensions={0}
  negated = f32[<=8,4] negate(dynamic)
  ROOT result = f32[4,<=8] transpose(negated), dimensions={1,0}
}
)";
  ASSERT_OK_AND_ASSIGN(module, run_bridge(kUnsupported));
  ASSERT_OK_AND_ASSIGN(valid_prefix_fusion,
                       find_valid_prefix_fusion(module.get()));
  EXPECT_EQ(valid_prefix_fusion, nullptr);
}

TEST_F(ValidPrefixExtentRewriterTest, GroupedGemmConsumesValidPrefixFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule GroupedGemmConsumesValidPrefix

fused_computation {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  bounded = f32[8,4] custom-call(x, n),
    custom_call_target="xla.valid_prefix"
  ROOT negated = f32[8,4] negate(bounded)
}

ENTRY main {
  data = f32[8,4] parameter(0)
  weights = f32[2,4,4] parameter(1)
  group_sizes = s32[2] parameter(2)
  n = s32[] parameter(3)
  epilogue = f32[8,4] fusion(data, n), kind=kLoop,
    calls=fused_computation
  ROOT grouped = f32[8,4] custom-call(epilogue, weights, group_sizes),
    custom_call_target="__cublas$lt$groupedMatmul"
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion =
      module->entry_computation()->GetInstructionWithName("epilogue");
  ASSERT_NE(fusion, nullptr);
  HloInstruction* extent = fusion->mutable_operand(1);
  ValidPrefixExtentRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(std::optional<ValidPrefixExtent> contract,
                       GetValidPrefixExtent(*fusion));
  ASSERT_TRUE(contract.has_value());
  EXPECT_EQ(fusion->operand(contract->operand_index), extent);
}

TEST_F(ValidPrefixExtentRewriterTest, UnknownConsumerKeepsCapacitySemantics) {
  constexpr absl::string_view kHlo = R"(
HloModule OrdinaryConsumer

fused_computation {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  bounded = f32[8,4] custom-call(x, n),
    custom_call_target="xla.valid_prefix"
  ROOT negated = f32[8,4] negate(bounded)
}

ENTRY main {
  data = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  epilogue = f32[8,4] fusion(data, n), kind=kLoop, calls=fused_computation
  ROOT result = f32[8,4] custom-call(epilogue),
    custom_call_target="unknown"
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion =
      module->entry_computation()->GetInstructionWithName("epilogue");
  ASSERT_NE(fusion, nullptr);
  ValidPrefixExtentRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
  ASSERT_OK_AND_ASSIGN(std::optional<ValidPrefixExtent> contract,
                       GetValidPrefixExtent(*fusion));
  EXPECT_FALSE(contract.has_value());
}

TEST_F(ValidPrefixExtentRewriterTest,
       RejectionPropagatesThroughSameExtentFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule PropagatedRejection

first_computation {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  bounded = f32[8,4] custom-call(x, n),
    custom_call_target="xla.valid_prefix"
  ROOT negated = f32[8,4] negate(bounded)
}

second_computation {
  x = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  bounded = f32[8,4] custom-call(x, n),
    custom_call_target="xla.valid_prefix"
  ROOT result = f32[8,4] exponential(bounded)
}

ENTRY main {
  data = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  first = f32[8,4] fusion(data, n), kind=kLoop,
    calls=first_computation
  second = f32[8,4] fusion(first, n), kind=kLoop,
    calls=second_computation
  ROOT result = f32[8,4] custom-call(second),
    custom_call_target="unknown"
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* first =
      module->entry_computation()->GetInstructionWithName("first");
  HloInstruction* second =
      module->entry_computation()->GetInstructionWithName("second");
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);

  ValidPrefixExtentRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
  ASSERT_OK_AND_ASSIGN(std::optional<ValidPrefixExtent> first_contract,
                       GetValidPrefixExtent(*first));
  ASSERT_OK_AND_ASSIGN(std::optional<ValidPrefixExtent> second_contract,
                       GetValidPrefixExtent(*second));
  EXPECT_FALSE(first_contract.has_value());
  EXPECT_FALSE(second_contract.has_value());
}

TEST_F(ValidPrefixExtentRewriterTest, RemovesUnfusedMarkers) {
  constexpr absl::string_view kHlo = R"(
HloModule UnfusedValidPrefix

ENTRY main {
  data = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  ROOT bounded = f32[8,4] custom-call(data, n),
    custom_call_target="xla.valid_prefix"
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ValidPrefixExtentRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction(),
            module->entry_computation()->parameter_instruction(0));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
