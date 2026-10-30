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

#include <gtest/gtest.h>

#include <memory>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/frontend_attributes.h"
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
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

class ValidPrefixExtentRewriterTest
    : public HloHardwareIndependentTestBase {
 public:
  ValidPrefixExtentRewriterTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

 protected:
  se::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  GpuAliasInfo alias_info_{device_info_};
  mlir::MLIRContext mlir_context_;
};

TEST_F(ValidPrefixExtentRewriterTest,
       DynamicPadderAssociationSurvivesGpuFusionPipeline) {
  auto run_bridge = [&](absl::string_view hlo, bool expect_annotation)
      -> absl::StatusOr<std::unique_ptr<HloModule>> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    DynamicPadder dynamic_padder;
    ABSL_RETURN_IF_ERROR(dynamic_padder.Run(module.get()).status());
    if (expect_annotation) {
      bool found_annotation = false;
      for (HloComputation* computation : module->computations()) {
        for (HloInstruction* instruction : computation->instructions()) {
          found_annotation |= instruction->frontend_attributes().map().contains(
              kXlaValidPrefixExtentId);
        }
      }
      if (!found_annotation) {
        return absl::InternalError(
            absl::StrCat("DynamicPadder lost valid-prefix metadata:\n",
                         module->ToString()));
      }
    }

    auto shape_size = [](const Shape& shape) {
      return ShapeUtil::ByteSizeOf(shape);
    };
    ABSL_RETURN_IF_ERROR(
        FusionPipeline(module->config().debug_options(), shape_size,
                       &alias_info_, /*thread_pool=*/nullptr, device_info_,
                       &mlir_context_)
            .Run(module.get())
            .status());

    ValidPrefixExtentRewriter rewriter;
    ABSL_RETURN_IF_ERROR(rewriter.Run(module.get()).status());
    HloDCE dce;
    ABSL_RETURN_IF_ERROR(dce.Run(module.get()).status());
    return module;
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          run_bridge(kSupported, /*expect_annotation=*/true));

  HloInstruction* valid_prefix_fusion = nullptr;
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kFusion &&
        instruction->frontend_attributes().map().contains(
            kXlaValidPrefixExtentOperand)) {
      valid_prefix_fusion = instruction;
      break;
    }
  }
  ASSERT_NE(valid_prefix_fusion, nullptr) << module->ToString();
  ASSERT_EQ(valid_prefix_fusion->operand_count(), 2);
  EXPECT_EQ(valid_prefix_fusion->frontend_attributes().map().at(
                kXlaValidPrefixExtentOperand),
            "1");
  EXPECT_FALSE(valid_prefix_fusion->fused_parameter(1)->users().empty());
  const Shape& extent_shape = valid_prefix_fusion->operand(1)->shape();
  EXPECT_TRUE(extent_shape.IsArray());
  EXPECT_TRUE(extent_shape.dimensions().empty());
  EXPECT_EQ(extent_shape.element_type(), S32);

  constexpr absl::string_view kUnsupported = R"(
HloModule UnsupportedValidPrefixExtent

ENTRY main {
  data = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  dynamic = f32[<=8,4] set-dimension-size(data, n), dimensions={0}
  ROOT result = f32[4,<=8] transpose(dynamic), dimensions={1,0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(module,
                          run_bridge(kUnsupported,
                                     /*expect_annotation=*/false));
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    EXPECT_FALSE(instruction->frontend_attributes().map().contains(
        kXlaValidPrefixExtentOperand));
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
