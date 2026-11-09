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

#include <memory>

#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

class ValidPrefixMarkerTest : public HloHardwareIndependentTestBase {
 public:
  ValidPrefixMarkerTest() { RegisterSymbolicExprStorage(&mlir_context_); }

 protected:
  se::DeviceDescription device_info_ = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  GpuAliasInfo alias_info_{device_info_};
  mlir::MLIRContext mlir_context_;
};

TEST_F(ValidPrefixMarkerTest, ExtentRemainsAFusionOperand) {
  constexpr char kHlo[] = R"(
HloModule ValidPrefixMarker

ENTRY main {
  data = f32[8,4] parameter(0)
  n = s32[] parameter(1)
  dynamic = f32[<=8,4] set-dimension-size(data, n), dimensions={0}
  negated = f32[<=8,4] negate(dynamic)
  ROOT result = f32[<=8,4] exponential(negated)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  DynamicPadderOptions options;
  options.preserve_valid_prefix = true;
  DynamicPadder dynamic_padder(options);
  ASSERT_OK(dynamic_padder.Run(module.get()).status());

  auto shape_size = [](const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
  };
  ASSERT_OK(FusionPipeline(module->config().debug_options(), shape_size,
                           &alias_info_, /*thread_pool=*/nullptr, device_info_,
                           &mlir_context_)
                .Run(module.get())
                .status());

  HloInstruction* marked_fusion = nullptr;
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() != HloOpcode::kFusion) {
      continue;
    }
    for (HloInstruction* fused :
         instruction->fused_instructions_computation()->instructions()) {
      if (fused->IsCustomCall("xla.valid_prefix")) {
        marked_fusion = instruction;
        break;
      }
    }
  }

  ASSERT_NE(marked_fusion, nullptr) << module->ToString();
  ASSERT_EQ(marked_fusion->operand_count(), 2);
  EXPECT_EQ(marked_fusion->operand(1),
            module->entry_computation()->parameter_instruction(1));
  const HloInstruction* extent_parameter = marked_fusion->fused_parameter(1);
  ASSERT_EQ(extent_parameter->user_count(), 1);
  EXPECT_TRUE(
      extent_parameter->users().front()->IsCustomCall("xla.valid_prefix"));
}

}  // namespace
}  // namespace xla::gpu
