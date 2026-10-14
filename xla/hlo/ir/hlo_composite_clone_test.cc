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

#include <gtest/gtest.h>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(HloCompositeCloneTest, PreservesCompositeMetadata) {
  const Shape shape = ShapeUtil::MakeShape(F32, {2});
  HloComputation::Builder builder("decomposition");
  builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "value"));
  auto decomposition = builder.Build();
  auto operand = HloInstruction::CreateParameter(0, shape, "operand");
  auto call =
      HloInstruction::CreateCall(shape, {operand.get()}, decomposition.get());
  EXPECT_FALSE(call->CloneWithNewOperands(shape, {operand.get()})
                   ->is_composite());

  call->set_is_composite(true);
  FrontendAttributes attributes;
  (*attributes.mutable_map())["composite.name"] = "test.identity";
  (*attributes.mutable_map())["composite.attributes"] = "{}";
  (*attributes.mutable_map())["composite.version"] = "0";
  call->set_frontend_attributes(attributes);

  auto clone = call->CloneWithNewOperands(shape, {operand.get()});
  EXPECT_TRUE(clone->is_composite());
  EXPECT_EQ(clone->to_apply(), decomposition.get());
  EXPECT_EQ(clone->frontend_attributes().map().at("composite.name"),
            "test.identity");
  EXPECT_EQ(clone->frontend_attributes().map().at("composite.attributes"), "{}");
  EXPECT_EQ(clone->frontend_attributes().map().at("composite.version"), "0");
}

}  // namespace
}  // namespace xla
