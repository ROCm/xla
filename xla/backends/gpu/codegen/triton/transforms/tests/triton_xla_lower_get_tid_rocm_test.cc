/* Copyright 2025 The OpenXLA Authors.

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
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "tsl/platform/test.h"

namespace mlir::triton::xla {
namespace {

class TritonXLALowerGetTidROCmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.loadDialect<mlir::triton::TritonDialect>();
    context_.loadDialect<mlir::triton::gpu::TritonGPUDialect>();
    context_.loadDialect<mlir::triton::xla::XlaTritonDialect>();
  }

  mlir::MLIRContext context_;
};

TEST_F(TritonXLALowerGetTidROCmTest, LowerGetTidOp) {
  constexpr char kMLIR[] = R"(
    module {
      tt.func @test_get_tid() -> i32 {
        %tid = triton_xla.get_tid : () -> i32
        tt.return %tid : i32
      }
    }
  )";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMLIR, &context_);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  pm.addPass(CreateTritonXLALowerGetTidROCmPass());

  ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

  // Verify that GetTidOp was lowered to GetThreadIdOp
  bool found_get_thread_id = false;
  module->walk([&](mlir::triton::gpu::GetThreadIdOp op) {
    found_get_thread_id = true;
  });
  EXPECT_TRUE(found_get_thread_id);

  // Verify that GetTidOp was removed
  bool found_get_tid = false;
  module->walk([&](GetTidOp op) { found_get_tid = true; });
  EXPECT_FALSE(found_get_tid);
}

}  // namespace
}  // namespace mlir::triton::xla
