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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

class TritonXLALowerAtomicsROCmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.loadDialect<mlir::triton::TritonDialect>();
    context_.loadDialect<mlir::triton::gpu::TritonGPUDialect>();
    context_.loadDialect<mlir::triton::xla::XlaTritonDialect>();
    context_.loadDialect<mlir::arith::ArithDialect>();
    context_.loadDialect<mlir::scf::SCFDialect>();
  }

  mlir::MLIRContext context_;
};

TEST_F(TritonXLALowerAtomicsROCmTest, LowerAtomicWriteOp) {
  constexpr char kMLIR[] = R"(
    module {
      tt.func @test_atomic_write(%ptr: !tt.ptr<i32>, %value: i32) {
        triton_xla.atomic_write sys, release, %ptr, %value : (!tt.ptr<i32>, i32) -> ()
        tt.return
      }
    }
  )";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMLIR, &context_);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  pm.addPass(CreateTritonXLALowerAtomicsROCmPass());

  ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

  // Verify that AtomicWriteOp was lowered to AtomicRMWOp
  bool found_atomic_rmw = false;
  module->walk([&](mlir::triton::AtomicRMWOp op) {
    found_atomic_rmw = true;
    EXPECT_EQ(op.getAtomicRmwOp(), mlir::triton::RMWOp::XCHG);
  });
  EXPECT_TRUE(found_atomic_rmw);

  // Verify that AtomicWriteOp was removed
  bool found_atomic_write = false;
  module->walk([&](AtomicWriteOp op) { found_atomic_write = true; });
  EXPECT_FALSE(found_atomic_write);
}

TEST_F(TritonXLALowerAtomicsROCmTest, LowerAtomicSpinWaitOp) {
  constexpr char kMLIR[] = R"(
    module {
      tt.func @test_atomic_spin_wait(%ptr: !tt.ptr<i32>, %expected: i32) {
        triton_xla.atomic_spin_wait sys, acquire, %ptr, lt, %expected : (!tt.ptr<i32>, i32) -> ()
        tt.return
      }
    }
  )";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMLIR, &context_);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  pm.addPass(CreateTritonXLALowerAtomicsROCmPass());

  ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

  // Verify that AtomicSpinWaitOp was lowered to a while loop with AtomicCASOp
  bool found_while_op = false;
  module->walk([&](mlir::scf::WhileOp op) { found_while_op = true; });
  EXPECT_TRUE(found_while_op);

  bool found_atomic_cas = false;
  module->walk([&](mlir::triton::AtomicCASOp op) { found_atomic_cas = true; });
  EXPECT_TRUE(found_atomic_cas);

  // Verify that AtomicSpinWaitOp was removed
  bool found_atomic_spin_wait = false;
  module->walk([&](AtomicSpinWaitOp op) { found_atomic_spin_wait = true; });
  EXPECT_FALSE(found_atomic_spin_wait);
}

TEST_F(TritonXLALowerAtomicsROCmTest, LowerAtomicWriteOpWithMask) {
  constexpr char kMLIR[] = R"(
    module {
      tt.func @test_atomic_write_masked(%ptr: !tt.ptr<i32>, %value: i32, %mask: i1) {
        triton_xla.atomic_write sys, release, %ptr, %value, %mask : (!tt.ptr<i32>, i32, i1) -> ()
        tt.return
      }
    }
  )";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMLIR, &context_);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context_);
  pm.addPass(CreateTritonXLALowerAtomicsROCmPass());

  ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

  // Verify that masked AtomicWriteOp was lowered correctly
  bool found_atomic_rmw = false;
  module->walk([&](mlir::triton::AtomicRMWOp op) {
    found_atomic_rmw = true;
    EXPECT_TRUE(op.getMask() != nullptr);
  });
  EXPECT_TRUE(found_atomic_rmw);
}

}  // namespace
}  // namespace mlir::triton::xla
