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

// ROCm-specific implementation of extern_elementwise atomic functions.
// This pass runs in the Triton ROCm pipeline and inlines the implementations
// of custom atomic functions by replacing llvm.call operations with actual
// LLVM dialect operations (intrinsics, atomics, ...).

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAIMPLEMENTEXTERNATOMICSROCMPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// MLIR pass that inlines extern function calls with actual implementations
class TritonXLAImplementExternAtomicsROCmPass
    : public impl::TritonXLAImplementExternAtomicsROCmPassBase<
          TritonXLAImplementExternAtomicsROCmPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());

    // Find all llvm.call operations to our extern functions
    llvm::SmallVector<LLVM::CallOp> calls_to_replace;
    module.walk([&](LLVM::CallOp call_op) {
      if (auto callee = call_op.getCallee()) {
        llvm::StringRef name = *callee;
        if (name.starts_with("xla_atomic_write_") ||
            name.starts_with("xla_atomic_spin_wait_") ||
            name.starts_with("xla_get_thread_id")) {
          calls_to_replace.push_back(call_op);
        }
      }
    });

    // Replace each call inline
    for (auto call_op : calls_to_replace) {
      std::string callee_name = call_op.getCallee()->str();
      builder.setInsertionPoint(call_op);
      auto loc = call_op.getLoc();

      if (absl::StartsWith(callee_name, "xla_get_thread_id")) {
        // Replace with direct intrinsic call
        auto i32_type = builder.getI32Type();
        auto intrinsic_name =
            builder.getStringAttr("llvm.amdgcn.workitem.id.x");
        auto intrinsic_call = LLVM::CallIntrinsicOp::create(
            builder, loc, i32_type, intrinsic_name, mlir::ValueRange{});
        call_op.replaceAllUsesWith(intrinsic_call->getResults());
        call_op.erase();

      } else if (absl::StartsWith(callee_name, "xla_atomic_write_")) {
        // Replace with atomicrmw xchg
        auto operands = call_op.getOperands();
        auto addr = operands[0];
        auto value = operands[1];

        // Parse scope from function name
        std::string syncscope;
        if (callee_name.find("_system") != std::string::npos) {
          syncscope = "one-as";
        } else if (callee_name.find("_gpu") != std::string::npos) {
          syncscope = "agent";
        } else {
          syncscope = "workgroup";
        }

        auto atomic_xchg = LLVM::AtomicRMWOp::create(
            builder, loc, LLVM::AtomicBinOp::xchg, addr, value,
            LLVM::AtomicOrdering::release, builder.getStringAttr(syncscope));

        call_op.replaceAllUsesWith(atomic_xchg->getResults());
        call_op.erase();

      } else if (absl::StartsWith(callee_name, "xla_atomic_spin_wait_")) {
        // Replace with inline loop
        auto operands = call_op.getOperands();
        auto addr = operands[0];
        auto expected = operands[1];

        // Parse scope and comparator
        std::string syncscope;
        if (callee_name.find("_system") != std::string::npos) {
          syncscope = "one-as";
        } else if (callee_name.find("_gpu") != std::string::npos) {
          syncscope = "agent";
        } else {
          syncscope = "workgroup";
        }

        bool is_lt = callee_name.find("_lt") != std::string::npos;

        // Create loop blocks
        auto* current_block = call_op->getBlock();
        auto* loop_block = current_block->splitBlock(call_op);
        auto* exit_block = loop_block->splitBlock(call_op);

        // Entry: branch to loop
        builder.setInsertionPointToEnd(current_block);
        LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loop_block);

        // Loop: atomic load + compare + conditional branch
        builder.setInsertionPointToStart(loop_block);
        auto i32_type = builder.getI32Type();
        auto loaded = LLVM::LoadOp::create(
            builder, loc, i32_type, addr, 0, false, false, false, false,
            LLVM::AtomicOrdering::acquire, builder.getStringAttr(syncscope));

        mlir::Value condition;
        if (is_lt) {
          condition =
              LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::slt,
                                   loaded->getResult(0), expected)
                  ->getResult(0);
        } else {
          condition =
              LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne,
                                   loaded->getResult(0), expected)
                  ->getResult(0);
        }

        LLVM::CondBrOp::create(builder, loc, condition, loop_block, exit_block);

        // Exit: use loaded value as result
        call_op.replaceAllUsesWith(loaded->getResults());
        call_op.erase();
      }
    }

    // Clean up unused extern function declarations
    llvm::SmallVector<LLVM::LLVMFuncOp> to_erase;
    module.walk([&](LLVM::LLVMFuncOp func) {
      if (func.isExternal()) {
        llvm::StringRef name = func.getName();
        if (name.starts_with("xla_atomic_write_") ||
            name.starts_with("xla_atomic_spin_wait_") ||
            name.starts_with("xla_get_thread_id")) {
          to_erase.push_back(func);
        }
      }
    });

    for (auto func : to_erase) {
      func.erase();
    }

    VLOG(2) << "TritonXLAImplementExternAtomicsROCmPass: Replaced "
            << calls_to_replace.size() << " calls, removed " << to_erase.size()
            << " declarations";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTritonXLAImplementExternAtomicsROCmPass() {
  return std::make_unique<TritonXLAImplementExternAtomicsROCmPass>();
}

}  // namespace mlir::triton::xla
