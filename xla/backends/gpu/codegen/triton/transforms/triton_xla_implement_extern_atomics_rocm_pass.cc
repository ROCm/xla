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

// Helper to parse syncscope from function name
// Function names follow pattern: xla_atomic_*_<semantic>_<scope>[_<comparator>]
std::string ParseSyncScope(const std::string& func_name) {
  // Per AMDGPU memory model (Table 31):
  // - "" (empty) = system scope (cross-device visibility)
  // - "agent" = GPU scope (single device)
  // - "workgroup" = CTA/block scope
  if (func_name.find("_system") != std::string::npos) {
    return "";  // System scope for cross-GPU visibility
  } else if (func_name.find("_gpu") != std::string::npos) {
    return "agent";
  } else if (func_name.find("_cta") != std::string::npos) {
    return "workgroup";
  }
  LOG(FATAL) << "Unable to parse syncscope from function name: " << func_name;
}

// Helper to check if function name ends with given suffix
bool EndsWithComparator(const std::string& func_name,
                        const std::string& comparator) {
  if (func_name.size() < comparator.size()) return false;
  return func_name.compare(func_name.size() - comparator.size(),
                           comparator.size(), comparator) == 0;
}

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
        mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

        std::string syncscope = ParseSyncScope(callee_name);

        if (mask) {
          // Masked atomic: use conditional execution
          // if (mask != 0) { atomic_xchg } else { return 0 }
          auto* current_block = call_op->getBlock();
          auto* atomic_block = current_block->splitBlock(call_op);
          auto* skip_block = atomic_block->splitBlock(call_op);
          auto* exit_block = skip_block->splitBlock(call_op);

          // Current block: check mask and branch
          builder.setInsertionPointToEnd(current_block);
          auto i32_type = builder.getI32Type();
          auto zero = LLVM::ConstantOp::create(builder, loc, i32_type,
                                               builder.getI32IntegerAttr(0));
          auto mask_nonzero = LLVM::ICmpOp::create(
              builder, loc, LLVM::ICmpPredicate::ne, mask, zero);
          LLVM::CondBrOp::create(builder, loc, mask_nonzero->getResult(0),
                                 atomic_block, skip_block);

          // Atomic block: perform atomic exchange
          builder.setInsertionPointToStart(atomic_block);
          auto atomic_xchg = LLVM::AtomicRMWOp::create(
              builder, loc, LLVM::AtomicBinOp::xchg, addr, value,
              LLVM::AtomicOrdering::release, builder.getStringAttr(syncscope));
          LLVM::BrOp::create(builder, loc, atomic_xchg->getResults(),
                             exit_block);

          // Skip block: return zero
          builder.setInsertionPointToStart(skip_block);
          LLVM::BrOp::create(builder, loc, mlir::ValueRange{zero}, exit_block);

          // Exit block: phi node to select result
          exit_block->addArgument(i32_type, loc);
          call_op.replaceAllUsesWith(
              mlir::ValueRange{exit_block->getArgument(0)});
          call_op.erase();
        } else {
          // Unmasked atomic: direct atomic exchange
          auto atomic_xchg = LLVM::AtomicRMWOp::create(
              builder, loc, LLVM::AtomicBinOp::xchg, addr, value,
              LLVM::AtomicOrdering::release, builder.getStringAttr(syncscope));

          call_op.replaceAllUsesWith(atomic_xchg->getResults());
          call_op.erase();
        }

      } else if (absl::StartsWith(callee_name, "xla_atomic_spin_wait_")) {
        // Replace with inline loop
        auto operands = call_op.getOperands();
        auto addr = operands[0];
        auto expected = operands[1];
        mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

        std::string syncscope = ParseSyncScope(callee_name);

        // Check comparator suffix (more robust than substring search)
        bool is_lt = EndsWithComparator(callee_name, "_lt");

        auto i32_type = builder.getI32Type();

        if (mask) {
          // Masked spin wait: check mask first, skip if mask is zero
          // if (mask == 0) goto done; else { spin wait loop }
          auto* current_block = call_op->getBlock();
          auto* loop_block = current_block->splitBlock(call_op);
          auto* exit_block = loop_block->splitBlock(call_op);

          // Current block: check mask
          builder.setInsertionPointToEnd(current_block);
          auto zero = LLVM::ConstantOp::create(builder, loc, i32_type,
                                               builder.getI32IntegerAttr(0));
          auto mask_nonzero = LLVM::ICmpOp::create(
              builder, loc, LLVM::ICmpPredicate::ne, mask, zero);
          LLVM::CondBrOp::create(builder, loc, mask_nonzero->getResult(0),
                                 loop_block, exit_block);

          // Loop block: spin wait
          builder.setInsertionPointToStart(loop_block);
          auto loaded = LLVM::LoadOp::create(
              builder, loc, i32_type, addr, 0, false, false, false, false,
              LLVM::AtomicOrdering::acquire, builder.getStringAttr(syncscope));

          mlir::Value condition;
          if (is_lt) {
            condition =
                LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ult,
                                     loaded->getResult(0), expected)
                    ->getResult(0);
          } else {
            condition =
                LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne,
                                     loaded->getResult(0), expected)
                    ->getResult(0);
          }

          LLVM::CondBrOp::create(builder, loc, condition, loop_block,
                                 exit_block);

          // Exit block
          call_op.replaceAllUsesWith(loaded->getResults());
          call_op.erase();
        } else {
          // Unmasked spin wait: direct loop
          auto* current_block = call_op->getBlock();
          auto* loop_block = current_block->splitBlock(call_op);
          auto* exit_block = loop_block->splitBlock(call_op);

          // Entry: branch to loop
          builder.setInsertionPointToEnd(current_block);
          LLVM::BrOp::create(builder, loc, mlir::ValueRange{}, loop_block);

          // Loop: atomic load + compare + conditional branch
          builder.setInsertionPointToStart(loop_block);
          auto loaded = LLVM::LoadOp::create(
              builder, loc, i32_type, addr, 0, false, false, false, false,
              LLVM::AtomicOrdering::acquire, builder.getStringAttr(syncscope));

          mlir::Value condition;
          if (is_lt) {
            condition =
                LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ult,
                                     loaded->getResult(0), expected)
                    ->getResult(0);
          } else {
            condition =
                LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne,
                                     loaded->getResult(0), expected)
                    ->getResult(0);
          }

          LLVM::CondBrOp::create(builder, loc, condition, loop_block,
                                 exit_block);

          // Exit: use loaded value as result
          call_op.replaceAllUsesWith(loaded->getResults());
          call_op.erase();
        }
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
