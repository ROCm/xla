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

// GPU-specific implementation of extern_elementwise atomic functions.
// This pass runs in the Triton GPU pipeline and inlines the implementations
// of custom atomic functions by replacing llvm.call operations with LLVM
// intrinsics. Supports both CUDA and ROCM backends.

#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/extern_function_helper.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAIMPLEMENTEXTERNELEMENTWISEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Pattern to rewrite llvm.call operations to XLA extern functions
class RewriteExternCallPattern : public OpRewritePattern<LLVM::CallOp> {
 public:
  explicit RewriteExternCallPattern(MLIRContext* context, TargetBackend target)
      : OpRewritePattern<LLVM::CallOp>(context), target_(target) {}

  LogicalResult matchAndRewrite(LLVM::CallOp call_op,
                                PatternRewriter& rewriter) const override {
    // Check if this is a call to one of our extern functions
    auto callee = call_op.getCallee();
    if (!callee) {
      return failure();
    }

    llvm::StringRef callee_name = *callee;
    if (!callee_name.starts_with("xla_atomic_write_") &&
        !callee_name.starts_with("xla_atomic_spin_wait_") &&
        !callee_name.starts_with("xla_get_thread_id")) {
      return failure();
    }

    // Parse the function name to get the instruction
    auto parsed = ParseExternFunctionName(callee_name.str());
    if (!parsed.ok()) {
      return rewriter.notifyMatchFailure(
          call_op, "Failed to parse extern function name: " +
                       callee_name.str() + " - " + parsed.status().ToString());
    }

    // Validate memory semantic
    auto validation = ValidateMemorySemantic(*parsed);
    if (!validation.ok()) {
      return rewriter.notifyMatchFailure(
          call_op, "Invalid memory semantic for function: " +
                       callee_name.str() + " - " + validation.ToString());
    }

    // Create LLVM operations for the instruction
    LLVMOpCreationParams params{.builder = rewriter,
                                .loc = call_op.getLoc(),
                                .target = target_,
                                .operands = call_op.getOperands()};

    mlir::Value result = CreateLLVMOpsForInstruction(*parsed, params);

    // Replace the call with the generated operations
    rewriter.replaceOp(call_op, result);
    return success();
  }

 private:
  TargetBackend target_;
};

// Pattern to erase unused extern function declarations
class EraseUnusedExternFuncPattern : public OpRewritePattern<LLVM::LLVMFuncOp> {
 public:
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp func_op,
                                PatternRewriter& rewriter) const override {
    // Only match external functions
    if (!func_op.isExternal()) {
      return failure();
    }

    llvm::StringRef name = func_op.getName();
    if (!name.starts_with("xla_atomic_write_") &&
        !name.starts_with("xla_atomic_spin_wait_") &&
        !name.starts_with("xla_get_thread_id")) {
      return failure();
    }

    // Check if the function has any uses
    if (!func_op.use_empty()) {
      return failure();
    }

    // Erase the unused function declaration
    rewriter.eraseOp(func_op);
    return success();
  }
};

// MLIR pass that inlines extern function calls with LLVM intrinsics
class TritonXLAImplementExternElementWisePass
    : public impl::TritonXLAImplementExternElementWisePassBase<
          TritonXLAImplementExternElementWisePass> {
 public:
  using impl::TritonXLAImplementExternElementWisePassBase<
      TritonXLAImplementExternElementWisePass>::
      TritonXLAImplementExternElementWisePassBase;

 private:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();

    // Get the target string from the option
    std::string target_str = target_.getValue();

    // Parse target string to TargetBackend enum
    TargetBackend target;
    if (target_str == "cuda") {
      target = TargetBackend::CUDA;
    } else if (target_str == "rocm") {
      target = TargetBackend::ROCM;
    } else {
      module.emitError("Invalid target backend: ")
          << target_str << ". Expected 'cuda' or 'rocm'.";
      return signalPassFailure();
    }

    // Apply rewrite patterns
    RewritePatternSet patterns(context);
    patterns.add<RewriteExternCallPattern>(context, target);
    patterns.add<EraseUnusedExternFuncPattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::triton::xla

std::unique_ptr<mlir::Pass>
mlir::triton::xla::CreateTritonXLAImplementExternElementWisePass(
    TargetBackend target) {
  TritonXLAImplementExternElementWisePassOptions options;
  options.target_ = target == TargetBackend::CUDA ? "cuda" : "rocm";
  return std::make_unique<TritonXLAImplementExternElementWisePass>(options);
}
