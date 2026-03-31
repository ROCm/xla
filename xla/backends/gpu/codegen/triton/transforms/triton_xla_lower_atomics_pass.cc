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

// Generic lowering of atomic operations for Triton XLA using
// tt.extern_elementwise. This implementation uses tt.extern_elementwise to call
// custom atomic functions that will be implemented in platform-specific passes
// later in the Triton pipeline.

#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERATOMICSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Helper function to convert MemSemantic enum to string
absl::string_view MemSemanticToString(triton::MemSemantic semantic) {
  switch (semantic) {
    case triton::MemSemantic::RELAXED:
      return "relaxed";
    case triton::MemSemantic::ACQUIRE:
      return "acquire";
    case triton::MemSemantic::RELEASE:
      return "release";
    case triton::MemSemantic::ACQUIRE_RELEASE:
      return "acq_rel";
  }
  LOG(FATAL) << "Unknown MemSemantic value";
}

// Helper function to convert MemSyncScope enum to string
absl::string_view MemSyncScopeToString(triton::MemSyncScope scope) {
  switch (scope) {
    case triton::MemSyncScope::SYSTEM:
      return "system";
    case triton::MemSyncScope::GPU:
      return "gpu";
    case triton::MemSyncScope::CTA:
      return "cta";
  }
  LOG(FATAL) << "Unknown MemSyncScope value";
}

// Helper function to convert Comparator enum to string
absl::string_view ComparatorToString(Comparator comparator) {
  switch (comparator) {
    case Comparator::EQ:
      return "eq";
    case Comparator::LT:
      return "lt";
  }
  LOG(FATAL) << "Unknown Comparator value";
}

mlir::Type GetResultType(mlir::Type ptr_type, PatternRewriter& rewriter) {
  mlir::Type result_type = rewriter.getI32Type();
  auto ranked_tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(ptr_type);
  // Tensor arguments must have tensor result type.
  if (ranked_tensor_type) {
    result_type = mlir::RankedTensorType::get(ranked_tensor_type.getShape(),
                                              rewriter.getI32Type());
  }
  return result_type;
}

// Lower AtomicWriteOp to tt.extern_elementwise.
// This creates extern calls that will be implemented in a separate ROCm pass.
LogicalResult LowerAtomicWriteOp(AtomicWriteOp atomic_write,
                                 PatternRewriter& rewriter) {
  VLOG(2) << "LowerAtomicWriteOp: Starting tt.extern_elementwise lowering";
  mlir::ImplicitLocOpBuilder builder(atomic_write.getLoc(), rewriter);

  mlir::Value ptr = atomic_write.getPtr();
  mlir::Value value = atomic_write.getValue();
  mlir::Value mask = atomic_write.getMask();

  triton::MemSemantic semantic = atomic_write.getMemSyncSemantic();
  triton::MemSyncScope scope = atomic_write.getMemSyncScope();

  // Validate memory semantics
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::RELEASE) {
    return rewriter.notifyMatchFailure(
        atomic_write,
        "AtomicWriteOp only supports RELAXED or RELEASE semantics");
  }

  // Build function name: xla_atomic_write_<semantic>_<scope>
  std::string func_name =
      absl::StrFormat("xla_atomic_write_%s_%s", MemSemanticToString(semantic),
                      MemSyncScopeToString(scope));

  VLOG(3) << "LowerAtomicWriteOp: Creating extern_elementwise call to "
          << func_name;

  // Get result type (handles both tensor and scalar pointers)
  mlir::Type result_type = GetResultType(ptr.getType(), rewriter);

  // Prepare operands: ptr (tensor or scalar), value (always scalar)
  llvm::SmallVector<mlir::Value> operands = {ptr, value};

  // If mask is provided, pass it as third argument
  if (mask) {
    operands.push_back(mask);
  }

  // Create tt.extern_elementwise call
  // The function will perform atomic exchange and return the old value
  // Note: extern_elementwise handles broadcasting scalar value to tensor
  // automatically
  builder.create<triton::ExternElementwiseOp>(
      /*resultType=*/result_type,
      /*srcs=*/operands,
      /*libname=*/"",
      /*libpath=*/"",
      /*symbol=*/func_name,
      /*pure=*/false);

  rewriter.eraseOp(atomic_write);
  return success();
}

// Lower AtomicSpinWaitOp to tt.extern_elementwise.
// This creates extern calls that will be implemented in a separate ROCm pass.
LogicalResult LowerAtomicSpinWaitOp(AtomicSpinWaitOp atomic_wait,
                                    PatternRewriter& rewriter) {
  VLOG(2) << "LowerAtomicSpinWaitOp: Starting tt.extern_elementwise lowering";
  mlir::ImplicitLocOpBuilder builder(atomic_wait.getLoc(), rewriter);

  mlir::Value ptr = atomic_wait.getPtr();
  mlir::Value expected = atomic_wait.getExpected();
  mlir::Value mask = atomic_wait.getMask();

  triton::MemSemantic semantic = atomic_wait.getMemSyncSemantic();
  triton::MemSyncScope scope = atomic_wait.getMemSyncScope();
  Comparator comparator = atomic_wait.getComparator();

  // Validate memory semantics
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::ACQUIRE) {
    return rewriter.notifyMatchFailure(
        atomic_wait,
        "AtomicSpinWaitOp only supports RELAXED or ACQUIRE semantics");
  }

  // Build function name: xla_atomic_spin_wait_<semantic>_<scope>_<comparator>
  std::string func_name = absl::StrFormat(
      "xla_atomic_spin_wait_%s_%s_%s", MemSemanticToString(semantic),
      MemSyncScopeToString(scope), ComparatorToString(comparator));

  VLOG(3) << "LowerAtomicSpinWaitOp: Creating extern_elementwise call to "
          << func_name;

  // Get result type (handles both tensor and scalar pointers)
  mlir::Type result_type = GetResultType(ptr.getType(), rewriter);

  // Prepare operands: ptr (tensor or scalar), expected (always scalar)
  llvm::SmallVector<mlir::Value> operands = {ptr, expected};

  // If mask is provided, pass it as third argument
  if (mask) {
    operands.push_back(mask);
  }

  // Create tt.extern_elementwise call
  // The function will spin-wait until the condition is met
  // Note: extern_elementwise handles broadcasting scalar expected to tensor
  // automatically
  builder.create<triton::ExternElementwiseOp>(
      /*resultType=*/result_type,
      /*srcs=*/operands,
      /*libname=*/"",
      /*libpath=*/"",
      /*symbol=*/func_name,
      /*pure=*/false);

  rewriter.eraseOp(atomic_wait);
  return success();
}

class TritonXLALowerAtomicsPass
    : public impl::TritonXLALowerAtomicsPassBase<TritonXLALowerAtomicsPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerAtomicWriteOp);
    patterns.add(LowerAtomicSpinWaitOp);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerAtomicsPass() {
  return std::make_unique<TritonXLALowerAtomicsPass>();
}

}  // namespace mlir::triton::xla
