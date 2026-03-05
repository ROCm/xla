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

// ROCm-specific lowering of atomic operations for Triton XLA.
// This implementation uses pure Triton atomic operations instead of
// platform-specific inline assembly, making it compatible with ROCm/HIP.
//
// Based on the IRIS one-shot AllReduce implementation which demonstrates
// that pure Triton atomics work correctly on AMD GPUs.

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

#define GEN_PASS_DEF_TRITONXLALOWERATOMICSROCMPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Lower AtomicWriteOp to pure Triton atomic_xchg operation.
// This is compatible with ROCm and doesn't require inline assembly.
LogicalResult LowerAtomicWriteOpROCm(AtomicWriteOp atomic_write,
                                     PatternRewriter& rewriter) {
  mlir::ImplicitLocOpBuilder builder(atomic_write.getLoc(), rewriter);

  mlir::Value ptr = atomic_write.getPtr();
  mlir::Value value = atomic_write.getValue();
  mlir::Value mask = atomic_write.getMask();
  
  triton::MemSemantic semantic = atomic_write.getMemSyncSemantic();
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::RELEASE) {
    return rewriter.notifyMatchFailure(
        atomic_write, absl::StrFormat("Unsupported memory semantic: %s",
                                      stringifyMemSemantic(semantic)));
  }

  // Use Triton's atomic_xchg which works on both CUDA and ROCm
  // atomic_xchg(ptr, value, sem=semantic, scope=scope) -> old_value
  // We don't care about the old value for writes
  triton::AtomicRMWOp::create(
      builder,
      /*result_type=*/value.getType(),
      triton::RMWOp::XCHG,
      /*ptr=*/ptr,
      /*val=*/value,
      /*mask=*/mask,
      /*sem=*/semantic,
      /*scope=*/atomic_write.getMemSyncScope());

  rewriter.eraseOp(atomic_write);
  return success();
}

// Lower AtomicSpinWaitOp to a loop with Triton atomic_cas.
// This uses pure Triton operations compatible with ROCm.
LogicalResult LowerAtomicSpinWaitOpROCm(AtomicSpinWaitOp atomic_wait,
                                        PatternRewriter& rewriter) {
  mlir::ImplicitLocOpBuilder builder(atomic_wait.getLoc(), rewriter);

  mlir::Value ptr = atomic_wait.getPtr();
  mlir::Value expected = atomic_wait.getExpected();
  mlir::Value mask = atomic_wait.getMask();
  
  triton::MemSemantic semantic = atomic_wait.getMemSyncSemantic();
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::ACQUIRE) {
    return rewriter.notifyMatchFailure(
        atomic_wait, absl::StrFormat("Unsupported memory semantic: %s",
                                     stringifyMemSemantic(semantic)));
  }

  Comparator comparator = atomic_wait.getComparator();
  
  // Create a while loop that spins until the condition is met
  // while (atomic_load(ptr) <comparator> expected) { /* spin */ }
  
  auto ptr_type = mlir::cast<mlir::RankedTensorType>(ptr.getType());
  bool is_tensor = ptr_type != nullptr;
  
  if (is_tensor) {
    // For tensor operations, we need to handle each element
    // This is more complex and would require element-wise loop
    // For now, we'll use a simpler approach with atomic_cas
    
    // Create a loop that continuously checks the value
    mlir::scf::WhileOp::create(
        builder,
        /*resultTypes=*/mlir::TypeRange{},
        /*operands=*/mlir::ValueRange{},
        /*beforeBuilder=*/
        [&](mlir::OpBuilder& op_builder, mlir::Location location,
            mlir::ValueRange args) {
          mlir::ImplicitLocOpBuilder loop_builder(location, op_builder);
          
          // Load current value using atomic_cas with same value
          // This ensures proper memory ordering
          mlir::Type result_type = expected.getType();
          mlir::Value loaded = triton::AtomicCASOp::create(
              loop_builder,
              /*result=*/result_type,
              /*ptr=*/ptr,
              /*cmp=*/expected,
              /*val=*/expected,
              /*sem=*/semantic,
              /*scope=*/atomic_wait.getMemSyncScope());
          
          // Check condition based on comparator
          mlir::Value condition;
          if (comparator == Comparator::LT) {
            condition = mlir::arith::CmpIOp::create(
                loop_builder, mlir::arith::CmpIPredicate::ult,
                loaded, expected);
          } else {  // Comparator::EQ
            condition = mlir::arith::CmpIOp::create(
                loop_builder, mlir::arith::CmpIPredicate::eq,
                loaded, expected);
          }
          
          // Apply mask if present
          if (mask) {
            condition = mlir::arith::AndIOp::create(
                loop_builder, condition, mask);
          }
          
          mlir::scf::ConditionOp::create(loop_builder, condition,
                                         mlir::ValueRange{});
        },
        /*afterBuilder=*/
        [&](mlir::OpBuilder& op_builder, mlir::Location location,
            mlir::ValueRange args) {
          mlir::ImplicitLocOpBuilder loop_builder(location, op_builder);
          // Just yield to continue the loop
          mlir::scf::YieldOp::create(loop_builder, mlir::ValueRange{});
        });
  } else {
    // Scalar version - simpler loop
    mlir::scf::WhileOp::create(
        builder,
        /*resultTypes=*/mlir::TypeRange{},
        /*operands=*/mlir::ValueRange{},
        /*beforeBuilder=*/
        [&](mlir::OpBuilder& op_builder, mlir::Location location,
            mlir::ValueRange args) {
          mlir::ImplicitLocOpBuilder loop_builder(location, op_builder);
          
          // Atomic load via CAS
          mlir::Type result_type = expected.getType();
          mlir::Value loaded = triton::AtomicCASOp::create(
              loop_builder,
              /*result=*/result_type,
              /*ptr=*/ptr,
              /*cmp=*/expected,
              /*val=*/expected,
              /*sem=*/semantic,
              /*scope=*/atomic_wait.getMemSyncScope());
          
          // Check condition
          mlir::Value condition;
          if (comparator == Comparator::LT) {
            condition = mlir::arith::CmpIOp::create(
                loop_builder, mlir::arith::CmpIPredicate::ult,
                loaded, expected);
          } else {
            condition = mlir::arith::CmpIOp::create(
                loop_builder, mlir::arith::CmpIPredicate::eq,
                loaded, expected);
          }
          
          mlir::scf::ConditionOp::create(loop_builder, condition,
                                         mlir::ValueRange{});
        },
        /*afterBuilder=*/
        [&](mlir::OpBuilder& op_builder, mlir::Location location,
            mlir::ValueRange args) {
          mlir::ImplicitLocOpBuilder loop_builder(location, op_builder);
          mlir::scf::YieldOp::create(loop_builder, mlir::ValueRange{});
        });
  }

  rewriter.eraseOp(atomic_wait);
  return success();
}

class TritonXLALowerAtomicsROCmPass
    : public impl::TritonXLALowerAtomicsROCmPassBase<
          TritonXLALowerAtomicsROCmPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerAtomicWriteOpROCm);
    patterns.add(LowerAtomicSpinWaitOpROCm);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerAtomicsROCmPass() {
  return std::make_unique<TritonXLALowerAtomicsROCmPass>();
}

}  // namespace mlir::triton::xla
