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
#include <utility>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
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

#define GEN_PASS_DEF_TRITONXLALOWERGETTIDPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Helper to detect if we're compiling for AMD/ROCm target
bool IsAMDGPUTarget(mlir::Operation* op) {
  //TODO: implement this function
  return true;
}

LogicalResult LowerGetTidOp(GetTidOp get_flat_tid, PatternRewriter& rewriter) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(get_flat_tid);
  const Location loc = get_flat_tid.getLoc();
  const mlir::Type i32_type = rewriter.getI32Type();

  // Check if we're compiling for AMD GPUs
  bool is_amd = IsAMDGPUTarget(get_flat_tid);

  if (is_amd) {
    // AMD GCN: Read thread ID from hardware register
    constexpr absl::string_view get_tid_asm = R"(
      v_mbcnt_lo_u32_b32 $0, -1, 0
      v_mbcnt_hi_u32_b32 $0, -1, $0
    )";
    auto tid_op = mlir::triton::ElementwiseInlineAsmOp::create(
        rewriter, loc,
        /*result_types=*/i32_type,
        /*asm_string=*/rewriter.getStringAttr(get_tid_asm),
        /*constraints=*/rewriter.getStringAttr("=v"),
        /*pure=*/rewriter.getBoolAttr(true),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args*/ mlir::ValueRange{});
    rewriter.replaceOp(get_flat_tid, tid_op->getResults());
  } else {
    // NVIDIA PTX: Read thread ID from special register
    constexpr absl::string_view get_tid_asm = R"(
      mov.u32 $0, %tid.x;
    )";
    auto tid_op = mlir::triton::ElementwiseInlineAsmOp::create(
        rewriter, loc,
        /*result_types=*/i32_type,
        /*asm_string=*/rewriter.getStringAttr(get_tid_asm),
        /*constraints=*/rewriter.getStringAttr("=r"),
        /*pure=*/rewriter.getBoolAttr(true),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args*/ mlir::ValueRange{});
    rewriter.replaceOp(get_flat_tid, tid_op->getResults());
  }
  return success();
}

class TritonXLALowerGetTidPass
    : public impl::TritonXLALowerGetTidPassBase<TritonXLALowerGetTidPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerGetTidOp);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerGetTidPass() {
  return std::make_unique<TritonXLALowerGetTidPass>();
}

}  // namespace mlir::triton::xla
