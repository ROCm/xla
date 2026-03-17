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

// ROCm-specific lowering of GetTidOp using inline GCN assembly.
// This implementation uses inline assembly to retrieve the thread ID
// directly from hardware registers.

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "mlir/IR/Builders.h"
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

#define GEN_PASS_DEF_TRITONXLALOWERGETTIDROCMASMPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

LogicalResult LowerGetTidOpROCmAsm(GetTidOp get_flat_tid,
                                   PatternRewriter& rewriter) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(get_flat_tid);
  const Location loc = get_flat_tid.getLoc();

  const mlir::Type i32_type = rewriter.getI32Type();
  
  // GCN inline assembly to get thread ID
  // In GCN/RDNA, the thread ID within a workgroup is available via v0
  // which contains the flattened thread ID (similar to CUDA's %tid.x)
  // We use the special register to read the thread ID
  const absl::string_view get_tid_asm = R"(
    v_mov_b32 $0, v0
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
  return success();
}

class TritonXLALowerGetTidROCmAsmPass
    : public impl::TritonXLALowerGetTidROCmAsmPassBase<
          TritonXLALowerGetTidROCmAsmPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerGetTidOpROCmAsm);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerGetTidROCmAsmPass() {
  return std::make_unique<TritonXLALowerGetTidROCmAsmPass>();
}

}  // namespace mlir::triton::xla
