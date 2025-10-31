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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_SIMPLIFYMATHPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

class SimplifyMathPass
    : public impl::SimplifyMathPassBase<SimplifyMathPass> {
 public:
  using SimplifyMathPassBase::SimplifyMathPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    // Collect built-in math algebraic simplification patterns.
    mlir::populateMathAlgebraicSimplificationPatterns(patterns);
    mlir::GreedyRewriteConfig config;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateSimplifyMathPass() {
  return std::make_unique<SimplifyMathPass>();
}

}  // namespace emitters
}  // namespace xla

