/* Copyright 2024 The OpenXLA Authors.

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

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_CONVERTFLOATPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;
namespace ml = ::mlir::LLVM;
namespace mscf = ::mlir::scf;
namespace mv = ::mlir::vector;

using mlir::Value;


struct RewriteTruncFPattern : public mlir::OpRewritePattern<ma::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::TruncFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());
    if (!dst_ty.isFloat8E4M3FNUZ() && !dst_ty.isFloat8E5M2FNUZ()) {
      return rewriter.notifyMatchFailure(op, "unsupported float conversion");
    }

    // auto loop = mlir::dyn_cast_or_null<mscf::ForOp>(op->getParentOp());

    // if (loop) {
    //   if (op->hasOneUse()) {
    //     auto insert =
    //         mlir::dyn_cast_or_null<mv::InsertOp>(*op->user_begin());
    //     if (insert) {
    //       if (insert->hasOneUse() &&
    //           mlir::isa<mscf::YieldOp>(*insert->user_begin())) {
    //         auto bbarg =
    //             mlir::dyn_cast_or_null<mlir::BlockArgument>(insert.getDest());
    //         if (bbarg) {
    //           if (bbarg.hasOneUse() &&
    //               bbarg.getOwner()->getParentOp() == loop) {
    //             if (mlir::getConstantIntValue(loop.getStep()) == 1 &&
    //                 mlir::getConstantIntValue(loop.getLowerBound()) == 0 &&
    //                 mlir::getConstantIntValue(loop.getUpperBound()) == 4) {
    //               auto pos = insert.getDynamicPosition();
    //               if ((pos.size() == 1) &&
    //                   (pos[0] == loop.getInductionVar())) {
    //                 rewriter.setInsertionPoint(loop);
    //                 auto init = rewriter.create<ml::PoisonOp>(loop.getLoc(),
    //                                                           src.getType());
    //                 rewriter.setInsertionPointAfter(insert);
    //                 mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    //                 auto cnd = b.create<ma::CmpIOp>(
    //                     ma::CmpIPredicate::eq, loop.getInductionVar(),
    //                     b.create<ma::ConstantOp>(
    //                         b.getZeroAttr(loop.getInductionVar().getType())));
    //                 auto if_op = b.create<mlir::scf::IfOp>(
    //                     cnd,
    //                     [&](mlir::OpBuilder& b, mlir::Location loc) {
    //                       b.create<mscf::YieldOp>(loc, mlir::ValueRange{insert.getDest(),
    //                                                    op.getOperand()});
    //                     },
    //                     [&](mlir::OpBuilder& b, mlir::Location loc) {
    //                       b.create<mscf::YieldOp>(loc, mlir::ValueRange{insert.getDest(),
    //                                                    op.getOperand()});
    //                     });
    //                 insert.getResult().replaceAllUsesWith(if_op.getResult(0));
    //                 rewriter.eraseOp(insert);
    //                 rewriter.eraseOp(op);
    //                 auto new_for = *loop.replaceWithAdditionalYields(
    //                     rewriter, mlir::ValueRange{init},
    //                     /*replaceInitOperandUsesInLoop=*/true,
    //                     [&](mlir::OpBuilder& yield_b, mlir::Location yield_loc,
    //                         llvm::ArrayRef<mlir::BlockArgument> bbarg) {
    //                       return llvm::SmallVector<Value>{if_op.getResult(1)};
    //                     });
    //                 return mlir::success();
    //               }
    //             }
    //           }
    //         }
    //       }
    //     }
    //   }
    // }

    auto match = MatchVectorizablePair(op,src, dst_ty);

    if (succeeded(match)) {
      auto [first, second, vector_input, vector_result, pos] = *match;
      rewriter.setInsertionPointAfter(vector_result.getDefiningOp());
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      rewriter.replaceOp(vector_result.getDefiningOp(),
                         EmitVectorizedTruncToF8Intrinsic(first, second, vector_input, pos, dst_ty, b));
      return mlir::success();
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOp(op, EmitTruncToF8Intrinsic(src, dst_ty, b));
    return mlir::success();
  }

  mlir::FailureOr<
      std::tuple<mlir::Value, mlir::Value, mlir::Value, mlir::Value, size_t>>
  MatchVectorizablePair(ma::TruncFOp op, Value value, mlir::FloatType to_ty) const {
    Value vector;
    llvm::APInt pos;
    if (!op->hasOneUse() ||
        !mlir::matchPattern(
            *op->user_begin(),
            mlir::m_Op<mv::InsertOp>(mlir::matchers::m_Val(op->getResult(0)),
                                     mlir::matchers::m_Any(&vector),
                                     mlir::m_ConstantInt(&pos)))) {
      std::cerr << "PERA" << std::endl;
      return mlir::failure();
    }

    if (vector.getType() != ml::getFixedVectorType(to_ty, 4)) {
      std::cerr << "PERA2" << std::endl;
      return mlir::failure();
    }

    auto insert = mlir::cast<mv::InsertOp>(*op->user_begin());

    if (pos.getZExtValue() % 2 == 0) {
      llvm::APInt pos2;
      Value input2;
      Value cvt;
      if (!insert->hasOneUse() ||
          !mlir::matchPattern(*insert->user_begin(),
                              mlir::m_Op<mv::InsertOp>(
                                  mlir::matchers::m_Any(&cvt),
                                  mlir::matchers::m_Val(insert.getResult()),
                                  mlir::m_ConstantInt(&pos2))) ||
          !mlir::matchPattern(
              cvt.getDefiningOp(),
              mlir::m_Op<ma::TruncFOp>(mlir::matchers::m_Any(&input2))) ||
          input2.getType() != value.getType() ||
          pos2.getZExtValue() != (pos.getZExtValue() + 1)) {
          std::cerr << "PERA3" << std::endl;
          cvt.dump();
        return mlir::failure();
      }

      return std::make_tuple(
          value, input2, vector,
          static_cast<mlir::Value>(insert->user_begin()->getResult(0)),
          pos.getZExtValue());
    } else {
      llvm::APInt pos1;
      Value vector1;
      Value input1;
      Value cvt;
      if (!mlir::matchPattern(
              insert->getOperand(1),
              mlir::m_Op<mv::InsertOp>(mlir::matchers::m_Any(&cvt),
                                       mlir::matchers::m_Any(&vector1),
                                       mlir::m_ConstantInt(&pos1))) ||
          !mlir::matchPattern(
              cvt.getDefiningOp(),
              mlir::m_Op<ma::TruncFOp>(mlir::matchers::m_Any(&input1))) ||
          input1.getType() != value.getType()  ||
          pos1.getZExtValue() != (pos.getZExtValue() - 1)) {
        return mlir::failure();
      }

      return std::make_tuple(input1, value, vector1,
                             static_cast<mlir::Value>(insert.getResult()),
                             pos1.getZExtValue());
    }
  }

  Value EmitVectorizedTruncToF8Intrinsic(Value valueA, Value valueB,
                                         Value vector, size_t pos,
                                         mlir::FloatType to_ty,
                                         mlir::ImplicitLocOpBuilder& b) const {
    assert(to_ty.isFloat8E4M3FNUZ() || to_ty.isFloat8E5M2FNUZ());

    mlir::FloatType f32_ty = b.getF32Type();
    mlir::IntegerType i32_ty = b.getI32Type();
    if (valueA.getType().getIntOrFloatBitWidth() < f32_ty.getWidth()) {
      valueA = b.create<ma::ExtFOp>(f32_ty, valueA);
    } else if (valueA.getType() != f32_ty) {
      valueA = b.create<ma::TruncFOp>(f32_ty, valueA);
    }

    if (valueB.getType().getIntOrFloatBitWidth() < f32_ty.getWidth()) {
      valueB = b.create<ma::ExtFOp>(f32_ty, valueB);
    } else if (valueB.getType() != f32_ty) {
      valueB = b.create<ma::TruncFOp>(f32_ty, valueB);
    }

    auto cvtIntr = to_ty.isFloat8E4M3FNUZ() ? "llvm.amdgcn.cvt.pk.fp8.f32"
                                            : "llvm.amdgcn.cvt.pk.bf8.f32";

    auto i8_4_ty = ml::getFixedVectorType(b.getI8Type(), 4);

    ml::CallIntrinsicOp cvtOp = b.create<ml::CallIntrinsicOp>(
        i32_ty, b.getStringAttr(cvtIntr),
        mlir::ValueRange{valueA, valueB,
                         b.create<ml::BitcastOp>(
                             i32_ty, mlir::ValueRange{b.create<mlir::UnrealizedConversionCastOp>(
                                         i8_4_ty, mlir::ValueRange{vector}).getResult(0)}),
                         b.create<ml::ConstantOp>(b.getI1Type(), pos != 0)});
    return b.create<mlir::UnrealizedConversionCastOp>(
        vector.getType(),
        mlir::ValueRange{b.create<ml::BitcastOp>(i8_4_ty, cvtOp.getResults())}).getResult(0);
  }

  Value EmitTruncToF8Intrinsic(Value value, mlir::FloatType to_ty,
                               mlir::ImplicitLocOpBuilder& b) const {
    assert(to_ty.isFloat8E4M3FNUZ() || to_ty.isFloat8E5M2FNUZ());

    mlir::FloatType f32_ty = b.getF32Type();
    mlir::IntegerType i32_ty = b.getI32Type();
    if (value.getType().getIntOrFloatBitWidth() < f32_ty.getWidth()) {
      value = b.create<ma::ExtFOp>(f32_ty, value);
    } else if (value.getType() != f32_ty) {
      value = b.create<ma::TruncFOp>(f32_ty, value);
    }

    auto cvtIntr = to_ty.isFloat8E4M3FNUZ() ? "llvm.amdgcn.cvt.pk.fp8.f32"
                                          : "llvm.amdgcn.cvt.pk.bf8.f32";

    ml::CallIntrinsicOp cvtOp = b.create<ml::CallIntrinsicOp>(
        i32_ty, b.getStringAttr(cvtIntr),
        mlir::ValueRange{value, b.create<ml::UndefOp>(f32_ty),
                         b.create<ml::UndefOp>(i32_ty),
                         b.create<ml::ConstantOp>(b.getI1Type(), 0)});
    Value res = b.create<ml::TruncOp>(b.getI8Type(), cvtOp.getResults());
    return b.create<ma::BitcastOp>(to_ty, res);
  }
};
  

struct RewriteExtFPattern : public mlir::OpRewritePattern<ma::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());
    if (!src.getType().isFloat8E4M3FNUZ() && !src.getType().isFloat8E5M2FNUZ()) {
      return rewriter.notifyMatchFailure(op, "unsupported float conversion");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOp(op, EmitExtFromF8Intrinsic(src, dst_ty, b));
    return mlir::success();
  }

  Value EmitExtFromF8Intrinsic(Value value, mlir::FloatType to_ty,
                               mlir::ImplicitLocOpBuilder& b) const {
    assert(value.getType().isFloat8E4M3FNUZ() || value.getType().isFloat8E5M2FNUZ());

    mlir::FloatType f32_ty = b.getF32Type();
    mlir::IntegerType i32_ty = b.getI32Type();
    mlir::IntegerType i8_ty = b.getI8Type();
    Value zero_cst = b.create<ml::ConstantOp>(i32_ty, 0);
    // Emulate anyext
    Value input = b.create<ml::BitcastOp>(
        i32_ty, b.create<ml::InsertElementOp>(
                    b.create<ml::UndefOp>(ml::getFixedVectorType(i8_ty, 4)),
                    b.create<ma::BitcastOp>(i8_ty, value),
                    zero_cst));
    auto cvtIntr = value.getType().isFloat8E4M3FNUZ()
                       ? "llvm.amdgcn.cvt.f32.fp8"
                       : "llvm.amdgcn.cvt.f32.bf8";

    auto cvtOp = b.create<ml::CallIntrinsicOp>(
        mlir::TypeRange{f32_ty}, b.getStringAttr(cvtIntr),
        mlir::ValueRange{input, zero_cst},
        ml::FastmathFlagsAttr::get(b.getContext(), ml::FastmathFlags::ninf));

    auto res = cvtOp.getResult(0);

    if (to_ty == f32_ty) {
      return res;
    }

    if (to_ty.getWidth() > f32_ty.getWidth()) {
      return b.create<ma::ExtFOp>(to_ty, res);
    }

    if (to_ty.isBF16()) {
      return b.create<ma::BitcastOp>(
          to_ty,
          b.create<ml::TruncOp>(
              b.getI16Type(),
              b.create<ml::LShrOp>(b.create<ma::BitcastOp>(i32_ty, res),
                                   b.create<ml::ConstantOp>(i32_ty, 16))));
    }

    assert(to_ty.getWidth() < f32_ty.getWidth());
    return b.create<ma::TruncFOp>(to_ty, res);
  }
};

class ConvertFloatPass
    : public impl::ConvertFloatPassBase<ConvertFloatPass> {
 public:
  using ConvertFloatPassBase::ConvertFloatPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteTruncFPattern, RewriteExtFPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateConvertFloatPass() {
  return std::make_unique<ConvertFloatPass>();
}

std::optional<std::unique_ptr<mlir::Pass>> MaybeCreateConvertFloatPass(
    const se::DeviceDescription& device_description) {
  auto cc = device_description.rocm_compute_capability();

  if (cc.has_fp8_support()) {
    return CreateConvertFloatPass();
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
