/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/topk_kernel.h"

#include <cstdint>
#include <utility>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/topk_config.h"
#include "xla/comparison_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace xla::gpu {
namespace {

namespace arith = ::mlir::arith;
namespace ttir = ::mlir::triton;

using ::mlir::ImplicitLocOpBuilder;
using ::mlir::RankedTensorType;
using ::mlir::Type;
using ::mlir::Value;

class TopKKernelBuilder {
 public:
  TopKKernelBuilder(const TopKKernelSpec& spec, const TopKKernelConfig& config,
                    ImplicitLocOpBuilder& b)
      : spec_(spec),
        config_(config),
        b_(b),
        i32_(b.getI32Type()),
        i64_(b.getI64Type()),
        rows_(config.rows_per_program),
        block_k_(static_cast<int64_t>(
            absl::bit_ceil(static_cast<uint64_t>(spec.k)))) {
    // Rows of the last program can lie past `batch`, so the offsets are sized
    // for the padded batch.
    const int64_t padded_batch = config.num_programs * config.rows_per_program;
    const bool fits_i32 = padded_batch * spec.n < (int64_t{1} << 31) &&
                          padded_batch * spec.k < (int64_t{1} << 31);
    offset_type_ = fits_i32 ? i32_ : i64_;
  }

  // Emits the body of a function with the arguments (input, values, indices).
  void EmitBody(Value input, Value values, Value indices) {
    const bool partial = spec_.order == ComparisonOrder::kPartial;

    Value pid = ttir::GetProgramIdOp::create(b_, 0);
    Value first_row = arith::MulIOp::create(b_, Cast(pid, offset_type_),
                                            Const(offset_type_, rows_));
    // [rows]
    Value row_ids = arith::AddIOp::create(b_, Splat(first_row, {rows_}),
                                          Cast(Range(rows_), offset_type_));
    Value row_mask =
        arith::CmpIOp::create(b_, arith::CmpIPredicate::slt, row_ids,
                              Const(offset_type_, spec_.batch, {rows_}));

    // [rows, block_n]
    const llvm::SmallVector<int64_t> block = {rows_, config_.block_n};
    Value cols = Range(config_.block_n);
    Value col_mask =
        arith::CmpIOp::create(b_, arith::CmpIPredicate::slt, cols,
                              Const(i32_, spec_.n, {config_.block_n}));
    Value mask =
        arith::AndIOp::create(b_, Rows(row_mask, block), Cols(col_mask, block));
    Value in_offsets = arith::AddIOp::create(
        b_,
        Rows(arith::MulIOp::create(b_, row_ids,
                                   Const(offset_type_, spec_.n, {rows_})),
             block),
        Cols(Cast(cols, offset_type_), block));
    Type in_type =
        mlir::cast<ttir::PointerType>(input.getType()).getPointeeType();
    Value x = ttir::LoadOp::create(b_, Ptrs(input, in_offsets, block), mask,
                                   Zero(in_type, block),
                                   /*isVolatile=*/false);
    Value col_ids = Cols(cols, block);
    Value keys = arith::SelectOp::create(b_, mask, Pack(x, col_ids, partial),
                                         Const(i64_, 0, block));

    // [rows, block_k]
    const llvm::SmallVector<int64_t> out_block = {rows_, block_k_};
    Value out_cols = Range(block_k_);
    Value out_col_ids = Cols(out_cols, out_block);
    Value top_init = Const(i64_, 0, out_block);

    // Round j moves the largest remaining key of each row to column j. Keys
    // within a row are distinct, so exactly one element is removed per row.
    auto loop = mlir::scf::ForOp::create(b_, Const(i32_, 0),
                                         Const(i32_, spec_.k), Const(i32_, 1),
                                         mlir::ValueRange{keys, top_init});
    {
      mlir::OpBuilder::InsertionGuard guard(b_);
      b_.setInsertionPointToStart(loop.getBody());
      Value j = loop.getInductionVar();
      Value cur_keys = loop.getRegionIterArgs()[0];
      Value cur_top = loop.getRegionIterArgs()[1];
      Value row_max = RowMax(cur_keys);
      Value is_max = arith::CmpIOp::create(b_, arith::CmpIPredicate::eq,
                                           cur_keys, Rows(row_max, block));
      Value next_keys =
          arith::SelectOp::create(b_, is_max, Const(i64_, 0, block), cur_keys);
      Value is_col_j = arith::CmpIOp::create(b_, arith::CmpIPredicate::eq,
                                             out_col_ids, Splat(j, out_block));
      Value next_top = arith::SelectOp::create(
          b_, is_col_j, Rows(row_max, out_block), cur_top);
      mlir::scf::YieldOp::create(b_, mlir::ValueRange{next_keys, next_top});
    }
    Value top = loop.getResult(1);

    Value out_mask = arith::AndIOp::create(
        b_, Rows(row_mask, out_block),
        Cols(arith::CmpIOp::create(b_, arith::CmpIPredicate::slt, out_cols,
                                   Const(i32_, spec_.k, {block_k_})),
             out_block));
    Value out_offsets = arith::AddIOp::create(
        b_,
        Rows(arith::MulIOp::create(b_, row_ids,
                                   Const(offset_type_, spec_.k, {rows_})),
             out_block),
        Cols(Cast(out_cols, offset_type_), out_block));
    auto [top_values, top_indices] = Unpack(top, in_type, partial);
    ttir::StoreOp::create(b_, Ptrs(values, out_offsets, out_block), top_values,
                          out_mask, /*ignore_cta=*/false);
    ttir::StoreOp::create(b_, Ptrs(indices, out_offsets, out_block),
                          top_indices, out_mask, /*ignore_cta=*/false);
  }

 private:
  // The key of upstream's BuildPackF32ToU64. BF16 bits shifted left by 16 are
  // the F32 bits of the same value, so BF16 uses the F32 key.
  //   radix = bits ^ ((bits >>s 31) | 0x80000000)
  //   TOTAL:   hi = radix,              lo = 0xFFFFFFFF - idx
  //   PARTIAL: hi = radix + is_neg_zero, lo = ((0x7FFFFFFF - idx) << 1)
  //                                           | is_neg_zero
  // PARTIAL ranks -0.0 like +0.0 by the lowest index, and the low bit lets the
  // decode restore -0.0.
  Value Pack(Value x, Value idx, bool partial) {
    auto shape = mlir::cast<RankedTensorType>(x.getType()).getShape();
    Value bits = F32Bits(x);
    Value sign_bit = Const(i32_, 0x80000000u, shape);
    Value radix = arith::XOrIOp::create(
        b_, bits,
        arith::OrIOp::create(
            b_, arith::ShRSIOp::create(b_, bits, Const(i32_, 31, shape)),
            sign_bit));
    Value hi = radix;
    Value lo;
    if (partial) {
      Value is_neg_zero = arith::ExtUIOp::create(
          b_, Tensor(i32_, shape),
          arith::CmpIOp::create(b_, arith::CmpIPredicate::eq, bits, sign_bit));
      hi = arith::AddIOp::create(b_, radix, is_neg_zero);
      lo = arith::OrIOp::create(
          b_,
          arith::ShLIOp::create(
              b_,
              arith::SubIOp::create(b_, Const(i32_, 0x7FFFFFFFu, shape), idx),
              Const(i32_, 1, shape)),
          is_neg_zero);
    } else {
      lo = arith::SubIOp::create(b_, Const(i32_, 0xFFFFFFFFu, shape), idx);
    }
    Type key_type = Tensor(i64_, shape);
    return arith::OrIOp::create(
        b_,
        arith::ShLIOp::create(b_, arith::ExtUIOp::create(b_, key_type, hi),
                              Const(i64_, 32, shape)),
        arith::ExtUIOp::create(b_, key_type, lo));
  }

  // The inverse of Pack, as upstream's BuildUnpackU64ToF32.
  std::pair<Value, Value> Unpack(Value key, Type value_type, bool partial) {
    auto shape = mlir::cast<RankedTensorType>(key.getType()).getShape();
    Type i32_tensor = Tensor(i32_, shape);
    Value lo = arith::TruncIOp::create(b_, i32_tensor, key);
    Value hi = arith::TruncIOp::create(
        b_, i32_tensor,
        arith::ShRUIOp::create(b_, key, Const(i64_, 32, shape)));
    Value one = Const(i32_, 1, shape);
    Value idx;
    if (partial) {
      idx = arith::SubIOp::create(b_, Const(i32_, 0x7FFFFFFFu, shape),
                                  arith::ShRUIOp::create(b_, lo, one));
      hi = arith::SubIOp::create(b_, hi, arith::AndIOp::create(b_, lo, one));
    } else {
      idx = arith::SubIOp::create(b_, Const(i32_, 0xFFFFFFFFu, shape), lo);
    }
    // bits = r ^ (((r >>u 31) - 1) | 0x80000000)
    Value bits = arith::XOrIOp::create(
        b_, hi,
        arith::OrIOp::create(
            b_,
            arith::SubIOp::create(
                b_, arith::ShRUIOp::create(b_, hi, Const(i32_, 31, shape)),
                one),
            Const(i32_, 0x80000000u, shape)));
    Value value;
    if (value_type.isF32()) {
      value = ttir::BitcastOp::create(b_, Tensor(value_type, shape), bits);
    } else {
      Value bf16_bits = arith::TruncIOp::create(
          b_, Tensor(b_.getI16Type(), shape),
          arith::ShRUIOp::create(b_, bits, Const(i32_, 16, shape)));
      value = ttir::BitcastOp::create(b_, Tensor(value_type, shape), bf16_bits);
    }
    return {value, idx};
  }

  // The F32 bit pattern of each element of `x` (F32 or BF16), as i32.
  Value F32Bits(Value x) {
    auto x_type = mlir::cast<RankedTensorType>(x.getType());
    auto shape = x_type.getShape();
    if (x_type.getElementType().isF32()) {
      return ttir::BitcastOp::create(b_, Tensor(i32_, shape), x);
    }
    Value bf16_bits =
        ttir::BitcastOp::create(b_, Tensor(b_.getI16Type(), shape), x);
    return arith::ShLIOp::create(
        b_, arith::ExtUIOp::create(b_, Tensor(i32_, shape), bf16_bits),
        Const(i32_, 16, shape));
  }

  // The unsigned maximum of each row of a [rows, n] tensor of keys.
  Value RowMax(Value keys) {
    Type row_type = Tensor(i64_, {rows_});
    auto reduce = ttir::ReduceOp::create(b_, row_type, keys, /*axis=*/1);
    mlir::OpBuilder::InsertionGuard guard(b_);
    mlir::Block* combiner =
        b_.createBlock(&reduce.getCombineOp(), reduce.getCombineOp().begin(),
                       {i64_, i64_}, {b_.getLoc(), b_.getLoc()});
    ttir::ReduceReturnOp::create(
        b_, mlir::ValueRange{arith::MaxUIOp::create(
                b_, combiner->getArgument(0), combiner->getArgument(1))});
    return reduce->getResult(0);
  }

  RankedTensorType Tensor(Type element_type, llvm::ArrayRef<int64_t> shape) {
    return RankedTensorType::get(shape, element_type);
  }

  // A constant of type `type`, or a tensor of `shape` filled with it. `value`
  // is taken modulo 2^width.
  Value Const(Type type, uint64_t value, llvm::ArrayRef<int64_t> shape = {}) {
    const unsigned width = type.getIntOrFloatBitWidth();
    mlir::IntegerAttr attr =
        b_.getIntegerAttr(type, llvm::APInt(width, value, /*isSigned=*/false,
                                            /*implicitTrunc=*/true));
    if (shape.empty()) {
      return arith::ConstantOp::create(b_, attr);
    }
    return Fill(attr, shape);
  }

  Value Zero(Type type, llvm::ArrayRef<int64_t> shape) {
    return Fill(b_.getZeroAttr(type), shape);
  }

  Value Fill(mlir::TypedAttr element, llvm::ArrayRef<int64_t> shape) {
    mlir::Attribute attr = element;
    return arith::ConstantOp::create(
        b_,
        mlir::DenseElementsAttr::get(Tensor(element.getType(), shape), attr));
  }

  Value Range(int64_t size) {
    return ttir::MakeRangeOp::create(b_, Tensor(i32_, {size}), 0, size);
  }

  Value Splat(Value scalar, llvm::ArrayRef<int64_t> shape) {
    return ttir::SplatOp::create(b_, Tensor(scalar.getType(), shape), scalar);
  }

  // Sign-extends an i32 scalar or tensor to the integer type `type`.
  Value Cast(Value x, Type type) {
    Type x_type = x.getType();
    Type element_type = mlir::getElementTypeOrSelf(x_type);
    if (element_type == type) {
      return x;
    }
    Type result_type = type;
    if (auto tensor = mlir::dyn_cast<RankedTensorType>(x_type)) {
      result_type = Tensor(type, tensor.getShape());
    }
    return arith::ExtSIOp::create(b_, result_type, x);
  }

  // Broadcasts a [rows] tensor along the columns of `shape`.
  Value Rows(Value x, llvm::ArrayRef<int64_t> shape) {
    return Broadcast(ttir::ExpandDimsOp::create(b_, x, 1), shape);
  }

  // Broadcasts a [columns] tensor along the rows of `shape`.
  Value Cols(Value x, llvm::ArrayRef<int64_t> shape) {
    return Broadcast(ttir::ExpandDimsOp::create(b_, x, 0), shape);
  }

  Value Broadcast(Value x, llvm::ArrayRef<int64_t> shape) {
    Type element_type = mlir::getElementTypeOrSelf(x.getType());
    return ttir::BroadcastOp::create(b_, Tensor(element_type, shape), x);
  }

  Value Ptrs(Value base, Value offsets, llvm::ArrayRef<int64_t> shape) {
    Value ptrs = Splat(base, shape);
    return ttir::AddPtrOp::create(b_, ptrs.getType(), ptrs, offsets);
  }

  const TopKKernelSpec& spec_;
  const TopKKernelConfig& config_;
  ImplicitLocOpBuilder& b_;
  Type i32_;
  Type i64_;
  Type offset_type_;
  const int64_t rows_;
  const int64_t block_k_;
};

bool IsPowerOfTwo(int64_t x) {
  return x > 0 && absl::has_single_bit(static_cast<uint64_t>(x));
}

}  // namespace

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> BuildTopKKernel(
    const TopKKernelSpec& spec, const TopKKernelConfig& config,
    absl::string_view name, mlir::MLIRContext& context) {
  TF_RETURN_IF_ERROR(CheckTopKKernelLimits(spec));
  if (!IsPowerOfTwo(config.block_n) || config.block_n < spec.n ||
      !IsPowerOfTwo(config.rows_per_program) || config.num_programs < 1 ||
      config.num_programs * config.rows_per_program < spec.batch) {
    return InvalidArgument("TopK kernel config %s does not cover %s",
                           config.ToString(), spec.ToString());
  }

  ImplicitLocOpBuilder b(
      mlir::NameLoc::get(mlir::StringAttr::get(&context, name)), &context);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(b);
  b.setInsertionPointToEnd(module->getBody());

  Type value_type =
      spec.input_type == F32 ? Type(b.getF32Type()) : Type(b.getBF16Type());
  Type value_ptr = ttir::PointerType::get(value_type);
  Type index_ptr = ttir::PointerType::get(b.getI32Type());
  auto function_type =
      b.getFunctionType({value_ptr, value_ptr, index_ptr}, /*results=*/{});
  auto func = ttir::FuncOp::create(b, name, function_type);
  for (int i = 0; i < func.getNumArguments(); ++i) {
    func.setArgAttr(i, "tt.divisibility", b.getI32IntegerAttr(16));
  }

  b.setInsertionPointToStart(func.addEntryBlock());
  TopKKernelBuilder(spec, config, b)
      .EmitBody(func.getArgument(0), func.getArgument(1), func.getArgument(2));
  ttir::ReturnOp::create(b);

  if (mlir::failed(mlir::verify(*module))) {
    return Internal("Invalid TopK kernel for %s", spec.ToString());
  }
  return module;
}

}  // namespace xla::gpu
