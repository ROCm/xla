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

#include "xla/codegen/xtile/codegen/conv_algorithms.h"

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/codegen/dot_algorithms.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace xtile {

namespace {

namespace stablehlo = ::mlir::stablehlo;
using ::llvm::SmallVector;
using ::mlir::ArrayRef;
using ::mlir::Type;
using ::mlir::Value;

bool IsIdentityPermutation(ArrayRef<int64_t> permutation) {
  for (int64_t i = 0; i < static_cast<int64_t>(permutation.size()); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

SmallVector<int64_t> InversePermutation(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inverse(permutation.size());
  for (int64_t i = 0; i < static_cast<int64_t>(permutation.size()); ++i) {
    inverse[permutation[i]] = i;
  }
  return inverse;
}

TensorValue MaybeTranspose(mlir::ImplicitLocOpBuilder& b, TensorValue input,
                           ArrayRef<int64_t> permutation) {
  if (IsIdentityPermutation(permutation)) {
    return input;
  }
  ArrayRef<int64_t> input_shape = input.getType().getShape();
  SmallVector<int64_t> output_shape(permutation.size());
  for (int64_t i = 0; i < static_cast<int64_t>(permutation.size()); ++i) {
    output_shape[i] = input_shape[permutation[i]];
  }
  auto output_type = mlir::RankedTensorType::get(
      output_shape, input.getType().getElementType());
  return mlir::cast<TensorValue>(
      stablehlo::TransposeOp::create(b, output_type, input,
                                     b.getDenseI64ArrayAttr(permutation))
          .getResult());
}

TensorValue EmitReshape(mlir::ImplicitLocOpBuilder& b, TensorValue input,
                        ArrayRef<int64_t> new_shape) {
  auto output_type =
      mlir::RankedTensorType::get(new_shape, input.getType().getElementType());
  return mlir::cast<TensorValue>(
      stablehlo::ReshapeOp::create(b, output_type, input).getResult());
}

}  // namespace

absl::StatusOr<ConvDotShim> MakeConvDotShim(
    const HloConvolutionInstruction& conv) {
  PrimitiveType lhs_et = conv.operand(0)->shape().element_type();
  PrimitiveType rhs_et = conv.operand(1)->shape().element_type();
  PrimitiveType out_et = conv.shape().element_type();
  ConvDotShim shim;
  shim.lhs = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(lhs_et, {1, 1}), "lhs");
  shim.rhs = HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(rhs_et, {1, 1}), "rhs");
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  shim.dot = HloInstruction::CreateDot(ShapeUtil::MakeShape(out_et, {1, 1}),
                                       shim.lhs.get(), shim.rhs.get(), dnums,
                                       conv.precision_config());
  return shim;
}

absl::StatusOr<Type> GetConvAccumulatorType(
    mlir::ImplicitLocOpBuilder& b, const HloConvolutionInstruction& conv) {
  PrimitiveType lhs_et = conv.operand(0)->shape().element_type();
  PrimitiveType rhs_et = conv.operand(1)->shape().element_type();

  if (lhs_et != rhs_et) {
    return absl::UnimplementedError(absl::StrCat(
        "Convolution requires matching operand element types, got ",
        primitive_util::LowercasePrimitiveTypeName(lhs_et), " and ",
        primitive_util::LowercasePrimitiveTypeName(rhs_et), "."));
  }

  TF_ASSIGN_OR_RETURN(ConvDotShim shim, MakeConvDotShim(conv));
  TF_ASSIGN_OR_RETURN(Type accumulator_type,
                      GetDotAccumulatorType(
                          b, *::xla::Cast<HloDotInstruction>(shim.dot.get())));

  // `GetAlgUnsetAccumulatorType` only yields an integer accumulator for the
  // `i8 x i8 -> i32` case and otherwise falls through to f32. For an
  // integral convolution that means accumulating in floating point and then
  // truncating back, which is not the requested arithmetic, so reject it
  // instead of silently emitting it. 
  TF_ASSIGN_OR_RETURN(Type lhs_type, PrimitiveTypeToMlirType(b, lhs_et));
  if (mlir::isa<mlir::IntegerType>(lhs_type) &&
      !mlir::isa<mlir::IntegerType>(accumulator_type)) {
    return absl::UnimplementedError(absl::StrCat(
        "Convolution with integral operand types is not supported: ",
        primitive_util::LowercasePrimitiveTypeName(lhs_et),
        " operands would accumulate in a non-integral type."));
  }

  return accumulator_type;
}

absl::StatusOr<TensorValue> CanonicalizeConvKernelToKN(
    mlir::ImplicitLocOpBuilder& b, TensorValue kernel_tile,
    const HloConvolutionInstruction& conv) {
  const auto& dnums = conv.convolution_dimension_numbers();
  const int64_t rank = kernel_tile.getType().getRank();
  const int64_t spatial_rank = dnums.kernel_spatial_dimensions_size();
  if (rank != spatial_rank + 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("CanonicalizeConvKernelToKN: kernel rank ", rank,
                     " does not match spatial_rank + 2 = ", spatial_rank + 2));
  }

  SmallVector<int64_t> permutation;
  permutation.reserve(rank);
  for (int64_t i = 0; i < spatial_rank; ++i) {
    permutation.push_back(dnums.kernel_spatial_dimensions(i));
  }
  permutation.push_back(dnums.kernel_input_feature_dimension());
  permutation.push_back(dnums.kernel_output_feature_dimension());

  TensorValue transposed = MaybeTranspose(b, kernel_tile, permutation);
  ArrayRef<int64_t> transposed_shape = transposed.getType().getShape();
  int64_t k_size = 1;
  for (int64_t i = 0; i < rank - 1; ++i) {
    k_size *= transposed_shape[i];
  }
  int64_t n_size = transposed_shape[rank - 1];
  return EmitReshape(b, transposed, {k_size, n_size});
}

absl::StatusOr<TensorValue> CanonicalizeConvInputToMK(
    mlir::ImplicitLocOpBuilder& b, TensorValue input_tile,
    const HloConvolutionInstruction& conv) {
  const auto& dnums = conv.convolution_dimension_numbers();
  const int64_t rank = input_tile.getType().getRank();
  const int64_t spatial_rank = dnums.input_spatial_dimensions_size();
  if (rank != spatial_rank + 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("CanonicalizeConvInputToMK: input rank ", rank,
                     " does not match spatial_rank + 2 = ", spatial_rank + 2));
  }

  SmallVector<int64_t> permutation;
  permutation.reserve(rank);
  permutation.push_back(dnums.input_batch_dimension());
  for (int64_t i = 0; i < spatial_rank; ++i) {
    permutation.push_back(dnums.input_spatial_dimensions(i));
  }
  permutation.push_back(dnums.input_feature_dimension());

  TensorValue transposed = MaybeTranspose(b, input_tile, permutation);
  ArrayRef<int64_t> transposed_shape = transposed.getType().getShape();
  int64_t m_size = 1;
  for (int64_t i = 0; i < rank - 1; ++i) {
    m_size *= transposed_shape[i];
  }
  int64_t k_size = transposed_shape[rank - 1];
  return EmitReshape(b, transposed, {m_size, k_size});
}

absl::StatusOr<TensorValue> CanonicalizeConvAccToMN(
    mlir::ImplicitLocOpBuilder& b, Value acc,
    const HloConvolutionInstruction& conv) {
  auto acc_tv = mlir::dyn_cast<TensorValue>(acc);
  if (!acc_tv) {
    return absl::InvalidArgumentError(
        "CanonicalizeConvAccToMN: acc is not a ranked tensor");
  }
  const auto& dnums = conv.convolution_dimension_numbers();
  const int64_t rank = acc_tv.getType().getRank();
  const int64_t spatial_rank = dnums.output_spatial_dimensions_size();
  if (rank != spatial_rank + 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("CanonicalizeConvAccToMN: acc rank ", rank,
                     " does not match spatial_rank + 2 = ", spatial_rank + 2));
  }

  SmallVector<int64_t> permutation;
  permutation.reserve(rank);
  permutation.push_back(dnums.output_batch_dimension());
  for (int64_t i = 0; i < spatial_rank; ++i) {
    permutation.push_back(dnums.output_spatial_dimensions(i));
  }
  permutation.push_back(dnums.output_feature_dimension());

  TensorValue transposed = MaybeTranspose(b, acc_tv, permutation);
  ArrayRef<int64_t> transposed_shape = transposed.getType().getShape();
  int64_t m_size = 1;
  for (int64_t i = 0; i < rank - 1; ++i) {
    m_size *= transposed_shape[i];
  }
  int64_t n_size = transposed_shape[rank - 1];
  return EmitReshape(b, transposed, {m_size, n_size});
}

absl::StatusOr<Value> RestoreConvAccFromMN(
    mlir::ImplicitLocOpBuilder& b, Value acc_2d,
    const HloConvolutionInstruction& conv, ArrayRef<int64_t> acc_tile_shape) {
  auto acc_tv = mlir::dyn_cast<TensorValue>(acc_2d);
  if (!acc_tv) {
    return absl::InvalidArgumentError(
        "RestoreConvAccFromMN: accumulator is not a ranked tensor");
  }
  if (acc_tv.getType().getRank() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("RestoreConvAccFromMN: expected rank-2 input, got rank ",
                     acc_tv.getType().getRank()));
  }

  const auto& dnums = conv.convolution_dimension_numbers();
  const int64_t spatial_rank = dnums.output_spatial_dimensions_size();
  const int64_t rank = spatial_rank + 2;

  if (static_cast<int64_t>(acc_tile_shape.size()) != rank) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RestoreConvAccFromMN: accumulator rank ", acc_tile_shape.size(),
        " does not match conv output rank ", rank));
  }

  SmallVector<int64_t> canonical_shape;
  canonical_shape.reserve(rank);
  canonical_shape.push_back(acc_tile_shape[dnums.output_batch_dimension()]);
  for (int64_t i = 0; i < spatial_rank; ++i) {
    canonical_shape.push_back(
        acc_tile_shape[dnums.output_spatial_dimensions(i)]);
  }
  canonical_shape.push_back(acc_tile_shape[dnums.output_feature_dimension()]);

  // Sanity check: the products must match the rank-2 tile dimensions.
  int64_t expected_m = 1;
  for (int64_t i = 0; i < rank - 1; ++i) {
    expected_m *= canonical_shape[i];
  }
  int64_t expected_n = canonical_shape[rank - 1];
  ArrayRef<int64_t> tile_shape = acc_tv.getType().getShape();
  if (tile_shape[0] != expected_m || tile_shape[1] != expected_n) {
    return absl::InvalidArgumentError(
        absl::StrCat("RestoreConvAccFromMN: rank-2 tile shape [", tile_shape[0],
                     ",", tile_shape[1], "] does not match expected [",
                     expected_m, ",", expected_n, "] from conv output shape."));
  }

  // Reshape and then un-permute to the original output dnums layout
  TensorValue reshaped = EmitReshape(b, acc_tv, canonical_shape);

  SmallVector<int64_t> permutation;
  permutation.reserve(rank);
  permutation.push_back(dnums.output_batch_dimension());
  for (int64_t i = 0; i < spatial_rank; ++i) {
    permutation.push_back(dnums.output_spatial_dimensions(i));
  }
  permutation.push_back(dnums.output_feature_dimension());

  SmallVector<int64_t> inverse = InversePermutation(permutation);
  return Value(MaybeTranspose(b, reshaped, inverse));
}

}  // namespace xtile
}  // namespace xla
