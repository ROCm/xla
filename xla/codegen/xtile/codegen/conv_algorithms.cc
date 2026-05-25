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

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/algorithm_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace xtile {

absl::StatusOr<::mlir::Type> GetConvAccumulatorType(
    mlir::ImplicitLocOpBuilder& b,
    const HloConvolutionInstruction& conv) {
  const PrecisionConfig::Algorithm algorithm =
      conv.precision_config().algorithm();
  if (algorithm == PrecisionConfig::ALG_UNSET) {
    TF_ASSIGN_OR_RETURN(::mlir::Type input_type,
                      PrimitiveTypeToMlirType(b, conv.operand(0)->shape().element_type()));
    TF_ASSIGN_OR_RETURN(::mlir::Type accumulator_type,
                      PrimitiveTypeToMlirType(b, conv.shape().element_type()));
    return (accumulator_type.isF64() && input_type.isF64()) ? b.getF64Type()
                                                            : b.getF32Type();
  }
  TF_ASSIGN_OR_RETURN(PrimitiveType accumulator_type,
                      algorithm_util::GetDotAccumulatorType(algorithm));
  return PrimitiveTypeToMlirType(b, accumulator_type);
}

}  // namespace xtile
}  // namespace xla
