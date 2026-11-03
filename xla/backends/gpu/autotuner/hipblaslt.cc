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

#include "xla/backends/gpu/autotuner/hipblaslt.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/hipblaslt_scale_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;
using se::gpu::BlasLt;

using HipblasLtBackendConfig = AutotuneResult::GemmKey;
using MxScaleLayout = HipblasLtBackendConfig::MxScaleLayout;

namespace {

absl::StatusOr<se::gpu::ScaleMode> MxScaleMode(MxScaleLayout layout) {
  switch (layout) {
    case HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR:
      return se::gpu::ScaleMode::kBlockScaling;
    case HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8:
      return se::gpu::ScaleMode::kBlockScaling32x8;
    default:
      return absl::InvalidArgumentError("Unknown hipBLASLt MX scale layout");
  }
}

absl::StatusOr<absl::InlinedVector<MxScaleLayout, 2>> MxScaleLayoutsToTune(
    const HloInstruction& scaled_dot, const DebugOptions& options,
    const se::GpuComputeCapability& gpu_version) {
  absl::InlinedVector<MxScaleLayout, 2> layouts;
  switch (options.xla_gpu_experimental_hipblaslt_mx_scale_layout()) {
    case DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_LINEAR:
      layouts.push_back(HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR);
      break;
    case DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_AUTO:
      layouts.push_back(HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR);
      [[fallthrough]];
    case DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8:
      if (CanUseHipblasLtScale32x8(scaled_dot, gpu_version).IsAllowed()) {
        layouts.push_back(
            HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8);
      }
      break;
    default:
      return absl::InvalidArgumentError(
          "Unknown hipBLASLt MX scale layout mode");
  }
  return layouts;
}

absl::StatusOr<BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::SILU:
      return BlasLt::Epilogue::kSILU;
    case GemmBackendConfig::BIAS:
      return BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return BlasLt::Epilogue::kBiasThenGELUWithAux;
    case GemmBackendConfig::BIAS_SILU:
      return BlasLt::Epilogue::kBiasThenSILU;
    default:
      return Internal("Unsupported Epilogue.");
  }
}

bool IsValidMxScaledDot(const HloInstruction* scaled_dot) {
  const Shape& lhs_shape = scaled_dot->operand(0)->shape();
  const Shape& rhs_shape = scaled_dot->operand(1)->shape();
  const Shape& lhs_scale_shape = scaled_dot->operand(2)->shape();
  const Shape& rhs_scale_shape = scaled_dot->operand(3)->shape();
  const Shape& output_shape = scaled_dot->shape();
  const DotDimensionNumbers& dot_dims = scaled_dot->dot_dimension_numbers();

  auto IsValidInputType = [](PrimitiveType type) {
    return type == F8E4M3FN || type == F8E5M2 || type == F4E2M1FN;
  };

  PrimitiveType lhs_type = lhs_shape.element_type();
  PrimitiveType rhs_type = rhs_shape.element_type();
  if (!IsValidInputType(lhs_type) || !IsValidInputType(rhs_type)) {
    VLOG(2) << "hipBLASLt MX: unsupported data type, lhs="
            << PrimitiveType_Name(lhs_type)
            << " rhs=" << PrimitiveType_Name(rhs_type);
    return false;
  }

  if (lhs_scale_shape.element_type() != F8E8M0FNU ||
      rhs_scale_shape.element_type() != F8E8M0FNU) {
    VLOG(2) << "hipBLASLt MX: scale type must be F8E8M0FNU";
    return false;
  }

  PrimitiveType out_type = output_shape.element_type();
  if (out_type != F32 && out_type != F16 && out_type != BF16) {
    VLOG(2) << "hipBLASLt MX: output type must be F32/F16/BF16, got "
            << PrimitiveType_Name(out_type);
    return false;
  }

  int64_t batch_size = 1;
  for (int64_t dim : dot_dims.lhs_batch_dimensions()) {
    batch_size *= lhs_shape.dimensions(dim);
  }
  if (batch_size != 1) {
    VLOG(2) << "hipBLASLt MX: batch_size > 1 not supported, got " << batch_size;
    return false;
  }

  int64_t m = 1;
  for (int64_t i = 0; i < lhs_shape.dimensions().size(); ++i) {
    if (!absl::c_linear_search(dot_dims.lhs_batch_dimensions(), i) &&
        !absl::c_linear_search(dot_dims.lhs_contracting_dimensions(), i)) {
      m *= lhs_shape.dimensions(i);
    }
  }

  int64_t n = 1;
  for (int64_t i = 0; i < rhs_shape.dimensions().size(); ++i) {
    if (!absl::c_linear_search(dot_dims.rhs_batch_dimensions(), i) &&
        !absl::c_linear_search(dot_dims.rhs_contracting_dimensions(), i)) {
      n *= rhs_shape.dimensions(i);
    }
  }

  int64_t k = 1;
  for (int64_t dim : dot_dims.lhs_contracting_dimensions()) {
    k *= lhs_shape.dimensions(dim);
  }

  if (m % 16 != 0 || n % 16 != 0) {
    VLOG(2) << "hipBLASLt MX: M and N must be divisible by 16, got M=" << m
            << " N=" << n;
    return false;
  }

  if (k % 32 != 0) {
    VLOG(2) << "hipBLASLt MX: K must be divisible by 32, got K=" << k;
    return false;
  }

  int64_t lhs_scale_k = 1;
  for (int64_t dim : dot_dims.lhs_contracting_dimensions()) {
    lhs_scale_k *= lhs_scale_shape.dimensions(dim);
  }
  int64_t rhs_scale_k = 1;
  for (int64_t dim : dot_dims.rhs_contracting_dimensions()) {
    rhs_scale_k *= rhs_scale_shape.dimensions(dim);
  }
  if (lhs_scale_k == 0 || k / lhs_scale_k != 32 || rhs_scale_k == 0 ||
      k / rhs_scale_k != 32) {
    VLOG(2) << "hipBLASLt MX: block size must be 32, got lhs="
            << (lhs_scale_k > 0 ? k / lhs_scale_k : 0)
            << " rhs=" << (rhs_scale_k > 0 ? k / rhs_scale_k : 0);
    return false;
  }

  return true;
}

// Returns the scaled-dot instruction inside a Triton GEMM fusion, or nullptr
// if `instr` is not such a fusion.
const HloInstruction* GetScaledDotFromFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return nullptr;
  }
  if (gpu_config->fusion_backend_config().kind() != kTritonGemmFusionKind) {
    return nullptr;
  }
  return hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
}

// The structure of a scaled-dot fusion that ApplyConfig can lower to
// __cublas$lt$matmul$mx: each scaled-dot operand comes from its own fusion
// parameter through bitcasts, and every other instruction belongs to an
// optional elementwise epilogue, which runs after the GEMM in a loop fusion.
struct MxFusionStructure {
  // The fusion parameter that feeds each scaled-dot operand.
  std::array<int64_t, 4> dot_parameter_numbers = {};
  // The fusion parameters that feed only the epilogue, in increasing order.
  std::vector<int64_t> epilogue_parameter_numbers;
  // The epilogue instructions other than parameters, in post order. Empty if
  // and only if the scaled-dot is the root of the fusion.
  std::vector<const HloInstruction*> epilogue;

  // Whether the epilogue only reinterprets the scaled-dot result: every
  // epilogue instruction is a bitcast or a reshape, which the analysis accepts
  // only if it keeps the element type and order.
  bool EpilogueIsBitcast() const {
    return epilogue_parameter_numbers.empty() &&
           absl::c_all_of(epilogue, [](const HloInstruction* instr) {
             return instr->opcode() == HloOpcode::kBitcast ||
                    instr->opcode() == HloOpcode::kReshape;
           });
  }
};

// Returns an error unless `instr`, an epilogue instruction of a scaled-dot
// fusion, can run in a loop fusion after the GEMM: an elementwise instruction
// whose operands have its dimensions, or a bitcast or reshape that keeps the
// element order. Transposes, broadcasts, reductions, slices and tuples are
// refused.
absl::Status CheckMxEpilogueInstruction(const HloInstruction& instr) {
  if (!instr.shape().IsArray() || !instr.shape().has_layout()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "hipBLASLt MX epilogue instruction is not an array with a layout: ",
        instr.ToString()));
  }
  if (instr.opcode() == HloOpcode::kBitcast ||
      instr.opcode() == HloOpcode::kReshape) {
    const Shape& operand_shape = instr.operand(0)->shape();
    if (!operand_shape.IsArray() || !operand_shape.has_layout() ||
        !ShapeUtil::ReshapeIsBitcast(operand_shape, instr.shape())) {
      return absl::InvalidArgumentError(
          absl::StrCat("hipBLASLt MX epilogue changes the element order: ",
                       instr.ToString()));
    }
    return absl::OkStatus();
  }
  if (!instr.IsElementwise() || instr.operand_count() == 0 ||
      instr.has_called_computations() || instr.HasSideEffect()) {
    return absl::InvalidArgumentError(
        absl::StrCat("hipBLASLt MX epilogue instruction is not elementwise: ",
                     instr.ToString()));
  }
  for (const HloInstruction* operand : instr.operands()) {
    if (!operand->shape().IsArray() ||
        !ShapeUtil::SameDimensions(operand->shape(), instr.shape())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "hipBLASLt MX epilogue instruction changes the dimensions: ",
          instr.ToString()));
    }
  }
  return absl::OkStatus();
}

// Analyzes `fusion`, a Triton GEMM fusion around `scaled_dot`, for lowering
// with `scale_mode`. Returns an InvalidArgument error that names the reason if
// the fusion cannot be lowered.
absl::StatusOr<MxFusionStructure> AnalyzeMxScaledDotFusion(
    const HloInstruction& fusion, const HloInstruction& scaled_dot,
    se::gpu::ScaleMode scale_mode) {
  // ReshapeIsBitcast compares element order, not element storage width.
  // Eliding the operand path must not reinterpret byte-stored FP4 as packed
  // nibbles, including when an intermediate shape changes storage width.
  auto validate_fp4_storage = [scale_mode](const Shape& shape) -> absl::Status {
    if (scale_mode == se::gpu::ScaleMode::kBlockScaling32x8 &&
        shape.element_type() == F4E2M1FN &&
        (!shape.has_layout() || shape.layout().element_size_in_bits() != 4)) {
      return absl::InvalidArgumentError(
          "Pre-swizzled FP4 operands require packed E(4) storage throughout "
          "the operand path");
    }
    return absl::OkStatus();
  };

  const HloComputation* fused = fusion.fused_instructions_computation();
  MxFusionStructure structure;
  // The operand parameters and the instructions between them and the
  // scaled-dot, all of which the lowering elides.
  absl::flat_hash_set<const HloInstruction*> operand_paths;
  // Walk back from each scaled-dot operand to the fusion parameter it comes
  // from. The fusion's external operands may have a different rank than the
  // scaled-dot operands: when quantization happens in-graph the operand is in
  // block-scaled [..., K/block, block] form and is only flattened to [..., K]
  // by a bitcast/reshape inside the fusion. Only such trivial rank-changing
  // ops are allowed on the path, and each instruction on it must have a single
  // user, so that no data or scale value reaches the epilogue.
  for (int64_t dot_index = 0; dot_index < 4; ++dot_index) {
    const HloInstruction* cur = scaled_dot.operand(dot_index);
    while (true) {
      ABSL_RETURN_IF_ERROR(validate_fp4_storage(cur->shape()));
      if (cur->user_count() != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "The path to scaled-dot operand ", dot_index,
            " has an instruction with several users: ", cur->ToString()));
      }
      if (!operand_paths.insert(cur).second) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Several scaled-dot operands come from ", cur->ToString()));
      }
      if (cur->opcode() == HloOpcode::kParameter) {
        break;
      }
      if (cur->opcode() != HloOpcode::kBitcast &&
          cur->opcode() != HloOpcode::kReshape) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unexpected op between fusion parameter and scaled-dot: ",
            cur->ToString()));
      }
      // The whole path is elided. Equal endpoint shapes do not make that safe:
      // an intermediate bitcast followed by a materializing reshape can change
      // the byte order and still restore the same shape.
      if (cur->opcode() == HloOpcode::kReshape &&
          !ShapeUtil::ReshapeIsBitcast(cur->operand(0)->shape(),
                                       cur->shape())) {
        return absl::InvalidArgumentError(
            "hipBLASLt MX lowering cannot discard a materializing reshape");
      }
      cur = cur->operand(0);
    }
    const HloInstruction* ext = fusion.operand(cur->parameter_number());
    ABSL_RETURN_IF_ERROR(validate_fp4_storage(ext->shape()));
    const Shape& inner_shape = scaled_dot.operand(dot_index)->shape();
    if (scale_mode == se::gpu::ScaleMode::kBlockScaling32x8 &&
        !ShapeUtil::Equal(ext->shape(), inner_shape) &&
        !ShapeUtil::ReshapeIsBitcast(ext->shape(), inner_shape)) {
      return absl::InvalidArgumentError(
          "Pre-swizzled MX operands require layout-preserving reshapes");
    }
    structure.dot_parameter_numbers[dot_index] = cur->parameter_number();
  }

  // Every other instruction belongs to the epilogue. It cannot feed the
  // scaled-dot, whose operand paths start at parameters, so it consumes only
  // the scaled-dot result, the other parameters and other epilogue values.
  const HloInstruction* root = fused->root_instruction();
  if (root != &scaled_dot && scaled_dot.user_count() == 0) {
    return absl::InvalidArgumentError(
        "The scaled-dot result does not reach the fusion root");
  }
  for (const HloInstruction* instr : fused->MakeInstructionPostOrder()) {
    if (instr == &scaled_dot || operand_paths.contains(instr)) {
      continue;
    }
    if (instr != root && instr->user_count() == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unused instruction in the scaled-dot fusion: ", instr->ToString()));
    }
    if (instr->opcode() == HloOpcode::kParameter) {
      structure.epilogue_parameter_numbers.push_back(instr->parameter_number());
      continue;
    }
    ABSL_RETURN_IF_ERROR(CheckMxEpilogueInstruction(*instr));
    structure.epilogue.push_back(instr);
  }
  absl::c_sort(structure.epilogue_parameter_numbers);
  return structure;
}

// Builds the loop fusion that runs the epilogue of the scaled-dot fusion
// `fusion` on `gemm_result`, the result of the GEMM that replaces the
// scaled-dot. Its parameters are the GEMM result and then the operands of
// `fusion` that feed only the epilogue, in their order in `fusion`. Its body is
// a clone of the epilogue with the scaled-dot replaced by parameter 0, so its
// root has the shape and layout of the root of `fusion`.
absl::StatusOr<HloInstruction*> CreateMxEpilogueFusion(
    HloInstruction& fusion, const HloInstruction& scaled_dot,
    const MxFusionStructure& structure, HloInstruction* gemm_result) {
  const HloComputation* fused = fusion.fused_instructions_computation();
  HloComputation::Builder builder("hipblaslt_mx_epilogue");
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> clones;
  clones[&scaled_dot] = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scaled_dot.shape(), "gemm_result"));
  std::vector<HloInstruction*> operands = {gemm_result};
  for (int64_t parameter_number : structure.epilogue_parameter_numbers) {
    const HloInstruction* parameter =
        fused->parameter_instruction(parameter_number);
    clones[parameter] = builder.AddInstruction(
        HloInstruction::CreateParameter(static_cast<int64_t>(operands.size()),
                                        parameter->shape(), parameter->name()));
    operands.push_back(fusion.mutable_operand(parameter_number));
  }
  for (const HloInstruction* instr : structure.epilogue) {
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(instr->operand_count());
    for (const HloInstruction* operand : instr->operands()) {
      new_operands.push_back(clones.at(operand));
    }
    clones[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), new_operands));
  }
  TF_RET_CHECK(clones.contains(fused->root_instruction()));
  HloComputation* computation = fusion.GetModule()->AddEmbeddedComputation(
      builder.Build(clones.at(fused->root_instruction())));
  computation->SetExecutionThread(fusion.parent()->execution_thread());
  // ApplyConfig names the fusion after the replacement, which would give it
  // the name of `fusion` because both are fusions.
  return fusion.parent()->AddInstruction(HloInstruction::CreateFusion(
      fusion.shape(), HloInstruction::FusionKind::kLoop, operands,
      computation));
}

// Replaces `fusion` by `result`. The control predecessors of `fusion` become
// predecessors of `gemm`, which `result` depends on, and its control
// successors become successors of `result`.
absl::Status ReplaceMxScaledDotFusion(HloInstruction& fusion,
                                      HloInstruction* gemm,
                                      HloInstruction* result) {
  ABSL_RETURN_IF_ERROR(fusion.CopyAllControlDepsTo(gemm, result));
  ABSL_RETURN_IF_ERROR(fusion.DropAllControlDeps());
  ABSL_ASSIGN_OR_RETURN(bool replaced,
                        fusion.parent()->ReplaceInstruction(
                            &fusion, result, /*preserve_sharding=*/false,
                            /*relay_control_dependency=*/true));
  if (!replaced) {
    return absl::InternalError(absl::StrCat(
        "hipBLASLt MX lowering could not replace ", fusion.name()));
  }
  return absl::OkStatus();
}

// Decides whether `gemm` is an FP8-output __cublas$lt$matmul$f8, without amax
// or auxiliary output, that can run as a GEMM with a wider output followed by
// the D scale, the saturation and the conversion to FP8. A forbidding decision
// explains what prevents the rewrite.
Decision CanUnfuseFp8Output(const HloInstruction& gemm) {
  if (!IsCublasLtMatmulF8(gemm)) {
    return Decision::Forbid("not a __cublas$lt$matmul$f8 call");
  }
  if (!gemm.shape().IsTuple() || gemm.shape().tuple_shapes().size() != 2) {
    return Decision::Forbid(absl::StrCat(
        "expected an output and a workspace, got ", gemm.shape().ToString()));
  }
  PrimitiveType d_type = gemm.shape().tuple_shapes(0).element_type();
  if (d_type != F8E4M3FN && d_type != F8E5M2 && d_type != F8E4M3FNUZ &&
      d_type != F8E5M2FNUZ) {
    return Decision::Forbid(absl::StrCat(
        "the output is ", PrimitiveType_Name(d_type), ", not FP8"));
  }
  absl::StatusOr<GpuBackendConfig> gpu_config =
      gemm.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return Decision(gpu_config.status());
  }
  const GemmBackendConfig& config = gpu_config->gemm_backend_config();
  if (config.beta() != 0) {
    return Decision::Forbid(
        absl::StrCat("beta is ", config.beta(), ", so the output is read"));
  }
  absl::StatusOr<bool> has_vector_bias =
      gpublas_lt::EpilogueAddsVectorBias(config.epilogue());
  if (!has_vector_bias.ok()) {
    return Decision(has_vector_bias.status());
  }
  // An amax or auxiliary output takes an operand the unfused GEMM cannot keep.
  int64_t expected_operands =
      4 + int{*has_vector_bias} + int{config.has_d_scale()};
  if (gemm.operand_count() != expected_operands) {
    return Decision::Forbid(absl::StrCat("expected ", expected_operands,
                                         " operands, got ",
                                         gemm.operand_count()));
  }
  // The vector bias follows the A and B scales.
  if (*has_vector_bias) {
    PrimitiveType bias_type = gemm.operand(4)->shape().element_type();
    if (bias_type != F16 && bias_type != BF16) {
      return Decision::Forbid(absl::StrCat("the vector bias is ",
                                           PrimitiveType_Name(bias_type),
                                           ", not F16 or BF16"));
    }
  }
  return Decision::Allow();
}

// Adds a copy of `gemm` without the D scale whose output is F32, or has the
// type of the vector bias, since hipBLASLt reads the bias with the output type
// for FP8 inputs.
absl::StatusOr<HloInstruction*> AddWideOutputGemm(HloInstruction* gemm) {
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        gemm->backend_config<GpuBackendConfig>());
  GemmBackendConfig& config = *gpu_config.mutable_gemm_backend_config();
  std::vector<HloInstruction*> operands(gemm->operands().begin(),
                                        gemm->operands().end());
  if (config.has_d_scale()) {
    operands.pop_back();
    // Cleared, not set to false, to print like a GEMM without a D scale.
    config.clear_has_d_scale();
  }
  ABSL_ASSIGN_OR_RETURN(bool has_vector_bias,
                        gpublas_lt::EpilogueAddsVectorBias(config.epilogue()));
  Shape shape = gemm->shape();
  *shape.mutable_tuple_shapes(0) = ShapeUtil::ChangeElementType(
      shape.tuple_shapes(0),
      has_vector_bias ? operands[4]->shape().element_type() : F32);
  HloInstruction* wide = gemm->parent()->AddInstruction(
      gemm->CloneWithNewOperands(shape, operands));
  ABSL_RETURN_IF_ERROR(wide->set_backend_config(gpu_config));
  return wide;
}

// Rewrites `gemm` into `wide`, made by AddWideOutputGemm, followed by a loop
// fusion that applies the D scale, saturates like hipBLASLt and converts the
// result to the FP8 output type of `gemm`.
absl::Status RewriteWithWideOutputGemm(HloInstruction* gemm,
                                       HloInstruction* wide) {
  HloComputation* computation = gemm->parent();
  const Shape& d_shape = gemm->shape().tuple_shapes(0);
  Shape f32_shape = ShapeUtil::ChangeElementType(d_shape, F32);
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        gemm->backend_config<GpuBackendConfig>());

  // The instructions to fuse, in topological order.
  std::vector<HloInstruction*> fused;
  auto add = [&](std::unique_ptr<HloInstruction> instr) {
    fused.push_back(computation->AddInstruction(std::move(instr)));
    return fused.back();
  };
  auto add_broadcast_constant = [&](float value) {
    HloInstruction* constant =
        add(HloInstruction::CreateConstant(LiteralUtil::CreateR0(value)));
    return add(HloInstruction::CreateBroadcast(f32_shape, constant, {}));
  };
  HloInstruction* result = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(wide, 0));
  if (result->shape().element_type() != F32) {
    result = add(HloInstruction::CreateConvert(f32_shape, result));
  }
  if (gpu_config.gemm_backend_config().has_d_scale()) {
    HloInstruction* d_scale = add(HloInstruction::CreateBroadcast(
        f32_shape, gemm->operands().back(), {}));
    result = add(HloInstruction::CreateBinary(f32_shape, HloOpcode::kMultiply,
                                              result, d_scale));
  }
  ABSL_ASSIGN_OR_RETURN(
      Literal max,
      LiteralUtil::MaxFiniteValue(d_shape.element_type()).Convert(F32));
  HloInstruction* lower = add_broadcast_constant(-max.GetFirstElement<float>());
  HloInstruction* upper = add_broadcast_constant(max.GetFirstElement<float>());
  result = add(HloInstruction::CreateTernary(f32_shape, HloOpcode::kClamp,
                                             lower, result, upper));
  add(HloInstruction::CreateConvert(d_shape, result));
  absl::c_reverse(fused);
  HloInstruction* fusion = computation->CreateFusionInstruction(
      fused, HloInstruction::FusionKind::kLoop);
  fusion->fused_instructions_computation()->SetExecutionThread(
      computation->execution_thread());

  ABSL_RETURN_IF_ERROR(gemm->CopyAllControlDepsTo(wide, fusion));
  ABSL_RETURN_IF_ERROR(gemm->DropAllControlDeps());
  HloInstruction* workspace = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(wide, 1));
  for (HloInstruction* user : std::vector<HloInstruction*>(gemm->users())) {
    TF_RET_CHECK(user->opcode() == HloOpcode::kGetTupleElement)
        << user->ToString();
    ABSL_RETURN_IF_ERROR(computation
                             ->ReplaceInstructionWithDifferentShape(
                                 user,
                                 user->tuple_index() == 0 ? fusion : workspace,
                                 /*preserve_sharding=*/false,
                                 /*relay_control_dependency=*/true,
                                 /*remove_unused_operands=*/false)
                             .status());
  }
  if (computation->root_instruction() == gemm) {
    computation->set_root_instruction(computation->AddInstruction(
        HloInstruction::CreateTuple({fusion, workspace})));
  }
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(gemm));
  return workspace->IsDead() ? computation->RemoveInstruction(workspace)
                             : absl::OkStatus();
}

}  // namespace

bool HipblasLtBackend::IsSupported(const HloInstruction& instr) {
  if (IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr) ||
      IsCublasLtGroupedMatmul(instr)) {
    return true;
  }
  if (GetScaledDotFromFusion(instr) != nullptr) {
    const auto& gpu_cc =
        target_config().device_description.gpu_compute_capability();
    const auto* rocm_cc = gpu_cc.rocm_compute_capability();
    return rocm_cc != nullptr && rocm_cc->has_mx_type_support();
  }
  return false;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
HipblasLtBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }

  if (stream_executor() == nullptr) {
    return absl::InvalidArgumentError(
        "HipblasLtBackend cannot enumerate configs in deviceless mode.");
  }

  if (IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr)) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instr.backend_config<GpuBackendConfig>());
    const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();

    ABSL_ASSIGN_OR_RETURN(
        GemmConfig gemm_config,
        GemmConfig::For(
            &instr,
            target_config().device_description.gpu_compute_capability()));

    ABSL_ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                          AsBlasLtEpilogue(backend_config.epilogue()));

    ABSL_ASSIGN_OR_RETURN(BlasLt * blas_lt,
                          se::gpu::BlasLt::Get(stream_executor()));

    ABSL_ASSIGN_OR_RETURN(BlasLt::MatmulPlanPtr plan,
                          blas_lt->GetMatmulPlan(gemm_config, epilogue));

    const Shape& output_shape = instr.shape();
    if (!output_shape.IsTuple() || output_shape.tuple_shapes().empty()) {
      return Internal(
          "Invalid shape for HipblasLt matmul: output is not a non-empty "
          "tuple.");
    }
    const int64_t workspace_size =
        ShapeUtil::ByteSizeOf(output_shape.tuple_shapes().back());

    int max_algorithms = debug_options().xla_gpu_blas_max_algorithms();
    if (max_algorithms <= 0) {
      max_algorithms = GemmConfig::kNumAlgorithms;
    }

    ABSL_ASSIGN_OR_RETURN(std::vector<BlasLt::MatmulAlgorithm> algorithms,
                          plan->GetAlgorithms(max_algorithms, workspace_size));
    int num_algorithms = algorithms.size();
    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.reserve(num_algorithms);
    for (int i = 0; i < num_algorithms; ++i) {
      auto config = std::make_unique<BackendConfig>();
      auto* gemm_key = config->mutable_gemm();
      gemm_key->set_algorithm(i);
      gemm_key->set_autotune_workspace_size(workspace_size);
      configs.push_back(std::move(config));
    }
    if (configs.empty()) {
      Decision can_unfuse = CanUnfuseFp8Output(instr);
      if (can_unfuse.IsForbidden()) {
        VLOG(2) << "hipBLASLt: not unfusing the output of " << instr.name()
                << ": " << can_unfuse.Explain();
      } else {
        // Query the wider GEMM in a scratch module, leaving `instr` unchanged.
        std::unique_ptr<HloModule> module =
            ExtractInstructionIntoNewModule(instr);
        ABSL_ASSIGN_OR_RETURN(
            HloInstruction * wide,
            AddWideOutputGemm(module->entry_computation()->root_instruction()));
        ABSL_ASSIGN_OR_RETURN(configs, GetSupportedConfigs(*wide));
        for (std::unique_ptr<BackendConfig>& config : configs) {
          config->mutable_gemm()->set_unfuse_fp8_output(true);
        }
        if (!configs.empty()) {
          LOG(WARNING)
              << "hipBLASLt: no algorithms found for the FP8 output of "
              << instr.name()
              << "; using a wider output and a separate FP8 conversion.";
        }
      }
    }
    return configs;
  }

  if (const auto* scaled_dot = GetScaledDotFromFusion(instr)) {
    if (!IsValidMxScaledDot(scaled_dot)) {
      LOG(WARNING) << "hipBLASLt MX: IsValidMxScaledDot failed";
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    ABSL_ASSIGN_OR_RETURN(
        auto layouts,
        MxScaleLayoutsToTune(
            *scaled_dot, debug_options(),
            target_config().device_description.gpu_compute_capability()));
    if (layouts.empty()) {
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    const Shape& lhs_shape = scaled_dot->operand(0)->shape();
    const Shape& rhs_shape = scaled_dot->operand(1)->shape();
    const DotDimensionNumbers& dot_dims = scaled_dot->dot_dimension_numbers();
    const Shape& output_shape = scaled_dot->shape();

    auto gemm_config_or = GemmConfig::For(
        lhs_shape, dot_dims.lhs_batch_dimensions(),
        dot_dims.lhs_contracting_dimensions(), rhs_shape,
        dot_dims.rhs_batch_dimensions(), dot_dims.rhs_contracting_dimensions(),
        output_shape,
        /*alpha_real=*/1.0, /*alpha_imag=*/0.0, /*beta=*/0.0,
        PrecisionConfig::ALG_UNSET, /*algorithm=*/std::nullopt,
        se::blas::kDefaultComputePrecision, /*grad_x=*/false,
        /*grad_y=*/false, /*scale_mode=*/se::gpu::ScaleMode::kBlockScaling,
        target_config().device_description.gpu_compute_capability());
    if (!gemm_config_or.ok()) {
      LOG(WARNING) << "hipBLASLt MX: GemmConfig::For failed: "
                   << gemm_config_or.status();
      return std::vector<std::unique_ptr<BackendConfig>>();
    }

    ABSL_ASSIGN_OR_RETURN(BlasLt * blas_lt,
                          se::gpu::BlasLt::Get(stream_executor()));
    int64_t workspace_size = GemmConfig::kGFX950Workspace;
    int max_algorithms = debug_options().xla_gpu_blas_max_algorithms();
    if (max_algorithms <= 0) {
      max_algorithms = GemmConfig::kNumAlgorithms;
    }
    std::vector<std::unique_ptr<BackendConfig>> configs;
    for (MxScaleLayout layout : layouts) {
      ABSL_ASSIGN_OR_RETURN(gemm_config_or->scale_mode, MxScaleMode(layout));
      // Offer only configs that ApplyConfig can lower.
      absl::Status lowerable =
          AnalyzeMxScaledDotFusion(instr, *scaled_dot,
                                   gemm_config_or->scale_mode)
              .status();
      if (!lowerable.ok()) {
        VLOG(2) << "hipBLASLt MX: cannot lower " << instr.name()
                << " with scale layout " << layout << ": "
                << lowerable.message();
        continue;
      }
      auto plan_or =
          blas_lt->GetMatmulPlan(*gemm_config_or, BlasLt::Epilogue::kDefault);
      if (!plan_or.ok()) {
        VLOG(2) << "hipBLASLt MX: GetMatmulPlan failed for scale layout "
                << layout << ": " << plan_or.status();
        continue;
      }
      auto algorithms_or =
          (*plan_or)->GetAlgorithms(max_algorithms, workspace_size);
      if (!algorithms_or.ok()) {
        VLOG(2) << "hipBLASLt MX: GetAlgorithms failed for scale layout "
                << layout << ": " << algorithms_or.status();
        continue;
      }
      for (int64_t i = 0; i < static_cast<int64_t>(algorithms_or->size());
           ++i) {
        auto config = std::make_unique<BackendConfig>();
        auto* gemm_key = config->mutable_gemm();
        gemm_key->set_algorithm(i);
        gemm_key->set_autotune_workspace_size(workspace_size);
        gemm_key->set_mx_scale_layout(layout);
        configs.push_back(std::move(config));
      }
    }
    return configs;
  }
  if (IsCublasLtGroupedMatmul(instr)) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instr.backend_config<GpuBackendConfig>());

    ABSL_ASSIGN_OR_RETURN(BlasLt * blas_lt,
                          se::gpu::BlasLt::Get(stream_executor()));

    int64_t workspace_size;
    const GroupedGemmBackendConfig& grouped_config =
        gpu_config.grouped_gemm_backend_config();
    const GemmBackendConfig& backend_config =
        grouped_config.gemm_backend_config();

    ABSL_ASSIGN_OR_RETURN(
        GroupedGemmConfig grouped_gemm_config,
        GroupedGemmConfig::For(
            &instr,
            target_config().device_description.gpu_compute_capability()));

    ABSL_ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                          AsBlasLtEpilogue(backend_config.epilogue()));

    ABSL_ASSIGN_OR_RETURN(
        BlasLt::MatmulPlanPtr plan,
        blas_lt->GetMatmulPlan(grouped_gemm_config, epilogue));

    const Shape& output_shape = instr.shape();
    if (!output_shape.IsTuple() || output_shape.tuple_shapes().empty()) {
      return Internal(
          "Invalid shape for HipblasLt grouped matmul: output is not a "
          "non-empty tuple.");
    }
    workspace_size = ShapeUtil::ByteSizeOf(output_shape.tuple_shapes().back());

    int max_algorithms = debug_options().xla_gpu_blas_max_algorithms();
    if (max_algorithms <= 0) {
      max_algorithms = GemmConfig::kNumAlgorithms;
    }

    ABSL_ASSIGN_OR_RETURN(std::vector<BlasLt::MatmulAlgorithm> algorithms,
                          plan->GetAlgorithms(max_algorithms, workspace_size));
    int num_algorithms = algorithms.size();
    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.reserve(num_algorithms);
    for (int i = 0; i < num_algorithms; ++i) {
      auto config = std::make_unique<BackendConfig>();
      auto* gemm_key = config->mutable_gemm();
      gemm_key->set_algorithm(i);
      gemm_key->set_autotune_workspace_size(workspace_size);
      configs.push_back(std::move(config));
    }

    return configs;
  }

  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
HipblasLtBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError("Not a supported HipblasLt instruction.");
  }

  auto config = std::make_unique<BackendConfig>();
  auto* gemm_key = config->mutable_gemm();
  gemm_key->set_algorithm(0);
  // Standard GEMMs were already lowered into custom calls with a workspace
  // buffer tuple by GemmRewriter, so we can preserve that allocated workspace
  // size. In contrast, scaled-dot GEMMs enter this pass as a fusion and are
  // only lowered to a custom call in ApplyConfig(), so they do not yet have a
  // workspace buffer in their shape. MX scaled dots on ROCm are currently
  // supported on gfx950, so we use kGFX950Workspace (matching
  // GetSupportedConfigs).
  if (const auto* scaled_dot = GetScaledDotFromFusion(instr)) {
    ABSL_ASSIGN_OR_RETURN(
        auto layouts,
        MxScaleLayoutsToTune(
            *scaled_dot, debug_options(),
            target_config().device_description.gpu_compute_capability()));
    if (layouts.empty()) {
      return absl::InvalidArgumentError(
          "The requested hipBLASLt MX scale layout is not supported");
    }
    // The default config must be one that ApplyConfig can lower.
    if (!IsValidMxScaledDot(scaled_dot)) {
      return absl::InvalidArgumentError(
          "hipBLASLt does not support this MX scaled-dot");
    }
    ABSL_ASSIGN_OR_RETURN(se::gpu::ScaleMode scale_mode,
                          MxScaleMode(layouts.front()));
    ABSL_RETURN_IF_ERROR(
        AnalyzeMxScaledDotFusion(instr, *scaled_dot, scale_mode).status());
    gemm_key->set_mx_scale_layout(layouts.front());
    gemm_key->set_autotune_workspace_size(GemmConfig::kGFX950Workspace);
  } else if (instr.shape().IsTuple() && !instr.shape().tuple_shapes().empty()) {
    gemm_key->set_autotune_workspace_size(
        ShapeUtil::ByteSizeOf(instr.shape().tuple_shapes().back()));
  }
  return config;
}

absl::Status HipblasLtBackend::ApplyConfig(HloInstruction& instr,
                                           const BackendConfig& config) {
  if (!config.has_gemm()) {
    return absl::InvalidArgumentError(
        "Expected GemmKey config for HipblasLtBackend.");
  }
  const AutotuneResult::GemmKey& gemm_key = config.gemm();
  if (gemm_key.mx_scale_layout() !=
          HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR &&
      GetScaledDotFromFusion(instr) == nullptr) {
    return absl::InvalidArgumentError(
        "A pre-swizzled MX scale layout requires a scaled-dot fusion");
  }

  if (gemm_key.unfuse_fp8_output()) {
    Decision can_unfuse = CanUnfuseFp8Output(instr);
    if (can_unfuse.IsForbidden()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "unfuse_fp8_output is set for an instruction whose FP8 output "
          "cannot be unfused: ",
          can_unfuse.Explain()));
    }
    BackendConfig gemm_config = config;
    gemm_config.mutable_gemm()->clear_unfuse_fp8_output();
    ABSL_RETURN_IF_ERROR(ApplyConfig(instr, gemm_config));
    ABSL_ASSIGN_OR_RETURN(HloInstruction * wide, AddWideOutputGemm(&instr));
    return RewriteWithWideOutputGemm(&instr, wide);
  }

  if (IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr)) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instr.backend_config<GpuBackendConfig>());
    GemmBackendConfig& backend_config =
        *gpu_config.mutable_gemm_backend_config();
    backend_config.set_selected_algorithm(gemm_key.algorithm());
    backend_config.set_autotune_workspace_size(
        gemm_key.autotune_workspace_size());
    ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));

    if (instr.shape().IsTuple() && !instr.shape().tuple_shapes().empty()) {
      Shape* workspace_shape = instr.mutable_shape()->mutable_tuple_shapes(
          instr.shape().tuple_shapes().size() - 1);
      if (workspace_shape->element_type() == S8 &&
          workspace_shape->dimensions().size() == 1) {
        workspace_shape->set_dimensions(0, gemm_key.autotune_workspace_size());
        if (HloModule* module = instr.GetModule()) {
          if (module->entry_computation() &&
              module->entry_computation()->root_instruction() == &instr) {
            *module->mutable_entry_computation_layout()
                 ->mutable_result_layout() = ShapeLayout(instr.shape());
          }
        }
      }
    }
    return absl::OkStatus();
  }

  if (const auto* scaled_dot = GetScaledDotFromFusion(instr)) {
    ABSL_ASSIGN_OR_RETURN(se::gpu::ScaleMode scale_mode,
                          MxScaleMode(gemm_key.mx_scale_layout()));
    if (!IsValidMxScaledDot(scaled_dot)) {
      return absl::InvalidArgumentError(
          "hipBLASLt does not support this MX scaled-dot");
    }
    if (scale_mode == se::gpu::ScaleMode::kBlockScaling32x8 &&
        !CanUseHipblasLtScale32x8(
             *scaled_dot,
             target_config().device_description.gpu_compute_capability())
             .IsAllowed()) {
      return absl::InvalidArgumentError(
          "The cached hipBLASLt 32x8 scale configuration is not supported");
    }
    HloComputation* parent = instr.parent();
    ABSL_ASSIGN_OR_RETURN(
        MxFusionStructure structure,
        AnalyzeMxScaledDotFusion(instr, *scaled_dot, scale_mode));

    // The operand paths are elided: each external operand is bitcast to the
    // shape of the scaled-dot operand it feeds.
    absl::InlinedVector<HloInstruction*, 4> operands;
    for (int64_t dot_index = 0; dot_index < 4; ++dot_index) {
      HloInstruction* ext =
          instr.mutable_operand(structure.dot_parameter_numbers[dot_index]);
      const Shape& inner_shape = scaled_dot->operand(dot_index)->shape();
      if (!ShapeUtil::Equal(ext->shape(), inner_shape)) {
        ext = parent->AddInstruction(
            HloInstruction::CreateBitcast(inner_shape, ext));
      }
      operands.push_back(ext);
    }
    if (scale_mode == se::gpu::ScaleMode::kBlockScaling32x8) {
      ABSL_ASSIGN_OR_RETURN(operands[2],
                            SwizzleHipblasLtScale32x8(operands[2]));
      ABSL_ASSIGN_OR_RETURN(operands[3],
                            SwizzleHipblasLtScale32x8(operands[3]));
    }
    const Shape& result_shape = scaled_dot->shape();
    int64_t workspace_size = gemm_key.autotune_workspace_size();
    Shape workspace_shape = ShapeUtil::MakeShape(S8, {workspace_size});
    Shape output_shape =
        ShapeUtil::MakeTupleShape({result_shape, workspace_shape});

    GpuBackendConfig gpu_backend_config;
    GemmBackendConfig& gemm_config =
        *gpu_backend_config.mutable_gemm_backend_config();
    *gemm_config.mutable_dot_dimension_numbers() =
        scaled_dot->dot_dimension_numbers();
    gemm_config.set_alpha_real(1.0);
    gemm_config.set_alpha_imag(0.0);
    gemm_config.set_beta(0.0);
    gemm_config.set_scale_mode(static_cast<int32_t>(scale_mode));
    gemm_config.set_selected_algorithm(gemm_key.algorithm());
    gemm_config.set_autotune_workspace_size(workspace_size);

    HloInstruction* custom_call =
        parent->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, operands, kCublasLtMatmulMxCallTarget));
    ABSL_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));
    HloInstruction* gte = parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(result_shape, custom_call, 0));
    if (structure.epilogue.empty()) {
      return ReplaceMxScaledDotFusion(instr, custom_call, gte);
    }
    if (structure.EpilogueIsBitcast()) {
      // Reinterpreting the GEMM result needs no kernel.
      HloInstruction* result =
          ShapeUtil::Equal(gte->shape(), instr.shape())
              ? gte
              : parent->AddInstruction(
                    HloInstruction::CreateBitcast(instr.shape(), gte));
      return ReplaceMxScaledDotFusion(instr, custom_call, result);
    }
    // The instructions that consume the scaled-dot result run after the GEMM.
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * epilogue,
        CreateMxEpilogueFusion(instr, *scaled_dot, structure, gte));
    // The replacement renames a fusion that replaces a fusion to the old name.
    const std::string epilogue_name = absl::StrCat(instr.name(), "_epilogue");
    HloModule* module = instr.GetModule();
    ABSL_RETURN_IF_ERROR(
        ReplaceMxScaledDotFusion(instr, custom_call, epilogue));
    module->SetAndUniquifyInstrName(epilogue, epilogue_name);
    return absl::OkStatus();
  }
  if (IsCublasLtGroupedMatmul(instr)) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instr.backend_config<GpuBackendConfig>());
    GemmBackendConfig* backend_config =
        gpu_config.mutable_grouped_gemm_backend_config()
            ->mutable_gemm_backend_config();

    backend_config->set_selected_algorithm(gemm_key.algorithm());
    backend_config->set_autotune_workspace_size(
        gemm_key.autotune_workspace_size());
    ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      "ApplyConfig called on unsupported instruction.");
}

}  // namespace gpu
}  // namespace xla
