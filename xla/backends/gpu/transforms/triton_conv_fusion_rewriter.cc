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

#include "xla/backends/gpu/transforms/triton_conv_fusion_rewriter.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla::gpu {
namespace {

// Builds a single-instruction fusion around `conv`. Parameters mirror the
// convolution's operands one for one; nothing else is pulled in.
absl::StatusOr<HloInstruction*> WrapConvInFusion(HloInstruction* conv) {
  HloComputation::Builder builder(absl::StrCat(conv->name(), "_computation"));
  std::vector<HloInstruction*> fusion_operands;
  std::vector<HloInstruction*> fused_operands;
  fusion_operands.reserve(conv->operand_count());
  fused_operands.reserve(conv->operand_count());

  for (int64_t i = 0; i < conv->operand_count(); ++i) {
    HloInstruction* operand = conv->mutable_operand(i);
    fusion_operands.push_back(operand);
    fused_operands.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            i, operand->shape(), absl::StrCat("p", i))));
  }

  HloInstruction* fused_conv = builder.AddInstruction(
      conv->CloneWithNewOperands(conv->shape(), fused_operands));

  HloComputation* computation =
      conv->GetModule()->AddComputationAndUnifyNamesAndIds(
          builder.Build(fused_conv), /*is_entry=*/false);

  HloInstruction* fusion =
      conv->parent()->AddInstruction(HloInstruction::CreateFusion(
          conv->shape(), HloInstruction::FusionKind::kCustom, fusion_operands,
          computation));

  GpuBackendConfig gpu_config;
  gpu_config.mutable_fusion_backend_config()->set_kind(kTritonConvFusionKind);
  ABSL_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));
  // triton_conv_config is deliberately left unset: the config assigner fills
  // it.
  return fusion;
}

}  // namespace

absl::StatusOr<bool> TritonConvFusionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->config()
           .debug_options()
           .xla_gpu_experimental_enable_triton_conv_fusion()) {
    return false;
  }

  XLA_VLOG_LINES(
      2, "TritonConvFusionRewriter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    // Two-phase: collect, then rewrite. Mutating a computation while iterating
    // its instruction list is undefined.
    std::vector<HloInstruction*> claimed;
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kConvolution) {
        continue;
      }
      if (Cast<HloConvolutionInstruction>(instruction)->convolution_kind() !=
          CONVOLUTION_KIND_FPROP) {
        continue;
      }
      claimed.push_back(instruction);
    }

    for (HloInstruction* conv : claimed) {
      ABSL_ASSIGN_OR_RETURN(HloInstruction * fusion, WrapConvInFusion(conv));
      VLOG(1) << "Wrapped " << conv->name() << " into " << fusion->name();
      ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(conv, fusion));
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "TritonConvFusionRewriter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
