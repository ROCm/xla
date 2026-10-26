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

#include "xla/backends/gpu/transforms/triton_conv_eligibility.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/conv_utils.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

Decision IsEligibleForTritonConv(
    const HloConvolutionInstruction& conv,
    const se::GpuComputeCapability& compute_capability) {
  if (conv.convolution_kind() != CONVOLUTION_KIND_UNSET) {
    return Decision::Forbid("Convolution kind is already assigned.");
  }

  if (!LooksLikeForwardConvolution(&conv)) {
    return Decision::Forbid(
        "Only forward convolutions are supported by Triton; this looks like a "
        "backward-filter convolution.");
  }
  return IsTritonSupportedInstruction(conv, compute_capability);
}

absl::StatusOr<bool> TritonConvEligibility::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->config()
           .debug_options()
           .xla_gpu_experimental_enable_triton_conv_fusion()) {
    return false;
  }
  XLA_VLOG_LINES(
      2, "TritonConvEligibility::Run(), before:\n" + module->ToString());

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kConvolution) {
        continue;
      }
      HloConvolutionInstruction* conv =
          Cast<HloConvolutionInstruction>(instruction);
      Decision decision = IsEligibleForTritonConv(*conv, compute_capability_);
      if (decision.IsForbidden()) {
        VLOG(2) << "Not routing " << conv->name()
                << " to Triton: " << decision.Explain();
        continue;
      }
      VLOG(1) << "Claiming " << conv->name()
              << " for the Triton convolution emitter.";
      conv->set_convolution_kind(CONVOLUTION_KIND_FPROP);
      changed = true;
    }
  }
  XLA_VLOG_LINES(2,
                 "TritonConvEligibility::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
