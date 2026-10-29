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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_TRITON_CONV_ELIGIBILITY_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_TRITON_CONV_ELIGIBILITY_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/decision.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

Decision IsEligibleForTritonConv(
    const HloConvolutionInstruction& conv,
    const se::GpuComputeCapability& compute_capability);

// Claims eligible forward convs for the Triton emitter by setting their
// conv kind.
class TritonConvEligibility : public HloModulePass {
 public:
  explicit TritonConvEligibility(
      const se::GpuComputeCapability& compute_capability)
      : compute_capability_(compute_capability) {}

  absl::string_view name() const override { return "triton-conv-eligibility"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::GpuComputeCapability compute_capability_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_TRITON_CONV_ELIGIBILITY_H_
