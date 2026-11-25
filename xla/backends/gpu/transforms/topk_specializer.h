/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_TOPK_SPECIALIZER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_TOPK_SPECIALIZER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/transforms/topk_rewriter.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Whether TopkSpecializer will route `topk`, a TopK op, to the Triton TopK
// kernels on `device` with the flags of its module. It makes the same decision
// as TopkSpecializer, on the shape of `topk`; SPMD partitioning may change the
// shape that TopkSpecializer sees.
bool WillUseTritonTopK(const HloInstruction* topk,
                       const se::DeviceDescription& device);

// The TopkDecomposer of TopK ops before SPMD partitioning for the ROCm Triton
// TopK path. The BF16 TopK ops that TopkSpecializer will route to the Triton
// TopK kernels (WillUseTritonTopK) get the generic sort instead of the packed
// BF16 sort, so that TopkRewriter turns them into "TopK" custom calls. A
// deferred TopK that is not routed after partitioning gets the packed BF16 sort
// from a later TopkDecomposer. On other GPUs, or with the flag off, it defers
// nothing.
class TopkDecomposerForTriton : public TopkDecomposer {
 public:
  explicit TopkDecomposerForTriton(
      const se::DeviceDescription& device_description);
  absl::string_view name() const override {
    return "topk-decomposer-for-triton";
  }
};

// This pass transforms eligible TopK CustomCall into a call to be executed by
// runtime/topk.cc, or, on ROCm, by the Triton TopK kernels.
class TopkSpecializer : public HloModulePass {
 public:
  explicit TopkSpecializer(const se::DeviceDescription& device_description)
      : device_description_(device_description) {}
  absl::string_view name() const override { return "topk-specializer"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::DeviceDescription device_description_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_TOPK_SPECIALIZER_H_
