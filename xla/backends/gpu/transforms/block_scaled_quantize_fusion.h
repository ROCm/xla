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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_BLOCK_SCALED_QUANTIZE_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_BLOCK_SCALED_QUANTIZE_FUSION_H_

#include <functional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/decision.h"

namespace xla::gpu {

// Wraps each block-scaled quantization (max |x| over blocks of 16 or 32, scale,
// divide, convert to FP8 or FP4) with its absorbed producers and the other
// orientation of its source in a "__block_scaled_quantize" kCustom fusion,
// rooted at data outputs, then scale outputs, then predicates, row first.
class BlockScaledQuantizeFusion : public HloModulePass {
 public:
  // Decides whether a fusion built by the pass can be emitted.
  using CanEmit = std::function<Decision(const HloFusionInstruction&)>;

  explicit BlockScaledQuantizeFusion(CanEmit can_emit)
      : can_emit_(std::move(can_emit)) {}

  absl::string_view name() const override {
    return "block-scaled-quantize-fusion";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  CanEmit can_emit_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_BLOCK_SCALED_QUANTIZE_FUSION_H_
