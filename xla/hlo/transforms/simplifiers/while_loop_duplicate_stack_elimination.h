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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_WHILE_LOOP_DUPLICATE_STACK_ELIMINATION_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_WHILE_LOOP_DUPLICATE_STACK_ELIMINATION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Finds two while-loop elements that every iteration overwrites with a
// dynamic-update-slice of the same value at the same offsets, where the writes
// of all iterations cover the buffer. Neither initial value then reaches the
// output, so the users of one output are made to read the other, and
// WhileLoopSimplifier removes the dead element. Offsets are compared by value
// (EvaluateScalarPerIteration), so the two elements may start from different
// values and use different index expressions, unlike in
// WhileLoopSimplifier's removal of repeated tuple indices.
//
// Typical source: a jax.lax.scan that writes a value into a carried buffer and
// also saves it as an autodiff residual, which the scan stacks again.
class WhileLoopDuplicateStackElimination : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "while-loop-duplicate-stack-elimination";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_WHILE_LOOP_DUPLICATE_STACK_ELIMINATION_H_
