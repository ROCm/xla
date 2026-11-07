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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_WHILE_LOOP_DUPLICATE_STACK_ELIMINATION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_WHILE_LOOP_DUPLICATE_STACK_ELIMINATION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Finds while loops that carry two stacked buffers which every iteration
// overwrites, with a dynamic-update-slice, with the same value at the same
// offsets, where the writes of all iterations together cover each buffer, so
// that neither initial value reaches the loop's output. Both loop outputs then
// hold the same value: the users of one are made to read the other, and the
// first becomes a dead loop element (WhileLoopSimplifier removes it, with its
// initialization and its per-iteration write).
//
// Typical source: a jax.lax.scan whose body writes a value into a carried
// buffer (buf = buf.at[i].set(v)) and also uses v in a way that makes autodiff
// save v as a residual for the backward pass, which the scan stacks as a
// separate output. For ZeRO-1 with weights gathered layer by layer in the
// first microbatch's forward, the gathered expert weights are written both
// into the carried compute copy and into the residual stack.
//
// The offsets are compared by evaluating the index expressions for every
// iteration (constants, the induction variable, integer arithmetic and
// elements of loop-invariant constant or iota arrays), so loops need a trivial
// induction variable and a known trip count.
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

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_WHILE_LOOP_DUPLICATE_STACK_ELIMINATION_H_
