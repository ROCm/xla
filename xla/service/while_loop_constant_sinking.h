/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_WHILE_LOOP_CONSTANT_SINKING_H_
#define XLA_SERVICE_WHILE_LOOP_CONSTANT_SINKING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Sinks while loop invariant values that happen to be constants into the while
// loop body and conditional. This is probably not a win in isolation but may
// unlock further optimizations like constant folding.
//
//   state = (..., const, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(v)
//     state = (..., v, ...)
//   }
//
// =>
//
//   state = (..., const, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(const)
//     state = (..., v, ...)
//   }
//
// Note that it leaves the `v` in place to keep that component of the state
// tuple trivially loop invariant.  WhileLoopSimplifier will later get rid of
// `v`.
//
class WhileLoopConstantSinking : public HloModulePass {
 public:
  // With `skip_unfusible_broadcasts`, a broadcast of a constant is not sunk
  // when no use of it in the body could fuse it: every use, possibly through
  // bitcasts, reshapes, transposes, copies and converts, is an operand that
  // its user updates in place (e.g. operand 0 of a dynamic-update-slice or a
  // scatter) or an operand of a custom call. Sinking would only add a
  // broadcast kernel to every iteration, where a custom call reads the carried
  // buffer as is and an in-place user needs a copy of the same size.
  explicit WhileLoopConstantSinking(bool sink_broadcast_of_constants = false,
                                    bool sink_only_scalar_constants = false,
                                    bool skip_unfusible_broadcasts = false)
      : sink_broadcast_of_constants_(sink_broadcast_of_constants),
        sink_only_scalar_constants_(sink_only_scalar_constants),
        skip_unfusible_broadcasts_(skip_unfusible_broadcasts) {}

  ~WhileLoopConstantSinking() override = default;

  static constexpr absl::string_view kName = "while-loop-constant-sinking";
  absl::string_view name() const override { return kName; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> TrySinkingConstantsIntoWhileLoop(
      HloModule* module, HloInstruction* while_instr);

  const bool sink_broadcast_of_constants_;
  const bool sink_only_scalar_constants_;
  const bool skip_unfusible_broadcasts_;
};
}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_CONSTANT_SINKING_H_
