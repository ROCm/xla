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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_STORE_FORWARDING_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_STORE_FORWARDING_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Prepares values that a loop stores into a stacked buffer (a
// dynamic-update-slice whose result is the loop state, e.g. the residuals
// jax.lax.scan saves for the backward pass) for dynamic slice fusion, so that
// the value is written into its slice of the stacked buffer instead of into a
// separate buffer that the dynamic-update-slice then copies.
//
// Dynamic slice fusion only redirects a hero's result into a slice when the
// dynamic-update-slice is the result's only user. This pass makes that true:
//
//  * The value is a hero (a custom call the fusion can wrap) with other users:
//    they read the stored slice back, dynamic-slice(stack', i), instead.
//
//  * The value is a hero that updates its operand in place
//    (output_to_operand_aliasing), and the operand is the only use of another
//    hero's result. The operand is stored first and the in-place hero runs on
//    the stored slice:
//
//      stack1 = dynamic-update-slice(stack, operand, i)
//      value' = hero(dynamic-slice(stack1, i))     // in place on the slice
//      stack' = dynamic-update-slice(stack1, value', i)
//
//    Other users read dynamic-slice(stack', i).
//
//  * The value is a fusible instruction whose other users are all heroes: the
//    heroes read dynamic-slice(stack', i), and the producer, now with a single
//    user, fuses into the dynamic-update-slice instead of being computed twice.
//
// Every rewrite preserves values: dynamic-slice(stack', i) is the stored value
// (both clamp `i` the same way). Requires DynamicSliceAnnotator to have run;
// the new slices get the dynamic-update-slice's DynamicSliceConfig.
class DynamicSliceStoreForwarding : public HloModulePass {
 public:
  // Selects the custom calls that dynamic slice fusion wraps (its hero
  // predicate).
  using HeroPredicate = absl::AnyInvocable<bool(const HloInstruction*) const>;

  explicit DynamicSliceStoreForwarding(HeroPredicate is_hero)
      : is_hero_(std::move(is_hero)) {}

  absl::string_view name() const override {
    return "dynamic-slice-store-forwarding";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  HeroPredicate is_hero_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_STORE_FORWARDING_H_
