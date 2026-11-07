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

#include <gtest/gtest.h>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class WhileLoopDuplicateStackEliminationExecutionTest : public HloTestBase {
 protected:
  HloModuleConfig ConfigWithFlag(bool enable) {
    HloModuleConfig config = GetModuleConfigForTest();
    DebugOptions debug_options = config.debug_options();
    debug_options.set_xla_gpu_enable_while_loop_duplicate_stack_elimination(
        enable);
    config.set_debug_options(debug_options);
    return config;
  }
};

// A scan that stores each iteration's value both into a carried buffer (at
// layer_indices[i]) and into a residual stack (at i), and a second loop that
// reads the residual stack back in reverse, as a backward pass does. With the
// flag the second loop reads the carried buffer: the results are bitwise equal.
TEST_F(WhileLoopDuplicateStackEliminationExecutionTest, ResidualReadBack) {
  constexpr absl::string_view kHlo = R"(
HloModule scan

body {
  p = (s32[], f32[8,16,32], f32[8,16,32], s32[8], f32[16,32]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  buf = f32[8,16,32] get-tuple-element(p), index=1
  res = f32[8,16,32] get-tuple-element(p), index=2
  layer_indices = s32[8] get-tuple-element(p), index=3
  x = f32[16,32] get-tuple-element(p), index=4
  i_f = f32[] convert(i)
  i_b = f32[16,32] broadcast(i_f), dimensions={}
  w = f32[16,32] multiply(x, i_b)
  zero = s32[] constant(0)
  li_slice = s32[1] dynamic-slice(layer_indices, i), dynamic_slice_sizes={1}
  li = s32[] reshape(li_slice)
  w_buf = f32[1,16,32] reshape(w)
  buf_new = f32[8,16,32] dynamic-update-slice(buf, w_buf, li, zero, zero)
  w_res = f32[1,16,32] reshape(w)
  res_new = f32[8,16,32] dynamic-update-slice(res, w_res, i, zero, zero)
  x_next = f32[16,32] sine(w)
  one = s32[] constant(1)
  next = s32[] add(i, one)
  ROOT t = (s32[], f32[8,16,32], f32[8,16,32], s32[8], f32[16,32]) tuple(next, buf_new, res_new, layer_indices, x_next)
}

cond {
  p = (s32[], f32[8,16,32], f32[8,16,32], s32[8], f32[16,32]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(8)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

back_body {
  p = (s32[], f32[8,16,32], f32[16,32]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  stack = f32[8,16,32] get-tuple-element(p), index=1
  acc = f32[16,32] get-tuple-element(p), index=2
  seven = s32[] constant(7)
  j = s32[] subtract(seven, i)
  zero = s32[] constant(0)
  slice = f32[1,16,32] dynamic-slice(stack, j, zero, zero), dynamic_slice_sizes={1,16,32}
  row = f32[16,32] reshape(slice)
  acc_next = f32[16,32] add(acc, row)
  one = s32[] constant(1)
  next = s32[] add(i, one)
  ROOT t = (s32[], f32[8,16,32], f32[16,32]) tuple(next, stack, acc_next)
}

back_cond {
  p = (s32[], f32[8,16,32], f32[16,32]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(8)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

ENTRY main {
  buf0 = f32[8,16,32] parameter(0)
  x0 = f32[16,32] parameter(1)
  c0 = f32[] constant(0)
  res0 = f32[8,16,32] broadcast(c0), dimensions={}
  iota = s32[8] iota(), iota_dimension=0
  i0 = s32[] constant(0)
  init = (s32[], f32[8,16,32], f32[8,16,32], s32[8], f32[16,32]) tuple(i0, buf0, res0, iota, x0)
  loop = (s32[], f32[8,16,32], f32[8,16,32], s32[8], f32[16,32]) while(init), condition=cond, body=body
  buf_out = f32[8,16,32] get-tuple-element(loop), index=1
  res_out = f32[8,16,32] get-tuple-element(loop), index=2
  acc0 = f32[16,32] broadcast(c0), dimensions={}
  back_init = (s32[], f32[8,16,32], f32[16,32]) tuple(i0, res_out, acc0)
  back = (s32[], f32[8,16,32], f32[16,32]) while(back_init), condition=back_cond, body=back_body
  acc_out = f32[16,32] get-tuple-element(back), index=2
  ROOT out = (f32[8,16,32], f32[16,32]) tuple(buf_out, acc_out)
}
)";
  EXPECT_TRUE(RunAndCompareTwoModules(
      kHlo, kHlo, ConfigWithFlag(/*enable=*/false),
      ConfigWithFlag(/*enable=*/true), ErrorSpec(/*aabs=*/0.0, /*arel=*/0.0)));
}

}  // namespace
}  // namespace xla::gpu
