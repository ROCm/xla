/* Copyright 2021 The OpenXLA Authors.

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

#include <string>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla::gpu {
namespace {

class KernelLaunchTest : public HloInterpreterReferenceMixin<HloTestBase> {};

TEST_F(KernelLaunchTest, Basic) {
  const char* hlo_text = R"(
    HloModule Test1

    add_F32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY Test1 {
      a = f32[2, 2]{1,0} parameter(0)
      b = f32[2, 2]{1,0} parameter(1)
      ROOT r1 = f32[2, 2]{1,0} map(a, b), dimensions={0, 1}, to_apply=add_F32
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(KernelLaunchTest, KernelWithConstants) {
  const char* hlo_text = R"(
    HloModule Test2

    add_F32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY Test2 {
      a = f32[2, 2]{1,0} constant({{1, 2}, {3, 4}})
      b = f32[2, 2]{1,0} constant({{5, 6}, {7, 8}})
      ROOT r1 = f32[2, 2]{1,0} map(a, b), dimensions={0, 1}, to_apply=add_F32
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(KernelLaunchTest, ValidPrefixExtentBoundsLoopFusion) {
  for (int live_rows : {-3, 0, 1, 3, 7, 8, 11}) {
    std::string hlo_text = R"(
      HloModule ValidPrefixExtent,
        input_output_alias={ {}: (0, {}, must-alias) }

      fused_computation {
        x = f32[8,4] parameter(0)
        n = s32[] parameter(1)
        rows = s32[8,4] iota(), iota_dimension=0
        broadcast_n = s32[8,4] broadcast(n), dimensions={}
        is_live = pred[8,4] compare(rows, broadcast_n), direction=LT
        squared = f32[8,4] multiply(x, x)
        ROOT result = f32[8,4] select(is_live, squared, x)
      }

      ENTRY main {
        x = f32[8,4] parameter(0)
        n = s32[] constant(LIVE_ROWS)
        ROOT fusion = f32[8,4] fusion(x, n), kind=kLoop,
          calls=fused_computation,
          backend_config={"fusion_backend_config":{"kind":"kLoop"},"valid_prefix_extent":{"operand_index":"1","dimension":"0"}}
      })";
    hlo_text.replace(hlo_text.find("LIVE_ROWS"),
                     std::string("LIVE_ROWS").size(),
                     std::to_string(live_rows));

    EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, ErrorSpec{1e-5, 1e-5}));
  }
}

TEST_F(KernelLaunchTest, ValidPrefixExtentDoesNotReuseUnboundedKernel) {
  const char* hlo_text = R"(
    HloModule ValidPrefixExtentKernelReuse

    fused_computation {
      x = f32[8,4] parameter(0)
      n = s32[] parameter(1)
      rows = s32[8,4] iota(), iota_dimension=0
      broadcast_n = s32[8,4] broadcast(n), dimensions={}
      is_live = pred[8,4] compare(rows, broadcast_n), direction=LT
      squared = f32[8,4] multiply(x, x)
      ROOT result = f32[8,4] select(is_live, squared, x)
    }

    unbounded_computation {
      x = f32[8,4] parameter(0)
      n = s32[] parameter(1)
      rows = s32[8,4] iota(), iota_dimension=0
      broadcast_n = s32[8,4] broadcast(n), dimensions={}
      is_live = pred[8,4] compare(rows, broadcast_n), direction=LT
      squared = f32[8,4] multiply(x, x)
      ROOT result = f32[8,4] select(is_live, squared, x)
    }

    ENTRY main {
      x = f32[8,4] parameter(0)
      n = s32[] constant(3)
      bounded = f32[8,4] fusion(x, n), kind=kLoop,
        calls=fused_computation,
        backend_config={"fusion_backend_config":{"kind":"kLoop"},"valid_prefix_extent":{"operand_index":"1","dimension":"0"}}
      unbounded = f32[8,4] fusion(x, n), kind=kLoop,
        calls=unbounded_computation,
        backend_config={"fusion_backend_config":{"kind":"kLoop"}}
      bounded_prefix = f32[3,4] slice(bounded), slice={[0:3], [0:4]}
      ROOT result = (f32[3,4], f32[8,4]) tuple(bounded_prefix, unbounded)
    }
  )";

  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla::gpu
