/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/autotuner_pass.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class AutotunerPassTest : public HloTestBase {};

TEST_F(AutotunerPassTest, TritonFusionIsAutotuned) {
  const char kCublasCustomCallHlo[] = R"(
    HloModule module, entry_computation_layout={(f32[100,100]{1,0}, f32[100,100]{1,0})->f32[100,100]{1,0}}
  
    ENTRY %main (arg0: f32[100,100], arg1: f32[100,100]) -> f32[100,100] {
      %arg0 = f32[100,100]{1,0} parameter(0)
      %arg1 = f32[100,100]{1,0} parameter(1)
      %custom-call.1 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%arg0, %arg1), 
      custom_call_target="__cublas$gemm", 
      backend_config={
        "gemm_backend_config":{
          "dot_dimension_numbers":
            {
              "lhs_contracting_dimensions":["1"],
              "rhs_contracting_dimensions":["0"],
              "lhs_batch_dimensions":[],
              "rhs_batch_dimensions":[]
          }
        }
      }
      ROOT %get-tuple-element = f32[100,100]{1,0} get-tuple-element(%custom-call.1), index=0
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  AutotunerPass pass(backend().default_stream_executor(), &thread_pool);
  EXPECT_THAT(pass.Run(module.get(), /*execution_threads=*/{}),
              IsOkAndHolds(true));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
