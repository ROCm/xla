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

#include "xla/backends/gpu/transforms/convert_triton_conv_config.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

class ConvertTritonConvConfigTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<VerifiedHloModule> RunConvertTritonConvConfig(
      absl::string_view hlo, bool expect_change = true) {
    std::unique_ptr<VerifiedHloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    EXPECT_THAT(ConvertTritonConvConfig().Run(module.get()),
                IsOkAndHolds(expect_change));
    EXPECT_OK(verifier().Run(module.get()).status());
    return module;
  }
};

TEST_F(ConvertTritonConvConfigTest, BasicNhwcConv) {
  constexpr absl::string_view hlo = R"(
conv_computation {
  input = f32[1,8,8,32] parameter(0)
  kernel = f32[3,3,32,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(input, kernel),
    window={size=3x3}, dim_labels=b01f_01io->b01f
}

ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  ROOT fusion = f32[1,6,6,16] fusion(p0, p1), kind=kCustom,
    calls=conv_computation, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_conv", "triton_conv_config": {
          "batch_tile":"1", "output_spatial_tiles":["8","8"],
          "output_feature_tile":"16", "kernel_spatial_tiles":["1","1"],
          "input_feature_tile":"32",
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"
        }
      }
    }
})";

  std::unique_ptr<VerifiedHloModule> module = RunConvertTritonConvConfig(hlo);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK: ROOT {{.*}} = f32[1,6,6,16]{{.*}} convolution(
    CHECK-SAME: backend_config={"sizes":["1","1","32"]}
    CHECK: ENTRY
    CHECK: ROOT{{.*}}fusion(
    CHECK-SAME: kind=kCustom
    CHECK-SAME: "block_level_fusion_config"
    CHECK-SAME: "num_ctas":1
    CHECK-SAME: "num_stages":1
    CHECK-SAME: "num_warps":"4"
    CHECK-SAME: "output_tiles":[{"sizes":["1","8","8","16"]}]
    CHECK-SAME: "kind":"__triton"
  )"));

  const HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  EXPECT_FALSE(fusion->backend_config<GpuBackendConfig>()
                   ->fusion_backend_config()
                   .has_triton_conv_config());
}

TEST_F(ConvertTritonConvConfigTest, MapsOutputTileThroughDimNumbers) {
  // The key names its tiles by convolution semantics, so a different
  // dim_labels must permute the output tile accordingly: here the feature tile
  // of 16 lands at index 1 and the spatial tiles at 2 and 3.
  constexpr absl::string_view hlo = R"(
conv_computation {
  input = f32[1,32,8,8] parameter(0)
  kernel = f32[16,32,3,3] parameter(1)
  ROOT conv = f32[1,16,6,6] convolution(input, kernel),
    window={size=3x3}, dim_labels=bf01_oi01->bf01
}

ENTRY entry {
  p0 = f32[1,32,8,8] parameter(0)
  p1 = f32[16,32,3,3] parameter(1)
  ROOT fusion = f32[1,16,6,6] fusion(p0, p1), kind=kCustom,
    calls=conv_computation, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_conv", "triton_conv_config": {
          "batch_tile":"1", "output_spatial_tiles":["8","8"],
          "output_feature_tile":"16", "kernel_spatial_tiles":["1","1"],
          "input_feature_tile":"32",
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"
        }
      }
    }
})";

  std::unique_ptr<VerifiedHloModule> module = RunConvertTritonConvConfig(hlo);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK: "output_tiles":[{"sizes":["1","16","8","8"]}]
  )"));
}

TEST_F(ConvertTritonConvConfigTest, HandlesThreeSpatialDimensions) {
  constexpr absl::string_view hlo = R"(
conv_computation {
  input = f32[1,8,8,8,16] parameter(0)
  kernel = f32[3,3,3,16,8] parameter(1)
  ROOT conv = f32[1,6,6,6,8] convolution(input, kernel),
    window={size=3x3x3}, dim_labels=b012f_012io->b012f
}

ENTRY entry {
  p0 = f32[1,8,8,8,16] parameter(0)
  p1 = f32[3,3,3,16,8] parameter(1)
  ROOT fusion = f32[1,6,6,6,8] fusion(p0, p1), kind=kCustom,
    calls=conv_computation, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_conv", "triton_conv_config": {
          "batch_tile":"1", "output_spatial_tiles":["4","4","4"],
          "output_feature_tile":"8", "kernel_spatial_tiles":["1","1","1"],
          "input_feature_tile":"16",
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
})";

  std::unique_ptr<VerifiedHloModule> module = RunConvertTritonConvConfig(hlo);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK: convolution(
    CHECK-SAME: backend_config={"sizes":["1","1","1","16"]}
    CHECK: "output_tiles":[{"sizes":["1","4","4","4","8"]}]
  )"));
}

TEST_F(ConvertTritonConvConfigTest, ErrorsOnSpatialArityMismatch) {
  constexpr absl::string_view kMismatch = R"(
conv_computation {
  input = f32[1,8,8,32] parameter(0)
  kernel = f32[3,3,32,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(input, kernel),
    window={size=3x3}, dim_labels=b01f_01io->b01f
}

ENTRY entry {
  p0 = f32[1,8,8,32] parameter(0)
  p1 = f32[3,3,32,16] parameter(1)
  ROOT fusion = f32[1,6,6,16] fusion(p0, p1), kind=kCustom,
    calls=conv_computation, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_conv", "triton_conv_config": {
          "batch_tile":"1", "output_spatial_tiles":["8"],
          "output_feature_tile":"16", "kernel_spatial_tiles":["1"],
          "input_feature_tile":"32",
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"
        }
      }
    }
})";

  std::unique_ptr<VerifiedHloModule> module =
      ParseAndReturnVerifiedModule(kMismatch).value();
  EXPECT_THAT(ConvertTritonConvConfig().Run(module.get()),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace xla::gpu
