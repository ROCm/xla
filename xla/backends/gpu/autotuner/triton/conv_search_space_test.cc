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

#include "xla/backends/gpu/autotuner/triton/conv_search_space.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/conv_utils.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::testing::IsEmpty;
using ::testing::Not;

class ConvSearchSpaceTest : public HloHardwareIndependentTestBase {
 protected:
  const se::DeviceDescription device_description_{
      TestGpuDeviceInfo::AMDMI300DeviceInfo()};

  absl::StatusOr<const HloConvolutionInstruction*> ParseConv(
      absl::string_view hlo) {
    ABSL_ASSIGN_OR_RETURN(module_, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* conv = hlo_query::GetFirstInstructionWithOpcode(
        *module_->entry_computation(), HloOpcode::kConvolution);
    if (conv == nullptr) {
      return absl::InvalidArgumentError("HLO contains no convolution.");
    }
    return Cast<HloConvolutionInstruction>(conv);
  }

  std::unique_ptr<VerifiedHloModule> module_;
};

constexpr absl::string_view kNhwcConv = R"(
ENTRY entry {
  p0 = f32[20,32,32,120] parameter(0)
  p1 = f32[5,5,120,64] parameter(1)
  ROOT conv = f32[20,28,28,64] convolution(p0, p1),
    window={size=5x5}, dim_labels=b01f_01io->b01f
})";

constexpr absl::string_view kNchwConv = R"(
ENTRY entry {
  p0 = f32[20,120,32,32] parameter(0)
  p1 = f32[5,5,120,64] parameter(1)
  ROOT conv = f32[20,64,28,28] convolution(p0, p1),
    window={size=5x5}, dim_labels=bf01_01io->bf01
})";

TEST_F(ConvSearchSpaceTest, GeneratesNonEmptyConfigListNhwc) {
  ASSERT_OK_AND_ASSIGN(const HloConvolutionInstruction* conv,
                       ParseConv(kNhwcConv));
  TritonConvFusionSearchSpace space(device_description_, conv);
  EXPECT_THAT(space.GenerateConfigs(), Not(IsEmpty()));
  EXPECT_THAT(space.ToString(), ::testing::HasSubstr("channels_minor:1"));
}

TEST_F(ConvSearchSpaceTest, GeneratesNonEmptyConfigListNchw) {
  ASSERT_OK_AND_ASSIGN(const HloConvolutionInstruction* conv,
                       ParseConv(kNchwConv));
  TritonConvFusionSearchSpace space(device_description_, conv);
  EXPECT_THAT(space.GenerateConfigs(), Not(IsEmpty()));
  EXPECT_THAT(space.ToString(), ::testing::HasSubstr("channels_minor:0"));
}

TEST_F(ConvSearchSpaceTest, EveryConfigRespectsSharedMemoryBound) {
  ASSERT_OK_AND_ASSIGN(const HloConvolutionInstruction* conv,
                       ParseConv(kNhwcConv));
  TritonConvFusionSearchSpace space(device_description_, conv);
  const int64_t limit = device_description_.shared_memory_per_block_optin();
  for (const TritonConvConfig& config : space.GenerateConfigs()) {
    EXPECT_LE(EstimateSharedMemoryBytes(config, /*element_bytes=*/4,
                                        /*input_is_channels_minor=*/true),
              limit)
        << config.ToString();
  }
}

TEST_F(ConvSearchSpaceTest, IncludesTheFloorConfig) {
  ASSERT_OK_AND_ASSIGN(const HloConvolutionInstruction* conv,
                       ParseConv(kNhwcConv));
  TritonConvFusionSearchSpace space(device_description_, conv);
  const TritonConvConfig floor = GetFloorConvConfig(conv);
  std::vector<TritonConvConfig> configs = space.GenerateConfigs();
  EXPECT_THAT(configs, ::testing::Contains(floor));
  EXPECT_LE(EstimateSharedMemoryBytes(floor, /*element_bytes=*/4,
                                      /*input_is_channels_minor=*/true),
            device_description_.shared_memory_per_block_optin() / 2)
      << "The floor config must fit with margin, not just barely.";
}

TEST_F(ConvSearchSpaceTest, SharedMemoryEstimateScalesWithStages) {
  TritonConvConfig one_stage;
  one_stage.batch_tile = 1;
  one_stage.output_spatial_tiles = {8, 8};
  one_stage.output_feature_tile = 16;
  one_stage.kernel_spatial_tiles = {1, 1};
  one_stage.input_feature_tile = 32;
  one_stage.num_stages = 1;
  one_stage.num_warps = 4;
  one_stage.num_ctas = 1;

  TritonConvConfig two_stages = one_stage;
  two_stages.num_stages = 2;

  EXPECT_EQ(EstimateSharedMemoryBytes(two_stages, 4,
                                      /*input_is_channels_minor=*/true),
            2 * EstimateSharedMemoryBytes(one_stage, 4,
                                          /*input_is_channels_minor=*/true));
}

TEST_F(ConvSearchSpaceTest, EstimateMatchesMeasuredHardwareNumbers) {
  struct Case {
    int64_t batch_tile;
    int64_t spatial_tile;
    int64_t output_feature_tile;
    int64_t input_feature_tile;
    int64_t measured_bytes;
  };

  constexpr Case kCases[] = {
      {1, 16, 16, 32, 32768},   // M=256
      {2, 16, 16, 32, 65536},   // M=512, K=32 exactly at the limit
      {4, 16, 16, 16, 65536},   // M=1024, K=16, exactly at the limit
      {4, 16, 16, 32, 131072},  // M=1024, K=32, exceeds the limit
  };

  for (const Case& c : kCases) {
    TritonConvConfig config;
    config.batch_tile = c.batch_tile;
    config.output_spatial_tiles = {c.spatial_tile, c.spatial_tile};
    config.output_feature_tile = c.output_feature_tile;
    config.kernel_spatial_tiles = {1, 1};
    config.input_feature_tile = c.input_feature_tile;
    config.num_stages = 1;
    config.num_warps = 1;
    config.num_ctas = 1;

    const int64_t predicted =
        EstimateSharedMemoryBytes(config, 4, /*input_is_channels_minor=*/true);
    EXPECT_EQ(predicted, c.measured_bytes);
  }
}

}  // namespace
}  // namespace xla::gpu
