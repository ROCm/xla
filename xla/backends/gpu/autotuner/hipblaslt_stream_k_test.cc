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

#include "xla/backends/gpu/autotuner/hipblaslt_stream_k.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

// Solution names of hipBLASLt 1.4 on gfx950 (getSolutionNameFromAlgo): 14828
// and 20152 are data-parallel, 12110 is a stream-K hybrid.
constexpr char kSolution14828[] =
    "Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_CMS_"
    "SN_GSU1_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_WGM16_WG32_8_1";
constexpr char kSolution12110[] =
    "Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x288x64_MI16x16x1_CMS_"
    "SN_GSU0_SVW8_SK5_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_WGM0_WG32_8_1";

TEST(HipblasLtStreamKTest, ParsesStreamKMode) {
  EXPECT_EQ(HipblasLtStreamKMode(kSolution14828), 0);
  EXPECT_EQ(HipblasLtStreamKMode(kSolution12110), 5);
  EXPECT_EQ(HipblasLtStreamKMode("MT128x128x64_SK3_GSU1"), 3);
  EXPECT_EQ(HipblasLtStreamKMode("MT128x128x64_SK12"), 12);
}

TEST(HipblasLtStreamKTest, IgnoresOtherStreamKSettings) {
  EXPECT_EQ(HipblasLtStreamKMode("MT128x128x64_SKFTR1_SKWS0_SKXCCM8_GSU1"),
            std::nullopt);
  EXPECT_EQ(HipblasLtStreamKMode("MT128x128x64_SK_GSU1"), std::nullopt);
  EXPECT_EQ(HipblasLtStreamKMode("MT128x128x64_SKn1"), std::nullopt);
  EXPECT_EQ(HipblasLtStreamKMode(""), std::nullopt);
  EXPECT_EQ(HipblasLtStreamKMode("ASK5_SK0"), 0);
}

TEST(HipblasLtStreamKTest, DropsStreamKSolutions) {
  std::vector<std::string> names = {kSolution12110, kSolution14828,
                                    "MT64x64x64_SK3", "unnamed",
                                    "MT256x256x64_SK0"};
  EXPECT_THAT(SelectNonStreamKSolutions(names), ElementsAre(1, 3, 4));
}

TEST(HipblasLtStreamKTest, KeepsAllWhenEverySolutionIsStreamK) {
  std::vector<std::string> names = {kSolution12110, "MT64x64x64_SK3"};
  EXPECT_THAT(SelectNonStreamKSolutions(names), ElementsAre(0, 1));
  EXPECT_THAT(SelectNonStreamKSolutions({}), IsEmpty());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
