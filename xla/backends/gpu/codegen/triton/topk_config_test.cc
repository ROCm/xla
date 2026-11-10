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

#include "xla/backends/gpu/codegen/triton/topk_config.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TopKKernelSpec Spec(int64_t batch, int64_t n, int64_t k,
                    PrimitiveType type = F32,
                    ComparisonOrder order = ComparisonOrder::kTotal) {
  return TopKKernelSpec{type, order, batch, n, k};
}

TEST(TopKKernelLimitsTest, AcceptsSupportedSpecs) {
  EXPECT_THAT(CheckTopKKernelLimits(Spec(24576, 64, 6, BF16)), IsOk());
  EXPECT_THAT(CheckTopKKernelLimits(Spec(1, 8, 1)), IsOk());
  EXPECT_THAT(CheckTopKKernelLimits(Spec(7, 1000, 128)), IsOk());
  EXPECT_THAT(
      CheckTopKKernelLimits(Spec(8, 1024, 32, F32, ComparisonOrder::kPartial)),
      IsOk());
}

TEST(TopKKernelLimitsTest, RejectsUnsupportedSpecs) {
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 1024, 8, F16)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 1024, 8, S32)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 1024, 129)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 1024, 0)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 16, 14)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(1, int64_t{1} << 31, 8)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      CheckTopKKernelLimits(Spec(8, 1024, 8, F32, ComparisonOrder::kWeak)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TopKKernelConfigTest, ShortRowsShareAProgram) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  std::optional<TopKKernelConfig> config =
      GetTopKKernelConfig(Spec(24576, 64, 6, BF16), mi350);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 64);
  EXPECT_EQ(config->rows_per_program, 64);
  EXPECT_EQ(config->num_warps, 8);
  EXPECT_EQ(config->num_programs, 384);
}

TEST(TopKKernelConfigTest, PadsRowsToAPowerOfTwo) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  std::optional<TopKKernelConfig> config =
      GetTopKKernelConfig(Spec(7, 1000, 128), mi350);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 1024);
  EXPECT_EQ(config->rows_per_program, 4);
  EXPECT_EQ(config->num_programs, 2);
}

TEST(TopKKernelConfigTest, DoesNotGiveASmallBatchMoreRowsThanItHas) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  std::optional<TopKKernelConfig> config =
      GetTopKKernelConfig(Spec(3, 16, 2), mi350);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->rows_per_program, 4);
  EXPECT_EQ(config->num_warps, 1);
  EXPECT_EQ(config->num_programs, 1);
}

TEST(TopKKernelConfigTest, RowLimitDependsOnTheWaveSize) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  se::DeviceDescription rx7900 = TestGpuDeviceInfo::AMDRX7900DeviceInfo();
  ASSERT_EQ(mi350.threads_per_warp(), 64);
  ASSERT_EQ(rx7900.threads_per_warp(), 32);

  EXPECT_TRUE(GetTopKKernelConfig(Spec(8, 16384, 64), mi350).has_value());
  EXPECT_FALSE(GetTopKKernelConfig(Spec(8, 32768, 64), mi350).has_value());
  EXPECT_TRUE(GetTopKKernelConfig(Spec(8, 8192, 64), rx7900).has_value());
  EXPECT_FALSE(GetTopKKernelConfig(Spec(8, 16384, 64), rx7900).has_value());
}

TEST(TopKSpeedTableTest, Mi350TakesTheMoEShapes) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  EXPECT_TRUE(IsTritonTopKFaster(Spec(24576, 64, 6, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(196608, 64, 6, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(24576, 32, 4, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(24576, 8, 2, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(24576, 64, 6, F32), mi350));
}

TEST(TopKSpeedTableTest, Mi350KeepsSlowerShapes) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  // Large k: the k rounds cost more than today's sort.
  EXPECT_FALSE(IsTritonTopKFaster(Spec(512, 1024, 128), mi350));
  // Batches beyond the measured ones.
  EXPECT_FALSE(IsTritonTopKFaster(Spec(1 << 20, 64, 6, BF16), mi350));
  // Short rows that are not a power of two pad to the next one.
  EXPECT_FALSE(IsTritonTopKFaster(Spec(262144, 33, 8), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(262144, 64, 8), mi350));
}

TEST(TopKSpeedTableTest, OtherDevicesHaveNoTable) {
  EXPECT_FALSE(IsTritonTopKFaster(Spec(24576, 64, 6, BF16),
                                  TestGpuDeviceInfo::AMDMI300DeviceInfo()));
  EXPECT_FALSE(IsTritonTopKFaster(Spec(24576, 64, 6, BF16),
                                  TestGpuDeviceInfo::H100SXMDeviceInfo()));
}

class TopKCustomCallTest : public HloHardwareIndependentTestBase {
 protected:
  HloCustomCallInstruction* ParseTopK(absl::string_view backend_config,
                                      absl::string_view shape = "f32[8,1000]",
                                      absl::string_view out = "f32[8,32]",
                                      absl::string_view idx = "s32[8,32]") {
    std::string config_attr =
        backend_config.empty()
            ? ""
            : absl::StrCat(", backend_config=", backend_config);
    auto module = ParseAndReturnVerifiedModule(absl::Substitute(
        R"(
      HloModule m
      ENTRY e {
        p = $0 parameter(0)
        ROOT t = ($1, $2) custom-call(p), custom_call_target="TopK"$3
      })",
        shape, out, idx, config_attr));
    CHECK_OK(module.status());
    module_ = *std::move(module);
    return Cast<HloCustomCallInstruction>(
        module_->entry_computation()->root_instruction());
  }

  std::unique_ptr<HloModule> module_;
};

TEST_F(TopKCustomCallTest, ReadsTheSpec) {
  HloCustomCallInstruction* topk =
      ParseTopK(R"({is_stable = false, order = "PARTIAL"})", "bf16[8,1000]",
                "bf16[8,32]");
  ASSERT_OK_AND_ASSIGN(TopKKernelSpec spec, GetTopKKernelSpec(*topk));
  EXPECT_EQ(spec.input_type, BF16);
  EXPECT_EQ(spec.order, ComparisonOrder::kPartial);
  EXPECT_EQ(spec.batch, 8);
  EXPECT_EQ(spec.n, 1000);
  EXPECT_EQ(spec.k, 32);
}

TEST_F(TopKCustomCallTest, ReadsTheSpecOfARankOneTopK) {
  HloCustomCallInstruction* topk =
      ParseTopK("", "f32[4096]", "f32[16]", "s32[16]");
  ASSERT_OK_AND_ASSIGN(TopKKernelSpec spec, GetTopKKernelSpec(*topk));
  EXPECT_EQ(spec.batch, 1);
  EXPECT_EQ(spec.n, 4096);
  EXPECT_EQ(spec.k, 16);
  EXPECT_EQ(spec.order, ComparisonOrder::kTotal);
}

TEST_F(TopKCustomCallTest, MarkerIsAppendedAndKeepsTheConfig) {
  HloCustomCallInstruction* topk =
      ParseTopK(R"({is_stable = false, order = "PARTIAL"})");
  EXPECT_FALSE(IsTritonTopK(*topk));
  ASSERT_THAT(MarkAsTritonTopK(*topk), IsOk());
  EXPECT_TRUE(IsTritonTopK(*topk));
  EXPECT_EQ(topk->raw_backend_config_string(),
            R"({is_stable = false, order = "PARTIAL", impl = "triton"})");
  EXPECT_FALSE(hlo_instruction_utils::IsTopKStable(topk));
  EXPECT_EQ(hlo_instruction_utils::GetTopKOrder(topk),
            ComparisonOrder::kPartial);

  ASSERT_THAT(MarkAsTritonTopK(*topk), IsOk());
  EXPECT_EQ(topk->raw_backend_config_string(),
            R"({is_stable = false, order = "PARTIAL", impl = "triton"})");
}

TEST_F(TopKCustomCallTest, MarkerOnAnEmptyConfig) {
  HloCustomCallInstruction* topk = ParseTopK("");
  ASSERT_THAT(MarkAsTritonTopK(*topk), IsOk());
  EXPECT_EQ(topk->raw_backend_config_string(), R"({impl = "triton"})");
  EXPECT_TRUE(hlo_instruction_utils::IsTopKStable(topk));
}

}  // namespace
}  // namespace xla::gpu
