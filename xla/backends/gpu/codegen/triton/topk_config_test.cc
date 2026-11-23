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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
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
using ::testing::HasSubstr;
using ::testing::Optional;

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
  EXPECT_THAT(CheckTopKKernelLimits(Spec(24576, 64, 6, U64)), IsOk());
}

TEST(TopKKernelLimitsTest, RejectsUnsupportedSpecs) {
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 1024, 8, F16)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported type f16")));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 1024, 8, S32)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported type s32")));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 1024, 129)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("k=129 outside [1, 128]")));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 1024, 0)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("k=0 outside [1, 128]")));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(0, 1024, 8)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("batch=0 is not at least 1")));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(8, 16, 14)),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("k/n=")));
  EXPECT_THAT(CheckTopKKernelLimits(Spec(1, int64_t{1} << 31, 8)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is not below 2^31")));
  EXPECT_THAT(
      CheckTopKKernelLimits(Spec(8, 1024, 8, F32, ComparisonOrder::kWeak)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("unsupported order")));
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

  auto num_chunks = [](const TopKKernelSpec& spec,
                       const se::DeviceDescription& device) {
    std::optional<TopKKernelConfig> config = GetTopKKernelConfig(spec, device);
    return config.has_value() ? std::optional(config->num_chunks)
                              : std::nullopt;
  };
  EXPECT_THAT(num_chunks(Spec(8, 16384, 64), mi350), Optional(1));
  EXPECT_THAT(num_chunks(Spec(8, 32768, 64), mi350), Optional(8));
  EXPECT_THAT(num_chunks(Spec(8, 8192, 64), rx7900), Optional(1));
  EXPECT_THAT(num_chunks(Spec(8, 16384, 64), rx7900), Optional(4));
}

TEST(TopKKernelConfigTest, RowLimitDependsOnTheLdsSize) {
  se::DeviceDescription mi300 = TestGpuDeviceInfo::AMDMI300DeviceInfo();
  ASSERT_EQ(mi300.threads_per_warp(), 64);
  ASSERT_EQ(mi300.shared_memory_per_block(), 64 * 1024);

  // The 64-bit keys of a program take 8 bytes each in LDS.
  std::optional<TopKKernelConfig> config =
      GetTopKKernelConfig(Spec(8, 8192, 64), mi300);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->num_chunks, 1);
  config = GetTopKKernelConfig(Spec(8, 16384, 64), mi300);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 4096);
  EXPECT_EQ(config->num_chunks, 4);
  // The merge of 128 chunks of k = 128 would hold 16384 keys.
  EXPECT_FALSE(GetTopKKernelConfig(Spec(1, 1 << 20, 128), mi300).has_value());
}

TEST(TopKKernelConfigTest, LongRowsAreSplitIntoChunks) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  const TopKKernelSpec spec = Spec(2, 20000, 16);
  std::optional<TopKKernelConfig> config = GetTopKKernelConfig(spec, mi350);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 4096);
  EXPECT_EQ(config->num_chunks, 5);
  EXPECT_EQ(config->rows_per_program, 1);
  EXPECT_EQ(config->num_warps, 8);
  EXPECT_EQ(config->num_programs, 10);
  EXPECT_EQ(GetTopKScratchElements(spec, *config), 2 * 5 * 16);

  TopKKernelSpec merge_spec = GetTopKMergeSpec(spec, *config);
  EXPECT_EQ(merge_spec.input_type, spec.input_type);
  EXPECT_EQ(merge_spec.order, spec.order);
  EXPECT_EQ(merge_spec.batch, 2);
  EXPECT_EQ(merge_spec.n, 80);
  EXPECT_EQ(merge_spec.k, 16);
  std::optional<TopKKernelConfig> merge_config =
      GetTopKMergeConfig(spec, *config, mi350);
  ASSERT_TRUE(merge_config.has_value());
  EXPECT_EQ(merge_config->block_n, 128);
  EXPECT_EQ(merge_config->num_chunks, 1);
  EXPECT_EQ(merge_config->num_programs, 1);
}

TEST(TopKKernelConfigTest, ChunksGrowUntilTheMergeFits) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  // 4096 chunks of 4096 leave 32768 candidates, more than one program takes.
  std::optional<TopKKernelConfig> config =
      GetTopKKernelConfig(Spec(1, 1 << 24, 8), mi350);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 8192);
  EXPECT_EQ(config->num_chunks, 2048);
  // Even the longest chunk leaves too many candidates of k = 128.
  EXPECT_FALSE(GetTopKKernelConfig(Spec(1, 1 << 24, 128), mi350).has_value());
  EXPECT_FALSE(GetTopKKernelConfig(Spec(2, 1 << 24, 128), mi350).has_value());
  // With 32-wide wavefronts a program holds half as many keys, so the merge
  // of k = 128 over 2^20 keys no longer fits.
  EXPECT_TRUE(GetTopKKernelConfig(Spec(1, 1 << 20, 128), mi350).has_value());
  EXPECT_FALSE(GetTopKKernelConfig(Spec(1, 1 << 20, 128),
                                   TestGpuDeviceInfo::AMDRX7900DeviceInfo())
                   .has_value());
}

TEST(TopKKernelConfigTest, SingleKernelHasNoMergeOrScratch) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  const TopKKernelSpec spec = Spec(8, 1000, 32);
  std::optional<TopKKernelConfig> config = GetTopKKernelConfig(spec, mi350);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->num_chunks, 1);
  EXPECT_EQ(GetTopKScratchElements(spec, *config), 0);
  EXPECT_FALSE(GetTopKMergeConfig(spec, *config, mi350).has_value());
}

// The specializer configures the values of a TopK; the emitter configures the
// u64 adapter's TopK of their keys.
TEST(TopKKernelConfigTest, KeysGetTheConfigOfTheirValues) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  for (PrimitiveType type : {F32, BF16}) {
    for (TopKKernelSpec spec :
         {Spec(24576, 64, 6, type), Spec(7, 1000, 128, type),
          Spec(2, 20000, 16, type), Spec(2, 1 << 20, 128, type)}) {
      std::optional<TopKKernelConfig> values = GetTopKKernelConfig(spec, mi350);
      TopKKernelSpec keys_spec = spec;
      keys_spec.input_type = U64;
      std::optional<TopKKernelConfig> keys =
          GetTopKKernelConfig(keys_spec, mi350);
      ASSERT_TRUE(values.has_value()) << spec.ToString();
      ASSERT_TRUE(keys.has_value()) << keys_spec.ToString();
      EXPECT_EQ(keys->ToString(), values->ToString());
      EXPECT_EQ(GetTopKScratchElements(keys_spec, *keys),
                GetTopKScratchElements(spec, *values));
    }
  }
}

TEST(TopKKernelConfigTest, SelectionDependsOnKAndTheRowLength) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  auto selection = [&](const TopKKernelSpec& spec) {
    std::optional<TopKKernelConfig> config = GetTopKKernelConfig(spec, mi350);
    EXPECT_TRUE(config.has_value());
    return config.has_value() ? config->selection : TopKSelection::kRounds;
  };
  EXPECT_EQ(selection(Spec(24576, 64, 6, BF16)), TopKSelection::kRounds);
  EXPECT_EQ(selection(Spec(8, 64, 9)), TopKSelection::kBitonic);
  EXPECT_EQ(selection(Spec(8, 65, 16)), TopKSelection::kRounds);
  EXPECT_EQ(selection(Spec(8, 2048, 17)), TopKSelection::kBitonic);
  EXPECT_EQ(selection(Spec(8, 2049, 32)), TopKSelection::kRounds);
  EXPECT_EQ(selection(Spec(8, 4096, 33)), TopKSelection::kBitonic);
  EXPECT_EQ(selection(Spec(8, 8192, 128)), TopKSelection::kBitonic);
  EXPECT_EQ(selection(Spec(8, 8193, 128)), TopKSelection::kRounds);
}

TEST(TopKKernelConfigTest, ChunkAndMergeChooseTheirSelectionsApart) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  // Chunks of 4096 for k = 9, and a merge of 5 * 9 candidates.
  const TopKKernelSpec spec = Spec(2, 16385, 9);
  std::optional<TopKKernelConfig> config = GetTopKKernelConfig(spec, mi350);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 4096);
  EXPECT_EQ(config->selection, TopKSelection::kRounds);
  std::optional<TopKKernelConfig> merge_config =
      GetTopKMergeConfig(spec, *config, mi350);
  ASSERT_TRUE(merge_config.has_value());
  EXPECT_EQ(merge_config->block_n, 64);
  EXPECT_EQ(merge_config->selection, TopKSelection::kBitonic);
}

TEST(TopKKernelConfigTest, ToStringShowsChunksOnlyWhenSplit) {
  TopKKernelConfig config{4096, 1, 8, 64};
  EXPECT_EQ(config.ToString(),
            "block_n=4096 rows_per_program=1 num_warps=8 num_programs=64");
  config.num_chunks = 8;
  EXPECT_EQ(config.ToString(),
            "block_n=4096 rows_per_program=1 num_warps=8 num_programs=64 "
            "num_chunks=8");
}

TEST(TopKKernelConfigTest, ToStringShowsOnlyTheBitonicSelection) {
  TopKKernelConfig config{4096, 1, 8, 64};
  config.selection = TopKSelection::kBitonic;
  EXPECT_EQ(config.ToString(),
            "block_n=4096 rows_per_program=1 num_warps=8 num_programs=64 "
            "selection=bitonic");
}

TEST(TopKRoutingTest, Mi350TakesTheMoEShapes) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  EXPECT_TRUE(IsTritonTopKFaster(Spec(24576, 64, 6, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(196608, 64, 6, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(24576, 32, 4, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(24576, 8, 2, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(24576, 64, 6, F32), mi350));
}

TEST(TopKRoutingTest, Mi350KeepsPaddedShortRowsWithMidK) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  EXPECT_FALSE(IsTritonTopKFaster(Spec(64, 65, 8), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(64, 64, 8), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(1, 65, 8), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(1024, 65, 8), mi350));
  EXPECT_FALSE(IsTritonTopKFaster(Spec(1024, 257, 16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(1024, 513, 16), mi350));
  EXPECT_FALSE(IsTritonTopKFaster(Spec(1024, 513, 16, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(1024, 33, 16, BF16), mi350));
}

TEST(TopKRoutingTest, Mi350KeepsLargeKOnRowsOf4KTo16K) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  EXPECT_FALSE(IsTritonTopKFaster(Spec(1024, 8192, 64, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(1024, 8192, 64), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(1024, 8192, 32, BF16), mi350));
  EXPECT_FALSE(IsTritonTopKFaster(Spec(8, 16384, 128), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(8, 8192, 128), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(16384, 16384, 128), mi350));
}

TEST(TopKRoutingTest, Mi350KeepsOneLongF32RowWithLargeK) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  EXPECT_FALSE(IsTritonTopKFaster(Spec(1, 1 << 20, 128), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(2, 1 << 20, 128), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(1, 1 << 20, 32), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(1, 1 << 20, 128, BF16), mi350));
}

TEST(TopKRoutingTest, Mi350TakesOtherShapes) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  EXPECT_TRUE(IsTritonTopKFaster(Spec(512, 1024, 128), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(8, 65536, 128), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(8, 32768, 64, BF16), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(4096, 32768, 8), mi350));
  EXPECT_TRUE(IsTritonTopKFaster(Spec(8, 1 << 24, 16), mi350));
}

TEST(TopKRoutingTest, Mi350KeepsOtherTypes) {
  EXPECT_FALSE(IsTritonTopKFaster(Spec(24576, 64, 6, F16),
                                  TestGpuDeviceInfo::AMDMI350DeviceInfo()));
  EXPECT_FALSE(IsTritonTopKFaster(Spec(24576, 64, 6, U64),
                                  TestGpuDeviceInfo::AMDMI350DeviceInfo()));
}

TEST(TopKRoutingTest, OtherDevicesKeepTheDefaultPath) {
  EXPECT_FALSE(IsTritonTopKFaster(Spec(24576, 64, 6, BF16),
                                  TestGpuDeviceInfo::AMDMI300DeviceInfo()));
  EXPECT_FALSE(IsTritonTopKFaster(Spec(24576, 64, 6, BF16),
                                  TestGpuDeviceInfo::H100SXMDeviceInfo()));
}

class TopKCustomCallTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<HloCustomCallInstruction*> ParseTopK(
      absl::string_view backend_config, absl::string_view shape = "f32[8,1000]",
      absl::string_view out = "f32[8,32]",
      absl::string_view idx = "s32[8,32]") {
    std::string config_attr =
        backend_config.empty()
            ? ""
            : absl::StrCat(", backend_config=", backend_config);
    ABSL_ASSIGN_OR_RETURN(module_,
                          ParseAndReturnVerifiedModule(absl::Substitute(
                              R"hlo(
      HloModule m
      ENTRY e {
        p = $0 parameter(0)
        ROOT t = ($1, $2) custom-call(p), custom_call_target="TopK"$3
      })hlo",
                              shape, out, idx, config_attr)));
    return Cast<HloCustomCallInstruction>(
        module_->entry_computation()->root_instruction());
  }

  std::unique_ptr<HloModule> module_;
};

TEST_F(TopKCustomCallTest, ReadsTheSpec) {
  ASSERT_OK_AND_ASSIGN(HloCustomCallInstruction * topk,
                       ParseTopK(R"({is_stable = false, order = "PARTIAL"})",
                                 "bf16[8,1000]", "bf16[8,32]"));
  ASSERT_OK_AND_ASSIGN(TopKKernelSpec spec, GetTopKKernelSpec(*topk));
  EXPECT_EQ(spec.input_type, BF16);
  EXPECT_EQ(spec.order, ComparisonOrder::kPartial);
  EXPECT_EQ(spec.batch, 8);
  EXPECT_EQ(spec.n, 1000);
  EXPECT_EQ(spec.k, 32);
}

TEST_F(TopKCustomCallTest, ReadsTheSpecOfARankOneTopK) {
  ASSERT_OK_AND_ASSIGN(HloCustomCallInstruction * topk,
                       ParseTopK("", "f32[4096]", "f32[16]", "s32[16]"));
  ASSERT_OK_AND_ASSIGN(TopKKernelSpec spec, GetTopKKernelSpec(*topk));
  EXPECT_EQ(spec.batch, 1);
  EXPECT_EQ(spec.n, 4096);
  EXPECT_EQ(spec.k, 16);
  EXPECT_EQ(spec.order, ComparisonOrder::kTotal);
}

TEST_F(TopKCustomCallTest, MarkerIsAppendedAndKeepsTheConfig) {
  ASSERT_OK_AND_ASSIGN(HloCustomCallInstruction * topk,
                       ParseTopK(R"({is_stable = false, order = "PARTIAL"})"));
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
  ASSERT_OK_AND_ASSIGN(HloCustomCallInstruction * topk, ParseTopK(""));
  ASSERT_THAT(MarkAsTritonTopK(*topk), IsOk());
  EXPECT_EQ(topk->raw_backend_config_string(), R"({impl = "triton"})");
  EXPECT_TRUE(hlo_instruction_utils::IsTopKStable(topk));
}

// The u64 adapter clears the stability of its TopK after copying the config.
TEST_F(TopKCustomCallTest, MarkerSurvivesClearingTheStability) {
  ASSERT_OK_AND_ASSIGN(HloCustomCallInstruction * topk,
                       ParseTopK(R"({is_stable = true, order = "TOTAL"})"));
  ASSERT_THAT(MarkAsTritonTopK(*topk), IsOk());
  hlo_instruction_utils::SetTopKStability(topk, false);
  EXPECT_TRUE(IsTritonTopK(*topk));
  EXPECT_FALSE(hlo_instruction_utils::IsTopKStable(topk));
  EXPECT_EQ(hlo_instruction_utils::GetTopKOrder(topk), ComparisonOrder::kTotal);
}

}  // namespace
}  // namespace xla::gpu
