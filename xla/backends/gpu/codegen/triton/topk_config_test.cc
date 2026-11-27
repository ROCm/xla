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
  EXPECT_THAT(num_chunks(Spec(8, 32768, 64), mi350), Optional(16));
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

TEST(TopKKernelConfigTest, Mi350ChunksOfLargeKLeaveAbout4096Candidates) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  auto expect_chunks = [&](const TopKKernelSpec& spec, int64_t block_n,
                           int64_t merge_block_n) {
    SCOPED_TRACE(spec.ToString());
    std::optional<TopKKernelConfig> config = GetTopKKernelConfig(spec, mi350);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->block_n, block_n);
    EXPECT_EQ(config->rows_per_program, 1);
    std::optional<TopKKernelConfig> merge_config =
        GetTopKMergeConfig(spec, *config, mi350);
    ASSERT_TRUE(merge_config.has_value());
    EXPECT_EQ(merge_config->block_n, merge_block_n);
  };
  expect_chunks(Spec(16, 1 << 17, 128), 4096, 4096);
  expect_chunks(Spec(16, 1 << 18, 128), 8192, 4096);
  // Chunks of 2048 at least, one per program.
  expect_chunks(Spec(8, 1 << 15, 128), 2048, 2048);
  expect_chunks(Spec(8, 1 << 17, 32), 2048, 2048);
  // Chunks longer than 64 x block_k would select by k rounds, so the merge
  // grows.
  expect_chunks(Spec(16, 1 << 20, 128), 8192, 16384);
  expect_chunks(Spec(16, 1 << 20, 64), 4096, 16384);
  // With few rows, the merge keeps the bitonic selection instead, or the
  // chunks start from 4096 if the merge would not fill a program.
  expect_chunks(Spec(1, 1 << 20, 128), 16384, 8192);
  expect_chunks(Spec(8, 1 << 20, 64), 16384, 4096);
  expect_chunks(Spec(1, 1 << 19, 40), 4096, 8192);
}

TEST(TopKKernelConfigTest, Mi350ChunksOfSmallKFillTheDevice) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  ASSERT_EQ(mi350.core_count(), 256);
  auto chunk = [&](const TopKKernelSpec& spec) {
    std::optional<TopKKernelConfig> config = GetTopKKernelConfig(spec, mi350);
    EXPECT_TRUE(config.has_value());
    return config.has_value() ? config->block_n : 0;
  };
  // 512 chunk programs of 16384, and exactly one per CU.
  EXPECT_EQ(chunk(Spec(64, 1 << 17, 8)), 16384);
  EXPECT_EQ(chunk(Spec(256, 1 << 15, 16)), 16384);
  EXPECT_EQ(chunk(Spec(128, 1 << 15, 16)), 16384);
  // Too few rows for longer chunks than 4096.
  EXPECT_EQ(chunk(Spec(16, 1 << 15, 8)), 4096);
  EXPECT_EQ(chunk(Spec(1, 1 << 20, 16)), 4096);
  // Rows that are not a multiple of the longer chunks.
  EXPECT_EQ(chunk(Spec(64, 262145, 16)), 4096);
  // Below k = 8.
  EXPECT_EQ(chunk(Spec(64, 1 << 17, 4)), 4096);
}

TEST(TopKKernelConfigTest, Mi350ChunksGrowWhenTheMergeDoesNotFit) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  // 1024 chunks of 4096 would leave 65536 candidates of k = 64.
  std::optional<TopKKernelConfig> config =
      GetTopKKernelConfig(Spec(4, 1 << 22, 64), mi350);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 16384);
  EXPECT_EQ(config->num_chunks, 256);
}

TEST(TopKKernelConfigTest, OtherDevicesTakeChunksFrom4096) {
  std::optional<TopKKernelConfig> config = GetTopKKernelConfig(
      Spec(64, 1 << 17, 8), TestGpuDeviceInfo::AMDMI300DeviceInfo());
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 4096);
  EXPECT_EQ(config->num_chunks, 32);
}

TEST(TopKKernelConfigTest, ChunksGrowUntilTheMergeFits) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  // 2048 chunks of 4096 leave 16384 candidates, more than an MI300 program
  // takes.
  std::optional<TopKKernelConfig> config = GetTopKKernelConfig(
      Spec(1, 1 << 23, 8), TestGpuDeviceInfo::AMDMI300DeviceInfo());
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 8192);
  EXPECT_EQ(config->num_chunks, 1024);
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

TEST(TopKKernelConfigTest, ScratchKeepsTheChunksItWasSizedFor) {
  se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  for (const TopKKernelSpec& spec :
       {Spec(8, 16384, 64), Spec(8, 32768, 64), Spec(64, 262145, 16),
        Spec(1, 1 << 20, 128), Spec(4, 1 << 22, 64)}) {
    SCOPED_TRACE(spec.ToString());
    std::optional<TopKKernelConfig> config = GetTopKKernelConfig(spec, mi350);
    ASSERT_TRUE(config.has_value());
    std::optional<TopKKernelConfig> from_scratch =
        GetTopKKernelConfigForScratch(
            spec, GetTopKScratchElements(spec, *config), mi350);
    ASSERT_TRUE(from_scratch.has_value());
    EXPECT_EQ(from_scratch->block_n, config->block_n);
    EXPECT_EQ(from_scratch->num_chunks, config->num_chunks);
    EXPECT_EQ(from_scratch->rows_per_program, config->rows_per_program);
    EXPECT_EQ(from_scratch->num_warps, config->num_warps);
    EXPECT_EQ(from_scratch->selection, config->selection);
  }
  // A module compiled for 8 chunks of 4096 keeps them.
  const TopKKernelSpec spec = Spec(8, 32768, 64);
  std::optional<TopKKernelConfig> config =
      GetTopKKernelConfigForScratch(spec, 8 * 8 * 64, mi350);
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->block_n, 4096);
  EXPECT_EQ(config->num_chunks, 8);
  // No chunk length splits a row of 32768 into 3 chunks.
  EXPECT_FALSE(
      GetTopKKernelConfigForScratch(spec, 8 * 3 * 64, mi350).has_value());
  EXPECT_FALSE(
      GetTopKKernelConfigForScratch(spec, 8 * 8 * 64 + 1, mi350).has_value());
  // A row of 32768 needs chunks.
  EXPECT_FALSE(GetTopKKernelConfigForScratch(spec, 0, mi350).has_value());
  // Chunks hold at least k keys: 16 chunks of 64 for k = 64, but not 32 of 32.
  const TopKKernelSpec short_row = Spec(1, 1024, 64);
  EXPECT_TRUE(
      GetTopKKernelConfigForScratch(short_row, 16 * 64, mi350).has_value());
  EXPECT_FALSE(
      GetTopKKernelConfigForScratch(short_row, 32 * 64, mi350).has_value());
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

TEST_F(TopKCustomCallTest, ReadsTheSpecOfATopKOp) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"hlo(
      HloModule m
      ENTRY e {
        p = bf16[4,6144,64] parameter(0)
        ROOT t = (bf16[4,6144,6], s32[4,6144,6]) topk(p), k=6, largest=true
      })hlo"));
  ASSERT_OK_AND_ASSIGN(TopKKernelSpec spec,
                       GetTopKKernelSpec(*Cast<HloTopKInstruction>(
                           module->entry_computation()->root_instruction())));
  EXPECT_EQ(spec.input_type, BF16);
  EXPECT_EQ(spec.order, ComparisonOrder::kTotal);
  EXPECT_EQ(spec.batch, 24576);
  EXPECT_EQ(spec.n, 64);
  EXPECT_EQ(spec.k, 6);
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

}  // namespace
}  // namespace xla::gpu
