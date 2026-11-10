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

#include "xla/backends/gpu/codegen/triton/topk_kernel.h"

#include <cstdint>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/triton/topk_config.h"
#include "xla/comparison_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

TopKKernelSpec Spec(int64_t batch, int64_t n, int64_t k,
                    PrimitiveType type = F32,
                    ComparisonOrder order = ComparisonOrder::kTotal) {
  return TopKKernelSpec{type, order, batch, n, k};
}

class TopKKernelTest : public ::testing::Test {
 protected:
  TopKKernelTest() {
    context_.loadDialect<mlir::triton::TritonDialect, mlir::arith::ArithDialect,
                         mlir::scf::SCFDialect>();
  }

  TopKKernelConfig Config(const TopKKernelSpec& spec) {
    std::optional<TopKKernelConfig> config =
        GetTopKKernelConfig(spec, TestGpuDeviceInfo::AMDMI350DeviceInfo());
    CHECK(config.has_value()) << spec.ToString();
    return *config;
  }

  // Builds the kernel `kind` with the configurations of the routing rule and
  // returns its IR as text.
  std::string Build(const TopKKernelSpec& spec,
                    TopKKernelKind kind = TopKKernelKind::kSingle) {
    TopKKernelSpec kernel_spec = spec;
    TopKKernelConfig config = Config(spec);
    if (kind == TopKKernelKind::kMerge) {
      std::optional<TopKKernelConfig> merge_config = GetTopKMergeConfig(
          spec, config, TestGpuDeviceInfo::AMDMI350DeviceInfo());
      CHECK(merge_config.has_value()) << spec.ToString();
      kernel_spec = GetTopKMergeSpec(spec, config);
      config = *merge_config;
    }
    auto module = BuildTopKKernel(kernel_spec, config, kind, "topk", context_);
    CHECK_OK(module.status()) << spec.ToString();
    std::string ir;
    llvm::raw_string_ostream os(ir);
    (*module)->print(os);
    return ir;
  }

  mlir::MLIRContext context_;
};

TEST_F(TopKKernelTest, F32TotalKernel) {
  // block_n = 128, rows_per_program = 32, k rounded up to 8.
  std::string ir = Build(Spec(64, 128, 8));
  EXPECT_THAT(ir,
              HasSubstr("tt.func @topk(%arg0: !tt.ptr<f32> "
                        "{tt.divisibility = 16 : i32}, %arg1: "
                        "!tt.ptr<f32> {tt.divisibility = 16 : i32}, "
                        "%arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32})"));
  EXPECT_THAT(ir, HasSubstr("tt.load"));
  EXPECT_THAT(ir,
              HasSubstr(": tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>"));
  EXPECT_THAT(ir, HasSubstr("scf.for"));
  EXPECT_THAT(ir, HasSubstr("\"tt.reduce\""));
  EXPECT_THAT(ir, HasSubstr("arith.maxui"));
  EXPECT_THAT(ir, HasSubstr("tensor<32x8xi64>"));
  EXPECT_THAT(ir, HasSubstr("tt.store"));
  // TOTAL keys have no -0.0 bit.
  EXPECT_THAT(ir, Not(HasSubstr("2147483647")));
}

TEST_F(TopKKernelTest, Bf16PartialKernel) {
  // block_n = 64, rows_per_program = 64, k = 6 rounded up to 8.
  std::string ir = Build(Spec(24576, 64, 6, BF16, ComparisonOrder::kPartial));
  EXPECT_THAT(ir, HasSubstr("!tt.ptr<bf16>"));
  EXPECT_THAT(ir, HasSubstr("tt.bitcast %"));
  EXPECT_THAT(ir, HasSubstr("tensor<64x64xbf16> -> tensor<64x64xi16>"));
  EXPECT_THAT(ir, HasSubstr("tensor<64x8xi16> -> tensor<64x8xbf16>"));
  // The PARTIAL index is 0x7FFFFFFF - idx.
  EXPECT_THAT(ir, HasSubstr("2147483647"));
}

TEST_F(TopKKernelTest, LargeInputsUseI64Offsets) {
  // 2^21 rows of 2^11 elements need offsets of 32 bits or more.
  std::string ir = Build(Spec(int64_t{1} << 21, 2048, 8, BF16));
  EXPECT_THAT(ir, HasSubstr(": tensor<2x2048x!tt.ptr<bf16>>, "
                            "tensor<2x2048xi64>"));
}

TEST_F(TopKKernelTest, ChunkKernelWritesKeysToScratch) {
  // 5 chunks of 4096 per row, one chunk per program.
  std::string ir = Build(Spec(2, 20000, 16), TopKKernelKind::kChunk);
  EXPECT_THAT(ir,
              HasSubstr("%arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}"));
  EXPECT_THAT(ir, HasSubstr("arith.divsi"));
  EXPECT_THAT(ir, HasSubstr("arith.remsi"));
  EXPECT_THAT(ir,
              HasSubstr(": tensor<1x4096x!tt.ptr<f32>>, tensor<1x4096xi32>"));
  EXPECT_THAT(ir, HasSubstr("tt.store %"));
  EXPECT_THAT(ir, HasSubstr(": tensor<1x16x!tt.ptr<i64>>"));
  // Keys are stored, not decoded values.
  EXPECT_THAT(ir, Not(HasSubstr("tensor<1x16x!tt.ptr<f32>>")));
}

TEST_F(TopKKernelTest, MergeKernelReadsKeysFromScratch) {
  // 5 x 16 candidates per row, padded to 128.
  std::string ir = Build(Spec(2, 20000, 16, BF16), TopKKernelKind::kMerge);
  EXPECT_THAT(ir,
              HasSubstr("%arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}"));
  EXPECT_THAT(ir, HasSubstr(": tensor<2x128x!tt.ptr<i64>>, tensor<2x128xi32>"));
  EXPECT_THAT(ir, Not(HasSubstr("arith.divsi")));
  EXPECT_THAT(ir, HasSubstr("tensor<2x16xi16> -> tensor<2x16xbf16>"));
  EXPECT_THAT(ir, HasSubstr(": tensor<2x16x!tt.ptr<i32>>"));
}

TEST_F(TopKKernelTest, RejectsChunkConfigsThatDoNotCoverTheRow) {
  TopKKernelSpec spec = Spec(2, 20000, 16);
  TopKKernelConfig config = Config(spec);
  ASSERT_EQ(config.num_chunks, 5);
  EXPECT_THAT(
      BuildTopKKernel(spec, config, TopKKernelKind::kChunk, "topk", context_)
          .status(),
      IsOk());
  EXPECT_THAT(
      BuildTopKKernel(spec, config, TopKKernelKind::kSingle, "topk", context_)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
  TopKKernelConfig few_chunks = config;
  few_chunks.num_chunks = 4;
  few_chunks.num_programs = 8;
  EXPECT_THAT(BuildTopKKernel(spec, few_chunks, TopKKernelKind::kChunk, "topk",
                              context_)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  TopKKernelConfig empty_chunk = config;
  empty_chunk.num_chunks = 6;
  empty_chunk.num_programs = 12;
  EXPECT_THAT(BuildTopKKernel(spec, empty_chunk, TopKKernelKind::kChunk, "topk",
                              context_)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  TopKKernelConfig few_programs = config;
  few_programs.num_programs = 9;
  EXPECT_THAT(BuildTopKKernel(spec, few_programs, TopKKernelKind::kChunk,
                              "topk", context_)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TopKKernelTest, RejectsConfigsThatDoNotCoverTheInput) {
  TopKKernelSpec spec = Spec(64, 128, 8);
  TopKKernelConfig config = Config(spec);
  TopKKernelConfig short_rows = config;
  short_rows.block_n = 64;
  EXPECT_THAT(BuildTopKKernel(spec, short_rows, TopKKernelKind::kSingle, "topk",
                              context_)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  TopKKernelConfig few_programs = config;
  few_programs.num_programs = 1;
  EXPECT_THAT(BuildTopKKernel(spec, few_programs, TopKKernelKind::kSingle,
                              "topk", context_)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  TopKKernelConfig odd_rows = config;
  odd_rows.rows_per_program = 3;
  odd_rows.num_programs = 22;
  EXPECT_THAT(
      BuildTopKKernel(spec, odd_rows, TopKKernelKind::kSingle, "topk", context_)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TopKKernelTest, RejectsSpecsOutsideTheLimits) {
  TopKKernelSpec spec = Spec(64, 128, 8, F16);
  TopKKernelConfig config = Config(Spec(64, 128, 8));
  EXPECT_THAT(
      BuildTopKKernel(spec, config, TopKKernelKind::kSingle, "topk", context_)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

// Every configuration the routing rule picks must give a valid kernel.
TEST_F(TopKKernelTest, EveryConfigOfTheRuleBuilds) {
  const se::DeviceDescription mi350 = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  int built = 0;
  for (PrimitiveType type : {F32, BF16}) {
    for (ComparisonOrder order :
         {ComparisonOrder::kTotal, ComparisonOrder::kPartial}) {
      for (int64_t batch : {1, 3, 64, 1000, 24576}) {
        for (int64_t n :
             {2, 16, 64, 100, 256, 1000, 4096, 8192, 20000, 32768, 1 << 20}) {
          for (int64_t k : {1, 2, 6, 8, 16, 100, 128}) {
            TopKKernelSpec spec{type, order, batch, n, k};
            if (!CheckTopKKernelLimits(spec).ok()) {
              continue;
            }
            std::optional<TopKKernelConfig> config =
                GetTopKKernelConfig(spec, mi350);
            if (!config.has_value()) {
              continue;
            }
            if (config->num_chunks == 1) {
              EXPECT_THAT(
                  BuildTopKKernel(spec, *config, TopKKernelKind::kSingle,
                                  "topk", context_)
                      .status(),
                  IsOk())
                  << spec.ToString() << " " << config->ToString();
              ++built;
              continue;
            }
            EXPECT_THAT(BuildTopKKernel(spec, *config, TopKKernelKind::kChunk,
                                        "topk", context_)
                            .status(),
                        IsOk())
                << spec.ToString() << " " << config->ToString();
            std::optional<TopKKernelConfig> merge_config =
                GetTopKMergeConfig(spec, *config, mi350);
            ASSERT_TRUE(merge_config.has_value()) << spec.ToString();
            EXPECT_THAT(
                BuildTopKKernel(GetTopKMergeSpec(spec, *config), *merge_config,
                                TopKKernelKind::kMerge, "topk", context_)
                    .status(),
                IsOk())
                << spec.ToString() << " " << merge_config->ToString();
            ++built;
          }
        }
      }
    }
  }
  EXPECT_GT(built, 0);
}

}  // namespace
}  // namespace xla::gpu
