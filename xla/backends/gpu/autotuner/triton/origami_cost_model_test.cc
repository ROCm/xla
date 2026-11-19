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

#include "xla/backends/gpu/autotuner/triton/origami_cost_model.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::UnorderedElementsAreArray;

se::DeviceDescription MI300X() {
  se::DeviceDescription d;
  d.set_rocm_compute_capability("gfx942:sramecc+:xnack-");
  d.set_core_count(304);
  d.set_shared_memory_per_core(65536);
  d.set_registers_per_core_limit(65536 * 4);
  d.set_l2_cache_size(4 * 1024 * 1024);
  d.set_clock_rate_ghz(2.1);
  return d;
}

se::DeviceDescription MI350X() {
  se::DeviceDescription d = MI300X();
  d.set_rocm_compute_capability("gfx950");
  d.set_core_count(256);
  d.set_shared_memory_per_core(163840);
  return d;
}

TritonGemmConfig Config(int m, int n, int k, int stages = 2, int warps = 4) {
  TritonGemmConfig c;
  c.block_m = m;
  c.block_n = n;
  c.block_k = k;
  c.num_stages = stages;
  c.num_warps = warps;
  c.num_ctas = 1;
  return c;
}

class OrigamiDotCostModelTest : public HloHardwareIndependentTestBase {
 protected:
  const HloDotInstruction* ParseDot(absl::string_view hlo) {
    module_ = ParseAndReturnVerifiedModule(hlo).value();
    for (HloInstruction* instr : module_->entry_computation()->instructions()) {
      if (instr->opcode() == HloOpcode::kDot) {
        return Cast<HloDotInstruction>(instr);
      }
    }
    return nullptr;
  }

  std::unique_ptr<VerifiedHloModule> module_;
};

constexpr absl::string_view kBf16Gemm = R"(
HloModule m
ENTRY e {
  a = bf16[4096,4096]{1,0} parameter(0)
  b = bf16[4096,4096]{1,0} parameter(1)
  ROOT d = bf16[4096,4096]{1,0} dot(a, b),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

TEST_F(OrigamiDotCostModelTest, CreateSucceedsWithoutGpu) {
  TF_ASSERT_OK_AND_ASSIGN(auto model, OrigamiDotCostModel::Create(MI300X()));
  EXPECT_NE(model, nullptr);
}

TEST_F(OrigamiDotCostModelTest, CreateRejectsUnknownArch) {
  se::DeviceDescription d = MI300X();
  d.set_rocm_compute_capability("gfx1234");
  EXPECT_EQ(OrigamiDotCostModel::Create(d).status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST_F(OrigamiDotCostModelTest, CreateRejectsIncompleteDeviceDescription) {
  se::DeviceDescription d = MI300X();
  d.set_clock_rate_ghz(0);
  EXPECT_EQ(OrigamiDotCostModel::Create(d).status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST_F(OrigamiDotCostModelTest, EstimatesAreFiniteAndOrdered) {
  TF_ASSERT_OK_AND_ASSIGN(auto model, OrigamiDotCostModel::Create(MI300X()));
  std::vector<TritonGemmConfig> configs = {
      Config(256, 256, 64), Config(128, 128, 64), Config(32, 32, 32)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto estimates, model->EstimateRunTimes(ParseDot(kBf16Gemm), configs));
  ASSERT_THAT(estimates, Not(IsEmpty()));
  for (const auto& [config, duration] : estimates) {
    EXPECT_GT(duration, absl::ZeroDuration());
  }
  // A large square GEMM should prefer the larger tile.
  if (estimates.contains(configs[0]) && estimates.contains(configs[2])) {
    EXPECT_LT(estimates.at(configs[0]), estimates.at(configs[2]));
  }
}

TEST_F(OrigamiDotCostModelTest, ConfigsSharingATileShapeTie) {
  TF_ASSERT_OK_AND_ASSIGN(auto model, OrigamiDotCostModel::Create(MI300X()));
  std::vector<TritonGemmConfig> configs = {
      Config(128, 128, 64, /*stages=*/2, /*warps=*/4),
      Config(128, 128, 64, /*stages=*/3, /*warps=*/8)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto estimates, model->EstimateRunTimes(ParseDot(kBf16Gemm), configs));
  ASSERT_EQ(estimates.size(), 2);
  EXPECT_EQ(estimates.at(configs[0]), estimates.at(configs[1]));
}

TEST_F(OrigamiDotCostModelTest, SelectTopTileShapesKeepsAllVariants) {
  TF_ASSERT_OK_AND_ASSIGN(auto model, OrigamiDotCostModel::Create(MI300X()));
  std::vector<TritonGemmConfig> configs = {
      Config(128, 128, 64, 2, 4), Config(128, 128, 64, 3, 8),
      Config(64, 64, 64, 2, 4),   Config(64, 64, 64, 3, 8),
      Config(32, 32, 32, 2, 4),   Config(32, 32, 32, 3, 8),
  };
  TF_ASSERT_OK_AND_ASSIGN(
      auto kept, model->SelectTopTileShapes(ParseDot(kBf16Gemm), configs, 2));

  EXPECT_EQ(kept.size(), 4);
  for (const TritonGemmConfig& c : kept) {
    EXPECT_EQ(c.block_m, c.block_n);
  }
  // Both variants of each surviving tile shape must be present.
  for (int i = 0; i + 1 < static_cast<int>(kept.size()); i += 2) {
    EXPECT_EQ(kept[i].block_m, kept[i + 1].block_m);
    EXPECT_EQ(kept[i].block_k, kept[i + 1].block_k);
    EXPECT_NE(kept[i].num_stages, kept[i + 1].num_stages);
  }
}

TEST_F(OrigamiDotCostModelTest, SelectTopTileShapesPreservesInputOrder) {
  TF_ASSERT_OK_AND_ASSIGN(auto model, OrigamiDotCostModel::Create(MI300X()));
  std::vector<TritonGemmConfig> configs = {
      Config(32, 32, 32), Config(256, 256, 64), Config(128, 128, 64)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto kept, model->SelectTopTileShapes(ParseDot(kBf16Gemm), configs, 3));
  ASSERT_EQ(kept.size(), 3);
  EXPECT_EQ(kept[0].block_m, 32);
  EXPECT_EQ(kept[1].block_m, 256);
  EXPECT_EQ(kept[2].block_m, 128);
}

TEST_F(OrigamiDotCostModelTest, DropsConfigsExceedingTritonSharedMemory) {
  TF_ASSERT_OK_AND_ASSIGN(auto model, OrigamiDotCostModel::Create(MI300X()));
  std::vector<TritonGemmConfig> configs = {Config(256, 256, 256, /*stages=*/4)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto estimates, model->EstimateRunTimes(ParseDot(kBf16Gemm), configs));
  EXPECT_THAT(estimates, IsEmpty());
}

TEST_F(OrigamiDotCostModelTest, Gfx950AcceptsLargerTilesThanGfx942) {
  // 98KB under gfx942's model, 148KB under gfx950's padded async-copy layout.
  std::vector<TritonGemmConfig> configs = {Config(256, 128, 64, /*stages=*/3)};
  const HloDotInstruction* dot = ParseDot(kBf16Gemm);

  TF_ASSERT_OK_AND_ASSIGN(auto mi300, OrigamiDotCostModel::Create(MI300X()));
  TF_ASSERT_OK_AND_ASSIGN(auto mi350, OrigamiDotCostModel::Create(MI350X()));
  TF_ASSERT_OK_AND_ASSIGN(auto on_300, mi300->EstimateRunTimes(dot, configs));
  TF_ASSERT_OK_AND_ASSIGN(auto on_350, mi350->EstimateRunTimes(dot, configs));

  EXPECT_THAT(on_300, IsEmpty());
  EXPECT_THAT(on_350, Not(IsEmpty()));
}

TEST_F(OrigamiDotCostModelTest, IsSupportedAcceptsPlainGemm) {
  EXPECT_OK(IsSupportedByOrigami(ParseDot(kBf16Gemm), MI300X()));
}

TEST_F(OrigamiDotCostModelTest, IsSupportedRejectsMultipleContractingDims) {
  constexpr absl::string_view kHlo = R"(
HloModule m
ENTRY e {
  a = bf16[16,32,64]{2,1,0} parameter(0)
  b = bf16[32,64,16]{2,1,0} parameter(1)
  ROOT d = bf16[16,16]{1,0} dot(a, b),
      lhs_contracting_dims={1,2}, rhs_contracting_dims={0,1}
})";
  EXPECT_EQ(IsSupportedByOrigami(ParseDot(kHlo), MI300X()).code(),
            absl::StatusCode::kUnimplemented);
}

TEST_F(OrigamiDotCostModelTest, IsSupportedRejectsUnmodelledDtype) {
  constexpr absl::string_view kHlo = R"(
HloModule m
ENTRY e {
  a = c64[64,64]{1,0} parameter(0)
  b = c64[64,64]{1,0} parameter(1)
  ROOT d = c64[64,64]{1,0} dot(a, b),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  EXPECT_EQ(IsSupportedByOrigami(ParseDot(kHlo), MI300X()).code(),
            absl::StatusCode::kUnimplemented);
}

TEST_F(OrigamiDotCostModelTest, BatchedDotIsModelled) {
  constexpr absl::string_view kHlo = R"(
HloModule m
ENTRY e {
  a = bf16[8,1024,1024]{2,1,0} parameter(0)
  b = bf16[8,1024,1024]{2,1,0} parameter(1)
  ROOT d = bf16[8,1024,1024]{2,1,0} dot(a, b),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto model, OrigamiDotCostModel::Create(MI300X()));
  std::vector<TritonGemmConfig> configs = {Config(128, 128, 64)};
  TF_ASSERT_OK_AND_ASSIGN(auto estimates,
                          model->EstimateRunTimes(ParseDot(kHlo), configs));
  EXPECT_THAT(estimates, Not(IsEmpty()));
}

}  // namespace
}  // namespace xla::gpu
