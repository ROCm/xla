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

#include "xla/backends/gpu/autotuner/hipblaslt.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/amdgpu_compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using HipblasLtBackendConfig = AutotuneResult::GemmKey;

const char kHipblasLtCustomCallHlo[] = R"(
HloModule module

ENTRY main {
  p0 = f32[100,100] parameter(0)
  p1 = f32[100,100] parameter(1)
  %custom-call.1 = (f32[100,100]{1,0}, s8[4194304]{0}) custom-call(p0, p1),
    custom_call_target="__cublas$lt$matmul",
    backend_config={
      "gemm_backend_config":{
        "selected_algorithm":"0",
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{
          "operand_precision":["DEFAULT","DEFAULT"],
          "algorithm":"ALG_UNSET"
        },
        "epilogue":"DEFAULT",
        "lhs_stride":"10000",
        "rhs_stride":"10000",
        "grad_x":false,
        "grad_y":false,
        "damax_output":false
      }
    }
  ROOT %get-tuple-element = f32[100,100]{1,0} get-tuple-element(%custom-call.1), index=0
})";

const char kUnsupportedHlo[] = R"(
  HloModule module
  
  computation {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    convert0 = f32[1024,1024]{1,0} convert(p0)
    p1 = s8[1024,1024]{1,0} parameter(1)
    convert1 = f32[1024,1024]{1,0} convert(p1)
    ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
  
  ENTRY main {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    p1 = s8[1024,1024]{1,0} parameter(1)
    ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

class HipblasLtBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  AMDGPUCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  HipblasLtBackend backend_;

  HipblasLtBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        backend_(stream_executor_, &debug_options_, &compiler_,
                 &target_config_) {}

  HipblasLtBackendConfig ExpectedDefaultAlgorithm() {
    auto config = AutotuneResult::GemmKey();
    config.set_algorithm(se::blas::kDefaultAlgorithm);
    return config;
  }
};

TEST_F(HipblasLtBackendTest, CanCreateHipblasLtBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(HipblasLtBackendTest, GetSupportedConfigs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *hlo_module->entry_computation()->root_instruction()->operand(0));
  EXPECT_THAT(configs,
              absl_testing::IsOkAndHolds(testing::SizeIs(testing::Gt(0))));
}

TEST_F(HipblasLtBackendTest, GetSupportedConfigsReturnsErrorForDeviceless) {
  HipblasLtBackend backend_without_stream_executor(
      /*stream_executor=*/nullptr, &debug_options_, &compiler_,
      &target_config_);
  auto hlo_module_or = ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo);
  ASSERT_THAT(hlo_module_or, absl_testing::IsOk());
  std::unique_ptr<HloModule> hlo_module = std::move(hlo_module_or).value();
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_without_stream_executor.GetSupportedConfigs(
          *hlo_module->entry_computation()->root_instruction()->operand(0));
  EXPECT_THAT(configs,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HipblasLtBackendTest,
       GetSupportedConfigsReturnsEmptyVectorForNonHipblasLtCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kUnsupportedHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *hlo_module->entry_computation()->root_instruction());
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(testing::SizeIs(0)));
}

TEST_F(HipblasLtBackendTest, GetDefaultConfig) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()->operand(0)));
  ASSERT_THAT(config, absl_testing::IsOk());
  ASSERT_TRUE((*config)->has_gemm());
  EXPECT_THAT((*config)->gemm().algorithm(), 0);
  EXPECT_NE((*config)->gemm().autotune_workspace_size(), 0);
}

TEST_F(HipblasLtBackendTest, GetDefaultConfigFailsWithoutAHipblasLtCustomCall) {
  std::string hlo = R"(
    HloModule module

    ENTRY main {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[1024,1024]{1,0} parameter(1)
      ROOT dot = f32[1024,1024]{1,0} dot(p0, p1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(config,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HipblasLtBackendTest, ApplyConfig) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));
  HipblasLtBackendConfig config;
  config.set_algorithm(2);
  config.set_autotune_workspace_size(42);
  BackendConfig backend_config;
  *backend_config.mutable_gemm() = config;
  TF_EXPECT_OK(backend_.ApplyConfig(*hlo_module->entry_computation()
                                         ->root_instruction()
                                         ->mutable_operands()
                                         .at(0),
                                    backend_config));
  EXPECT_THAT(RunFileCheck(hlo_module->ToString(),
                           R"(CHECK: (f32[100,100]{1,0}, s8[42]{0}) custom-call
                              CHECK: "selected_algorithm":"2")"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(HipblasLtBackendTest, Compile) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction()->operand(0))));
  absl::StatusOr<std::unique_ptr<Executable>> executable = backend_.Compile(
      *(module->entry_computation()->root_instruction()->operand(0)), *config);
  EXPECT_THAT(executable, absl_testing::IsOk());
}

const char kScaledDotFp8FusionHlo[] = R"(
HloModule ScaledDotFusion

%fusion_dot (p0: f8e4m3fn[32,256], p1: f8e4m3fn[16,256], p2: f8e8m0fnu[32,8], p3: f8e8m0fnu[16,8]) -> f32[32,16] {
  %p0 = f8e4m3fn[32,256]{1,0} parameter(0)
  %p1 = f8e4m3fn[16,256]{1,0} parameter(1)
  %p2 = f8e8m0fnu[32,8]{1,0} parameter(2)
  %p3 = f8e8m0fnu[16,8]{1,0} parameter(3)
  ROOT %scaled_dot = f32[32,16]{1,0} scaled-dot(%p0, %p1, %p2, %p3), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY %main (lhs: f8e4m3fn[32,256], rhs: f8e4m3fn[16,256], lhs_scale: f8e8m0fnu[32,8], rhs_scale: f8e8m0fnu[16,8]) -> f32[32,16] {
  %lhs = f8e4m3fn[32,256]{1,0} parameter(0)
  %rhs = f8e4m3fn[16,256]{1,0} parameter(1)
  %lhs_scale = f8e8m0fnu[32,8]{1,0} parameter(2)
  %rhs_scale = f8e8m0fnu[16,8]{1,0} parameter(3)
  ROOT %fusion = f32[32,16]{1,0} fusion(%lhs, %rhs, %lhs_scale, %rhs_scale), kind=kCustom, calls=%fusion_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

const char kScaledDotFp4FusionHlo[] = R"(
HloModule ScaledDotFp4Fusion

%fusion_dot (p0: f4e2m1fn[32,256], p1: f4e2m1fn[16,256], p2: f8e8m0fnu[32,8], p3: f8e8m0fnu[16,8]) -> f32[32,16] {
  %p0 = f4e2m1fn[32,256]{1,0} parameter(0)
  %p1 = f4e2m1fn[16,256]{1,0} parameter(1)
  %p2 = f8e8m0fnu[32,8]{1,0} parameter(2)
  %p3 = f8e8m0fnu[16,8]{1,0} parameter(3)
  ROOT %scaled_dot = f32[32,16]{1,0} scaled-dot(%p0, %p1, %p2, %p3), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY %main (lhs: f4e2m1fn[32,256], rhs: f4e2m1fn[16,256], lhs_scale: f8e8m0fnu[32,8], rhs_scale: f8e8m0fnu[16,8]) -> f32[32,16] {
  %lhs = f4e2m1fn[32,256]{1,0} parameter(0)
  %rhs = f4e2m1fn[16,256]{1,0} parameter(1)
  %lhs_scale = f8e8m0fnu[32,8]{1,0} parameter(2)
  %rhs_scale = f8e8m0fnu[16,8]{1,0} parameter(3)
  ROOT %fusion = f32[32,16]{1,0} fusion(%lhs, %rhs, %lhs_scale, %rhs_scale), kind=kCustom, calls=%fusion_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

class HipblasLtScaledDotTest : public HipblasLtBackendTest {
 protected:
  void SetUp() override {
    const auto& gpu_cc =
        target_config_.device_description.gpu_compute_capability();
    const auto* rocm_cc = gpu_cc.rocm_compute_capability();
    if (rocm_cc == nullptr || !rocm_cc->has_mx_type_support()) {
      GTEST_SKIP() << "Scaled dot requires MX type support (gfx950+).";
    }
  }

  static constexpr const char* kScaledDotHlos[] = {kScaledDotFp8FusionHlo,
                                                   kScaledDotFp4FusionHlo};
};

TEST_F(HipblasLtScaledDotTest, GetSupportedConfigs) {
  for (const char* hlo : kScaledDotHlos) {
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    auto configs = backend_.GetSupportedConfigs(*fusion);
    EXPECT_THAT(configs,
                absl_testing::IsOkAndHolds(testing::SizeIs(testing::Gt(0))));
  }
}

TEST_F(HipblasLtScaledDotTest, ApplyConfig) {
  for (const char* hlo : kScaledDotHlos) {
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    TF_ASSERT_OK_AND_ASSIGN(auto config, backend_.GetDefaultConfig(*fusion));

    TF_EXPECT_OK(backend_.ApplyConfig(*fusion, *config));

    EXPECT_THAT(RunFileCheck(module->ToString(),
                             R"(CHECK: custom-call
                        CHECK-SAME: custom_call_target="__cublas$lt$matmul$mx"
                        CHECK: "scale_mode":2)"),
                absl_testing::IsOkAndHolds(true));
  }
}

TEST_F(HipblasLtScaledDotTest, Compile) {
  for (const char* hlo : kScaledDotHlos) {
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    TF_ASSERT_OK_AND_ASSIGN(auto config, backend_.GetDefaultConfig(*fusion));

    auto executable = backend_.Compile(*fusion, *config);
    EXPECT_THAT(executable, absl_testing::IsOk());
  }
}

namespace {

std::string PreswizzledScaledDotFusionHlo(bool fp4, int n = 96) {
  return absl::Substitute(R"(
HloModule PreswizzledScaledDotFusion
fusion_dot {
  a = $0[64,512]{1,0$1} parameter(0)
  b = $0[$2,512]{1,0$1} parameter(1)
  sa = f8e8m0fnu[64,16]{1,0} parameter(2)
  sb = f8e8m0fnu[$2,16]{1,0} parameter(3)
  ROOT dot = f32[64,$2]{1,0} scaled-dot(a, b, sa, sb),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
ENTRY main {
  a = $0[64,512]{1,0$1} parameter(0)
  b = $0[$2,512]{1,0$1} parameter(1)
  sa = f8e8m0fnu[64,16]{1,0} parameter(2)
  sb = f8e8m0fnu[$2,16]{1,0} parameter(3)
  ROOT fusion = f32[64,$2]{1,0} fusion(a,b,sa,sb), kind=kCustom,
      calls=fusion_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})",
                          fp4 ? "f4e2m1fn" : "f8e4m3fn", fp4 ? ":E(4)" : "", n);
}

class HipblasLtPreswizzledScaleTest : public HipblasLtBackendTest {
 protected:
  void SetUp() override {
    const auto* rocm =
        target_config_.device_description.gpu_compute_capability()
            .rocm_compute_capability();
    if (rocm == nullptr || !rocm->gfx9_mi350()) {
      GTEST_SKIP() << "Pre-swizzled scale tests require gfx950";
    }
  }
};

class HipblasLtPreswizzledScaleTypeTest
    : public HipblasLtPreswizzledScaleTest,
      public ::testing::WithParamInterface<bool> {};

TEST_P(HipblasLtPreswizzledScaleTypeTest, EnumeratesBothScaleLayouts) {
  debug_options_.set_xla_gpu_blas_max_algorithms(4);
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_AUTO);
  auto module_or =
      ParseAndReturnVerifiedModule(PreswizzledScaledDotFusionHlo(GetParam()));
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  auto configs_or = backend_.GetSupportedConfigs(
      *module->entry_computation()->root_instruction());
  ASSERT_THAT(configs_or, absl_testing::IsOk());
  auto configs = std::move(configs_or).value();
  bool has_linear = false;
  bool has_32x8 = false;
  for (const auto& config : configs) {
    has_linear |= config->gemm().mx_scale_layout() ==
                  HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR;
    has_32x8 |= config->gemm().mx_scale_layout() ==
                HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8;
  }
  EXPECT_TRUE(has_linear);
  EXPECT_TRUE(has_32x8);
  auto default_or =
      backend_.GetDefaultConfig(*module->entry_computation()->root_instruction());
  ASSERT_THAT(default_or, absl_testing::IsOk());
  EXPECT_EQ((*default_or)->gemm().mx_scale_layout(),
            HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR);
}

TEST_P(HipblasLtPreswizzledScaleTypeTest, AppliesSerializedLayoutNotCurrentFlag) {
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_LINEAR);
  auto module_or =
      ParseAndReturnVerifiedModule(PreswizzledScaledDotFusionHlo(GetParam()));
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  BackendConfig original;
  original.mutable_gemm()->set_algorithm(0);
  original.mutable_gemm()->set_autotune_workspace_size(64 * 1024 * 1024);
  original.mutable_gemm()->set_mx_scale_layout(
      HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8);
  BackendConfig reloaded;
  ASSERT_TRUE(reloaded.ParseFromString(original.SerializeAsString()));
  ASSERT_THAT(backend_.ApplyConfig(
                  *module->entry_computation()->root_instruction(), reloaded),
              absl_testing::IsOk());
  const HloInstruction* call =
      module->entry_computation()->root_instruction()->operand(0);
  auto config_or = call->backend_config<GpuBackendConfig>();
  ASSERT_THAT(config_or, absl_testing::IsOk());
  auto config = std::move(config_or).value();
  EXPECT_EQ(config.gemm_backend_config().scale_mode(), 3);
  for (int operand : {0, 1}) {
    EXPECT_EQ(call->operand(operand),
              module->entry_computation()->parameter_instruction(operand));
    EXPECT_EQ(call->operand(operand)->shape().layout().element_size_in_bits(),
              GetParam() ? 4 : 0);
  }
  EXPECT_EQ(call->operand(2)->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(call->operand(3)->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(call->operand(2)->operand(0),
            module->entry_computation()->parameter_instruction(2));
  EXPECT_EQ(call->operand(3)->operand(0),
            module->entry_computation()->parameter_instruction(3));
}

TEST_P(HipblasLtPreswizzledScaleTypeTest, UnsupportedShapeRetainsLinearCandidate) {
  debug_options_.set_xla_gpu_blas_max_algorithms(4);
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_AUTO);
  auto module_or =
      ParseAndReturnVerifiedModule(PreswizzledScaledDotFusionHlo(GetParam(), 16));
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto configs_or = backend_.GetSupportedConfigs(*fusion);
  ASSERT_THAT(configs_or, absl_testing::IsOk());
  auto configs = std::move(configs_or).value();
  ASSERT_FALSE(configs.empty());
  for (const auto& config : configs) {
    EXPECT_EQ(config->gemm().mx_scale_layout(),
              HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR);
  }
  BackendConfig packed;
  packed.mutable_gemm()->set_mx_scale_layout(
      HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8);
  EXPECT_THAT(backend_.ApplyConfig(*fusion, packed),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(module->entry_computation()->root_instruction(), fusion);
}

TEST_P(HipblasLtPreswizzledScaleTypeTest, CompileIncludesScaleFusions) {
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8);
  auto module_or =
      ParseAndReturnVerifiedModule(PreswizzledScaledDotFusionHlo(GetParam()));
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto config_or = backend_.GetDefaultConfig(*fusion);
  ASSERT_THAT(config_or, absl_testing::IsOk());
  auto config = std::move(config_or).value();
  EXPECT_EQ(config->gemm().mx_scale_layout(),
            HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8);
  EXPECT_THAT(backend_.Compile(*fusion, *config), absl_testing::IsOk());
}

TEST_P(HipblasLtPreswizzledScaleTypeTest,
       RejectsMaterializingReshapeWithIdenticalEndpointShapes) {
  for (auto layout : {HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR,
                      HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8}) {
    // Exercise both data and scale operands. The bitcast reinterprets the
    // row-major parameter as column-major; the reshape then materializes its
    // logical values back into row-major storage. The endpoint shapes match,
    // but bypassing the chain would silently change the GEMM's inputs.
    for (int operand_index : {0, 2}) {
      auto module_or =
          ParseAndReturnVerifiedModule(PreswizzledScaledDotFusionHlo(GetParam()));
      ASSERT_THAT(module_or, absl_testing::IsOk());
      auto module = std::move(module_or).value();
      HloInstruction* fusion = module->entry_computation()->root_instruction();
      HloComputation* fused = fusion->fused_instructions_computation();
      HloInstruction* parameter = fused->parameter_instruction(operand_index);
      Shape column_major = parameter->shape();
      *column_major.mutable_layout() = LayoutUtil::MakeLayout({0, 1});
      column_major.mutable_layout()->set_element_size_in_bits(
          parameter->shape().layout().element_size_in_bits());
      HloInstruction* bitcast = fused->AddInstruction(
          HloInstruction::CreateBitcast(column_major, parameter));
      HloInstruction* reshape = fused->AddInstruction(
          HloInstruction::CreateReshape(parameter->shape(), bitcast));
      ASSERT_FALSE(
          ShapeUtil::ReshapeIsBitcast(bitcast->shape(), reshape->shape()));
      ASSERT_THAT(
          fused->root_instruction()->ReplaceOperandWith(operand_index, reshape),
          absl_testing::IsOk());
      ASSERT_THAT(module->Verify(), absl_testing::IsOk());

      BackendConfig config;
      config.mutable_gemm()->set_mx_scale_layout(layout);
      EXPECT_THAT(backend_.ApplyConfig(*fusion, config),
                  absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
      EXPECT_EQ(module->entry_computation()->root_instruction(), fusion);
    }
  }
}

TEST_P(HipblasLtPreswizzledScaleTypeTest, PreservesContiguousFlattenedOperands) {
  std::string hlo = absl::Substitute(R"(
HloModule FlattenedScaledDotFusion
fusion_dot {
  a = $0[64,16,32]{2,1,0$1} parameter(0)
  b = $0[96,16,32]{2,1,0$1} parameter(1)
  sa = f8e8m0fnu[64,16]{1,0} parameter(2)
  sb = f8e8m0fnu[96,16]{1,0} parameter(3)
  a_flat = $0[64,512]{1,0$1} reshape(a)
  b_flat = $0[96,512]{1,0$1} reshape(b)
  ROOT dot = f32[64,96]{1,0} scaled-dot(a_flat, b_flat, sa, sb),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
ENTRY main {
  a = $0[64,16,32]{2,1,0$1} parameter(0)
  b = $0[96,16,32]{2,1,0$1} parameter(1)
  sa = f8e8m0fnu[64,16]{1,0} parameter(2)
  sb = f8e8m0fnu[96,16]{1,0} parameter(3)
  ROOT fusion = f32[64,96]{1,0} fusion(a,b,sa,sb), kind=kCustom,
      calls=fusion_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})",
                                   GetParam() ? "f4e2m1fn" : "f8e4m3fn",
                                   GetParam() ? ":E(4)" : "");
  auto module_or = ParseAndReturnVerifiedModule(hlo);
  ASSERT_THAT(module_or, absl_testing::IsOk());
  auto module = std::move(module_or).value();
  BackendConfig config;
  config.mutable_gemm()->set_autotune_workspace_size(64 * 1024 * 1024);
  config.mutable_gemm()->set_mx_scale_layout(
      HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8);
  ASSERT_THAT(backend_.ApplyConfig(
                  *module->entry_computation()->root_instruction(), config),
              absl_testing::IsOk());
  const HloInstruction* call =
      module->entry_computation()->root_instruction()->operand(0);
  for (int operand : {0, 1}) {
    const HloInstruction* data = call->operand(operand);
    ASSERT_EQ(data->opcode(), HloOpcode::kBitcast);
    EXPECT_EQ(data->operand(0),
              module->entry_computation()->parameter_instruction(operand));
    EXPECT_EQ(data->shape().layout().element_size_in_bits(), GetParam() ? 4 : 0);
    EXPECT_EQ(ShapeUtil::ByteSizeOf(data->shape()),
              ShapeUtil::ByteSizeOf(data->operand(0)->shape()));
  }
  EXPECT_THAT(module->Verify(), absl_testing::IsOk());
}

INSTANTIATE_TEST_SUITE_P(
    MxTypes, HipblasLtPreswizzledScaleTypeTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Fp4" : "Fp8";
    });

TEST_F(HipblasLtPreswizzledScaleTest, RejectsUnpackedFp4OperandPath) {
  // Check external buffers, fusion parameters and intermediate shapes. The
  // dot's immediate operands stay packed, so its eligibility check alone cannot
  // detect these storage mismatches.
  for (int location : {0, 1, 2}) {
    for (int element_size : {0, 8}) {
      SCOPED_TRACE(::testing::Message()
                   << "location=" << location
                   << " element_size=" << element_size);
      auto module_or =
          ParseAndReturnVerifiedModule(PreswizzledScaledDotFusionHlo(true));
      ASSERT_THAT(module_or, absl_testing::IsOk());
      auto module = std::move(module_or).value();
      HloInstruction* fusion = module->entry_computation()->root_instruction();
      HloComputation* fused = fusion->fused_instructions_computation();
      HloInstruction* parameter = fused->parameter_instruction(0);
      Shape packed_shape = parameter->shape();
      Shape unpacked_shape = packed_shape;
      unpacked_shape.mutable_layout()->set_element_size_in_bits(element_size);
      HloInstruction* source = parameter;
      if (location == 0) {
        *fusion->mutable_operand(0)->mutable_shape() = unpacked_shape;
      } else if (location == 1) {
        *parameter->mutable_shape() = unpacked_shape;
      } else {
        source = fused->AddInstruction(
            HloInstruction::CreateReshape(unpacked_shape, parameter));
      }
      HloInstruction* reshape = fused->AddInstruction(
          HloInstruction::CreateReshape(packed_shape, source));
      ASSERT_TRUE(ShapeUtil::ReshapeIsBitcast(source->shape(), reshape->shape()));
      ASSERT_THAT(fused->root_instruction()->ReplaceOperandWith(0, reshape),
                  absl_testing::IsOk());

      // These deliberately invalid storage layouts must be rejected even when
      // ApplyConfig is called without running the normal layout verifier.
      BackendConfig config;
      config.mutable_gemm()->set_mx_scale_layout(
          HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8);
      EXPECT_THAT(
          backend_.ApplyConfig(*fusion, config),
          absl_testing::StatusIs(
              absl::StatusCode::kInvalidArgument,
              ::testing::HasSubstr("packed E(4) storage throughout")));
      EXPECT_EQ(module->entry_computation()->root_instruction(), fusion);
    }
  }
}

TEST_F(HipblasLtPreswizzledScaleTest,
       LayoutPolicySeparatesAutotuneCacheVersions) {
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_LINEAR);
  std::string linear_version = backend_.version();
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8);
  EXPECT_NE(linear_version, backend_.version());
}

}  // namespace

}  // namespace gpu
}  // namespace xla
