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
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
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
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

// Returns the first device executor of the default platform.
se::StreamExecutor* GetDefaultStreamExecutor() {
  absl::StatusOr<se::Platform*> platform = PlatformUtil::GetDefaultPlatform();
  CHECK_OK(platform.status());
  absl::StatusOr<se::StreamExecutor*> stream_executor =
      (*platform)->ExecutorForDevice(0);
  CHECK_OK(stream_executor.status());
  return *stream_executor;
}

}  // namespace

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
      : debug_options_(
            HloHardwareIndependentTestBase::GetDebugOptionsForTest()),
        stream_executor_(GetDefaultStreamExecutor()),
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kUnsupportedHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *hlo_module->entry_computation()->root_instruction());
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(testing::SizeIs(0)));
}

TEST_F(HipblasLtBackendTest, GetDefaultConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(config,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HipblasLtBackendTest, ApplyConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));
  HipblasLtBackendConfig config;
  config.set_algorithm(2);
  config.set_autotune_workspace_size(42);
  BackendConfig backend_config;
  *backend_config.mutable_gemm() = config;
  EXPECT_OK(backend_.ApplyConfig(*hlo_module->entry_computation()
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHipblasLtCustomCallHlo));
  ASSERT_OK_AND_ASSIGN(
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
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    auto configs = backend_.GetSupportedConfigs(*fusion);
    EXPECT_THAT(configs,
                absl_testing::IsOkAndHolds(testing::SizeIs(testing::Gt(0))));
  }
}

TEST_F(HipblasLtScaledDotTest, ApplyConfig) {
  for (const char* hlo : kScaledDotHlos) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    ASSERT_OK_AND_ASSIGN(auto config, backend_.GetDefaultConfig(*fusion));

    EXPECT_OK(backend_.ApplyConfig(*fusion, *config));

    EXPECT_THAT(RunFileCheck(module->ToString(),
                             R"(CHECK: custom-call
                        CHECK-SAME: custom_call_target="__cublas$lt$matmul$mx"
                        CHECK: "scale_mode":2)"),
                absl_testing::IsOkAndHolds(true));
  }
}

TEST_F(HipblasLtScaledDotTest, Compile) {
  for (const char* hlo : kScaledDotHlos) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    ASSERT_OK_AND_ASSIGN(auto config, backend_.GetDefaultConfig(*fusion));

    auto executable = backend_.Compile(*fusion, *config);
    EXPECT_THAT(executable, absl_testing::IsOk());
  }
}

// A hand-written FP8-output GEMM with an A of type <<A>> and an optional D
// scale. GemmRewriter passes no D scale with the DEFAULT epilogue, so a GEMM
// with one stands in for one created with a wrong SupportsF8OutputDScale.
const char kFp8OutputGemmHlo[] = R"(
HloModule module

ENTRY main {
  a = <<A>>[32,64]{1,0} parameter(0)
  b = f8e4m3fn[16,64]{1,0} parameter(1)
  a_scale = f32[] parameter(2)
  b_scale = f32[] parameter(3)
  d_scale = f32[] parameter(4)
  gemm = (f8e4m3fn[32,16]{1,0}, s8[67108864]{0}) custom-call(a, b, a_scale, b_scale<<D_SCALE>>),
    custom_call_target="__cublas$lt$matmul$f8",
    backend_config={"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","scale_mode":1<<HAS_D_SCALE>>}}
  ROOT out = f8e4m3fn[32,16]{1,0} get-tuple-element(gemm), index=0
})";

class HipblasLtFp8OutputTest : public HipblasLtBackendTest {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> ParseFp8OutputGemm(
      absl::string_view a_type, bool has_d_scale) {
    return ParseAndReturnVerifiedModule(absl::StrReplaceAll(
        kFp8OutputGemmHlo,
        {{"<<A>>", a_type},
         {"<<D_SCALE>>", has_d_scale ? ", d_scale" : ""},
         {"<<HAS_D_SCALE>>", has_d_scale ? R"(,"has_d_scale":true)" : ""}}));
  }
};

// The output is unfused only if hipBLASLt has no algorithm for the FP8 output.
// It has D-scaled algorithms only for E4M3 x E4M3 -> E4M3 on gfx950.
TEST_F(HipblasLtFp8OutputTest, UnfusesOutputOnlyWithoutAlgorithms) {
  const auto* rocm_cc =
      target_config_.device_description.gpu_compute_capability()
          .rocm_compute_capability();
  if (rocm_cc == nullptr || !rocm_cc->has_ocp_fp8_support()) {
    GTEST_SKIP() << "Requires OCP FP8 support.";
  }
  // {A type, has a D scale, unfuses the output}
  for (auto [a_type, has_d_scale, unfused] :
       {std::tuple{"f8e5m2", false, false}, std::tuple{"f8e4m3fn", true, false},
        std::tuple{"f8e5m2", true, true}}) {
    SCOPED_TRACE(testing::Message() << a_type << " " << has_d_scale);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseFp8OutputGemm(a_type, has_d_scale));
    ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<BackendConfig>> configs,
        backend_.GetSupportedConfigs(
            *module->entry_computation()->root_instruction()->operand(0)));
    ASSERT_THAT(configs, testing::Not(testing::IsEmpty()));
    if (unfused && !configs[0]->gemm().unfuse_fp8_output()) {
      GTEST_SKIP() << "hipBLASLt has D-scaled algorithms for E5M2 x E4M3.";
    }
    for (const std::unique_ptr<BackendConfig>& config : configs) {
      EXPECT_EQ(config->gemm().unfuse_fp8_output(), unfused);
    }
  }
}

TEST_F(HipblasLtFp8OutputTest, ApplyUnfusedOutputConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseFp8OutputGemm("f8e5m2", true));
  BackendConfig config;
  config.mutable_gemm()->set_algorithm(2);
  config.mutable_gemm()->set_autotune_workspace_size(42);
  config.mutable_gemm()->set_unfuse_fp8_output(true);
  EXPECT_OK(backend_.ApplyConfig(
      *module->entry_computation()->root_instruction()->mutable_operand(0),
      config));
  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
CHECK:      fused_computation
CHECK-DAG:    multiply(
CHECK-DAG:    f32[] constant(-448)
CHECK-DAG:    f32[] constant(448)
CHECK-DAG:    clamp(
CHECK:        ROOT {{.*}} = f8e4m3fn[32,16]{1,0} convert(
CHECK:      ENTRY
CHECK:        (f32[32,16]{1,0}, s8[42]{0}) custom-call({{[^)]*}} %b_scale)
CHECK-SAME:     custom_call_target="__cublas$lt$matmul$f8"
CHECK-SAME:     "selected_algorithm":"2"
CHECK-NOT:      has_d_scale
CHECK:        ROOT {{.*}} = f8e4m3fn[32,16]{1,0} fusion({{.*}}), kind=kLoop
)"),
              absl_testing::IsOkAndHolds(true));
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

// Scaled-dot fusions with the epilogues that GemmFusionV2 fuses in a Llama
// layer, with the extra BF16 parameters first: a residual add, and the
// accumulation of an activation gradient with two others.
const char kScaledDotResidualFusionHlo[] = R"(
HloModule ScaledDotResidualFusion
fusion_dot {
  r0 = bf16[1,64,96]{2,1,0} parameter(0)
  a = $0[1,64,512]{2,1,0$1} parameter(1)
  b = $0[1,96,512]{2,1,0$1} parameter(2)
  sa = f8e8m0fnu[1,64,16]{2,1,0} parameter(3)
  sb = f8e8m0fnu[1,96,16]{2,1,0} parameter(4)
  dot = bf16[1,64,96]{2,1,0} scaled-dot(a, b, sa, sb),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
  r0_f32 = f32[1,64,96]{2,1,0} convert(r0)
  dot_f32 = f32[1,64,96]{2,1,0} convert(dot)
  sum0 = f32[1,64,96]{2,1,0} add(r0_f32, dot_f32)
  ROOT out = bf16[1,64,96]{2,1,0} convert(sum0)
}
ENTRY main {
  r0 = bf16[1,64,96]{2,1,0} parameter(0)
  a = $0[1,64,512]{2,1,0$1} parameter(1)
  b = $0[1,96,512]{2,1,0$1} parameter(2)
  sa = f8e8m0fnu[1,64,16]{2,1,0} parameter(3)
  sb = f8e8m0fnu[1,96,16]{2,1,0} parameter(4)
  ROOT fusion = bf16[1,64,96]{2,1,0} fusion(r0, a, b, sa, sb), kind=kCustom,
      calls=fusion_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

const char kScaledDotAccumulationFusionHlo[] = R"(
HloModule ScaledDotAccumulationFusion
fusion_dot {
  r0 = bf16[1,64,96]{2,1,0} parameter(0)
  r1 = bf16[1,64,96]{2,1,0} parameter(1)
  a = $0[1,64,512]{2,1,0$1} parameter(2)
  b = $0[1,96,512]{2,1,0$1} parameter(3)
  sa = f8e8m0fnu[1,64,16]{2,1,0} parameter(4)
  sb = f8e8m0fnu[1,96,16]{2,1,0} parameter(5)
  dot = bf16[1,64,96]{2,1,0} scaled-dot(a, b, sa, sb),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
  dot_f32 = f32[1,64,96]{2,1,0} convert(dot)
  r1_f32 = f32[1,64,96]{2,1,0} convert(r1)
  sum1 = f32[1,64,96]{2,1,0} add(dot_f32, r1_f32)
  r0_f32 = f32[1,64,96]{2,1,0} convert(r0)
  sum0 = f32[1,64,96]{2,1,0} add(sum1, r0_f32)
  ROOT out = bf16[1,64,96]{2,1,0} convert(sum0)
}
ENTRY main {
  r0 = bf16[1,64,96]{2,1,0} parameter(0)
  r1 = bf16[1,64,96]{2,1,0} parameter(1)
  a = $0[1,64,512]{2,1,0$1} parameter(2)
  b = $0[1,96,512]{2,1,0$1} parameter(3)
  sa = f8e8m0fnu[1,64,16]{2,1,0} parameter(4)
  sb = f8e8m0fnu[1,96,16]{2,1,0} parameter(5)
  ROOT fusion = bf16[1,64,96]{2,1,0} fusion(r0, r1, a, b, sa, sb),
      kind=kCustom, calls=fusion_dot,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

// The parameter is {FP4 data, number of extra BF16 parameters}.
class HipblasLtMxEpilogueTest
    : public HipblasLtPreswizzledScaleTest,
      public ::testing::WithParamInterface<std::tuple<bool, int>> {
 protected:
  bool fp4() const { return std::get<0>(GetParam()); }
  int num_addends() const { return std::get<1>(GetParam()); }
  std::string Hlo() const {
    return absl::Substitute(
        num_addends() == 1 ? kScaledDotResidualFusionHlo
                           : kScaledDotAccumulationFusionHlo,
        fp4() ? "f4e2m1fn" : "f8e4m3fn", fp4() ? ":E(4)" : "");
  }
};

TEST_P(HipblasLtMxEpilogueTest, OffersBothScaleLayouts) {
  debug_options_.set_xla_gpu_blas_max_algorithms(4);
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_AUTO);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(Hlo()));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
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
}

TEST_P(HipblasLtMxEpilogueTest, MovesEpilogueAfterTheGemm) {
  for (auto layout : {HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR,
                      HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8}) {
    SCOPED_TRACE(::testing::Message() << "layout=" << layout);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(Hlo()));
    HloComputation* entry = module->entry_computation();
    Shape fusion_shape = entry->root_instruction()->shape();
    BackendConfig config;
    config.mutable_gemm()->set_autotune_workspace_size(64 * 1024 * 1024);
    config.mutable_gemm()->set_mx_scale_layout(layout);
    ASSERT_THAT(backend_.ApplyConfig(*entry->root_instruction(), config),
                absl_testing::IsOk());
    EXPECT_THAT(module->Verify(), absl_testing::IsOk());

    // The root is a loop fusion of the GEMM result and the extra parameters.
    const HloInstruction* epilogue = entry->root_instruction();
    ASSERT_EQ(epilogue->opcode(), HloOpcode::kFusion);
    EXPECT_EQ(epilogue->fusion_kind(), HloInstruction::FusionKind::kLoop);
    EXPECT_THAT(epilogue->name(), ::testing::StartsWith("fusion_epilogue"));
    EXPECT_TRUE(ShapeUtil::Equal(epilogue->shape(), fusion_shape));
    ASSERT_EQ(epilogue->operand_count(), 1 + num_addends());
    for (int addend = 0; addend < num_addends(); ++addend) {
      EXPECT_EQ(epilogue->operand(1 + addend),
                entry->parameter_instruction(addend));
    }
    const HloInstruction* gte = epilogue->operand(0);
    ASSERT_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
    const HloInstruction* call = gte->operand(0);
    ASSERT_EQ(call->opcode(), HloOpcode::kCustomCall);
    EXPECT_EQ(call->custom_call_target(), "__cublas$lt$matmul$mx");
    ASSERT_EQ(call->operand_count(), 4);
    ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                         call->backend_config<GpuBackendConfig>());
    EXPECT_EQ(gpu_config.gemm_backend_config().scale_mode(),
              layout == HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR ? 2 : 3);

    // The data and scales feed only the GEMM, the scales through the swizzle
    // fusions under 32x8.
    for (int operand = 0; operand < 4; ++operand) {
      const HloInstruction* parameter =
          entry->parameter_instruction(num_addends() + operand);
      ASSERT_EQ(parameter->user_count(), 1);
      if (operand >= 2 &&
          layout == HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8) {
        EXPECT_EQ(call->operand(operand)->opcode(), HloOpcode::kFusion);
        EXPECT_EQ(call->operand(operand)->operand(0), parameter);
        EXPECT_EQ(parameter->users()[0], call->operand(operand));
      } else {
        EXPECT_EQ(call->operand(operand), parameter);
      }
    }

    // The body takes the GEMM result in place of the scaled-dot and applies
    // the converts and adds; no data or scale reaches it.
    const HloComputation* body = epilogue->fused_instructions_computation();
    ASSERT_EQ(body->num_parameters(), 1 + num_addends());
    EXPECT_TRUE(ShapeUtil::Equal(body->parameter_instruction(0)->shape(),
                                 call->shape().tuple_shapes(0)));
    EXPECT_TRUE(
        ShapeUtil::Equal(body->root_instruction()->shape(), fusion_shape));
    EXPECT_EQ(body->root_instruction()->opcode(), HloOpcode::kConvert);
    int num_adds = 0;
    for (const HloInstruction* instr : body->instructions()) {
      EXPECT_NE(instr->opcode(), HloOpcode::kScaledDot);
      EXPECT_THAT(instr->shape().element_type(), ::testing::AnyOf(BF16, F32));
      num_adds += instr->opcode() == HloOpcode::kAdd;
    }
    EXPECT_EQ(num_adds, num_addends());
  }
}

// Every config that GetSupportedConfigs offers can be applied, which is what
// filtering the configs with the lowering's analysis promises.
TEST_P(HipblasLtMxEpilogueTest, AppliesEverySupportedConfig) {
  debug_options_.set_xla_gpu_blas_max_algorithms(4);
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_AUTO);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(Hlo()));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  bool has_linear = false;
  bool has_32x8 = false;
  for (const std::unique_ptr<BackendConfig>& config : configs) {
    SCOPED_TRACE(config->ShortDebugString());
    has_linear |= config->gemm().mx_scale_layout() ==
                  HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR;
    has_32x8 |= config->gemm().mx_scale_layout() ==
                HipblasLtBackendConfig::MX_SCALE_LAYOUT_HIPBLASLT_32X8;
    // A fresh copy of the module for each config.
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> copy,
                         ParseAndReturnVerifiedModule(Hlo()));
    EXPECT_THAT(backend_.ApplyConfig(
                    *copy->entry_computation()->root_instruction(), *config),
                absl_testing::IsOk());
    EXPECT_THAT(copy->Verify(), absl_testing::IsOk());
  }
  EXPECT_TRUE(has_linear);
  EXPECT_TRUE(has_32x8);
}

TEST_P(HipblasLtMxEpilogueTest, CompilesWithEpilogue) {
  for (auto policy :
       {DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_LINEAR,
        DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8}) {
    SCOPED_TRACE(::testing::Message() << "policy=" << policy);
    debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(policy);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(Hlo()));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                         backend_.GetDefaultConfig(*fusion));
    EXPECT_THAT(backend_.Compile(*fusion, *config), absl_testing::IsOk());
  }
}

INSTANTIATE_TEST_SUITE_P(
    MxEpilogues, HipblasLtMxEpilogueTest,
    ::testing::Combine(::testing::Bool(), ::testing::Values(1, 2)),
    [](const ::testing::TestParamInfo<std::tuple<bool, int>>& info) {
      return absl::StrCat(std::get<0>(info.param) ? "Fp4" : "Fp8", "With",
                          std::get<1>(info.param), "Addends");
    });

// A scaled-dot fusion whose root <<ROOT>> of shape <<SHAPE>> changes the shape
// of the scaled-dot result.
const char kScaledDotShapeChangingEpilogueHlo[] = R"(
HloModule ScaledDotShapeChangingEpilogue
fusion_dot {
  a = f8e4m3fn[1,64,512]{2,1,0} parameter(0)
  b = f8e4m3fn[1,96,512]{2,1,0} parameter(1)
  sa = f8e8m0fnu[1,64,16]{2,1,0} parameter(2)
  sb = f8e8m0fnu[1,96,16]{2,1,0} parameter(3)
  dot = bf16[1,64,96]{2,1,0} scaled-dot(a, b, sa, sb),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
  ROOT out = <<SHAPE>> <<ROOT>>
}
ENTRY main {
  a = f8e4m3fn[1,64,512]{2,1,0} parameter(0)
  b = f8e4m3fn[1,96,512]{2,1,0} parameter(1)
  sa = f8e8m0fnu[1,64,16]{2,1,0} parameter(2)
  sb = f8e8m0fnu[1,96,16]{2,1,0} parameter(3)
  ROOT fusion = <<SHAPE>> fusion(a, b, sa, sb), kind=kCustom,
      calls=fusion_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

TEST_F(HipblasLtScaledDotTest, RefusesNonElementwiseEpilogue) {
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_AUTO);
  // {shape, root}
  for (auto [shape, root] :
       {std::tuple{"bf16[1,96,64]{2,1,0}",
                   "transpose(dot), dimensions={0,2,1}"},
        std::tuple{"bf16[1,64,96,2]{3,2,1,0}",
                   "broadcast(dot), dimensions={0,1,2}"}}) {
    SCOPED_TRACE(root);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                             kScaledDotShapeChangingEpilogueHlo,
                             {{"<<SHAPE>>", shape}, {"<<ROOT>>", root}})));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    EXPECT_THAT(backend_.GetSupportedConfigs(*fusion),
                absl_testing::IsOkAndHolds(testing::IsEmpty()));
    EXPECT_THAT(backend_.GetDefaultConfig(*fusion),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
    BackendConfig config;
    config.mutable_gemm()->set_mx_scale_layout(
        HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR);
    EXPECT_THAT(
        backend_.ApplyConfig(*fusion, config),
        absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               ::testing::HasSubstr("not elementwise")));
    EXPECT_EQ(module->entry_computation()->root_instruction(), fusion);
  }
}

// An epilogue that only bitcasts the [1,M,N] result to [M,N] needs no kernel.
TEST_F(HipblasLtScaledDotTest, BitcastsResultWithoutEpilogueFusion) {
  for (absl::string_view root : {"bitcast(dot)", "reshape(dot)"}) {
    SCOPED_TRACE(root);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VerifiedHloModule> module,
        ParseAndReturnVerifiedModule(absl::StrReplaceAll(
            kScaledDotShapeChangingEpilogueHlo,
            {{"<<SHAPE>>", "bf16[64,96]{1,0}"}, {"<<ROOT>>", root}})));
    HloComputation* entry = module->entry_computation();
    BackendConfig config;
    config.mutable_gemm()->set_autotune_workspace_size(64 * 1024 * 1024);
    config.mutable_gemm()->set_mx_scale_layout(
        HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR);
    ASSERT_THAT(backend_.ApplyConfig(*entry->root_instruction(), config),
                absl_testing::IsOk());
    EXPECT_THAT(module->Verify(), absl_testing::IsOk());

    // The entry holds the four parameters, the custom call, its result and
    // the bitcast of the result, and no fusion.
    const HloInstruction* result = entry->root_instruction();
    ASSERT_EQ(result->opcode(), HloOpcode::kBitcast);
    EXPECT_TRUE(ShapeUtil::Equal(
        result->shape(),
        ShapeUtil::MakeShapeWithDenseLayout(BF16, {64, 96}, {1, 0})));
    const HloInstruction* gte = result->operand(0);
    ASSERT_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
    const HloInstruction* call = gte->operand(0);
    ASSERT_EQ(call->opcode(), HloOpcode::kCustomCall);
    EXPECT_EQ(call->custom_call_target(), "__cublas$lt$matmul$mx");
    for (int operand = 0; operand < 4; ++operand) {
      EXPECT_EQ(call->operand(operand), entry->parameter_instruction(operand));
    }
    EXPECT_EQ(entry->instruction_count(), 7);
    for (const HloInstruction* instr : entry->instructions()) {
      EXPECT_NE(instr->opcode(), HloOpcode::kFusion) << instr->ToString();
    }
  }
}

// An FP8 scaled-dot fusion over [1,64,64] operands, so that the data have the
// dimensions of the result. Each case of RefusesFusionStructure supplies the
// fused computation's body, the operands of the fusion, the shape of `a` and
// the result shape.
const char kMxFusionStructureHlo[] = R"(
HloModule MxFusionStructure
fusion_dot {
<<BODY>>
}
ENTRY main {
  a = <<A>> parameter(0)
  b = f8e4m3fn[1,64,64]{2,1,0} parameter(1)
  sa = f8e8m0fnu[1,64,2]{2,1,0} parameter(2)
  sb = f8e8m0fnu[1,64,2]{2,1,0} parameter(3)
  ROOT fusion = <<RESULT>> fusion(<<OPERANDS>>), kind=kCustom,
      calls=fusion_dot, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

TEST_F(HipblasLtScaledDotTest, RefusesFusionStructure) {
  debug_options_.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
      DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_AUTO);
  constexpr absl::string_view kParams = R"(
  b = f8e4m3fn[1,64,64]{2,1,0} parameter(1)
  sa = f8e8m0fnu[1,64,2]{2,1,0} parameter(2)
  sb = f8e8m0fnu[1,64,2]{2,1,0} parameter(3))";
  constexpr absl::string_view kDims =
      "lhs_batch_dims={0}, lhs_contracting_dims={2}, "
      "rhs_batch_dims={0}, rhs_contracting_dims={2}";
  struct Case {
    absl::string_view name;
    absl::string_view a_shape;
    absl::string_view body;
    absl::string_view operands;
    absl::string_view result_shape;
    absl::string_view error;
  };
  const Case kCases[] = {
      {"data parameter also feeds the epilogue", "f8e4m3fn[1,64,64]{2,1,0}",
       R"(
  a = f8e4m3fn[1,64,64]{2,1,0} parameter(0)<<PARAMS>>
  dot = bf16[1,64,64]{2,1,0} scaled-dot(a, b, sa, sb), <<DIMS>>
  a_bf16 = bf16[1,64,64]{2,1,0} convert(a)
  ROOT out = bf16[1,64,64]{2,1,0} add(dot, a_bf16))",
       "a, b, sa, sb", "bf16[1,64,64]{2,1,0}", "several users"},
      {"one parameter feeds two operands", "f8e4m3fn[1,64,64]{2,1,0}", R"(
  a = f8e4m3fn[1,64,64]{2,1,0} parameter(0)
  sa = f8e8m0fnu[1,64,2]{2,1,0} parameter(1)
  sb = f8e8m0fnu[1,64,2]{2,1,0} parameter(2)
  ROOT dot = bf16[1,64,64]{2,1,0} scaled-dot(a, a, sa, sb), <<DIMS>>)",
       "a, sa, sb", "bf16[1,64,64]{2,1,0}", "Several scaled-dot operands"},
      {"operand path value has two users", "f8e4m3fn[64,64]{1,0}", R"(
  a = f8e4m3fn[64,64]{1,0} parameter(0)<<PARAMS>>
  a_3d = f8e4m3fn[1,64,64]{2,1,0} bitcast(a)
  dot = bf16[1,64,64]{2,1,0} scaled-dot(a_3d, b, sa, sb), <<DIMS>>
  a_bf16 = bf16[1,64,64]{2,1,0} convert(a_3d)
  ROOT out = bf16[1,64,64]{2,1,0} add(dot, a_bf16))",
       "a, b, sa, sb", "bf16[1,64,64]{2,1,0}", "several users"},
      {"transposing reshape on an operand path", "f8e4m3fn[64,64]{0,1}", R"(
  a = f8e4m3fn[64,64]{0,1} parameter(0)<<PARAMS>>
  a_3d = f8e4m3fn[1,64,64]{2,1,0} reshape(a)
  ROOT dot = bf16[1,64,64]{2,1,0} scaled-dot(a_3d, b, sa, sb), <<DIMS>>)",
       "a, b, sa, sb", "bf16[1,64,64]{2,1,0}", "materializing reshape"},
      {"slice in the epilogue", "f8e4m3fn[1,64,64]{2,1,0}", R"(
  a = f8e4m3fn[1,64,64]{2,1,0} parameter(0)<<PARAMS>>
  dot = bf16[1,64,64]{2,1,0} scaled-dot(a, b, sa, sb), <<DIMS>>
  ROOT out = bf16[1,64,32]{2,1,0} slice(dot), slice={[0:1], [0:64], [0:32]})",
       "a, b, sa, sb", "bf16[1,64,32]{2,1,0}", "not elementwise"},
  };
  for (const Case& c : kCases) {
    SCOPED_TRACE(c.name);
    std::string body = absl::StrReplaceAll(
        c.body, {{"<<PARAMS>>", kParams}, {"<<DIMS>>", kDims}});
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VerifiedHloModule> module,
        ParseAndReturnVerifiedModule(absl::StrReplaceAll(
            kMxFusionStructureHlo, {{"<<BODY>>", body},
                                    {"<<A>>", c.a_shape},
                                    {"<<OPERANDS>>", c.operands},
                                    {"<<RESULT>>", c.result_shape}})));
    HloInstruction* fusion = module->entry_computation()->root_instruction();
    const std::string before = module->ToString();
    EXPECT_THAT(backend_.GetSupportedConfigs(*fusion),
                absl_testing::IsOkAndHolds(testing::IsEmpty()));
    EXPECT_THAT(backend_.GetDefaultConfig(*fusion),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       ::testing::HasSubstr(c.error)));
    BackendConfig config;
    config.mutable_gemm()->set_autotune_workspace_size(64 * 1024 * 1024);
    config.mutable_gemm()->set_mx_scale_layout(
        HipblasLtBackendConfig::MX_SCALE_LAYOUT_LINEAR);
    EXPECT_THAT(backend_.ApplyConfig(*fusion, config),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       ::testing::HasSubstr(c.error)));
    EXPECT_EQ(module->ToString(), before);
  }
}

}  // namespace

}  // namespace gpu
}  // namespace xla
