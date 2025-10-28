/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/triton_fusion_numerics_verifier.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

class TritonFusionNumericsVerifierTest
    : public HloTestBase,
      public ::testing::WithParamInterface<PrimitiveType> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    auto options = HloTestBase::GetDebugOptionsForTest();
    options.set_xla_gpu_verify_triton_fusion_numerics(true);
    return options;
  }

 protected:
  std::unique_ptr<xla::HloModule> Module(absl::string_view hlo_text_template,
                                         absl::string_view type) {
    auto m = ParseAndReturnVerifiedModule(
        absl::Substitute(hlo_text_template, type), GetModuleConfigForTest());
    TF_EXPECT_OK(m);
    return std::move(m.value());
  }

  const HloFusionInstruction* TritonFusion(const xla::HloModule& module) {
    const HloFusionInstruction* fusion_result = nullptr;

    absl::Status res =
        triton_fusion_numerics_pass_internal::ForAllTritonFusions(
            module, /*execution_threads=*/{},
            [&](const HloFusionInstruction& fusion) -> absl::Status {
              EXPECT_EQ(fusion_result, nullptr);
              fusion_result = &fusion;
              return absl::OkStatus();
            });
    return fusion_result;
  }

  AutotuneConfig CreateAutotuneConfig() {
    se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
    auto executors_or = PlatformUtil::GetStreamExecutors(platform);
    TF_EXPECT_OK(executors_or);
    return AutotuneConfig{DeviceConfig{executors_or->at(0), nullptr},
                          GetDebugOptionsForTest()};
  }

  AutotunerCompileUtil CreateAutotunerCompileUtil(AutotuneConfig& config) {
    auto compile_util_or =
        AutotunerCompileUtil::Create(config, GetDebugOptionsForTest());
    TF_EXPECT_OK(compile_util_or);
    return std::move(compile_util_or).value();
  }
};

constexpr absl::string_view kSoftmaxHlo = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
triton_softmax_computation {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  exponential = $0[127,125]{1,0} exponential(subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = $0[127,125]{1,0} divide(exponential, second_broadcast)
}
ENTRY main{
  p = $0[127,125] parameter(0)
  ROOT triton_softmax = $0[127,125] fusion(p), kind=kCustom,
    calls=triton_softmax_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","125"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1"}}}
})";


TEST_F(TritonFusionNumericsVerifierTest, CheckMismatch) {
  // This test intentionally compares two different Triton modules to each
  // other. This is to test that the verifier functions correctly catch and
  // report mismatches.
  //
  // Note that as part of computing the two modules below, the numerics verifier
  // pass also runs individually for each module. These runs compare the
  // modules to the corresponding emitters generated version, which matches. In
  // that sense this test covers what is being tested by
  // VerifyExactSoftmaxFusionNumerics. The reason to keep two tests is that
  // VerifyExactSoftmaxFusionNumerics is minimal and will be easier to debug if
  // it fails.

  auto module_f64 = Module(kSoftmaxHlo, "f64");
  auto fusion_f64 = TritonFusion(*module_f64);
  EXPECT_NE(fusion_f64, nullptr);

  auto module_f32 = Module(kSoftmaxHlo, "f32");
  auto fusion_f32 = TritonFusion(*module_f32);
  EXPECT_NE(fusion_f32, nullptr);

  AutotuneConfig autotune_config = CreateAutotuneConfig();
  AutotunerCompileUtil compile_util =
      CreateAutotunerCompileUtil(autotune_config);
  const DebugOptions& debug_options = GetDebugOptionsForTest();

  auto f64_result = triton_fusion_numerics_pass_internal::CompileAndRunFusion(
      compile_util, *fusion_f64, autotune_config, debug_options,
      /*clear_backend_config=*/false);
  TF_EXPECT_OK(f64_result);

  auto f32_result = triton_fusion_numerics_pass_internal::CompileAndRunFusion(
      compile_util, *fusion_f32, autotune_config, debug_options,
      /*clear_backend_config=*/false);
  TF_EXPECT_OK(f32_result);

  auto stream = autotune_config.GetStream();
  TF_EXPECT_OK(stream);

  // Intentionally compare the fusions from the different modules, triggering a
  // mismatch.
  auto cmp = triton_fusion_numerics_pass_internal::CompareBuffers(
      *f64_result, *f32_result, fusion_f64->shape(),
      fusion_f64->GetModule()->config(), *stream);

  EXPECT_FALSE(cmp.ok());
}

// By default, AutotunerCompileUtil filters out kernels that cause registers to
// spill. Verify that the numerics verifier still runs on those kernels.
TEST_F(TritonFusionNumericsVerifierTest,
       CompilationSucceedsEvenIfKernelWillSpillRegisters) {
  auto module = Module(R"(
HloModule m

add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

triton_softmax_computation {
  param_0 = f32[16,256000] parameter(0)
  constant_0 = f32[] constant(0)
  reduce_0 = f32[16]{0} reduce(param_0, constant_0), dimensions={1}, to_apply=add
  broadcast_0 = f32[16,256000]{1,0} broadcast(reduce_0), dimensions={0}
  ROOT multiply = f32[16,256000]{1,0} multiply(param_0, broadcast_0)
}

ENTRY main {
  param_0 = f32[16,256000] parameter(0)
  ROOT triton_softmax = f32[16,256000]{1,0} fusion(param_0), kind=kCustom,
    calls=triton_softmax_computation, backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","256000"]}],
          "num_warps":"32",
          "num_ctas":"1",
          "num_stages":"1"}}}
})",
                       "");

  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
  auto fusion = TritonFusion(*module);
  EXPECT_NE(fusion, nullptr);

  AutotuneConfig autotune_config = CreateAutotuneConfig();
  AutotunerCompileUtil compile_util =
      CreateAutotunerCompileUtil(autotune_config);
  auto compilation_result =
      triton_fusion_numerics_pass_internal::CompileAndRunFusion(
          compile_util, *fusion, autotune_config, GetDebugOptionsForTest(),
          /*disable_triton=*/false);

  // Verify that the compilation with default flags fails. The compilation
  // fails, because the kernel will spill registers, but the error is
  // overwritten inside the autotuner utils and returns a generic error.
  EXPECT_FALSE(compilation_result.ok());
  EXPECT_THAT(compilation_result.status(),
              tsl::testing::StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(compilation_result.status().message(),
              ::testing::HasSubstr("Failed to compile Triton fusion"));
}

TEST_F(TritonFusionNumericsVerifierTest, CacheIsUsed) {
  absl::string_view hlo_text = R"(
add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] maximum(p0, p1)
}

reduce_0 {
  p = f32[16,16] parameter(0)
  c = f32[] constant(0)
  ROOT reduce_0 = f32[16]{0} reduce(p, c), dimensions={1}, to_apply=add
}

reduce_1 {
  p = f32[16,16] parameter(0)
  c = f32[] constant(0)
  ROOT reduce_0 = f32[16]{0} reduce(p, c), dimensions={1}, to_apply=max
}

// Identical to reduce_0.
reduce_2 {
  p = f32[16,16] parameter(0)
  c = f32[] constant(0)
  ROOT reduce_0 = f32[16]{0} reduce(p, c), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[16,16] parameter(0)
  p1 = f32[16,16] parameter(1)
  p2 = f32[16,16] parameter(2)
  r0 = f32[16] fusion(p0), kind=kCustom, calls=reduce_0, backend_config={"fusion_backend_config": {"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["16"]}],"num_warps":"1","num_ctas":"1","num_stages":"1"}}}
  r1 = f32[16] fusion(p1), kind=kCustom, calls=reduce_1, backend_config={"fusion_backend_config": {"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["16"]}],"num_warps":"1","num_ctas":"1","num_stages":"1"}}}
  r2 = f32[16] fusion(p2), kind=kCustom, calls=reduce_2, backend_config={"fusion_backend_config": {"kind":"__triton","block_level_fusion_config":{"output_tiles":[{"sizes":["16"]}],"num_warps":"1","num_ctas":"1","num_stages":"1"}}}
  add_0_1 = f32[16] add(r0, r1)
  ROOT add_0_2 = f32[16] add(add_0_1, r2)
}
  )";

  std::unique_ptr<HloModule> module = Module(hlo_text, "");
  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
  EXPECT_EQ(verifier.CacheHitsForTestingOnly(), 1);
}

TEST_F(TritonFusionNumericsVerifierTest, VerifyThatDisablingTritonIsFast) {
  // This computation results in a single Triton fusion. If that fusion is
  // compiled without Triton and without rerunning the fusion pass, the
  // resulting kernel is extremely slow and the test will timeout. This test
  // ensures that the fusion pass is rerun.
  absl::string_view hlo_text = R"(
max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT max = f32[] maximum(p0, p1)
}

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

triton_softmax_computation {
  p0 = f32[16384,16384] parameter(0)
  reshape1 = f32[1,1,16384,16384] reshape(p0)
  reshape2 = f32[1,16384,16384] reshape(p0)
  constant3 = f32[] constant(-inf)
  reduce0 = f32[1,16384] reduce(reshape2, constant3), dimensions={2}, to_apply=max
  broadcast3 = f32[1,1,16384,16384] broadcast(reduce0), dimensions={1,2}
  sub = f32[1,1,16384,16384] subtract(reshape1, broadcast3)
  exp = f32[1,1,16384,16384] exponential(sub)
  reshape3 = f32[1,16384,16384] reshape(exp)
  constant4 = f32[] constant(0)
  reduce1 = f32[1,16384] reduce(reshape3, constant4), dimensions={2}, to_apply=add
  broadcast4 = f32[1,1,16384,16384] broadcast(reduce1), dimensions={1,2}
  ROOT div = f32[1,1,16384,16384] divide(exp, broadcast4)
}

ENTRY main {
  p = f32[16384,16384] parameter(0)
  ROOT triton_softmax = f32[1,1,16384,16384] fusion(p), kind=kCustom,
    calls=triton_softmax_computation, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1","1","16384"]}],
        "num_warps":"32",
        "num_ctas":"1",
        "num_stages":"1"}}}
}
  )";
  auto module = Module(hlo_text, "");
  EXPECT_NE(TritonFusion(*module), nullptr);
  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
}

TEST_F(TritonFusionNumericsVerifierTest, SoftmaxDebug) {
  absl::string_view hlo_text = R"(
HloModule fused_computation_1333_standalone

%region_1.1358 (param_0: f32[], param_1: f32[]) -> f32[] {
  %param_0 = f32[] parameter(0)
  %param_1 = f32[] parameter(1)
  ROOT %add = f32[] add(%param_0, %param_1)
}

%fused_computation.1333 (param_0.6350: pred[], param_1.6600: bf16[6,5120]{1,0}, param_2.4391: f32[1,11160,6,5120]{3,1,2,0}, param_3.3137: bf16[11160,5120]{1,0}, param_4.2552: bf16[1,11160,5120]{2,1,0}, param_5.2297: bf16[11160,5120]{1,0}) -> f32[93,5120]{1,0} {
  %param_0.6350 = pred[] parameter(0), metadata={scheduling_name="param_0.6350"}
  %broadcast.11424.20 = pred[1,11160,5120]{2,1,0} broadcast(%param_0.6350), dimensions={}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/cross_attn/cross_attn.forward/o/o.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/linear.py" source_line=75 scheduling_name="broadcast.11424.20"}
  %param_4.2552 = bf16[1,11160,5120]{2,1,0} parameter(4), metadata={scheduling_name="param_4.2552"}
  %convert.7998.42 = f32[1,11160,5120]{2,1,0} convert(%param_4.2552), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=140 scheduling_name="convert.7998.42"}
  %param_3.3137 = bf16[11160,5120]{1,0} parameter(3), metadata={scheduling_name="param_3.3137"}
  %bitcast.54878.21 = bf16[1,11160,5120]{2,1,0} bitcast(%param_3.3137), metadata={scheduling_name="bitcast.54878.21"}
  %convert.8020.21 = f32[1,11160,5120]{2,1,0} convert(%bitcast.54878.21), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=141 scheduling_name="convert.8020.21"}
  %param_2.4391 = f32[1,11160,6,5120]{3,1,2,0} parameter(2), metadata={scheduling_name="param_2.4391"}
  %bitcast.54624.69 = f32[1,6,11160,5120]{3,2,1,0} bitcast(%param_2.4391), metadata={scheduling_name="bitcast.54624.69"}
  %param_1.6600 = bf16[6,5120]{1,0} parameter(1), metadata={scheduling_name="param_1.6600"}
  %convert.7999.30 = f32[6,5120]{1,0} convert(%param_1.6600), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=218 scheduling_name="convert.7999.30"}
  %broadcast.11347.62 = f32[1,6,11160,5120]{3,2,1,0} broadcast(%convert.7999.30), dimensions={1,3}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=218 scheduling_name="broadcast.11347.62"}
  %add.5150.69 = f32[1,6,11160,5120]{3,2,1,0} add(%bitcast.54624.69, %broadcast.11347.62), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=218 scheduling_name="add.5150.69"}
  %slice.1536.15 = f32[1,1,11160,5120]{3,2,1,0} slice(%add.5150.69), slice={[0:1], [2:3], [0:11160], [0:5120]}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/slice" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=244 scheduling_name="slice.1536.15"}
  %bitcast.54884.16 = f32[1,11160,5120]{2,1,0} bitcast(%slice.1536.15), metadata={scheduling_name="bitcast.54884.16"}
  %multiply.5900.13 = f32[1,11160,5120]{2,1,0} multiply(%convert.8020.21, %bitcast.54884.16), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=178 scheduling_name="multiply.5900.13"}
  %add.5157.11 = f32[1,11160,5120]{2,1,0} add(%convert.7998.42, %multiply.5900.13), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=178 scheduling_name="add.5157.11"}
  %bitcast.76333 = f32[11160,5120]{1,0} bitcast(%add.5157.11), metadata={scheduling_name="bitcast.76333"}
  %constant_7904 = f32[] constant(0)
  %reduce.1528 = f32[11160]{0} reduce(%bitcast.76333, %constant_7904), dimensions={1}, to_apply=%region_1.1358, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117 scheduling_name="reduce.1528"}
  %bitcast.76332 = f32[11160]{0} bitcast(%reduce.1528), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117 scheduling_name="bitcast.76332"}
  %constant_7903 = f32[] constant(0.000195312503)
  %broadcast.14341 = f32[11160]{0} broadcast(%constant_7903), dimensions={}, metadata={scheduling_name="broadcast.14341"}
  %multiply.7090 = f32[11160]{0} multiply(%bitcast.76332, %broadcast.14341), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117 scheduling_name="multiply.7090"}
  %bitcast.76331 = f32[11160]{0} bitcast(%multiply.7090), metadata={scheduling_name="bitcast.76331"}
  %broadcast.14340 = f32[1,11160,5120]{2,1,0} broadcast(%bitcast.76331), dimensions={1}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/sub" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214 scheduling_name="broadcast.14340"}
  %subtract.534 = f32[1,11160,5120]{2,1,0} subtract(%add.5157.11, %broadcast.14340), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/sub" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214 scheduling_name="subtract.534"}
  %multiply.5901.15 = f32[1,11160,5120]{2,1,0} multiply(%subtract.534, %subtract.534), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/square" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=57 scheduling_name="multiply.5901.15"}
  %bitcast.54894.15 = f32[11160,5120]{1,0} bitcast(%multiply.5901.15), metadata={scheduling_name="bitcast.54894.15"}
  %reduce.885.15 = f32[11160]{0} reduce(%bitcast.54894.15, %constant_7904), dimensions={1}, to_apply=%region_1.1358, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117 scheduling_name="reduce.885.15"}
  %bitcast.54899.13 = f32[11160]{0} bitcast(%reduce.885.15), metadata={scheduling_name="bitcast.54899.13"}
  %multiply.5902.13 = f32[11160]{0} multiply(%bitcast.54899.13, %broadcast.14341), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117 scheduling_name="multiply.5902.13"}
  %constant_2767_6 = f32[] constant(1e-06)
  %broadcast.11345.36 = f32[11160]{0} broadcast(%constant_2767_6), dimensions={}, metadata={scheduling_name="broadcast.11345.36"}
  %add.5158.11 = f32[11160]{0} add(%multiply.5902.13, %broadcast.11345.36), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215 scheduling_name="add.5158.11"}
  %bitcast.54909.12 = f32[11160]{0} bitcast(%add.5158.11), metadata={scheduling_name="bitcast.54909.12"}
  %rsqrt.381.7 = f32[11160]{0} rsqrt(%bitcast.54909.12), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/rsqrt" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215 scheduling_name="rsqrt.381.7"}
  %bitcast.54912.19 = f32[11160]{0} bitcast(%rsqrt.381.7), metadata={scheduling_name="bitcast.54912.19"}
  %broadcast.11362.19 = f32[1,11160,5120]{2,1,0} broadcast(%bitcast.54912.19), dimensions={1}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217 scheduling_name="broadcast.11362.19"}
  %param_5.2297 = bf16[11160,5120]{1,0} parameter(5), metadata={scheduling_name="param_5.2297"}
  %bitcast.55544.19 = bf16[1,11160,5120]{2,1,0} bitcast(%param_5.2297), metadata={scheduling_name="bitcast.55544.19"}
  %convert.8083.19 = f32[1,11160,5120]{2,1,0} convert(%bitcast.55544.19), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=223 scheduling_name="convert.8083.19"}
  %multiply.5962.3 = f32[1,11160,5120]{2,1,0} multiply(%subtract.534, %convert.8083.19), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=219 scheduling_name="multiply.5962.3"}
  %multiply.6003.5 = f32[1,11160,5120]{2,1,0} multiply(%broadcast.11362.19, %multiply.5962.3), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217 scheduling_name="multiply.6003.5"}
  %broadcast.11430.11 = f32[1,11160,5120]{2,1,0} broadcast(%constant_7904), dimensions={}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/cross_attn/cross_attn.forward/q_norm/q_norm.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=48 scheduling_name="broadcast.11430.11"}
  %select.619.3 = f32[1,11160,5120]{2,1,0} select(%broadcast.11424.20, %multiply.6003.5, %broadcast.11430.11), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217 scheduling_name="select.619.3"}
  %bitcast.56458.1 = f32[93,120,5120]{2,1,0} bitcast(%select.619.3), metadata={scheduling_name="bitcast.56458.1"}
  ROOT %reduce.1203.1 = f32[93,5120]{1,0} reduce(%bitcast.56458.1, %constant_7904), dimensions={1}, to_apply=%region_1.1358, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217 scheduling_name="reduce.1203.1"}
}

ENTRY main {
  %param_0 = pred[] parameter(0)
  %param_1 = bf16[6,5120]{1,0} parameter(1)
  %param_2 = f32[1,11160,6,5120]{3,1,2,0} parameter(2)
  %param_3 = bf16[11160,5120]{1,0} parameter(3)
  %param_4 = bf16[1,11160,5120]{2,1,0} parameter(4)
  %param_5 = bf16[11160,5120]{1,0} parameter(5)
  ROOT %fusion = f32[93,5120]{1,0} fusion(%param_0, %param_1, %param_2, %param_3, %param_4, %param_5), kind=kCustom,
    calls=%fused_computation.1333, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1"]}],
        "num_warps":"8",
        "num_ctas":"1",
        "num_stages":"1"}}}
}
  )";
  auto module = Module(hlo_text, "");
  EXPECT_NE(TritonFusion(*module), nullptr);
  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
}


TEST_F(TritonFusionNumericsVerifierTest, SoftmaxDebugNVH100) {
  absl::string_view hlo_text = R"(
HloModule fused_computation.1242_standalone

%region_1.1358.clone.19 (Arg_0.109: f32[], Arg_1.109: f32[]) -> f32[] {
  %Arg_0.109 = f32[] parameter(0), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum"}
  %Arg_1.109 = f32[] parameter(1), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum"}
  ROOT %add.2499.0 = f32[] add(%Arg_0.109, %Arg_1.109), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum" source_file="/workspace/k-diffusion-jax/t5x/layers_scalable.py" source_line=722}
}

%region_1.1358.clone.31 (Arg_0.121: f32[], Arg_1.121: f32[]) -> f32[] {
  %Arg_0.121 = f32[] parameter(0), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum"}
  %Arg_1.121 = f32[] parameter(1), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum"}
  ROOT %add.2515.0 = f32[] add(%Arg_0.121, %Arg_1.121), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum" source_file="/workspace/k-diffusion-jax/t5x/layers_scalable.py" source_line=722}
}

%region_1.1358.clone.32 (Arg_0.122: f32[], Arg_1.122: f32[]) -> f32[] {
  %Arg_0.122 = f32[] parameter(0), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum"}
  %Arg_1.122 = f32[] parameter(1), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum"}
  ROOT %add.2516.0 = f32[] add(%Arg_0.122, %Arg_1.122), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum" source_file="/workspace/k-diffusion-jax/t5x/layers_scalable.py" source_line=722}
}

%region_1.1358.clone.30 (Arg_0.120: f32[], Arg_1.120: f32[]) -> f32[] {
  %Arg_0.120 = f32[] parameter(0), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum"}
  %Arg_1.120 = f32[] parameter(1), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum"}
  ROOT %add.2513.0 = f32[] add(%Arg_0.120, %Arg_1.120), metadata={op_name="jit(train_step)/jit(main)/Transformer.encode/encoder/while/body/encoder/pre_attention_layer_norm/reduce_sum" source_file="/workspace/k-diffusion-jax/t5x/layers_scalable.py" source_line=722}
}

%fused_computation.1242 (param_0.4366: f32[1,11160,6,5120], param_1.4391: f32[1,11160,1], param_2.3040: bf16[11160,5120], param_3.1916: bf16[1,11160,5120], param_4.1406: f32[1,11160,5120], param_5.1139: bf16[6,5120], param_6.1021: bf16[5120], param_7.1027: f32[1,11160], param_8.684: f32[1,11160,5120]) -> (bf16[1,11160,5120], bf16[1,11160,5120]) {
  %param_3.1916 = bf16[1,11160,5120]{2,1,0} parameter(3)
  %param_4.1406 = f32[1,11160,5120]{2,1,0} parameter(4)
  %bitcast.12036.clone.1 = f32[11160,5120]{1,0} bitcast(%param_4.1406)
  %constant_7179_clone_1 = f32[] constant(0)
  %reduce.1451.clone.1 = f32[11160]{0} reduce(%bitcast.12036.clone.1, %constant_7179_clone_1), dimensions={1}, to_apply=%region_1.1358.clone.19, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %bitcast.12035.clone.1 = f32[1,11160]{1,0} bitcast(%reduce.1451.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %constant_7178_clone_1 = f32[] constant(0.000195312503)
  %broadcast.8054.clone.1 = f32[1,11160]{1,0} broadcast(%constant_7178_clone_1), dimensions={}
  %multiply.3320.clone.1 = f32[1,11160]{1,0} multiply(%bitcast.12035.clone.1, %broadcast.8054.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %bitcast.12034.clone.1 = f32[11160]{0} bitcast(%multiply.3320.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %broadcast.8053.clone.1 = f32[1,11160,5120]{2,1,0} broadcast(%bitcast.12034.clone.1), dimensions={1}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/sub" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %subtract.334.clone.1 = f32[1,11160,5120]{2,1,0} subtract(%param_4.1406, %broadcast.8053.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/sub" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %param_8.684 = f32[1,11160,5120]{2,1,0} parameter(8)
  %param_6.1021 = bf16[5120]{0} parameter(6)
  %convert.452.10 = f32[5120]{0} convert(%param_6.1021), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %broadcast.3888.24 = f32[1,11160,5120]{2,1,0} broadcast(%convert.452.10), dimensions={2}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %multiply.1681.5 = f32[1,11160,5120]{2,1,0} multiply(%param_8.684, %broadcast.3888.24), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %bitcast.9479.3 = f32[11160,5120]{1,0} bitcast(%multiply.1681.5)
  %reduce.875.3 = f32[11160]{0} reduce(%bitcast.9479.3, %constant_7179_clone_1), dimensions={1}, to_apply=%region_1.1358.clone.30, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %bitcast.374.3 = f32[1,11160,1]{2,1,0} bitcast(%reduce.875.3), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %param_1.4391 = f32[1,11160,1]{2,1,0} parameter(1)
  %param_7.1027 = f32[1,11160]{1,0} parameter(7)
  %bitcast.316.6 = f32[1,11160,1]{2,1,0} bitcast(%param_7.1027), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215}
  %divide.340.5 = f32[1,11160,1]{2,1,0} divide(%param_1.4391, %bitcast.316.6), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215}
  %constant_2726_5 = f32[] constant(-0.5)
  %broadcast.3925.30 = f32[1,11160,1]{2,1,0} broadcast(%constant_2726_5), dimensions={}
  %multiply.1682.5 = f32[1,11160,1]{2,1,0} multiply(%divide.340.5, %broadcast.3925.30), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215}
  %multiply.1683.3 = f32[1,11160,1]{2,1,0} multiply(%bitcast.374.3, %multiply.1682.5), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215}
  %constant_2727_5 = f32[] constant(0.000390625)
  %broadcast.3926.8 = f32[1,11160,1]{2,1,0} broadcast(%constant_2727_5), dimensions={}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln2/ln2.forward/mul" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=57}
  %multiply.1684.3 = f32[1,11160,1]{2,1,0} multiply(%multiply.1683.3, %broadcast.3926.8), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=57}
  %bitcast.375.11.clone.1 = f32[11160]{0} bitcast(%multiply.1684.3), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=57}
  %broadcast.3955.11.clone.1 = f32[1,11160,5120]{2,1,0} broadcast(%bitcast.375.11.clone.1), dimensions={1}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=57}
  %multiply.1685.9.clone.1 = f32[1,11160,5120]{2,1,0} multiply(%subtract.334.clone.1, %broadcast.3955.11.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=57}
  %bitcast.9481.7 = f32[11160,5120]{1,0} bitcast(%multiply.1685.9.clone.1)
  %reduce.876.7 = f32[11160]{0} reduce(%bitcast.9481.7, %constant_7179_clone_1), dimensions={1}, to_apply=%region_1.1358.clone.31, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=141}
  %bitcast.9482.5 = f32[1,11160]{1,0} bitcast(%reduce.876.7), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=141}
  %negate.142.5 = f32[1,11160]{1,0} negate(%bitcast.9482.5), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/neg" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=141}
  %param_2.3040 = bf16[11160,5120]{1,0} parameter(2)
  %bitcast.373.19 = bf16[1,11160,5120]{2,1,0} bitcast(%param_2.3040), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/cross_attn/cross_attn.forward/q/q.forward/...b,ba->...a/dot_general" source_file="/workspace/k-diffusion-jax/szg_lib/nn/linear.py" source_line=65}
  %convert.1568.19 = f32[1,11160,5120]{2,1,0} convert(%bitcast.373.19), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=223}
  %bitcast.317.27 = f32[11160]{0} bitcast(%param_1.4391), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/rsqrt" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215}
  %broadcast.3886.27 = f32[1,11160,5120]{2,1,0} broadcast(%bitcast.317.27), dimensions={1}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %multiply.1611.19 = f32[1,11160,5120]{2,1,0} multiply(%broadcast.3886.27, %broadcast.3888.24), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %multiply.1686.11 = f32[1,11160,5120]{2,1,0} multiply(%convert.1568.19, %multiply.1611.19), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=219}
  %bitcast.9483.5 = f32[11160,5120]{1,0} bitcast(%multiply.1686.11)
  %reduce.877.5 = f32[11160]{0} reduce(%bitcast.9483.5, %constant_7179_clone_1), dimensions={1}, to_apply=%region_1.1358.clone.32, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %bitcast.9484.3 = f32[1,11160]{1,0} bitcast(%reduce.877.5), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %negate.143.3 = f32[1,11160]{1,0} negate(%bitcast.9484.3), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/neg" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %add.1089.3 = f32[1,11160]{1,0} add(%negate.142.5, %negate.143.3), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/add_any" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %multiply.1687.3 = f32[1,11160]{1,0} multiply(%add.1089.3, %broadcast.8054.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %bitcast.376.7.clone.1 = f32[11160]{0} bitcast(%multiply.1687.3), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %broadcast.3958.7.clone.1 = f32[1,11160,5120]{2,1,0} broadcast(%bitcast.376.7.clone.1), dimensions={1}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/broadcast_in_dim" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %add.1090.5.clone.1 = f32[1,11160,5120]{2,1,0} add(%multiply.1685.9.clone.1, %broadcast.3958.7.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/add_any" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %convert.482.3.clone.1 = bf16[1,11160,5120]{2,1,0} convert(%add.1090.5.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/convert_element_type" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=113}
  %add.1091.1.clone.1 = bf16[1,11160,5120]{2,1,0} add(%param_3.1916, %convert.482.3.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/add_any" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=113}
  %convert.483.3.clone.1 = bf16[1,11160,5120]{2,1,0} convert(%multiply.1686.11), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %add.1092.1.clone.1 = bf16[1,11160,5120]{2,1,0} add(%add.1091.1.clone.1, %convert.483.3.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/add_any" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %convert.484.4 = f32[1,11160,5120]{2,1,0} convert(%add.1092.1.clone.1), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=180}
  %param_0.4366 = f32[1,11160,6,5120]{3,1,2,0} parameter(0)
  %bitcast.2686.37 = f32[1,6,11160,5120]{3,2,1,0} bitcast(%param_0.4366)
  %param_5.1139 = bf16[6,5120]{1,0} parameter(5)
  %convert.432.7 = f32[6,5120]{1,0} convert(%param_5.1139), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=218}
  %broadcast.6560.37 = f32[1,6,11160,5120]{3,2,1,0} broadcast(%convert.432.7), dimensions={1,3}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=218}
  %add.2798.37 = f32[1,6,11160,5120]{3,2,1,0} add(%bitcast.2686.37, %broadcast.6560.37), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=218}
  %slice.957.3 = f32[1,1,11160,5120]{3,2,1,0} slice(%add.2798.37), slice={[0:1], [2:3], [0:11160], [0:5120]}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/slice" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=244}
  %bitcast.312.4 = f32[1,11160,5120]{2,1,0} bitcast(%slice.957.3), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/slice" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=244}
  %multiply.1690.3 = f32[1,11160,5120]{2,1,0} multiply(%convert.484.4, %bitcast.312.4), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=178}
  %convert.485.1 = bf16[1,11160,5120]{2,1,0} convert(%multiply.1690.3), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=141}
  ROOT %tuple.113 = (bf16[1,11160,5120]{2,1,0}, bf16[1,11160,5120]{2,1,0}) tuple(%convert.485.1, %add.1092.1.clone.1)
}

ENTRY main {
  %param_0 = f32[1,11160,6,5120]{3,1,2,0} parameter(0)
  %param_1 = f32[1,11160,1]{2,1,0} parameter(1)
  %param_2 = bf16[11160,5120]{1,0} parameter(2)
  %param_3 = bf16[1,11160,5120]{2,1,0} parameter(3)
  %param_4 = f32[1,11160,5120]{2,1,0} parameter(4)
  %param_5 = bf16[6,5120]{1,0} parameter(5)
  %param_6 = bf16[5120]{0} parameter(6)
  %param_7 = f32[1,11160]{1,0} parameter(7)
  %param_8 = f32[1,11160,5120]{2,1,0} parameter(8)
  ROOT %fusion = (bf16[1,11160,5120]{2,1,0}, bf16[1,11160,5120]{2,1,0}) fusion(%param_0, %param_1, %param_2, %param_3, %param_4, %param_5, %param_6, %param_7, %param_8), kind=kCustom,
  calls=%fused_computation.1242, backend_config={
    "fusion_backend_config":{
    "kind":"__triton",
    "block_level_fusion_config":{
      "output_tiles":[{"sizes":["1","2","5120"]},{"sizes":["1","2","5120"]}],
      "num_warps":"8",
      "num_ctas":"1",
      "num_stages":"1"}}}
}
  )";
  auto module = Module(hlo_text, "");
  EXPECT_NE(TritonFusion(*module), nullptr);
  auto verifier = TritonFusionNumericsVerifier(CreateAutotuneConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
}


}  // namespace
}  // namespace xla::gpu
