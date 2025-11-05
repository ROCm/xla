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
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class TritonFusionNumericsVerifierTest
    : public HloPjRtTestBase,
      public ::testing::WithParamInterface<PrimitiveType> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    auto options = HloPjRtTestBase::GetDebugOptionsForTest();
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

  DeviceOrDevicelessConfig CreateDeviceOrDevicelessConfig() {
    se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
    auto executors_or = PlatformUtil::GetStreamExecutors(platform);
    TF_EXPECT_OK(executors_or);
    return DeviceOrDevicelessConfig{DeviceConfig{executors_or->at(0), nullptr}};
  }

  AutotunerCompileUtil CreateAutotunerCompileUtil(
      DeviceOrDevicelessConfig& config) {
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

TEST_P(TritonFusionNumericsVerifierTest, VerifyExactSoftmaxFusionNumerics) {
  auto module = Module(kSoftmaxHlo,
                       primitive_util::LowercasePrimitiveTypeName(GetParam()));

  EXPECT_NE(TritonFusion(*module), nullptr);
  auto verifier = TritonFusionNumericsVerifier(CreateDeviceOrDevicelessConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
}

TEST_P(TritonFusionNumericsVerifierTest, HugeHLOV71) {
  constexpr absl::string_view kHloText = R"(
HloModule m

%region_1.1358.clone.30 (Arg_0: f32[], Arg_1: f32[]) -> f32[] {
  %Arg_0 = f32[] parameter(0)
  %Arg_1 = f32[] parameter(1)
  ROOT %add = f32[] add(%Arg_0, %Arg_1)
}

%region_1.1358.clone.31 (Arg_0: f32[], Arg_1: f32[]) -> f32[] {
  %Arg_0 = f32[] parameter(0)
  %Arg_1 = f32[] parameter(1)
  ROOT %add = f32[] add(%Arg_0, %Arg_1)
}

%region_1.1358.clone.52 (Arg_0: f32[], Arg_1: f32[]) -> f32[] {
  %Arg_0 = f32[] parameter(0)
  %Arg_1 = f32[] parameter(1)
  ROOT %add = f32[] add(%Arg_0, %Arg_1)
}

%fused_computation.1337 (param_0.6629: pred[], param_1.6894: bf16[6,5120], param_2.4554: f32[1,11160,6,5120], param_3.3300: bf16[5120], param_4.2710: bf16[11160,5120], param_5.2438: bf16[1,11160,5120], param_6.2060: bf16[11160,5120]) -> f32[93,5120] {
  %param_0.6629 = pred[] parameter(0)
  %broadcast.6173.20 = pred[1,11160,5120]{2,1,0} broadcast(%param_0.6629), dimensions={}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/self_attn/self_attn.forward/custom_partitioning" source_file="/usr/local/lib/python3.10/dist-packages/transformer_engine/jax/attention.py" source_line=977}
  %param_5.2438 = bf16[1,11160,5120]{2,1,0} parameter(5)
  %convert.469.42 = f32[1,11160,5120]{2,1,0} convert(%param_5.2438), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=140}
  %param_4.2710 = bf16[11160,5120]{1,0} parameter(4)
  %convert.3999.17 = f32[11160,5120]{1,0} convert(%param_4.2710)
  %param_3.3300 = bf16[5120]{0} parameter(3)
  %convert.6198.13 = f32[5120]{0} convert(%param_3.3300)
  %broadcast.7705.19 = f32[11160,5120]{1,0} broadcast(%convert.6198.13), dimensions={1}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/self_attn/self_attn.forward/o/o.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/linear.py" source_line=75}
  %add.2270.19 = f32[11160,5120]{1,0} add(%convert.3999.17, %broadcast.7705.19), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/self_attn/self_attn.forward/o/o.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/linear.py" source_line=75}
  %convert.4001.17 = bf16[11160,5120]{1,0} convert(%add.2270.19)
  %bitcast.311.21 = bf16[1,11160,5120]{2,1,0} bitcast(%convert.4001.17), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/self_attn/self_attn.forward/o/o.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/linear.py" source_line=75}
  %convert.1597.21 = f32[1,11160,5120]{2,1,0} convert(%bitcast.311.21), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=141}
  %param_2.4554 = f32[1,11160,6,5120]{3,1,2,0} parameter(2)
  %bitcast.2788.63 = f32[1,6,11160,5120]{3,2,1,0} bitcast(%param_2.4554)
  %param_1.6894 = bf16[6,5120]{1,0} parameter(1)
  %convert.474.25 = f32[6,5120]{1,0} convert(%param_1.6894), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=218}
  %broadcast.6572.56 = f32[1,6,11160,5120]{3,2,1,0} broadcast(%convert.474.25), dimensions={1,3}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=218}
  %add.2836.63 = f32[1,6,11160,5120]{3,2,1,0} add(%bitcast.2788.63, %broadcast.6572.56), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=218}
  %slice.965.15 = f32[1,1,11160,5120]{3,2,1,0} slice(%add.2836.63), slice={[0:1], [2:3], [0:11160], [0:5120]}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/slice" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=244}
  %bitcast.312.16 = f32[1,11160,5120]{2,1,0} bitcast(%slice.965.15), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/slice" source_file="/workspace/k-diffusion-jax/szg_lib/nn/transformer.py" source_line=244}
  %multiply.1662.13 = f32[1,11160,5120]{2,1,0} multiply(%convert.1597.21, %bitcast.312.16), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=178}
  %add.1104.11 = f32[1,11160,5120]{2,1,0} add(%convert.469.42, %multiply.1662.13), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=178}
  %bitcast.10783 = f32[11160,5120]{1,0} bitcast(%add.1104.11)
  %constant_8003 = f32[] constant(0)
  %reduce.1513 = f32[11160]{0} reduce(%bitcast.10783, %constant_8003), dimensions={1}, to_apply=%region_1.1358.clone.30, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %bitcast.10782 = f32[1,11160]{1,0} bitcast(%reduce.1513), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %constant_8002 = f32[] constant(0.000195312503)
  %broadcast.8215 = f32[1,11160]{1,0} broadcast(%constant_8002), dimensions={}
  %multiply.3367 = f32[1,11160]{1,0} multiply(%bitcast.10782, %broadcast.8215), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %bitcast.10781 = f32[11160]{0} bitcast(%multiply.3367), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %broadcast.8213 = f32[1,11160,5120]{2,1,0} broadcast(%bitcast.10781), dimensions={1}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/sub" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %subtract.359 = f32[1,11160,5120]{2,1,0} subtract(%add.1104.11, %broadcast.8213), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/sub" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=214}
  %multiply.1664.11 = f32[1,11160,5120]{2,1,0} multiply(%subtract.359, %subtract.359), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/square" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=57}
  %bitcast.8184.11 = f32[11160,5120]{1,0} bitcast(%multiply.1664.11)
  %reduce.885.11 = f32[11160]{0} reduce(%bitcast.8184.11, %constant_8003), dimensions={1}, to_apply=%region_1.1358.clone.31, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %bitcast.8185.9 = f32[1,11160]{1,0} bitcast(%reduce.885.11), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %multiply.1665.9 = f32[1,11160]{1,0} multiply(%bitcast.8185.9, %broadcast.8215), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/div" source_file="/usr/local/lib/python3.10/dist-packages/flax/linen/normalization.py" source_line=117}
  %constant_2798_15 = f32[] constant(1e-06)
  %broadcast.3929.30 = f32[1,11160]{1,0} broadcast(%constant_2798_15), dimensions={}
  %add.1106.7 = f32[1,11160]{1,0} add(%multiply.1665.9, %broadcast.3929.30), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215}
  %bitcast.316.8 = f32[1,11160,1]{2,1,0} bitcast(%add.1106.7), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/add" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215}
  %rsqrt.229.3 = f32[1,11160,1]{2,1,0} rsqrt(%bitcast.316.8), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/rsqrt" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215}
  %bitcast.317.19 = f32[11160]{0} bitcast(%rsqrt.229.3), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/rsqrt" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=215}
  %broadcast.3977.19 = f32[1,11160,5120]{2,1,0} broadcast(%bitcast.317.19), dimensions={1}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/rematted_computation/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %param_6.2060 = bf16[11160,5120]{1,0} parameter(6)
  %bitcast.373.21 = bf16[1,11160,5120]{2,1,0} bitcast(%param_6.2060), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/cross_attn/cross_attn.forward/q/q.forward/...b,ba->...a/dot_general" source_file="/workspace/k-diffusion-jax/szg_lib/nn/linear.py" source_line=65}
  %convert.1599.21 = f32[1,11160,5120]{2,1,0} convert(%bitcast.373.21), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/convert_element_type" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=223}
  %multiply.1735.5 = f32[1,11160,5120]{2,1,0} multiply(%subtract.359, %convert.1599.21), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=219}
  %multiply.1786.5 = f32[1,11160,5120]{2,1,0} multiply(%broadcast.3977.19, %multiply.1735.5), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/mul" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %broadcast.4098.11 = f32[1,11160,5120]{2,1,0} broadcast(%constant_8003), dimensions={}, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/cross_attn/cross_attn.forward/q_norm/q_norm.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=48}
  %select.135.3 = f32[1,11160,5120]{2,1,0} select(%broadcast.6173.20, %multiply.1786.5, %broadcast.4098.11), metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
  %bitcast.8928.1 = f32[93,120,5120]{2,1,0} bitcast(%select.135.3)
  ROOT %reduce.1206.1 = f32[93,5120]{1,0} reduce(%bitcast.8928.1, %constant_8003), dimensions={1}, to_apply=%region_1.1358.clone.52, metadata={op_name="jit(train_step)/jit(main)/transpose(jvp(CausalWanXModel.loss_ode_regression))/CausalWanXModel.forward/transformer/transformer.forward/while/body/checkpoint/blocks/blocks.forward/ln3/ln3.forward/reduce_sum" source_file="/workspace/k-diffusion-jax/szg_lib/nn/norm.py" source_line=217}
}

ENTRY main {
  p0 = pred[] parameter(0)
  p1 = bf16[6,5120] parameter(1)
  p2 = f32[1,11160,6,5120] parameter(2)
  p3 = bf16[5120] parameter(3)
  p4 = bf16[11160,5120] parameter(4)
  p5 = bf16[1,11160,5120] parameter(5)
  p6 = bf16[11160,5120] parameter(6)
  ROOT fusion = f32[93,5120] fusion(p0, p1, p2, p3, p4, p5, p6), kind=kCustom,
    calls=%fused_computation.1337, backend_config={
      "fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1"]}],
        "num_warps":"8",
        "num_ctas":"1",
        "num_stages":"1"}}}
}
  
)";
  auto module = Module(kHloText,
                       primitive_util::LowercasePrimitiveTypeName(GetParam()));

  EXPECT_NE(TritonFusion(*module), nullptr);
  auto verifier = TritonFusionNumericsVerifier(CreateDeviceOrDevicelessConfig());
  TF_EXPECT_OK(verifier.Run(module.get(), /*execution_threads=*/{}));
}

INSTANTIATE_TEST_SUITE_P(TritonFusionNumericsVerifierTestSuite,
                         TritonFusionNumericsVerifierTest,
                         ::testing::Values(F32));
  
}  // namespace
}  // namespace xla::gpu
