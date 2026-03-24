#!/usr/bin/env bash
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

set -e
set -x

SCRIPT_DIR=$(realpath $(dirname $0))
TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh)

mkdir -p /tf/pkg

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu"
    fi
    if [[ "$arg" == "--config=asan" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-noasan"
    fi
    if [[ "$arg" == "--config=tsan" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-notsan"
    fi
done

TEST_FILTER=(
    F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f8e5m2_from_cc_8_9_rocm_63_no_restriction_c_32_nc_32
    F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f8e5m2_from_cc_8_9_rocm_63_no_restriction_c_16_nc_2
    DotBf16Bf16F32X6Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_bf16_bf16_f32_x6_with_lhs_f32_rhs_f32_output_f32_from_cc_8_0_rocm_60_no_restriction_c_16_nc_2
    DotBf16Bf16F32X6Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_bf16_bf16_f32_x6_with_lhs_f32_rhs_f32_output_f32_from_cc_8_0_rocm_60_no_restriction_c_32_nc_32
    DotBf16Bf16F32X9Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_bf16_bf16_f32_x9_with_lhs_f32_rhs_f32_output_f32_from_cc_8_0_rocm_60_no_restriction_c_32_nc_32
    DotBf16Bf16F32X9Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_bf16_bf16_f32_x9_with_lhs_f32_rhs_f32_output_f32_from_cc_8_0_rocm_60_no_restriction_c_16_nc_2
    CubScanThunkTest.ToProto
)

SCRIPT_DIR=$(dirname $0)
bazel --bazelrc="$SCRIPT_DIR/rocm_xla_ci.bazelrc" test \
    "$@" \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --profile=/tf/pkg/profile.json.gz \
    --keep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16" \
    --test_output=errors
