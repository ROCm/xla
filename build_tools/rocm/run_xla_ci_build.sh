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

EXCLUDED_TESTS=(
    "F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f8e5m2_from_cc_8_9_rocm_63_no_restriction_c_32_nc_32"
    "F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f8e5m2_from_cc_8_9_rocm_63_no_restriction_c_16_nc_2"
    # Aligned with upstream openxla/xla ROCm CI EXCLUDED_TESTS: known
    # ROCm-unsupported / hipBLASLt-gap cases (e.g. f64 cublasLt + activation).
    "HostMemoryAllocateTest.Numa"
    "*IotaR1Test*"
    "NumericTestsForBlas/NumericTestsForBlas.Infinity/dot_tf32_tf32_f32_x3"
    "TritonAndBlasSupportForDifferentTensorSizes/TritonAndBlasSupportForDifferentTensorSizes.IsDotAlgorithmSupportedByTriton/dot_bf16_bf16_f32_x*"
    "F8E5M2Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_*"
    "DotOperationTestWithCublasLt_F16F32F64CF64/1.GeneralMatMulActivation"
    "MatmulTestWithCublas.GemmRewriter_RegressionTestF64"
    "TritonEmitterTest.ScaledDotIsSupportedByReferencePlatform"
    "SampleFileTest.Convolution"
)

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},requires-gpu-rocm,requires-gpu-amd,multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},requires-gpu-rocm,requires-gpu-amd,-multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_rocm_cpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-requires-gpu-rocm,-requires-gpu-amd"
    fi
done

bazel --bazelrc="$SCRIPT_DIR/rocm_xla_ci.bazelrc" test \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --profile=/tf/pkg/profile.json.gz \
    --nokeep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16" \
    --test_output=errors \
    --run_under=//build_tools/rocm:parallel_gpu_execute \
    --test_filter=-$(
        IFS=:
        echo "${EXCLUDED_TESTS[*]}"
    ) \
    --color=yes \
    "$@" \
    -- \
    //xla/... \
    -//xla/pjrt/gpu:se_gpu_pjrt_client_test_amdgpu_any \
    -//xla/tests:iota_test_amdgpu_any \
    -//xla/backends/gpu/codegen:dynamic_slice_fusion_test_amdgpu_any \
    -//xla/backends/gpu/tests:ragged_all_to_all_e2e_test_amdgpu_any \
    -//xla/tests:local_client_execute_test_amdgpu_any
