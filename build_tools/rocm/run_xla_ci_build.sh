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
    "CollectivePipelineParallelismTestWithAndWithoutOpts/CollectivePipelineParallelismTest.PartiallyPipelinedAsyncSendRecvLoop/1"
    "CublasFissionBackendTest.CublasFallbackForBf16Bf16F32Algorithm"
    "TritonBackendTestSuite/TritonBackendTest.CostModelOptions_Filter/0"
    "TritonBackendTestSuite/TritonBackendTest.CostModelOptions_TopFromDefault/0"
    "TritonBackendTestSuite/TritonBackendTest.CostModelOptions_Filter/1"
    "TritonBackendTestSuite/TritonBackendTest.CostModelOptions_TopFromDefault/1"
    "TritonBackendTestSuite/TritonBackendTest.CostModelOptions_Combination/0"
    "TritonBackendTestSuite/TritonBackendTest.CostModelOptions_Combination/1"
    "GemmRewriteTest.CheckCustomCallHipblasLtBF16"
    "ParameterizedGemmRewriteTest.GemmTypeCombinationCheck"
    "FloatSupportTestWithCublas.MixedTypeDotIsNotUpcasted"
    "BlasAlgorithmTest.Algorithm_BF16_BF16_F32_X6"
    "BlasAlgorithmTest.Algorithm_BF16_BF16_F32_X3"
    "BlasAlgorithmTest.Algorithm_BF16_BF16_F32"
    "HostMemoryAllocateTest.Numa"
    "F8E5M2Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_with_lhs_f8e5m2_rhs_f8e4m3fn_output_f32_from_cc_8_9_rocm_63_no_restriction_c_32_nc_32"
    "F8E5M2Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e5m2_rhs_f8e4m3fn_output_f32_from_cc_8_9_rocm_63_no_restriction_c_16_nc_2"
    "F8E5M2Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_with_lhs_f8e5m2_rhs_f8e4m3fn_output_f32_from_cc_8_9_rocm_63_no_restriction_c_16_nc_2"
    "F8E5M2Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e5m2_rhs_f8e4m3fn_output_f32_from_cc_8_9_rocm_63_no_restriction_c_32_nc_32"
    "F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f32_from_cc_8_9_rocm_63_no_restriction_c_16_nc_2"
    "F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f32_from_cc_8_9_rocm_63_no_restriction_c_32_nc_32"
    "F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f32_from_cc_8_9_rocm_63_no_restriction_c_16_nc_2"
    "F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_fast_accum_with_lhs_f8e4m3fn_rhs_f8e4m3fn_output_f32_from_cc_8_9_rocm_63_no_restriction_c_32_nc_32"
    # mGPU tests
    "CollectiveOpsTestE2E.CollectiveGroupAllReduceDifferentReplicaGroups"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_2GPUs_*"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncRaggedAllToAll_2GPUs_BF16/sync"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncRaggedAllToAll_2GPUs_BF16/async"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncRaggedAllToAll_2GPUs_BF16/sync_symmetric"
    "AsyncCollectiveOps/AsyncCollectiveOps.AsyncRaggedAllToAll_2GPUs_BF16/async_symmetric"
    # Cherry-pick https://github.com/openxla/xla/pull/46650 once it is merged and remove tests below
    "GpuCollectivesTest.CreateRegisteredMemory"
    "GpuCollectivesTest.CreateSymmetricMemory"
    "GpuCollectivesTest.CreateWithMultipleIds"
    "GpuCollectivesTest.SplitCommunicators"
    "GpuCollectivesTest.PutAndWaitSignal"
    # Cherry-pick https://github.com/openxla/xla/pull/46832 once it is approved and merged and remove tests below
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_2GPUs/sync_nccl_private"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_SeveralOps_2GPUs/async_nccl_private"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_SeveralOps_2GPUs/async_nccl_peer"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_2GPUs/async_nccl_peer"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_SeveralOps_2GPUs/sync_nccl_private"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_2GPUs/async_nccl_private"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_SeveralOps_2GPUs/async_one_shot_with_multi_gpu_barrier_with_nccl_private"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_SeveralOps_2GPUs/sync_nccl_peer"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_2GPUs/async_one_shot_with_multi_gpu_barrier_with_nccl_private"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_SeveralOps_2GPUs/sync_one_shot_with_multi_gpu_barrier_with_nccl_private"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_2GPUs/sync_one_shot_with_multi_gpu_barrier_with_nccl_private"
    "RaggedAllToAllTest/RaggedAllToAllTest.RaggedAllToAll_2GPUs/sync_nccl_peer"

)

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu"
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
    --test_timeout=920,2400,7200,9600 \
    --profile=/tf/pkg/profile.json.gz \
    --test_sharding_strategy=disabled \
    --flaky_test_attempts=3 \
    --nokeep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16" \
    --test_output=errors \
    --run_under=//build_tools/rocm:parallel_gpu_execute \
    --test_filter=-$(
        IFS=:
        echo "${EXCLUDED_TESTS[*]}"
    ) \
    --color=yes \
    "$@" \
    //xla/...
