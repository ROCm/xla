#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")

EXCLUDED_TESTS=(
    "HostMemoryAllocateTest.Numa"                                                                                                                  # Failing on RBE
    "*IotaR1Test*"                                                                                                                                 # Taking too many CI nodes
    "HipblasLtMxExecutionTest*"
    "TritonBackendTest.CostModelOptions_Combination"
    "TritonBackendTest.CostModelOptions_Filter"
    "TritonBackendTest.CostModelOptions_TopFromDefaul"
    "NumericTestsForBlas/NumericTestsForBlas.Infinity/dot_tf32_tf32_f32_x3"
    "TritonAndBlasSupportForDifferentTensorSizes/TritonAndBlasSupportForDifferentTensorSizes.IsDotAlgorithmSupportedByTriton/dot_bf16_bf16_f32_x*"
    "DeterminismTest.CublasDot"
    "F8E5M2Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_*"
)

TAG_FILTERS=$("${SCRIPT_DIR}/rocm_tag_filters.sh")

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multi_gpu"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu"
    fi
done

"${SCRIPT_DIR}/run_xla_ci_build.sh" \
    "$@" \
    --build_tag_filters="$TAG_FILTERS" \
    --test_tag_filters="$TAG_FILTERS" \
    --execution_log_compact_file=execution_log.binpb.zst \
    --spawn_strategy=local \
    --repo_env=REMOTE_GPU_TESTING=1 \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx90a,gfx942,gfx950 \
    --remote_download_outputs=minimal \
    --grpc_keepalive_time=30s \
    --test_sharding_strategy=disabled \
    --test_verbose_timeout_warnings \
    --sandbox_add_mount_pair=/dev/null:/etc/ld.so.cache \
    --curses=no \
    --color=yes \
    --test_filter=-$(
        IFS=:
        echo "${EXCLUDED_TESTS[*]}"
    ) \
    --cache_test_results=yes \
    --keep_going \
    --repo_env=TF_ROCM_RBE_SINGLE_GPU_POOL=${RBE_POOL} \
    --repo_env=TF_ROCM_RBE_SINGLE_GPU_POOL=linux_x64_gpu_do_gfx950 \
    -- \
    //xla/... \
    -//xla/tests:dot_operation_test_amdgpu_any \
    -//xla/backends/gpu/autotuner:triton_test_amdgpu_any \
    -//xla/backends/gpu/transforms:gemm_rewriter_group_gemm_test_amdgpu_any 
    # TODO: skippped tests from https://wardite.cluster.engflow.com/invocations/default/f6e1d975-7f66-4b51-8430-d79e0ab0493a