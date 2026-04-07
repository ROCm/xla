#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")

EXCLUDED_TESTS=(
    "*ParametersUsedByCollectiveMosaicShouldBeCopiedToCollectiveMemory"
    "SortingTest*"
    "*IotaR1Test*"
    "HostMemoryAllocateTest.Numa"
    "CubSort*"
)

EXCLUDED_TARGETS_SGPU=(
    "//xla/service/gpu:dot_algorithm_support_test_amdgpu_any"
    "//xla/service/gpu:float_support_test_amdgpu_any"
    "//xla/backends/gpu/transforms:scatter_determinism_expander_test_amdgpu_any"
    "//xla/backends/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any"
    "//xla/backends/gpu/codegen/triton:dot_algorithms_test_amdgpu_any"
)

TEST_TARGETS_SGPU=(
    "//xla/..."
    "-//xla/service/gpu:dot_algorithm_support_test_amdgpu_any"
    "-//xla/service/gpu:float_support_test_amdgpu_any"
    "-//xla/backends/gpu/transforms:scatter_determinism_expander_test_amdgpu_any"
    "-//xla/backends/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any"
    "-//xla/backends/gpu/codegen/triton:dot_algorithms_test_amdgpu_any"
)

TEST_TARGETS_MGPU=(
    "//xla/backends/gpu/tests:collective_pipeline_parallelism_test"
    "//xla/backends/gpu/collectives:gpu_clique_key_test"
    "//xla/service:collective_ops_utils_test"
    "//xla/service:collective_pipeliner_test"
    "//xla/service:collective_permute_cycle_test"
    "//xla/service:batched_gather_scatter_normalizer_test"
    "//xla/service:all_reduce_simplifier_test"
    "//xla/service:all_gather_simplifier_test"
    "//xla/service:reduce_scatter_decomposer_test"
    "//xla/service:reduce_scatter_reassociate_test"
    "//xla/service:reduce_scatter_combiner_test"
    "//xla/service:scatter_simplifier_test"
    "//xla/service:sharding_propagation_test"
    "//xla/service:sharding_remover_test"
    "//xla/service:p2p_schedule_preparation_test"
    "//xla/pjrt/distributed:topology_util_test"
    "//xla/pjrt/distributed:client_server_test"
)

TAG_FILTERS=$("${SCRIPT_DIR}/rocm_tag_filters.sh")
TEST_TARGETS=("${TEST_TARGETS_SGPU[@]}")
SGPU_AMDGPU_TARGETS="${TF_ROCM_SGPU_AMDGPU_TARGETS:-gfx90a,gfx942}"
MGPU_AMDGPU_TARGETS="${TF_ROCM_MGPU_AMDGPU_TARGETS:-gfx950}"
AMDGPU_TARGETS="${SGPU_AMDGPU_TARGETS}"

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS=""
        TEST_TARGETS=("${TEST_TARGETS_MGPU[@]}")
        AMDGPU_TARGETS="${MGPU_AMDGPU_TARGETS}"
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu,-no_oss"
        TEST_TARGETS=("${TEST_TARGETS_SGPU[@]}")
        AMDGPU_TARGETS="${SGPU_AMDGPU_TARGETS}"
    fi
done

"${SCRIPT_DIR}/run_xla_ci_build.sh" \
    "$@" \
    --build_tag_filters="$TAG_FILTERS" \
    --test_tag_filters="$TAG_FILTERS" \
    --execution_log_compact_file=execution_log.binpb.zst \
    --spawn_strategy=local \
    --repo_env=REMOTE_GPU_TESTING=1 \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=${AMDGPU_TARGETS} \
    --remote_download_outputs=minimal \
    --grpc_keepalive_time=30s \
    --test_sharding_strategy=disabled \
    --test_verbose_timeout_warnings \
    --curses=no \
    --color=yes \
    --test_filter=-$(
        IFS=:
        echo "${EXCLUDED_TESTS[*]}"
    ) \
    -- "${TEST_TARGETS[@]}"
