#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$(realpath "$(dirname "$0")")


EXCLUDED_TESTS=()

EXCLUDED_TARGETS_SGPU=(
    "//xla/tests:iota_test_amdgpu_any"   # Taking too many CI nodes
    "//xla/backends/gpu/codegen/triton:dot_algorithms_test_amdgpu_any"
)

TEST_TARGETS_SGPU=(
    "//xla/..."
    "-//xla/tests:iota_test_amdgpu_any"   # Taking too many CI nodes
    "-//xla/backends/gpu/codegen/triton:dot_algorithms_test_amdgpu_any"
)

TEST_TARGETS_MGPU=(
    "//xla/tests:collective_ops_test"
    "//xla/backends/gpu/collectives:gpu_clique_key_test"
    "//xla/backends/gpu/runtime:all_reduce_test"
    "//xla/backends/gpu/runtime:collective_kernel_thunk_test"
    "//xla/backends/gpu/runtime:buffers_checksum_thunk_test"
    "//xla/backends/gpu/tests:collective_ops_command_buffer_test"
    "//xla/backends/gpu/tests:collective_pipeline_parallelism_test"
    "//xla/backends/gpu/tests:nccl_group_execution_test"
    "//xla/backends/gpu/tests:collective_ops_e2e_test"
    "//xla/backends/gpu/tests:collective_ops_ffi_test"
    "//xla/backends/gpu/tests:collective_ops_sharded_unsharded_e2e_test"
    "//xla/backends/gpu/tests:ragged_all_to_all_e2e_test"
    "//xla/backends/gpu/tests:replicated_io_feed_test"
    "//xla/backends/gpu/tests:all_reduce_e2e_test"
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
    "//xla/tools/multihost_hlo_runner:functional_hlo_runner_test"
    "//xla/pjrt/distributed:topology_util_test"
    "//xla/pjrt/distributed:client_server_test"
    "//xla/pjrt/extensions/cross_host_transfers:pjrt_c_api_cross_host_transfers_extension_gpu_test"
    "//xla/pjrt/gpu/tfrt:tfrt_gpu_client_test"
    "//xla/pjrt/gpu:se_gpu_pjrt_client_test")

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
