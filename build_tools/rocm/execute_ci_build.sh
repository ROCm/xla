#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(dirname $0)

EXCLUDED_TESTS=(
    HostMemoryAllocateTest.Numa          # numa tests will fail on rbe
)

TEST_TARGETS_SGPU=(
    //xla/...
    -//xla/backends/gpu/codegen/triton:dot_algorithms_test_amdgpu_any
    -//xla/backends/gpu/codegen/triton:fusion_emitter_parametrized_test_amdgpu_any
    -//xla/backends/gpu/codegen/triton:triton_gemm_fusion_test_amdgpu_any
    -//xla/backends/gpu/runtime:command_buffer_conversion_pass_test_amdgpu_any
    -//xla/backends/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any
    -//xla/tests:collective_ops_sharded_unsharded_e2e_test_amdgpu_any
    -//xla/backends/gpu/transforms:sort_rewriter_test_amdgpu_any
    -//xla/hlo/builder/lib:self_adjoint_eig_test_amdgpu_any
    -//xla/hlo/builder/lib:svd_test_amdgpu_any
    -//xla/service/gpu:dot_algorithm_support_test_amdgpu_any
    -//xla/service/gpu:float_support_test_amdgpu_any
    -//xla/service/gpu/tests:gpu_cub_sort_test_amdgpu_any
    -//xla/service/gpu/tests:sorting_test_amdgpu_any
    -//xla/tests:iota_test_amdgpu_any
)

TEST_TARGETS_MGPU=(
    //xla/tests:collective_ops_e2e_test
    //xla/tests:collective_ops_test
    //xla/tests:collective_pipeline_parallelism_test
    //xla/tests:replicated_io_feed_test
    //xla/backends/gpu/collectives:gpu_clique_key_test
    //xla/backends/gpu/runtime:all_reduce_test
    //xla/service:collective_ops_utils_test
    //xla/service:collective_pipeliner_test
    //xla/service:collective_permute_cycle_test
    //xla/service:batched_gather_scatter_normalizer_test
    //xla/service:all_reduce_simplifier_test
    //xla/service:all_gather_simplifier_test
    //xla/service:reduce_scatter_decomposer_test
    //xla/service:reduce_scatter_reassociate_test
    //xla/service:reduce_scatter_combiner_test
    //xla/service:scatter_simplifier_test
    //xla/service:sharding_propagation_test
    //xla/service:sharding_remover_test
    //xla/service:p2p_schedule_preparation_test
    //xla/tools/multihost_hlo_runner:functional_hlo_runner_test
    //xla/pjrt/distributed:topology_util_test
    //xla/pjrt/distributed:client_server_test
)

TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh)
TEST_TARGETS=("//xla/...")
for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="" # in mgpu we have a standard set of tests from the xla_mgpu config
        TEST_TARGETS=("${TEST_TARGETS_MGPU[@]}")
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TEST_TARGETS=("${TEST_TARGETS_SGPU[@]}")
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu,-no_oss"
    fi
done

${SCRIPT_DIR}/run_xla_ci_build.sh \
    --config=rocm_ci \
    --config=rocm_rbe \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --execution_log_compact_file=execution_log.binpb.zst \
    --spawn_strategy=local \
    --repo_env=REMOTE_GPU_TESTING=1 \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx90a,gfx942 \
    --repo_env=TF_ROCM_RBE_SINGLE_GPU_POOL=linux_x64_gpu_gfx90a \
    --remote_download_outputs=minimal \
    --grpc_keepalive_time=30s \
    --test_sharding_strategy=disabled \
    --test_verbose_timeout_warnings \
    --test_filter=-$(IFS=: ; echo "${EXCLUDED_TESTS[*]}") \
    --curses=no \
    --color=yes \
    $@ \
    -- \
    "${TEST_TARGETS[@]}"
