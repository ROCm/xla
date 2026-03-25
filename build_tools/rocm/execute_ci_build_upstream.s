#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(realpath "$(dirname "$0")")

TAG_FILTERS=$("$SCRIPT_DIR/rocm_tag_filters.sh")

# Temporary target skips for known upstream ROCm CI failures.
TEMPORARY_SKIPS=(
    "-//xla/backends/gpu/codegen/triton:dot_algorithms_test_amdgpu_any"
    "-//xla/backends/gpu/codegen/triton:fusion_emitter_int4_device_test_amdgpu_any"
    "-//xla/backends/gpu/transforms:gemm_rewriter_test_amdgpu_any"
    "-//xla/backends/gpu/transforms:sort_rewriter_test_amdgpu_any"
    "-//xla/backends/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any"
    "-//xla/hlo/builder/lib:self_adjoint_eig_test_amdgpu_any"
    "-//xla/hlo/builder/lib:svd_test_amdgpu_any"
    "-//xla/service:elemental_ir_emitter_test_amdgpu_any"
    "-//xla/service/gpu:dot_algorithm_support_test_amdgpu_any"
    "-//xla/service/gpu:float_support_test_amdgpu_any"
    "-//xla/service/gpu/tests:gpu_cub_sort_test_amdgpu_any"
    "-//xla/service/gpu/tests:sorting_test_amdgpu_any"
    "-//xla/tests:dot_operation_single_threaded_runtime_test_amdgpu_any"
    "-//xla/tests:dot_operation_test_amdgpu_any"
    "-//xla/backends/gpu/runtime:all_reduce_test_amdgpu_any"
    "-//xla/tests:collective_ops_e2e_test_amdgpu_any"
    "-//xla/tests:collective_ops_test_amdgpu_any"
    "-//xla/tools/multihost_hlo_runner:functional_hlo_runner_test_amdgpu_any"
)

for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="" # in mgpu we have a standard set of tests from the xla_mgpu config
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu,-no_oss"
    fi
done

"$SCRIPT_DIR/run_xla_ci_build.sh" \
    "$@" \
    --build_tag_filters="$TAG_FILTERS" \
    --test_tag_filters="$TAG_FILTERS" \
    --execution_log_compact_file=execution_log.binpb.zst \
    --spawn_strategy=local \
    --repo_env=REMOTE_GPU_TESTING=1 \
    --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx90a,gfx942 \
    --remote_download_outputs=minimal \
    --grpc_keepalive_time=30s \
    --test_sharding_strategy=disabled \
    --test_verbose_timeout_warnings \
    --curses=no \
    --color=yes \
    "${TEMPORARY_SKIPS[@]}"
