#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(dirname $0)

TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh)
for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="" # in mgpu we have a standard set of tests from the xla_mgpu config
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu,-no_oss"
    fi
done

${SCRIPT_DIR}/run_xla_ci_build.sh \
    --config=rocm_ci \
    --config=rocm_rbe_dynamic \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
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
    $@
