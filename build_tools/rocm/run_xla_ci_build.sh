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

SCRIPT_DIR=$(realpath $(dirname $0))
TAG_FILTERS=$($SCRIPT_DIR/rocm_tag_filters.sh),-skip_rocprofiler_sdk,-oss_excluded,-oss_serial
BAZEL_DISK_CACHE_DIR="/tf/disk_cache/rocm-jaxlib"
mkdir -p ${BAZEL_DISK_CACHE_DIR}
mkdir -p /tf/pkg

clean_up() {
    # clean up nccl- files
    rm -rf /dev/shm/nccl-*

    # clean up bazel disk_cache
    bazel shutdown \
        --disk_cache=${BAZEL_DISK_CACHE_DIR} \
        --experimental_disk_cache_gc_max_size=100G
}

trap clean_up EXIT

TARGETS_TO_EXCLUDE=()
EXTRA_CONFIGS=()
FILTERED_ARGS=()
IS_MULTI_GPU=false
AFTER_SEPARATOR=false

# Extract multi-GPU targets from xla_mgpu config in rocm_xla.bazelrc
# This is the single source of truth for multi-GPU test targets
get_mgpu_targets() {
    grep -A 100 "^test:xla_mgpu" "$SCRIPT_DIR/rocm_xla.bazelrc" | \
        grep "^//xla" | \
        sed 's/ *\\$//'
}

# First pass: detect if multi-GPU mode is requested
for arg in "$@"; do
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        IS_MULTI_GPU=true
        break
    fi
done

# Second pass: process arguments and filter targets for multi-GPU
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        AFTER_SEPARATOR=true
        # For multi-GPU, don't pass the separator (xla_mgpu provides targets)
        if [[ "$IS_MULTI_GPU" == false ]]; then
            FILTERED_ARGS+=("$arg")
        fi
        continue
    fi

    if [[ "$arg" == "--config=asan" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-noasan"
    fi
    if [[ "$arg" == "--config=tsan" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-notsan"
        TARGETS_TO_EXCLUDE+=(
            -//xla/tests:iota_test_amdgpu_any
            -//xla/tests:iota_test_amdgpu_any_notfrt
        )
    fi
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        # Use xla_mgpu config from rocm_xla.bazelrc for multi-GPU test targets
        EXTRA_CONFIGS+=(--config=xla_mgpu)
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu,-no_oss"
        # Exclude multi-GPU targets from single-GPU runs (generated from xla_mgpu)
        while IFS= read -r target; do
            TARGETS_TO_EXCLUDE+=("-$target")
        done < <(get_mgpu_targets)
        # Additional single-GPU specific exclusion
        TARGETS_TO_EXCLUDE+=(-//xla/service/gpu/tests:gpu_cub_sort_test_amdgpu_any)
    fi

    # For multi-GPU: filter out target arguments (after -- separator)
    # For single-GPU: keep all arguments
    if [[ "$IS_MULTI_GPU" == true && "$AFTER_SEPARATOR" == true ]]; then
        # Skip target arguments like //xla/... for multi-GPU (xla_mgpu provides targets)
        continue
    else
        FILTERED_ARGS+=("$arg")
    fi
done

set -x

bazel --bazelrc="$SCRIPT_DIR/rocm_xla.bazelrc" test \
    --disk_cache=${BAZEL_DISK_CACHE_DIR} \
    --config=rocm_rbe \
    --disk_cache=${BAZEL_DISK_CACHE_DIR} \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --test_timeout=920,2400,7200,9600 \
    --profile=/tf/pkg/profile.json.gz \
    --keep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16" \
    --test_output=errors \
    --local_test_jobs=4 \
    "${EXTRA_CONFIGS[@]}" \
    "${FILTERED_ARGS[@]}" \
    "${TARGETS_TO_EXCLUDE[@]}" \
