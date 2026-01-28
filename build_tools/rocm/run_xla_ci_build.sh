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

# Split arguments: configs (before --) and targets (after --)
CONFIGS=()
TARGETS=()
TARGETS_TO_EXCLUDE=()
in_targets=false

for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        in_targets=true
    elif [[ "$in_targets" == true ]]; then
        TARGETS+=("$arg")
    else
        CONFIGS+=("$arg")
    fi
done

# Process config arguments
for arg in "${CONFIGS[@]}"; do
    case "$arg" in
        --config=asan)
            TAG_FILTERS="${TAG_FILTERS},-noasan"
            ;;
        --config=tsan)
            TAG_FILTERS="${TAG_FILTERS},-notsan"
            TARGETS_TO_EXCLUDE+=(-//xla/tests:iota_test_amdgpu_any -//xla/tests:iota_test_amdgpu_any_notfrt)
            ;;
        --config=ci_single_gpu)
            TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu,-no_oss"
            # Exclude multi-GPU targets (parsed from xla_mgpu in rocm_xla.bazelrc)
            while IFS= read -r t; do
                TARGETS_TO_EXCLUDE+=("-$t")
            done < <(grep -A 100 "^test:xla_mgpu" "$SCRIPT_DIR/rocm_xla.bazelrc" | grep "^//xla" | sed 's/ *\\$//')
            TARGETS_TO_EXCLUDE+=(-//xla/service/gpu/tests:gpu_cub_sort_test_amdgpu_any)
            ;;
        --config=ci_multi_gpu)
            # For multi-GPU: use xla_mgpu targets, ignore user-provided targets
            TARGETS=()
            CONFIGS+=(--config=xla_mgpu)
            ;;
    esac
done

set -x

bazel --bazelrc="$SCRIPT_DIR/rocm_xla.bazelrc" test \
    --disk_cache=${BAZEL_DISK_CACHE_DIR} \
    --config=rocm_rbe \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --test_timeout=920,2400,7200,9600 \
    --profile=/tf/pkg/profile.json.gz \
    --keep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16" \
    --test_output=errors \
    --local_test_jobs=4 \
    "${CONFIGS[@]}" \
    ${TARGETS:+-- "${TARGETS[@]}"} \
    "${TARGETS_TO_EXCLUDE[@]}" \
