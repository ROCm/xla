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
}

trap clean_up EXIT

TEST_FILTER=()

for arg in "$@"; do
    if [[ "$arg" == "--config=asan" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-noasan"
    fi
    if [[ "$arg" == "--config=tsan" ]]; then
        TAG_FILTERS="${TAG_FILTERS},-notsan"
    fi
    if [[ "$arg" == "--config=ci_multi_gpu" ]]; then
        TAG_FILTERS="" # in mgpu we have a standard set of tests
    fi
    if [[ "$arg" == "--config=ci_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multi_gpu,-no_oss"
    fi
    if [[ "$arg" == "--config=rocm_ci_hermetic" ]]; then
        TEST_FILTER+=(
            LegacyCublasGemmRewriteTest*
        )
    fi
done

set -x

bazel --bazelrc="$SCRIPT_DIR/rocm_xla.bazelrc" test \
    --build_tag_filters=$TAG_FILTERS \
    --test_tag_filters=$TAG_FILTERS \
    --test_timeout=920,2400,7200,9600 \
    --profile=/tf/pkg/profile.json.gz \
    --keep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --action_env=XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16" \
    --test_output=errors \
    --test_filter=-$(
        IFS=:
        echo "${TEST_FILTER[*]}"
    ) \
    --spawn_strategy=local \
    "$@" \
