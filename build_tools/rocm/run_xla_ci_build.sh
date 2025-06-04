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

CONFIG=$1
DISK_CACHE_PATH=$2

ASAN_ARGS=()
if [[ $CONFIG == "rocm_ci_hermetic" ]]; then
	ASAN_ARGS+=("--test_env=ASAN_OPTIONS=suppressions=$(realpath $(dirname $0))/asan_ignore_list.txt")
	ASAN_ARGS+=("--test_env=LSAN_OPTIONS=suppressions=$(realpath $(dirname $0))/lsan_ignore_list.txt")
	ASAN_ARGS+=("--config=asan")
fi

bazel \
    --bazelrc=/usertools/rocm.bazelrc \
    --output_base=/tmp/bzl \
    test \
    --config=${CONFIG} \
    --config=xla_cpp \
    --disk_cache=${DISK_CACHE_PATH} \
    --keep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --test_env=TF_GPU_COUNT=2 \
    --action_env=XLA_FLAGS=--xla_gpu_force_compilation_parallelism=16 \
    --action_env=XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=true \
    --test_output=errors \
    --local_test_jobs=2 \
    --run_under=//tools/ci_build/gpu_build:parallel_gpu_execute \
    "${ASAN_ARGS[@]}" \
    //xla/service:compiler_test_gpu_amd_any \
    //xla/service:elemental_ir_emitter_test_gpu_amd_any \
    //xla/service/gpu:gpu_compiler_test_gpu_amd_any \
    //xla/tests:matmul_test_gpu_amd_any \
    //xla/service/gpu/tests:kernel_launch_test_gpu_amd_any \
    //xla/stream_executor/gpu:gpu_kernel_test_gpu_amd_any \
    //xla/tests:client_test_gpu_amd_any \
    //xla/tests:convolution_test_gpu_amd_any

