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

ASAN_ARGS=()
if [[ $CONFIG == "rocm_ci_hermetic" ]]; then
    ASAN_ARGS+=("--test_env=ASAN_OPTIONS=suppressions=$(realpath $(dirname $0))/asan_ignore_list.txt")
    ASAN_ARGS+=("--test_env=LSAN_OPTIONS=suppressions=$(realpath $(dirname $0))/lsan_ignore_list.txt")
    ASAN_ARGS+=("--config=asan")
fi

TAGS_FILTER="gpu,requires-gpu-amd,-requires-gpu-nvidia,-no_oss,-oss_excluded,-oss_serial,-no_gpu,-cuda-only"

bazel \
    --output_base=/tmp/bzl \
    test \
    --build_tag_filters=${TAGS_FILTER} \
    --test_tag_filters=${TAGS_FILTER} \
    --config=${CONFIG} \
    --disk_cache=/github/home/.cache/bazel_disk_cache \
    --experimental_disk_cache_gc_max_size=100G \
    --keep_going \
    --test_env=TF_TESTS_PER_GPU=1 \
    --test_env=TF_GPU_COUNT=2 \
    --action_env=XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16 --xla_gpu_enable_llvm_module_compilation_parallelism=true" \
    --test_output=errors \
    --local_test_jobs=2 \
    --run_under=//tools/ci_build/gpu_build:parallel_gpu_execute \
    "${ASAN_ARGS[@]}" \
    -- \
    //xla/... \
    -//xla/tests:grouped_convolution_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:support_test \
    -//xla/tests:convolution_test_gpu_alternative_layout_amdgpu_any \
    -//xla/tests:conv_depthwise_test_amdgpu_any \
    -//xla/tests:reshape_test_amdgpu_any \
    -//xla/tests:conv_depthwise_backprop_filter_test_amdgpu_any \
    -//xla/tests:convolution_test_amdgpu_any \
    -//xla/backends/gpu/codegen/emitters/tests:loop/broadcast_constant_block_dim_limit.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:reduce_row/mof_scalar_variadic.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:reduce_row/side_output_broadcast.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/multiple_roots.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_bf16.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_f16.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_multiple_heroes.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_multiple_roots.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_s16.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_s4.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_s8.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_side_output.hlo.test \
    -//xla/backends/gpu/codegen/emitters/tests:transpose/packed_transpose_two_heroes.hlo.test \
    -//xla/backends/gpu/codegen/triton:fusion_emitter_device_legacy_port_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:fusion_emitter_parametrized_test_amdgpu_any \
    -//xla/backends/gpu/codegen/triton:support_legacy_test_amdgpu_any \
    -//xla/backends/gpu/runtime:topk_test_amdgpu_any \
    -//xla/backends/profiler/gpu:cupti_error_manager_test_amdgpu_any \
    -//xla/pjrt/c:pjrt_c_api_gpu_test_amdgpu_any \
    -//xla/service/gpu/tests:command_buffer_test_amdgpu_any \
    -//xla/service/gpu/tests:dynamic_shared_memory_test_amdgpu_any \
    -//xla/service/gpu/tests:gpu_cub_sort_test_amdgpu_any \
    -//xla/service/gpu/tests:gpu_kernel_tiling_test_amdgpu_any \
    -//xla/service/gpu/tests:gpu_triton_custom_call_test_amdgpu_any \
    -//xla/service/gpu/tests:sorting_test_amdgpu_any \
    -//xla/service/gpu/transforms:triton_fusion_numerics_verifier_test_amdgpu_any \
    -//xla/tests:multioutput_fusion_test_amdgpu_any \
    -//xla/tools/hlo_opt:tests/gpu_hlo_llvm.hlo.test \
