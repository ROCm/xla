#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

# This script runs XLA unit tests on ROCm platform by selecting tests that are
# tagged with requires-gpu-amd

set -e
set -x

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
# JAX images provide amd-smi.
command -v amd-smi >/dev/null 2>&1 || { echo "ERROR: amd-smi is required"; exit 1; }
amd-smi list
TF_GPU_COUNT=$(amd-smi list | grep -cE '^GPU:' || true)
if [ "${TF_GPU_COUNT:-0}" -eq 0 ]; then
  echo "ERROR: amd-smi list reported no GPUs"
  exit 1
fi
TF_TESTS_PER_GPU=1
N_TEST_JOBS=$(expr ${TF_GPU_COUNT} \* ${TF_TESTS_PER_GPU})
amdgpuname=(`rocminfo | grep gfx | head -n 1`)
AMD_GPU_GFX_ID=${amdgpuname[1]}
echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s) for gpu ${AMD_GPU_GFX_ID}."
echo ""

export ROCM_PATH=/opt/rocm

SCRIPT_DIR=$(realpath $(dirname $0))

$SCRIPT_DIR/run_xla_ci_build.sh \
    --config=rocm_ci \
    --config=ci_single_gpu \
    --local_test_jobs=${N_TEST_JOBS} \
    --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
    --keep_going \
    --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
    --action_env=TF_ROCM_AMDGPU_TARGETS=${AMD_GPU_GFX_ID} \
    --repo_env="ROCM_PATH=$ROCM_PATH" \
    -- \
    //xla/...
