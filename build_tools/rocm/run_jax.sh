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

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

pushd $TMP_DIR

git clone --single-branch --branch rocm-jaxlib-v0.6.0 https://github.com/ROCm/rocm-jax
pushd rocm-jax

if [[ $1 == "single_gpu" ]]; then
    python jax_rocm_plugin/build/rocm/run_single_gpu.py -c
else
    python jax_rocm_plugin/build/rocm/run_multi_gpu.py -c
fi

popd
popd
