#!/bin/bash

if [ "$#" -eq 1 ]; then 
   TYPE="build"
else
   TYPE=$1
   shift 1
fi

BAZEL_CACHE=/tf/bazel_xla_cache2
CLANG=$(realpath /opt/rocm/llvm/bin/clang)

bazel --output_base=$BAZEL_CACHE $TYPE \
        --config=rocm_clang_official \
        --repo_env=ROCM_PATH=$(realpath /opt/rocm) \
        --action_env=CLANG_COMPILER_PATH=$CLANG \
        --action_env=TF_ROCM_AMDGPU_TARGETS=gfx942,gfx950 \
        --repo_env=TF_ROCM_AMDGPU_TARGETS=gfx942,gfx950 \
        --copt="-Wno-c23-extensions" \
        --cache_test_results=yes \
        --local_test_jobs=1 \
        --keep_going \
        --run_under=//tools/ci_build/gpu_build:parallel_gpu_execute -- \
        $@ $@"_gpu_amd_any" $@"_amd_gpu_any" $@"_amdgpu_any" 2>&1 | tee yyybuild.log
