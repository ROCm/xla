#!/bin/bash

if [ "$#" -eq 1 ]; then 
   TYPE="build"
else
   TYPE=$1
   shift 1
fi

BAZEL_CACHE=/tf/bazel_xla_cache
CLANG=$(realpath /opt/rocm/llvm/bin/clang)

bazel --output_base=$BAZEL_CACHE $TYPE \
        --config=rocm_clang_official \
        --action_env=TF_ROCM_AMDGPU_TARGETS=gfx942 \
        --test_env=TF_TESTS_PER_GPU=2 \
        --test_env=TF_PER_DEVICE_MEMORY_LIMIT_MB=0 \
        --copt="-Wno-c23-extensions" \
        --action_env=CLANG_COMPILER_PATH=$CLANG \
        --test_env=TF_GPU_COUNT=2 \
        --test_env=TF_CPP_MAX_VLOG_LEVEL=0 \
        --test_env=XLA_FLAGS="--xla_gpu_autotune_level=4" \
        --test_tag_filters=-oss_excluded,-oss_serial,-no_rocm,-tpu,-no_oss \
        --test_output=all \
        --cache_test_results=yes \
        --local_test_jobs=1 \
        --keep_going \
        --run_under=//tools/ci_build/gpu_build:parallel_gpu_execute -- \
        $@ $@"_gpu_amd_any" $@"_amd_gpu_any" $@"_amdgpu_any" 2>&1 | tee yyybuild.log
