#!/bin/bash

set -e

JAX_DIR=$1
XLA_DIR=$(pwd)

pushd $JAX_DIR

python build/build.py build \
        --wheels=jax-rocm-plugin \
        --configure_only \
        --local_xla_path=${XLA_DIR} \
        --python_version=3.12

bazel test \
    --config=rocm \
    --action_env=TF_ROCM_AMDGPU_TARGETS=gfx908 \
    --//jax:build_jaxlib=true \
    --test_output=streamed \
    "//tests/..."

popd
