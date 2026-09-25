#!/usr/bin/env bash
set -euo pipefail

TE_ROOT="${TE_ROOT:-/workspace/TransformerEngine-mxfp8}"
BRANCH="experiment/v2.17-gfx950-mxfp8-workspace"

if [[ ! -d "$TE_ROOT/.git" ]]; then
  git clone --recursive --branch "$BRANCH" \
    https://github.com/clarkechong/TransformerEngine.git "$TE_ROOT"
fi

git -C "$TE_ROOT" checkout "$BRANCH"
git -C "$TE_ROOT" submodule update --init --recursive

export ROCM_PATH="${ROCM_PATH:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel}"
export HIP_DEVICE_LIB_PATH="$ROCM_PATH/llvm/amdgcn/bitcode"
export CMAKE_PREFIX_PATH="$ROCM_PATH/lib/cmake"
export NVTE_FRAMEWORK=jax
export NVTE_ROCM_ARCH=gfx950
export NVTE_USE_ROCM=1

rm -f "$TE_ROOT"/dist/transformer_engine-*.whl
python3 -m pip wheel "$TE_ROOT" --no-build-isolation --no-deps --wheel-dir "$TE_ROOT/dist"
python3 -m pip uninstall -y transformer_engine transformer_engine_rocm_jax transformer_engine_rocm7
python3 -m pip install --no-deps "$TE_ROOT"/dist/transformer_engine-*.whl
