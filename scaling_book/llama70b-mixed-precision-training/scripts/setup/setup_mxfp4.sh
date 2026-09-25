#!/usr/bin/env bash
set -euo pipefail

MAXTEXT_MXFP4_ROOT="${MAXTEXT_MXFP4_ROOT:-/workspace/maxtext-mxfp4}"
JAX_AITER_ROOT="${JAX_AITER_ROOT:-/workspace/jax-aiter-alpha2}"

if [[ ! -d "$MAXTEXT_MXFP4_ROOT/.git" ]]; then
  git clone --branch feature/jax-aiter-mxfp4-v26.6 \
    https://github.com/ROCm/maxtext.git "$MAXTEXT_MXFP4_ROOT"
fi
git -C "$MAXTEXT_MXFP4_ROOT" checkout b437942a5f33704f8438deb948488ad08164285c

if [[ ! -d "$JAX_AITER_ROOT/.git" ]]; then
  git clone --recursive --branch release/v0.1.0-alpha2 \
    https://github.com/ROCm/jax-aiter.git "$JAX_AITER_ROOT"
fi
git -C "$JAX_AITER_ROOT" checkout 35b7175c763153ddb5da50c47d33dec436d5f191
git -C "$JAX_AITER_ROOT" submodule update --init --recursive
test "$(git -C "$JAX_AITER_ROOT/third_party/aiter" rev-parse HEAD)" = \
  31350226161346314b3d8882c8085bd31dce6a34

export JA_ROOT_DIR="$JAX_AITER_ROOT"
export GPU_ARCHS=gfx950
export AITER_SYMBOL_VISIBLE=1
make -C "$JAX_AITER_ROOT"
make -C "$JAX_AITER_ROOT" ja_mods_nomha
