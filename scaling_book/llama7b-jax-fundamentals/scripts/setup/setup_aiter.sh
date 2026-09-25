#!/usr/bin/env bash
set -euo pipefail

JAX_AITER_ROOT="${JAX_AITER_ROOT:-/workspace/jax-aiter-alpha2}"

python3 -m pip install \
  "jax==0.11.0" \
  "jaxlib==0.11.0" \
  "jax-rocm7-plugin==0.11.0" \
  "jax-rocm7-pjrt==0.11.0" \
  "tokamax==0.0.14"

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

bash "$JAX_AITER_ROOT/ci/fetch_jit_libs.sh"
test -s "$JAX_AITER_ROOT/build/aiter_build/libmha_fwd.so"
test -s "$JAX_AITER_ROOT/build/aiter_build/libmha_bwd.so"
make -C "$JAX_AITER_ROOT"
make -C "$JAX_AITER_ROOT" ja_mods

PYTHONPATH="$JAX_AITER_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -c "from jax_aiter.mha import flash_attn_func; print(flash_attn_func)"
