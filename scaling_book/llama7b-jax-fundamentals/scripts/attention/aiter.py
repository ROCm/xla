#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JAX_AITER = Path(os.environ.get("JAX_AITER_ROOT", "/workspace/jax-aiter-alpha2"))
CONFIG = REPO / "configs" / "jax.yml"
OUTPUT = Path(os.environ.get("OUTPUT_ROOT", "/tmp/llama7b"))
OUTPUT.mkdir(parents=True, exist_ok=True)

required = [
    JAX_AITER / "build" / "jax_aiter_build" / "libjax_aiter.so",
    JAX_AITER / "build" / "jax_aiter_build" / "mha_fwd_ja.so",
    JAX_AITER / "build" / "jax_aiter_build" / "mha_bwd_ja.so",
    JAX_AITER / "build" / "aiter_build" / "libmha_fwd.so",
    JAX_AITER / "build" / "aiter_build" / "libmha_bwd.so",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"JAX-AITER MHA setup is incomplete; missing: {', '.join(missing)}")

flags = " ".join(
    line.split("#", 1)[0].strip()
    for line in (REPO / "configs" / "flags" / "rocm.txt").read_text().splitlines()
    if line.split("#", 1)[0].strip()
)
env = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
env.update(
    {
        "HIP_VISIBLE_DEVICES": "0",
        "JAX_PLATFORMS": "rocm",
        "XLA_FLAGS": flags,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        "PYTHONPATH": f"{JAX_AITER}:{env.get('PYTHONPATH', '')}".rstrip(":"),
        "JA_ROOT_DIR": str(JAX_AITER),
        "AITER_ASM_DIR": str(JAX_AITER / "third_party" / "aiter" / "hsa"),
        "AITER_SYMBOL_VISIBLE": "1",
        "GPU_ARCHS": "gfx950",
    }
)
command = [
    sys.executable,
    str(REPO / "model" / "train.py"),
    str(CONFIG),
    f"base_output_directory={OUTPUT / 'attention-aiter'}",
    "dtype=bfloat16",
    "attention=aiter",
    "remat_policy=minimal_with_context",
    *sys.argv[1:],
]
raise SystemExit(subprocess.run(command, cwd=REPO, env=env, check=False).returncode)
