#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "jax.yml"
OUTPUT = Path(os.environ.get("OUTPUT_ROOT", "/tmp/llama7b"))
OUTPUT.mkdir(parents=True, exist_ok=True)

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
        "NVTE_FUSED_ATTN_CK": "1",
        "NVTE_FUSED_ATTN_AOTRITON": "0",
        "XLA_FLAGS": flags,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    }
)
command = [
    sys.executable,
    str(REPO / "model" / "train.py"),
    str(CONFIG),
    f"base_output_directory={OUTPUT / 'remat-full'}",
    "dtype=bfloat16",
    "attention=te",
    "remat_policy=full",
    *sys.argv[1:],
]
raise SystemExit(subprocess.run(command, cwd=REPO, env=env, check=False).returncode)
