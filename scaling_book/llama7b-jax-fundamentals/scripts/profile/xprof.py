#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "jax.yml"
OUTPUT = Path(os.environ.get("OUTPUT_ROOT", "/tmp/llama7b")) / "profile-xprof"
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
        "XLA_FLAGS": flags,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    }
)
command = [
    sys.executable,
    str(REPO / "model" / "train.py"),
    str(CONFIG),
    f"base_output_directory={OUTPUT}",
    "dtype=bfloat16",
    "attention=xla",
    "remat_policy=minimal_with_context",
    "profiler=xplane",
    "skip_first_n_steps_for_profiler=10",
    "profiler_steps=5",
    *sys.argv[1:],
]
raise SystemExit(subprocess.run(command, cwd=REPO, env=env, check=False).returncode)
