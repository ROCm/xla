#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAXTEXT = Path(os.environ.get("MAXTEXT_ROOT", "/workspace/maxtext"))
CONFIG = REPO / "configs" / "maxtext.yml"
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
        "XLA_FLAGS": flags,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        "PYTHONPATH": f"{MAXTEXT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":"),
    }
)
command = [
    sys.executable,
    "-m",
    "maxtext.trainers.pre_train.train",
    str(CONFIG),
    f"base_output_directory={OUTPUT}",
    "run_name=maxtext-bf16",
    "dtype=bfloat16",
    "weight_dtype=float32",
    "grad_dtype=float32",
    "mu_dtype=float32",
    "attention=dot_product",
    "remat_policy=minimal_with_context",
    *sys.argv[1:],
]
raise SystemExit(subprocess.run(command, cwd=REPO, env=env, check=False).returncode)
