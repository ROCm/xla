#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAXTEXT = Path(os.environ.get("MAXTEXT_ROOT", "/workspace/maxtext"))
CONFIG = REPO / "configs" / "llama70b.yml"
OUTPUT = Path(os.environ.get("OUTPUT_ROOT", "/tmp/llama70b"))
OUTPUT.mkdir(parents=True, exist_ok=True)

flags = " ".join(
    line.split("#", 1)[0].strip()
    for line in (REPO / "configs" / "flags" / "rocm.txt").read_text().splitlines()
    if line.split("#", 1)[0].strip()
)
env = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
env.pop("NVTE_ROCM_USE_HIPBLASLT_MXFP8", None)
env.update(
    {
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "JAX_PLATFORMS": "rocm",
        "NVTE_ROCM_ENABLE_MXFP8": "1",
        "XLA_FLAGS": flags,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.94",
        "PYTHONPATH": f"{MAXTEXT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":"),
    }
)
probe = subprocess.run(
    [
        sys.executable,
        "-c",
        (
            "import importlib, inspect;"
            "m=importlib.import_module('transformer_engine.jax.cpp_extensions.gemm');"
            "f=getattr(m,'_get_gemm_workspace_size',None);"
            "assert f and 'rhs_scale_inv.size + 2 * lhs_scale_inv.size' "
            "in inspect.getsource(f)"
        ),
    ],
    env=env,
    capture_output=True,
    text=True,
    check=False,
)
if probe.returncode:
    raise SystemExit(
        "MXFP8 requires the patched TransformerEngine wheel; see README.md"
    )
command = [
    sys.executable,
    "-m",
    "maxtext.trainers.pre_train.train",
    str(CONFIG),
    f"base_output_directory={OUTPUT}",
    "run_name=train-step-mxfp8",
    "dtype=bfloat16",
    "quantization=te_mxfp8",
    *sys.argv[1:],
]
raise SystemExit(subprocess.run(command, cwd=REPO, env=env, check=False).returncode)
