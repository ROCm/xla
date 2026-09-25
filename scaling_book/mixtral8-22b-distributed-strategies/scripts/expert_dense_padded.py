#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAXTEXT = Path(os.environ.get("MAXTEXT_ROOT", "/workspace/maxtext"))
CONFIG = REPO / "configs" / "mixtral8-22b.yml"
FLAG_FILE = REPO / "configs" / "flags" / "rocm.txt"
OUTPUT = Path(os.environ.get("OUTPUT_ROOT", "/tmp/mixtral8-22b"))
OUTPUT.mkdir(parents=True, exist_ok=True)

flags = " ".join(
    line.split("#", 1)[0].strip()
    for line in FLAG_FILE.read_text().splitlines()
    if line.split("#", 1)[0].strip()
)
flags += (
    " --xla_gpu_enable_cublaslt=true"
    " --xla_gpu_experimental_use_ragged_dot_grouped_gemm=false"
    " --xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false"
    " --xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true"
)
env = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
env.update(
    {
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "JAX_PLATFORMS": "rocm",
        "XLA_FLAGS": flags,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97",
        "PYTHONPATH": f"{MAXTEXT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":"),
    }
)
recipe = [
    "dtype=float16",
    "quantization=",
    "ici_fsdp_parallelism=1",
    "ici_expert_parallelism=8",
    "sparse_matmul=true",
    "megablox=false",
    "capacity_factor=-1.0",
    "ragged_buffer_factor=-1.0",
    "moe_dispatch_no_expert_sharding=true",
    "num_experts=8",
    "num_experts_per_tok=2",
    "use_custom_sort_vjp=true",
    "use_ragged_sort=false",
    "use_ring_of_experts=false",
    "use_tokamax_gmm=false",
    f"base_output_directory={OUTPUT}",
    "run_name=expert-ragged-dot-dense-padded-fp16-fsdp1-ep8-lhs-on",
]
command = [
    sys.executable,
    "-m",
    "maxtext.trainers.pre_train.train",
    str(CONFIG),
    *recipe,
    *sys.argv[1:],
]
raise SystemExit(subprocess.run(command, cwd=REPO, env=env, check=False).returncode)
