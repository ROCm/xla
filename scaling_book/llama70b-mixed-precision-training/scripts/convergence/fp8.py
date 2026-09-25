#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAXTEXT = Path(os.environ.get("MAXTEXT_ROOT", "/workspace/maxtext"))
CONFIG = REPO / "configs" / "llama70b.yml"
OUTPUT = Path(os.environ.get("OUTPUT_ROOT", "/tmp/llama70b"))
DATA = Path(os.environ.get("DATA_ROOT", str(REPO / "data")))
OUTPUT.mkdir(parents=True, exist_ok=True)

flags = " ".join(
    line.split("#", 1)[0].strip()
    for line in (REPO / "configs" / "flags" / "rocm.txt").read_text().splitlines()
    if line.split("#", 1)[0].strip()
)
env = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
env.update(
    {
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "JAX_PLATFORMS": "rocm",
        "XLA_FLAGS": flags,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97",
        "PYTHONPATH": f"{MAXTEXT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":"),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
)
recipe = [
    "dtype=bfloat16",
    "quantization=te_fp8_delayedscaling",
    "dataset_type=hf",
    "hf_path=json",
    f"hf_train_files={DATA / 'c4-en' / 'train'}/*.json.gz",
    "train_split=train",
    f"hf_eval_files={DATA / 'c4-en' / 'validation'}/*.json.gz",
    "hf_eval_split=train",
    f"tokenizer_path={DATA / 'tokenizer-llama2-hf'}",
    "reuse_example_batch=0",
    "enable_data_shuffling=true",
    "data_shuffle_seed=20260823",
    "packing=true",
    "max_segments_per_seq=32",
    "steps=2034",
    "learning_rate_schedule_steps=2034",
    "warmup_steps_fraction=0.05",
    "eval_start_step=0",
    "eval_interval=100",
    "eval_steps=20",
    "eval_per_device_batch_size=15",
    "target_eval_loss=0.",
    "enable_checkpointing=false",
    "async_checkpointing=false",
    "save_checkpoint_on_completion=false",
    f"base_output_directory={OUTPUT}",
    "run_name=convergence-fp8",
    f"metrics_file={OUTPUT / 'fp8-metrics.jsonl'}",
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
