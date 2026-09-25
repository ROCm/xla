#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAXTEXT = Path(os.environ.get("MAXTEXT_MXFP4_ROOT", "/workspace/maxtext-mxfp4"))
JAX_AITER = Path(os.environ.get("JAX_AITER_ROOT", "/workspace/jax-aiter-alpha2"))
CONFIG = REPO / "configs" / "llama70b.yml"
OUTPUT = Path(os.environ.get("OUTPUT_ROOT", "/tmp/llama70b"))
DATA = Path(os.environ.get("DATA_ROOT", str(REPO / "data")))
OUTPUT.mkdir(parents=True, exist_ok=True)
required = [
    MAXTEXT / "src" / "maxtext" / "layers" / "quantizations.py",
    JAX_AITER / "build" / "jax_aiter_build" / "libjax_aiter.so",
    JAX_AITER / "build" / "jax_aiter_build" / "gemm_fp4_ja.so",
    JAX_AITER / "build" / "jax_aiter_build" / "cast_mxfp4_ja.so",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"MXFP4 setup is incomplete; missing: {', '.join(missing)}")

flags = " ".join(
    line.split("#", 1)[0].strip()
    for line in (REPO / "configs" / "flags" / "rocm.txt").read_text().splitlines()
    if line.split("#", 1)[0].strip()
)
env = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
for key in tuple(env):
    if key.startswith(("AITER_FP4_", "JA_FP4_")) or key == "FP4_SELECT":
        env.pop(key)
env.update(
    {
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "JAX_PLATFORMS": "rocm",
        "XLA_FLAGS": flags,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97",
        "PYTHONPATH": (
            f"{MAXTEXT / 'src'}:{JAX_AITER}:{env.get('PYTHONPATH', '')}"
        ).rstrip(":"),
        "JA_ROOT_DIR": str(JAX_AITER),
        "AITER_ASM_DIR": str(JAX_AITER / "third_party" / "aiter" / "hsa"),
        "AITER_SYMBOL_VISIBLE": "1",
        "GPU_ARCHS": "gfx950",
        "AITER_FP4_MLP": "1",
        "AITER_FP4_ATTN": "1",
        "AITER_BF16_HIPBLASLT": "1",
        "JA_FP4_HADAMARD_PASSES": "wgrad",
        "JA_FP4_SR_PASSES": "wgrad_col",
        "JA_FP4_DGRAD_PARTITION": "gather_packed",
        "JA_FP4_DGRAD_REUSE_FWD_COL": "1",
        "JA_FP4_REMAT_SAVE_COL": "wt",
        "JA_FP4_PACK_GATEUP_AG": "0",
        "FP4_SELECT": "dispatch",
        "AITER_FP4_DISPATCH": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
)
recipe = [
    "dtype=bfloat16",
    "quantization=aiter_fp4",
    "use_jax_aiter=true",
    "aiter_attention=false",
    "aiter_rmsnorm=false",
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
    "run_name=convergence-mxfp4",
    f"metrics_file={OUTPUT / 'mxfp4-metrics.jsonl'}",
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
