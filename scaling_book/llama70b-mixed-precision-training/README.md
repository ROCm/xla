# Llama 2 70B mixed precision quantization with MaxText

- Docker: `docker.io/rocm/jax-training:maxtext-v26.6`.

## Requirements

- Docker: `docker.io/rocm/jax-training:maxtext-v26.6`
  - Stock MaxText `release/v26.6` at `b47d74bf` is under `/workspace/maxtext/`.
- MXFP8: build and install the [patched TransformerEngine 2.17 branch](https://github.com/clarkechong/TransformerEngine/tree/fix/jax-gfx950-mxfp8-workspace).
- MXFP4:
  - Clone the [ROCm MaxText feature branch](https://github.com/ROCm/maxtext/tree/feature/jax-aiter-mxfp4-v26.6) at `b437942a`.
  - Clone [JAX-AITER alpha2](https://github.com/ROCm/jax-aiter/tree/release/v0.1.0-alpha2) at `35b7175c`, initialize AITER at `31350226`, and build its FP4 FFI libraries.

Setup scripts are provided:

- MXFP8: `bash scripts/setup/install_te_mxfp8.sh`
- MXFP4: `bash scripts/setup/setup_mxfp4.sh`

You should double check the paths in the runner `.py` files match your installation setup.
You can override the filepaths with the environment variables:  `MAXTEXT_ROOT`, `MAXTEXT_MXFP4_ROOT`, `JAX_AITER_ROOT`, `DATA_ROOT`, or `OUTPUT_ROOT`.

## Train-step timing

Each run performs 30 total train steps. The first 10 steps are intended as JIT warmup runs.

```bash
python3 scripts/train_step/bf16.py
python3 scripts/train_step/fp16.py
python3 scripts/train_step/fp8.py
python3 scripts/train_step/mxfp8.py
python3 scripts/train_step/mxfp4.py
python3 scripts/train_step/fp32.py
```

Additional arguments are passed to MaxText, for example `steps=2`.

Prior v26.6 measurements:

| precision | seconds/step | status |
|---|---:|---|
| BF16 | 26.486 | stock |
| FP16 | 24.552 | stock |
| FP8 | 14.950 | stock |
| MXFP8 | 29.938 | unpatched fallback |
| MXFP4 | 11.692 | all dense projections |

## Convergence

Prepare the pinned C4 data and tokenizer once:

```bash
python3 scripts/setup/fetch_c4.py
python3 scripts/setup/make_hf_tokenizer.py
```

Then run one of `scripts/convergence/{bf16,fp16,fp8,mxfp8,mxfp4}.py`. Each file contains the complete 2,034-step recipe and writes MaxText metrics under `/tmp/llama70b`.

## rocprof

Wrap any timing file directly:

```bash
NVTE_FRAMEWORK=jax /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3 \
  --kernel-trace -- python3 scripts/train_step/mxfp8.py
```
