# Llama 2 7B in JAX and MaxText

- Docker: `docker.io/rocm/jax-training:maxtext-v26.6`.
- Expect a total runtime of 30 minutes to run these experiments (8x MI355X node).

## Setup

Stock MaxText is read from `/workspace/maxtext`. Set `MAXTEXT_ROOT` or
`OUTPUT_ROOT` to override paths.

Align to JAX ROCm plugin/PJRT 0.11.0 and build JAX-AITER attention:

```bash
bash scripts/setup/setup_aiter.sh
```

## Raw JAX and MaxText

FP32 and BF16 use XLA attention in both implementations:

```bash
python3 scripts/precision/jax_fp32.py
python3 scripts/precision/jax_bf16.py
python3 scripts/precision/maxtext_fp32.py
python3 scripts/precision/maxtext_bf16.py
```

## Profiling

XProf is demonstrated on raw JAX.

```bash
python3 scripts/profile/xprof.py
```

Wrap the command for `rocprof`:

```bash
ROCPROF=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3
$ROCPROF --kernel-trace -- python3 scripts/precision/jax_bf16.py
$ROCPROF --kernel-trace --pmc "SQ_WAVES GRBM_GUI_ACTIVE" -- \
  python3 scripts/precision/jax_bf16.py
```

Note PMC timings are invalid as dispatches are serialized in order to collect hardware counters.

## Attention

Raw JAX, BF16, and `minimal_with_context` remat:

```bash
python3 scripts/attention/xla.py
python3 scripts/attention/te.py
python3 scripts/attention/aiter.py
python3 scripts/attention/triton.py
```

`triton.py` uses Tokamax Pallas-Triton. `aiter.py` calls JAX-AITER.

## Remat

Raw JAX BF16 using TE attention, on one GPU:

```bash
python3 scripts/remat/minimal_1gpu.py
python3 scripts/remat/full_1gpu.py
python3 scripts/remat/none_1gpu.py
```

The same three policies sharded across 8 GPUs, in MaxText with
`ici_fsdp_parallelism: 8` and TE attention:

```bash
python3 scripts/remat/minimal_fsdp8.py
python3 scripts/remat/full_fsdp8.py
python3 scripts/remat/none_fsdp8.py
```

`per_device_batch_size` is 4 in both, so the sharded runs have a global batch of 32
and their step times are comparable per device.
