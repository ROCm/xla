# Mixtral 8x22B distributed strategies

- Docker: `docker.io/rocm/jax-training:maxtext-v26.6`.

## Requirements

- `docker.io/rocm/jax-training:maxtext-v26.6`
- Stock MaxText at `/workspace/maxtext`
- Sequence length 4096, global batch 64, 30 synthetic steps

Override paths with `MAXTEXT_ROOT` or `OUTPUT_ROOT`.

## Run

```bash
python3 scripts/baseline.py # FSDP1, EP8, fixed capacity MoE.
python3 scripts/mesh_fsdp8_ep1.py
python3 scripts/mesh_fsdp4_ep2.py
python3 scripts/mesh_fsdp2_ep4.py
python3 scripts/expert_dense_masked.py
python3 scripts/expert_dense_padded.py
python3 scripts/expert_ragged_dot.py
python3 scripts/overlap_lhs_off.py
```

Additional arguments are passed directly to MaxText, for example `steps=2`.

- `baseline.py`: FSDP-1, EP-8, fixed-capacity one-hot experts, latency hiding on.
- `mesh_*.py`: vary FSDP/EP while keeping expert execution fixed.
- `expert_dense_masked.py`: BF16 dropless dense-masked expert execution.
- `expert_ragged_dot.py` and `expert_dense_padded.py`: matched FP16 sparse-routing arms that differ only in GroupedGemm versus XLA's dense-padded lowering.
- Both sparse-routing arms retain one-shot ragged all-to-all and disable its unsupported RCCL device barrier.
- `overlap_lhs_off.py`: disable the latency-hiding scheduler at the baseline.

No v26.6 timings have been recorded yet.

## rocprof

```bash
NVTE_FRAMEWORK=jax /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3 \
  --kernel-trace -- python3 scripts/baseline.py
```
