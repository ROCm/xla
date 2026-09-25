# Mixtral 8x22B experiment

## Fixed recipe

- One node with 8× MI355X.
- MaxText `mixtral-8x22b`: 56 layers, 8 experts, top-2 routing.
- BF16 weights and compute, FP32 gradients. The matched GroupedGemm/dense-padded pair uses FP16 compute because gfx950 hipBLASLt GroupedGemm does not support BF16.
- Sequence length 4096.
- Microbatch 4 per device with two accumulation steps.
- Global batch 64 sequences, 262,144 tokens per update.
- Synthetic reused data and 30 steps.
- Shardy and TransformerEngine fused attention.

## Eight train-step cells

The baseline is FSDP-1, EP-8, fixed-capacity one-hot expert execution, and latency hiding enabled.

1. Baseline.
2. FSDP-8, EP-1.
3. FSDP-4, EP-2.
4. FSDP-2, EP-4.
5. Dropless dense-masked experts.
6. FP16 dropless `jax.lax.ragged_dot` lowered to hipBLASLt GroupedGemm.
7. The same FP16 sparse route lowered to dense-padded dots.
8. Baseline with latency hiding disabled.

Both sparse-routing cells keep one-shot token exchange and disable the NCCL device barrier unsupported by the pinned ROCm XLA/RCCL communicator.

Each cell is an independent timing script. Convergence and precision sweeps are out of scope.
