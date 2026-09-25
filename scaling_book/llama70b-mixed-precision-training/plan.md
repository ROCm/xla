# Llama 2 70B experiment

## Train-step study

- Hardware: one node, 8× MI355X, FSDP-8.
- Model: MaxText `llama2-70b`.
- Sequence length: 4096.
- Global batch: 120 sequences, 491,520 tokens per step.
- Workload: 30 synthetic steps with a reused batch.
- Axis: FP32, FP16, BF16, FP8 delayed scaling, MXFP8 block scaling, and MXFP4.
- Timing baseline: BF16.

FP16, BF16, FP8 and MX formats keep FP32 master weights, gradients, and Adam moments. FP32 uses microbatch 1 with 15 accumulation steps and dot-product attention because TransformerEngine has no FP32 fused-attention backend.

MXFP8 requires the gfx950 TransformerEngine workspace patch. MXFP4 uses the ROCm MaxText feature branch and JAX-AITER alpha2; MLP, Q/K/V/O, and logits projections use MXFP4 while the fused attention core remains BF16.

## Convergence study

- Arms: BF16, FP16, FP8, MXFP8, MXFP4.
- Data: pinned local C4 shards with one verified Llama 2 tokenizer.
- Budget: 2,034 steps, approximately 1B tokens.
- Schedule: 5% warmup followed by cosine decay.
- Evaluation: 20 batches every 100 train steps.
- Comparison: validation loss and time to quality relative to BF16.
- Checkpointing: disabled.
