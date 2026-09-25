# Llama 2 7B experiment

- MI355X node
- Sequence 4096, batch 4 per device, 30 synthetic steps.
- Llama 2 7B: 32 layers, width 4096, MLP 11008, 32 heads.
- FP32 master weights and Adam moments.
- Scanned layers, no disk checkpointing.
- JAX/JAXLIB 0.11.0 and ROCm plugin/PJRT 0.11.0.

## Stage 1: implementation and precision

Compare raw JAX and MaxText in FP32 and BF16. Fix attention to XLA and remat to
`minimal_with_context`.

## Stage 2: attention

Compare raw-JAX BF16 XLA, TransformerEngine CK, direct JAX-AITER, and Tokamax
Pallas-Triton attention. Hold all other settings fixed.

## Stage 3: remat

Compare `minimal_with_context`, `full`, and `none` using raw-JAX BF16 with TE
attention fixed. Report median step time and peak memory.

## Stage 4: remat under FSDP

Repeat the remat experiments under `fsdp-8`. Under a multi-GPU setup, rematerialization of activations can influence collectives and affect the compute-memory tradeoff.

## Stage 5: profiling workflow

Use raw JAX/XLA for XProf, rocprof kernel tracing, and rocprof PMC collection.
Run timing and each profiler in separate processes; PMC timings are invalid.
