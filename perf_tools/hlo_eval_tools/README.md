# HLO Eval Tools

Per-module microbenchmarks for XLA on ROCm. This tool takes the **pre-optimization
HLO dumps** produced by a real JAX or TF model, runs each HLO module on the GPU
through `multihost_hlo_runner`, and records the measured per-module execution
time. Accumulating those times into a CSV over many runs gives a clear picture of
**how much each HLO submodule of a model costs on XLA**, and lets you track that
cost across ROCm versions / XLA revisions.

It is built on the upstream `multihost_hlo_runner` tool and its CSV profiling
output (the `--append_profile_to_csv_file` flag), from [openxla/xla#42696](https://github.com/openxla/xla/pull/42696)

## What this measures (and what it does *not*)

These are **clean, per-kernel microbenchmarks** run one HLO module at a time. Each
number answers:

> "How fast is *this one HLO module* on *this GPU / this ROCm version*?"

That makes them excellent for regression tracking, per-op/per-fusion cost
attribution, and ROCm-version comparisons.

They are **not** end-to-end model latency. Because modules are replayed in
isolation, the numbers deliberately exclude:

- Python / framework dispatch overhead,
- host<->device transfer and input-pipeline effects seen in a live model,
- cross-module scheduling, overlap, and memory-pressure interactions,
- the full sequence/quantity of modules an actual inference or training step runs.

So do **not** read these to answer "how long does my model take." Use them to
answer "how fast is this module, and did it regress?"

> Compare with the sibling [`rocm_latency_check_tool`](../rocm_latency_check_tool/README.md),
> which is a synthetic HIP-runtime latency benchmark. This tool instead replays
> **real HLO modules** from real models.

## Where the HLO dumps come from

Run your JAX / TF model with XLA HLO dumping enabled and collect the
**unoptimized** (`before_optimizations`) HLO modules. A common way to dump is:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/hlo_dump --xla_dump_hlo_as_text" python my_model.py
```

For JAX you can also use `jax.jit(...).lower(...).compiler_ir("hlo")` /
`.as_text()` to write the module out. The tool consumes the **pre-optimization**
HLO so that the timing reflects the full XLA compile + execute path (XLA still
optimizes the module internally before running it), not a pre-optimized artifact.

## Directory layout

Keep one directory of HLO dumps per model. The intended set of model directories
is:

```
hlo_eval_tools/
  deepseek2_16b/
  gemma3_4b/
  gp3_oss_20b/
  llama3_8b/
  mixtral_8x7b/
  qwen3_14b/
  alphafold3/
  colabfold/
  stable_diffusion/
  resnet50/
  gpt_j_6b/
  flan_t5_large/
```

Each directory holds the `*.hlo` (or `*.txt` / proto) modules dumped from that
model. Grouping by model keeps the per-module CSV headers scoped to a single
model and makes cross-run comparison straightforward.

| Category | Models |
| --- | --- |
| Large language models | `deepseek2_16b`, `gemma3_4b`, `gp3_oss_20b`, `llama3_8b`, `mixtral_8x7b`, `qwen3_14b`, `gpt_j_6b`, `flan_t5_large` |
| Vision / diffusion | `stable_diffusion`, `resnet50` |
| Science | `alphafold3`, `colabfold` |

## Building `multihost_hlo_runner`

The tool ships with XLA and is built with Bazel from an XLA checkout. For ROCm:

```bash
bazel build -c opt --config=rocm \
  //xla/tools/multihost_hlo_runner:hlo_runner_main
```

The resulting binary is `bazel-bin/xla/tools/multihost_hlo_runner/hlo_runner_main`
(referred to below simply as `multihost_hlo_runner`).

## Usage

Profile every HLO module in one model directory and append the averaged timings
to a CSV. Note that the CSV path is given **without** the `.csv` extension — it is
added automatically:

```bash
./multihost_hlo_runner \
  --device_type=gpu \
  --profile_execution=true \
  --append_profile_to_csv_file=results/llama3_8b \
  hlo_eval_tools/llama3_8b/*.hlo
```

- `--profile_execution=true` enables execution profiling and per-module timing.
- `--append_profile_to_csv_file=<path>` redirects the averaged timings to
  `<path>.csv`. In a multi-node run the file name is prefixed with the task id so
  each node writes its own file (no collisions on a shared filesystem).
- The trailing positional arguments are the HLO files to run (a glob expands to
  one module per file).

Run the whole suite, one CSV per model:

```bash
for model in deepseek2_16b gemma3_4b gp3_oss_20b llama3_8b mixtral_8x7b \
             qwen3_14b alphafold3 colabfold stable_diffusion resnet50 gpt_j_6b flan_t5_large; do
  ./multihost_hlo_runner \
    --device_type=gpu \
    --profile_execution=true \
    --append_profile_to_csv_file=results/${model} \
    hlo_eval_tools/${model}/*.hlo
done
```

## CSV output

The CSV writer accumulates the per-file **averaged** execution time (the warmup
repeat is skipped) and writes it on process exit:

- **First run (new file):** writes a header row — a `Datetime` column followed by
  one column per HLO file. Common directory prefixes are stripped from the
  filenames to keep the header concise.
- **Subsequent runs (existing file):** appends one data row with the run
  timestamp and the timing values formatted to 4 significant figures (e.g.
  `3.142ms`).
- HLO files are kept in a stable sorted order so the header and every data row
  line up column-for-column.

Example (`results/llama3_8b.csv`):

```
Datetime           , all_gather.hlo, all_gather2.hlo, all_gather3.hlo, gemm.hlo, gemm_failed.hlo, transpose_large.hlo
2026-05-15 16:14:25, 8.284ms       , 8.328ms        , 8.343ms        , 2.508ms , 1.082ms        , 0.07167ms
2026-05-15 16:16:50, 8.296ms       , 8.325ms        , 8.313ms        , 2.518ms , 1.057ms        , 0.05767ms
```

Because each row is timestamped and appended to the same file, the CSV can be
loaded directly into pandas/Excel and pivoted by date (or annotated with the
ROCm/XLA version under test) to spot per-module regressions over time.

## Regression workflow

1. Dump the pre-optimization HLO for each model once and store it under the
   matching `hlo_eval_tools/<model>/` directory.
2. On each ROCm version / XLA revision you want to compare, build
   `multihost_hlo_runner` against it.
3. Run the suite with identical flags, appending into the **same** per-model CSVs.
4. Compare rows across timestamps (or versions) to see which HLO modules got
   faster or slower.
