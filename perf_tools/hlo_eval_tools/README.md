# HLO Eval Tools

Per-module microbenchmarks for XLA on ROCm. This tool takes the **pre-optimization
HLO dumps** produced by a real JAX or TF model, runs each HLO module on the GPU
through `multihost_hlo_runner`, and records the measured per-module execution
time. Accumulating those times into a CSV over many runs gives a clear picture of
**how much each HLO submodule of a model costs on XLA**, and lets you track that
cost across ROCm versions / XLA revisions.

It covers **both training and inference workloads**. Training and inference emit
different HLO (the training step adds the backward pass, gradient/all-reduce, and
optimizer-update modules that inference never runs), so each model is split into a
`training/` and an `inference/` subdirectory of HLO dumps that are benchmarked
independently.

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

Dump the model **twice** — once in a training configuration (forward + backward +
optimizer step) and once in an inference/serving configuration (forward only) —
and store the resulting modules under the model's `training/` and `inference/`
subdirectories respectively.

> **Training HLO is dumped from a single node with 8 GPUs.** This is intentional
> and does not limit the tool's validity: the number of *nodes* does not change
> the per-submodule compute kernels, so the performance numbers this tool
> verifies hold regardless of how many nodes a real training run uses. For a
> given device count and sharding, the logical HLO is essentially the same across
> node counts — the only real difference is how the collectives are implemented
> and mapped onto the network fabric, not the compute modules being benchmarked
> here.

## Directory layout

Models are grouped into category folders; each model has a `training/` directory
and an `inference/` directory that is further split by device count
(`1gpu/`, `2gpu/`, `4gpu/`, `8gpu/`):

```
hlo_eval_tools/
  <category>/
    <model>/
      training/               # training step (fwd + bwd + optimizer)
      inference/
        1gpu/                 # single-device inference (no sharding)
        2gpu/  4gpu/  8gpu/   # multi-GPU inference (see "Multi-GPU inference")
```

The intended set of model directories is:

```
hlo_eval_tools/
  large_language_models/
    deepseek2_16b/{training,inference}/
    gemma3_4b/{training,inference}/
    gp3_oss_20b/{training,inference}/
    llama3_8b/{training,inference}/
    mixtral_8x7b/{training,inference}/
    qwen3_14b/{training,inference}/
    gpt_j_6b/{training,inference}/
    flan_t5_large/{training,inference}/
  vision_diffusion/
    resnet50/{training,inference}/
    imagenette/{training,inference}/
    efficientnet/{training,inference}/
    vit/{training,inference}/
    mlp_mixer/{training,inference}/
    clip/{training,inference}/
    detr/{training,inference}/
    stable_diffusion_1_5/{training,inference}/
    sdxl/{training,inference}/
    dit/{training,inference}/
  multimodal/
    paligemma/{training,inference}/
    siglip/{training,inference}/
    lit/{training,inference}/
    cappa/{training,inference}/
  science/
    alphafold3/{training,inference}/
    colabfold/{training,inference}/
```

Each leaf directory holds the `*.hlo` (or `*.txt` / proto) modules dumped from
that model in that workload. Grouping by model + workload keeps the per-module
CSV headers scoped to a single run and makes cross-run comparison straightforward.

| Category folder | Models |
| --- | --- |
| `large_language_models/` | `deepseek2_16b`, `gemma3_4b`, `gp3_oss_20b`, `llama3_8b`, `mixtral_8x7b`, `qwen3_14b`, `gpt_j_6b`, `flan_t5_large` |
| `vision_diffusion/` | `resnet50`, `imagenette`, `efficientnet`, `vit`, `mlp_mixer`, `clip`, `detr`, `stable_diffusion_1_5`, `sdxl`, `dit` |
| `multimodal/` | `paligemma`, `siglip`, `lit`, `cappa` |
| `science/` | `alphafold3`, `colabfold` |

The Vision-language / multimodal models (plus `vit` and `mlp_mixer`) are drawn
from Google Research's [`big_vision`](https://github.com/google-research/big_vision)
codebase, which covers a range of complementary op mixes: `vit` (pure attention +
GEMM), `mlp_mixer` (all-MLP, no attention/conv), `siglip` / `lit` (contrastive
dual-encoder image-text), `cappa` (image-captioning encoder-decoder), and
`paligemma` (a full vision-language generative model).

## Collection status

The table below reflects the HLO currently checked in under each model directory
(models are grouped on disk under the category folders listed above). All files
are pre-optimization (`before_optimizations`) HLO text modules. A *training*
module is the fused training step (forward + backward + optimizer); *inference*
modules are the forward/serving step.

The **Training** and **Inference** columns give the **GPU counts** each workload
was dumped at (not module counts). Training is dumped once per model at the count
shown; inference is dumped separately at each listed count under
`inference/<N>gpu/`.

Provenance: dumped with a from-source build of ROCm JAX/XLA (`jax 0.10.2` built
against the `rocm-jaxlib-v0.10.2` XLA branch) on a gfx950 node. Scope is
compilation-only — synthetic inputs and randomly-initialized weights are enough to
emit the `before_optimizations` HLO, so no trained checkpoints or datasets are
needed. The six MaxText LLMs' training is dumped on 8 GPUs (FSDP,
`num_partitions=8`); all other training is single-GPU. Multi-GPU inference uses
tensor-parallel for the MaxText LLMs and FSDP for everything else (see
[Multi-GPU inference](#multi-gpu-inference)).

| Model | Training (GPUs) | Inference (GPUs) | Source / notes |
| --- | :---: | :---: | --- |
| `llama3_8b` | 8 | 1 / 2 / 4 / 8 | MaxText; tensor-parallel; prefill + generate |
| `deepseek2_16b` | 8 | 1 / 2 / 4 / 8 | MaxText MLA; decode needs `mla_naive_kvcache=False` |
| `gemma3_4b` | 8 | 1 / 2 / 4 | MaxText; no 8-GPU (only 4 KV heads to shard) |
| `gp3_oss_20b` | 8 | 1 / 2 / 4 / 8 | MaxText (gpt-oss-20b) |
| `mixtral_8x7b` | 8 | 1 / 2 / 4 / 8 | MaxText; 1-GPU leaf has 2 prefill buckets + generate |
| `qwen3_14b` | 8 | 1 / 2 / 4 / 8 | MaxText |
| `gpt_j_6b` | 1 | 1 / 2 / 4 / 8 | HF `FlaxGPTJForCausalLM`; multi-GPU = FSDP |
| `flan_t5_large` | 1 | 1 / 2 / 4 / 8 | HF `FlaxT5ForConditionalGeneration`; multi-GPU = FSDP |
| `resnet50` | 1 | 1 / 2 / 4 / 8 | big_vision BiT ResNetV2-50 |
| `imagenette` | 1 | 1 / 2 / 4 / 8 | flax ResNet-18 (10-class); added beyond the canonical set |
| `efficientnet` | 1 | 1 / 2 / 4 / 8 | EfficientNet-B0 (flax implementation) |
| `vit` | 1 | 1 / 2 / 4 / 8 | big_vision ViT-B/16 |
| `mlp_mixer` | 1 | 1 / 2 / 4 / 8 | big_vision Mixer-B/16 |
| `clip` | 1 | 1 / 2 / 4 / 8 | big_vision `two_towers` (softmax contrastive) |
| `detr` | 1 | 1 / 2 / 4 / 8 | DETR (flax impl); Hungarian matching runs host-side |
| `stable_diffusion_1_5` | 1 | 1 / 2 / 4 / 8 | diffusers Flax SD v1.x UNet + pipeline |
| `sdxl` | 1 | 1 / 2 / 4 / 8 | diffusers Flax SDXL UNet |
| `dit` | 1 | 1 / 2 / 4 / 8 | DiT-B/2 (flax implementation) |
| `paligemma` | 1 | 1 / 2 / 4 / 8 | big_vision PaliGemma (ViT + Gemma-2B) |
| `siglip` | 1 | 1 / 2 / 4 / 8 | big_vision `two_towers` (sigmoid loss) |
| `lit` | 1 | 1 / 2 / 4 / 8 | big_vision `two_towers` (image tower locked) |
| `cappa` | 1 | 1 / 2 / 4 / 8 | big_vision CapPa (ViT encoder + AR decoder) |
| `alphafold3` | — | 1 / 2 / 4 / 8 | inference-only (no public training path) |
| `colabfold` | — | 1 / 2 / 4 / 8 | inference-only; AlphaFold2 single-sequence |

`—` in Training means the model has no public training path (dumped
inference-only). In total the tree holds 141 HLO modules across 24 models: 22
training + 119 inference (summed over the 1/2/4/8-GPU inference leaves). Each LLM
inference leaf holds its `prefill` + `generate` modules (mixtral's 1-GPU leaf: 2
prefill buckets + generate); every other inference leaf holds one forward module.

Known gaps: `alphafold3` / `colabfold` have no public training path (scoped
inference-only).
`efficientnet`, `dit`, and `detr` have no canonical JAX/Flax reference in the
source stack, so they use compact, architecturally-faithful flax implementations
(representative op mixes for benchmarking, not weight-identical to reference
checkpoints).

## Multi-GPU inference

Inference is dumped at 1, 2, 4 and 8 GPUs, one subdirectory per device count:
`<model>/inference/{1gpu,2gpu,4gpu,8gpu}/`. The `1gpu` modules are single-device
(no sharding). The `2gpu`/`4gpu`/`8gpu` modules are SPMD programs
(`num_partitions=N`) carrying the mesh + **Shardy** sharding annotations
(`xla.sdy.*`). Because the `before_optimizations` HLO is captured *before* the
partitioner runs, the cross-device collectives (all-gather / reduce-scatter) are
inserted by XLA at replay time inside `multihost_hlo_runner` — which is why
replaying them requires `--num_partitions=N --use_shardy_partitioner` (see
[Usage](#usage)); the checked-in module holds the shardings and `num_partitions=N`
that drive that partitioning.

Sharding strategy by model family:

- Large language models (MaxText: `llama3_8b`, `gemma3_4b`, `gp3_oss_20b`,
  `mixtral_8x7b`, `qwen3_14b`): tensor-parallel decode
  (`ici_tensor_parallelism=N`); each device count keeps the prefill + generate
  modules.
- All other models (vision / diffusion / multimodal, plus `gpt_j_6b` and
  `flan_t5_large`): FSDP - every parameter is sharded along its largest
  N-divisible axis across an `("fsdp",)` mesh; params/state with no N-divisible
  axis are replicated.
- `alphafold3` / `colabfold`: the haiku forward is FSDP-sharded and AOT-compiled
  under a mesh (no weights or execution needed) to emit the `num_partitions=N`
  module.

Multi-GPU coverage is complete (1/2/4/8 for every model) except:

- `gemma3_4b`: no `8gpu` (it has only 4 KV heads, which cannot be tensor-sharded
  across 8 devices; 1/2/4 are present).

`deepseek2_16b` uses MLA and requires `mla_naive_kvcache=False` at decode time
(the default naive cache mis-sizes the 192-wide MLA key against a 128-wide
buffer); with that flag it dumps cleanly at 1/2/4/8 GPUs.

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
  --num_partitions=8 --use_shardy_partitioner \
  --hlo_argument_mode=uninitialized \
  --profile_execution=true \
  --append_profile_to_csv_file=results/llama3_8b_inference_8gpu \
  hlo_eval_tools/large_language_models/llama3_8b/inference/8gpu/*.txt
```

(The collected modules are `*.before_optimizations.txt`; adjust the glob to
`*.hlo` if you add modules in that format.)

**Multi-GPU modules need explicit flags.** The runner takes the device/partition
count from `--num_partitions` (default 1), **not** from the module. So for any
module with `num_partitions=N > 1` — the `inference/<N>gpu/` leaves and the
8-GPU MaxText LLM `training/` modules — you must pass `--num_partitions=N
--use_shardy_partitioner` (and have N GPUs visible). These modules were lowered by
JAX with **Shardy**, so they contain `xla.sdy.*` custom calls: use
`--use_shardy_partitioner`, **not** `--use_spmd_partitioning` (the GSPMD path
`RET_CHECK`s on the `xla.sdy.*` calls). Omit the partition flags for the
single-device (`1gpu`, and the 1-GPU `training/`) modules.

`--hlo_argument_mode=uninitialized` is recommended for every run: it feeds
uninitialized argument buffers instead of generating random inputs, which greatly
reduces startup time for the large LLM / diffusion modules.

- `--profile_execution=true` enables execution profiling and per-module timing.
- `--append_profile_to_csv_file=<path>` redirects the averaged timings to
  `<path>.csv`. In a multi-node run the file name is prefixed with the task id so
  each node writes its own file (no collisions on a shared filesystem).
- The trailing positional arguments are the HLO files to run (a glob expands to
  one module per file).

### Automated: `run_hlo_eval.sh`

`run_hlo_eval.sh` (in this directory) packages everything below — leaf discovery,
per-module partition-count detection, the Shardy / uninitialized flags, device
selection and per-leaf CSV naming — behind three arguments:

```bash
./run_hlo_eval.sh <hlo_runner_main> <hlo_path> <out> [num_repeats]
```

- `<hlo_runner_main>` — the built binary.
- `<hlo_path>` — the `hlo_eval_tools` root, **or** any subtree (a single category
  or model dir), **or** a single leaf dir, **or** a single `.txt`/`.hlo` module.
  Every `training/` and `inference/<N>gpu/` leaf beneath it is profiled; empty
  leaves (e.g. `alphafold3/training`) are skipped.
- `<out>` — a **directory** (one CSV per leaf, e.g.
  `large_language_models_llama3_8b_inference_8gpu.csv`) **or** a path ending in
  `.csv` (all results append into that single file).
- `[num_repeats]` — executions per module (default 5; the warmup is skipped).

```bash
# Whole suite -> one CSV per model+workload under results/
./run_hlo_eval.sh /tf/xla/bazel-bin/xla/tools/multihost_hlo_runner/hlo_runner_main \
  hlo_eval_tools results

# Narrow the scope to one category, one model, or one leaf:
./run_hlo_eval.sh …/hlo_runner_main hlo_eval_tools/vision_diffusion results
./run_hlo_eval.sh …/hlo_runner_main hlo_eval_tools/large_language_models/llama3_8b results
./run_hlo_eval.sh …/hlo_runner_main hlo_eval_tools/large_language_models/llama3_8b/inference/8gpu results
```

For each leaf the script reads `num_partitions=N` from the first module and, when
`N>1`, adds `--num_partitions=N --use_shardy_partitioner` and sets
`HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES=0..N-1` (N GPUs must be visible). It
always passes `--hlo_argument_mode=uninitialized`, **disables HIP command buffers**
(see below), prints a summary of profiled / skipped / failed leaves, and exits
non-zero if any leaf failed (so you can re-run just those by pointing at the leaf).

Behavior is tunable with environment variables:

| Variable | Default | Effect |
|---|---|---|
| `CMD_BUFFER` | `off` | `off` adds `XLA_FLAGS=--xla_gpu_enable_command_buffer=`, disabling XLA's HIP command buffers (graphs). **Required on ROCm 7.2.4**: with them enabled, ~half the models `SIGSEGV` in `libamdhip64`'s graph launch (`RocmCommandBuffer::LaunchGraph`). Set `CMD_BUFFER=on` to re-enable. |
| `RESUME` | `0` | `1` skips leaves whose CSV already exists (resume an interrupted run / re-run only the leaves that lack a CSV). |
| `ORDER` | `size` | `size` profiles smallest-HLO leaves first (fast models first, biggest last); `path` = alphabetical. |
| `ARG_MODE` | `uninitialized` | The runner's `--hlo_argument_mode`. |
| `SETTLE_SEC` | `2` | Seconds paused between runner processes so GPU memory is reclaimed (helps back-to-back multi-GPU runs). |

### Manual equivalent

The script automates exactly this loop — one CSV per model **and** workload:

```bash
# Auto-discover every training and inference/<N>gpu leaf and write one CSV each,
# e.g. results/large_language_models_llama3_8b_inference_8gpu.csv.
# The partition count is read from each module's header (num_partitions=N), so
# this correctly handles both the multi-GPU inference leaves and the 8-GPU LLM
# training modules; N GPUs must be visible for the multi-GPU runs.
for leaf in hlo_eval_tools/*/*/training hlo_eval_tools/*/*/inference/*gpu; do
  files=("$leaf"/*.txt)
  # Skip empty leaf dirs (e.g. gemma3_4b/inference/8gpu).
  [ -e "${files[0]}" ] || continue
  n=$(grep -oE 'num_partitions=[0-9]+' "${files[0]}" | head -1 | cut -d= -f2)
  n=${n:-1}
  spmd=""; [ "$n" -gt 1 ] && spmd="--num_partitions=$n --use_shardy_partitioner"
  csv="results/$(echo "${leaf#hlo_eval_tools/}" | tr '/' '_')"
  ./multihost_hlo_runner \
    --device_type=gpu $spmd --hlo_argument_mode=uninitialized \
    --profile_execution=true \
    --append_profile_to_csv_file="$csv" \
    "${files[@]}"
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

Example — real output from profiling `llama3_8b` 8-GPU inference on gfx950 with
the built `hlo_runner_main`
(`results/large_language_models_llama3_8b_inference_8gpu.csv`):

```
Datetime,module_0035.jit__prefill_jit.before_optimizations.txt,module_0090.jit__generate_jit.before_optimizations.txt
2026-07-08 19:57:37, 222.6ms, 463.4ms
```

Each subsequent run appends another dated row with the same columns, so
re-profiling on a new ROCm / XLA build lets you diff the per-module timings over
time. As another real data point, the same `imagenette` (ResNet-18) inference
module measured **72.89ms on 1 GPU** vs **94.98ms on 2 GPUs** — for a model this
small the increase at higher device counts is FSDP all-gather / communication
overhead, which is exactly what this per-module timing is meant to surface.

Because each row is timestamped and appended to the same file, the CSV can be
loaded directly into pandas/Excel and pivoted by date (or annotated with the
ROCm/XLA version under test) to spot per-module regressions over time.

## Regression workflow

1. Dump the pre-optimization HLO for each model's training and inference
   workloads and store it under the matching
   `hlo_eval_tools/<category>/<model>/{training,inference}/` directory.
2. On each ROCm version / XLA revision you want to compare, build
   `multihost_hlo_runner` against it.
3. Run the suite with identical flags, appending into the **same** per-model CSVs.
4. Compare rows across timestamps (or versions) to see which HLO modules got
   faster or slower.

## Collected results

Profiling results for a given build are checked in under each model's
`results/` folder, one CSV per workload
(`<category>/<model>/results/{inference_1gpu,…,inference_8gpu,training}.csv`).
Every CSV starts with a provenance header recording the build it came from.

The currently checked-in results were produced on:

- **XLA build:** `rocm-jaxlib-v0.10.2` — [ROCm/xla @ `7b5ecf1c`](https://github.com/ROCm/xla/commit/7b5ecf1c9282fdf1039211e0d45216980058beda)
- **GPU:** MI350
- **Docker image:** `ghcr.io/rocm/jax-ubu22.rocm7.2.4`
- **Runner flags:** `--hlo_argument_mode=uninitialized --num_repeats=2`, HIP command buffers disabled

105 of 117 leaves are present; the 12 missing are the deterministic failures below
(`mixtral_8x7b` has no `results/` folder for this reason).

## Known build limitations (ROCm 7.2.4 / rocm-jaxlib-v0.10.2)

On the validated build, **105 of 117** non-empty leaves profile successfully. The
remaining 12 fail for reasons intrinsic to this ROCm/XLA build (not the tool), and
are **deterministic** — retrying (even in isolation) does not help:

- **`mixtral_8x7b`** (inference 1/2/4/8gpu + training) — `RESOURCE_EXHAUSTED`: the
  `jit__prefill_jit` op needs a single **173.9 GiB** allocation, which does not fit
  a device even when sharded.
- **8-GPU LLM `training`** (`deepseek2_16b`, `gemma3_4b`, `gp3_oss_20b`,
  `llama3_8b`, `qwen3_14b`) — `SIGSEGV` at execution.
- **`alphafold3` inference 2gpu / 8gpu** — `SIGSEGV` in
  `xla::gpu::CustomKernelThunk` (a custom GPU kernel inside the evoformer
  while-loops).

Two distinct backend bugs were found. The first — HIP command buffers crashing in
`libamdhip64` (`RocmCommandBuffer::LaunchGraph`) — is worked around by the default
`CMD_BUFFER=off` and recovered ~40 models. The remaining crashes above are a
**second, per-kernel** issue that needs an upstream ROCm/XLA fix rather than a
runner flag.

### Command buffers: which models require them OFF

The first bug is model-specific. With command buffers **enabled** (the XLA
default), these models `SIGSEGV` at execution and therefore require
`CMD_BUFFER=off` (i.e. `--xla_gpu_enable_command_buffer=`) — verified per model by
running `inference/1gpu` with command buffers on:

| Category | Models that **require command buffers OFF** |
|---|---|
| multimodal | `cappa`, `lit`, `paligemma`, `siglip` |
| vision_diffusion | `clip`, `detr`, `dit`, `mlp_mixer`, `sdxl`, `stable_diffusion_1_5`, `vit` |
| science | `alphafold3`, `colabfold` |
| large_language_models | `gpt_j_6b`, `flan_t5_large` |

These run fine **with command buffers ON** (the flag is not required, but the
default `CMD_BUFFER=off` is harmless for them):

| Category | Models that work with command buffers ON |
|---|---|
| large_language_models (MaxText) | `deepseek2_16b`, `gemma3_4b`, `gp3_oss_20b`, `llama3_8b`, `qwen3_14b` |
| vision_diffusion (Flax) | `resnet50`, `efficientnet`, `imagenette` |

`mixtral_8x7b` is unaffected by this flag — it hits the OOM above regardless.
Because the split is model-dependent and disabling command buffers is harmless for
the models that don't need it, `run_hlo_eval.sh` disables them **globally** by
default (`CMD_BUFFER=off`).
