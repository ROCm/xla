---
name: run-hlo-perf-rbe
description: Run the hlo_eval_tools HLO perf suite (multihost_hlo_runner on the pre-optimization model modules) on the ROCm RBE GPU pools with rbe_hlo_eval.sh, read summary.tsv, and compare runs or an A/B between two XLA revisions. Use when asked to run perf, benchmark, measure or A/B XLA changes on RBE / Buildbarn / MI350, or to interpret hlo_eval_results.
argument-hint: [all | model[,model...]] [leaf...]
---

# Run the HLO perf suite on ROCm RBE

The tools live in `perf_tools/hlo_eval_tools/` of the xla-dev-infra checkout
that contains this skill (resolve symlinks: `realpath` of this skill's base
directory, then `../../../perf_tools/hlo_eval_tools`). Call that `$T`. Its
`README.md`, section "On ROCm RBE", lists every variable.

## Before running

- Pick the XLA revision to measure with `--xla`: a ref (`--xla rocm/main`,
  a branch or SHA) is checked out detached into `worktrees/<ref>` of the
  xla-dev-infra repo and reused; a directory (`--xla ~/xla`) is used as is.
  Without `--xla` the current directory must be an XLA checkout. Never switch
  branches or edit sources in the user's own checkouts without asking; prefer
  `--xla <ref>`, which only touches `worktrees/`.
- The Buildbarn client certificates must exist: `/data/bb-rbe-client.crt`,
  `/data/bb-rbe-client.key`, `/data/bb-rbe-ca.crt`. If they are missing, stop
  and tell the user; nothing else will work.
- Build and tests run remotely, but `bazel` still runs locally. If the user's
  instructions forbid bazel on this machine, ask first.

## Run

```bash
$T/rbe_hlo_eval.sh --xla rocm/main all -- --jobs=200          # 40 single-GPU leaves
$T/rbe_hlo_eval.sh --xla <ref> sdxl,vit inference/1gpu -- --jobs=200
```

- Models are bare names (`sdxl`, `llama3_8b`) or `all`; leaves are
  `inference/1gpu`, `training`, ... Only 1-GPU leaves run unless `MAX_GPUS=4`
  (multi-GPU pool unverified).
- The ROCm tarball defaults to what the checkout's `rocm_ci.yml` pins. When
  comparing runs, every run must use the same tarball: pass
  `ROCM_DISTRO_URL` and `ROCM_DISTRO_HASH` explicitly if in doubt.
- A full run takes 5-10 minutes when the build is cached and 30 minutes or
  more after a new XLA revision or tarball. Start it in the background and
  wait for it to finish; do not poll.
  While it runs, do not edit the scripts or the XLA sources, and do not start
  a second run from the same checkout (they share `_hlo_eval_rbe/` and
  `bazel-testlogs`).

## Read the results

The script prints the summary table and the results directory
(`hlo_eval_results/<xla commit>-<models>-<timestamp>/` under the directory it
was started from; `run_info.txt` records the commit, tarball and settings).
`summary.tsv` there has one
row per module: `median_ms` is the number to compare, `iqr_sigma_pct` is the
within-run spread in % of the median.

- `OK`: the median is trustworthy.
- `NOISY`: spread still above 2% after the automatic rerun. Inconclusive;
  leave it out of conclusions.
- `NO_DATA`: no timings; read the leaf's `test.log`. A common cause is that
  a large model does not fit in the device memory of these partitions
  (`RESOURCE_EXHAUSTED: Out of memory`): that is a capacity limit, not a
  regression, and repeats on every run.

## Compare runs and A/B

```bash
$T/hlo_eval_summarize.py <results dir A> <results dir B> [...]
```

- Judge on the suite geomean line, "without 3 largest". Measured A/A noise:
  identical runs agree within 0.25% there.
- Per module, run-to-run noise is large: median range 1.1%, 90th percentile
  4.4%. Do not report a single module's change below about 5% (beyond that
  90th percentile) as real unless it reproduces in repeated runs of both arms.
- `stable_diffusion_1_5` and `dit` inference, `clip` inference and
  `colabfold` settle at different levels from run to run (autotuning or host),
  so treat their differences as noise unless they are large and repeatable.
- For a real A/B, run each arm at least twice and alternate them (A, B, A, B).

## When it fails

- `lost inputs with digests`: Buildbarn evicted cached outputs. Rerun the same
  command; `bazel clean` does not help. If it persists, rerun once with
  `-- --noremote_accept_cached` (rebuilds everything remotely).
- `is not an XLA checkout`: run from an XLA checkout or pass `--xla <dir|ref>`.
- `has local changes`: the `worktrees/<ref>` checkout was edited; ask the user
  before discarding anything.
- TLS or certificate errors: the `/data/bb-rbe-*` files are missing or stale.

## Report

State the XLA revision(s), ROCm tarball, `REPEATS`, and results directories;
the suite geomean (and "without 3 largest"); modules beyond the noise bands
above; and any `NOISY` or `NO_DATA` leaves.
