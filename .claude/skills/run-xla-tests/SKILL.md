---
name: run-xla-tests
description: Run XLA unit tests the way ROCm CI does with dev_tools/run_test.sh, locally on this machine or on the ROCm RBE pools (Buildbarn, EngFlow; MI350 GPUs), for a checkout, worktree or git ref; pick the tests a change affects and the CI config from their tags, and read the failures. Use when asked which tests to run for an XLA change, to run XLA tests locally, on a local GPU or on RBE, or to check a branch against ROCm CI before pushing.
argument-hint: "[--xla <dir|ref>] [--config=rocm_rbe_dev|rocm_rbe] [--config=ci_*] <bazel targets...>"
---

# Run XLA tests locally or on RBE

The script is `dev_tools/run_test.sh`, three directories up from this skill's
base directory. Call it `$S`. Its options, configs, ROCm settings and parallel
runs are documented in `dev_tools/README.md`; read it before the first run.

## Local or RBE

- Use what the user asked for. Otherwise use RBE (`--config=rocm_rbe_dev`): a
  local run builds XLA on this machine, which takes hours when cold.
- Where the user's instructions forbid builds here, do not run locally; where
  they forbid bazel, do not run at all (RBE also starts a local bazel client).
  Hand over the command instead.
- For RBE, check the backend's certificate files (listed in the README); if
  they are missing, stop and tell the user.
- A local GPU run tests this machine's GPU, not the CI's gfx950. Check the GPUs
  first (`rocminfo` or `amd-smi list`), and say which GPU it ran on. When
  several runs need GPUs at once, prefer RBE: local ones share the GPUs.

## Choose the tests

1. Find the change: the user's own commits in the checkout (`git log`,
   `git show --stat`); with a stale merge base, ignore the upstream files it
   drags in.
2. Direct tests: the `*_test` targets in the `BUILD` package of each changed
   file.
3. Dependent tests: grep `BUILD` files for the changed `cc_library` (for
   example `"//xla/backends/gpu/cost_model:coalescing_analysis"` or
   `":coalescing_analysis"`), then for its dependents, and take their tests.
   Stop at generic dependents (compiler, pipeline); CI covers those.
4. Drop tests that cannot see the change. For ROCm-only behavior (warp size
   64, gfx predicates), a test on a fixed CUDA device description
   (`TestGpuDeviceInfo::RTXA6000DeviceInfo()`, `H100SXMDeviceInfo()`, ...)
   does not change; one on `AMDMI210/MI300/MI350DeviceInfo()`, or on the real
   device (`GetDeviceDescription()`) of an AMD GPU, does. Keep unaffected ones
   only as cheap sanity checks, and say which is which.

## Pick the CI config from the tags

A target that does not match the CI config is skipped silently. Check each
target's rule and tags in its `BUILD`:

| Target | Config |
|--------|--------|
| `xla_cc_test` tagged `gpu` (the macro adds no tags) | `--config=ci_rocm_cpu` (default) |
| `xla_cc_test` without the `gpu` tag | `--config=ci_rocm_untagged` |
| `xla_test(backends = ["gpu"])`, a suite of `<name>_<backend>` variants | `--config=ci_single_gpu` |
| tagged `multi_gpu` | `--config=ci_multi_gpu` |
| tagged `cuda-only`, `requires-gpu-nvidia`, `requires-gpu-sm*`, `no_oss`, `no_gpu` or `flaky` | never runs here; leave it out and say so |

Run each config as a separate command, one after the other. "All specified
test targets were excluded by filters" means the tags do not match the config.

## Run

```bash
$S --xla <dir|ref> --config=rocm_rbe_dev //xla/pkg:a_test //xla/pkg:b_test
$S --xla <dir|ref> --config=rocm_rbe_dev --config=ci_single_gpu //xla/pkg:c_test
```

- Useful extra arguments: `--keep_going` (the default stops at the first
  failure), `--test_filter=...`, `--nocache_test_results` (rerun a passing
  test), `--runs_per_test=N` (flakes), `--test_output=all`.
- Start it in the background and wait for the notification; do not poll. Do
  not edit the script or the tested sources while it runs.
- A second run in the same checkout fails with "another run is using the XLA
  checkout"; wait for the first. Other worktrees can run at the same time.

## Read the result

- The first line names the checkout, commit (`-dirty` if uncommitted
  changes), CI config, backend and ROCm tarball: check they are what you meant.
- Failing tests' logs are printed inline; on RBE the invocation link has all
  logs.
- A failure that also happens on the base revision is not caused by the
  change: rerun that test with `--xla <base ref>` before blaming it.
- RBE: `lost inputs with digests` means the remote cache evicted outputs;
  rerun, and if it persists add `--noremote_accept_cached`. TLS errors mean
  missing or stale certificates. A long `Waiting for build events upload`
  after the result is the BES endpoint; the result is final, and
  `--bes_upload_mode=fully_async` avoids the wait.
- Local: a GPU test failing with no device or HIP errors at startup means this
  machine lacks a usable GPU for that config; rerun on RBE.

## Report

Where it ran (local and which GPU, or the RBE backend), checkout and commit,
CI config(s), targets run with pass/fail, targets skipped by tags and why, and
for each failure the failing case, the relevant log lines and whether it
reproduces on the base revision.
