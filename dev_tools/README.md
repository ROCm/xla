# Dev Tools

Day-to-day developer scripts for working on XLA with ROCm.

## `run_test.sh`

Runs `bazel test` the way ROCm CI does, through the checkout's
`build_tools/rocm/run_xla_ci_build.sh`: on this machine by default, or on the
ROCm RBE pools with a backend config.

```bash
# From an XLA checkout or worktree, on this machine:
../../dev_tools/run_test.sh //xla/backends/gpu/cost_model:coalescing_analysis_test

# On Buildbarn RBE, single-GPU tests:
../../dev_tools/run_test.sh --config=rocm_rbe_dev --config=ci_single_gpu //xla/backends/gpu/autotuner:fission_backend_test

# From this repo's root, for a git ref:
dev_tools/run_test.sh --xla rocm/main --config=rocm_rbe_dev //xla/backends/gpu/cost_model/...
```

`--xla` selects the checkout: a directory, used as is, or a git ref, tested in
the worktree [`xla_checkout.sh`](#xla_checkoutsh) creates for it. Without it,
the current directory, which must be the root of an XLA checkout. All other
arguments go to `bazel test` after the script's own, so they override them
(`--keep_going`, `--test_filter=...`, `--jobs=N`).

One CI config selects the tests by their tags:

| Config | Tests |
|--------|-------|
| `--config=ci_rocm_cpu` (default) | deviceless tests tagged `gpu`, as CI's `rocm_cpu` step |
| `--config=ci_rocm_untagged` | deviceless tests without the `gpu` tag, which no ROCm CI job runs (needs a checkout whose `run_xla_ci_build.sh` has it) |
| `--config=ci_single_gpu` | single-GPU tests |
| `--config=ci_multi_gpu` | multi-GPU tests |

One backend config, or none, selects where it runs:

- None: on this machine. A cold XLA build takes hours; GPU tests use its GPUs
  (four for `ci_multi_gpu`). Only the amdgpu driver is needed; the ROCm
  tarball is hermetic.
- `--config=rocm_rbe_dev`: Buildbarn. Needs `/data/bb-rbe-client.crt`,
  `/data/bb-rbe-client.key` and `/data/bb-rbe-ca.crt`.
- `--config=rocm_rbe`: EngFlow. Needs `/data/temp-rbe-cert.crt` and `.key`.

`--config=rocm_rbe_dynamic` can be added to either backend, as CI does.

The ROCm settings (tarball, RBE docker image, GPU target, single-GPU pool)
default to what the checkout's `.github/workflows/rocm_ci.yml` pins. Override
them with `--repo_env=<VAR>=<value>`; the variables are documented in the
checkout's `third_party/gpus/rocm_configure.bzl`.

### Parallel runs

Worktrees run in parallel, each with bazel's default output base (one per
workspace) and its own bazel server; downloads are shared through bazel's
repository cache. One run per worktree: a second stops with `another run is
using the XLA checkout ...`. The lock is `_hlo_eval_rbe/.lock` in the
checkout, shared with `perf_tools/hlo_eval_tools/rbe_hlo_eval.sh`. Local GPU
tests of parallel runs share this machine's GPUs.

After the ROCm tarball changes in a worktree, branches whose rules_ml_toolchain
predates its fix for this (`rocm-jaxlib-v0.12.1` and older) need
`bazel clean --expunge` there.

## `xla_checkout.sh`

Creates or reuses a worktree of this repo for an XLA ref and prints its path.

```bash
dev_tools/xla_checkout.sh <user>/my-fix        # worktrees/my-fix, on branch <user>/my-fix
dev_tools/xla_checkout.sh rocm/main            # worktrees/rocm-main, detached at rocm/main
dev_tools/xla_checkout.sh <sha> bisect-good    # worktrees/bisect-good, detached at <sha>
```

A local branch gets `worktrees/<last part of its name>` on the branch. A remote
ref (fetched first), tag or SHA gets `worktrees/<ref with / replaced by ->`,
detached. A second argument sets the name. An existing worktree is reused: one
on a branch is never moved; a detached one is moved to the new commit unless it
has local changes or a run is using it.
