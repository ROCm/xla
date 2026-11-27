# AMD ROCm Continuous Integration

This document explains how XLA is built and tested on AMD ROCm in CI: which jobs
run, which files define them, and the concepts behind them that the files do
not spell out. It links to the files instead of quoting them, so read the linked
file for exact flags, tag lists and versions.

## What runs and why

Three CI jobs cover ROCm. All of them run on pull requests against `main`.

| Job (GitHub check name)             | Workflow      | What it does            |
| ----------------------------------- | ------------- | ----------------------- |
| `XLA Linux x86 AMD Instinct GPU`    | `rocm_ci.yml` | Builds XLA and runs its |
:                                     :               : tests on AMD GPUs.      :
| `JAX Linux x86 AMD Instinct GPU`    | `rocm_ci.yml` | Builds JAX against the  |
:                                     :               : XLA from the PR and     :
:                                     :               : runs JAX's GPU tests.   :
| `XLA Linux x86 GPU Hermetic ROCm`   | `ci.yml`      | Compiles XLA for ROCm   |
:                                     :               : on a generic runner, no :
:                                     :               : tests.                  :

*   The **XLA job** runs two test phases, one after the other: a single-GPU
    phase (`ci_single_gpu`) for tests that need an AMD GPU, and a CPU phase
    (`ci_rocm_cpu`) for GPU-tagged tests that can run without one. Both phases
    call `build_tools/rocm/run_xla_ci_build.sh` directly.
*   The **JAX job** checks out `jax-ml/jax` and runs JAX's own ROCm RBE script
    with `--override_repository`/`--override_module` pointing JAX's `xla`
    dependency at the PR checkout, so it tests the PR's XLA rather than the
    version JAX pins. It only runs for PRs targeting `main`.
*   The **compile-only job** is defined in `build_tools/ci/build.py` like every
    other `ci.yml` build. It uses the hermetic ROCm toolchain and CPU-only RBE,
    so it catches ROCm build breaks even when the AMD GPU runners are down.

In `rocm_ci.yml`, a small `rocm-config` job runs first. It publishes the
pinned Docker image and runs `build_tools/ci/change_detector.py` to skip the XLA
or JAX job when a PR cannot affect it. The XLA job then runs
`build_tools/ci/bazel_diff.py` to restrict both test phases to the targets the
PR impacts (the label `ci:disable-bazel-diff` turns this off). The workflow can
also be started by hand (`workflow_dispatch`) or from another workflow
(`workflow_call`), which can override the runner label.

## Map of the files

*   [`.github/workflows/rocm_ci.yml`](https://github.com/openxla/xla/blob/main/.github/workflows/rocm_ci.yml):
    the XLA and JAX GPU jobs, the Docker image, the ROCm distribution and GPU
    target, and the Bazel configs shared by every XLA invocation.
*   [`build_tools/rocm/run_xla_ci_build.sh`](https://github.com/openxla/xla/blob/main/build_tools/rocm/run_xla_ci_build.sh):
    the driver the XLA job calls. It turns `--config=ci_*` arguments into tag
    filters, applies the bazel-diff target list if there is one, and runs
    `bazel test`.
*   [`build_tools/rocm/rocm_tag_filters.sh`](https://github.com/openxla/xla/blob/main/build_tools/rocm/rocm_tag_filters.sh):
    the base tag filter: tags that ROCm CI never runs (other vendors' GPUs,
    `no_oss`, and so on).
*   [`build_tools/rocm/rocm_xla_ci.bazelrc`](https://github.com/openxla/xla/blob/main/build_tools/rocm/rocm_xla_ci.bazelrc):
    the bazelrc the CI passes to Bazel. It imports the bazelrc below plus an
    optional one provided by the Docker image.
*   [`build_tools/rocm/rocm_xla.bazelrc`](https://github.com/openxla/xla/blob/main/build_tools/rocm/rocm_xla.bazelrc):
    the ROCm CI configs: `rocm_rbe_base`, `rocm_rbe` (EngFlow), `rocm_rbe_dev`
    (Buildbarn), `rocm_rbe_dynamic`, `ci_single_gpu`, `ci_multi_gpu`,
    `ci_rocm_cpu`, `no_flaky`, `asan`, `tsan`, and the `xla_sgpu`/`xla_mgpu`
    target lists.
*   [`tensorflow.bazelrc`](https://github.com/openxla/xla/blob/main/tensorflow.bazelrc):
    the general ROCm configs (`rocm_base`, `rocm_clang_hermetic`,
    `rocm_ci_hermetic`).
*   [`third_party/gpus/rocm_configure.bzl`](https://github.com/openxla/xla/blob/main/third_party/gpus/rocm_configure.bzl):
    the `@local_config_rocm` repository rule. It reads the `ROCM_*` and
    `TF_ROCM_*` repo env variables set by the workflow (hermetic ROCm download,
    GPU targets, RBE Docker image and pools) and defines the RBE execution
    platforms.
*   [`build_tools/rocm/BUILD`](https://github.com/openxla/xla/blob/main/build_tools/rocm/BUILD):
    the `sanitizer_wrapper` and GPU-lock (`parallel_gpu_execute`) targets used
    with `--run_under`, and the sanitizer ignore lists next to it.
*   [`build_tools/ci/build.py`](https://github.com/openxla/xla/blob/main/build_tools/ci/build.py):
    the compile-only job (`XLA_LINUX_X86_GPU_HERMETIC_ROCM_GITHUB_ACTIONS`).
    Its `rocm_tag_filter` is a hand-maintained copy of `rocm_tag_filters.sh`
    plus `gpu`, so keep the two in sync.
*   `build_tools/rocm/run_xla.sh` and `build_tools/rocm/run_xla_multi_gpu.sh`:
    helpers for running the single-GPU and multi-GPU suites by hand on a
    machine with local AMD GPUs. CI does not use them.

## Concepts

### How the test phases select tests

`run_xla_ci_build.sh` starts from the base filter in `rocm_tag_filters.sh` and
adds tags based on the `ci_*` config it is given. The same filter is used for
both `--build_tag_filters` and `--test_tag_filters`.

*   `ci_single_gpu` keeps tests tagged `requires-gpu-rocm` or
    `requires-gpu-amd` and drops `multi_gpu` tests.
*   `ci_rocm_cpu` keeps tests tagged `gpu` and drops those that need an AMD
    GPU. These tests exercise GPU code paths (codegen, passes, autotuner
    plumbing) without a device, so they run as ordinary CPU actions with high
    parallelism.
*   `ci_multi_gpu` keeps `multi_gpu` tests. No job in `rocm_ci.yml` runs it
    today.
*   `no_flaky` drops `flaky` tests, and `asan`/`tsan` drop `noasan`/`notsan`.

To change which tests run, change the tags on the test target, not the scripts.

### Where tests run: RBE

The GitHub runner does not run the GPU tests itself. Builds and tests go to a
remote build execution (RBE) cluster. Test targets tagged `requires-gpu-rocm` or
`requires-gpu-amd` get execution properties that send them to a GPU worker pool
(`TF_ROCM_RBE_SINGLE_GPU_POOL`, or the multi-GPU pool for `multi_gpu`); see
`tf_exec_properties` in `xla/tsl/platform/default/build_config_root.bzl`.

There are two RBE backends, both built on `rocm_rbe_base`:

*   **EngFlow** (`--config=rocm_rbe`): the backend the CI uses. Its endpoints
    and client certificate paths are in `rocm_xla.bazelrc`. Links are done
    locally because large link actions exceed EngFlow's limits.
*   **Buildbarn** (`--config=rocm_rbe_dev`): an AMD-hosted backend at
    `bb-rbe.amd.com` with its own execution platform (`linux_x64_bb`) and
    certificates. CI does not use it yet.

`rocm_rbe_dynamic` is a separate config, not part of `rocm_rbe`. It turns on
Bazel's dynamic execution, which races each action locally and remotely and
gives the remote side a head start. The XLA job combines it with `rocm_rbe`.

### Hermetic ROCm

The CI does not use a ROCm install on the runner. `rocm_ci_hermetic` makes
`@local_config_rocm` download the ROCm distribution named by `ROCM_DISTRO_URL`
and `ROCM_DISTRO_HASH` in `rocm_ci.yml`, and `ROCM_PATH=""` keeps a system ROCm
from being picked up. To bump ROCm, update those two values (and the Docker
image, if needed) in `rocm_ci.yml`.

### Sanitizers and the GPU lock

ASan and TSan builds need suppression files for ROCm libraries, and the tests
need to find them at run time. `//build_tools/rocm:sanitizer_wrapper` is a
`--run_under` wrapper (from `rules_ml_toolchain`) that adds the ignore lists and
`llvm-symbolizer` to the test's runfiles and sets the sanitizer options. Because
the ignore lists are inputs of the test action, changing them reruns the
affected tests. The `asan` and `tsan` configs use this wrapper; no job in
`rocm_ci.yml` runs them.

`parallel_gpu_execute.sh` is a `--run_under` wrapper for machines with several
local GPUs: it counts GPUs with `rocminfo`, takes a `flock` slot per GPU and
sets `HIP_VISIBLE_DEVICES` so tests do not share a GPU. None of the `ci_*`
configs and no script in `rocm_ci.yml` uses it any more, because GPU tests now
run on RBE workers. It is still referenced by the
`sanitizer_wrapper_parallel_gpu_execute` target in `build_tools/rocm/BUILD`,
which pipelines outside this repository can pass with `--run_under`.

## Reproducing a CI run locally

1.  Use the Docker image from the `rocm-config` job in `rocm_ci.yml`.
2.  Set the environment variables the XLA job uses (`DOCKER_IMAGE`,
    `ROCM_DISTRO_URL`, `ROCM_DISTRO_HASH`, `ROCM_AMD_GPU_TARGETS`,
    `ROCM_SINGLE_GPU_RBE_POOL`).
3.  Run `build_tools/rocm/run_xla_ci_build.sh` with the same arguments as the
    "Test XLA" step you want to reproduce, including the flags from
    `ROCM_COMMON_BAZEL_ARGS`, and replace `//xla/...` with the targets you care
    about.

Using `--config=rocm_rbe` requires the EngFlow client certificate, which only
the CI runners have. Without it, drop `rocm_rbe` and `rocm_rbe_dynamic` and run
on a machine with a local AMD GPU instead; `run_xla.sh` does this for the
single-GPU suite.

## Triage

*   **Invocation link.** Bazel prints a link to the RBE invocation
    (`bes_results_url` in `rocm_xla.bazelrc`) near the start of each test step.
    It has per-action logs, cache hits and the full output of each test.
*   **Failure annotations.** On failure, `build_tools/ci/bep_annotations.py`
    adds the failing targets to the job summary, and the Build Event Protocol
    files are uploaded as the `xla-bep` and `jax-bep` artifacts.
*   **Host information.** The `CPU Info` step prints `lscpu` at the top of
    each job log.
*   **Skipped jobs.** If a job ran on `ubuntu-latest` or skipped its test
    steps, the change detector or bazel-diff decided the PR does not affect
    it. Add the `ci:disable-bazel-diff` label to force a full run.
