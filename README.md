# ROCm/XLA Dev Infrastructure Branch
This is the `rocm-dev-infra` branch — the default branch of the [`ROCm/xla`](https://github.com/ROCm/xla) fork of [`openxla/xla`](https://github.com/openxla/xla).
It hosts GitHub Actions workflows, CI/CD infrastructure, Debug and Performance Tools, the JAX Scaling Book for AMD GPUs, and serves as a central index for all ROCm JAX/XLA release branches.

The **`main`** branch is an exact mirror of `openxla/xla:main` and should never
contain fork-specific commits. This separation ensures the `merge-upstream` API
(fast-forward only) continues to work for automatic syncing.

## Table of Contents

- [XLA Branches for JAX](#xla-branches-for-jax)
- [Recommended Docker Image && Build Instructions](#recommended-docker-image--build-instructions)
- [OpenXLA Upstream PR Status](#openxla-upstream-pr-status)
- [OpenXLA Upstream CI Checks](#openxla-upstream-ci-checks)
- [Debug Tools](#debug-tools)
- [Performance Tools](#performance-tools)
- [Scaling Book](#scaling-book)
- [Workflows](#workflows)

## XLA Branches for JAX

For supported ROCm versions and GFX targets per release, see the [AMD ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

| JAX Release | XLA Branch | JAX Branch |
|-------------|------------|------------|
| jax-ml/jax:main | [`openxla/xla:main`](https://github.com/openxla/xla) | [`jax-ml/jax:main`](https://github.com/jax-ml/jax) | 
| jax-v0.11.2 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.11.2`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.11.2) | [`rocm-jaxlib-v0.11.2`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.11.2) |
| jax-v0.11.1 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.11.1`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.11.1) | [`rocm-jaxlib-v0.11.1`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.11.1) |
| jax-v0.11.0 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.11.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.11.0) | [`rocm-jaxlib-v0.11.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.11.0) |
| jax-v0.10.2 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.10.2`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.10.2) | [`rocm-jaxlib-v0.10.2`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.10.2) |
| jax-v0.10.1 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.10.1`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.10.1) | [`rocm-jaxlib-v0.10.1`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.10.1) |
| jax-v0.10.0 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.10.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.10.0) | [`rocm-jaxlib-v0.10.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.10.0) |
| jax-v0.9.2 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.9.2`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.9.2) | [`rocm-jaxlib-v0.9.2`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.9.2) |
| jax-v0.9.1 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.9.1`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.9.1) | [`rocm-jaxlib-v0.9.1`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.9.1) |
| jax-v0.9.0 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.9.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.9.0) | [`rocm-jaxlib-v0.9.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.9.0) |
| jax-v0.8.2 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.8.2`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.8.2) | [`rocm-jaxlib-v0.8.2`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.8.2) |
| jax-v0.8.0 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.8.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.8.0) | [`rocm-jaxlib-v0.8.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.8.0) |
| jax-v0.7.x (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.7.1`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.7.1) | [`rocm-jaxlib-v0.7.1`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.7.1) |
| jax-v0.6.x (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.6.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.6.0) | [`rocm-jaxlib-v0.6.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.6.0) and [`rocm-jaxlib-v0.6.2`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.6.2) |
| jax-v0.5.0 | [`rocm-jaxlib-v0.5.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.5.0) | [`rocm-jaxlib-v0.5.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.5.0) |
| jax-v0.4.35 | [`rocm-jaxlib-v0.4.35-qa`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.4.35-qa) | [`rocm-jaxlib-v0.4.35-qa`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.35-qa) |
| jax-v0.4.31 | [`rocm-jaxlib-v0.4.31-qa`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.4.31-qa) | [`rocm-jaxlib-v0.4.31-qa`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.31-qa) |
| jax-v0.4.30 | [`rocm-jaxlib-v0.4.30-qa`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.4.30-qa) | [`rocm-jaxlib-v0.4.30-qa`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.30-qa) |
| jax-v0.4.28 | [`rocm-jaxlib-v0.4.28-qa`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.4.28-qa) | [`rocm-jaxlib-v0.4.28-qa`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.28-qa) |

## Recommended Docker Image && Build Instructions

Use the same ROCm-less build container as CI for reproducible local builds and CI parity:

```bash
docker pull ghcr.io/rocm/jax-build-ubu24.rocmless@sha256:68854ed30b800d3e8c390f6e651721e2bead58830da1c59e1b828354eb8f077f
```

This is the digest pinned in the `rocm-config` job of the CI workflows; the image's `latest` tag is a newer build that CI does not use. The image is Ubuntu 24.04 with Python 3.12, Bazel (via Bazelisk), and the build dependencies needed by JAX and XLA, but it has **no ROCm install** (no `/opt/rocm`). Instead, Bazel downloads a [TheRock](https://github.com/ROCm/TheRock) ROCm tarball and builds against it with the hermetic Clang toolchain, so neither a system ROCm nor a system Clang is required. Mount your local `jax` and `xla` source checkouts into the container (e.g. as `/tf/jax` and `/tf/xla`), and pass `--device=/dev/kfd --device=/dev/dri --group-add video` to `docker run` so tests can see the GPU, before running the build commands below.

Bazel picks the TheRock tarball from these flags, which every CI job passes:

```bash
--repo_env=ROCM_PATH=
--repo_env=ROCM_DISTRO_URL=https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-10.0.0.tar.gz
--repo_env=ROCM_DISTRO_HASH=1c5e807875d26a2470ecc7323daa5b5b9009208a55c3290ac255a909cde15fc6
```

- **Keep `ROCM_PATH` empty, and don't pass `--rocm_path` to `build/build.py`.** A non-empty `ROCM_PATH` takes priority over `ROCM_DISTRO_URL`, so Bazel would use a local ROCm install instead of the tarball.
- TheRock 10.0.0 is also XLA's default hermetic ROCm (`rocm_latest_multiarch` in [`third_party/gpus/rocm/rocm_redist.bzl`](https://github.com/openxla/xla/blob/main/third_party/gpus/rocm/rocm_redist.bzl)); the two `ROCM_DISTRO_*` flags pin it explicitly.
- To build against a TheRock nightly, set `ROCM_DISTRO_URL` to `https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-<version>.tar.gz` and `ROCM_DISTRO_HASH=no_hash` (which skips the checksum). This prints the latest 10.2 nightly `<version>`:

  ```bash
  GPU_TARGET=device-all THEROCK_INDEX_URL=https://nightly.repo.amd.com/rocm/whl-next/ ROCM_SDK_VERSION=10.2 \
    bash build_tools/rocm/detect-therock-version.sh
  ```

- CI runs the XLA unit tests with [`build_tools/rocm/run_xla_ci_build.sh`](https://github.com/openxla/xla/blob/main/build_tools/rocm/run_xla_ci_build.sh) from `openxla/xla`, and builds and tests JAX with [`ci/run_bazel_test_rocm_rbe.sh`](https://github.com/jax-ml/jax/blob/main/ci/run_bazel_test_rocm_rbe.sh) from `jax-ml/jax`, both with these flags on top of the ROCm RBE configs. See [`rocm_xla_ci_nightly.yml`](.github/workflows/rocm_xla_ci_nightly.yml) and [`rocm_xla_ci_jax.yml`](.github/workflows/rocm_xla_ci_jax.yml) for the exact commands.


### Building `jax-ml/jax:main` against `openxla/xla:main`

Inside the [recommended container](#recommended-docker-image--build-instructions), with the JAX repo mounted at `/tf/jax` and the XLA repo mounted at `/tf/xla`:

```bash
cd /tf/jax

# Remove any pre-existing JAX/jaxlib installs first, in case a mixed-up
# jaxlib is present that would otherwise shadow the freshly built wheels.
python3 -m pip uninstall jax jaxlib jax-rocm10-pjrt jax-rocm10-plugin jax-rocm7-pjrt jax-rocm7-plugin -y
rm -rf dist/*

python build/build.py build \
  --wheels=jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt \
  --rocm_amdgpu_targets=gfx942 \
  --rocm_version=10 \
  --bazel_options=--override_module=xla=/tf/xla \
  --bazel_options=--config=rocm_clang_hermetic \
  --bazel_options=--repo_env=ROCM_PATH= \
  --bazel_options=--repo_env=ROCM_DISTRO_URL=https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-10.0.0.tar.gz \
  --bazel_options=--repo_env=ROCM_DISTRO_HASH=1c5e807875d26a2470ecc7323daa5b5b9009208a55c3290ac255a909cde15fc6 \
  --bazel_options=--repo_env=TF_ROCM_AMDGPU_TARGETS=gfx942 \
  --bazel_startup_options="--bazelrc=build/rocm/rocm.bazelrc"

pip install dist/*.whl

# The rocmless image has no ROCm runtime, so point the loader at the TheRock
# libraries Bazel extracted during the build.
ROCM_DIST="$(bazel --bazelrc=build/rocm/rocm.bazelrc info output_base)/external/xla++rocm_configure_ext+local_config_rocm/rocm/rocm_dist"
export LD_LIBRARY_PATH="${ROCM_DIST}/lib:${ROCM_DIST}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Run outside /tf/jax so Python imports the installed wheels, not the source tree.
cd /tmp && python3 -c "import jax; print(jax.devices())"
```

Notes:

- **Use `--override_module=xla=/tf/xla`, NOT `--override_repository=xla=...`.** [Current JAX builds run under **bzlmod**](https://github.com/jax-ml/jax/pull/40053) (`.bazelrc` sets `--enable_bzlmod --noenable_workspace`), where XLA is a Bazel *module* whose canonical repo name is **`xla+`** (note the trailing `+`). The legacy `--override_repository=xla=...` targets a repo literally named `xla`, which does not exist under bzlmod, so it is **silently ignored** — Bazel then builds XLA from the bzlmod-pinned commit and your local checkout is never compiled.
- Adjust `--rocm_amdgpu_targets` and `--repo_env=TF_ROCM_AMDGPU_TARGETS` to match your hardware (e.g. `gfx90a`, `gfx942`, `gfx950`, `gfx1100`), keeping both the same. Multiple targets can be passed comma-separated. `--rocm_amdgpu_targets` alone does not reach Bazel's ROCm setup, because `build/rocm/rocm.bazelrc` sets `TF_ROCM_AMDGPU_TARGETS` to 27 architectures with `--repo_env`, which makes the build much slower and the wheels much larger.
- The `--wheels=jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt` list produces all four required wheels in `dist/`; `pip install dist/*.whl` installs the matching set. Built against TheRock 10.0.0, the ROCm wheels are named `jax_rocm10_plugin` and `jax_rocm10_pjrt`.
- The last command should print `[RocmDevice(id=0), ...]`. Without the `LD_LIBRARY_PATH` step, JAX prints a plugin error (`librocprofiler-sdk.so.1: cannot open shared object file`) and falls back to `[CpuDevice(id=0)]`. Set `LD_LIBRARY_PATH` again in every new shell, and rebuild after a `bazel clean --expunge`, which deletes the extracted libraries.

### Building the `rocm-jaxlib-v0.8.x` / `rocm-jaxlib-v0.9.x` release branches

For the **0.8.x** and **0.9.x** JAX releases (rows in the [XLA Branches for JAX](#xla-branches-for-jax) table above), the wheel build is driven by the [`ROCm/rocm-jax`](https://github.com/ROCm/rocm-jax/tree/rocm-jax-infra) repository on its `rocm-jax-infra` branch — **not** by `build/build.py` in `ROCm/jax` directly. This is the legacy `stack.py` / `build/ci_build` path that pairs the matching `rocm-jaxlib-v0.8.x` / `rocm-jaxlib-v0.9.x` branches of `ROCm/xla` and `ROCm/jax`.

| JAX Version | Repo used for building | Builds `jaxlib` wheel? |
|-------------|------------------------|------------------------|
| 0.8.x       | [`ROCm/rocm-jax`](https://github.com/ROCm/rocm-jax/tree/rocm-jax-infra) | Yes |
| 0.9.x       | [`ROCm/rocm-jax`](https://github.com/ROCm/rocm-jax/tree/rocm-jax-infra) | Yes |
| 0.10.x and newer | [`ROCm/jax`](https://github.com/ROCm/jax) (artifact workflow / `build/build.py`) | No (only when `jaxlib` changes) |

Starting with **JAX 0.10.0**, wheel builds move to the `ROCm/jax` fork's artifact workflow (`ci/build_rocm_artifacts.sh` → `build/build.py`); the `stack.py` entrypoint in `rocm-jax` is deprecated and the `rocm-jax-infra` branch is retained only for Dockerfiles and image-build infrastructure.

To build a 0.8.x / 0.9.x wheel set, follow the build and CI scripts in the `rocm-jax-infra` branch of `ROCm/rocm-jax`, pointing them at the corresponding `rocm-jaxlib-v0.8.x` / `rocm-jaxlib-v0.9.x` branches of this repository ([`ROCm/xla`](https://github.com/ROCm/xla)) and [`ROCm/jax`](https://github.com/ROCm/jax). For `main` and the 0.11.x+ branches, use the `build/build.py` invocation shown in the previous section instead. The 0.10.x branches' XLA predates TheRock tarball support (`ROCM_DISTRO_URL`), so build those in the ROCm 7.2.4 image (`ghcr.io/rocm/jax-ubu22.rocm7.2.4`) with the same command, replacing the three `--repo_env=ROCM_*` options with `--rocm_path=/opt/rocm` and using `--rocm_version=7`.

## OpenXLA Upstream PR Status

Track pending and in-progress ROCm upstream contributions to OpenXLA on the [OpenXLA:GPU - AMD/ROCm project board](https://github.com/orgs/openxla/projects/27).

## OpenXLA Upstream CI Checks

Every `openxla/xla` PR runs the upstream [`rocm_ci.yml`](https://github.com/openxla/xla/blob/main/.github/workflows/rocm_ci.yml) workflow on a gfx950 runner, using the ROCm-less image and TheRock 10.0.0 tarball described in [Recommended Docker Image && Build Instructions](#recommended-docker-image--build-instructions). It has two jobs:

| Job | Entry point | Purpose |
|-----|-------------|---------|
| `xla` | [`build_tools/rocm/run_xla_ci_build.sh`](https://github.com/openxla/xla/blob/main/build_tools/rocm/run_xla_ci_build.sh) (in `openxla/xla`) | **XLA test sweep.** Runs the `//xla/...` single-GPU and CPU test suites. |
| `jax` | [`ci/run_bazel_test_rocm_rbe.sh`](https://github.com/jax-ml/jax/blob/main/ci/run_bazel_test_rocm_rbe.sh) (in `jax-ml/jax`) | **JAX test sweep against XLA-from-source.** Builds jaxlib and JAX with Bazel against the PR's XLA and runs the JAX GPU tests. Runs only for PRs targeting `main`. |

Both jobs use the EngFlow `internal.actinolite.cluster.engflow.com` RBE cluster through the `rocm_rbe` / `rocm_rbe_dynamic` configs in XLA's `build_tools/rocm/rocm_xla.bazelrc` and JAX's `build/rocm/rocm.bazelrc`.

## Debug Tools

Debugging utilities for triaging issues while running MaxText-TE / JAX / TF workloads on AMD GPUs (ROCm). See [`debug_tools/`](debug_tools/) for full details. Individual tools live in their own subdirectories with self-contained usage notes.

| Tool | Purpose |
|------|---------|
| [`debug_tools/queue_dump`](debug_tools/queue_dump) | Hang investigation. Drives `gdb` / `rocgdb` / `umr` / the ROCR Debug Agent / RDL to inspect a hung process and its GPU state (queues, kernels, wavefronts, fences). |
| [`debug_tools/nan_check_tool`](debug_tools/nan_check_tool) | NaN / numerical-correctness investigation. Scans intermediate buffers for NaN / Inf and localises the first offending HLO instruction, buffer index, and rank. |

## Performance Tools

Performance and regression micro-benchmarks for AMD GPUs (ROCm). See [`perf_tools/`](perf_tools/) for full details.

| Tool | Purpose |
|------|---------|
| [`perf_tools/rocm_latency_check_tool`](perf_tools/rocm_latency_check_tool) | HIP-runtime-only micro-benchmark measuring end-to-end per-request dispatch latency (H2D → compute → D2D → compute → D2H) and comparing it across ROCm versions. Reports a latency percentile histogram plus QPS and can append version-stamped rows to a CSV for regression tracking. |
| [`perf_tools/hlo_eval_tools`](perf_tools/hlo_eval_tools) | Per-module HLO micro-benchmarks built on `multihost_hlo_runner`. Replays the pre-optimization HLO dumps (training and inference) from real JAX models one module at a time to measure each submodule's cost on XLA, and appends the timings to a CSV for regression tracking across ROCm/XLA versions. Covers models across four categories: LLM, vision, multimodal, and science. |

## Scaling Book

The scaling book is a companion to [Google's JAX Scaling Book](https://jax-ml.github.io/scaling-book/) for AMD GPU users, covering performance features for JAX training at the implementation layer (ROCm libraries, MaxText, XLA internals, and JAX APIs).

The `scaling_book/` directory contains the Jekyll site served on GitHub Pages. Page content can be found at `scaling_book/pages/` with images at `scaling_book/pages/img/`.

Reference reproductions for the case studies can be found at the following locations:

| Folder | Description |
|--------|-------------|
| [`scaling_book/llama7b-jax-fundamentals`](scaling_book/llama7b-jax-fundamentals) | Rematerialisation, attention backends, and profiling examples |
| [`scaling_book/llama70b-mixed-precision-training`](scaling_book/llama70b-mixed-precision-training) | Mixed quantization experiments in BF16, FP16, FP8, MXFP8, and MXFP4 |
| [`scaling_book/mixtral8-22b-distributed-strategies`](scaling_book/mixtral8-22b-distributed-strategies) | Sharding and grouped GEMM experiments |

## Workflows

GitHub Actions workflows and PR labels that drive automation in this fork.

### Scheduled / event-driven workflows

| Workflow                     | Schedule                                                          | Description                                                                                                                                                                                           |
| ----------                   | ----------                                                        | -------------                                                                                                                                                                                         |
| `sync_upstream.yml`          | Every 3h (weekdays), daily (weekends)                             | Syncs `main` with `openxla/xla:main` via the GitHub merge-upstream API                                                                                                                                |
| `claude_auto_review.yml`     | Every PR opened against a branch that has `pr_event_dispatch.yml` | Automatically runs Claude-Opus-5.5 powered code review on every pull request                                                                                                                          |
| `rocm_xla_ci_nightly.yml`    | Daily at 3:47 AM UTC                                              | Runs the latest XLA (openxla/xla) unit test suite, the TSAN and ASAN sanitizer builds (`rocm_xla_ci_sanitizers.yml`) and the latest JAX (jax-ml/jax) unit tests (`rocm_xla_ci_jax.yml`), each against the TheRock 10.0.0 release tarball and the latest TheRock nightly tarball (the XLA tests track the [10.2 nightlies](https://nightly.repo.amd.com/rocm/whl-next/rocm-sdk-devel/)) |
| `rocm_infra_monitor.yml`     | Every 1h                                                          | Analyzes xla upstream PRs to detect infrastructure problems. If infra problem is detected then new issue is created.                                                                                                                                |
| `scaling_book_pages.yml`     | Push to `rocm-dev-infra` when `scaling_book/` changes             | Builds the scaling book from `scaling_book/` and deploys to GitHub Pages at [rocm.github.io/xla/scaling-book](https://rocm.github.io/xla/scaling-book/)                                                                                          |

### Opt-in PR labels

PRs opened against branches in the `ROCm/xla` fork can opt-in to additional automated checks by attaching labels:

| Label          | Effect                                                                                |
|----------------|---------------------------------------------------------------------------------------|
| `claude-review` | Triggers an automated Claude code review on the PR.                                  |
| `sanitizers`    | Runs sanitizer-instrumented builds and tests: ThreadSanitizer (TSAN) and AddressSanitizer (ASAN). |
