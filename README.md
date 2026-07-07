# ROCm/XLA Dev Infrastructure Branch
This is the `rocm-dev-infra` branch — the default branch of the [`ROCm/xla`](https://github.com/ROCm/xla) fork of [`openxla/xla`](https://github.com/openxla/xla).
It hosts GitHub Actions workflows, CI/CD infrastructure, and serves as a central index for all ROCm JAX/XLA release branches and their hardware support matrix.

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
- [Workflows](#workflows)

## XLA Branches for JAX

For supported ROCm versions and GFX targets per release, see the [AMD ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

| JAX Release | XLA Branch | JAX Branch |
|-------------|------------|------------|
| jax-ml/jax:main | [`openxla/xla:main`](https://github.com/openxla/xla) | [`jax-ml/jax:main`](https://github.com/jax-ml/jax) |
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

Use the prebuilt ROCm JAX build container for reproducible local builds and CI parity:

```bash
docker pull ghcr.io/rocm/jax-ubu22.rocm7.2.4
```

This image ships ROCm 7.2.4 at `/opt/rocm`, multiple Python versions, Bazel, and all build dependencies needed by JAX and XLA. The build uses the hermetic Clang-18 toolchain, so no system Clang is required. Mount your local `jax` and `xla` source checkouts into the container (e.g. as `/tf/jax` and `/tf/xla`) before running the build commands below.


### Building `jax-ml/jax:main` against `openxla/xla:main`

Inside the [recommended container](#recommended-docker-image--build-instructions), with the JAX repo mounted at `/tf/jax` and the XLA repo mounted at `/tf/xla`:

```bash
cd /tf/jax

# Remove any pre-existing JAX/jaxlib installs first, in case a mixed-up
# jaxlib is present that would otherwise shadow the freshly built wheels.
python3 -m pip uninstall jax jaxlib jax-rocm-pjrt jax-rocm-plugin -y

python build/build.py build \
  --wheels=jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt \
  --rocm_path=/opt/rocm \
  --rocm_amdgpu_targets=gfx942 \
  --rocm_version=7 \
  --bazel_options=--override_repository=xla=/tf/xla \
  --bazel_options=--config=rocm_clang_hermetic \
  --bazel_startup_options="--bazelrc=build/rocm/rocm.bazelrc"

pip install dist/*.whl
```

Notes:

- `--bazel_options=--override_repository=xla=/tf/xla` redirects the `@xla` repository to your local checkout, replacing the upstream-pinned commit in JAX's `WORKSPACE`.
- Adjust `--rocm_amdgpu_targets` to match your hardware (e.g. `gfx90a`, `gfx942`, `gfx950`, `gfx1100`). Multiple targets can be passed comma-separated.
- The `--wheels=jax,jaxlib,jax-rocm-plugin,jax-rocm-pjrt` list produces all four required wheels in `dist/`; `pip install dist/*.whl` installs the matching set.

### Building the `rocm-jaxlib-v0.8.x` / `rocm-jaxlib-v0.9.x` release branches

For the **0.8.x** and **0.9.x** JAX releases (rows in the [XLA Branches for JAX](#xla-branches-for-jax) table above), the wheel build is driven by the [`ROCm/rocm-jax`](https://github.com/ROCm/rocm-jax/tree/rocm-jax-infra) repository on its `rocm-jax-infra` branch — **not** by `build/build.py` in `ROCm/jax` directly. This is the legacy `stack.py` / `build/ci_build` path that pairs the matching `rocm-jaxlib-v0.8.x` / `rocm-jaxlib-v0.9.x` branches of `ROCm/xla` and `ROCm/jax`.

| JAX Version | Repo used for building | Builds `jaxlib` wheel? |
|-------------|------------------------|------------------------|
| 0.8.x       | [`ROCm/rocm-jax`](https://github.com/ROCm/rocm-jax/tree/rocm-jax-infra) | Yes |
| 0.9.x       | [`ROCm/rocm-jax`](https://github.com/ROCm/rocm-jax/tree/rocm-jax-infra) | Yes |
| 0.10.x and newer | [`ROCm/jax`](https://github.com/ROCm/jax) (artifact workflow / `build/build.py`) | No (only when `jaxlib` changes) |

Starting with **JAX 0.10.0**, wheel builds move to the `ROCm/jax` fork's artifact workflow (`ci/build_rocm_artifacts.sh` → `build/build.py`); the `stack.py` entrypoint in `rocm-jax` is deprecated and the `rocm-jax-infra` branch is retained only for Dockerfiles and image-build infrastructure.

To build a 0.8.x / 0.9.x wheel set, follow the build and CI scripts in the `rocm-jax-infra` branch of `ROCm/rocm-jax`, pointing them at the corresponding `rocm-jaxlib-v0.8.x` / `rocm-jaxlib-v0.9.x` branches of this repository ([`ROCm/xla`](https://github.com/ROCm/xla)) and [`ROCm/jax`](https://github.com/ROCm/jax). For 0.10.x+ branches and `main`, use the `build/build.py` invocation shown in the previous section instead.

## OpenXLA Upstream PR Status

Track pending and in-progress ROCm upstream contributions to OpenXLA on the [OpenXLA:GPU - AMD/ROCm project board](https://github.com/orgs/openxla/projects/27).

## OpenXLA Upstream CI Checks

PRs that touch upstream-tracking branches (e.g. `main`, which mirrors `openxla/xla:main`) run two CI entry-points on the ROCm RBE pool to validate the change against the latest upstream XLA and JAX:

| Script | Purpose |
|--------|---------|
| [`build_tools/rocm/execute_ci_build_upstream.sh`](build_tools/rocm/execute_ci_build_upstream.sh) | **XLA test sweep.** Shim script that runs the `//xla/...` test suite on ROCm RBE |
| [`build_tools/rocm/execute_ci_build_upstream_jax.sh`](build_tools/rocm/execute_ci_build_upstream_jax.sh) | **JAX test sweep against XLA-from-source.** Shim script that runs the JAX test suite against XLA built from source on ROCm RBE. |

Both scripts run on the EngFlow `wardite.cluster.engflow.com` RBE cluster and consume the `rocm_rbe` / `rocm_rbe_dynamic` configs from `build/rocm/rocm.bazelrc` (in the JAX repo) and XLA's own RBE bazelrc.

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

## Workflows

GitHub Actions workflows and PR labels that drive automation in this fork.

### Scheduled / event-driven workflows

| Workflow | Schedule | Description |
|----------|----------|-------------|
| `sync_upstream.yml` | Every 3h (weekdays), daily (weekends) | Syncs `main` with `openxla/xla:main` via the GitHub merge-upstream API |
| `claude_auto_review.yml` | Every PR opened against a branch that has `pr_event_dispatch.yml` | Automatically runs Claude Opus 4.6-powered code review on every pull request |

### Opt-in PR labels

PRs opened against branches in the `ROCm/xla` fork can opt-in to additional automated checks by attaching labels:

| Label          | Effect                                                                                |
|----------------|---------------------------------------------------------------------------------------|
| `claude-review` | Triggers an automated Claude code review on the PR.                                  |
| `sanitizers`    | Runs sanitizer-instrumented builds and tests: ThreadSanitizer (TSAN) and AddressSanitizer (ASAN). |
