# ROCm/XLA Dev Infrastructure Branch

This is the `rocm-dev-infra` branch — the default branch of the [`ROCm/xla`](https://github.com/ROCm/xla) fork of [`openxla/xla`](https://github.com/openxla/xla).
It hosts GitHub Actions workflows, CI/CD infrastructure, and serves as a central index for all ROCm JAX/XLA release branches and their hardware support matrix.

The **`main`** branch is an exact mirror of `openxla/xla:main` and should never
contain fork-specific commits. This separation ensures the `merge-upstream` API
(fast-forward only) continues to work for automatic syncing.

## Table of Contents

- [XLA Branches for JAX](#xla-branches-for-jax)
- [OpenXLA upstream PRs Status](#upstream-pr-status)
- [Workflows](#workflows)

## XLA Branches for JAX

| JAX Release | XLA Branch | JAX Branch | ROCm Version | GFX Target |
|-------------|------------|------------|--------------|------------|
| jax-ml/jax:main | [`openxla/xla:main`](https://github.com/openxla/xla) | [`jax-ml/jax:main`](https://github.com/jax-ml/jax) | | |
| jax-v0.9.x (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.9.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.9.0) | [`rocm-jaxlib-v0.9.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.9.0) | | |
| jax-v0.8.2 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.8.2`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.8.2) | [`rocm-jaxlib-v0.8.2`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.8.2) | | |
| jax-v0.8.0 (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.8.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.8.0) | [`rocm-jaxlib-v0.8.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.8.0) | | |
| jax-v0.7.x (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.7.1`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.7.1) | [`rocm-jaxlib-v0.7.1`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.7.1) | | |
| jax-v0.6.x (rocm-pjrt-plugin) | [`rocm-jaxlib-v0.6.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.6.0) | [`rocm-jaxlib-v0.6.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.6.0) and [`rocm-jaxlib-v0.6.2`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.6.2) | | |
| jax-v0.5.0 | [`rocm-jaxlib-v0.5.0`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.5.0) | [`rocm-jaxlib-v0.5.0`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.5.0) | | |
| jax-v0.4.35 | [`rocm-jaxlib-v0.4.35-qa`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.4.35-qa) | [`rocm-jaxlib-v0.4.35-qa`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.35-qa) | | |
| jax-v0.4.31 | [`rocm-jaxlib-v0.4.31-qa`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.4.31-qa) | [`rocm-jaxlib-v0.4.31-qa`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.31-qa) | | |
| jax-v0.4.30 | [`rocm-jaxlib-v0.4.30-qa`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.4.30-qa) | [`rocm-jaxlib-v0.4.30-qa`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.30-qa) | | |
| jax-v0.4.28 | [`rocm-jaxlib-v0.4.28-qa`](https://github.com/ROCm/xla/tree/rocm-jaxlib-v0.4.28-qa) | [`rocm-jaxlib-v0.4.28-qa`](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.28-qa) | | |

## Upstream PR Status

Track pending and in-progress ROCm/AMD upstream contributions to OpenXLA on the [OpenXLA:GPU - AMD/ROCm project board](https://github.com/orgs/openxla/projects/27).

## Workflows

| Workflow | Schedule | Description |
|----------|----------|-------------|
| `sync_upstream.yml` | Every 3h (weekdays), daily (weekends) | Syncs `main` with `openxla/xla:main` via the GitHub merge-upstream API |
| `claude_auto_review.yml` | Every PR (`pull_request`) | Automatically runs Claude-powered code review on every pull request |
