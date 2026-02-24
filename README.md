# rocm/xla Infrastructure Branch

This is the `rocm-dev-infra` branch — the default branch of the `rocm/xla` fork.
It exists solely to host GitHub Actions workflows and other CI/CD infrastructure.

The **`main`** branch is an exact mirror of `openxla/xla:main` and should never
contain fork-specific commits. This separation ensures the `merge-upstream` API
(fast-forward only) continues to work for automatic syncing.

## Workflows

| Workflow | Schedule | Description |
|----------|----------|-------------|
| `sync_upstream.yml` | Every 3h (weekdays), daily (weekends) | Syncs `main` with `openxla/xla:main` via the GitHub merge-upstream API |
| `claude_auto_review.yml` | Every PR (`pull_request`) | Automatically runs Claude-powered code review on every pull request |
