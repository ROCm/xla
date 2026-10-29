# CI metrics: upstream JAX & OpenXLA on ROCm

Window **${window}** (${span}). Generated ${generated}.

**DO** = Digital Ocean self-hosted runners (`amd-do-*`), where presubmits build. **RBE** = EngFlow remote pool (`wardite`), where tests dispatch. **NV** = Google-hosted NVIDIA runners, the reference lane. **Presubmit** = `pull_request` runs only; one commit = one PR head SHA.

**${win}** marks a like-for-like ROCm win; plain = trails, or the suites are not comparable. **Bold** in a phase row is that row's largest phase, and *italics* is a figure that was not measured.

Peak hours (CDT (UTC-5), busiest 8 by measured DO arrivals): ${peak_hours}  ·  nominal (quietest 8): ${nominal_hours}

${repo_sections}

## Infrastructure

${infra_table}

## Peak versus nominal: measured

${arrivals_table}

> Arrival volume swings **${swing}x** across the day (${arrivals_lo} to ${arrivals_hi} jobs/hour). DO jobs = every ROCm job on DO in the collected workflows; NV jobs = jobs of the NV presubmit lanes, by the hour they were created. NV's busiest 8 hours are ${nv_peak_hours}, ${shared_peak_hours} of them DO peak hours.

${pool_component_table}

> Arrivals swing ${swing}x. ${pool_verdicts}.

## DO scale-sets

${scale_set_table}

> GPUs per job read from the label suffix. All ROCm jobs in the collected workflows, any conclusion except skipped.

### Runner teardown: tests done, check still pending

Job wall time minus the sum of its steps. Almost all of it sits after the `Complete job` step, before GitHub marks the job finished.

${teardown_table}

## NV runner pools

Google-hosted runners of the NV presubmit lanes, by runner label and job type. Queue = `started_at - created_at`. A label whose last day seen is before the end of the window stopped appearing partway through the window.

${nv_pool_table}

## RBE (EngFlow) detail

From bazel's `Elapsed time / Critical Path` line in sampled job logs across the whole window. The gap between wall time and critical path is scheduling, queueing, cache lookup and local work, not test execution on the critical path.

${rbe_table}

> Both ROCm presubmits run `--config=rocm_rbe_dynamic`: tests race a local run on the DO GPU against the EngFlow run (`--strategy=TestRunner=dynamic` in JAX, `--spawn_strategy=dynamic` in XLA), so RBE wall time includes work that may have finished locally.

### Executor queue by pool

> Not refreshed. This table came from a manual AWS dashboard export; it needs a new export for this window (owner: user).

## The ask: more runners

${ask_table}

> Google's CUDA presubmit compiles remotely (`--spawn_strategy=remote,worker,standalone,local`, seen in this window's logs); JAX ROCm pins builds to the GPU box, while `openxla/xla` ROCm already uses `--spawn_strategy=dynamic`.

${ask_coverage}

**Gating coverage:** with more hardware we can pitch Google to gate PRs on far more ROCm tests than the single target we gate on today.

${changes_section}
