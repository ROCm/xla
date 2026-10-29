## ${repo}

### Test coverage and cost per test

Test counts come from bazel's `Executed X out of Y tests` line in sampled job logs from the last 48h of the window, so suite sizes are current. Test-step p50 and seconds per test use only runs that executed at least 90% of the suite (remote-cache hits make the rest cheap and unrepresentative).

${coverage_table}

${coverage_note}

### Metrics

${metrics_table}

> DO queue wait = `started_at - created_at` for every DO job in the band (all workflows in the repo). DO time to signal = PR head SHA first triggered (or approved, for fork PRs) to its last ROCm job finishing. RBE cells = wall time of the bazel test step that dispatches to EngFlow, for jobs created in the band.${nightly_note}

### Where the time to signal goes (ROCm on DO, NV on Google runners)

For each commit, the lane's job that finishes last is followed from the moment the SHA was first triggered. Every second lands in exactly one named phase, and each row adds up to its total (checked in code), so there is no unaccounted time. One row per lane, over all hours of the window.

${families_note}

#### Typical commit (commits between p40 and p60 of time to signal, mean of each phase)

${phase_typical_table}

#### Slowest 10% of commits (at or above p90, mean of each phase)

${phase_slowest_table}

${phase_notes}

### ROCm versus the NVIDIA reference lane

Scoped to the presubmit test lane, `pull_request` runs only.

${rocm_vs_nv_table}

n = ${rocm_commits} ROCm commits, ${nv_commits} NV commits.

${compare_note}

### Peak versus nominal: ROCm and NV side by side

Both lanes are measured over the same clock hours: the peak and nominal hours listed at the top, which come from DO arrivals pooled across every repo, so all sections share one clock. Jobs are placed by the hour they were created, commits by the hour their SHA was first triggered.

${band_table}

p90 by time of day (3-hour blocks, CDT (UTC-5)):

${band_by_hour_table}

> Queue wait covers every job of the presubmit lane, so the ROCm figure differs slightly from the Metrics table, which takes every DO job in the repo. Job execution and test step use only the matched test job types listed above; job types with no counterpart on the other lane are left out. Highlights follow the suite-size rule used above, and queue wait is never highlighted. In the time-of-day tables a block is tagged peak or nominal when at least 2 of its 3 hours are, and test-job rows pool the matched job types; read p90 with its commit count.

### Tail latency and same-commit comparison

Job execution = `completed_at - started_at` for the jobs of the run that signalled, per job type. The same-commit tables compare ROCm and NV on the **same PR head SHA**, so both lanes saw the same code change and the same set of tests invalidated in the cache.

#### Percentiles by job type

${tail_percentile_table}

> p99 of a few hundred jobs rests on a handful of samples; read it with its n.

${same_commit_section}
