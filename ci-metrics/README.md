# ci-metrics: Nightly CI report for upstream JAX and OpenXLA

Builds a one-page HTML + Markdown report of presubmit CI health over a 1-day
window, with **one section per repo** (`REPOS` in `metrics_lib.py`) followed by
the shared infrastructure sections. Each repo section holds: time to
signal, queue wait, job and test-step execution, where every second of time to
signal goes, at peak and nominal hours, tail percentiles for PR same-commit on various runner pools and test coverage.

Everything comes from the public GitHub Actions API and job logs. Nothing reads
EngFlow, DO dashboards or other internal systems.

## Requirements

- Python 3.10 or newer, standard library only (tested on 3.10).
- `gh` CLI, logged in (tested with 2.97). The scripts take the token from
  `$GH_TOKEN` / `$GITHUB_TOKEN`, or from `gh auth token`. Read access to public
  repos is enough.
- GitHub allows 5000 API requests per hour per token. A full week needs more than
  that, so `collect.py` pauses once until the limit resets. Every step is
  resumable: if one stops, run it again and it continues where it left off.

## Run it

Pick a Tuesday-to-Tuesday window in UTC (the reports so far use 15:00 UTC) and an
empty data directory:

```bash
D=~/ci-metrics-data/2026-10-13 S=2026-10-06T15:00 U=2026-10-13T15:00
python3 collect.py        --since $S --until $U --out $D
python3 fetch_attempts.py --data $D
python3 sample_logs.py    --data $D
python3 report.py         --data $D --since $S --until $U --html report.html
```

`report.py` writes `report.md` alongside `report.html`, converted from the same
rendered HTML so both hold identical numbers and prose. Use `--md PATH` to put it
elsewhere, or `--no-md` to skip it. In Markdown the green "ROCm wins" highlight
becomes a bold value with a `▲`, the dominant phase of a row becomes bold, and
a not-measured figure becomes italic.

`report.py` only reads the data directory (plus a few cached API lookups for
fork-PR numbers), so you can re-run it as often as you like while editing.

For this nightly previous day to current timestamp is used as time window

## What each script does

| Script | Role | Writes to the data dir |
|---|---|---|
| `collect.py` | Fetches every run of the six workflows in `WORKFLOWS`, with each job's queue/start/end and per-step timestamps. | `jobs.jsonl`, `runs--<repo>--<workflow>.json` |
| `fetch_attempts.py` | Fetches earlier attempts of re-run PR runs; `collect.py` only stores the latest attempt, and a failed first attempt is already a signal. | `attempts.jsonl` |
| `sample_logs.py` | Downloads a random sample of job logs and parses bazel's `Executed X out of Y tests` and `Elapsed time / Critical Path` lines. | `logs.jsonl` |
| `metrics_lib.py` | `REPOS` (the repo/lane config) plus shared definitions: how a step maps to a phase, the time zone. Every CI naming convention lives here. | - |
| `report.py` | Computes all statistics (`build()`) and renders the HTML (`render()`). | `pr_lookup.json` (cache) |
| `html_md.py` | Converts the rendered HTML to Markdown, so there is one renderer and one set of numbers. | - |

## How the numbers are defined

- **Repos and lanes.** `REPOS` lists the upstream repos; each gets a ROCm lane
  and an NV reference lane, from which `PRESUBMIT_LANES` is derived with the
  stable keys `<short>_rocm` / `<short>_nv`. Presubmit means `pull_request` runs
  only; one commit is one PR head SHA.
- **Job-family sub-lanes.** `openxla/xla` runs its own tests and the JAX
  integration tests in one workflow, so each OpenXLA lane is also measured one
  family at a time (`SPLIT_LANES`, keyed `xla_rocm:xla`, `xla_rocm:jax` and the
  NV equivalents, derived from the entry's `families`). In the phase tables such
  a lane is one row per family, both over the same commits and so
  sharing one `n`. `Total` is that family's own signal; `Lane signal` is the
  combined one the PR waits for - the slower family on each commit, not the sum
  - so it repeats down the lane, and is `—` for the single-family JAX lanes.
  `last on N/n` counts the commits where that family finished last; a family at
  `n/n` necessarily matches the lane signal, because on those commits it is the
  lane.
- **Phase tables are not banded.** The "Where the time to signal goes" tables
  use one row per lane over all hours. Peak/nominal there left only a handful
  of commits per row once the p90 tail was taken; the band split still applies
  in the Metrics, band and tail sections.
- **Time to signal.** From the SHA's first trigger to the last lane job
  finishing, in the run that completed a full signal earliest (every lane job
  succeeded, failed or timed out; cancelled runs never count). For fork
  PRs the clock starts when a maintainer approves CI; the approval wait is
  reported separately.
- **Phases.** Each commit's time to signal is split into superseded runs, waiting
  on earlier jobs, queue, runner set-up, docker, checkout/env, test execution,
  close, and runner teardown (wall time minus the sum of steps). The phases add up
  to the total, and the code asserts it.
- **Peak and nominal hours.** The 8 busiest and 8 quietest hours of the day by
- **Like-for-like.* comparisons use matched job types (`PAIRS`
  in `report.py`).
- **Re-runs.** When "re-run failed jobs" copies finished jobs into a new attempt,
  the copy keeps the old start/end but gets a new creation time. Those copies are
  dropped (`carried_over()` in `metrics_lib.py`).
- **Test counts and cost per test.** Taken from the sampled logs, not from every
  job. Seconds per test only uses runs that executed at least 90% of the suite.

## Adapting it

- **Another repo, or a renamed workflow/job/step.** Add or edit an entry in
  `REPOS` in `metrics_lib.py`. That one table drives everything: lane keys,
  display names, job families, the DO pool, the nightly workflows, job-type
  classification, the like-for-like `pairs`, the RBE rows, and `WORKFLOWS` in
  `collect.py`. `report.py` loops over it and needs no change — a new entry
  produces a full report section on its own. Step regexes are still shared
  (`TEST_STEP_RE`, `CLOSE_STEP_RE`).
- **New runner pool labels.** The `pool` field of the repo's `REPOS` entry; both
  the peak-vs-nominal table and `POOL_NOW` read it. `POOL_NOW`'s keys are the
  names the `--baseline` JSON uses, so they must stay stable.
- **Time zone.** `CDT` in `metrics_lib.py` is fixed at UTC-5. Change it to UTC-6
  for winter windows.
- **New HTML construct in the report.** `html_md.py` handles only the tags
  `report.py` emits; anything else degrades to its text. Add it there if a new
  section needs it in the Markdown.
- **Hand-written text.** Some sections are not computed: the "pending" rows under
  Infrastructure, "Executor queue by pool", "The ask", and the notes about bazel
  strategies. Review them before sharing a report.
- **Week-over-week comparison.** `report.py --baseline BASE.json` adds a "Changes
  since" section from an earlier report's figures. The format is in the
  `report.py` docstring.

## Limits

- GitHub expires job logs (its default retention is 90 days), so `sample_logs.py`
  only works on recent windows. Run and job metadata last longer.
- The venodr lanes run on Google-hosted runners and Google RBE. The report sees them
  only through GitHub job and step timestamps and their logs.
