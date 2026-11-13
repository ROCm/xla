# ci-metrics: Nightly CI report for upstream JAX and OpenXLA

Builds a one-page Markdown report of presubmit CI health over a 1-day window, with **one section per repo** (`REPOS` in `metrics_lib.py`) followed by
the shared infrastructure sections. Each repo section holds: time to
signal, queue wait, job and test-step execution, where every second of time to
signal goes, at peak and nominal hours, tail percentiles for PR same-commit on various runner pools and test coverage.

Everything comes from the public GitHub Actions API and job logs. Nothing reads
EngFlow, DO dashboards or other internal systems.

## Requirements

- Python 3.10 or newer (tested on 3.10). `collect.py`, `fetch_attempts.py` and
  `sample_logs.py` are standard library only; `report.py` needs Jinja2 to render
  the template: `pip install 'jinja2>=3.1'`.
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
python3 report.py         --data $D --since $S --until $U --md report.md --json report.json
```

The report is GitHub-flavoured Markdown, and **the whole document lives in
`templates/report.md.j2`** — every heading, sentence, table and piece of markup.
`report.py` computes figures and nothing else: it builds one dict and hands it to
Jinja2, and never assembles a string of Markdown. To reword the report, or to
move or restyle a table, edit the template and leave Python alone.

`--json` writes that same dict beside the report, so any figure in it can be
looked up by name (`jq '.repos[0].metrics.tts_peak.nv' report.json`) and fed back
in as next week's `--baseline`.

Two conventions in the template are worth knowing:

- Tables whose rows are fixed — Metrics, ROCm vs NV, Infrastructure, the ask —
  are written out row by row, each figure a named lookup like
  `{{ M.tts_peak.do }}`. Only tables whose rows are discovered at run time
  (scale-sets, NV runner pools, per-SHA rows) use a `{% for %}` loop, because
  there is nothing to name in advance.
- Meaning that a stylesheet would have carried is in the markup: a figure may
  arrive as `{text, mark}`, and the `c()` macro at the top of the template turns
  `win` into a bold value with a `▲`, `dominant` into bold, and `pending` into
  italics. What each looks like is decided there, not in Python.

Rendering uses `StrictUndefined`, so a name the template asks for and the data
does not carry raises an error rather than silently blanking a figure.

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
| `report.py` | Computes all statistics (`build()`) and fills the templates (`render()`). | `pr_lookup.json` (cache) |
| `templates/*.md` | The document structure: `report.md` (whole report), `repo_section.md` (one repo), `same_commit.md`. | - |
| `trends.py` | Rolls *every* published snapshot up into one day-by-day series for the web dashboard. Calls `report.build()` for the timing series and the p90 tail, and reads the sampled `logs.jsonl` and the cached `pr_lookup.json` for the test counts and PR numbers, so it adds no definitions of its own. | `trends.json` |
| `site/*` | The dashboard: `index.html`, `app.js`, `styles.css`. Plain SVG, no dependencies. | - |

## The trend dashboard

The Markdown report covers one 24h window. `trends.py` answers the other
question -- what changed over the last fortnight -- and
[`ci_metrics_pages.yml`](../.github/workflows/ci_metrics_pages.yml) publishes it
to <https://rocm.github.io/xla/ci-metrics/> after every nightly.

> **Not the perf trends.** The site carries two trend dashboards, and this is
> the one about **CI**: how long CI takes to answer a pull request. Its sibling
> [`perf-metrics/`](../perf-metrics) is about how fast the code CI builds
> actually runs -- different question, different data branch, different
> collector, and a point there is one benchmark run rather than one day. They
> share chart primitives, a stylesheet and a layout so that knowing one is
> knowing the other, and nothing else: neither reads the other's data, and no
> figure on one has a counterpart on the other. The two are compared
> side by side in the [root README](../README.md#dashboards).

```bash
git clone --branch ci-metrics-data --depth 1 <repo> /tmp/ci-data
python3 trends.py --data /tmp/ci-data/data --out site/trends.json
python3 -m http.server -d site 8000     # then open localhost:8000
```

GitHub Pages serves one deployment per repository, and a deployment replaces
the whole site, so **all three** Pages workflows -- `scaling_book_pages.yml`,
`ci_metrics_pages.yml` and `perf_metrics_pages.yml` -- build **all three**
sections through the shared
[`build-pages-site`](../.github/actions/build-pages-site/action.yml) action.
A workflow that published only its own section would delete the others. The
three are serialised by the `pages` concurrency group.

Two things about the snapshots shape the output:

- **They overlap, unevenly.** The nightly collects the previous 24h, but it is
  also run by hand, so six snapshots landed on 2026-10-29 (94-99% the same
  commits) against one on 2026-10-26. Plotting one point per snapshot would
  chart how often the collector ran. Instead every snapshot is pooled, commits
  are deduplicated by `(lane, head sha)`, and each commit is bucketed by its
  own arrival time, in the same CDT the report uses.
- **Some days are only partly covered.** A day the snapshots do not fully cover
  is marked `complete: false` and drawn provisionally rather than dropped, so a
  half-collected day never reads as a sudden improvement. Nightlies drift by a
  few minutes, so gaps under `GAP_TOLERANCE` are not treated as missing data.

`PHASE_GROUPS` folds the nine `PHASES` into seven for the stacked chart (seven
is the point past which adjacent colours stop being tellable apart); the groups
still cover `PHASES` exactly, and an `assert` fails if a phase is ever added
without being placed in one. The stack uses **means**, not medians: medians of
the parts would not sum to the median of the whole.

Low-sample days are kept but drawn hollow: weekends really do fall to a handful
of commits, where a p90 is one slow run. The count is in every tooltip.

Both tabs are always shown, including against a `trends.json` that predates
them: the view then says which key its data lacks and when that file was
generated, rather than disappearing. A tab that is simply absent looks
identical to a deploy that never happened, which is a slow thing to diagnose --
the site and its `trends.json` are built by two different steps of
`build-pages-site`, from whatever `ci-metrics/` is on `rocm-dev-infra`.

The page has two tabs over that one day axis, sharing the lane filter:

- **Timing** -- time to signal, queue wait, where the time goes, commit volume
  and pass rate. The percentile chips scope this tab only.
- **Tests** -- how many tests one test job of each lane runs, from the
  `Executed X out of Y tests` lines `sample_logs.py` pulled out of the logs: a
  job's suite is the sum of `Y` over its bazel invocations, which is the suite
  size `report.coverage()` reports, taken per job instead of per 48h window.
  One series per suite, so the `openxla/xla` lanes are split into their XLA and
  JAX job families (`TEST_CATS`); the family is drawn as a dash on the lane's
  own colour, never as a new hue, so a lane keeps one identity across both tabs.
  Cached tests count -- the suite is what the lane covers, not what it had to
  re-run.

  The day's figure is the **largest** of its sampled jobs, not a median or a
  mean. A job that fails early reports only the tests it reached (`Executed 1
  out of 1 tests` is a common sample), and the XLA lanes run a second bazel
  invocation that not every job gets to, so a central statistic measures how
  the day's jobs ended rather than how big the suite is: on 2026-11-11 the
  median for the ROCm XLA jobs is 549 where the Markdown report publishes
  576 (419 + 157) -- which is the max. It is still a max over a *sample*, so a
  thin day can fall short of the real suite; `n` is in every tooltip and those
  days are drawn hollow. The table carries the median beside it, where the gap
  between the two is the day's truncation rate.

  Never a sum: `sample_logs.py` caps how many logs it downloads per category,
  so a total would plot that cap rather than the lane.

  The same tab carries **what the p90 is made of**: per day and lane, the
  slowest three commits at or above that lane's p90 time to signal, with the
  cut, how many commits sat at or above it, how many of those the other lane
  also put in its tail, and the phase group that took most of each wait. A percentile is a position in a sorted list, so those commits
  are the tail -- move them and the p90 moves. `TAIL_PCT` must stay the
  percentile the page plots as its tail, or the table would explain a number
  that is not on screen.

  Naming them has to work offline, because the Pages job has no token: a branch
  PR's run carries its PR number, a fork or copybara-export run carries none,
  and for those the row falls back to the head branch linked to the workflow
  run, which names the pull request on its own page. `report.build()` therefore
  keeps `run` and `branch` on every commit. Whatever fork PRs the snapshots
  already resolved into their `pr_lookup.json` (the report looks a few up for
  its own tables) are merged in and shown as numbers; nothing new is fetched.
  On a representative fortnight that is about half the tail by number.

As the data branch grows (~14 MB a night), `--window-days N` limits the read to
recent snapshots, which also allows a narrower sparse checkout in CI.

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
- **The Metrics table's three columns.** `DO` and `RBE` are the two halves of the
  ROCm lane: the self-hosted runner and the EngFlow pool its tests dispatch to.
  `NV` is the same metric on the NVIDIA reference lane, so a row reads across as
  "us, where our tests actually run, and them". It is `n/a` where that lane has
  no counterpart (the ROCm-only nightlies). NV queue wait covers the NV
  presubmit lane's jobs — only that workflow is collected, so unlike the DO cell
  it is not every NV job in the repo.
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
- **The p90 tail.** The commits of a lane-day whose time to signal is at or
  above that day's p90 for that lane. They are what the percentile is made of,
  so the dashboard names the slowest three; a commit's "mostly" is the phase
  group that took the largest share of its wait. One PR can own several, one
  per head commit that reached a signal.
- **ROCm and NV tails compared.** A repo's two lanes run on the same head
  commits, so each tail commit is also placed in the other lane: in its tail
  too (`both`), signalled there but outside it (`only`), or never signalled
  there at all. `only` is the case the page marks, because the lanes disagree
  about one commit and the difference is therefore the lane, not the PR. It
  does not mean the commit was fast on the other lane -- only that it was not
  among that lane's own slowest tenth. `shared` counts the whole tail this way,
  not just the three named, and is computed over all lanes regardless of what
  the page's lane filter is showing.
- **Re-runs.** When "re-run failed jobs" copies finished jobs into a new attempt,
  the copy keeps the old start/end but gets a new creation time. Those copies are
  dropped (`carried_over()` in `metrics_lib.py`).
- **Test counts and cost per test.** Taken from the sampled logs, not from every
  job. Seconds per test only uses runs that executed at least 90% of the suite.
  The report takes the suite size from the most common shape in the last 48h;
  the dashboard's Tests tab takes the same per-job count a day at a time, as
  the largest in that day's sample. Both answer "how big is the suite", so they
  agree to within a test or two; a median would not, because the sample
  includes jobs that ended before the suite did.

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
- **Wording or layout of the report.** Edit `templates/report.md.j2` and nothing
  else. Reordering sections, rewording a sentence, restyling or moving a table,
  even dropping a column, is a template change. Only a genuinely new *figure*
  needs `report.py` to compute and add it to the dict.
- **Time zone.** `CDT` in `metrics_lib.py` is fixed at UTC-5. Change it to UTC-6
  for winter windows.
- **Hand-written text.** Some sections are not computed: the "pending" rows under
  Infrastructure, "Executor queue by pool", "The ask", and the notes about bazel
  strategies. They sit in the template as literal Markdown — review them before
  sharing a report.
- **Week-over-week comparison.** `report.py --baseline BASE.json` adds a "Changes
  since" section from an earlier report's figures. Pass last week's `report.json`
  (written by `--json`); the keys it reads are in the `report.py` docstring.

## Limits

- GitHub expires job logs (its default retention is 90 days), so `sample_logs.py`
  only works on recent windows. Run and job metadata last longer.
- The venodr lanes run on Google-hosted runners and Google RBE. The report sees them
  only through GitHub job and step timestamps and their logs.
