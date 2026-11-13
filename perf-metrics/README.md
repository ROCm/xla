# perf-metrics: the trend dashboard for the ROCm perf nightly

[`rocm_xla_perf_nightly.yml`](../.github/workflows/rocm_xla_perf_nightly.yml)
measures one night at a time and publishes it to the `perf-metrics-data`
branch. This turns every published night into one page:
<https://rocm.github.io/xla/perf-metrics/>.

The measuring lives in [`perf_tools/`](../perf_tools) and nothing here
re-measures anything. `trends.py` reads what the nightly wrote and computes
only ratios, sums and the noise floor those ratios are read against; the page
draws that and adds no definitions of its own.

> **Not the CI trends.** The site carries two trend dashboards, and this is the
> one about **performance**: how fast the benchmarked code runs, per ROCm
> stack. Its sibling [`ci-metrics/`](../ci-metrics) is about how long CI takes
> to answer a pull request -- different question, different data branch,
> different collector, and a point there is one day pooled from overlapping
> snapshots rather than one benchmark run. The two pages are deliberately built
> the same way -- same chart primitives, same filter row, same tabs -- so that
> knowing one is knowing both, and they share nothing else: neither reads the
> other's data, and no figure on one has a counterpart on the other. They are
> compared side by side in the [root README](../README.md#dashboards).

## What the nightly publishes

One directory per run, named `<stamp>-<sha6>`:

```
data/20261113T021256Z-ced35e/
  run.json                              timestamp, the openxla/xla commit, the matrix
  therock_release_10.0/hlo_eval.tsv     model, leaf, module, duration_ms
  therock_release_10.1/hlo_eval.tsv
  therock_nightly/hlo_eval.tsv
  therock_nightly/latency.csv           the HIP latency tool, one row per report
```

All three ROCm stacks replay the same HLO modules against the **same** pinned
`openxla/xla` commit, which is what makes a difference between them a statement
about the stack. Which stack each one *was* is in `run.json`: `matrix` records
the whole benchmark matrix as `rocm-config` built it -- every mode with the
TheRock release it pulled, the tarball URL and the digest that pinned it --
while `modes` lists the subset that actually published results, so a mode in
one list and not the other is a mode that failed. Nothing in the TSV or CSV
rows carries that, which is why the manifest has to.

`latency.csv` is written for the newest stack only: it
characterises the ROCm runtime rather than the XLA tree, so measuring it on the
release stacks would only re-measure the same interconnect.

> **The stamped columns carry a stray `$`.** The nightly prepends the
> timestamp, commit and mode to each row with `awk -v OFS="$sep"`, where the
> workflow passes `$'\t'` as `sep`; the shell that step runs under does not
> interpret that quoting, so `OFS` becomes `$` followed by a tab and every
> stamped field is published as `therock_nightly$`. `clean()` in `trends.py`
> strips it, which reads the snapshots already on the branch and is a no-op on
> a clean field. Fixing the nightly would not remove the need for it.

## Run it

```bash
git clone --branch perf-metrics-data --depth 1 <repo> /tmp/perf-data
python3 trends.py --data /tmp/perf-data/data --out site/trends.json
python3 -m http.server -d site 8000     # then open localhost:8000
```

Python 3.10 or newer, standard library only -- no Jinja2, no pip install. That
is why the Pages job can build this section without installing anything.
`--window-days N` trims the read to recent runs, which is what would let a
caller sparse-checkout only those.

## The figures

**One point per run, not per day.** `ci-metrics` pools its snapshots by day
because their 24h windows overlap unevenly. Perf runs do not overlap at all:
each is one measurement of one pinned commit, and that commit is what a
regression gets blamed on. Two runs on one day are two points, and the x axis
is labelled by date, with the UTC time added only where a date repeats.

Three figures describe a run, because none of them is honest alone:

| | What it is | What it hides |
|---|---|---|
| **Index** | `100 x geomean(d / d_baseline)`. The typical module, with alphafold3 at 1.6s counting exactly as much as detr at 1.4ms. | That one big module can swing the suite's real cost without moving this. |
| **Total** | The plain sum. What the suite actually costs. | That it is mostly the three heavy modules (alphafold3, stable diffusion 1.5, sdxl). |
| **Per module** | Every module's own line, one panel each, split into the inference and training suites. | Nothing -- this is where a regression is actually visible, which is why it gets a tab per suite. |

A geometric mean because these are ratios: an arithmetic mean of a module that
halved (0.5) and one that doubled (2.0) is 1.25, reporting a 25% regression
where nothing changed. In logs they cancel.

**The core set** of a mode is the modules its *latest* run measured. Anchoring
on the latest run rather than on the intersection of every run keeps one
truncated night from permanently shrinking the suite, and keeps a module added
today from being excluded because last week's runs predate it. A run that did
not cover the whole core set reports **no total at all** -- a sum over a subset
is smaller than a sum over the set, which on a chart is indistinguishable from
things getting faster. The index tolerates the same run, because it averages
ratios rather than adding them; those points are drawn hollow and carry their
module count in the tooltip.

**The noise floor** is the 90th percentile of every module's change between
consecutive runs: about 2.4% on the release stacks and 3.1% on nightly, against
medians near 0.5%. The tool times each module once per run, so that spread is
the only evidence available for what a quiet night looks like, and it is the
level a per-module line has to clear to be worth chasing. Every card that plots
a module quotes it, because the panels are close-ups and a close-up makes 1%
look like an event. A window of one run has no floor at all -- there is no
previous run to difference against -- and the page says so rather than leaving
the sentence blank.

**Reading a per-module panel.** Each has its **own** y axis, bounded by its own
highest and lowest readings. The suite runs from detr at 1.4ms to alphafold3 at
1597ms, so a shared scale would flatten all but the heaviest into one flat line;
the cost is that panels are comparable with themselves over time and never with
each other. That is the division of labour on the page: the Models tab compares
modules with each other, these tabs compare a module with its own past.

The branch shows what that is for. qwen3 `module_0082` went to 451ms on
2026-11-07 and back to 44.7ms the next night, while reading 44.6-45.0ms on every
other run and on both release stacks -- a single-stack excursion on a shared
runner, which is what nearly every large move on these charts turns out to be.
A module that jumps and *stays* is the other kind, and the panel shows the
difference at a glance where an aggregate would not.

**Latency** is `rocm_latency_check_tool`, read from its `final` rows only --
the tool also appends an `interval` row every few seconds, and pooling those
would weight a run by how long it happened to take. **Only the nightly stack is
plotted**: the figure characterises the ROCm runtime rather than the XLA tree,
so the workflow runs the tool on the newest stack alone, and the three early
runs that measured all three stacks are discarded rather than drawn as lines
that stop in November and never resume. The stack filter therefore does not
apply to that tab and is hidden on it. The p50 holds steady near 4ms; the **p99
is a handful of requests** and one stall moves it by an order of magnitude
(4.4ms to 67ms on 2026-11-12), so the page defaults to p50 and says so.

The `rocm_version` column is not used: the tool writes the `--label` value into
it, so on these snapshots it reads `therock_nightly`. `hip_version` beside it is
correct and *is* read for every stack, including the ones whose timings are
discarded -- a pinned release's HIP version is a fact that does not go stale, so
it still names the build behind each stack chip.

## The page

```
site/index.html   the document: every heading and every sentence of guidance
site/app.js       plain SVG built in the DOM -- no chart library, no build step
site/styles.css   a copy of ci-metrics/site/styles.css, perf rules appended
```

Four tabs over one stack filter:

- **Models** -- the index, the total, and the total split by model category.
- **Inference** and **Training** -- one small chart per HLO module of that
  suite, grouped by model category, one line per stack. Two tabs rather than
  one list because they are two suites: training HLO carries the backward pass,
  the gradient all-reduce and the optimizer update, so a training module has no
  inference counterpart to sit beside. A tab whose suite the data does not carry
  is removed rather than shown empty.
- **Latency** -- the HIP latency tool, nightly stack only. The stack filter is
  hidden here, since it would scope nothing.

`styles.css` is a copy rather than a shared file. The two dashboards are
separate sections of the Pages site deployed by separate workflows, a shared
stylesheet would let a restyle of one silently repaint the other, and each
`site/` has to be servable on its own for the preview command above to work.

A view whose `trends.json` lacks its keys says so rather than disappearing: the
page and its data are deployed together but built by different steps, and a
tab that is simply absent looks exactly like a deploy that never happened.

## How it reaches the web

[`perf_metrics_pages.yml`](../.github/workflows/perf_metrics_pages.yml) deploys
after every perf nightly, on a push to `perf-metrics/`, or on demand.

GitHub Pages serves one deployment per repository and a deployment replaces the
whole site, so **all three** Pages workflows -- `scaling_book_pages.yml`,
`ci_metrics_pages.yml` and this one -- build **all three** sections through the
shared [`build-pages-site`](../.github/actions/build-pages-site/action.yml)
action. A workflow that published only its own section would delete the others.
The three are serialised by the `pages` concurrency group.

Two things about the trigger are worth knowing:

- It deploys on a **failed** perf nightly too, and only skips a cancelled one.
  A nightly that loses one of its three matrix modes still publishes the modes
  that finished -- `publish-perf-metrics` runs `if: always()` -- so a `failure`
  conclusion routinely carries new results.
- `workflow_run` only fires for workflow files on the **default branch**, so
  until this file is merged there the `push` and `workflow_dispatch` triggers
  are what reach the page.
