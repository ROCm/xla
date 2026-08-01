# HLO Stability Tools

Standalone one-HLO stability collection, analysis, and reporting for XLA/ROCm.
The tool builds its own immutable runners by default and can reuse only native
runner bundles produced by a previous stability run.

Its sole cross-tool executable interface is:

```text
perf_tools/hlo_eval_tools/run_hlo_eval.sh
```

The stability tool does not read multi-branch campaign manifests, import the
campaign orchestrator, or modify the HLO evaluation workflow.
For provenance and mutation detection, it also fingerprints the evaluator's
internal `legacy_profile_to_csv.py` dependency without invoking that helper
directly.

## Layout

```text
hlo_stability_tools/
  README.md
  PHASE_II_CANDIDATE_EVALUATION.md
  configs/
    stability_profile.json
    xla_targets.json
    xla_targets.template.json
  examples/
    run_4target_24round.sh
  scripts/
    run_hlo_stability.py
    xla_runner_bundle.py
    hlo_stability.py
    analyze_hlo_stability.py
    render_hlo_stability_report.py
    show_hlo_stability_status.py
    file_util.py
    test_hlo_stability.py
```

## Build runners and collect evidence

Use a clean, dedicated XLA source checkout. The collector checks out each
resolved commit sequentially, builds
`//xla/tools/multihost_hlo_runner:hlo_runner_main`, copies each executable into
the output bundle, and restores the original checkout on success, failure, or
interruption.

The advisory lock file is intentionally compatible with other XLA runner-build
tools so two processes cannot switch the same source checkout concurrently.
This is lock coordination only; no campaign code or schema is imported.

From the XLA/perf-tools repository root:

```bash
python3 perf_tools/hlo_stability_tools/scripts/run_hlo_stability.py \
  --xla-source-repo /path/to/clean/source-xla \
  --output-dir /path/to/new-stability-output \
  --hlo-path /path/to/one_module.txt \
  --targets-file /path/to/xla_targets.json \
  --rounds 24 \
  --reference-csv /path/to/optional_historical.csv
```

`--targets-file` defaults to `configs/xla_targets.json`. It contains one to
three candidate branches, tags, or commits. The pinned live control and fixed
runner protocol are owned by `configs/stability_profile.json`.

Copy `configs/xla_targets.template.json` when starting a focused branch/commit
experiment. A ready-to-run four-target example is also available:

```bash
OUTPUT_ROOT=/workspace/debug_space \
bash perf_tools/hlo_stability_tools/examples/run_4target_24round.sh \
  /workspace/codeRepo/xla_debug/xla
```

The output directory must be absent or empty and outside both Git checkouts.
The source checkout must have no tracked, untracked, or ignored-file conflicts;
the tool never stashes, resets, cleans, or deletes source changes.
Unless `--skip-fetch` is used, runner preparation fetches referenced remotes
with `--prune` and adds the public OpenXLA `upstream` remote when required.
Branch/commit/worktree state is restored; fetched remote-tracking refs and an
added remote are intentionally retained Git administrative changes.

## Reuse a native runner bundle

To skip Git checkouts and Bazel builds:

```bash
python3 perf_tools/hlo_stability_tools/scripts/run_hlo_stability.py \
  --runner-bundle /path/to/previous-output/runner_bundle \
  --output-dir /path/to/new-stability-output \
  --hlo-path /path/to/one_module.txt \
  --rounds 24
```

Only a completed `hlo_stability_runner_bundle` is accepted. An optional
`--targets-file` can select a subset of candidates already present in that
bundle. Every commit identity, executable path, and runner SHA256 is validated
before collection.

Interrupted or failed runs retain their bundle, metadata, logs, round order,
and completed raw measurements. Collection itself does not resume; retry with a
new output directory and reuse the completed bundle when available.

Run long sessions inside `tmux`. SIGHUP is handled as an interruption, so
`nohup` is not the recommended launcher.

## Progress and logs

Console phase, target, commit, and log-path messages are flushed immediately.
Long runner builds and evaluations emit a heartbeat every 30 seconds.

Follow structured progress with:

```bash
python3 perf_tools/hlo_stability_tools/scripts/show_hlo_stability_status.py \
  --output-dir /path/to/stability-output \
  --follow
```

Detailed logs follow the verified HLO evaluator layout:

```text
<output>/runner_bundle/<target>/build.log
<output>/runner_bundle/<target>/metadata.json
<output>/warmup/<role>/eval.log
<output>/<role>/round_NN/eval.log
```

The frontend log identifies the active branch and prints the relevant detailed
log path. Bazel and runner output remain in separate files so the console stays
readable.

## Measurement policy

Defaults:

- 12 balanced measured rounds; use 24 for stronger confirmation.
- One unrecorded warmup evaluation per target.
- 8 seconds between target evaluations.
- 30 seconds between complete rounds.
- 2 seconds after each runner process for GPU resource cleanup.

Round counts must contain complete schedule cycles:

- One candidate plus control: multiple of 2.
- Two candidates plus control: multiple of 6.
- Three candidates plus control: multiple of 4.

The runner repeat count is fixed at two to match the reference protocol. The
first repeat is the internal runner warmup and is excluded. Statistical
repetition is controlled by `--rounds`, not a runner-repeat CLI option.

`--reference-csv` is optional. It adds historical context for the same HLO but
never defines a pass/fail threshold.

## Optional system context

Enable coarse context capture with:

```bash
--capture-system-snapshots
```

Before and after each measured evaluation, the collector records `uptime` and
point-in-time `rocm-smi` utilization, memory, clocks, power, temperature, and
process IDs. Snapshot failures do not fail collection, and snapshot content is
not used by the evidence classifier.

These samples can support manual outlier investigation, but they can miss
transient activity and do not replace rocprofv3 or Compute Viewer. Use dedicated
profiling after balanced evidence establishes a reproducible target/HLO delta.

## Evidence outputs

```text
<output>/
  runner_bundle/                    # build mode only; reusable
    manifest.json
    <target>/runner/hlo_runner_main
  experiment_metadata.json
  round_orders.csv
  <role>/round_NN/csv/*.csv
  <role>/round_NN/eval.log
  <role>/round_NN/system_before.txt  # optional
  <role>/round_NN/system_after.txt   # optional
  stability_analysis.json
  stability_summary.csv
  raw_rounds_long.csv
  paired_deltas.csv
  stability_report.html
```

Raw samples are never deleted. Outliers are flagged, and paired comparisons
exclude a round only when either side of that pair is flagged. Reports are
descriptive evidence, not release certification or CI gating. A frequent
outlier pattern (at least three samples and at least 10% of rounds) is reported
as overall instability even when the retained clean mode remains within the
candidate/control reporting band. Reports show both raw and clean CV, paired
exclusion rate, early/late outlier incidence, and byte-identical runner
warnings.

The analyzer and renderer can be rerun independently:

```bash
python3 perf_tools/hlo_stability_tools/scripts/analyze_hlo_stability.py \
  --experiment-dir /path/to/stability-output \
  --reference-csv /path/to/optional_historical.csv

python3 perf_tools/hlo_stability_tools/scripts/render_hlo_stability_report.py \
  --experiment-dir /path/to/stability-output
```

## Tests

```bash
python3 -m unittest discover -v \
  -s perf_tools/hlo_stability_tools/scripts \
  -p 'test_*.py'
```

The process-group termination test is POSIX-only and must pass in the target
Linux/ROCm environment before a full verification run.
