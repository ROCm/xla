#!/usr/bin/env python3
"""Roll the perf-nightly runs up into one run-by-run trend series.

  python3 trends.py --data <perf-metrics-data checkout>/data --out trends.json

`rocm_xla_perf_nightly.yml` publishes one `data/<stamp>-<sha6>/` directory per
run, holding `<mode>/hlo_eval.tsv` (one duration per HLO module) and
`<mode>/latency.csv` (the HIP latency tool). This turns every published run into
the series the dashboard plots, and computes nothing the raw files do not say
beyond ratios and the noise floor those ratios are read against.

One point per *run*, not per day. The sibling `ci-metrics/trends.py` pools its
snapshots by day because their 24h windows overlap unevenly; perf runs do not
overlap at all -- each is one measurement of one pinned `openxla/xla` commit,
and that commit is the thing a regression gets blamed on. Two runs on one day
are therefore two points, and the x axis is labelled by date only where the date
changes.

Three figures describe a run, because no single one is honest on its own:

  index     geomean of each module's ratio to its baseline. The typical module,
            with alphafold3 (1597ms) counting exactly as much as detr (1.4ms).
  total_ms  the sum. What the suite actually costs, and therefore dominated by
            the three heavy modules.
  movers    the modules that moved, which is the only view that names a cause.

Nothing is called a move until it clears `noise_pct`, the p90 of the
module-level run-to-run change -- about 2.4% on the release stacks and 3.1% on
nightly. The tool reports one duration per module per run, so that spread is the
only evidence available for what run-to-run quiet looks like.
"""
import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import statistics

# The modes the dashboard plots, in a fixed order. The order is also the
# categorical colour order on the page, so a mode keeps its hue no matter how
# many modes a filter leaves on screen. Oldest stack first, so the colours run
# in the same direction as the versions.
MODES = (
    ("therock_release_10.0", "ROCm 10.0"),
    ("therock_release_10.1", "ROCm 10.1"),
    ("therock_nightly", "ROCm nightly"),
)

# The model categories, as `perf_tools/hlo_eval_tools/` lays them out on disk.
# A model name is `<category>_<model>`, and the categories themselves contain
# underscores, so this is a prefix match against the known set rather than a
# split -- `vision_diffusion_vit` is one model in `vision_diffusion`, not a
# model called `diffusion_vit` in `vision`. Longest first, so a category that
# is a prefix of another could never swallow it.
GROUPS = (
    ("large_language_models", "Language models"),
    ("vision_diffusion", "Vision + diffusion"),
    ("multimodal", "Multimodal"),
    ("science", "Science"),
)

# The latency tool's figures, as `rocm_latency_check_tool.cpp` names them.
# p50 leads: it is the one that holds still. The p99 of a 10-second run is a
# handful of requests, and a single stall moves it by an order of magnitude
# (4.4ms -> 67ms on 2026-11-12), so it reports stalls, not latency.
LATENCY_METRICS = (
    ("p50_us", "Median (p50)", "us"),
    ("p99_us", "Tail (p99)", "us"),
    ("mean_us", "Mean", "us"),
    ("qps", "Throughput", "qps"),
)

# How many movers one run's row can show without the table becoming the page.
# The count of everything that moved travels with them, so a run where more
# moved is not mistaken for one where it did not.
MOVERS_SHOW = 3

# The percentile of the module-level run-to-run change that counts as noise.
# At p90, one module-pair in ten clears the floor on a quiet night, which is
# what keeps a quiet night from filling the movers table.
NOISE_PCT = 90


def pct(values, p):
    """The p-th percentile, nearest-rank, of an unsorted list."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def group_of(model):
    """The category a model name belongs to.

    An unknown model is kept under its own first segment rather than dropped:
    a model added to `perf_tools/hlo_eval_tools/` under a new category should
    appear on the page before anyone remembers to edit `GROUPS`.
    """
    for key, _ in GROUPS:
        if model == key or model.startswith(key + "_"):
            return key
    return model.split("_", 1)[0]


def clean(s):
    """Strip the stamping step's stray `$` from one field.

    `rocm_xla_perf_nightly.yml` stamps each row by prepending the timestamp,
    commit and mode with `awk -v OFS="$sep"`, where the workflow passes `$'\\t'`
    as `sep`. The shell that step runs under does not interpret that quoting, so
    `OFS` ends up as `$` followed by a tab and every stamped field is published
    as `therock_nightly$`. Reading it off here keeps the dashboard working on
    the snapshots already on the branch; it is a no-op on a clean field.
    """
    return s.strip().rstrip("$")


def read_table(path, delimiter):
    """One delimited file as a list of dicts, with the `$` taken off."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter=delimiter))
    if not rows:
        return []
    header = [clean(h) for h in rows[0]]
    return [dict(zip(header, (clean(v) for v in r))) for r in rows[1:] if any(r)]


def read_runs(data_dir):
    """Every published run, oldest first.

    Directory names are `<stamp>-<sha6>` as the nightly writes them
    (`20261113T021256Z-ced35e`), so a lexical sort is chronological. `run.json`
    is the manifest the publish step wrote; a directory without one is not a
    run directory and is left alone.
    """
    runs = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        manifest = os.path.join(path, "run.json")
        if not os.path.isdir(path) or not os.path.exists(manifest):
            continue
        with open(manifest) as f:
            meta = json.load(f)
        runs.append({"name": name, "path": path, "meta": meta})
    return runs


def read_modules(run):
    """`{mode: {(model, leaf, module): duration_ms}}` for one run.

    Keyed by the whole triple because the module id alone is not unique:
    `module_3910` is a different module in `multimodal_lit` than it is in
    `vision_diffusion_clip`, and both are measured.
    """
    out = {}
    for mode, _ in MODES:
        path = os.path.join(run["path"], mode, "hlo_eval.tsv")
        if not os.path.exists(path):
            continue
        durations = {}
        for row in read_table(path, "\t"):
            try:
                durations[(row["model"], row["leaf"], row["module"])] = float(row["duration_ms"])
            except (KeyError, ValueError):
                continue  # a row the runner could not time; absent, not zero
        if durations:
            out[mode] = durations
    return out


def read_latency(run):
    """`{mode: {num_gpus: row}}` for one run, from the `final` rows only.

    The tool appends a row every `--report-interval-sec` as well as one summary
    row per invocation; the interval rows are snapshots of the same run, so
    pooling them would weight a run by how long it happened to take. `final` is
    the figure the tool prints as its result.
    """
    out = {}
    for mode, _ in MODES:
        path = os.path.join(run["path"], mode, "latency.csv")
        if not os.path.exists(path):
            continue
        by_gpus = {}
        for row in read_table(path, ","):
            if row.get("phase") != "final":
                continue
            try:
                by_gpus[int(row["num_gpus"])] = row
            except (KeyError, ValueError):
                continue
        if by_gpus:
            out[mode] = by_gpus
    return out


def geomean(ratios):
    """The geometric mean -- the right average for a set of ratios.

    An arithmetic mean of ratios is not symmetric: a module that halves (0.5)
    and one that doubles (2.0) average to 1.25, reporting a 25% regression
    where nothing changed. In logs they cancel, which is what this does.
    """
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None


def core_set(per_run, mode):
    """The modules a mode's series is defined over: those its latest run timed.

    Anchoring on the latest run rather than on the intersection of every run
    keeps one truncated night from permanently shrinking the suite, and keeps
    a module added today from being excluded because last week's runs predate
    it.
    """
    for run in reversed(per_run):
        if mode in run:
            return set(run[mode])
    return set()


def noise_floor(per_run, mode, core):
    """The run-to-run change a quiet night produces anyway, as a fraction.

    Every module measured by two consecutive runs contributes one |change|, and
    the p90 of that pile is the floor. It is a property of the measurement, not
    of any one night: the modules are replayed in isolation on a shared runner,
    and nothing here reports a variance that could be used instead.
    """
    deltas = []
    for before, after in zip(per_run, per_run[1:]):
        a, b = before.get(mode, {}), after.get(mode, {})
        for key in core:
            if key in a and key in b and a[key] > 0:
                deltas.append(abs(b[key] / a[key] - 1))
    return pct(deltas, NOISE_PCT)


def baselines(per_run, mode, core):
    """Each core module's first measurement, and which run that was.

    The index is read as "against where this window started", so the baseline
    is per module: a module first timed halfway through the window still gets
    an index from that point rather than dropping the whole run's geomean.
    """
    base = {}
    for i, run in enumerate(per_run):
        for key, value in run.get(mode, {}).items():
            if key in core and key not in base and value > 0:
                base[key] = (value, i)
    return base


def mode_series(per_run, mode):
    """One mode's whole series: index, totals, per-group totals, coverage."""
    core = core_set(per_run, mode)
    base = baselines(per_run, mode, core)
    groups = [key for key, _ in GROUPS if any(group_of(k[0]) == key for k in core)]

    out = {
        "index": [],
        "total_ms": [],
        "n": [],
        "complete": [],
        **{f"group_{g}": [] for g in groups},
    }
    for run in per_run:
        durations = run.get(mode, {})
        measured = {k: v for k, v in durations.items() if k in core}
        whole = len(measured) == len(core) and bool(core)

        out["n"].append(len(measured) or None)
        out["complete"].append(whole)
        # A sum over a subset is smaller than a sum over the set, which on a
        # chart is indistinguishable from things getting faster. A run that did
        # not cover the suite reports no total at all.
        out["total_ms"].append(sum(measured.values()) if whole else None)
        for g in groups:
            out[f"group_{g}"].append(
                sum(v for k, v in measured.items() if group_of(k[0]) == g) if whole else None)
        # The index tolerates a partial run: it averages ratios, so a missing
        # module drops out of the average instead of deflating it. `n` says how
        # many modules are behind the point.
        ratios = [v / base[k][0] for k, v in measured.items() if k in base and v > 0]
        out["index"].append(geomean(ratios) * 100 if ratios else None)

    out["noise_pct"] = noise_floor(per_run, mode, core)
    out["core"] = len(core)
    return out, groups


def module_key(key):
    """One module's id in `trends.json`: `<model>|<leaf>|<module>`.

    A flat string because it is a JSON object key. `|` is the separator because
    none of the three parts can contain one -- they come from directory and
    file names under `perf_tools/hlo_eval_tools/`.
    """
    return "|".join(key)


def movers(per_run, mode, core, base, noise, others):
    """What moved between each run and the one before it, newest change first.

    A percentile tail answers "which commits are slow"; this answers the
    question a perf series is actually opened for, "what changed last night".
    Each run reports every core module whose change from the previous run
    clears the noise floor, and names the largest `MOVERS_SHOW` of them.

    Each named module carries where the *other* stacks stand on it in the same
    run, which is the difference between an XLA regression and a ROCm one:

      `all`    every stack that measured this module moved the same way, so the
               cause is the thing they share -- the XLA commit under test.
      `only`   the other stacks measured it and held still. The stack is the
               variable, and this is the row worth chasing.
      None     no other stack measured it this run, so there is nothing to
               compare against.
    """
    rows = []
    for i, run in enumerate(per_run):
        now = run.get(mode, {})
        prev = per_run[i - 1].get(mode, {}) if i else {}
        if not now or not prev or noise is None:
            rows.append(None)
            continue

        moved = []
        for key in core:
            if key not in now or key not in prev or prev[key] <= 0:
                continue
            delta = now[key] / prev[key] - 1
            if abs(delta) > noise:
                moved.append((key, delta))
        if not moved:
            rows.append({"n": 0, "noise": noise, "modules": []})
            continue
        moved.sort(key=lambda m: -abs(m[1]))

        named = []
        for key, delta in moved[:MOVERS_SHOW]:
            elsewhere = []
            for other in others:
                a, b = per_run[i - 1].get(other, {}), run.get(other, {})
                if key in a and key in b and a[key] > 0:
                    elsewhere.append(b[key] / a[key] - 1)
            if not elsewhere:
                also = None
            elif all(abs(d) > noise and (d > 0) == (delta > 0) for d in elsewhere):
                also = "all"
            else:
                also = "only"
            named.append({
                "key": module_key(key),
                "model": key[0],
                "leaf": key[1],
                "module": key[2],
                "group": group_of(key[0]),
                "ms": now[key],
                "was": prev[key],
                "delta": delta,
                "also": also,
                # Where it stands against the start of the window: a module
                # that comes straight back the next night was a blip, and one
                # that stays put is the regression.
                "vs_base": (now[key] / base[key][0] - 1
                            if key in base and base[key][0] > 0 else None),
            })
        rows.append({"n": len(moved), "noise": noise, "modules": named})
    return rows


def module_table(per_run, modes):
    """Every core module's whole series, per mode, for the table and the picker.

    This is the raw material the two aggregates are made of, published as it is
    so that any figure on the page can be traced back to the durations the
    runner wrote.
    """
    keys = sorted({k for mode in modes for k in core_set(per_run, mode)})
    return (
        [{"key": module_key(k), "model": k[0], "leaf": k[1], "module": k[2],
          "group": group_of(k[0])} for k in keys],
        {mode: {module_key(k): [run.get(mode, {}).get(k) for run in per_run]
                for k in keys}
         for mode in modes},
    )


def latency_series(per_latency, modes):
    """`{"<mode>:<gpus>": {metric: [...]}}` plus the GPU counts in play."""
    gpu_counts = sorted({g for run in per_latency for by in run.values() for g in by})
    series = {}
    for mode in modes:
        for gpus in gpu_counts:
            row = {metric: [] for metric, _, _ in LATENCY_METRICS}
            for run in per_latency:
                found = run.get(mode, {}).get(gpus)
                for metric, _, _ in LATENCY_METRICS:
                    try:
                        row[metric].append(float(found[metric]))
                    except (TypeError, KeyError, ValueError):
                        row[metric].append(None)
            if any(v is not None for v in row["p50_us"]):
                series[f"{mode}:{gpus}"] = row
    return gpu_counts, series


def hip_versions(per_latency, modes):
    """The HIP runtime each mode measured with, per run.

    Only the latency tool records it, so this is the single place the page can
    say *which* ROCm a point was measured on. The `rocm_version` column beside
    it is not used: the tool writes the `--label` value into it, so on these
    snapshots it reads `therock_nightly`.
    """
    return {mode: [run.get(mode, {}).get(min(run[mode]), {}).get("hip_version")
                   if run.get(mode) else None
                   for run in per_latency]
            for mode in modes}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="the `data/` directory of a perf-metrics-data checkout")
    ap.add_argument("--out", required=True, help="where to write trends.json")
    ap.add_argument("--window-days", type=int, default=None,
                    help="only read runs from the last N days (default: all)")
    args = ap.parse_args()

    runs = read_runs(args.data)
    if args.window_days is not None:
        cut = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.window_days)
        runs = [r for r in runs
                if dt.datetime.strptime(r["name"].split("-")[0], "%Y%m%dT%H%M%SZ")
                .replace(tzinfo=dt.timezone.utc) >= cut]
    if not runs:
        raise SystemExit(f"no run directories under {args.data}")

    per_run = [read_modules(r) for r in runs]
    per_latency = [read_latency(r) for r in runs]

    # Only the modes that actually appear, so a mode retired from the matrix
    # stops being plotted rather than drawing an empty series forever.
    modes = [m for m, _ in MODES if any(m in r for r in per_run)]
    if not modes:
        raise SystemExit("no hlo_eval.tsv found in any run")

    series, groups = {}, []
    for mode in modes:
        series[mode], mode_groups = mode_series(per_run, mode)
        for g in mode_groups:
            if g not in groups:
                groups.append(g)

    move_rows = {}
    for mode in modes:
        core = core_set(per_run, mode)
        move_rows[mode] = movers(per_run, mode, core,
                                 baselines(per_run, mode, core),
                                 series[mode]["noise_pct"],
                                 [m for m in modes if m != mode])

    modules, module_values = module_table(per_run, modes)
    gpu_counts, latency = latency_series(per_latency, modes)

    labels = dict(MODES)
    group_labels = dict(GROUPS)
    doc = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository": runs[-1]["meta"].get("repository", "openxla/xla"),
        # One entry per published run, oldest first: the x axis of every chart.
        "runs": [{
            "stamp": r["name"],
            "iso": r["meta"].get("timestamp"),
            "day": (r["meta"].get("timestamp") or "")[:10],
            "commit": r["meta"].get("commit"),
            "sha": r["meta"].get("xla_sha"),
            "run": r["meta"].get("workflow_run"),
        } for r in runs],
        "modes": [{"key": m, "label": labels[m]} for m in modes],
        "groups": [{"key": g, "label": group_labels.get(g, g)} for g in groups],
        "series": series,
        # The modules the aggregates are made of, and what moved between runs.
        "modules": modules,
        "module_values": module_values,
        "movers": move_rows,
        "movers_show": MOVERS_SHOW,
        "hip": hip_versions(per_latency, modes),
        "latency": {
            "gpus": gpu_counts,
            "metrics": [{"key": k, "label": l, "unit": u} for k, l, u in LATENCY_METRICS],
            "series": latency,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))

    first, last = runs[0]["name"], runs[-1]["name"]
    print(f"{len(runs)} runs ({first} .. {last})")
    for mode in modes:
        s = series[mode]
        idx = [v for v in s["index"] if v is not None]
        partial = sum(1 for c in s["complete"] if not c)
        moved = sum(r["n"] for r in move_rows[mode] if r)
        where = (f"index {min(idx):.1f}-{max(idx):.1f} (now {idx[-1]:.1f})"
                 if idx else "no index")
        print(f"  {mode:21s} {s['core']:3d} modules, {where}")
        print(f"  {'':21s} noise p{NOISE_PCT} "
              f"{(s['noise_pct'] or 0) * 100:.1f}%, {moved} module changes above it"
              + (f", {partial} partial run(s)" if partial else ""))
    if latency:
        print(f"  latency: {len(latency)} series over {len(gpu_counts)} GPU counts "
              f"{gpu_counts}")
    else:
        print("  latency: no final rows found")
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
