#!/usr/bin/env python3
"""Summarize one rbe_hlo_eval.sh results dir, or compare several.

  hlo_eval_summarize.py <results_dir> [noisy_pct]      writes summary.tsv, noisy.txt
  hlo_eval_summarize.py <results_dir> <results_dir>...  run-to-run comparison

Samples are the "## Execution time ... repeat=I duration=Nns" lines of
<model>/<leaf>/attempts/<k>/test.log (repeat 0 is the warmup). Spread is the IQR
sigma, (q75 - q25) / 1.349 in % of the median. Per leaf the attempt with the
smallest worst-module spread is kept and copied to <model>/<leaf>/; a leaf whose
kept spread exceeds noisy_pct (default 2) is NOISY and listed in noisy.txt.
"""
import glob
import math
import os
import re
import shutil
import statistics as st
import sys

EXEC = re.compile(r"## Execution time, file=(\S+) repeat=(\d+) duration=(\d+)ns")
WORKER = re.compile(r"^worker: (\S+)")


def parse(log):
    """Returns (worker host, {module: [ms, ...]}) of one test.log."""
    host, samples = "?", {}
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            if m := WORKER.match(line):
                host = m.group(1)
            elif (m := EXEC.search(line)) and int(m.group(2)) > 0:
                mod = os.path.basename(m.group(1)).split(".before_optimizations")[0]
                samples.setdefault(mod, []).append(int(m.group(3)) / 1e6)
    return host, samples


def quantile(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def iqr_sigma(v):
    return 100 * (quantile(v, .75) - quantile(v, .25)) / 1.349 / st.median(v)


def print_table(rows):
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def summarize(run, noisy_pct):
    rows = [["model", "leaf", "module", "n", "median_ms", "min_ms", "max_ms",
             "iqr_sigma_pct", "attempt", "host", "status"]]
    noisy = []
    for leaf_dir in sorted(glob.glob(os.path.join(run, "*", "*", "attempts"))):
        leaf_dir = os.path.dirname(leaf_dir)
        model, leaf = leaf_dir.split(os.sep)[-2:]
        attempts = sorted(glob.glob(os.path.join(leaf_dir, "attempts", "*")),
                          key=lambda a: int(os.path.basename(a)))
        best = None  # (worst spread, attempt dir, host, samples)
        for a in attempts:
            host, samples = parse(os.path.join(a, "test.log"))
            if samples:
                worst = max(iqr_sigma(v) for v in samples.values())
                if best is None or worst < best[0]:
                    best = (worst, a, host, samples)
        if best is None:
            rows.append([model, leaf, "-", "0"] + ["-"] * 6 + ["NO_DATA"])
            continue
        worst, a, host, samples = best
        for name in os.listdir(a):  # Bazel outputs are read-only: replace, don't overwrite.
            if os.path.isfile(os.path.join(a, name)):
                dst = os.path.join(leaf_dir, name)
                if os.path.lexists(dst):
                    os.remove(dst)
                shutil.copyfile(os.path.join(a, name), dst)
        if worst > noisy_pct:
            noisy.append(f"//_hlo_eval_rbe/{model}:{leaf}")
        for mod, v in sorted(samples.items()):
            s = iqr_sigma(v)
            rows.append([model, leaf, mod, str(len(v)), f"{st.median(v):.4g}",
                         f"{min(v):.4g}", f"{max(v):.4g}", f"{s:.2f}",
                         f"{os.path.basename(a)}/{len(attempts)}", host,
                         "NOISY" if s > noisy_pct else "OK"])
    with open(os.path.join(run, "summary.tsv"), "w") as f:
        f.writelines("\t".join(r) + "\n" for r in rows)
    with open(os.path.join(run, "noisy.txt"), "w") as f:
        f.writelines(t + "\n" for t in noisy)
    print_table(rows)
    print(f"\nnoisy (IQR sigma > {noisy_pct}%): {' '.join(noisy) or 'none'}")


def compare(runs):
    data = []  # per run: {(model, leaf, module): (host, samples)}
    for run in runs:
        d = {}
        for log in glob.glob(os.path.join(run, "*", "*", "test.log")):
            model, leaf = log.split(os.sep)[-3:-1]
            host, samples = parse(log)
            for mod, v in samples.items():
                d[(model, leaf, mod)] = (host, v)
        data.append(d)
    rows = []  # (range %, row)
    for key in sorted(set.intersection(*(set(d) for d in data))):
        meds = [st.median(d[key][1]) for d in data]
        mean = st.mean(meds)
        rng = 100 * (max(meds) - min(meds)) / mean
        rows.append((rng, meds, ["/".join(key)] + [f"{m:.4g}" for m in meds] + [
            f"{rng:.2f}", f"{100 * st.stdev(meds) / mean:.2f}",
            f"{max(iqr_sigma(d[key][1]) for d in data):.2f}",
            ",".join(d[key][0].rsplit("-", 1)[-1] for d in data)]))
    if not rows:
        print("no module has results in every run")
        return
    rows.sort(key=lambda r: -r[0])
    print_table([["model/leaf/module"] + [f"run{i + 1}" for i in range(len(runs))] +
                 ["range%", "cv%", "worst_iqr_sigma%", "hosts"]] + [r[2] for r in rows])
    ranges = sorted(r[0] for r in rows)
    print(f"\n{len(rows)} modules; range%: median {st.median(ranges):.2f}, "
          f"p90 {quantile(ranges, .9):.2f}; > 2%: {sum(r > 2 for r in ranges)}")
    # Suite geomean ratio per pair of runs, also without the pair's 3 largest
    # per-module differences (when there are more than 3 modules).
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            logs = sorted((math.log(r[1][j] / r[1][i]) for r in rows), key=abs)
            trimmed = (f" (without 3 largest: {100 * math.expm1(st.mean(logs[:-3])):+.3f}%)"
                       if len(logs) > 3 else "")
            print(f"geomean run{j + 1}/run{i + 1}: {100 * math.expm1(st.mean(logs)):+.3f}%{trimmed}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 1 and os.path.isdir(args[1]):
        compare(args)
    else:
        summarize(args[0], float(args[1]) if len(args) > 1 else 2.0)
