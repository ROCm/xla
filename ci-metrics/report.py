#!/usr/bin/env python3
"""Build the CI metrics Markdown report.

Full pipeline (all steps resumable; ~1.5h wall, bounded by the 5000/h API budget):

  D=DATA_DIR; S=2026-09-22T15:00; U=2026-09-29T15:00         # UTC, Tuesday to Tuesday
  python3 collect.py        --since $S --until $U --out $D   # runs + jobs + steps
  python3 fetch_attempts.py --data $D                        # earlier attempts of re-runs
  python3 sample_logs.py    --data $D                        # test counts, critical path
  python3 report.py         --data $D --since $S --until $U --md OUT.md [--baseline BASE.json]

The report is GitHub-flavoured Markdown, rendered directly: one section per repo
from REPOS, then the shared infrastructure sections.

--baseline adds a "Changes since" section from an earlier report's figures:
  {"label": "...", "jax_rocm_tests": N, "xla_rocm_tests": N, "do_nodes": N, "do_gpus": N,
   "queue": {"JAX single-GPU pool": [old_label, p50, p90], ...}, "note": "markdown"}
Every key except label is optional; queue names must match POOL_NOW.
"""
import argparse
import collections
import datetime as dt
import json
import os
import re
import statistics
import string

from metrics_lib import (ALL_LANES, CDT, LANES, NIGHTLY_WORKFLOWS, PHASES, REPO_BY_SHORT, REPOS, carried_over,
                         cov_key, is_do, job_phases, lane_key, load, nightly_kind, test_step_seconds, ts)

SIGNAL = ("success", "failure", "timed_out")


# ---- small helpers ---------------------------------------------------------

def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt(s):
    if s is None:
        return "n/a"
    s = int(round(s))
    neg = s < 0
    s = abs(s)
    out = f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"
    return "-" + out if neg else out


def esc(s):
    """Escape the few characters that would be read as Markdown markup."""
    return re.sub(r"([\\`*_\[\]|])", r"\\\1", str(s))


def hour(t):
    return t.astimezone(CDT).hour


# ---- data model ------------------------------------------------------------

def build(data_dir):
    recs = list(load(data_dir))
    ap = os.path.join(data_dir, "attempts.jsonl")
    earlier = [json.loads(l) for l in open(ap)] if os.path.exists(ap) else []
    do_jobs = []
    for rec in recs:
        for j in rec["jobs"]:
            if is_do(j) and j["started_at"] and j["created_at"] and j["conclusion"] != "skipped" \
                    and not carried_over(j):
                do_jobs.append((rec, j))

    # Peak / nominal hours from DO arrivals.
    arrivals = collections.Counter(hour(ts(j["created_at"])) for _, j in do_jobs)
    by_vol = sorted(range(24), key=lambda h: (-arrivals[h], h))
    peak, nominal = sorted(by_vol[:8]), sorted(by_vol[-8:])

    def band(t):
        h = hour(t)
        return "peak" if h in peak else "nominal" if h in nominal else "mid"

    # Presubmit commits.
    commits = {k: [] for k in ALL_LANES}
    unsignalled = collections.Counter()
    approval = {k: [] for k in ALL_LANES}
    lane_jobs_all = {k: [] for k in ALL_LANES}
    for key, (repo, wf, pred, _) in ALL_LANES.items():
        groups = collections.defaultdict(list)
        for rec in recs + earlier:
            if rec["repo"] == repo and rec["workflow"] == wf and rec["run"]["event"] == "pull_request":
                groups[rec["run"]["head_sha"]].append(rec)
        seen_jobs = set()
        for sha, runs in groups.items():
            # Earliest attempt first: when a re-run only carried this lane's jobs
            # over, the attempt that really ran them wins the tie below.
            runs.sort(key=lambda r: r["run"]["run_attempt"])
            # Fork PRs sit in action_required until a maintainer approves; the
            # CI clock starts at approval and the wait is reported separately.
            gated = [r for r in runs if r["run"]["conclusion"] == "action_required" and not r["jobs"]]
            runs = [r for r in runs if r not in gated]
            if not runs:
                continue
            approved = {r["run"]["id"] for r in gated}

            def real_start(r):
                if r["run"]["run_attempt"] > 1 and r["run"]["id"] in approved:
                    return ts(r["run"]["run_started_at"])
                return ts(r["run"]["created_at"])

            t0 = min(real_start(r) for r in runs)
            if gated:
                approval[key].append((t0 - min(ts(r["run"]["created_at"]) for r in gated)).total_seconds())
            best = None
            for r in runs:
                lj = [j for j in r["jobs"] if pred(j) and j["conclusion"] != "skipped"]
                for j in lj:
                    if j["id"] not in seen_jobs and not carried_over(j):
                        seen_jobs.add(j["id"])
                        lane_jobs_all[key].append((r, j))
                if not lj or any(j["conclusion"] not in SIGNAL or not j["completed_at"] for j in lj):
                    continue
                crit = max(lj, key=lambda j: ts(j["completed_at"]))
                if best is None or ts(crit["completed_at"]) < ts(best[1]["completed_at"]):
                    best = (r, crit, lj)
            if best is None:
                if any(r["jobs"] for r in runs):
                    unsignalled[key] += 1
                continue
            r, crit, lj = best
            njobs = len(lj)
            start = ts(r["run"]["created_at"])
            if r["run"]["run_attempt"] > 1 and crit["run_attempt"] == r["run"]["run_attempt"]:
                start = ts(r["run"]["run_started_at"])
            ph = {"superseded": (start - t0).total_seconds(),
                  "upstream": (ts(crit["created_at"]) - start).total_seconds(),
                  "queue": (ts(crit["started_at"]) - ts(crit["created_at"])).total_seconds()}
            ph.update(job_phases(crit))
            total = (ts(crit["completed_at"]) - t0).total_seconds()
            assert abs(sum(ph.values()) - total) < 1e-6, (sha, ph, total)
            commits[key].append({"sha": sha, "t0": t0, "band": band(t0), "total": total, "phases": ph,
                                 "njobs": njobs, "attempt": r["run"]["run_attempt"], "prs": r["run"]["prs"],
                                 "passed": all(j["conclusion"] == "success" for j in lj),
                                 "jobs": [{"type": job_type(key, j["name"]),
                                           "exec": (ts(j["completed_at"]) - ts(j["started_at"])).total_seconds(),
                                           "test": test_step_seconds(j),
                                           "queue": (ts(j["started_at"]) - ts(j["created_at"])).total_seconds()}
                                          for j in lj],
                                 "rerun": r["run"]["run_attempt"] > 1 and r["run"]["id"] not in approved,
                                 "extra_runs": len(runs) - 1, "crit": crit["name"]})

    # Nightly / continuous ROCm legs, one signal per run per kind.
    nightly = {"bazel": [], "pytest": []}
    nightly_test_steps = {"bazel": [], "pytest": []}
    for rec in recs:
        if rec["workflow"] not in NIGHTLY_WORKFLOWS:
            continue
        kinds = collections.defaultdict(list)
        for j in rec["jobs"]:
            k = nightly_kind(rec, j)
            if k and j["conclusion"] != "skipped":
                kinds[k].append(j)
        for k, lj in kinds.items():
            done = [j for j in lj if j["conclusion"] in SIGNAL and j["completed_at"]]
            nightly_test_steps[k] += [test_step_seconds(j) for j in done]
            if done and len(done) == len(lj):
                last = max(ts(j["completed_at"]) for j in done)
                nightly[k].append((last - ts(rec["run"]["created_at"])).total_seconds())

    nv_arrivals = collections.Counter(hour(ts(j["created_at"])) for key in ("jax_nv", "xla_nv")
                                      for _, j in lane_jobs_all[key] if j["created_at"])

    return {"data_dir": data_dir, "recs": recs, "do_jobs": do_jobs, "arrivals": arrivals, "peak": peak, "nominal": nominal,
            "band": band, "commits": commits, "unsignalled": unsignalled, "approval": approval,
            "lane_jobs": lane_jobs_all, "nv_arrivals": nv_arrivals,
            "nightly": nightly, "nightly_test_steps": nightly_test_steps}


def load_logs(data_dir):
    p = os.path.join(data_dir, "logs.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ---- Markdown ---------------------------------------------------------------
# The report is plain GitHub-flavoured Markdown. There is no stylesheet, so
# anything the HTML version said with colour is said here with markup: WIN marks
# a like-for-like ROCm win, "dominant" bolds the largest phase of a row, and
# "pending" italicises a figure that was not measured.

WIN = "\u25b2"          # the green-cell marker
SUB = "\u21b3"          # job-family rows under a lane
BULLET = "\u00b7"
DASH = "\u2014"


def cell(c, numeric):
    """One table cell: either plain text, or (text, markers) where markers name
    the meaning the HTML carried in a CSS class."""
    txt, marks = c if isinstance(c, tuple) else (c, "")
    txt = str(txt).strip().replace("|", "\\|").replace("\n", " ") or " "
    if "dominant" in marks or "win" in marks:
        txt = f"**{txt}**"
    if "win" in marks:
        txt += " " + WIN
    if "pending" in marks:
        txt = f"*{txt}*"
    return txt


def table(head, rows, num_from=1):
    """A GFM pipe table. `rows` holds plain strings or (text, markers) tuples.

    `num_from` is the first right-aligned (numeric) column, which GFM expresses
    in the delimiter row.
    """
    n = len(head)
    align = ["---" if i < num_from else "---:" for i in range(n)]
    out = ["| " + " | ".join(cell(h, False) for h in head) + " |",
           "| " + " | ".join(align) + " |"]
    for r in rows:
        cells = [cell(c, i >= num_from) for i, c in enumerate(r)]
        cells += [" "] * (n - len(cells))
        out.append("| " + " | ".join(cells[:n]) + " |")
    return "\n".join(out)


def h(level, text):
    return f"{'#' * level} {text}"


def note(text):
    """An aside; a blockquote keeps it visually separate from the tables."""
    return "> " + text.replace("\n", " ")


def win_pair(rocm, nv, comparable):
    a, b = fmt(rocm), fmt(nv)
    if comparable and rocm is not None and nv is not None and rocm < nv:
        return (a, "win"), b
    return a, b


# ---- templates --------------------------------------------------------------
# The document's shape lives in templates/*.md, so the structure of the report
# can be read in one place. Python computes the values; the template decides
# where they go. Tables arrive pre-rendered because string.Template has no
# loops - the slot name says what the table is, which is the part worth seeing.

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def fill(name, **slots):
    """Substitute into templates/<name>.md.

    Missing slots are an error (a typo should not silently blank a section);
    empty ones are allowed, and the blank lines they leave behind are collapsed.
    """
    with open(os.path.join(TEMPLATES, name + ".md")) as f:
        out = string.Template(f.read()).substitute(**slots)
    return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"


def repo_section(m, logs, cov, R, span):
    """Every table scoped to one repo: coverage, metrics, phases, ROCm vs NV, bands, tail.

    Driven entirely by the repo's REPOS entry, so a new repo is a config entry.
    Sections about shared hardware (DO pools, NV pools, RBE, scale-sets) stay in
    render(), since they describe a fleet both repos sit on.
    """
    commits = m["commits"]
    rk, nk = lane_key(R, "rocm"), lane_key(R, "nv")
    slot = {"repo": esc(R["repo"])}

    # --- coverage -----------------------------------------------------------
    rows = []
    for suffix, keys in cov_rows(R):
        for lane, lane_lbl in [("rocm", "ROCm"), ("nv", "NV/CUDA")]:
            is_rocm = lane == "rocm"
            c = cov.get(keys[lane], {})
            spt = c.get("spt")
            other = cov.get(keys["nv" if is_rocm else "rocm"], {}).get("spt")
            spt_cell = ((f"{spt:.2f}s", "n win" if is_rocm and spt and other and spt < other else "n")
                        if spt else ("n/a", "n pending"))
            rows.append([f"{R['label']}{suffix}", (lane_lbl, "lane rocm" if is_rocm else "lane"),
                         c.get("suite_txt", "n/a"),
                         fmt(c.get("p50")) if c.get("p50") else ("no full-suite run sampled", "n pending"),
                         spt_cell, str(c.get("n_full", 0))])
    slot["coverage_table"] = table(["Job family", "Lane", "Tests in suite", "Test step p50 (full-suite runs)",
                                   "Seconds per test", "Full-suite runs sampled"], rows, num_from=2)
    slot["coverage_note"] = coverage_note(cov, R)

    # --- headline metrics ---------------------------------------------------

    def q_do(b):
        return [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for rec, j in m["do_jobs"]
                if rec["repo"] == R["repo"] and m["band"](ts(j["created_at"])) == b]

    def rbe_steps(key, b):
        xs = []
        for _, j in m["lane_jobs"][key]:
            if j["conclusion"] in SIGNAL and j["completed_at"] and m["band"](ts(j["created_at"])) == b:
                t = test_step_seconds(j)
                if t > 0:
                    xs.append(t)
        return xs

    def cell(xs):
        return f"{fmt(pct(xs, 90))}  (n={len(xs)})" if xs else "n/a"

    rows = []
    bands = [("peak", "peak times"), ("nominal", "nominal times")]
    for b, lbl in bands:
        rows.append([f"Queue wait (P90, {span}), {lbl}", cell(q_do(b)),
                     ("n/a: EngFlow scheduler queue is not exposed per action; see RBE detail", "n pending")])
    for b, lbl in bands:
        rows.append([f"Time to signal presubmit, {lbl}",
                     cell([c["total"] for c in commits[rk] if c["band"] == b]), cell(rbe_steps(rk, b))])
    if R["nightly"]:
        rows.append(["Time to signal nightly, bazel", cell(m["nightly"]["bazel"]),
                     cell([x for x in m["nightly_test_steps"]["bazel"] if x > 0])])
        rows.append(["Time to signal nightly, pytest", cell(m["nightly"]["pytest"]),
                     ("n/a, pytest runs local", "n pending")])
    for lbl, do, rbe in R.get("extra_metrics", ()):
        rows.append([lbl, (do, "pending"), rbe])
    rows = [[(str(i), "n")] + r for i, r in enumerate(rows, 1)]
    slot["metrics_table"] = table(["#", "Metric", "DO", "RBE"], rows, num_from=2)
    slot["nightly_note"] = ""
    if R["nightly"]:
        wfs = ", ".join(f"`{esc(w)}`" for w in R["nightly"])
        slot["nightly_note"] = (f" Nightly rows cover {wfs}, one signal per run: run created to its last "
                                f"ROCm job finishing.")

    # --- phase breakdown ----------------------------------------------------
    slot["families_note"] = ""
    if R["families"]:
        slot["families_note"] = (
            f"{esc(R['repo'])} runs more than one job family in a single workflow, so its lanes get "
            "one row per family. Both rows cover the same commits, so they share one **n**; **Total** is that "
            "family's own signal and **Lane signal** is the combined one the PR actually waits for - the slower "
            "family on each commit, which is why it repeats down a lane. The count beside the family name is how "
            "often it was the family that finished last.")
    phase_head = ["Line item", "Job family"] + [lbl for _, lbl in PHASES] + ["Total", "Lane signal", "n"]

    def phase_cells(chosen):
        """The per-phase means plus the row's own total, dominant phase marked."""
        means = {k: statistics.fmean(c["phases"][k] for c in chosen) for k, _ in PHASES}
        dom = max(means, key=means.get)
        return ([(fmt(means[k]), "n dominant" if k == dom else "n") for k, _ in PHASES]
                + [fmt(statistics.fmean(c["total"] for c in chosen))])

    def phase_rows(sel):
        rows = []
        for lane, lane_lbl in [("rocm", "ROCm"), ("nv", "NV")]:
            key = lane_key(R, lane)
            # One row per lane over all hours: splitting peak from nominal here
            # leaves a handful of commits per row once the tail is taken.
            chosen = sel(commits[key])
            if not chosen:
                continue
            scope = "DO" if lane == "rocm" else "NV"
            head = f"{lane_lbl} Time to signal presubmit ({scope})"
            if not R["families"]:
                # One family, so the row's own total already is the lane signal.
                rows.append([head, "all jobs"] + phase_cells(chosen)
                            + [("—", "n pending"), str(len(chosen))])
                continue
            lane_total = fmt(statistics.fmean(c["total"] for c in chosen))
            subs = [(lbl, f"{key}:{fam}") for lbl, fam in R["families"]]
            by = {sk: {c["sha"]: c for c in commits[sk]} for _, sk in subs}
            shas = [c["sha"] for c in chosen]
            # Which family finished last on each commit, i.e. set the signal.
            last = collections.Counter()
            for s in shas:
                totals = {sk: by[sk][s]["total"] for _, sk in subs if s in by[sk]}
                if totals:
                    last[max(totals, key=totals.get)] += 1
            for fam_lbl, sub_key in subs:
                cs = [by[sub_key][s] for s in shas if s in by[sub_key]]
                if not cs:
                    continue
                label = f"{fam_lbl} (last on {last[sub_key]}/{len(shas)})"
                rows.append([head, label] + phase_cells(cs) + [lane_total, str(len(cs))])
        return rows

    def middle(cs):
        if not cs:
            return []
        lo, hi = pct([c["total"] for c in cs], 40), pct([c["total"] for c in cs], 60)
        return [c for c in cs if lo <= c["total"] <= hi]

    def slowest(cs):
        if not cs:
            return []
        p90 = pct([c["total"] for c in cs], 90)
        return [c for c in cs if c["total"] >= p90]

    slot["phase_typical_table"] = table(phase_head, phase_rows(middle), num_from=2)
    slot["phase_slowest_table"] = table(phase_head, phase_rows(slowest), num_from=2)
    slot["phase_notes"] = phase_notes(m, R)

    # --- ROCm vs NV ---------------------------------------------------------
    ck = (cov_key(R, "rocm"), cov_key(R, "nv"))
    rs, ns = lane_stats(m, rk), lane_stats(m, nk)
    rc, nc = cov.get(ck[0], {}).get("suite"), cov.get(ck[1], {}).get("suite")
    comparable = bool(rc and nc and rc >= 0.9 * nc)
    rows = []
    for lbl, k in [("Time to signal, p50", "tts50"), ("Time to signal, p90", "tts90"),
                   ("Per-job execution, p50", "exe50"), ("Per-job execution, p90", "exe90"),
                   ("Queue wait, p50", "q50")]:
        a, b = win_pair(rs[k], ns[k], comparable and k != "q50")
        rows.append([(lbl, "lane"), a, b])
    for r in rows:
        r.append(f"ROCm {rs['jobs']} · NV {ns['jobs']}")
    slot["rocm_vs_nv_table"] = table(["Metric", "ROCm (DO)", "NV (Google)", "Jobs per commit"], rows)
    slot["rocm_commits"], slot["nv_commits"] = str(rs["n"]), str(ns["n"])
    slot["compare_note"] = compare_note(R["repo"], rc, nc, cov, ck, comparable)

    slot.update(band_section(m, cov, R))
    slot.update(tail_section(m, R))
    return fill("repo_section", **slot)


def render(m, logs, since, until, base=None):
    peak, nominal = m["peak"], m["nominal"]
    days = (until - since).total_seconds() / 86400
    # The window is whatever collect.py fetched; a nightly run is one day, not a week.
    span = f"{days:g} day{'' if days == 1 else 's'}"
    slot = {
        "window": f"{since:%Y-%m-%d %H:%M} to {until:%Y-%m-%d %H:%M} UTC",
        "span": (f"Tuesday to Tuesday, {span}"
                 if days == 7 and since.weekday() == 1 and until.weekday() == 1 else span),
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "win": WIN,
        "peak_hours": "`" + ", ".join(f"{h:02d}" for h in peak) + "`",
        "nominal_hours": "`" + ", ".join(f"{h:02d}" for h in nominal) + "`",
    }

    cov = coverage(logs, m)
    slot["repo_sections"] = "\n\n".join(repo_section(m, logs, cov, R, span) for R in REPOS)

    # --- infrastructure -----------------------------------------------------
    def was(k):
        return f" ({esc(base['label'])}: {base[k]})" if base and k in base else ""

    labels = collections.Counter()
    for rec, j in m["do_jobs"]:
        labels[next(l for l in j["labels"] if l.startswith("amd-do-"))] += 1
    slot["infra_table"] = table(["Item", "Value"], [
        ["DO node count (DPX)", (f"not re-measured: not derivable from job metadata{was('do_nodes')}", "pending")],
        ["DO GPUs (executors)", (f"not re-measured: not derivable from job metadata{was('do_gpus')}", "pending")],
        ["DO runner labels seen in the window", str(len(labels))],
        ["Local node count", ("pending: not derivable from CI metadata (owner: user)", "pending")],
        ["Local GPUs", ("pending: not derivable from CI metadata (owner: user)", "pending")]])

    # --- peak vs nominal ----------------------------------------------------
    arr, nva = m["arrivals"], m["nv_arrivals"]
    slot["arrivals_table"] = table(
        ["Hour CDT (UTC-5)", "DO jobs", "NV jobs", "", "Hour CDT (UTC-5)", "DO jobs", "NV jobs"],
        [[f"{h:02d}", str(arr[h]), str(nva[h]), "", f"{h + 12:02d}", str(arr[h + 12]), str(nva[h + 12])]
         for h in range(12)], num_from=0)
    lo, hi = min(arr.values()), max(arr.values())
    nv_peak = sorted(sorted(range(24), key=lambda h: (-nva[h], h))[:8])
    slot["swing"] = f"{hi / lo:.1f}"
    slot["arrivals_lo"], slot["arrivals_hi"] = str(lo), str(hi)
    slot["nv_peak_hours"] = "`" + ", ".join(f"{h:02d}" for h in nv_peak) + "`"
    slot["shared_peak_hours"] = str(len(set(nv_peak) & set(peak)))

    pools = [(R["label"], R["pool"]) for R in REPOS]
    comp = collections.defaultdict(list)
    for rec, j in m["do_jobs"]:
        if j["conclusion"] not in SIGNAL or not j["completed_at"]:
            continue
        pool = next((p for p, lab in pools if lab in j["labels"]), None)
        if not pool:
            continue
        b = m["band"](ts(j["created_at"]))
        ph = job_phases(j)
        comp[("queue", pool, b)].append((ts(j["started_at"]) - ts(j["created_at"])).total_seconds())
        comp[("setup", pool, b)].append(ph["setup"] + ph["docker"] + ph["env"])
        comp[("test", pool, b)].append(ph["test"])
    rows = []
    for k, lbl in [("queue", "Queue"), ("setup", "Set-up (runner, docker, checkout, env)"),
                   ("test", "Test execution")]:
        for pool, lab in pools:
            rows.append([f"{lbl} ({pool} pool)"]
                        + [fmt(pct(comp[(k, pool, b)], 50)) for b in ("peak", "nominal")]
                        + [fmt(pct(comp[(k, pool, b)], 90)) for b in ("peak", "nominal")])
    slot["pool_component_table"] = table(
        ["Per-job component (DO, single-GPU pools)", "Peak p50", "Nominal p50", "Peak p90", "Nominal p90"], rows)

    def flat(a, b):
        return abs(a - b) <= max(0.2 * max(a, b), 10)

    verdicts = []
    for pool, _ in pools:
        qp, qn = pct(comp[("queue", pool, "peak")], 50), pct(comp[("queue", pool, "nominal")], 50)
        tp, tn = pct(comp[("queue", pool, "peak")], 90), pct(comp[("queue", pool, "nominal")], 90)
        if qp is None or qn is None:
            continue
        if flat(qp, qn) and flat(tp, tn):
            v = "flat at p50 and p90, so the wait is provisioning latency, not contention"
        elif flat(qp, qn):
            v = (f"flat at p50 but p90 goes {fmt(tn)} to {fmt(tp)}, so the typical wait is provisioning "
                 f"latency and the tail is contention")
        else:
            v = "it moves with demand, so part of the wait is contention"
        verdicts.append(f"**{pool} pool**: queue p50 {fmt(qp)} at peak vs {fmt(qn)} at nominal; {v}")
    slot["pool_verdicts"] = "; ".join(verdicts)

    # --- scale sets ---------------------------------------------------------
    rows = []
    for lab, n in labels.most_common():
        xs = [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for rec, j in m["do_jobs"]
              if lab in j["labels"]]
        g = re.search(r"[.-](\d+)$", lab)
        gpus = "0 (CPU)" if "cpu" in lab else (g.group(1) if g else "?")
        rows.append([f"`{esc(lab)}`", gpus, str(n)] + [fmt(pct(xs, p)) for p in (50, 90, 99)] + [fmt(max(xs))])
    slot["scale_set_table"] = table(["Scale-set", "GPUs", "Jobs", "Queue p50", "p90", "p99", "Max"], rows)

    rows = []
    for lab, _ in labels.most_common():
        xs = [job_phases(j)["gaps"] for rec, j in m["do_jobs"]
              if lab in j["labels"] and j["conclusion"] in SIGNAL and j["completed_at"]]
        if not xs:
            continue
        rows.append([f"`{esc(lab)}`", str(len(xs))] + [fmt(pct(xs, p)) for p in (50, 90, 99)]
                    + [str(sum(x > 300 for x in xs)), fmt(max(xs))])
    slot["teardown_table"] = table(
        ["Scale-set", "Jobs", "Teardown p50", "p90", "p99", "Jobs over 5 min", "Worst"], rows)

    # --- NV runner pools ----------------------------------------------------
    nv_pools = collections.defaultdict(list)
    for R in REPOS:
        key = lane_key(R, "nv")
        for _, j in m["lane_jobs"][key]:
            if j["started_at"] and j["created_at"]:
                nv_pools[(R["repo"], ", ".join(j["labels"]), job_type(key, j["name"]))].append(j)
    rows = []
    for (repo, lab, ty), js in sorted(nv_pools.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        q = [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for j in js]
        days_seen = sorted(ts(j["created_at"]).astimezone(CDT) for j in js)
        rows.append([repo, f"`{esc(lab)}`", ty, str(len(js))] + [fmt(pct(q, p)) for p in (50, 90, 99)]
                    + [str(sum(x > 600 for x in q)), f"{days_seen[0]:%m-%d} to {days_seen[-1]:%m-%d}"])
    slot["nv_pool_table"] = table(["Repo", "Runner label", "Job type", "Jobs", "Queue p50", "p90", "p99",
                                   "Over 10 min", "Seen (CDT)"], rows, num_from=3)

    # --- RBE detail ---------------------------------------------------------
    rows = []
    for lbl, cat, inv_idx in [(f"{R['label']} {sfx}", cat, i) for R in REPOS for sfx, cat, i in R["rbe"]]:
        el = [l["elapsed"][inv_idx] for l in logs if l["cat"] == cat and l["elapsed"]]
        w, c = [e[0] for e in el], [e[1] for e in el]
        ratio = (pct(w, 50) / pct(c, 50)) if el and pct(c, 50) else None
        rows.append([lbl, str(len(el)), fmt(pct(w, 50)), fmt(pct(w, 90)), fmt(pct(c, 50)), fmt(pct(c, 90)),
                     f"{ratio:.0f}x" if ratio else "n/a"])
    slot["rbe_table"] = table(["Set", "Invocations sampled", "Wall p50", "Wall p90", "Critical path p50",
                               "Critical path p90", "Wall:cpath at p50"], rows)

    # --- the ask ------------------------------------------------------------
    slot.update(ask_section(cov, base))
    slot["changes_section"] = changes_section(m, cov, base) if base else ""
    return fill("report", **slot)


def job_type(lane, name):
    """Classify a job name, using the owning repo's rules. `lane` is a lane key."""
    return REPO_BY_SHORT[lane.split("_", 1)[0]]["job_type"](name)


def fam_keys(R, fam):
    """The per-lane coverage/log-sample keys for one job family, or for the lane."""
    return {lane: (f"{lane_key(R, lane)}:{fam}" if fam else cov_key(R, lane)) for lane in LANES}


def cov_rows(R):
    """(display suffix, {lane: key}) once per job family, or once for the whole lane."""
    if not R["families"]:
        return [("", fam_keys(R, None))]
    return [(f" ({lbl})", fam_keys(R, fam)) for lbl, fam in R["families"]]


def band_section(m, cov, R):
    """ROCm and NV over the same clock hours: peak/nominal p50 and p90, then p90 by time of day."""
    def comparable(rc, nc):
        a, b = cov.get(rc, {}).get("suite"), cov.get(nc, {}).get("suite")
        return bool(a and b and a >= 0.9 * b)

    def lane(key, keep, types):
        jobs = [j for _, j in m["lane_jobs"][key]
                if j["conclusion"] in SIGNAL and j["completed_at"] and j["started_at"] and keep(ts(j["created_at"]))]
        tj = [j for j in jobs if job_type(key, j["name"]) in types and test_step_seconds(j) > 0]
        return {"tts": [c["total"] for c in m["commits"][key] if keep(c["t0"])],
                "queue": [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for j in jobs],
                "exec": [(ts(j["completed_at"]) - ts(j["started_at"])).total_seconds() for j in tj],
                "test": [test_step_seconds(j) for j in tj]}

    bands = [(b, lambda t, b=b: m["band"](t) == b) for b in ("peak", "nominal")]

    def metric_rows(label, metric, rs, ns, comp):
        out = []
        for lname, s, o, is_rocm in [("ROCm", rs, ns, 1), ("NV", ns, rs, 0)]:
            cells = []
            for b, _ in bands:
                for p in (50, 90):
                    v, w = pct(s[b][metric], p), pct(o[b][metric], p)
                    win = is_rocm and comp and v is not None and w is not None and v < w
                    cells.append((fmt(v), "n win" if win else "n"))
            row = [label]
            out.append(row + [(lname, "lane rocm" if is_rocm else "lane")] + cells
                       + [f"{len(s['peak'][metric])} / {len(s['nominal'][metric])}"])
        return out

    rk, nk = lane_key(R, "rocm"), lane_key(R, "nv")
    ck = (cov_key(R, "rocm"), cov_key(R, "nv"))
    pairs = R["pairs"]
    r_types, n_types = {rt for _, rt, _, _ in pairs}, {nt for _, _, nt, _ in pairs}
    rs = {b: lane(rk, keep, r_types) for b, keep in bands}
    ns = {b: lane(nk, keep, n_types) for b, keep in bands}
    rows = metric_rows("Time to signal (per commit)", "tts", rs, ns, comparable(*ck))
    rows += metric_rows("Queue wait (every lane job)", "queue", rs, ns, False)
    for lbl, rt, nt, fam in pairs:
        cats = tuple(fam_keys(R, fam)[lane] for lane in LANES)
        rts = {b: lane(rk, keep, {rt}) for b, keep in bands}
        nts = {b: lane(nk, keep, {nt}) for b, keep in bands}
        rows += metric_rows(f"Job execution, {lbl}", "exec", rts, nts, comparable(*cats))
        rows += metric_rows(f"Test step, {lbl}", "test", rts, nts, comparable(*cats))
    out = {"band_table": table(["Metric", "Lane", "Peak p50", "Peak p90", "Nominal p50", "Nominal p90",
                                "n (peak / nominal)"], rows, num_from=2)}

    rows = []
    for blk in range(8):
        def keep(t, blk=blk):
            return hour(t) // 3 == blk
        r, n = lane(rk, keep, r_types), lane(nk, keep, n_types)
        hrs = range(3 * blk, 3 * blk + 3)
        tag = ("peak" if sum(h in m["peak"] for h in hrs) >= 2 else
               "nominal" if sum(h in m["nominal"] for h in hrs) >= 2 else "")
        rows.append([f"{3 * blk:02d}:00-{3 * blk + 3:02d}:00" + (f" ({tag})" if tag else ""),
                     f"{len(r['tts'])} / {len(n['tts'])}"]
                    + [f"{fmt(pct(r[k], 90))} / {fmt(pct(n[k], 90))}" for k in ("tts", "queue", "exec", "test")])
    out["band_by_hour_table"] = table(["Hours", "Commits ROCm / NV", "Time to signal ROCm / NV", "Queue ROCm / NV",
                                       "Test job execution ROCm / NV", "Test step ROCm / NV"], rows)
    return out


def lookup_pr(m, repo, sha):
    """Fork-PR runs carry no pull_requests; resolve via the commit, cached in the data dir."""
    path = os.path.join(m["data_dir"], "pr_lookup.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    k = f"{repo}@{sha}"
    if k not in cache:
        try:
            from collect import get
            prs = get(f"/repos/{repo}/commits/{sha}/pulls")
            cache[k] = prs[0]["number"] if prs else None
        except Exception:
            return None
        with open(path, "w") as f:
            json.dump(cache, f)
    return cache[k]


def tail_section(m, cfg):
    repo = cfg["repo"]
    rk, nk = lane_key(cfg, "rocm"), lane_key(cfg, "nv")
    ps = (50, 90, 95, 99)
    rows = []
    for lane_lbl, key, is_rocm in [("ROCm", rk, 1), ("NV", nk, 0)]:
        cs = m["commits"][key]
        tt = [c["total"] for c in cs]
        rows.append([(lane_lbl, "lane rocm" if is_rocm else "lane"), "*commit time to signal*",
                     str(len(tt))] + [fmt(pct(tt, p)) for p in ps])
        by = collections.defaultdict(list)
        for c in cs:
            for j in c["jobs"]:
                by[j["type"]].append(j["exec"])
        for t, xs in sorted(by.items()):
            rows.append([(lane_lbl, "lane rocm" if is_rocm else "lane"), t, str(len(xs))]
                        + [fmt(pct(xs, p)) for p in ps])
    out = {"tail_percentile_table": table(["Lane", "Job type", "n", "p50", "p90", "p95", "p99"], rows, num_from=2),
           "same_commit_section": ""}

    R = {c["sha"]: c for c in m["commits"][rk]}
    N = {c["sha"]: c for c in m["commits"][nk]}
    common = set(R) & set(N)
    both = sorted(s for s in common if R[s]["passed"] and N[s]["passed"])
    if not both:
        return out
    ratio = [R[s]["total"] / N[s]["total"] for s in both if N[s]["total"] > 0]
    faster = sum(1 for s in both if R[s]["total"] < N[s]["total"])
    sc = {"repo": esc(repo), "n_both": str(len(both)), "n_failed": str(len(common) - len(both))}
    sc["ratio_table"] = (table(["ROCm / NV time to signal, per PR", "p10", "p50", "p90", "p99",
                                "PRs where ROCm is faster"],
                   [["ratio"] + [f"{pct(ratio, p):.2f}x" for p in (10, 50, 90, 99)]
                    + [f"{faster} of {len(both)} ({100 * faster / len(both):.0f}%)"]]))

    def outlier_row(label, slow_side, shas):
        rt = [R[s]["total"] for s in shas]
        nt = [N[s]["total"] for s in shas]
        rq = [R[s]["phases"]["queue"] for s in shas]
        rx = [R[s]["phases"]["test"] for s in shas]
        rg = [R[s]["phases"]["gaps"] for s in shas]
        n90 = pct([N[s]["total"] for s in both], 90)
        r90 = pct([R[s]["total"] for s in both], 90)
        other_slow = sum(1 for s in shas if (N[s]["total"] >= n90 if slow_side == "rocm" else R[s]["total"] >= r90))
        return [label, str(len(shas)), fmt(pct(rt, 50)), fmt(pct(nt, 50)),
                f"{pct([a / b for a, b in zip(rt, nt) if b], 50):.2f}x",
                f"{other_slow} ({100 * other_slow / len(shas):.0f}%)",
                fmt(statistics.fmean(rq)), fmt(statistics.fmean(rx)), fmt(statistics.fmean(rg))]

    r90 = pct([R[s]["total"] for s in both], 90)
    n90 = pct([N[s]["total"] for s in both], 90)
    slow_r = [s for s in both if R[s]["total"] >= r90]
    slow_n = [s for s in both if N[s]["total"] >= n90]
    sc["outlier_table"] = table(["Outlier set", "PRs", "ROCm TTS p50", "NV TTS p50", "Median ROCm/NV",
                                 "Also in the other lane's slowest 10%", "ROCm queue (mean)",
                                 "ROCm test execution (mean)", "ROCm teardown (mean)"],
                                [outlier_row("ROCm's slowest 10%", "rocm", slow_r),
                                 outlier_row("NV's slowest 10%", "nv", slow_n)])

    rows = []
    for lbl, rt, nt, _fam in cfg["pairs"]:
        pr, pn = [], []
        for s in both:
            a = [j for j in R[s]["jobs"] if j["type"] == rt]
            b = [j for j in N[s]["jobs"] if j["type"] == nt]
            if a and b:
                pr.append(a[0]["test"])
                pn.append(b[0]["test"])
        if not pr:
            continue
        cut = pct(pr, 90)
        tail = [(x, y) for x, y in zip(pr, pn) if x >= cut]
        rows.append([lbl, str(len(pr))] + [f"{fmt(pct(pr, p))} / {fmt(pct(pn, p))}" for p in (50, 90, 99)]
                    + [f"{pct([x / y for x, y in zip(pr, pn) if y], 50):.2f}x",
                       f"{fmt(pct([x for x, _ in tail], 50))} / {fmt(pct([y for _, y in tail], 50))}"])
    sc["matched_job_table"] = table(["Matched job type (test step)", "Commits", "p50 ROCm / NV", "p90 ROCm / NV",
                                     "p99 ROCm / NV", "Median per-commit ratio",
                                     "ROCm's slowest 10%: ROCm / NV p50 on the same commits"], rows)

    worst = sorted(both, key=lambda s: -(R[s]["total"] / N[s]["total"] if N[s]["total"] else 0))[:10]
    rows = []
    for s in worst:
        c = R[s]
        pr = c["prs"][0] if c["prs"] else lookup_pr(m, repo, s)
        link = f"[#{pr}](https://github.com/{repo}/pull/{pr})" if pr else "(fork PR, number not in API)"
        top = sorted(((k, v) for k, v in c["phases"].items()), key=lambda kv: -kv[1])[:2]
        rows.append([f"`{s[:10]}` {link}", fmt(c["total"]), fmt(N[s]["total"]),
                     f"{c['total'] / N[s]['total']:.1f}x",
                     ", ".join(f"{dict(PHASES)[k].lower()} {fmt(v)}" for k, v in top)])
    sc["largest_gap_table"] = table(["Largest ROCm/NV gap (commit)", "ROCm TTS", "NV TTS", "Ratio",
                                     "Where ROCm's time went"], rows, num_from=1)
    out["same_commit_section"] = fill("same_commit", **sc)
    return out


def lane_stats(m, key):
    cs = m["commits"][key]
    tot = [c["total"] for c in cs]
    jobs = [(r, j) for r, j in m["lane_jobs"][key] if j["conclusion"] in SIGNAL and j["completed_at"]
            and j["started_at"]]
    exe = [(ts(j["completed_at"]) - ts(j["started_at"])).total_seconds() for _, j in jobs]
    q = [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for _, j in jobs]
    nj = statistics.median([c["njobs"] for c in cs]) if cs else 0
    return {"tts50": pct(tot, 50), "tts90": pct(tot, 90), "exe50": pct(exe, 50), "exe90": pct(exe, 90),
            "q50": pct(q, 50), "n": len(cs), "jobs": int(nj)}


def coverage(logs, m):
    """Suite size from the newest 48h of samples; cost per test from full-suite runs."""
    out = {}
    if not logs:
        return out
    end = max(l["completed_at"] for l in logs)
    cutoff = (ts(end) - dt.timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
    step_by_job = {}
    for rec in m["recs"]:
        for j in rec["jobs"]:
            step_by_job[j["id"]] = test_step_seconds(j)
    by = collections.defaultdict(list)
    for l in logs:
        by[l["cat"]].append(l)
    for cat, ls in by.items():
        recent = [l for l in ls if l["completed_at"] >= cutoff and l["executed"]]
        if not recent:
            continue
        # Per-invocation suite sizes, most common in the recent window.
        shapes = collections.Counter(tuple(t for _, t in l["executed"]) for l in recent)
        shape = shapes.most_common(1)[0][0]
        suite = sum(shape)
        full = [l for l in recent if tuple(t for _, t in l["executed"]) == shape
                and sum(e for e, _ in l["executed"]) >= 0.9 * suite]
        steps = [step_by_job[l["job_id"]] for l in full if step_by_job.get(l["job_id"])]
        p50 = pct(steps, 50)
        txt = str(suite) if len(shape) == 1 else f"{suite} ({' + '.join(map(str, shape))})"
        out[cat] = {"suite": suite, "shape": shape, "suite_txt": txt, "p50": p50,
                    "spt": p50 / suite if p50 else None, "n_full": len(full), "n_recent": len(recent)}
    return out


def coverage_note(cov, R):
    parts = []
    for suffix, keys in cov_rows(R):
        lbl, r, n = f"{R['label']}{suffix}", keys["rocm"], keys["nv"]
        a, b = cov.get(r, {}), cov.get(n, {})
        if a.get("suite") and b.get("suite"):
            s = f"**{lbl} runs {100 * a['suite'] / b['suite']:.0f}% of the CUDA suite** ({a['suite']} vs {b['suite']})"
            if a.get("spt") and b.get("spt"):
                r_ = a["spt"] / b["spt"]
                s += f" and is {r_:.2f}x {'slower' if r_ > 1 else 'faster'} per test" if r_ >= 1 else \
                    f" and is {1 / r_:.2f}x faster per test"
            parts.append(s)
    return f"> {'; '.join(parts)}." if parts else ""


def compare_note(title, rc, nc, cov, ck, comparable):
    if not (rc and nc):
        return ""
    share = 100 * rc / nc
    a, b = cov.get(ck[0], {}).get("spt"), cov.get(ck[1], {}).get("spt")
    per = ""
    if a and b:
        per = f"; per test ROCm is **{a / b:.2f}x slower**" if a > b else f"; per test ROCm is **{b / a:.2f}x faster**"
    if comparable:
        return (f"> **Comparable suites:** {rc} ROCm tests against NV's {nc} ({share:.0f}%), so "
                f"durations are highlighted where ROCm is faster{per}. Queue wait is never highlighted; NV queue by "
                f"runner pool is under NV runner pools.")
    return (f"> **Not highlighted:** {rc} ROCm tests against NV's {nc} ({share:.0f}%), so these "
            f"durations compare a smaller suite to a larger one{per}.")


def phase_notes(m, R):
    out = []
    for lane, other in [("ROCm", "NV"), ("NV", "ROCm")]:
        key = lane_key(R, lane.lower())
        cs = m["commits"][key]
        if not cs:
            continue
        sup = [c for c in cs if c["phases"]["superseded"] > 0]
        att = [c for c in cs if c["rerun"]]
        ap = m["approval"][key]
        out.append(f"**{lane}:** {len(cs)} commits signalled; {len(sup)} of them first had a run that was "
                   f"cancelled or re-triggered ({len(att)} signalled on a re-run attempt), and "
                   f"{m['unsignalled'][key]} commits never produced a {lane} signal because every run was cancelled, "
                   f"almost always superseded by a newer push. "
                   + (f"{len(ap)} fork-PR commits waited for a maintainer to approve CI (median {fmt(pct(ap, 50))}, "
                      f"max {fmt(max(ap))}); their clock starts at approval, since that wait is a human gate "
                      f"the {other} lane shares." if ap else ""))
    out.append("**Superseded runs + re-run delay** = time from the SHA's first trigger to the start of the run "
               "that finally signalled. **Waiting on earlier jobs** = `needs:` jobs ahead of the "
               "lane's last job in the same run. "
               "**Runner teardown + completion report** = job wall time minus the sum of "
               "its step durations; almost all of it sits after the `Complete job` step, before GitHub "
               "marks the job done, so the check stays pending while the ephemeral runner shuts down.")
    return "\n\n".join(f"> {s}" for s in out)


def ask_section(cov, base):
    """The ask table, plus one coverage sentence per repo. Prose lives in the template."""
    ask_table = table(["Pool", "What it does today", "What we need", "What it unblocks"], [
        [("DO", "lane rocm"), "Builds and launches every ROCm presubmit job",
         ("pending (owner: user)", "pending"),
         "Removes the provisioning wait paid on every job, and the multi-GPU starvation"],
        [("RBE", "lane rocm"), "Runs the bazel test actions dispatched from DO",
         ("pending (owner: user)", "pending"),
         "Shortens test execution, the largest phase of time to signal"],
        [("Remote build", "lane rocm"), "JAX: nothing. `rocm_rbe_dynamic` sets "
         "`--spawn_strategy=local`, so compile actions run on the GPU runner",
         ("pending (owner: user)", "pending"),
         "Stops GPU nodes being used as build machines, freeing the same hardware for tests without buying more"],
    ], num_from=9)
    lines = []
    for R in REPOS:
        parts = []
        for suffix, keys in cov_rows(R):
            a, b = cov.get(keys["rocm"], {}).get("suite"), cov.get(keys["nv"], {}).get("suite")
            if a and b:
                bk = f"{R['short']}_rocm_tests"
                up = (f", up from {base[bk]} in the {esc(base['label'])}"
                      if base and bk in base and not suffix else "")
                parts.append(f"the{suffix or ' presubmit'} runs {a} ROCm targets against NV's {b}{up}")
        if parts:
            lines.append(f"**{esc(R['repo'])}:** " + ", and ".join(parts) + ".")
    return {"ask_table": ask_table, "ask_coverage": "\n\n".join(lines)}


# Keyed by the names the --baseline JSON uses; keep them stable.
POOL_NOW = {f"{R['short'].upper()} single-GPU pool": R["pool"] for R in REPOS}


def changes_section(m, cov, base):
    """Compare against an earlier report's figures, read from the --baseline JSON."""
    rows = []
    for R in REPOS:
        bk = f"{R['short']}_rocm_tests"
        if bk in base:
            now = cov.get(cov_key(R, "rocm"), {}).get("suite")
            rows.append([f"{R['label']} ROCm presubmit tests", str(base[bk]), str(now or "n/a")])
    for name, (old_lab, p50, p90) in base.get("queue", {}).items():
        lab = POOL_NOW[name]
        xs = [(ts(jb["started_at"]) - ts(jb["created_at"])).total_seconds() for _, jb in m["do_jobs"]
              if lab in jb["labels"]]
        rows.append([f"{name} queue p50 / p90 (`{esc(old_lab)}` then, `{esc(lab)}` now)",
                     f"{p50} / {p90}", f"{fmt(pct(xs, 50))} / {fmt(pct(xs, 90))}"])
    out = [f"## Changes since the {esc(base['label'])}",
           table(["Item", esc(base["label"]), "This week"], rows)]
    if base.get("note"):
        out.append(note(base["note"]))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--md", required=True, help="path of the Markdown report to write")
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--baseline", help="JSON with an earlier report's figures; enables 'Changes since'")
    a = ap.parse_args()
    m = build(a.data)
    logs = load_logs(a.data)
    since = dt.datetime.fromisoformat(a.since)
    until = dt.datetime.fromisoformat(a.until)
    base = json.load(open(a.baseline)) if a.baseline else None
    with open(a.md, "w") as f:
        f.write(render(m, logs, since, until, base))
    print(f"wrote {a.md}")
    for k, cs in m["commits"].items():
        print(k, "commits", len(cs), "unsignalled", m["unsignalled"][k])


if __name__ == "__main__":
    main()
