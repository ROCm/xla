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

import jinja2

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
                                 # A fork PR's run carries no `prs`, so the run
                                 # id and branch are what name it without an
                                 # API call; `lookup_pr()` is the one that costs
                                 # a request.
                                 "run": r["run"]["id"], "branch": r["run"]["head_branch"],
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


# ---- rendering --------------------------------------------------------------
# This file computes values and nothing else. Every heading, sentence, table and
# piece of markup lives in templates/report.md.j2; report.py hands it one dict
# and never builds a string of Markdown. A cell that needs emphasis carries a
# marker naming the meaning the old HTML put in a CSS class - "win" a
# like-for-like ROCm win, "dominant" the largest phase of a row, "pending" a
# figure that was not measured - and the template decides what each looks like.

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def marked(text, mark):
    """A cell value plus its marker, or the bare value when it has none."""
    return {"text": text, "mark": mark} if mark else text


def win_pair(rocm, nv, comparable):
    """One metric's ROCm/NV pair, the ROCm side marked when it is the winner."""
    a, b = fmt(rocm), fmt(nv)
    return marked(a, "win" if comparable and None not in (rocm, nv) and rocm < nv else None), b


def _cell(v):
    """Make a value safe to sit in a pipe-table cell."""
    return str(v).strip().replace("|", "\\|").replace("\n", " ") or " "


def render(data):
    """templates/report.md.j2, filled from one dict.

    StrictUndefined so a name the template asks for and the data does not carry
    is an error, rather than a silently blank figure.
    """
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES),
                             trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
                             undefined=jinja2.StrictUndefined)
    env.filters["cell"] = _cell
    out = env.get_template("report.md.j2").render(**data)
    # A section that rendered to nothing leaves a hole behind; close it up.
    return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"


def repo_data(m, logs, cov, R, span):
    """Every figure scoped to one repo: coverage, metrics, phases, ROCm vs NV, bands, tail.

    Driven entirely by the repo's REPOS entry, so a new repo is a config entry.
    Figures about shared hardware (DO pools, NV pools, RBE, scale-sets) stay in
    report_data(), since they describe a fleet both repos sit on.
    """
    commits = m["commits"]
    rk, nk = lane_key(R, "rocm"), lane_key(R, "nv")
    d = {"repo": R["repo"], "label": R["label"], "has_families": bool(R["families"]),
         "nightly": list(R["nightly"])}

    # --- coverage -----------------------------------------------------------
    rows = []
    for suffix, keys in cov_rows(R):
        for lane, lane_lbl in [("rocm", "ROCm"), ("nv", "NV/CUDA")]:
            is_rocm = lane == "rocm"
            c = cov.get(keys[lane], {})
            spt = c.get("spt")
            other = cov.get(keys["nv" if is_rocm else "rocm"], {}).get("spt")
            rows.append({
                "family": f"{R['label']}{suffix}",
                "lane": lane_lbl,
                "suite": c.get("suite_txt", "n/a"),
                "p50": fmt(c["p50"]) if c.get("p50") else marked("no full-suite run sampled", "pending"),
                "per_test": (marked(f"{spt:.2f}s", "win" if is_rocm and other and spt < other else None)
                             if spt else marked("n/a", "pending")),
                "sampled": c.get("n_full", 0)})
    d["coverage_rows"] = rows
    d["coverage_clauses"] = coverage_clauses(cov, R)

    # --- headline metrics ---------------------------------------------------

    def q_do(b):
        return [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for rec, j in m["do_jobs"]
                if rec["repo"] == R["repo"] and m["band"](ts(j["created_at"])) == b]

    def q_nv(b):
        """Queue wait on the NV reference lane. Scoped to that lane's jobs, not
        every NV job in the repo: only the NV presubmit workflow is collected."""
        return [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for _, j in m["lane_jobs"][nk]
                if j["started_at"] and j["created_at"] and m["band"](ts(j["created_at"])) == b]

    def rbe_steps(key, b):
        xs = []
        for _, j in m["lane_jobs"][key]:
            if j["conclusion"] in SIGNAL and j["completed_at"] and m["band"](ts(j["created_at"])) == b:
                t = test_step_seconds(j)
                if t > 0:
                    xs.append(t)
        return xs

    def p90(xs):
        return f"{fmt(pct(xs, 90))}  (n={len(xs)})" if xs else "n/a"

    # The Metrics table's rows are fixed, so the template writes them out and
    # looks each figure up by name; only the numbers come from here.
    d["metrics"] = {
        "queue_peak": {"do": p90(q_do("peak")), "nv": p90(q_nv("peak"))},
        "queue_nominal": {"do": p90(q_do("nominal")), "nv": p90(q_nv("nominal"))},
        "tts_peak": {"do": p90([c["total"] for c in commits[rk] if c["band"] == "peak"]),
                     "rbe": p90(rbe_steps(rk, "peak")),
                     "nv": p90([c["total"] for c in commits[nk] if c["band"] == "peak"])},
        "tts_nominal": {"do": p90([c["total"] for c in commits[rk] if c["band"] == "nominal"]),
                        "rbe": p90(rbe_steps(rk, "nominal")),
                        "nv": p90([c["total"] for c in commits[nk] if c["band"] == "nominal"])},
        "nightly_bazel": {"do": p90(m["nightly"]["bazel"]),
                          "rbe": p90([x for x in m["nightly_test_steps"]["bazel"] if x > 0])},
        "nightly_pytest": {"do": p90(m["nightly"]["pytest"])},
    }
    # Hand-written rows a repo adds to the end of the table (metrics_lib).
    d["extra_metrics"] = [{"metric": lbl, "do": do, "rbe": rbe, "nv": rest[0] if rest else None}
                          for lbl, do, rbe, *rest in R.get("extra_metrics", ())]

    # --- phase breakdown ----------------------------------------------------
    def phase_cells(chosen):
        """The per-phase means plus the row's own total, dominant phase marked."""
        means = {k: statistics.fmean(c["phases"][k] for c in chosen) for k, _ in PHASES}
        dom = max(means, key=means.get)
        return ({k: marked(fmt(means[k]), "dominant" if k == dom else None) for k, _ in PHASES},
                fmt(statistics.fmean(c["total"] for c in chosen)))

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
                phases, total = phase_cells(chosen)
                rows.append({"line_item": head, "family": "all jobs", "phases": phases, "total": total,
                             "lane_signal": marked("—", "pending"), "n": len(chosen)})
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
                phases, total = phase_cells(cs)
                rows.append({"line_item": head, "family": label, "phases": phases, "total": total,
                             "lane_signal": lane_total, "n": len(cs)})
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

    d["phase_typical"] = phase_rows(middle)
    d["phase_slowest"] = phase_rows(slowest)
    d["phase_lanes"] = phase_lane_data(m, R)

    # --- ROCm vs NV ---------------------------------------------------------
    ck = (cov_key(R, "rocm"), cov_key(R, "nv"))
    rs, ns = lane_stats(m, rk), lane_stats(m, nk)
    rc, nc = cov.get(ck[0], {}).get("suite"), cov.get(ck[1], {}).get("suite")
    comparable = bool(rc and nc and rc >= 0.9 * nc)
    # Fixed rows, so the template names each one; queue wait is never a "win",
    # since a shorter queue says nothing about the work done after it.
    d["vs_nv"] = {}
    for k in ("tts50", "tts90", "exe50", "exe90", "q50"):
        a, b = win_pair(rs[k], ns[k], comparable and k != "q50")
        d["vs_nv"][k] = {"rocm": a, "nv": b}
    d["jobs_rocm"], d["jobs_nv"] = rs["jobs"], ns["jobs"]
    d["rocm_commits"], d["nv_commits"] = rs["n"], ns["n"]
    d["compare"] = compare_data(rc, nc, cov, ck, comparable)

    d.update(band_data(m, cov, R))
    d.update(tail_data(m, R))
    return d


def report_data(m, logs, since, until, base=None):
    peak, nominal = m["peak"], m["nominal"]
    days = (until - since).total_seconds() / 86400
    # The window is whatever collect.py fetched; a nightly run is one day, not a week.
    span = f"{days:g} day{'' if days == 1 else 's'}"
    slot = {
        "window": f"{since:%Y-%m-%d %H:%M} to {until:%Y-%m-%d %H:%M} UTC",
        "span": (f"Tuesday to Tuesday, {span}"
                 if days == 7 and since.weekday() == 1 and until.weekday() == 1 else span),
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "peak_hours": ", ".join(f"{h:02d}" for h in peak),
        "nominal_hours": ", ".join(f"{h:02d}" for h in nominal),
        # PHASES drives the phase table's columns, so the template loops them
        # rather than hard-coding nine headings that could drift out of step.
        "phase_keys": [k for k, _ in PHASES],
        "phase_labels": [lbl for _, lbl in PHASES],
        "baseline_label": base["label"] if base else None,
    }

    cov = coverage(logs, m)
    slot["repos"] = [repo_data(m, logs, cov, R, span) for R in REPOS]

    # --- infrastructure -----------------------------------------------------
    labels = collections.Counter()
    for rec, j in m["do_jobs"]:
        labels[next(l for l in j["labels"] if l.startswith("amd-do-"))] += 1
    slot["infra"] = {"labels_seen": len(labels),
                     "base_do_nodes": base.get("do_nodes") if base else None,
                     "base_do_gpus": base.get("do_gpus") if base else None}

    # --- peak vs nominal ----------------------------------------------------
    arr, nva = m["arrivals"], m["nv_arrivals"]
    slot["arrivals"] = [{"am_hour": f"{h:02d}", "am_do": arr[h], "am_nv": nva[h],
                         "pm_hour": f"{h + 12:02d}", "pm_do": arr[h + 12], "pm_nv": nva[h + 12]}
                        for h in range(12)]
    lo, hi = min(arr.values()), max(arr.values())
    nv_peak = sorted(sorted(range(24), key=lambda h: (-nva[h], h))[:8])
    slot["swing"] = f"{hi / lo:.1f}"
    slot["arrivals_lo"], slot["arrivals_hi"] = lo, hi
    slot["nv_peak_hours"] = ", ".join(f"{h:02d}" for h in nv_peak)
    slot["shared_peak_hours"] = len(set(nv_peak) & set(peak))

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
            rows.append({"component": f"{lbl} ({pool} pool)",
                         **{f"{b}_p{p}": fmt(pct(comp[(k, pool, b)], p))
                            for b in ("peak", "nominal") for p in (50, 90)}})
    slot["pool_components"] = rows

    def flat(a, b):
        return abs(a - b) <= max(0.2 * max(a, b), 10)

    verdicts = []
    for pool, _ in pools:
        qp, qn = pct(comp[("queue", pool, "peak")], 50), pct(comp[("queue", pool, "nominal")], 50)
        tp, tn = pct(comp[("queue", pool, "peak")], 90), pct(comp[("queue", pool, "nominal")], 90)
        if qp is None or qn is None:
            continue
        kind = "flat" if flat(qp, qn) and flat(tp, tn) else "tail" if flat(qp, qn) else "demand"
        verdicts.append({"pool": pool, "peak": fmt(qp), "nominal": fmt(qn), "kind": kind,
                         "tail": {"nominal": fmt(tn), "peak": fmt(tp)}})
    slot["pool_verdicts"] = verdicts

    # --- scale sets ---------------------------------------------------------
    rows = []
    for lab, n in labels.most_common():
        xs = [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for rec, j in m["do_jobs"]
              if lab in j["labels"]]
        g = re.search(r"[.-](\d+)$", lab)
        rows.append({"label": lab, "gpus": "0 (CPU)" if "cpu" in lab else (g.group(1) if g else "?"),
                     "jobs": n, **{f"p{p}": fmt(pct(xs, p)) for p in (50, 90, 99)}, "max": fmt(max(xs))})
    slot["scale_sets"] = rows

    rows = []
    for lab, _ in labels.most_common():
        xs = [job_phases(j)["gaps"] for rec, j in m["do_jobs"]
              if lab in j["labels"] and j["conclusion"] in SIGNAL and j["completed_at"]]
        if not xs:
            continue
        rows.append({"label": lab, "jobs": len(xs), **{f"p{p}": fmt(pct(xs, p)) for p in (50, 90, 99)},
                     "over": sum(x > 300 for x in xs), "worst": fmt(max(xs))})
    slot["teardown"] = rows

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
        seen = sorted(ts(j["created_at"]).astimezone(CDT) for j in js)
        rows.append({"repo": repo, "label": lab, "job_type": ty, "jobs": len(js),
                     **{f"p{p}": fmt(pct(q, p)) for p in (50, 90, 99)},
                     "over": sum(x > 600 for x in q),
                     "seen": f"{seen[0]:%m-%d} to {seen[-1]:%m-%d}"})
    slot["nv_pools"] = rows

    # --- RBE detail ---------------------------------------------------------
    rows = []
    for lbl, cat, inv_idx in [(f"{R['label']} {sfx}", cat, i) for R in REPOS for sfx, cat, i in R["rbe"]]:
        el = [l["elapsed"][inv_idx] for l in logs if l["cat"] == cat and l["elapsed"]]
        w, c = [e[0] for e in el], [e[1] for e in el]
        ratio = (pct(w, 50) / pct(c, 50)) if el and pct(c, 50) else None
        rows.append({"set": lbl, "sampled": len(el),
                     "wall_p50": fmt(pct(w, 50)), "wall_p90": fmt(pct(w, 90)),
                     "cpath_p50": fmt(pct(c, 50)), "cpath_p90": fmt(pct(c, 90)),
                     "ratio": f"{ratio:.0f}x" if ratio else "n/a"})
    slot["rbe"] = rows

    # --- the ask ------------------------------------------------------------
    slot["ask_coverage"] = ask_coverage_data(cov, base)
    slot["changes"] = changes_data(m, cov, base) if base else None
    return slot


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


def band_data(m, cov, R):
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
            cells = {}
            for b, _ in bands:
                for p in (50, 90):
                    v, w = pct(s[b][metric], p), pct(o[b][metric], p)
                    win = is_rocm and comp and v is not None and w is not None and v < w
                    cells[f"{b}_p{p}"] = marked(fmt(v), "win" if win else None)
            out.append({"metric": label, "lane": lname, **cells,
                        "n": f"{len(s['peak'][metric])} / {len(s['nominal'][metric])}"})
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
    out = {"band_rows": rows}

    rows = []
    for blk in range(8):
        def keep(t, blk=blk):
            return hour(t) // 3 == blk
        r, n = lane(rk, keep, r_types), lane(nk, keep, n_types)
        hrs = range(3 * blk, 3 * blk + 3)
        tag = ("peak" if sum(h in m["peak"] for h in hrs) >= 2 else
               "nominal" if sum(h in m["nominal"] for h in hrs) >= 2 else "")
        rows.append({"hours": f"{3 * blk:02d}:00-{3 * blk + 3:02d}:00" + (f" ({tag})" if tag else ""),
                     "commits": f"{len(r['tts'])} / {len(n['tts'])}",
                     **{k: f"{fmt(pct(r[k], 90))} / {fmt(pct(n[k], 90))}"
                        for k in ("tts", "queue", "exec", "test")}})
    out["band_hours"] = rows
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


def tail_data(m, cfg):
    repo = cfg["repo"]
    rk, nk = lane_key(cfg, "rocm"), lane_key(cfg, "nv")
    ps = (50, 90, 95, 99)
    rows = []
    for lane_lbl, key, is_rocm in [("ROCm", rk, 1), ("NV", nk, 0)]:
        cs = m["commits"][key]
        tt = [c["total"] for c in cs]
        rows.append({"lane": lane_lbl, "job_type": "*commit time to signal*", "n": len(tt),
                     **{f"p{p}": fmt(pct(tt, p)) for p in ps}})
        by = collections.defaultdict(list)
        for c in cs:
            for j in c["jobs"]:
                by[j["type"]].append(j["exec"])
        for t, xs in sorted(by.items()):
            rows.append({"lane": lane_lbl, "job_type": t, "n": len(xs),
                         **{f"p{p}": fmt(pct(xs, p)) for p in ps}})
    out = {"tail_rows": rows, "same_commit": None}

    R = {c["sha"]: c for c in m["commits"][rk]}
    N = {c["sha"]: c for c in m["commits"][nk]}
    common = set(R) & set(N)
    both = sorted(s for s in common if R[s]["passed"] and N[s]["passed"])
    if not both:
        return out
    ratio = [R[s]["total"] / N[s]["total"] for s in both if N[s]["total"] > 0]
    faster = sum(1 for s in both if R[s]["total"] < N[s]["total"])
    sc = {"n_both": len(both), "n_failed": len(common) - len(both)}
    sc["ratio"] = {**{f"p{p}": f"{pct(ratio, p):.2f}x" for p in (10, 50, 90, 99)},
                   "faster": faster, "faster_pct": f"{100 * faster / len(both):.0f}"}

    def outlier_row(slow_side, shas):
        rt = [R[s]["total"] for s in shas]
        nt = [N[s]["total"] for s in shas]
        rq = [R[s]["phases"]["queue"] for s in shas]
        rx = [R[s]["phases"]["test"] for s in shas]
        rg = [R[s]["phases"]["gaps"] for s in shas]
        n90 = pct([N[s]["total"] for s in both], 90)
        r90 = pct([R[s]["total"] for s in both], 90)
        other_slow = sum(1 for s in shas if (N[s]["total"] >= n90 if slow_side == "rocm" else R[s]["total"] >= r90))
        return {"prs": len(shas), "rocm_p50": fmt(pct(rt, 50)), "nv_p50": fmt(pct(nt, 50)),
                "ratio": f"{pct([a / b for a, b in zip(rt, nt) if b], 50):.2f}x",
                "other_slow": other_slow, "other_slow_pct": f"{100 * other_slow / len(shas):.0f}",
                "queue": fmt(statistics.fmean(rq)), "test": fmt(statistics.fmean(rx)),
                "teardown": fmt(statistics.fmean(rg))}

    r90 = pct([R[s]["total"] for s in both], 90)
    n90 = pct([N[s]["total"] for s in both], 90)
    slow_r = [s for s in both if R[s]["total"] >= r90]
    slow_n = [s for s in both if N[s]["total"] >= n90]
    sc["outliers"] = {"rocm": outlier_row("rocm", slow_r), "nv": outlier_row("nv", slow_n)}

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
        rows.append({"job_type": lbl, "commits": len(pr),
                     **{f"p{p}": f"{fmt(pct(pr, p))} / {fmt(pct(pn, p))}" for p in (50, 90, 99)},
                     "ratio": f"{pct([x / y for x, y in zip(pr, pn) if y], 50):.2f}x",
                     "tail": f"{fmt(pct([x for x, _ in tail], 50))} / {fmt(pct([y for _, y in tail], 50))}"})
    sc["matched"] = rows

    worst = sorted(both, key=lambda s: -(R[s]["total"] / N[s]["total"] if N[s]["total"] else 0))[:10]
    rows = []
    for s in worst:
        c = R[s]
        top = sorted(((k, v) for k, v in c["phases"].items()), key=lambda kv: -kv[1])[:2]
        rows.append({"sha": s[:10], "pr": c["prs"][0] if c["prs"] else lookup_pr(m, repo, s),
                     "rocm": fmt(c["total"]), "nv": fmt(N[s]["total"]),
                     "ratio": f"{c['total'] / N[s]['total']:.1f}x",
                     "where": [{"phase": dict(PHASES)[k].lower(), "value": fmt(v)} for k, v in top]})
    sc["largest_gap"] = rows
    out["same_commit"] = sc
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


def coverage_clauses(cov, R):
    """Per job family: the ROCm suite as a share of CUDA's, and cost per test."""
    out = []
    for suffix, keys in cov_rows(R):
        a, b = cov.get(keys["rocm"], {}), cov.get(keys["nv"], {})
        if not (a.get("suite") and b.get("suite")):
            continue
        per = None
        if a.get("spt") and b.get("spt"):
            r_ = a["spt"] / b["spt"]
            per = ({"factor": f"{r_:.2f}", "direction": "slower" if r_ > 1 else "faster"} if r_ >= 1
                   else {"factor": f"{1 / r_:.2f}", "direction": "faster"})
        out.append({"label": f"{R['label']}{suffix}", "share": f"{100 * a['suite'] / b['suite']:.0f}",
                    "rocm": a["suite"], "nv": b["suite"], "per_test": per})
    return out


def compare_data(rc, nc, cov, ck, comparable):
    """Whether the two suites are close enough in size for durations to be compared."""
    if not (rc and nc):
        return None
    a, b = cov.get(ck[0], {}).get("spt"), cov.get(ck[1], {}).get("spt")
    per = None
    if a and b:
        per = ({"factor": f"{a / b:.2f}", "direction": "slower"} if a > b
               else {"factor": f"{b / a:.2f}", "direction": "faster"})
    return {"comparable": comparable, "rocm": rc, "nv": nc, "share": f"{100 * rc / nc:.0f}", "per_test": per}


def phase_lane_data(m, R):
    """Per lane: how many commits signalled, and what happened to the ones that did not."""
    out = []
    for lane, other in [("ROCm", "NV"), ("NV", "ROCm")]:
        key = lane_key(R, lane.lower())
        cs = m["commits"][key]
        if not cs:
            continue
        ap = m["approval"][key]
        out.append({
            "lane": lane,
            "signalled": len(cs),
            "superseded": len([c for c in cs if c["phases"]["superseded"] > 0]),
            "rerun": len([c for c in cs if c["rerun"]]),
            "unsignalled": m["unsignalled"][key],
            "approval": ({"n": len(ap), "median": fmt(pct(ap, 50)), "max": fmt(max(ap)), "other": other}
                         if ap else None)})
    return out


def ask_coverage_data(cov, base):
    """Per repo: ROCm targets against NV's, and the gain since the baseline."""
    out = []
    for R in REPOS:
        clauses = []
        for suffix, keys in cov_rows(R):
            a, b = cov.get(keys["rocm"], {}).get("suite"), cov.get(keys["nv"], {}).get("suite")
            if not (a and b):
                continue
            bk = f"{R['short']}_rocm_tests"
            up = ({"count": base[bk], "label": base["label"]}
                  if base and bk in base and not suffix else None)
            clauses.append({"kind": suffix or " presubmit", "rocm": a, "nv": b, "up": up})
        if clauses:
            out.append({"repo": R["repo"], "clauses": clauses})
    return out


# Keyed by the names the --baseline JSON uses; keep them stable.
POOL_NOW = {f"{R['short'].upper()} single-GPU pool": R["pool"] for R in REPOS}


def changes_data(m, cov, base):
    """Compare against an earlier report's figures, read from the --baseline JSON."""
    rows = []
    for R in REPOS:
        bk = f"{R['short']}_rocm_tests"
        if bk in base:
            now = cov.get(cov_key(R, "rocm"), {}).get("suite")
            rows.append({"item": f"{R['label']} ROCm presubmit tests", "then": base[bk], "now": now or "n/a"})
    for name, (old_lab, p50, p90) in base.get("queue", {}).items():
        lab = POOL_NOW[name]
        xs = [(ts(jb["started_at"]) - ts(jb["created_at"])).total_seconds() for _, jb in m["do_jobs"]
              if lab in jb["labels"]]
        rows.append({"item": f"{name} queue p50 / p90 (`{old_lab}` then, `{lab}` now)",
                     "then": f"{p50} / {p90}", "now": f"{fmt(pct(xs, 50))} / {fmt(pct(xs, 90))}"})
    # The baseline note is hand-written Markdown; flatten it onto one line so it
    # cannot break out of its blockquote.
    return {"label": base["label"], "rows": rows,
            "note": base["note"].replace("\n", " ") if base.get("note") else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--md", required=True, help="path of the Markdown report to write")
    ap.add_argument("--json", help="also write the report's figures as JSON, for --baseline next time")
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--baseline", help="JSON with an earlier report's figures; enables 'Changes since'")
    a = ap.parse_args()
    m = build(a.data)
    logs = load_logs(a.data)
    since = dt.datetime.fromisoformat(a.since)
    until = dt.datetime.fromisoformat(a.until)
    base = json.load(open(a.baseline)) if a.baseline else None
    data = report_data(m, logs, since, until, base)
    with open(a.md, "w") as f:
        f.write(render(data))
    print(f"wrote {a.md}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        print(f"wrote {a.json}")
    for k, cs in m["commits"].items():
        print(k, "commits", len(cs), "unsignalled", m["unsignalled"][k])


if __name__ == "__main__":
    main()
