#!/usr/bin/env python3
"""Build the 7-day CI metrics HTML report.

Full pipeline (all steps resumable; ~1.5h wall, bounded by the 5000/h API budget):

  D=DATA_DIR; S=2026-09-22T15:00; U=2026-09-29T15:00         # UTC, Tuesday to Tuesday
  python3 collect.py        --since $S --until $U --out $D   # runs + jobs + steps
  python3 fetch_attempts.py --data $D                        # earlier attempts of re-runs
  python3 sample_logs.py    --data $D                        # test counts, critical path
  python3 report.py         --data $D --since $S --until $U --html OUT.html [--baseline BASE.json]

report.py writes OUT.md next to OUT.html (--md overrides the path, --no-md skips
it); the Markdown is converted from the rendered HTML, so both hold the same
numbers and prose.

--baseline adds a "Changes since" section from an earlier report's figures:
  {"label": "...", "jax_rocm_tests": N, "xla_rocm_tests": N, "do_nodes": N, "do_gpus": N,
   "queue": {"JAX single-GPU pool": [old_label, p50, p90], ...}, "note": "html"}
Every key except label is optional; queue names must match POOL_NOW.
"""
import argparse
import collections
import datetime as dt
import html
import json
import os
import re
import statistics

from html_md import to_markdown
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
    return html.escape(str(s))


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


# ---- HTML ------------------------------------------------------------------

CSS = """
:root{--ink:#12161c;--mute:#5d6b7a;--line:#dfe4ea;--bg:#fff;--soft:#f6f8fa;
      --win:#0f7b3d;--winbg:#e8f6ed;--accent:#1f4e79}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px 64px;background:#eef1f5;color:var(--ink);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;background:var(--bg);padding:40px 44px 48px;
      border-radius:10px;box-shadow:0 1px 3px rgba(16,24,40,.08),0 8px 24px rgba(16,24,40,.06)}
h1{margin:0 0 6px;font-size:27px;letter-spacing:-.02em}
h2{margin:38px 0 10px;font-size:19px;letter-spacing:-.01em;
   padding-bottom:7px;border-bottom:2px solid var(--line)}
h2.repo{margin-top:52px;font-size:23px;color:var(--accent);
        border-bottom:3px solid var(--accent);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
h4{margin:20px 0 6px;font-size:14px;color:var(--ink);font-weight:600}
h3{margin:26px 0 8px;font-size:15.5px;color:var(--accent)}
p{margin:9px 0}
.sub{color:var(--mute);font-size:14px;margin:0 0 4px}
.meta{color:var(--mute);font-size:12.5px;margin:10px 0 0}
.defs{background:var(--soft);border:1px solid var(--line);border-radius:8px;
      padding:14px 18px;margin:18px 0 6px;font-size:13.5px}
.defs b{color:var(--accent)}
table{border-collapse:collapse;width:100%;margin:12px 0 4px;font-size:13.5px}
th,td{padding:8px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:600;font-size:12.5px;letter-spacing:.02em;
   text-transform:uppercase;color:var(--mute);border-bottom:2px solid var(--line)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:#fbfcfd}
.win{color:var(--win);font-weight:700;background:var(--winbg)}
.win::after{content:" \\25B2";font-size:9px;vertical-align:2px}
.lane{font-weight:600}
.rocm{color:var(--accent)}
.pending{color:#9aa5b1;font-style:italic}
td.dominant{font-weight:700;color:var(--ink);background:#fbf6e8}
.scope{display:inline-block;margin-left:8px;padding:1px 8px;border-radius:99px;
       background:var(--soft);border:1px solid var(--line);color:var(--mute);
       font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;
       vertical-align:middle}
.note{font-size:13px;color:var(--mute);margin:10px 0 0;padding-left:12px;
      border-left:3px solid var(--line)}
.key{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:14px 0 0;
     font-size:13px;color:var(--mute)}
.chip{display:inline-block;padding:2px 9px;border-radius:99px;font-weight:700;
      font-size:12px;color:var(--win);background:var(--winbg)}
code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:12.5px}
.scroll{overflow-x:auto}
@media print{body{background:#fff;padding:0}.wrap{box-shadow:none;padding:0}
  h2{break-after:avoid}table{break-inside:avoid}}
"""


def table(head, rows, num_from=1):
    """head: list of str; rows: list of lists of (html, cls) or str."""
    th = "".join(f"<th class='n'>{h}</th>" if i >= num_from else f"<th>{h}</th>" for i, h in enumerate(head))
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            if isinstance(c, tuple):
                txt, cls = c
            else:
                txt, cls = c, ("n" if i >= num_from else "")
            tds.append(f"<td class='{cls}'>{txt}</td>" if cls else f"<td>{txt}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def win_pair(rocm, nv, comparable):
    a, b = fmt(rocm), fmt(nv)
    if comparable and rocm is not None and nv is not None and rocm < nv:
        return (a, "n win"), (b, "n")
    return (a, "n"), (b, "n")


def repo_section(m, logs, cov, R, span):
    """Every table scoped to one repo: coverage, metrics, phases, ROCm vs NV, bands, tail.

    Driven entirely by the repo's REPOS entry, so a new repo is a config entry.
    Sections about shared hardware (DO pools, NV pools, RBE, scale-sets) stay in
    render(), since they describe a fleet both repos sit on.
    """
    commits = m["commits"]
    rk, nk = lane_key(R, "rocm"), lane_key(R, "nv")
    P = [f"<h2 class='repo'>{esc(R['repo'])}</h2>"]
    add = P.append

    # --- coverage -----------------------------------------------------------
    add("<h3>Test coverage and cost per test</h3>")
    add("<p class='sub'>Test counts come from bazel's <code>Executed X out of Y tests</code> line in sampled job "
        "logs from the last 48h of the window, so suite sizes are current. Test-step p50 and seconds per test use "
        "only runs that executed at least 90% of the suite (remote-cache hits make the rest cheap and "
        "unrepresentative).</p>")
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
    add(table(["Job family", "Lane", "Tests in suite", "Test step p50 (full-suite runs)", "Seconds per test",
               "Full-suite runs sampled"], rows, num_from=2))
    add(coverage_note(cov, R))

    # --- headline metrics ---------------------------------------------------
    add("<h3>Metrics</h3>")

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
        return f"{fmt(pct(xs, 90))} &nbsp;(n={len(xs)})" if xs else "n/a"

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
    add(table(["#", "Metric", "DO", "RBE"], rows, num_from=2).replace("<th>#</th>", "<th class='n'>#</th>"))
    note = ("<p class='note'>DO queue wait = <code>started_at - created_at</code> for every DO job in the band "
            "(all workflows in the repo). DO time to signal = PR head SHA first triggered (or approved, for fork "
            "PRs) to its last ROCm job finishing. RBE cells = wall time of the bazel test step that dispatches to "
            "EngFlow, for jobs created in the band.")
    if R["nightly"]:
        wfs = ", ".join(f"<code>{esc(w)}</code>" for w in R["nightly"])
        note += f" Nightly rows cover {wfs}, one signal per run: run created to its last ROCm job finishing."
    add(note + "</p>")

    # --- phase breakdown ----------------------------------------------------
    add("<h3>Where the time to signal goes <span class='scope'>ROCm on DO, NV on Google runners</span></h3>")
    add("<p class='sub'>For each commit, the lane's job that finishes last is followed from the moment the SHA was "
        "first triggered. Every second lands in exactly one named phase, and each row adds up to its total "
        "(checked in code), so there is no unaccounted time. One row per lane, over all hours of the window.</p>")
    if R["families"]:
        add(f"<p class='sub'>{esc(R['repo'])} runs more than one job family in a single workflow, so its lanes get "
            "one row per family. Both rows cover the same commits, so they share one <b>n</b>; <b>Total</b> is that "
            "family's own signal and <b>Lane signal</b> is the combined one the PR actually waits for - the slower "
            "family on each commit, which is why it repeats down a lane. The count beside the family name is how "
            "often it was the family that finished last.</p>")
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
            head = f"{lane_lbl} Time to signal presubmit <span class='scope'>{scope}</span>"
            if not R["families"]:
                # One family, so the row's own total already is the lane signal.
                rows.append([head, "all jobs"] + phase_cells(chosen)
                            + [("&mdash;", "n pending"), str(len(chosen))])
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
                label = f"{fam_lbl} <span class='scope'>last on {last[sub_key]}/{len(shas)}</span>"
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

    add("<h4>Typical commit (commits between p40 and p60 of time to signal, mean of each phase)</h4>")
    add(table(phase_head, phase_rows(middle), num_from=2))
    add("<h4>Slowest 10% of commits (at or above p90, mean of each phase)</h4>")
    add(table(phase_head, phase_rows(slowest), num_from=2))
    add(phase_notes(m, R))

    # --- ROCm vs NV ---------------------------------------------------------
    add("<h3>ROCm versus the NVIDIA reference lane</h3>")
    add("<p class='sub'>Scoped to the presubmit test lane, <code>pull_request</code> runs only.</p>")
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
    rows[0].append((f"ROCm {rs['jobs']} &middot; NV {ns['jobs']}", "n' rowspan='5"))
    add(table(["Metric", "ROCm (DO)", "NV (Google)", "Jobs per commit"], rows))
    add(f"<p class='meta'>n = {rs['n']} ROCm commits, {ns['n']} NV commits.</p>")
    add(compare_note(R["repo"], rc, nc, cov, ck, comparable))

    add(band_section(m, cov, R))
    add(tail_section(m, R))
    return "".join(P)


def render(m, logs, since, until, base=None):
    P = []
    add = P.append
    commits, peak, nominal = m["commits"], m["peak"], m["nominal"]
    gen = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # The window is whatever collect.py fetched; a nightly run is one day, not a week.
    days = (until - since).total_seconds() / 86400
    span = f"{days:g} day{'' if days == 1 else 's'}"
    hdr_span = (f"Tuesday to Tuesday, {span}"
                if days == 7 and since.weekday() == 1 and until.weekday() == 1 else span)
    add(f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>CI metrics - upstream JAX &amp; OpenXLA (ROCm)</title><style>{CSS}</style></head>"
        f"<body><div class='wrap'>")
    add("<h1>CI metrics: upstream JAX &amp; OpenXLA on ROCm</h1>")
    add(f"<p class='sub'>Window <b>{since:%Y-%m-%d %H:%M} to {until:%Y-%m-%d %H:%M} UTC</b> "
        f"({hdr_span}). Generated {gen}.</p>")
    add("<div class='defs'><b>DO</b> = Digital Ocean self-hosted runners (<code>amd-do-*</code>), where presubmits "
        "build. &nbsp; <b>RBE</b> = EngFlow remote pool (<code>wardite</code>), where tests dispatch. &nbsp; "
        "<b>NV</b> = Google-hosted NVIDIA runners, the reference lane. &nbsp; <b>Presubmit</b> = "
        "<code>pull_request</code> runs only; one commit = one PR head SHA.</div>")
    add("<div class='key'><span><span class='chip'>green</span> ROCm beats NV like-for-like</span>"
        "<span>plain = trails, or suites are not comparable</span></div>")
    add(f"<p class='meta'>Peak hours (CDT (UTC-5), busiest 8 by measured DO arrivals): "
        f"<code>{', '.join(f'{h:02d}' for h in peak)}</code> &nbsp;&middot;&nbsp; nominal (quietest 8): "
        f"<code>{', '.join(f'{h:02d}' for h in nominal)}</code></p>")

    cov = coverage(logs, m)
    for R in REPOS:
        add(repo_section(m, logs, cov, R, span))

    # --- global sections ----------------------------------------------------

    # --- infrastructure -----------------------------------------------------
    add("<h2>Infrastructure</h2>")

    def was(k):
        return f" ({esc(base['label'])}: {base[k]})" if base and k in base else ""
    labels = collections.Counter()
    runners = collections.defaultdict(set)
    for rec, j in m["do_jobs"]:
        lab = next(l for l in j["labels"] if l.startswith("amd-do-"))
        labels[lab] += 1
        if j["runner_name"]:
            runners[lab].add(j["runner_name"])
    add(table(["Item", "Value"], [
        ["DO node count (DPX)", (f"not re-measured: not derivable from job metadata{was('do_nodes')}", "n pending")],
        ["DO GPUs (executors)", (f"not re-measured: not derivable from job metadata{was('do_gpus')}", "n pending")],
        ["DO runner labels seen this week", str(len(labels))],
        ["Local node count", ("pending: not derivable from CI metadata (owner: user)", "n pending")],
        ["Local GPUs", ("pending: not derivable from CI metadata (owner: user)", "n pending")]]))

    # --- peak vs nominal ----------------------------------------------------
    add("<h2>Peak versus nominal: measured</h2>")
    arr, nva = m["arrivals"], m["nv_arrivals"]
    rows = [[f"{h:02d}", str(arr[h]), str(nva[h]), "", f"{h + 12:02d}", str(arr[h + 12]), str(nva[h + 12])]
            for h in range(12)]
    add(table(["Hour CDT (UTC-5)", "DO jobs", "NV jobs", "", "Hour CDT (UTC-5)", "DO jobs", "NV jobs"], rows,
              num_from=0))
    lo, hi = min(arr.values()), max(arr.values())
    nv_peak = sorted(sorted(range(24), key=lambda h: (-nva[h], h))[:8])
    shared = len(set(nv_peak) & set(peak))
    add(f"<p class='note'>Arrival volume swings <b>{hi / lo:.1f}x</b> across the day ({lo} to {hi} jobs/hour). "
        f"DO jobs = every ROCm job on DO in the collected workflows; NV jobs = jobs of the two NV presubmit lanes, "
        f"by the hour they were created. NV's busiest 8 hours are <code>{', '.join(f'{h:02d}' for h in nv_peak)}</code>, "
        f"{shared} of them DO peak hours.</p>")
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
    for k, lbl in [("queue", "Queue"), ("setup", "Set-up (runner, docker, checkout, env)"), ("test", "Test execution")]:
        for pool, lab in pools:
            rows.append([f"{lbl} <span class='scope'>{pool} pool</span>"]
                        + [fmt(pct(comp[(k, pool, b)], 50)) for b in ("peak", "nominal")]
                        + [fmt(pct(comp[(k, pool, b)], 90)) for b in ("peak", "nominal")])
    add(table(["Per-job component (DO, single-GPU pools)", "Peak p50", "Nominal p50", "Peak p90", "Nominal p90"], rows))
    verdicts = []
    for pool, _ in pools:
        qp, qn = pct(comp[("queue", pool, "peak")], 50), pct(comp[("queue", pool, "nominal")], 50)
        tp, tn = pct(comp[("queue", pool, "peak")], 90), pct(comp[("queue", pool, "nominal")], 90)
        if qp is None or qn is None:
            continue

        def flat(a, b):
            return abs(a - b) <= max(0.2 * max(a, b), 10)

        if flat(qp, qn) and flat(tp, tn):
            v = "flat at p50 and p90, so the wait is provisioning latency, not contention"
        elif flat(qp, qn):
            v = (f"flat at p50 but p90 goes {fmt(tn)} to {fmt(tp)}, so the typical wait is provisioning "
                 f"latency and the tail is contention")
        else:
            v = "it moves with demand, so part of the wait is contention"
        verdicts.append(f"<b>{pool} pool</b>: queue p50 {fmt(qp)} at peak vs {fmt(qn)} at nominal; {v}")
    add(f"<p class='note'>Arrivals swing {hi / lo:.1f}x. {'; '.join(verdicts)}.</p>")

    # --- scale sets ---------------------------------------------------------
    add("<h2>DO scale-sets</h2>")
    rows = []
    for lab, n in labels.most_common():
        xs = [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for rec, j in m["do_jobs"]
              if lab in j["labels"]]
        g = re.search(r"[.-](\d+)$", lab)
        gpus = "0 (CPU)" if "cpu" in lab else (g.group(1) if g else "?")
        rows.append([f"<code>{esc(lab)}</code>", gpus, str(n)] + [fmt(pct(xs, p)) for p in (50, 90, 99)] + [fmt(max(xs))])
    add(table(["Scale-set", "GPUs", "Jobs", "Queue p50", "p90", "p99", "Max"], rows))
    add("<p class='note'>GPUs per job read from the label suffix. All ROCm jobs in the collected workflows, "
        "any conclusion except skipped.</p>")
    add("<h3>Runner teardown: tests done, check still pending</h3>")
    add("<p class='sub'>Job wall time minus the sum of its steps. Almost all of it sits after the "
        "<code>Complete job</code> step, before GitHub marks the job finished.</p>")
    rows = []
    for lab, _ in labels.most_common():
        xs = [job_phases(j)["gaps"] for rec, j in m["do_jobs"]
              if lab in j["labels"] and j["conclusion"] in SIGNAL and j["completed_at"]]
        if not xs:
            continue
        rows.append([f"<code>{esc(lab)}</code>", str(len(xs))] + [fmt(pct(xs, p)) for p in (50, 90, 99)]
                    + [str(sum(x > 300 for x in xs)), fmt(max(xs))])
    add(table(["Scale-set", "Jobs", "Teardown p50", "p90", "p99", "Jobs over 5 min", "Worst"], rows))

    # --- NV runner pools ----------------------------------------------------
    add("<h2>NV runner pools</h2>")
    add("<p class='sub'>Google-hosted runners of the two NV presubmit lanes, by runner label and job type. Queue = "
        "<code>started_at - created_at</code>. A label whose last day seen is before the end of the window "
        "stopped appearing partway through the week.</p>")
    nv_pools = collections.defaultdict(list)
    for R in REPOS:
        key = lane_key(R, "nv")
        for _, j in m["lane_jobs"][key]:
            if j["started_at"] and j["created_at"]:
                nv_pools[(R["repo"], ", ".join(j["labels"]), job_type(key, j["name"]))].append(j)
    rows = []
    for (repo, lab, ty), js in sorted(nv_pools.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        q = [(ts(j["started_at"]) - ts(j["created_at"])).total_seconds() for j in js]
        days = sorted(ts(j["created_at"]).astimezone(CDT) for j in js)
        rows.append([repo, f"<code>{esc(lab)}</code>", ty, str(len(js))] + [fmt(pct(q, p)) for p in (50, 90, 99)]
                    + [str(sum(x > 600 for x in q)), f"{days[0]:%m-%d} to {days[-1]:%m-%d}"])
    add(table(["Repo", "Runner label", "Job type", "Jobs", "Queue p50", "p90", "p99", "Over 10 min",
               "Seen (CDT)"], rows, num_from=3))

    # --- RBE detail ---------------------------------------------------------
    add("<h2>RBE (EngFlow) detail</h2>")
    add("<p class='sub'>From bazel's <code>Elapsed time / Critical Path</code> line in sampled job logs across the "
        "whole week. The gap between wall time and critical path is scheduling, queueing, cache lookup and "
        "local work, not test execution on the critical path.</p>")
    rows = []
    for lbl, cat, inv_idx in [(f"{R['label']} {sfx}", cat, i) for R in REPOS for sfx, cat, i in R["rbe"]]:
        el = [l["elapsed"][inv_idx] for l in logs if l["cat"] == cat and l["elapsed"]]
        w = [e[0] for e in el]
        c = [e[1] for e in el]
        ratio = (pct(w, 50) / pct(c, 50)) if el and pct(c, 50) else None
        rows.append([lbl, str(len(el)), fmt(pct(w, 50)), fmt(pct(w, 90)), fmt(pct(c, 50)), fmt(pct(c, 90)),
                     f"{ratio:.0f}x" if ratio else "n/a"])
    add(table(["Set", "Invocations sampled", "Wall p50", "Wall p90", "Critical path p50", "Critical path p90",
               "Wall:cpath at p50"], rows))
    add("<p class='note'>Both ROCm presubmits run <code>--config=rocm_rbe_dynamic</code>: tests race a local run on "
        "the DO GPU against the EngFlow run (<code>--strategy=TestRunner=dynamic</code> in JAX, "
        "<code>--spawn_strategy=dynamic</code> in XLA), so RBE wall time includes work that may have finished "
        "locally.</p>")
    add("<h3>Executor queue by pool</h3>")
    add("<p class='pending'>Not refreshed. This table came from a manual AWS dashboard export; it needs a new "
        "export for this window (owner: user).</p>")

    # --- ask ----------------------------------------------------------------
    add(ask_section(cov, base))
    if base:
        add(changes_section(m, cov, base))
    add("</div></body></html>")
    return "\n".join(P)


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
            row = [(label, "' rowspan='2")] if is_rocm else []
            out.append(row + [(lname, "lane rocm" if is_rocm else "lane")] + cells
                       + [f"{len(s['peak'][metric])} / {len(s['nominal'][metric])}"])
        return out

    P = ["<h3>Peak versus nominal: ROCm and NV side by side</h3>",
         "<p class='sub'>Both lanes are measured over the same clock hours: the peak and nominal hours listed at the "
         "top, which come from DO arrivals pooled across every repo, so all sections share one clock. Jobs are "
         "placed by the hour they were created, commits by the hour their SHA was first triggered.</p>"]
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
    P.append(table(["Metric", "Lane", "Peak p50", "Peak p90", "Nominal p50", "Nominal p90", "n (peak / nominal)"],
                   rows, num_from=2))

    rows = []
    for blk in range(8):
        def keep(t, blk=blk):
            return hour(t) // 3 == blk
        r, n = lane(rk, keep, r_types), lane(nk, keep, n_types)
        hrs = range(3 * blk, 3 * blk + 3)
        tag = ("peak" if sum(h in m["peak"] for h in hrs) >= 2 else
               "nominal" if sum(h in m["nominal"] for h in hrs) >= 2 else "")
        rows.append([f"{3 * blk:02d}:00-{3 * blk + 3:02d}:00" + (f" <span class='scope'>{tag}</span>" if tag else ""),
                     f"{len(r['tts'])} / {len(n['tts'])}"]
                    + [f"{fmt(pct(r[k], 90))} / {fmt(pct(n[k], 90))}" for k in ("tts", "queue", "exec", "test")])
    P.append("<p class='meta'>p90 by time of day (3-hour blocks, CDT (UTC-5)):</p>")
    P.append(table(["Hours", "Commits ROCm / NV", "Time to signal ROCm / NV", "Queue ROCm / NV",
                    "Test job execution ROCm / NV", "Test step ROCm / NV"], rows))
    P.append("<p class='note'>Queue wait covers every job of the presubmit lane, so the ROCm figure differs slightly "
             "from the Metrics table, which takes every DO job in the repo. Job execution and test step use only the "
             "matched test job types listed above; job types with no counterpart on the other lane are left out. "
             "Highlights follow the suite-size rule used above, and queue wait is never highlighted. In the "
             "time-of-day tables a block is tagged peak or nominal when at least 2 of its 3 hours are, and test-job "
             "rows pool the matched job types; read p90 with its commit count.</p>")
    return "".join(P)


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
    P = ["<h3>Tail latency and same-commit comparison</h3>",
         "<p class='sub'>Job execution = <code>completed_at - started_at</code> for the jobs of the run that "
         "signalled, per job type. The same-commit tables compare ROCm and NV on the <b>same PR head SHA</b>, "
         "so both lanes saw the same code change and the same set of tests invalidated in the cache.</p>"]
    ps = (50, 90, 95, 99)
    rows = []
    for lane_lbl, key, is_rocm in [("ROCm", rk, 1), ("NV", nk, 0)]:
        cs = m["commits"][key]
        tt = [c["total"] for c in cs]
        rows.append([(lane_lbl, "lane rocm" if is_rocm else "lane"), "<i>commit time to signal</i>",
                     str(len(tt))] + [fmt(pct(tt, p)) for p in ps])
        by = collections.defaultdict(list)
        for c in cs:
            for j in c["jobs"]:
                by[j["type"]].append(j["exec"])
        for t, xs in sorted(by.items()):
            rows.append([(lane_lbl, "lane rocm" if is_rocm else "lane"), t, str(len(xs))]
                        + [fmt(pct(xs, p)) for p in ps])
    P.append("<h4>Percentiles by job type</h4>")
    P.append(table(["Lane", "Job type", "n", "p50", "p90", "p95", "p99"], rows, num_from=2))
    P.append("<p class='note'>p99 of a few hundred jobs rests on a handful of samples; read it with its n.</p>")

    R = {c["sha"]: c for c in m["commits"][rk]}
    N = {c["sha"]: c for c in m["commits"][nk]}
    common = set(R) & set(N)
    both = sorted(s for s in common if R[s]["passed"] and N[s]["passed"])
    if not both:
        return "".join(P)
    ratio = [R[s]["total"] / N[s]["total"] for s in both if N[s]["total"] > 0]
    faster = sum(1 for s in both if R[s]["total"] < N[s]["total"])
    P.append(f"<h4>{repo}: same commits, {len(both)} PR SHAs where both lanes passed</h4>")
    P.append(f"<p class='meta'>{len(common) - len(both)} more SHAs signalled on both lanes but failed on at "
             f"least one; a failure stops early, so those are left out of the comparison.</p>")
    P.append(table(["ROCm / NV time to signal, per PR", "p10", "p50", "p90", "p99", "PRs where ROCm is faster"],
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
    P.append(table(["Outlier set", "PRs", "ROCm TTS p50", "NV TTS p50", "Median ROCm/NV", "Also in the other lane's slowest 10%",
                    "ROCm queue (mean)", "ROCm test execution (mean)", "ROCm teardown (mean)"],
                   [outlier_row("ROCm's slowest 10%", "rocm", slow_r),
                    outlier_row("NV's slowest 10%", "nv", slow_n)]))

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
    P.append(table(["Matched job type (test step)", "Commits", "p50 ROCm / NV", "p90 ROCm / NV", "p99 ROCm / NV",
                    "Median per-commit ratio", "ROCm's slowest 10%: ROCm / NV p50 on the same commits"], rows))

    worst = sorted(both, key=lambda s: -(R[s]["total"] / N[s]["total"] if N[s]["total"] else 0))[:10]
    rows = []
    for s in worst:
        c = R[s]
        pr = c["prs"][0] if c["prs"] else lookup_pr(m, repo, s)
        link = f"<a href='https://github.com/{repo}/pull/{pr}'>#{pr}</a>" if pr else "(fork PR, number not in API)"
        top = sorted(((k, v) for k, v in c["phases"].items()), key=lambda kv: -kv[1])[:2]
        rows.append([f"<code>{s[:10]}</code> {link}", fmt(c["total"]), fmt(N[s]["total"]),
                     f"{c['total'] / N[s]['total']:.1f}x",
                     ", ".join(f"{dict(PHASES)[k].lower()} {fmt(v)}" for k, v in top)])
    P.append(table(["Largest ROCm/NV gap (commit)", "ROCm TTS", "NV TTS", "Ratio", "Where ROCm's time went"],
                   rows, num_from=1))
    return "".join(P)


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
            s = f"<b>{lbl} runs {100 * a['suite'] / b['suite']:.0f}% of the CUDA suite</b> ({a['suite']} vs {b['suite']})"
            if a.get("spt") and b.get("spt"):
                r_ = a["spt"] / b["spt"]
                s += f" and is {r_:.2f}x {'slower' if r_ > 1 else 'faster'} per test" if r_ >= 1 else \
                    f" and is {1 / r_:.2f}x faster per test"
            parts.append(s)
    return f"<p class='note'>{'; '.join(parts)}.</p>" if parts else ""


def compare_note(title, rc, nc, cov, ck, comparable):
    if not (rc and nc):
        return ""
    share = 100 * rc / nc
    a, b = cov.get(ck[0], {}).get("spt"), cov.get(ck[1], {}).get("spt")
    per = ""
    if a and b:
        per = f"; per test ROCm is <b>{a / b:.2f}x slower</b>" if a > b else f"; per test ROCm is <b>{b / a:.2f}x faster</b>"
    if comparable:
        return (f"<p class='note'><b>Comparable suites:</b> {rc} ROCm tests against NV's {nc} ({share:.0f}%), so "
                f"durations are highlighted where ROCm is faster{per}. Queue wait is never highlighted; NV queue by "
                f"runner pool is under NV runner pools.</p>")
    return (f"<p class='note'><b>Not highlighted:</b> {rc} ROCm tests against NV's {nc} ({share:.0f}%), so these "
            f"durations compare a smaller suite to a larger one{per}.</p>")


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
        out.append(f"<b>{lane}:</b> {len(cs)} commits signalled; {len(sup)} of them first had a run that was "
                   f"cancelled or re-triggered ({len(att)} signalled on a re-run attempt), and "
                   f"{m['unsignalled'][key]} commits never produced a {lane} signal because every run was cancelled, "
                   f"almost always superseded by a newer push. "
                   + (f"{len(ap)} fork-PR commits waited for a maintainer to approve CI (median {fmt(pct(ap, 50))}, "
                      f"max {fmt(max(ap))}); their clock starts at approval, since that wait is a human gate "
                      f"the {other} lane shares." if ap else ""))
    out.append("<b>Superseded runs + re-run delay</b> = time from the SHA's first trigger to the start of the run "
               "that finally signalled. <b>Waiting on earlier jobs</b> = <code>needs:</code> jobs ahead of the "
               "lane's last job in the same run. "
               "<b>Runner teardown + completion report</b> = job wall time minus the sum of "
               "its step durations; almost all of it sits after the <code>Complete job</code> step, before GitHub "
               "marks the job done, so the check stays pending while the ephemeral runner shuts down.")
    return "".join(f"<p class='note'>{s}</p>" for s in out)


def ask_section(cov, base):
    s = ["<h2>The ask: more runners</h2>",
         table(["Pool", "What it does today", "What we need", "What it unblocks"], [
             [("DO", "lane rocm"), "Builds and launches every ROCm presubmit job", ("pending (owner: user)", "pending"),
              "Removes the provisioning wait paid on every job, and the multi-GPU starvation"],
             [("RBE", "lane rocm"), "Runs the bazel test actions dispatched from DO", ("pending (owner: user)", "pending"),
              "Shortens test execution, the largest phase of time to signal"],
             [("Remote build", "lane rocm"), "JAX: nothing. <code>rocm_rbe_dynamic</code> sets "
              "<code>--spawn_strategy=local</code>, so compile actions run on the GPU runner",
              ("pending (owner: user)", "pending"),
              "Stops GPU nodes being used as build machines, freeing the same hardware for tests without buying more"],
         ], num_from=9),
         "<p class='note'>Google's CUDA presubmit compiles remotely (<code>--spawn_strategy=remote,worker,standalone,local</code>, "
         "seen in this week's logs); JAX ROCm pins builds to the GPU box, while <code>openxla/xla</code> ROCm already "
         "uses <code>--spawn_strategy=dynamic</code>.</p>"]
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
            s.append(f"<p><b>{esc(R['repo'])}:</b> " + ", and ".join(parts) + ".</p>")
    s.append("<p><b>Gating coverage:</b> with more hardware we can pitch Google to gate PRs on far more ROCm "
             "tests than the single target we gate on today.</p>")
    return "".join(s)


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
        rows.append([f"{name} queue p50 / p90 (<code>{esc(old_lab)}</code> then, <code>{esc(lab)}</code> now)",
                     f"{p50} / {p90}", f"{fmt(pct(xs, 50))} / {fmt(pct(xs, 90))}"])
    return (f"<h2>Changes since the {esc(base['label'])}</h2>"
            + table(["Item", esc(base["label"]), "This week"], rows)
            + (f"<p class='note'>{base['note']}</p>" if base.get("note") else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--html", required=True)
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--md", help="also write the same report as Markdown (defaults to --html with a .md suffix)")
    ap.add_argument("--no-md", action="store_true", help="write only the HTML")
    ap.add_argument("--baseline", help="JSON with an earlier report's figures; enables 'Changes since'")
    a = ap.parse_args()
    m = build(a.data)
    logs = load_logs(a.data)
    since = dt.datetime.fromisoformat(a.since)
    until = dt.datetime.fromisoformat(a.until)
    base = json.load(open(a.baseline)) if a.baseline else None
    doc = render(m, logs, since, until, base)
    with open(a.html, "w") as f:
        f.write(doc)
    print(f"wrote {a.html}")
    if not a.no_md:
        md_path = a.md or re.sub(r"\.html?$", "", a.html) + ".md"
        with open(md_path, "w") as f:
            f.write(to_markdown(doc))
        print(f"wrote {md_path}")
    for k, cs in m["commits"].items():
        print(k, "commits", len(cs), "unsignalled", m["unsignalled"][k])


if __name__ == "__main__":
    main()
