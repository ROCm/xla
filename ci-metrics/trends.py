#!/usr/bin/env python3
"""Roll the nightly snapshots up into one day-by-day trend series.

`report.py` renders a single 24h window; this renders the change between them.
It reuses `report.build()` unchanged, so a metric means here exactly what it
means in the Markdown report -- nothing is redefined.

  python3 trends.py --data <ci-metrics-data checkout>/data --out trends.json

The nightly publishes one `data/<stamp>/` snapshot per run, each covering the
24h before it. Those windows overlap, and the overlap is uneven: six snapshots
landed on 2026-10-29 (94-99% the same commits) against one on 2026-10-26. One
point per snapshot would therefore plot the collection cadence rather than CI
behaviour, so every snapshot is pooled and deduplicated by (lane, head sha) and
each commit is then bucketed by its own arrival time. The result depends on
when CI ran, not on when the collector did.

Test counts come from the same snapshots but not from `build()`: the jobs API
carries none, so they are read from the sampled logs `sample_logs.py` wrote,
pooled and deduplicated by job id the same way. That sample is capped per
category, so the series is the suite size of the day's largest sampled job --
what one test job runs when nothing truncates it -- and never a lane-wide
total.

A day is only comparable to its neighbours if the snapshots actually cover all
of it. Each snapshot covers the 24h before its stamp; the union of those
windows is the covered span, and any day not wholly inside it -- the first, the
most recent, and anything either side of a night the nightly missed -- is kept
but flagged `complete: false` so the page can draw it as provisional instead of
as a real dip.
"""
import argparse
import collections
import datetime as dt
import json
import os

from metrics_lib import ALL_LANES, CDT, PHASES, REPO_BY_SHORT, ts
from report import build, load_logs, pct

# The lanes the dashboard plots, in a fixed order. The order is also the
# categorical colour order on the page, so a lane keeps its hue no matter how
# many lanes a filter leaves on screen.
LANES = ("xla_rocm", "xla_nv", "jax_rocm", "jax_nv")

# PHASES has nine entries, which is past the eight-series ceiling a categorical
# stack can stay readable at, and three of them are rounding error (`setup`
# 1.3s, `close` 3.2s, `superseded` 0.0s mean on a representative snapshot).
# Folding to seven keeps every second accounted for -- the groups still cover
# PHASES exactly -- while leaving the stack legible.
PHASE_GROUPS = (
    ("superseded", "Superseded + re-run delay", ("superseded",)),
    ("upstream", "Waiting on earlier jobs", ("upstream",)),
    ("queue", "Queue for a runner", ("queue",)),
    ("start", "Runner start + Docker", ("setup", "docker")),
    ("env", "Checkout, env, wheels", ("env",)),
    ("test", "Test execution", ("test",)),
    ("teardown", "Uploads + teardown", ("close", "gaps")),
)

assert {p for _, _, group in PHASE_GROUPS for p in group} == {k for k, _ in PHASES}, \
    "PHASE_GROUPS no longer covers PHASES; a phase would be silently dropped"


def test_cats():
    """The log-sample categories whose suite sizes the dashboard plots.

    One per lane, except where the repo splits into job families -- openxla/xla
    runs its own tests and the JAX integration tests in one workflow, and they
    are separate suites, so they are separate series. These are the keys
    `log_category()` writes into `logs.jsonl`, in lane order.
    """
    out = []
    for lane in LANES:
        R = REPO_BY_SHORT[lane.rsplit("_", 1)[0]]
        if R["families"]:
            out += [{"key": f"{lane}:{fam}", "lane": lane, "label": label}
                    for label, fam in R["families"]]
        else:
            out.append({"key": lane, "lane": lane, "label": "Tests"})
    return out


TEST_CATS = test_cats()


def snapshots(data_dir, window_days=None):
    """The snapshot directories to read, oldest first.

    Names are `<stamp>` as the nightly writes them (`20261103T034845Z`), so a
    lexical sort is chronological. `window_days` trims the read to recent
    snapshots, which is what lets a caller sparse-checkout only those.
    """
    names = sorted(n for n in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, n)))
    if window_days is not None:
        cut = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)
        keep = []
        for n in names:
            try:
                stamp = dt.datetime.strptime(n, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
            except ValueError:
                continue  # not a snapshot directory; leave it alone
            if stamp >= cut:
                keep.append(n)
        names = keep
    return names


def pool(data_dir, names):
    """Every commit each snapshot saw, deduplicated by (lane, head sha).

    The first snapshot to report a commit wins. Snapshots are read oldest
    first, and the earliest window containing a commit is the one that saw it
    closest to when it finished, so later re-observations add nothing.
    """
    seen = {lane: {} for lane in LANES}
    for name in names:
        m = build(os.path.join(data_dir, name))
        for lane in LANES:
            for c in m["commits"][lane]:
                seen[lane].setdefault(c["sha"], c)
    return seen


def pr_index(data_dir, names):
    """Every `<repo>@<sha> -> PR number` the snapshots resolved, merged.

    A run triggered by a branch PR names its PR; a fork PR's run does not, and
    `report.lookup_pr()` asks the API and caches the answer in the snapshot's
    `pr_lookup.json`. Reading those caches gets the fork PRs here without a
    token and without a request, which matters because the Pages job that runs
    this has neither.
    """
    out = {}
    for name in names:
        path = os.path.join(data_dir, name, "pr_lookup.json")
        if os.path.exists(path):
            with open(path) as f:
                out.update({k: v for k, v in json.load(f).items() if v})
    return out


def test_samples(data_dir, names):
    """Tests per test job, per category per day, from the sampled logs.

    `sample_logs.py` records every bazel `Executed X out of Y tests` line of a
    sampled job, so one job's suite is the sum of `Y` over its invocations --
    the same quantity `report.coverage()` reports as the suite size, taken per
    job rather than per 48h window. Cached tests count: the suite is what the
    lane covers, not what it had to re-run.

    Samples are pooled across snapshots and deduplicated by job id, exactly as
    commits are, and bucketed by the day the job finished. A sampled job, not
    every job: the counts here are a sample of each day, never a lane-wide
    total.
    """
    seen = {}
    for name in names:
        for l in load_logs(os.path.join(data_dir, name)):
            if l["executed"]:
                seen.setdefault(l["job_id"], l)
    by = {c["key"]: collections.defaultdict(list) for c in TEST_CATS}
    for l in seen.values():
        if l["cat"] in by:
            by[l["cat"]][ts(l["completed_at"]).astimezone(CDT).date()].append(
                sum(total for _, total in l["executed"]))
    return by


def day_of(commit):
    """The calendar day a commit belongs to, in the report's own timezone."""
    return commit["t0"].astimezone(CDT).date()


# Each snapshot collects the 24h before its stamp; see the collection window in
# rocm_xla_jax_metrics_nightly.yml.
SNAPSHOT_HOURS = 24

# Consecutive nightlies drift by a few minutes, so their 24h windows can fall
# minutes short of meeting (observed: 0.2-3.2 min). That is schedule jitter, not
# missing data, and must not flag the day around it as partial. A nightly that
# genuinely did not run leaves a hole of about 24h, far outside this tolerance.
GAP_TOLERANCE = dt.timedelta(hours=1)


def covered_spans(names):
    """The snapshot windows, merged into non-overlapping [start, end) spans."""
    spans = []
    for n in names:
        end = dt.datetime.strptime(n, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
        spans.append((end - dt.timedelta(hours=SNAPSHOT_HOURS), end))
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= GAP_TOLERANCE:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def is_complete(day, spans):
    """Whether a whole calendar day (in CDT) sits inside the covered span."""
    start = dt.datetime.combine(day, dt.time.min, tzinfo=CDT)
    end = start + dt.timedelta(days=1)
    return any(s <= start and end <= e for s, e in spans)


def summarise(commits):
    """One day's figures for one lane."""
    totals = [c["total"] for c in commits]
    queues = [c["phases"]["queue"] for c in commits]
    out = {
        "n": len(commits),
        "ttl_p50": pct(totals, 50),
        "ttl_p90": pct(totals, 90),
        "queue_p50": pct(queues, 50),
        "queue_p90": pct(queues, 90),
        "pass_rate": sum(1 for c in commits if c["passed"]) / len(commits),
    }
    # Means, not medians, for the stacked breakdown: the medians of the parts
    # do not sum to the median of the whole, so a median stack would draw a
    # total that no commit ever took.
    for key, _, group in PHASE_GROUPS:
        out[f"phase_{key}"] = sum(sum(c["phases"][p] for p in group) for c in commits) / len(commits)
    return out


def by_day(pooled):
    """Every lane's commits, grouped by the day they arrived."""
    out = {lane: collections.defaultdict(list) for lane in LANES}
    for lane in LANES:
        for c in pooled[lane].values():
            out[lane][day_of(c)].append(c)
    return out


def series(commit_days, days):
    """The per-day series for every lane, over one shared day axis."""
    metrics = ["n", "ttl_p50", "ttl_p90", "queue_p50", "queue_p90", "pass_rate"]
    metrics += [f"phase_{key}" for key, _, _ in PHASE_GROUPS]

    out = {}
    for lane in LANES:
        rows = {d: summarise(cs) for d, cs in commit_days[lane].items() if cs}
        # A day with no commits in this lane is a hole, not a zero.
        out[lane] = {m: [rows[d][m] if d in rows else None for d in days] for m in metrics}
    return out


# The tail the dashboard names. `TAIL_PCT` has to be the percentile the page
# plots as the tail, or the table would explain a number that is not on screen.
TAIL_PCT = 90
# Three is what a day's row can show without the table becoming the page. The
# count of everything at or above the cut travels with them, so a day where the
# tail is deeper than three is not mistaken for one where it is not.
TAIL_SHOW = 3


def dominant(phases):
    """The phase group that took the most of one commit's time to signal.

    Groups, not raw phases, so the word here is the word the stacked chart and
    its legend already use for that colour.
    """
    key, _, group = max(PHASE_GROUPS, key=lambda g: sum(phases[p] for p in g[2]))
    return key, sum(phases[p] for p in group)


def counterpart(lane):
    """The other side of the same repo's presubmit: ROCm <-> NV, or None."""
    short, side = lane.rsplit("_", 1)
    other = f"{short}_{'nv' if side == 'rocm' else 'rocm'}"
    return other if other in LANES else None


def tail_series(commit_days, days, prs):
    """The commits that put each lane's p90 where it is, day by day.

    A percentile is a position in a sorted list, so the commits at or above it
    are what it is made of: move them and the p90 moves. Each day reports the
    cut, how many commits sat at or above it, and the slowest `TAIL_SHOW` of
    them by name -- the PR, the head sha, and the phase that took the most of
    the wait, which is usually the whole explanation.

    A PR can appear twice in a day: two of its head commits each got a signal,
    and each is its own entry in the lane. That repetition is the finding, not
    a duplicate.

    Both lanes of a repo run on the same head commits, so each commit also
    carries where it sits in the other lane that day, which is the difference
    between a slow PR and a slow lane:

      `both`    in the other lane's tail too -- the commit is heavy everywhere,
                and nothing about this lane put it here.
      `only`    the other lane signalled this commit and did *not* put it in
                its tail -- the lanes disagree about it, which is the row worth
                chasing.
      `None`    the other lane never reached a signal on it, so there is
                nothing to compare against.

    `shared` counts the whole tail that way, not just the named three, so a day
    can be read as "ours alone" or "the same PRs as theirs" at a glance.
    """
    cuts, tails, signalled = {}, {}, {}
    for lane in LANES:
        for d in days:
            cs = commit_days[lane].get(d)
            if not cs:
                continue
            cut = pct([c["total"] for c in cs], TAIL_PCT)
            cuts[lane, d] = cut
            tails[lane, d] = sorted((c for c in cs if c["total"] >= cut),
                                    key=lambda c: -c["total"])
            signalled[lane, d] = {c["sha"] for c in cs}

    out = {}
    for lane in LANES:
        repo = REPO_BY_SHORT[lane.rsplit("_", 1)[0]]["repo"]
        other = counterpart(lane)
        rows = []
        for d in days:
            tail = tails.get((lane, d))
            if not tail:
                rows.append(None)
                continue
            their_tail = {c["sha"] for c in tails.get((other, d), ())}
            their_commits = signalled.get((other, d), set())

            def against(sha):
                if sha in their_tail:
                    return "both"
                return "only" if sha in their_commits else None

            rows.append({
                "cut": cuts[lane, d],
                "n": len(tail),
                "other": other,
                "shared": sum(1 for c in tail if c["sha"] in their_tail),
                "commits": [{
                    "also": against(c["sha"]),
                    "pr": c["prs"][0] if c["prs"] else prs.get(f"{repo}@{c['sha']}"),
                    # Half the tail is fork or export PRs, whose runs name no
                    # PR; the run does, on its own page, and costs no request.
                    "run": c["run"],
                    "branch": c["branch"],
                    "sha": c["sha"][:10],
                    "total": c["total"],
                    "phase": dominant(c["phases"])[0],
                    "phase_s": dominant(c["phases"])[1],
                    "passed": c["passed"],
                    # A re-run starts the clock again on the same commit, so it
                    # is a common reason for one to land in the tail.
                    "rerun": c["rerun"],
                } for c in tail[:TAIL_SHOW]],
            })
        out[lane] = rows
    return out


def test_series(samples, days):
    """The per-day suite size for every test category, over the same day axis.

    The headline is the day's **largest** sampled job, not a central statistic:
    a job that fails early reports what it got through (`Executed 1 out of 1
    tests` is a common value in the sample), and the XLA lanes run a second
    bazel invocation that not every job reaches, so a median tracks the day's
    mix of job outcomes rather than the suite. The largest is the suite with
    nothing truncating it, which is the figure `report.coverage()` publishes:
    on 2026-11-11 the report says 576 (419 + 157) for the ROCm XLA jobs, where
    the max is 576 and the median 549.

    It is still a maximum over a *sample*, so a thin day can under-report the
    suite; `n` travels with it, and the page draws those days hollow. `p50` is
    kept for the table, where the gap between the two is the truncation rate.

    Never a sum: the sample is capped per category per snapshot, so a total
    would plot the sampling cap rather than the lane.
    """
    return {cat["key"]: {
        "tests_max": [max(samples[cat["key"]][d]) if samples[cat["key"]].get(d) else None
                      for d in days],
        "tests_p50": [pct(samples[cat["key"]][d], 50) if samples[cat["key"]].get(d) else None
                      for d in days],
        "n": [len(samples[cat["key"]][d]) if samples[cat["key"]].get(d) else None
              for d in days],
    } for cat in TEST_CATS}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the `data/` directory of a ci-metrics-data checkout")
    ap.add_argument("--out", required=True, help="where to write trends.json")
    ap.add_argument("--window-days", type=int, default=None,
                    help="only read snapshots from the last N days (default: all)")
    args = ap.parse_args()

    names = snapshots(args.data, args.window_days)
    if not names:
        raise SystemExit(f"no snapshot directories under {args.data}")
    pooled = pool(args.data, names)
    spans = covered_spans(names)
    commit_days = by_day(pooled)
    samples = test_samples(args.data, names)
    # One day axis for both tabs. A sampled job whose day carries no commit
    # (the edge of the window) still gets a column rather than being dropped.
    days = sorted({d for lane in LANES for d in commit_days[lane]}
                  | {d for cat in samples.values() for d in cat})
    if not days:
        raise SystemExit("no commits found in any snapshot")
    complete = [is_complete(d, spans) for d in days]
    lanes = series(commit_days, days)
    tests = test_series(samples, days)
    tail = tail_series(commit_days, days, pr_index(args.data, names))
    days = [d.isoformat() for d in days]

    doc = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timezone": "CDT (UTC-5), as in the Markdown report",
        "snapshots": len(names),
        "first_snapshot": names[0],
        "last_snapshot": names[-1],
        "unique_commits": {lane: len(pooled[lane]) for lane in LANES},
        # `repo` is here so the page can link a PR number to its pull request.
        "lanes": [{"key": lane, "label": ALL_LANES[lane][3], "repo": ALL_LANES[lane][0]}
                  for lane in LANES],
        "phases": [{"key": key, "label": label} for key, label, _ in PHASE_GROUPS],
        # The test suites, one series per lane or per job family within a lane.
        "test_cats": TEST_CATS,
        "days": days,
        # False where the snapshots do not cover the whole day, so the page can
        # mark it provisional rather than let a short day read as an improvement.
        "complete": complete,
        "series": lanes,
        # Suite size per sampled test job, on the same day axis as `series`.
        "tests": tests,
        # The commits the p90 of `ttl_p90` is made of, same day axis again.
        "tail_pct": TAIL_PCT,
        "tail": tail,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    partial = [d for d, ok in zip(days, complete) if not ok]
    print(f"{len(names)} snapshots -> {len(days)} days ({days[0]} .. {days[-1]})")
    if partial:
        print(f"  partial (flagged, not dropped): {', '.join(partial)}")
    for lane in LANES:
        print(f"  {lane:9s} {len(pooled[lane]):5d} unique commits")
    for cat in TEST_CATS:
        got = [v for v in tests[cat["key"]]["tests_max"] if v is not None]
        n = sum(x for x in tests[cat["key"]]["n"] if x)
        print(f"  {cat['key']:13s} {n:5d} sampled jobs over {len(got)} days"
              + (f", {got[0]:.0f} -> {got[-1]:.0f} tests" if got else ", no suite sizes"))
    named = sum(1 for lane in LANES for r in tail[lane] if r for c in r["commits"] if c["pr"])
    total = sum(len(r["commits"]) for lane in LANES for r in tail[lane] if r)
    print(f"  p{TAIL_PCT} tail: {total} commits over {len(days)} days, {named} with a PR "
          f"number, {total - named} shown by run (fork and export PRs name none)")
    seen = collections.Counter(c["also"] for lane in LANES for r in tail[lane] if r
                               for c in r["commits"])
    print(f"    of those, {seen['only']} are in this lane's tail alone, {seen['both']} in both "
          f"lanes', {seen[None]} have no signal on the other lane")
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
