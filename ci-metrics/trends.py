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

from metrics_lib import ALL_LANES, CDT, PHASES
from report import build, pct

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


def series(pooled, spans):
    """The per-day series for every lane, over one shared day axis."""
    by_lane_day = {lane: collections.defaultdict(list) for lane in LANES}
    for lane in LANES:
        for c in pooled[lane].values():
            by_lane_day[lane][day_of(c)].append(c)

    days = sorted({d for lane in LANES for d in by_lane_day[lane]})

    metrics = ["n", "ttl_p50", "ttl_p90", "queue_p50", "queue_p90", "pass_rate"]
    metrics += [f"phase_{key}" for key, _, _ in PHASE_GROUPS]

    out = {}
    for lane in LANES:
        rows = {d: summarise(cs) for d, cs in by_lane_day[lane].items() if cs}
        # A day with no commits in this lane is a hole, not a zero.
        out[lane] = {m: [rows[d][m] if d in rows else None for d in days] for m in metrics}
    return ([d.isoformat() for d in days],
            [is_complete(d, spans) for d in days],
            out)


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
    days, complete, lanes = series(pooled, spans)
    if not days:
        raise SystemExit("no commits found in any snapshot")

    doc = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timezone": "CDT (UTC-5), as in the Markdown report",
        "snapshots": len(names),
        "first_snapshot": names[0],
        "last_snapshot": names[-1],
        "unique_commits": {lane: len(pooled[lane]) for lane in LANES},
        "lanes": [{"key": lane, "label": ALL_LANES[lane][3]} for lane in LANES],
        "phases": [{"key": key, "label": label} for key, label, _ in PHASE_GROUPS],
        "days": days,
        # False where the snapshots do not cover the whole day, so the page can
        # mark it provisional rather than let a short day read as an improvement.
        "complete": complete,
        "series": lanes,
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
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
