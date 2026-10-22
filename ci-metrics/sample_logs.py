#!/usr/bin/env python3
"""Sample job logs and extract bazel test counts, elapsed time and critical path.

The jobs API carries no test counts and no critical path, so both come from the
log. Per job it records every bazel invocation's
  "Executed X out of Y tests" and "Elapsed time: Ns, Critical Path: Ms".

  python3 sample_logs.py --data DATA_DIR [--per-cat 60 --recent 30]
"""
import argparse
import datetime as dt
import json
import os
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from metrics_lib import load, log_category

EXEC_RE = re.compile(r"Executed (\d+) out of (\d+) tests?")
ELAPSED_RE = re.compile(r"Elapsed time: ([0-9.]+)s, Critical Path: ([0-9.]+)s")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def fetch(repo, job_id):
    p = subprocess.run(["gh", "api", "--allow-escape-sequences", f"repos/{repo}/actions/jobs/{job_id}/logs"],
                       capture_output=True, text=True, errors="replace")
    if p.returncode:
        return None
    text = ANSI_RE.sub("", p.stdout)
    execs = [(int(a), int(b)) for a, b in EXEC_RE.findall(text)]
    elaps = [(float(a), float(b)) for a, b in ELAPSED_RE.findall(text)]
    return {"executed": execs, "elapsed": elaps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--per-cat", type=int, default=60)
    ap.add_argument("--recent", type=int, default=30)
    ap.add_argument("--recent-hours", type=int, default=48)
    a = ap.parse_args()
    out_path = os.path.join(a.data, "logs.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["job_id"] for l in open(out_path)}

    cands = {}
    end = None
    for rec in load(a.data):
        for j in rec["jobs"]:
            cat = log_category(rec, j)
            if not cat or j["conclusion"] not in ("success", "failure") or not j["completed_at"]:
                continue
            cands.setdefault(cat, []).append((rec, j))
            end = max(end or j["completed_at"], j["completed_at"])
    cutoff = (dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
              - dt.timedelta(hours=a.recent_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rng = random.Random(0)
    picks = []
    for cat, lst in sorted(cands.items()):
        chosen = {j["id"]: (rec, j) for rec, j in rng.sample(lst, min(a.per_cat, len(lst)))}
        recent = [x for x in lst if x[1]["completed_at"] >= cutoff and x[1]["id"] not in chosen]
        for rec, j in rng.sample(recent, min(a.recent, len(recent))):
            chosen[j["id"]] = (rec, j)
        print(f"{cat}: {len(lst)} candidates, sampling {len(chosen)}", flush=True)
        picks += [(cat, rec, j) for rec, j in chosen.values() if j["id"] not in done]

    def work(item):
        cat, rec, j = item
        return cat, rec, j, fetch(rec["repo"], j["id"])

    with open(out_path, "a") as out, ThreadPoolExecutor(8) as ex:
        for n, (cat, rec, j, res) in enumerate(ex.map(work, picks), 1):
            if res is None:
                continue
            out.write(json.dumps({"cat": cat, "repo": rec["repo"], "run_id": rec["run"]["id"],
                                  "job_id": j["id"], "job": j["name"], "completed_at": j["completed_at"],
                                  **res}) + "\n")
            if n % 50 == 0:
                print(f"  {n}/{len(picks)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
