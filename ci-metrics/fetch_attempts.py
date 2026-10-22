#!/usr/bin/env python3
"""Fetch earlier attempts of re-run workflow runs.

collect.py stores only the latest attempt. A failed first attempt is already a
signal, so without it a re-run days later would count as time to signal.

  python3 fetch_attempts.py --data DATA_DIR
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from collect import get, slim_run
from metrics_lib import load


def fetch(rec, n):
    repo, rid = rec["repo"], rec["run"]["id"]
    run = get(f"/repos/{repo}/actions/runs/{rid}/attempts/{n}")
    jobs, page = [], 1
    while True:
        d = get(f"/repos/{repo}/actions/runs/{rid}/attempts/{n}/jobs?per_page=100&page={page}")
        jobs += d["jobs"]
        if len(jobs) >= d["total_count"] or not d["jobs"]:
            break
        page += 1
    keep = ["id", "name", "labels", "runner_name", "runner_group_name", "status", "conclusion",
            "created_at", "started_at", "completed_at", "run_attempt"]
    return {"repo": repo, "workflow": rec["workflow"], "lane": rec["lane"], "run": slim_run(run),
            "jobs": [{**{k: j.get(k) for k in keep},
                      "steps": [{k: s.get(k) for k in ("number", "name", "conclusion", "started_at", "completed_at")}
                                for s in j.get("steps") or []]} for j in jobs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    path = os.path.join(a.data, "attempts.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(json.loads(l)["run"]["id"], json.loads(l)["run"]["run_attempt"]) for l in open(path)}
    todo = [(rec, n) for rec in load(a.data) if rec["run"]["event"] == "pull_request"
            for n in range(1, rec["run"]["run_attempt"]) if (rec["run"]["id"], n) not in done]
    print(f"fetching {len(todo)} earlier attempts", flush=True)
    with open(path, "a") as out, ThreadPoolExecutor(8) as ex:
        for rec in ex.map(lambda t: fetch(*t), todo):
            out.write(json.dumps(rec) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
