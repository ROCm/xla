#!/usr/bin/env python3
"""Fetch GitHub Actions runs + jobs (with step timings) for the CI metrics report.

Stdlib only. Token from $GH_TOKEN or `gh auth token`. Resumable: runs already in
jobs.jsonl are skipped, so re-running after a rate-limit stop just continues.

  python3 collect.py --since 2026-09-22 --until 2026-09-29 --out DATA_DIR
"""
import argparse
import datetime as dt
import http.client
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from metrics_lib import REPOS


def _workflows():
    """(repo, workflow file, lane) from REPOS, so collection can't drift from
    what the report expects. ROCm and the nightlies first, so a rate-limit stall
    hits the NV reference lane rather than the lanes the report is about."""
    out = [(R["repo"], R["rocm"][0], "rocm") for R in REPOS]
    out += [(R["repo"], wf, "rocm") for R in REPOS for wf in R["nightly"]]
    out += [(R["repo"], R["nv"][0], "nv") for R in REPOS]
    return out


WORKFLOWS = _workflows()
# Continuous runs whose trigger check skips everything finish in seconds and
# carry no ROCm jobs; do not spend an API call on them.
SKIP_SHORT_RUNS = {"wheel_tests_continuous.yml": 180}

API = "https://api.github.com"
_rl_lock = threading.Lock()
_rl = {"remaining": 5000, "reset": 0}


def token():
    t = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if t:
        return t
    return subprocess.check_output(["gh", "auth", "token"], text=True).strip()


TOKEN = token()


def get(path):
    url = path if path.startswith("http") else API + path
    for attempt in range(6):
        with _rl_lock:
            if _rl["remaining"] < 30 and _rl["reset"] > time.time():
                wait = _rl["reset"] - time.time() + 5
                print(f"[rate] {_rl['remaining']} left, sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
                _rl["remaining"] = 5000
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                with _rl_lock:
                    _rl["remaining"] = int(r.headers.get("X-RateLimit-Remaining", 5000))
                    _rl["reset"] = int(r.headers.get("X-RateLimit-Reset", 0))
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset = int(e.headers.get("X-RateLimit-Reset", time.time() + 60))
                with _rl_lock:
                    _rl["remaining"] = 0
                    _rl["reset"] = reset
                continue
            if e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead, ConnectionError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}")


def list_runs(repo, wf, since, until):
    runs = []
    day = since
    while day < until:
        nxt = min(day + dt.timedelta(hours=12), until)
        rng = f"{day:%Y-%m-%dT%H:%M:%SZ}..{nxt:%Y-%m-%dT%H:%M:%SZ}"
        page = 1
        while True:
            d = get(f"/repos/{repo}/actions/workflows/{wf}/runs?created={rng}&per_page=100&page={page}")
            runs += d["workflow_runs"]
            if len(d["workflow_runs"]) < 100:
                break
            page += 1
        day = nxt
    return runs


RUN_KEYS = ["id", "name", "path", "event", "head_sha", "head_branch", "run_attempt",
            "created_at", "run_started_at", "updated_at", "status", "conclusion"]
JOB_KEYS = ["id", "name", "labels", "runner_name", "runner_group_name", "status",
            "conclusion", "created_at", "started_at", "completed_at", "run_attempt"]


def slim_run(r):
    d = {k: r.get(k) for k in RUN_KEYS}
    d["prs"] = [p["number"] for p in r.get("pull_requests") or []]
    return d


def fetch_jobs(repo, run_id):
    jobs, page = [], 1
    while True:
        d = get(f"/repos/{repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100&page={page}")
        jobs += d["jobs"]
        if len(jobs) >= d["total_count"] or not d["jobs"]:
            break
        page += 1
    out = []
    for j in jobs:
        s = {k: j.get(k) for k in JOB_KEYS}
        s["steps"] = [{k: st.get(k) for k in ("number", "name", "conclusion", "started_at", "completed_at")}
                      for st in j.get("steps") or []]
        out.append(s)
    return out


def iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--workflows", help="comma-separated workflow files; default: all in WORKFLOWS")
    a = ap.parse_args()
    wanted = set(a.workflows.split(",")) if a.workflows else None
    since = dt.datetime.fromisoformat(a.since).replace(tzinfo=dt.timezone.utc)
    until = dt.datetime.fromisoformat(a.until).replace(tzinfo=dt.timezone.utc)
    os.makedirs(a.out, exist_ok=True)
    jobs_path = os.path.join(a.out, "jobs.jsonl")
    done = set()
    if os.path.exists(jobs_path):
        with open(jobs_path) as f:
            for line in f:
                done.add(json.loads(line)["run"]["id"])
    print(f"resume: {len(done)} runs already fetched", flush=True)
    wlock = threading.Lock()
    with open(jobs_path, "a") as out:
        for repo, wf, lane in WORKFLOWS:
            if wanted and wf not in wanted:
                continue
            cache = os.path.join(a.out, f"runs--{repo.replace('/', '--')}--{wf}.json")
            if os.path.exists(cache):
                runs = json.load(open(cache))
            else:
                runs = list_runs(repo, wf, since, until)
                with open(cache, "w") as f:
                    json.dump(runs, f)
            todo = []
            for r in runs:
                if r["id"] in done:
                    continue
                lim = SKIP_SHORT_RUNS.get(wf)
                if lim and r.get("run_started_at") and r.get("updated_at") and \
                        (iso(r["updated_at"]) - iso(r["run_started_at"])).total_seconds() < lim:
                    continue
                todo.append(r)
            print(f"{repo} {wf}: {len(runs)} runs, fetching jobs for {len(todo)}", flush=True)

            def work(r):
                return r, fetch_jobs(repo, r["id"])

            n = 0
            with ThreadPoolExecutor(a.workers) as ex:
                for fut in as_completed([ex.submit(work, r) for r in todo]):
                    r, jobs = fut.result()
                    rec = {"repo": repo, "workflow": wf, "lane": lane, "run": slim_run(r), "jobs": jobs}
                    with wlock:
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                    n += 1
                    if n % 200 == 0:
                        print(f"  {n}/{len(todo)}  rate_remaining={_rl['remaining']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
