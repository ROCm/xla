"""Lane definitions, step phases and helpers shared by the ci-metrics scripts.

Everything that encodes a CI naming convention lives here, so a renamed job or
step is a one-line fix.
"""
import datetime as dt
import json
import os
import re

CDT = dt.timezone(dt.timedelta(hours=-5), "CDT")


def load(data_dir):
    with open(os.path.join(data_dir, "jobs.jsonl")) as f:
        for line in f:
            yield json.loads(line)


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def is_do(job):
    return any(l.startswith("amd-do-") for l in job["labels"])


def carried_over(job):
    """A finished job that "re-run failed jobs" copied into the new attempt.

    The copy gets a new id and created_at but keeps the original start and end,
    so it duplicates the earlier attempt's job and its queue time is negative.
    """
    a, b = ts(job["created_at"]), ts(job["started_at"])
    return bool(a and b and b < a)


# ---- lanes -----------------------------------------------------------------
# A lane is the set of jobs whose completion is "the signal" for one commit.

PRESUBMIT_LANES = {
    # key: (repo, workflow, job predicate, label)
    "jax_rocm": ("jax-ml/jax", "bazel_rocm_presubmit.yml", lambda j: is_do(j), "JAX ROCm"),
    "jax_nv": ("jax-ml/jax", "bazel_cuda_presubmit.yml", lambda j: True, "JAX NV/CUDA"),
    "xla_rocm": ("openxla/xla", "rocm_ci.yml", lambda j: is_do(j), "OpenXLA ROCm"),
    "xla_nv": ("openxla/xla", "ci.yml",
               lambda j: re.search(r"\bGPU\b", j["name"]) and not re.search(r"ROCm|ONEAPI|AMD", j["name"], re.I),
               "OpenXLA NV/CUDA"),
}

def xla_family(name):
    """openxla/xla runs its own tests and the JAX integration tests in one workflow."""
    return "jax" if name.startswith("JAX") else "xla"


def _split_lane(key, fam):
    repo, wf, pred, label = PRESUBMIT_LANES[key]
    return (repo, wf, lambda j, p=pred, f=fam: p(j) and xla_family(j["name"]) == f,
            f"{label} ({fam.upper()} jobs)")


# Sub-lanes of the two openxla/xla lanes: the signal each job family would give
# on its own. Reported alongside the full lanes, never instead of them; the full
# lane's time to signal is the max of its two sub-lanes, not their sum.
SPLIT_LANES = {f"{k}:{f}": _split_lane(k, f) for k in ("xla_rocm", "xla_nv") for f in ("xla", "jax")}

NIGHTLY_WORKFLOWS = ("wheel_tests_continuous.yml", "wheel_tests_nightly_release.yml")


def nightly_kind(rec, job):
    if rec["workflow"] not in NIGHTLY_WORKFLOWS or not is_do(job):
        return None
    n = job["name"]
    if n.startswith("Pytest ROCm"):
        return "pytest"
    if n.startswith("Bazel ROCm"):
        return "bazel"
    return None


def log_category(rec, job):
    """Job types whose logs are sampled for test counts and critical path."""
    if rec["run"]["event"] != "pull_request":
        return None
    wf, n = rec["workflow"], job["name"]
    if wf == "bazel_rocm_presubmit.yml" and is_do(job) and n.startswith("Test /"):
        return "jax_rocm"
    if wf == "bazel_cuda_presubmit.yml" and n.startswith("Test /"):
        return "jax_nv"
    if wf == "rocm_ci.yml" and is_do(job):
        return "xla_rocm:" + xla_family(n)
    if wf == "ci.yml" and PRESUBMIT_LANES["xla_nv"][2](job):
        return "xla_nv:" + xla_family(n)
    return None


# ---- step phases -----------------------------------------------------------
# Every second of a job belongs to exactly one phase; the order is the order
# they appear in the report.

PHASES = [
    ("superseded", "Superseded runs + re-run delay"),
    ("upstream", "Waiting on earlier jobs"),
    ("queue", "Queue for a runner"),
    ("setup", "Runner set-up"),
    ("docker", "Docker pull + container start"),
    ("env", "Checkout, env, wheels"),
    ("test", "Test execution"),
    ("close", "Uploads, docker stop, job close"),
    ("gaps", "Runner teardown + completion report"),
]

TEST_STEP_RE = re.compile(
    r"^(Bazel (ROCm|CUDA) tests|Test XLA|Run JAX Unit Tests|Run build\.py|Run (py)?tests?|Pytest|Run pytest|Bazel test)",
    re.I)
CLOSE_STEP_RE = re.compile(r"^(Post |Stop containers|Complete job|Upload)", re.I)


def step_phase(name):
    if name == "Set up job":
        return "setup"
    if name == "Initialize containers":
        return "docker"
    if TEST_STEP_RE.search(name):
        return "test"
    if CLOSE_STEP_RE.search(name):
        return "close"
    return "env"


def job_phases(job):
    """Split one job's on-runner time into phases. Sums to completed - started."""
    out = {k: 0.0 for k in ("setup", "docker", "env", "test", "close")}
    for s in job["steps"]:
        a, b = ts(s["started_at"]), ts(s["completed_at"])
        if a and b and b > a:
            out[step_phase(s["name"])] += (b - a).total_seconds()
    wall = (ts(job["completed_at"]) - ts(job["started_at"])).total_seconds()
    out["gaps"] = wall - sum(out.values())
    return out


def test_step_seconds(job):
    tot = 0.0
    for s in job["steps"]:
        a, b = ts(s["started_at"]), ts(s["completed_at"])
        if a and b and TEST_STEP_RE.search(s["name"]):
            tot += (b - a).total_seconds()
    return tot
