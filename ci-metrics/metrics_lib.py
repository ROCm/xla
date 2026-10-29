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


# ---- repos -----------------------------------------------------------------
# One entry per upstream repo the report covers. Everything repo-specific lives
# here: lanes, display names, job families, runner pools, log sampling. Adding a
# repo is an entry in REPOS, not a change to report.py.
#
# A lane is the set of jobs whose completion is "the signal" for one commit; a
# repo has a ROCm lane (AMD self-hosted) and an NV lane (the reference).
#
# Fields:
#   short      key prefix for lane names and the PAIRS/REPO_URL lookups
#   repo       GitHub slug
#   label      display name, used for every heading and row label
#   rocm/nv    (workflow, job predicate, log-sample predicate) per lane
#   families   job families inside one workflow, or () when the repo has one.
#              openxla/xla runs its own tests and the JAX integration tests in
#              the same workflow, so its lanes split into "XLA jobs"/"JAX jobs".
#   pool       the DO scale-set this repo's ROCm jobs run on
#   nightly    nightly/continuous workflows, or () when the repo has none
#   job_type   classifies a job name for the per-job-type tables
#   pairs      (label, ROCm job type, NV job type) compared like-for-like


def _nv_gpu(j):
    return bool(re.search(r"\bGPU\b", j["name"]) and not re.search(r"ROCm|ONEAPI|AMD", j["name"], re.I))


def _jax_job_type(name):
    if "blocking gate" in name:
        return "blocking gate (1 test)"
    if name.startswith("Build wheels"):
        return "build wheels"
    b = re.search(r"build_jaxlib=(\w+)", name)
    return f"test, build_jaxlib={b.group(1)}" if b else name[:40]


def _xla_job_type(name):
    return "JAX job" if name.startswith("JAX") else "XLA job"


REPOS = [
    {
        "short": "xla",
        "repo": "openxla/xla",
        "label": "OpenXLA",
        "rocm": ("rocm_ci.yml", is_do, is_do),
        "nv": ("ci.yml", _nv_gpu, _nv_gpu),
        "families": (("XLA jobs", "xla"), ("JAX jobs", "jax")),
        "family_of": lambda name: "jax" if name.startswith("JAX") else "xla",
        "pool": "amd-do-linux-xla-gpu-gfx950-dpx-1",
        "nightly": (),
        "job_type": _xla_job_type,
        # (label, ROCm job type, NV job type, family the type belongs to)
        "pairs": (("XLA job", "XLA job", "XLA job", "xla"),
                  ("JAX job", "JAX job", "JAX job", "jax")),
        # (label suffix, log-sample category, which bazel invocation in the log)
        "rbe": (("ROCm presubmit, XLA job (<code>single_gpu</code>)", "xla_rocm:xla", 0),
                ("ROCm presubmit, JAX job", "xla_rocm:jax", -1),
                ("NV presubmit, XLA T4 job (Google RBE)", "xla_nv:xla", -1)),
    },
    {
        "short": "jax",
        "repo": "jax-ml/jax",
        "label": "JAX",
        "rocm": ("bazel_rocm_presubmit.yml", is_do,
                 lambda j: is_do(j) and j["name"].startswith("Test /")),
        "nv": ("bazel_cuda_presubmit.yml", lambda j: True,
               lambda j: j["name"].startswith("Test /")),
        "families": (),
        "family_of": None,
        "pool": "amd-do-linux.jax.gpu.gfx950.1",
        "nightly": ("wheel_tests_continuous.yml", "wheel_tests_nightly_release.yml"),
        "job_type": _jax_job_type,
        "pairs": (("tests, jaxlib from PyPI", "test, build_jaxlib=false", "test, build_jaxlib=false", None),
                  ("tests, jaxlib built from source", "test, build_jaxlib=wheel", "test, build_jaxlib=true", None)),
        "rbe": (("ROCm presubmit", "jax_rocm", -1),
                ("NV presubmit (Google RBE)", "jax_nv", -1)),
        # Hand-written Metrics rows that are not derivable from CI metadata.
        "extra_metrics": (("Jax-lab complete run (minutes)",
                           "pending: not a GitHub Actions workflow, needs definition (owner: user)", "n/a"),),
    },
]

REPO_BY_SHORT = {R["short"]: R for R in REPOS}
LANES = ("rocm", "nv")


def lane_key(R, lane):
    """The stable key a lane's commits and coverage figures are stored under."""
    return f"{R['short']}_{lane}"


def cov_key(R, lane):
    """The log-sample category carrying this lane's test counts.

    A repo with families reports coverage for the family that owns the repo's
    own tests, not for the whole lane.
    """
    return f"{lane_key(R, lane)}:{R['short']}" if R["families"] else lane_key(R, lane)


def _lane_entry(R, lane):
    wf, pred, _ = R[lane]
    return (R["repo"], wf, pred, f"{R['label']} {'ROCm' if lane == 'rocm' else 'NV/CUDA'}")


PRESUBMIT_LANES = {lane_key(R, lane): _lane_entry(R, lane) for R in REPOS for lane in LANES}


def _split_lane(R, lane, fam):
    repo, wf, pred, label = PRESUBMIT_LANES[lane_key(R, lane)]
    fam_of = R["family_of"]
    return (repo, wf, lambda j, p=pred, f=fam: p(j) and fam_of(j["name"]) == f,
            f"{label} ({fam.upper()} jobs)")


# Sub-lanes: the signal each job family would give on its own. Reported
# alongside the full lanes, never instead of them; a lane's time to signal is
# the slower of its families on each commit, not the sum of the two.
SPLIT_LANES = {f"{lane_key(R, lane)}:{fam}": _split_lane(R, lane, fam)
               for R in REPOS for lane in LANES for _, fam in R["families"]}

ALL_LANES = {**PRESUBMIT_LANES, **SPLIT_LANES}

NIGHTLY_WORKFLOWS = tuple(wf for R in REPOS for wf in R["nightly"])


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
    for R in REPOS:
        for lane in LANES:
            wf, _, sample = R[lane]
            if rec["workflow"] != wf or not sample(job):
                continue
            key = lane_key(R, lane)
            return f"{key}:{R['family_of'](job['name'])}" if R["families"] else key
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
