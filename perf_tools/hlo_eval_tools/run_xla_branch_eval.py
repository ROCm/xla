#!/usr/bin/env python3
"""Build and evaluate HLOs for the XLA targets in xla_targets.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_CONFIG = SCRIPT_DIR / "xla_targets.json"
EVAL_SCRIPT = SCRIPT_DIR / "run_hlo_eval.sh"
RUNNER_TARGET = "//xla/tools/multihost_hlo_runner:hlo_runner_main"
RUNNER_PATH = Path("xla/tools/multihost_hlo_runner/hlo_runner_main")
ROCM_BAZELRC = Path("build_tools/rocm/rocm_xla_ci.bazelrc")
FULL_SHA = re.compile(r"[0-9a-fA-F]{40}")
ACTIVE_PROCESS: subprocess.Popen[str] | None = None


class CampaignInterrupted(KeyboardInterrupt):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xla-source-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hlo-path", type=Path)
    parser.add_argument("--num-repeats", type=int, default=2)
    return parser.parse_args()


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def capture(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"command failed ({result.returncode}): "
            f"{command_text(command)}\n{detail}"
        )
    return result.stdout.strip()


def git(repo: Path, *args: str, check: bool = True) -> str:
    command = ["git", "-C", str(repo), *args]
    if check:
        return capture(command)
    result = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def write_header(
    path: Path, target: dict[str, Any], stage: str, mode: str = "w"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(
            f"target_label={target.get('label') or target['revision']}\n"
            f"target_role={target['role']}\n"
            f"target_revision={target['revision']}\n"
            f"target_commit={target['commit']}\n"
            f"stage={stage}\n\n"
        )


def run_logged(
    command: list[str],
    cwd: Path,
    log: Path,
    env: dict[str, str] | None = None,
) -> int:
    global ACTIVE_PROCESS
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {command_text(command)}\n\n")
        ACTIVE_PROCESS = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return ACTIVE_PROCESS.wait()
        finally:
            ACTIVE_PROCESS = None


def handle_signal(signum: int, _frame: object) -> None:
    if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
        ACTIVE_PROCESS.terminate()
    raise CampaignInterrupted(signum)


def load_targets() -> list[dict[str, Any]]:
    value = json.loads(TARGET_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "targets",
    }:
        raise ValueError(
            "xla_targets.json must contain exactly schema_version and targets"
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError(
            f"unsupported target schema: {value['schema_version']!r}"
        )
    if not isinstance(value["targets"], list) or not value["targets"]:
        raise ValueError("xla_targets.json requires a non-empty target list")

    targets: list[dict[str, Any]] = []
    controls = 0
    for index, raw in enumerate(value["targets"]):
        allowed = {"revision", "commit", "role", "label"}
        if not isinstance(raw, dict) or "revision" not in raw:
            raise ValueError(f"target {index} must contain revision")
        if set(raw) - allowed:
            raise ValueError(f"target {index} contains unsupported fields")
        revision = raw["revision"]
        if (
            not isinstance(revision, str)
            or not revision
            or revision != revision.strip()
            or any(char.isspace() for char in revision)
            or revision.startswith("-")
        ):
            raise ValueError(f"target {index} has an invalid revision")
        configured = raw.get("commit")
        if configured is not None and (
            not isinstance(configured, str) or not FULL_SHA.fullmatch(configured)
        ):
            raise ValueError(
                f"target {index} commit must be null or a full SHA"
            )
        role = raw.get("role", "candidate")
        if role not in {"control", "candidate"}:
            raise ValueError(f"target {index} has an invalid role")
        controls += role == "control"
        targets.append(
            {
                "revision": revision,
                "configured_commit": (
                    configured.lower() if isinstance(configured, str) else None
                ),
                "role": role,
                "label": raw.get("label"),
            }
        )
    if controls != 1:
        raise ValueError(
            f"xla_targets.json requires exactly one control; found {controls}"
        )
    return targets


def fetch_and_resolve(
    repo: Path, targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    remotes = set(git(repo, "remote").splitlines())
    required = {
        target["revision"].split("/", 1)[0]
        for target in targets
        if "/" in target["revision"]
    }
    missing = sorted(required - remotes)
    if missing:
        raise ValueError(f"missing Git remote(s): {', '.join(missing)}")
    for remote in sorted(required):
        print(f"Fetching {remote}", flush=True)
        git(repo, "fetch", remote, "--prune")

    resolved: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for target in targets:
        requested = target["configured_commit"] or target["revision"]
        canonical = requested
        if not requested.startswith("refs/") and "/" in requested:
            remote, _ = requested.split("/", 1)
            if remote in remotes:
                canonical = f"refs/remotes/{requested}"
        commit = git(
            repo,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{canonical}^{{commit}}",
        ).lower()
        if not FULL_SHA.fullmatch(commit):
            raise RuntimeError(f"failed to resolve target {target['revision']}")
        if commit in seen:
            raise ValueError(
                f"{seen[commit]!r} and {target['revision']!r} "
                f"resolve to the same commit"
            )
        seen[commit] = target["revision"]
        prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", target["revision"])
        resolved.append(
            {
                **target,
                "commit": commit,
                "slug": f"{prefix.strip('._-') or 'xla'}_{commit[:12]}",
            }
        )
    return resolved


def source_state(repo: Path) -> dict[str, str | None]:
    return {
        "branch": git(
            repo,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        )
        or None,
        "commit": git(repo, "rev-parse", "HEAD"),
        "status": git(
            repo, "status", "--porcelain", "--untracked-files=all"
        ),
    }


def require_clean(repo: Path) -> dict[str, str | None]:
    state = source_state(repo)
    if state["status"]:
        raise ValueError(f"XLA source checkout is not clean:\n{state['status']}")
    return state


def restore_source(repo: Path, original: dict[str, str | None]) -> None:
    require_clean(repo)
    if original["branch"]:
        git(repo, "checkout", "--no-overwrite-ignore", str(original["branch"]))
    else:
        git(
            repo,
            "checkout",
            "--no-overwrite-ignore",
            "--detach",
            str(original["commit"]),
        )


def acquire_lock(repo: Path, output: Path) -> int:
    if os.name != "posix":
        raise RuntimeError("campaign execution requires Linux")
    fcntl = __import__("fcntl")
    lock = Path(git(repo, "rev-parse", "--absolute-git-dir")) / (
        "hlo-eval-campaign.lock"
    )
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(f"another campaign holds {lock}") from error
    os.ftruncate(descriptor, 0)
    os.write(descriptor, str(output).encode())
    return descriptor


def evaluate_target(
    target: dict[str, Any],
    source_repo: Path,
    output: Path,
    hlo_path: Path,
    repeats: int,
) -> dict[str, Any]:
    target_dir = output / target["slug"]
    build_log = target_dir / "build.log"
    eval_log = target_dir / "eval.log"
    csv_dir = target_dir / "csv"
    csv_dir.mkdir(parents=True)
    result: dict[str, Any] = {
        **target,
        "status": "failed",
        "stage": "checkout",
        "exit_code": 2,
        "log": build_log,
        "csv_dir": csv_dir,
    }
    write_header(build_log, target, "checkout")
    try:
        require_clean(source_repo)
        git(
            source_repo,
            "checkout",
            "--no-overwrite-ignore",
            "--detach",
            target["commit"],
        )
        if git(source_repo, "rev-parse", "HEAD") != target["commit"]:
            raise RuntimeError("checked-out commit does not match target")

        bazelrc = source_repo / ROCM_BAZELRC
        if not bazelrc.is_file():
            raise RuntimeError(f"required ROCm BazelRC is missing: {bazelrc}")
        build = [
            "bazel",
            f"--bazelrc={bazelrc}",
            "build",
            "-c",
            "opt",
            "--config=rocm",
            RUNNER_TARGET,
        ]
        write_header(build_log, target, "build", mode="a")
        result["stage"] = "build"
        build_rc = run_logged(build, source_repo, build_log)
        if build_rc:
            result.update(exit_code=build_rc)
            return result

        bazel_bin = Path(
            capture(
                [
                    "bazel",
                    f"--bazelrc={bazelrc}",
                    "info",
                    "-c",
                    "opt",
                    "--config=rocm",
                    "bazel-bin",
                ],
                source_repo,
            ).splitlines()[-1]
        )
        runner = bazel_bin / RUNNER_PATH
        if not runner.is_file() or not os.access(runner, os.X_OK):
            raise RuntimeError(f"built runner is missing: {runner}")

        command = [
            "bash",
            str(EVAL_SCRIPT),
            str(runner),
            str(hlo_path),
            str(csv_dir),
            str(repeats),
        ]
        write_header(eval_log, target, "evaluation")
        environment = os.environ.copy()
        environment["SETTLE_SEC"] = "0"
        result.update(stage="evaluation", log=eval_log)
        eval_rc = run_logged(
            command, SCRIPT_DIR, eval_log, env=environment
        )
        result.update(
            status="completed" if eval_rc == 0 else "failed",
            exit_code=eval_rc,
        )
    except Exception as error:
        with result["log"].open("a", encoding="utf-8") as stream:
            stream.write(f"error={error}\n")
        result["error"] = str(error)
    return result


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n==== campaign summary ====")
    for result in results:
        status = "PASS" if result["status"] == "completed" else "FAIL"
        print(
            f"{status} {result['revision']}@{result['commit'][:12]} "
            f"stage={result['stage']} rc={result['exit_code']} "
            f"csv={result['csv_dir']}"
        )
        if status == "FAIL":
            print(f"  log={result['log']}")
            if result.get("error"):
                print(f"  error={result['error']}")


def main() -> int:
    args = parse_args()
    if args.num_repeats < 1:
        raise ValueError("--num-repeats must be at least 1")
    if not EVAL_SCRIPT.is_file():
        raise ValueError(f"evaluator not found: {EVAL_SCRIPT}")

    source_repo = args.xla_source_repo.expanduser().resolve(strict=True)
    root = Path(git(source_repo, "rev-parse", "--show-toplevel")).resolve()
    if root != source_repo:
        raise ValueError("--xla-source-repo must be a Git root")
    hlo_path = (
        SCRIPT_DIR if args.hlo_path is None else args.hlo_path
    ).expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    original = require_clean(source_repo)
    lock = acquire_lock(source_repo, output)
    results: list[dict[str, Any]] = []
    restore_error: Exception | None = None
    try:
        targets = fetch_and_resolve(source_repo, load_targets())
        for target in targets:
            print(
                f"[{target.get('label') or target['revision']}] "
                f"{target['revision']} -> {target['commit']}",
                flush=True,
            )
            results.append(
                evaluate_target(
                    target, source_repo, output, hlo_path, args.num_repeats
                )
            )
    finally:
        try:
            restore_source(source_repo, original)
        except Exception as error:
            restore_error = error
            print(f"FAIL source restore: {error}", file=sys.stderr)
        os.close(lock)

    print_summary(results)
    return 0 if (
        restore_error is None
        and results
        and all(result["status"] == "completed" for result in results)
    ) else 1


if __name__ == "__main__":
    for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled_signal, handle_signal)
    try:
        raise SystemExit(main())
    except CampaignInterrupted as error:
        print(f"Campaign interrupted by signal {error.signum}", file=sys.stderr)
        raise SystemExit(128 + error.signum)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
