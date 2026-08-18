#!/usr/bin/env python3
"""Build and evaluate HLOs for the XLA targets in xla_targets.json."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
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


def format_command_for_log(command: list[str]) -> str:
    """Format one argv sequence as shell-readable diagnostic text."""
    return " ".join(shlex.quote(item) for item in command)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_campaign_manifest_atomically(
    path: Path, value: dict[str, Any]
) -> None:
    """Replace the campaign manifest only after writing complete JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def collect_campaign_environment() -> dict[str, Any]:
    """Capture live host and visible-device metadata."""
    return {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "visible_devices": {
            name: os.environ.get(name)
            for name in (
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            )
        },
    }


def workload_leaf_relative_path(leaf: Path) -> str:
    """Return the corpus-relative category/model/workload leaf path."""
    try:
        return leaf.relative_to(SCRIPT_DIR).as_posix()
    except ValueError:
        count = 3 if leaf.name == "training" else 4
        return "/".join(leaf.parts[-count:])


def build_hlo_workload_inventory(hlo_path: Path) -> dict[str, Any]:
    """Describe every selected non-empty workload leaf and HLO module."""
    files = (
        [hlo_path]
        if hlo_path.is_file()
        else [
            path
            for path in hlo_path.rglob("*")
            if path.is_file() and path.suffix in {".txt", ".hlo"}
        ]
    )
    modules_by_leaf: dict[Path, list[str]] = {}
    for path in files:
        leaf = path.parent
        if not (
            leaf.name == "training"
            or (
                re.fullmatch(r"[0-9]+gpu", leaf.name) is not None
                and leaf.parent.name == "inference"
            )
        ):
            continue
        modules_by_leaf.setdefault(leaf, []).append(path.name)

    workloads = []
    for leaf, modules in sorted(
        modules_by_leaf.items(), key=lambda item: str(item[0])
    ):
        name = workload_leaf_relative_path(leaf)
        workloads.append(
            {
                "workload": name.replace("/", "_") + ".csv",
                "leaf": name,
                "modules": sorted(modules),
            }
        )
    return {
        "schema_version": 1,
        "selected_hlo_path": str(hlo_path),
        "workload_count": len(workloads),
        "workloads": workloads,
    }


def build_target_manifest_entry(target: dict[str, Any]) -> dict[str, Any]:
    """Convert one resolved target into its stable manifest representation."""
    target_id = f"{target['role']}:{target['revision']}"
    return {
        "id": target_id,
        "role": target["role"],
        "ref": target["revision"],
        "source_ref": target["revision"],
        "revision": target["revision"],
        "commit": target["commit"],
        "slug": target["slug"],
        "label": target.get("label") or target["revision"],
    }


def build_target_result_manifest_entry(
    result: dict[str, Any],
    output: Path,
    hlo_path: Path,
) -> dict[str, Any]:
    """Convert one target result into report-oriented manifest metadata."""
    stage = result["stage"]
    exit_code = result["exit_code"]
    target_dir = output / result["slug"]
    paths = {
        "build_log": str(target_dir / "build.log"),
        "eval_log": str(target_dir / "eval.log"),
        "results": str(target_dir / "csv"),
        "hlo_input": str(SCRIPT_DIR),
        "selected_hlo_path": str(hlo_path),
    }
    return {
        **build_target_manifest_entry(result),
        "status": result["status"],
        "stage": stage,
        "build_exit_code": (
            exit_code
            if stage == "build"
            else 0
            if stage == "evaluation"
            else None
        ),
        "evaluation_exit_code": (
            exit_code if stage == "evaluation" else None
        ),
        "error": result.get("error"),
        "paths": paths,
    }


def run_and_capture_stdout(
    command: list[str], cwd: Path | None = None
) -> str:
    """Run a command and return stdout, raising with captured diagnostics."""
    result = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"command failed ({result.returncode}): "
            f"{format_command_for_log(command)}\n{detail}"
        )
    return result.stdout.strip()


def run_git_command(
    repo: Path, *args: str, check: bool = True
) -> str:
    """Run Git against the campaign's dedicated source checkout."""
    command = ["git", "-C", str(repo), *args]
    if check:
        return run_and_capture_stdout(command)
    result = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def write_stage_log_header(
    path: Path, target: dict[str, Any], stage: str, mode: str = "w"
) -> None:
    """Write target identity and stage metadata at the start of a log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(
            f"target_label={target.get('label') or target['revision']}\n"
            f"target_role={target['role']}\n"
            f"target_revision={target['revision']}\n"
            f"target_commit={target['commit']}\n"
            f"stage={stage}\n\n"
        )


def run_command_with_log(
    command: list[str],
    cwd: Path,
    log: Path,
    env: dict[str, str] | None = None,
) -> int:
    """Run a foreground command while appending combined output to a log."""
    global ACTIVE_PROCESS
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {format_command_for_log(command)}\n\n")
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


def handle_campaign_signal(signum: int, _frame: object) -> None:
    """Terminate the active child and interrupt campaign orchestration."""
    if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
        ACTIVE_PROCESS.terminate()
    raise CampaignInterrupted(signum)


def load_campaign_targets() -> list[dict[str, Any]]:
    """Load and validate the configured control and candidate targets."""
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


def fetch_and_resolve_targets(
    repo: Path, targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fetch required remotes and resolve every target to one commit."""
    remotes = set(run_git_command(repo, "remote").splitlines())
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
        run_git_command(repo, "fetch", remote, "--prune")

    resolved: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for target in targets:
        requested = target["configured_commit"] or target["revision"]
        canonical = requested
        if not requested.startswith("refs/") and "/" in requested:
            remote, _ = requested.split("/", 1)
            if remote in remotes:
                canonical = f"refs/remotes/{requested}"
        commit = run_git_command(
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


def capture_source_checkout_state(
    repo: Path,
) -> dict[str, str | None]:
    """Capture the branch, commit, and cleanliness of the source checkout."""
    return {
        "branch": run_git_command(
            repo,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        )
        or None,
        "commit": run_git_command(repo, "rev-parse", "HEAD"),
        "status": run_git_command(
            repo, "status", "--porcelain", "--untracked-files=all"
        ),
    }


def require_clean_source_checkout(
    repo: Path,
) -> dict[str, str | None]:
    """Require and return a clean source checkout state."""
    state = capture_source_checkout_state(repo)
    if state["status"]:
        raise ValueError(f"XLA source checkout is not clean:\n{state['status']}")
    return state


def restore_source_checkout(
    repo: Path, original: dict[str, str | None]
) -> None:
    """Restore the source checkout to its original branch or commit."""
    require_clean_source_checkout(repo)
    if original["branch"]:
        run_git_command(
            repo,
            "checkout",
            "--no-overwrite-ignore",
            str(original["branch"]),
        )
    else:
        run_git_command(
            repo,
            "checkout",
            "--no-overwrite-ignore",
            "--detach",
            str(original["commit"]),
        )


def acquire_campaign_lock(repo: Path, output: Path) -> int:
    """Acquire the source-checkout lock for one campaign process."""
    if os.name != "posix":
        raise RuntimeError("campaign execution requires Linux")
    fcntl = __import__("fcntl")
    lock = Path(
        run_git_command(repo, "rev-parse", "--absolute-git-dir")
    ) / (
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


def build_and_evaluate_target(
    target: dict[str, Any],
    source_repo: Path,
    output: Path,
    hlo_path: Path,
    repeats: int,
) -> dict[str, Any]:
    """Build one resolved XLA target and evaluate the selected HLO corpus."""
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
    write_stage_log_header(build_log, target, "checkout")
    try:
        require_clean_source_checkout(source_repo)
        run_git_command(
            source_repo,
            "checkout",
            "--no-overwrite-ignore",
            "--detach",
            target["commit"],
        )
        if (
            run_git_command(source_repo, "rev-parse", "HEAD")
            != target["commit"]
        ):
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
        write_stage_log_header(build_log, target, "build", mode="a")
        result["stage"] = "build"
        build_rc = run_command_with_log(build, source_repo, build_log)
        if build_rc:
            result.update(exit_code=build_rc)
            return result

        bazel_bin = Path(
            run_and_capture_stdout(
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
        write_stage_log_header(eval_log, target, "evaluation")
        environment = os.environ.copy()
        environment["SETTLE_SEC"] = "0"
        result.update(stage="evaluation", log=eval_log)
        eval_rc = run_command_with_log(
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


def print_campaign_summary(results: list[dict[str, Any]]) -> None:
    """Print the final per-target campaign status and artifact locations."""
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
    root = Path(
        run_git_command(source_repo, "rev-parse", "--show-toplevel")
    ).resolve()
    if root != source_repo:
        raise ValueError("--xla-source-repo must be a Git root")
    hlo_path = (
        SCRIPT_DIR if args.hlo_path is None else args.hlo_path
    ).expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    original = require_clean_source_checkout(source_repo)
    lock = acquire_campaign_lock(source_repo, output)
    results: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None
    manifest_path = output / "manifest.json"
    interrupted = False
    restore_error: Exception | None = None
    try:
        targets = fetch_and_resolve_targets(
            source_repo, load_campaign_targets()
        )
        campaign_targets = [
            build_target_manifest_entry(target) for target in targets
        ]
        control = next(
            target
            for target in campaign_targets
            if target["role"] == "control"
        )
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "running",
            "benchmark": {
                "effective": {
                    "num_repeats": args.num_repeats,
                    "arg_mode": os.environ.get(
                        "ARG_MODE", "uninitialized"
                    ),
                    "cmd_buffer": os.environ.get("CMD_BUFFER", "off"),
                    "order": os.environ.get("ORDER", "size"),
                    "settle_sec": 0,
                }
            },
            "environment": collect_campaign_environment(),
            "targets": campaign_targets,
            "results": [],
            "live_control_id": control["id"],
            "reference_dataset": {
                "id": "not_recorded",
                "role": "historical_reference",
                "source": "unavailable",
                "inventory": build_hlo_workload_inventory(hlo_path),
            },
        }
        write_campaign_manifest_atomically(manifest_path, manifest)
        for target in targets:
            print(
                f"[{target.get('label') or target['revision']}] "
                f"{target['revision']} -> {target['commit']}",
                flush=True,
            )
            result = build_and_evaluate_target(
                target, source_repo, output, hlo_path, args.num_repeats
            )
            results.append(result)
            manifest["results"].append(
                build_target_result_manifest_entry(
                    result, output, hlo_path
                )
            )
            write_campaign_manifest_atomically(manifest_path, manifest)
    except CampaignInterrupted:
        interrupted = True
        raise
    finally:
        try:
            restore_source_checkout(source_repo, original)
        except Exception as error:
            restore_error = error
            print(f"FAIL source restore: {error}", file=sys.stderr)
        if manifest is not None:
            completed = sum(
                result["status"] == "completed" for result in results
            )
            manifest["status"] = (
                "interrupted"
                if interrupted
                else "completed"
                if restore_error is None and completed == len(targets)
                else "completed_with_failures"
            )
            manifest["finished_at"] = utc_now()
            manifest["source_restored"] = restore_error is None
            manifest["summary"] = {
                "total": len(targets),
                "completed": completed,
                "failed": len(results) - completed,
                "not_run": len(targets) - len(results),
            }
            write_campaign_manifest_atomically(manifest_path, manifest)
        os.close(lock)

    print_campaign_summary(results)
    if manifest_path.is_file():
        print(f"manifest={manifest_path}")
    return 0 if (
        restore_error is None
        and results
        and all(result["status"] == "completed" for result in results)
    ) else 1


if __name__ == "__main__":
    for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled_signal, handle_campaign_signal)
    try:
        raise SystemExit(main())
    except CampaignInterrupted as error:
        print(f"Campaign interrupted by signal {error.signum}", file=sys.stderr)
        raise SystemExit(128 + error.signum)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
