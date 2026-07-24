#!/usr/bin/env python3
"""Build and benchmark hlo_runner_main across a list of XLA Git refs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare_hlo_branch_results import select_comparison_targets, write_comparison


UPSTREAM_URL = "https://github.com/openxla/xla.git"
RUNNER_TARGET = "//xla/tools/multihost_hlo_runner:hlo_runner_main"
RUNNER_RELATIVE_PATH = Path(
    "xla/tools/multihost_hlo_runner/hlo_runner_main"
)
ACTIVE_PROCESS: subprocess.Popen[str] | None = None


class CampaignInterrupted(KeyboardInterrupt):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--perf-tools-repo",
        type=Path,
        help="default: Git repository containing this script",
    )
    parser.add_argument("--xla-source-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--refs-file",
        type=Path,
        help=(
            "default: <perf-tools-repo>/perf_tools/hlo_eval_tools/"
            "configs/xla_refs.txt"
        ),
    )
    parser.add_argument(
        "--profile-file",
        type=Path,
        help=(
            "default: <perf-tools-repo>/perf_tools/hlo_eval_tools/"
            "configs/benchmark_profile.json"
        ),
    )
    parser.add_argument("--num-repeats", type=positive_int)
    parser.add_argument("--arg-mode")
    parser.add_argument("--cmd-buffer", choices=("off", "on"))
    parser.add_argument("--order", choices=("size", "path"))
    parser.add_argument("--settle-sec", type=nonnegative_int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse built runners and skip leaves whose CSV already exists",
    )
    parser.add_argument(
        "--bazel-command",
        help="Bazel executable (default: bazel; it may be a Bazelisk symlink)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="use currently available remote-tracking refs without fetching",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve refs and print the plan without building or evaluating",
    )
    return parser.parse_args()


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def signal_process_group(process: subprocess.Popen[str], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def handle_campaign_signal(signum: int, _frame: Any) -> None:
    if ACTIVE_PROCESS is not None:
        signal_process_group(ACTIVE_PROCESS, signum)
    raise CampaignInterrupted(signum)


def run_capture(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> str:
    global ACTIVE_PROCESS
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    ACTIVE_PROCESS = process
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        signal_process_group(process, signal.SIGKILL)
        process.communicate()
        raise
    except CampaignInterrupted:
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                signal_process_group(process, signal.SIGKILL)
                process.communicate()
        raise
    except BaseException:
        signal_process_group(process, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            process.communicate()
        raise
    finally:
        ACTIVE_PROCESS = None
    if check and process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {command_text(command)}\n"
            f"{stdout.strip()}"
        )
    return stdout.strip()


def run_logged(
    command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None
) -> int:
    global ACTIVE_PROCESS
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {command_text(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        ACTIVE_PROCESS = process
        try:
            return_code = process.wait()
        except CampaignInterrupted:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_process_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    signal_process_group(process, signal.SIGKILL)
                    process.wait()
            raise
        except BaseException:
            signal_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_process_group(process, signal.SIGKILL)
                process.wait()
            raise
        finally:
            ACTIVE_PROCESS = None
        log.write(f"\n[exit_code={return_code}]\n")
    return return_code


def git(repo: Path, *args: str, check: bool = True) -> str:
    return run_capture(["git", *args], cwd=repo, check=check)


def validate_git_root(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{label} does not exist or is not a directory: {path}")
    root = Path(git(path, "rev-parse", "--show-toplevel")).resolve()
    if root != path:
        raise ValueError(f"{label} must be the Git repository root: {path} (root={root})")
    return path


def discover_perf_tools_repo() -> Path:
    script_directory = Path(__file__).resolve().parent
    try:
        return Path(git(script_directory, "rev-parse", "--show-toplevel")).resolve()
    except RuntimeError as error:
        raise ValueError(
            "could not discover the perf-tools Git repository from "
            f"{script_directory}; pass --perf-tools-repo explicitly"
        ) from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_refs(path: Path) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        ref = raw_line.strip()
        if not ref or ref.startswith("#"):
            continue
        if any(character.isspace() for character in ref):
            raise ValueError(f"{path}:{line_number}: whitespace is not allowed in a ref")
        if ref.startswith("-"):
            raise ValueError(f"{path}:{line_number}: ref cannot begin with '-': {ref}")
        if ref in seen:
            raise ValueError(f"{path}:{line_number}: duplicate ref: {ref}")
        refs.append(ref)
        seen.add(ref)
    if not refs:
        raise ValueError(f"no XLA refs found in {path}")
    return refs


def validate_resume_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("resume manifest must contain a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError(
            "cannot resume unsupported manifest schema: "
            f"{manifest.get('schema_version')}"
        )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("resume manifest has no valid inputs object")
    for name in (
        "perf_tools_repo",
        "xla_source_repo",
        "profile_file",
        "evaluation_script",
        "comparison_script",
    ):
        if not isinstance(inputs.get(name), dict):
            raise ValueError(f"resume manifest has no valid inputs.{name} object")
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, dict) or not isinstance(
        benchmark.get("effective"), dict
    ):
        raise ValueError("resume manifest has no valid benchmark.effective object")
    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("resume manifest has no valid target list")
    for index, target in enumerate(targets):
        if not isinstance(target, dict) or any(
            not isinstance(target.get(key), str) or not target[key]
            for key in ("ref", "commit", "slug")
        ):
            raise ValueError(f"resume manifest target {index} is invalid")
    if "source_original_state" in manifest:
        source_state = manifest["source_original_state"]
        if (
            not isinstance(source_state, dict)
            or not isinstance(source_state.get("commit"), str)
            or not source_state["commit"]
            or source_state.get("branch") is not None
            and not isinstance(source_state["branch"], str)
            or not isinstance(source_state.get("status"), str)
        ):
            raise ValueError("resume manifest source_original_state is invalid")


def load_profile(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise ValueError(f"unsupported benchmark profile schema: {profile.get('schema_version')}")
    defaults = profile.get("runner")
    if not isinstance(defaults, dict):
        raise ValueError(f"benchmark profile has no runner object: {path}")

    required = {
        "num_repeats": int,
        "arg_mode": str,
        "cmd_buffer": str,
        "order": str,
        "settle_sec": int,
    }
    for key, expected_type in required.items():
        if not isinstance(defaults.get(key), expected_type):
            raise ValueError(f"invalid or missing runner.{key} in {path}")
    if defaults["num_repeats"] < 1 or defaults["settle_sec"] < 0:
        raise ValueError(f"invalid repeat/settle value in {path}")
    if defaults["cmd_buffer"] not in {"off", "on"}:
        raise ValueError(f"invalid runner.cmd_buffer in {path}")
    if defaults["order"] not in {"size", "path"}:
        raise ValueError(f"invalid runner.order in {path}")

    cli_values = {
        "num_repeats": args.num_repeats,
        "arg_mode": args.arg_mode,
        "cmd_buffer": args.cmd_buffer,
        "order": args.order,
        "settle_sec": args.settle_sec,
    }
    effective = dict(defaults)
    overrides: dict[str, Any] = {}
    for key, value in cli_values.items():
        if value is not None:
            effective[key] = value
            if value != defaults[key]:
                overrides[key] = {"repository_default": defaults[key], "value": value}
    return profile, {
        "profile_name": profile.get("name", "unnamed"),
        "reference_aligned": not overrides,
        "effective": effective,
        "overrides": overrides,
    }


def ensure_and_fetch_remotes(repo: Path, refs: list[str], skip_fetch: bool) -> None:
    remotes = set(git(repo, "remote").splitlines())
    needed = {ref.split("/", 1)[0] for ref in refs if "/" in ref}

    if "upstream" in needed and "upstream" not in remotes:
        if skip_fetch:
            raise ValueError(
                f"upstream remote is required but missing; add it with: "
                f"git remote add upstream {UPSTREAM_URL}"
            )
        print(f"Adding upstream remote: {UPSTREAM_URL}", file=sys.stderr)
        git(repo, "remote", "add", "upstream", UPSTREAM_URL)
        remotes.add("upstream")

    missing = sorted(needed - remotes)
    if missing:
        raise ValueError(f"refs use unknown Git remote(s): {', '.join(missing)}")

    if not skip_fetch:
        for remote in sorted(needed):
            print(f"Fetching {remote}...", file=sys.stderr)
            git(repo, "fetch", remote, "--prune")


def resolve_refs(repo: Path, refs: list[str]) -> list[dict[str, str]]:
    resolved: list[dict[str, str]] = []
    for ref in refs:
        sha = git(
            repo,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{ref}^{{commit}}",
        )
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ref).strip("._-") or "xla"
        resolved.append({"ref": ref, "commit": sha, "slug": f"{slug}_{sha[:12]}"})
    return resolved


def validate_reference_target(
    profile: dict[str, Any], targets: list[dict[str, str]]
) -> None:
    reference = profile.get("reference", {})
    reference_ref = reference.get("xla_ref")
    reference_commit = reference.get("xla_commit")
    if not reference_ref or not reference_commit:
        return
    target = next((item for item in targets if item["ref"] == reference_ref), None)
    if target is None:
        raise ValueError(f"reference ref is not present in the target list: {reference_ref}")
    if target["commit"] != reference_commit:
        raise ValueError(
            f"reference ref {reference_ref} resolved to {target['commit']}, but the "
            f"benchmark profile requires {reference_commit}"
        )


def missing_target_commits(
    repo: Path, targets: list[dict[str, str]]
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for target in targets:
        try:
            git(
                repo,
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{target['commit']}^{{commit}}",
            )
        except RuntimeError:
            missing.append(target)
    return missing


def choose_bazel(requested: str | None) -> str:
    command = requested or "bazel"
    executable = shutil.which(command)
    if executable is None:
        raise ValueError(f"Bazel executable not found: {command}")
    return executable


def bazel_version(executable: str, cwd: Path) -> str:
    output = run_capture([executable, "--version"], cwd=cwd)
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
    return match.group(1) if match else output


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "visible_devices": {
            name: os.environ.get(name)
            for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        },
    }
    for name, command in {
        "rocm_smi": ["rocm-smi", "--showproductname", "--showdriverversion"],
        "hipcc": ["hipcc", "--version"],
    }.items():
        if shutil.which(command[0]):
            try:
                environment[name] = run_capture(command, check=False, timeout=15)
            except subprocess.TimeoutExpired:
                environment[name] = "timed out"
    return environment


def hlo_inventory(hlo_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in hlo_root.rglob("*")
        if path.is_file()
        and path.suffix in {".txt", ".hlo"}
    )
    for path in files:
        relative = path.relative_to(hlo_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def hlo_pathspecs(hlo_root: Path, repo: Path) -> list[str]:
    relative = hlo_root.relative_to(repo).as_posix()
    return [
        f":(glob){relative}/*.txt",
        f":(glob){relative}/*.hlo",
        f":(glob){relative}/**/*.txt",
        f":(glob){relative}/**/*.hlo",
    ]


def legacy_hlo_changes(
    repo: Path, hlo_root: Path, old_commit: str, new_commit: str
) -> str:
    return git(
        repo,
        "diff",
        "--name-only",
        old_commit,
        new_commit,
        "--",
        *hlo_pathspecs(hlo_root, repo),
    )


def require_unchanged_hlo_inventory(
    hlo_root: Path, expected: dict[str, Any], context: str
) -> None:
    current = hlo_inventory(hlo_root)
    if current != expected:
        raise ValueError(
            f"HLO corpus changed {context}; expected {expected}, found {current}. "
            "Restore the original HLO inputs or start a new campaign."
        )


def repository_metadata(repo: Path, hlo_root: Path | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(repo),
        "commit": git(repo, "rev-parse", "HEAD"),
        "status": git(repo, "status", "--short"),
    }
    if hlo_root is not None:
        relative = hlo_root.relative_to(repo)
        metadata["hlo_tree"] = git(repo, "rev-parse", f"HEAD:{relative.as_posix()}")
        metadata["hlo_inventory"] = hlo_inventory(hlo_root)
        metadata["hlo_status"] = git(
            repo,
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            *hlo_pathspecs(hlo_root, repo),
        )
    return metadata


def source_checkout_state(repo: Path) -> dict[str, str | None]:
    return {
        "branch": git(
            repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
        )
        or None,
        "commit": git(repo, "rev-parse", "HEAD"),
        "status": git(repo, "status", "--porcelain", "--untracked-files=all"),
    }


def acquire_source_lock(repo: Path, output_dir: Path) -> int:
    if os.name != "posix":
        raise RuntimeError("campaign locking requires a POSIX execution environment")
    fcntl = __import__("fcntl")
    git_dir = Path(git(repo, "rev-parse", "--absolute-git-dir"))
    lock_path = git_dir / "hlo-eval-campaign.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.lseek(descriptor, 0, os.SEEK_SET)
        owner = os.read(descriptor, 4096).decode("utf-8", errors="replace").strip()
        os.close(descriptor)
        details = f"\nLock owner:\n{owner}" if owner else ""
        raise ValueError(
            f"another HLO evaluation campaign is using {repo}.{details}"
        ) from error
    owner = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "output_dir": str(output_dir.expanduser().resolve()),
        "acquired_at": utc_now(),
    }
    os.ftruncate(descriptor, 0)
    os.write(descriptor, (json.dumps(owner, indent=2) + "\n").encode("utf-8"))
    os.fsync(descriptor)
    return descriptor


def release_source_lock(descriptor: int) -> None:
    fcntl = __import__("fcntl")
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def require_clean_source_repo(repo: Path) -> dict[str, str | None]:
    state = source_checkout_state(repo)
    if state["status"]:
        raise ValueError(
            f"XLA source repo is not clean: {repo}\n"
            "The campaign checks out multiple commits and will not stash, reset, "
            "or delete local changes.\n"
            f"Inspect it with: git -C {shlex.quote(str(repo))} status --short\n"
            "Commit, stash, or remove the changes before retrying:\n"
            f"{state['status']}"
        )
    return state


def restore_source_checkout(repo: Path, state: dict[str, str | None]) -> None:
    require_clean_source_repo(repo)
    branch = state["branch"]
    if branch:
        git(repo, "checkout", "--no-overwrite-ignore", branch)
    else:
        commit = state["commit"]
        if not commit:
            raise RuntimeError("original detached checkout has no recorded commit")
        git(repo, "checkout", "--no-overwrite-ignore", "--detach", commit)


def branch_metadata_path(branch_dir: Path) -> Path:
    return branch_dir / "metadata.json"


def evaluate_target(
    *,
    target: dict[str, str],
    source_repo: Path,
    output_dir: Path,
    bazel: str,
    eval_script: Path,
    hlo_root: Path,
    benchmark: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    branch_dir = output_dir / target["slug"]
    csv_dir = branch_dir / "csv"
    runner_copy = branch_dir / "runner" / "hlo_runner_main"
    build_log = branch_dir / "build.log"
    eval_log = branch_dir / "eval.log"
    metadata_file = branch_metadata_path(branch_dir)
    previous_metadata: dict[str, Any] = {}
    if resume and metadata_file.is_file():
        previous_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if (
            previous_metadata.get("status") in {"completed", "evaluation_failed"}
            and isinstance(previous_metadata.get("evaluation_exit_code"), int)
        ):
            previous_metadata.setdefault("resume_history", []).append(
                {
                    "skipped_at": utc_now(),
                    "reason": "target evaluation already finished",
                }
            )
            write_json(metadata_file, previous_metadata)
            print(
                f"[{target['ref']}] evaluation already finished; skipping on resume"
            )
            return previous_metadata
    metadata: dict[str, Any] = {
        **previous_metadata,
        **target,
        "started_at": utc_now(),
        "status": "running",
        "benchmark": benchmark,
        "paths": {
            "results": str(csv_dir),
            "build_log": str(build_log),
            "eval_log": str(eval_log),
            "runner": str(runner_copy),
        },
    }
    metadata.pop("error", None)
    metadata.pop("finished_at", None)
    write_json(metadata_file, metadata)

    try:
        runner_reusable = False
        if resume and runner_copy.is_file() and os.access(runner_copy, os.X_OK):
            expected_hash = previous_metadata.get("runner_sha256")
            runner_reusable = (
                previous_metadata.get("commit") == target["commit"]
                and isinstance(expected_hash, str)
                and sha256_file(runner_copy) == expected_hash
            )
            if not runner_reusable:
                metadata["runner_reuse_rejected"] = (
                    "missing or mismatched commit/checksum provenance"
                )
                write_json(metadata_file, metadata)

        if not runner_reusable:
            require_clean_source_repo(source_repo)
            metadata["status"] = "checking_out"
            write_json(metadata_file, metadata)
            git(
                source_repo,
                "checkout",
                "--no-overwrite-ignore",
                "--detach",
                target["commit"],
            )
            checked_out_commit = git(source_repo, "rev-parse", "HEAD")
            if checked_out_commit != target["commit"]:
                raise RuntimeError(
                    f"checked out {checked_out_commit}, expected {target['commit']}"
                )
            metadata["source_head"] = checked_out_commit

            expected_bazel = (source_repo / ".bazelversion").read_text(
                encoding="utf-8"
            ).strip()
            actual_bazel = bazel_version(bazel, source_repo)
            metadata["bazel"] = {
                "command": bazel,
                "expected_version": expected_bazel,
                "actual_version": actual_bazel,
            }
            write_json(metadata_file, metadata)
            if expected_bazel != actual_bazel:
                raise RuntimeError(
                    f"{target['ref']} requires Bazel {expected_bazel}, but {bazel} "
                    f"reports {actual_bazel}; ensure bazel invokes Bazelisk or pass "
                    f"--bazel-command with a compatible launcher"
                )

            build_command = [
                bazel,
                "build",
                "-c",
                "opt",
                "--config=rocm",
                RUNNER_TARGET,
            ]
            metadata["build_command"] = build_command
            metadata["status"] = "building"
            write_json(metadata_file, metadata)
            print(f"[{target['ref']}] building {RUNNER_TARGET}")
            build_rc = run_logged(build_command, cwd=source_repo, log_path=build_log)
            metadata["build_exit_code"] = build_rc
            if build_rc != 0:
                metadata["status"] = "build_failed"
                metadata["finished_at"] = utc_now()
                write_json(metadata_file, metadata)
                return metadata

            bazel_bin = Path(
                run_capture(
                    [bazel, "info", "bazel-bin"],
                    cwd=source_repo,
                ).splitlines()[-1]
            )
            built_runner = bazel_bin / RUNNER_RELATIVE_PATH
            if not built_runner.is_file():
                raise RuntimeError(f"built runner not found: {built_runner}")
            runner_copy.parent.mkdir(parents=True, exist_ok=True)
            temporary_runner = runner_copy.with_suffix(".tmp")
            shutil.copy2(built_runner, temporary_runner)
            temporary_runner.chmod(temporary_runner.stat().st_mode | 0o111)
            temporary_runner.replace(runner_copy)
            metadata["runner_sha256"] = sha256_file(runner_copy)
            metadata["status"] = "runner_built"
            write_json(metadata_file, metadata)
        else:
            metadata["runner_reused"] = True
            metadata["runner_sha256"] = sha256_file(runner_copy)

        effective = benchmark["effective"]
        environment = os.environ.copy()
        environment.update(
            {
                "ARG_MODE": str(effective["arg_mode"]),
                "CMD_BUFFER": str(effective["cmd_buffer"]),
                "ORDER": str(effective["order"]),
                "SETTLE_SEC": str(effective["settle_sec"]),
                "RESUME": "1" if resume else "0",
            }
        )
        eval_command = [
            str(eval_script),
            str(runner_copy),
            str(hlo_root),
            str(csv_dir),
            str(effective["num_repeats"]),
        ]
        metadata["evaluation_command"] = eval_command
        metadata["evaluation_environment"] = {
            key: environment[key]
            for key in ("ARG_MODE", "CMD_BUFFER", "ORDER", "SETTLE_SEC", "RESUME")
        }
        metadata["status"] = "evaluating"
        write_json(metadata_file, metadata)
        print(f"[{target['ref']}] evaluating HLO corpus")
        eval_rc = run_logged(
            eval_command,
            cwd=eval_script.parent,
            log_path=eval_log,
            env=environment,
        )
        metadata["evaluation_exit_code"] = eval_rc
        metadata["status"] = "completed" if eval_rc == 0 else "evaluation_failed"
        metadata["finished_at"] = utc_now()
        write_json(metadata_file, metadata)
        return metadata
    except Exception as error:
        metadata["status"] = "error"
        metadata["error"] = str(error)
        metadata["finished_at"] = utc_now()
        write_json(metadata_file, metadata)
        return metadata


def main() -> int:
    args = parse_args()
    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_signal_handlers = {
        signum: signal.getsignal(signum) for signum in handled_signals
    }
    for signum in handled_signals:
        signal.signal(signum, handle_campaign_signal)
    source_repo: Path | None = None
    source_original_state: dict[str, str | None] | None = None
    restore_source = False
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    campaign_manifest_written = False
    source_lock: int | None = None
    try:
        perf_repo = validate_git_root(
            args.perf_tools_repo or discover_perf_tools_repo(),
            "perf-tools repo",
        )
        source_repo = validate_git_root(args.xla_source_repo, "XLA source repo")
        if perf_repo == source_repo:
            raise ValueError("perf-tools repo and XLA source repo must be different checkouts")
        source_lock = acquire_source_lock(source_repo, args.output_dir)
        source_original_state = require_clean_source_repo(source_repo)

        hlo_root = perf_repo / "perf_tools/hlo_eval_tools"
        refs_file = (
            args.refs_file or hlo_root / "configs/xla_refs.txt"
        ).expanduser().resolve()
        profile_file = (
            args.profile_file or hlo_root / "configs/benchmark_profile.json"
        ).expanduser().resolve()
        eval_script = hlo_root / "run_hlo_eval.sh"
        comparison_script = hlo_root / "scripts/compare_hlo_branch_results.py"
        for required in (refs_file, profile_file, eval_script, comparison_script):
            if not required.is_file():
                raise ValueError(f"required file not found: {required}")
        if not os.access(eval_script, os.X_OK):
            raise ValueError(f"evaluation script is not executable: {eval_script}")

        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.is_relative_to(perf_repo) or output_dir.is_relative_to(source_repo):
            raise ValueError(
                "output directory must be outside both Git repositories: "
                f"{output_dir}"
            )
        manifest_path = output_dir / "manifest.json"
        previous: dict[str, Any] | None = None
        output_has_files = output_dir.exists() and any(output_dir.iterdir())
        if output_has_files and not args.resume and not args.dry_run:
            raise ValueError(
                f"output directory is not empty: {output_dir}; choose a new "
                "directory or pass --resume"
            )
        if args.resume:
            if not manifest_path.is_file():
                raise ValueError(
                    f"cannot resume without an existing manifest: {manifest_path}"
                )
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_resume_manifest(previous)

        refs = read_refs(refs_file)
        profile, benchmark = load_profile(profile_file, args)
        added_refs: list[str] = []
        removed_refs: list[str] = []
        if previous is None:
            ensure_and_fetch_remotes(source_repo, refs, args.skip_fetch)
            campaign_targets = resolve_refs(source_repo, refs)
            targets = campaign_targets
        else:
            previous_targets = previous["targets"]
            previous_by_ref = {
                target["ref"]: target for target in previous_targets
            }
            added_refs = [ref for ref in refs if ref not in previous_by_ref]
            removed_refs = [
                target["ref"]
                for target in previous_targets
                if target["ref"] not in refs
            ]
            if added_refs and not args.skip_fetch:
                ensure_and_fetch_remotes(source_repo, added_refs, skip_fetch=False)
            added_targets = resolve_refs(source_repo, added_refs) if added_refs else []
            campaign_targets = [*previous_targets, *added_targets]
            campaign_by_ref = {
                target["ref"]: target for target in campaign_targets
            }
            targets = [campaign_by_ref[ref] for ref in refs]
            missing = missing_target_commits(source_repo, targets)
            if missing and not args.skip_fetch:
                ensure_and_fetch_remotes(source_repo, refs, skip_fetch=False)
                missing = missing_target_commits(source_repo, targets)
            if missing:
                missing_text = ", ".join(
                    f"{target['ref']} ({target['commit']})" for target in missing
                )
                raise ValueError(
                    "cannot resume because these recorded commits are unavailable: "
                    f"{missing_text}"
                )
        validate_reference_target(profile, campaign_targets)
        bazel = choose_bazel(args.bazel_command)

        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": "planned" if args.dry_run else "running",
            "inputs": {
                "perf_tools_repo": repository_metadata(perf_repo, hlo_root),
                "xla_source_repo": repository_metadata(source_repo),
                "refs_file": {
                    "path": str(refs_file),
                    "sha256": sha256_file(refs_file),
                },
                "profile_file": {
                    "path": str(profile_file),
                    "sha256": sha256_file(profile_file),
                },
                "evaluation_script": {
                    "path": str(eval_script),
                    "sha256": sha256_file(eval_script),
                },
                "comparison_script": {
                    "path": str(comparison_script),
                    "sha256": sha256_file(comparison_script),
                },
            },
            "profile": profile,
            "benchmark": benchmark,
            "environment": collect_environment(),
            "bazel_command": bazel,
            "source_original_state": source_original_state,
            "targets": campaign_targets,
            "active_refs": refs,
            "target_selection": {
                "added_refs": added_refs,
                "removed_refs": removed_refs,
            },
        }

        if args.dry_run:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        if previous is not None:
            checks = {
                "benchmark profile": (
                    previous["inputs"]["profile_file"]["sha256"],
                    manifest["inputs"]["profile_file"]["sha256"],
                ),
                "effective benchmark settings": (
                    previous["benchmark"]["effective"],
                    manifest["benchmark"]["effective"],
                ),
                "XLA source repo": (
                    previous["inputs"]["xla_source_repo"]["path"],
                    manifest["inputs"]["xla_source_repo"]["path"],
                ),
                "perf-tools repo": (
                    previous["inputs"]["perf_tools_repo"]["path"],
                    manifest["inputs"]["perf_tools_repo"]["path"],
                ),
                "evaluation script": (
                    previous["inputs"]["evaluation_script"]["sha256"],
                    manifest["inputs"]["evaluation_script"]["sha256"],
                ),
                "comparison script": (
                    previous["inputs"]["comparison_script"]["sha256"],
                    manifest["inputs"]["comparison_script"]["sha256"],
                ),
            }
            previous_perf = previous["inputs"]["perf_tools_repo"]
            current_perf = manifest["inputs"]["perf_tools_repo"]
            if "hlo_inventory" in previous_perf:
                checks["HLO corpus"] = (
                    previous_perf["hlo_inventory"],
                    current_perf["hlo_inventory"],
                )
            else:
                if "hlo_status" not in previous_perf:
                    raise ValueError(
                        "cannot safely resume a legacy campaign without recorded "
                        "HLO cleanliness; start a new campaign"
                    )
                previous_hlo_status = previous_perf.get("hlo_status", "")
                if previous_hlo_status:
                    raise ValueError(
                        "cannot safely resume a legacy campaign that started with "
                        "modified or untracked HLO inputs; start a new campaign. "
                        f"Recorded HLO status:\n{previous_hlo_status}"
                    )
                legacy_changes = legacy_hlo_changes(
                    perf_repo,
                    hlo_root,
                    previous_perf["commit"],
                    current_perf["commit"],
                )
                checks["HLO corpus"] = (
                    ("", ""),
                    (legacy_changes, current_perf.get("hlo_status", "")),
                )
            changed = [name for name, values in checks.items() if values[0] != values[1]]
            if changed:
                raise ValueError(
                    "cannot resume because campaign inputs changed: "
                    + ", ".join(changed)
                )
            current_plan = manifest
            manifest = previous
            manifest["inputs"]["perf_tools_repo"]["hlo_inventory"] = (
                current_perf["hlo_inventory"]
            )
            manifest["targets"] = campaign_targets
            manifest["active_refs"] = refs
            manifest["current_refs_file"] = current_plan["inputs"]["refs_file"]
            manifest.setdefault("target_selection_history", []).append(
                {
                    "selected_at": utc_now(),
                    "active_refs": refs,
                    "added_refs": added_refs,
                    "removed_refs": removed_refs,
                }
            )
            source_original_state = manifest.get(
                "source_original_state", source_original_state
            )
            manifest["status"] = "running"
            manifest.setdefault("resume_history", []).append(
                {"resumed_at": utc_now(), "environment": collect_environment()}
            )
        baseline_ref = profile.get("reference", {}).get("xla_ref")
        if baseline_ref:
            comparison_refs = list(refs)
            if baseline_ref not in comparison_refs:
                comparison_refs.insert(0, baseline_ref)
            manifest["comparison_refs"] = comparison_refs
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        campaign_manifest_written = True

        expected_hlo_inventory = manifest["inputs"]["perf_tools_repo"]["hlo_inventory"]
        results_by_ref = {
            result["ref"]: result for result in manifest.get("results", [])
        }
        restore_source = True
        for target in targets:
            require_unchanged_hlo_inventory(
                hlo_root,
                expected_hlo_inventory,
                f"before evaluating {target['ref']}",
            )
            result = evaluate_target(
                target=target,
                source_repo=source_repo,
                output_dir=output_dir,
                bazel=bazel,
                eval_script=eval_script,
                hlo_root=hlo_root,
                benchmark=benchmark,
                resume=args.resume,
            )
            results_by_ref[target["ref"]] = result
            manifest["results"] = [
                results_by_ref[item["ref"]]
                for item in campaign_targets
                if item["ref"] in results_by_ref
            ]
            write_json(manifest_path, manifest)

        results = [results_by_ref[target["ref"]] for target in targets]
        failed = [result for result in results if result["status"] != "completed"]
        comparison_failed = False
        if baseline_ref:
            try:
                require_unchanged_hlo_inventory(
                    hlo_root,
                    expected_hlo_inventory,
                    "before comparison",
                )
                comparison_targets = select_comparison_targets(
                    manifest, baseline_ref
                )
                manifest["comparison"] = write_comparison(
                    output_dir=output_dir,
                    targets=comparison_targets,
                    baseline_ref=baseline_ref,
                )
                comparison_failed = (
                    manifest["comparison"]["validation"]["status"] != "passed"
                )
                if comparison_failed:
                    print("error: comparison validation failed", file=sys.stderr)
            except (KeyError, OSError, TypeError, ValueError) as error:
                comparison_failed = True
                manifest["comparison"] = {"error": str(error)}
                print(f"error: comparison generation failed: {error}", file=sys.stderr)
        manifest["status"] = (
            "completed"
            if not failed and not comparison_failed
            else "completed_with_failures"
        )
        manifest["finished_at"] = utc_now()
        manifest["summary"] = {
            "total": len(results),
            "completed": len(results) - len(failed),
            "failed": len(failed),
            "comparison_failed": comparison_failed,
        }
        write_json(manifest_path, manifest)
        return 0 if not failed and not comparison_failed else 1
    except CampaignInterrupted as error:
        if manifest is not None:
            manifest["status"] = "interrupted"
            manifest["interrupted_at"] = utc_now()
            manifest["signal"] = error.signum
            if manifest_path is not None and manifest_path.is_file():
                write_json(manifest_path, manifest)
        print(f"interrupted by signal {error.signum}", file=sys.stderr)
        return 128 + error.signum
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        if (
            campaign_manifest_written
            and manifest is not None
            and manifest_path is not None
        ):
            try:
                manifest["status"] = "error"
                manifest["error"] = str(error)
                manifest["failed_at"] = utc_now()
                write_json(manifest_path, manifest)
            except OSError as write_error:
                print(
                    f"error: failed to checkpoint campaign error: {write_error}",
                    file=sys.stderr,
                )
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        previous_signal_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, handled_signals
        )
        finalization_failed = False
        if restore_source and source_repo is not None and source_original_state is not None:
            try:
                restore_source_checkout(source_repo, source_original_state)
                restore_result = {
                    "status": "restored",
                    "restored_at": utc_now(),
                    **source_checkout_state(source_repo),
                }
            except (OSError, RuntimeError, ValueError) as error:
                finalization_failed = True
                restore_result = {
                    "status": "failed",
                    "attempted_at": utc_now(),
                    "error": str(error),
                }
                print(
                    "error: failed to restore the original XLA checkout: "
                    f"{error}",
                    file=sys.stderr,
                )
            if manifest is not None and manifest_path is not None and manifest_path.is_file():
                try:
                    manifest["source_restore"] = restore_result
                    write_json(manifest_path, manifest)
                except OSError as error:
                    finalization_failed = True
                    print(
                        f"error: failed to record source restore status: {error}",
                        file=sys.stderr,
                    )
        if source_lock is not None:
            try:
                release_source_lock(source_lock)
            except OSError as error:
                finalization_failed = True
                print(f"error: failed to release source lock: {error}", file=sys.stderr)
        for signum, previous_handler in previous_signal_handlers.items():
            signal.signal(signum, previous_handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
        if finalization_failed:
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
