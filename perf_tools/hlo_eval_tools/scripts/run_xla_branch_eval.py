#!/usr/bin/env python3
"""Build and benchmark hlo_runner_main across a list of XLA Git refs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare_hlo_branch_results import write_comparison


UPSTREAM_URL = "https://github.com/openxla/xla.git"
RUNNER_TARGET = "//xla/tools/multihost_hlo_runner:hlo_runner_main"
RUNNER_RELATIVE_PATH = Path(
    "xla/tools/multihost_hlo_runner/hlo_runner_main"
)


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
        help="Bazelisk/Bazel executable (default: bazelisk if available, else bazel)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="use currently available remote-tracking refs without fetching",
    )
    parser.add_argument(
        "--keep-worktrees",
        action="store_true",
        help="retain temporary source worktrees and Bazel output for debugging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve refs and print the plan without building or evaluating",
    )
    return parser.parse_args()


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_capture(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command_text(command)}\n"
            f"{result.stdout.strip()}"
        )
    return result.stdout.strip()


def run_logged(
    command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {command_text(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        log.write(f"\n[exit_code={result.returncode}]\n")
    return result.returncode


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


def choose_bazel(requested: str | None) -> str:
    if requested:
        executable = shutil.which(requested)
        if executable is None:
            raise ValueError(f"Bazel executable not found: {requested}")
        return executable
    for candidate in ("bazelisk", "bazel"):
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise ValueError("neither bazelisk nor bazel is available in PATH")


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


def repository_metadata(repo: Path, hlo_root: Path | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(repo),
        "commit": git(repo, "rev-parse", "HEAD"),
        "status": git(repo, "status", "--short"),
    }
    if hlo_root is not None:
        relative = hlo_root.relative_to(repo)
        metadata["hlo_tree"] = git(repo, "rev-parse", f"HEAD:{relative.as_posix()}")
        metadata["hlo_status"] = git(
            repo,
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            f":(glob){relative.as_posix()}/*/*/training/*.txt",
            f":(glob){relative.as_posix()}/*/*/training/*.hlo",
            f":(glob){relative.as_posix()}/*/*/inference/*gpu/*.txt",
            f":(glob){relative.as_posix()}/*/*/inference/*gpu/*.hlo",
        )
    return metadata


def branch_metadata_path(branch_dir: Path) -> Path:
    return branch_dir / "metadata.json"


def evaluate_target(
    *,
    target: dict[str, str],
    source_repo: Path,
    output_dir: Path,
    work_root: Path,
    bazel: str,
    eval_script: Path,
    hlo_root: Path,
    benchmark: dict[str, Any],
    resume: bool,
    keep_worktrees: bool,
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

    target_work_root = work_root / target["slug"]
    worktree = target_work_root / "source"
    output_base = target_work_root / "bazel-output"
    worktree_added = False

    try:
        if not (resume and runner_copy.is_file() and os.access(runner_copy, os.X_OK)):
            target_work_root.mkdir(parents=True, exist_ok=True)
            git(
                source_repo,
                "worktree",
                "add",
                "--detach",
                str(worktree),
                target["commit"],
            )
            worktree_added = True

            expected_bazel = (worktree / ".bazelversion").read_text(
                encoding="utf-8"
            ).strip()
            actual_bazel = bazel_version(bazel, worktree)
            metadata["bazel"] = {
                "command": bazel,
                "expected_version": expected_bazel,
                "actual_version": actual_bazel,
            }
            write_json(metadata_file, metadata)
            if expected_bazel != actual_bazel:
                raise RuntimeError(
                    f"{target['ref']} requires Bazel {expected_bazel}, but {bazel} "
                    f"reports {actual_bazel}; install Bazelisk or pass "
                    f"--bazel-command with a compatible launcher"
                )

            build_command = [
                bazel,
                f"--output_base={output_base}",
                "build",
                "-c",
                "opt",
                "--config=rocm",
                RUNNER_TARGET,
            ]
            metadata["build_command"] = build_command
            write_json(metadata_file, metadata)
            print(f"[{target['ref']}] building {RUNNER_TARGET}")
            build_rc = run_logged(build_command, cwd=worktree, log_path=build_log)
            metadata["build_exit_code"] = build_rc
            if build_rc != 0:
                metadata["status"] = "build_failed"
                metadata["finished_at"] = utc_now()
                write_json(metadata_file, metadata)
                return metadata

            bazel_bin = Path(
                run_capture(
                    [bazel, f"--output_base={output_base}", "info", "bazel-bin"],
                    cwd=worktree,
                ).splitlines()[-1]
            )
            built_runner = bazel_bin / RUNNER_RELATIVE_PATH
            if not built_runner.is_file():
                raise RuntimeError(f"built runner not found: {built_runner}")
            runner_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_runner, runner_copy)
            runner_copy.chmod(runner_copy.stat().st_mode | 0o111)
        else:
            metadata["runner_reused"] = True

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
    finally:
        if worktree_added and not keep_worktrees:
            git(
                source_repo,
                "worktree",
                "remove",
                "--force",
                str(worktree),
                check=False,
            )
        if not keep_worktrees:
            shutil.rmtree(target_work_root, ignore_errors=True)


def main() -> int:
    args = parse_args()
    try:
        perf_repo = validate_git_root(
            args.perf_tools_repo or discover_perf_tools_repo(),
            "perf-tools repo",
        )
        source_repo = validate_git_root(args.xla_source_repo, "XLA source repo")
        if perf_repo == source_repo:
            raise ValueError("perf-tools repo and XLA source repo must be different checkouts")

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

        refs = read_refs(refs_file)
        profile, benchmark = load_profile(profile_file, args)
        ensure_and_fetch_remotes(source_repo, refs, args.skip_fetch)
        targets = resolve_refs(source_repo, refs)
        bazel = choose_bazel(args.bazel_command)

        manifest: dict[str, Any] = {
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
            "targets": targets,
        }

        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.is_relative_to(perf_repo) or output_dir.is_relative_to(source_repo):
            raise ValueError(
                "output directory must be outside both Git repositories: "
                f"{output_dir}"
            )
        if args.dry_run:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        manifest_path = output_dir / "manifest.json"
        if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
            raise ValueError(
                f"output directory is not empty: {output_dir}; choose a new directory "
                f"or pass --resume"
            )
        if args.resume and output_dir.exists() and any(output_dir.iterdir()):
            if not manifest_path.is_file():
                raise ValueError(
                    f"cannot resume without an existing manifest: {manifest_path}"
                )
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            checks = {
                "refs file": (
                    previous["inputs"]["refs_file"]["sha256"],
                    manifest["inputs"]["refs_file"]["sha256"],
                ),
                "benchmark profile": (
                    previous["inputs"]["profile_file"]["sha256"],
                    manifest["inputs"]["profile_file"]["sha256"],
                ),
                "effective benchmark settings": (
                    previous["benchmark"]["effective"],
                    manifest["benchmark"]["effective"],
                ),
                "HLO corpus": (
                    (
                        previous["inputs"]["perf_tools_repo"]["hlo_tree"],
                        previous["inputs"]["perf_tools_repo"]["hlo_status"],
                    ),
                    (
                        manifest["inputs"]["perf_tools_repo"]["hlo_tree"],
                        manifest["inputs"]["perf_tools_repo"]["hlo_status"],
                    ),
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
            changed = [name for name, values in checks.items() if values[0] != values[1]]
            if changed:
                raise ValueError(
                    "cannot resume because campaign inputs changed: "
                    + ", ".join(changed)
                )
            targets = previous["targets"]
            manifest = previous
            manifest["status"] = "running"
            manifest.setdefault("resume_history", []).append(
                {"resumed_at": utc_now(), "environment": collect_environment()}
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)

        work_root = output_dir / ".work"
        results_by_ref = {
            result["ref"]: result for result in manifest.get("results", [])
        }
        for target in targets:
            result = evaluate_target(
                target=target,
                source_repo=source_repo,
                output_dir=output_dir,
                work_root=work_root,
                bazel=bazel,
                eval_script=eval_script,
                hlo_root=hlo_root,
                benchmark=benchmark,
                resume=args.resume,
                keep_worktrees=args.keep_worktrees,
            )
            results_by_ref[target["ref"]] = result
            manifest["results"] = [
                results_by_ref[item["ref"]]
                for item in targets
                if item["ref"] in results_by_ref
            ]
            write_json(manifest_path, manifest)

        if not args.keep_worktrees:
            shutil.rmtree(work_root, ignore_errors=True)
        results = manifest["results"]
        failed = [result for result in results if result["status"] != "completed"]
        comparison_failed = False
        baseline_ref = profile.get("reference", {}).get("xla_ref")
        if baseline_ref:
            try:
                manifest["comparison"] = write_comparison(
                    output_dir=output_dir,
                    targets=targets,
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
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
