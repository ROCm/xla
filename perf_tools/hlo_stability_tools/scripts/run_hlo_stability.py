#!/usr/bin/env python3
"""Build or reuse runners and collect balanced repeated one-HLO evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_hlo_stability import analyze, write_outputs
from hlo_stability import (
    build_stability_plan,
    load_json_object,
    load_target_specs,
    order_cycle_for_roles,
    orders_for_rounds,
    read_result,
    resolve_runner_bundle_targets,
    validate_runner,
)
from file_util import sha256_file
from render_hlo_stability_report import write_stability_report
import xla_runner_bundle


ACTIVE_PROCESS: subprocess.Popen[str] | None = None
# Fixed to the checked-in reference protocol. Statistical repetition is rounds.
STABILITY_NUM_REPEATS = 2
TOOLING_FILES = (
    "run_hlo_stability.py",
    "xla_runner_bundle.py",
    "hlo_stability.py",
    "analyze_hlo_stability.py",
    "render_hlo_stability_report.py",
    "file_util.py",
)


class CollectionInterrupted(KeyboardInterrupt):
    def __init__(self, signum: int, previous_mask: Any = None):
        super().__init__(f"received signal {signum}")
        self.signum = signum
        self.previous_mask = previous_mask


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be an integer"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    runner_source = parser.add_mutually_exclusive_group(required=True)
    runner_source.add_argument(
        "--xla-source-repo",
        type=Path,
        help="clean dedicated XLA checkout used to build runners",
    )
    runner_source.add_argument(
        "--runner-bundle",
        type=Path,
        help="reuse a completed bundle produced by this stability tool",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new output directory for runners, raw evidence, and report",
    )
    parser.add_argument(
        "--hlo-path",
        required=True,
        type=Path,
        help="exactly one .txt/.hlo module to evaluate",
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        help=(
            "schema-v1 target JSON; default in build mode: "
            "configs/xla_targets.json; optional selector in reuse mode"
        ),
    )
    parser.add_argument(
        "--bazel-command",
        help="Bazel executable used in build mode (default: bazel/Bazelisk)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="build mode: use locally available refs without fetching",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        help=(
            "optional historical timing for the selected HLO; comparison "
            "context only, never a pass/fail threshold"
        ),
    )
    parser.add_argument(
        "--rounds",
        type=positive_int,
        default=12,
        help="balanced measured rounds (default: 12; use 24 to confirm)",
    )
    parser.add_argument(
        "--warmup-cooldown-sec",
        type=nonnegative_float,
        default=8,
        help="idle time between unrecorded per-target warmups (default: 8)",
    )
    parser.add_argument(
        "--target-cooldown-sec",
        type=nonnegative_float,
        default=8,
        help="idle time between target evaluations in a round (default: 8)",
    )
    parser.add_argument(
        "--round-cooldown-sec",
        type=nonnegative_float,
        default=30,
        help="idle time between complete balanced rounds (default: 30)",
    )
    parser.add_argument(
        "--runner-settle-sec",
        type=nonnegative_int,
        default=2,
        help=(
            "integer seconds after each runner process for GPU resource "
            "cleanup (default: 2)"
        ),
    )
    parser.add_argument(
        "--capture-system-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "record optional point-in-time ROCm/host context before and after "
            "measurements; not used in analysis and not a profiler"
        ),
    )
    parser.add_argument("--modified-z-threshold", type=positive_float, default=3.5)
    parser.add_argument("--minimum-outlier-percent", type=nonnegative_float, default=2)
    parser.add_argument("--temporal-drift-percent", type=nonnegative_float, default=2)
    parser.add_argument(
        "--reporting-threshold-percent", type=nonnegative_float, default=2
    )
    parser.add_argument("--minimum-paired-rounds", type=positive_int, default=3)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def signal_process_group(process: subprocess.Popen[str], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def handle_collection_signal(signum: int, _frame: Any) -> None:
    xla_runner_bundle.signal_active_process(signum)
    if ACTIVE_PROCESS is not None:
        signal_process_group(ACTIVE_PROCESS, signum)
    if xla_runner_bundle.bundle_finalization_active():
        xla_runner_bundle.defer_finalization_signal(signum)
        return
    previous_mask = None
    if hasattr(signal, "pthread_sigmask"):
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            {
                signal.SIGINT,
                signal.SIGTERM,
                getattr(signal, "SIGHUP", signal.SIGTERM),
            },
        )
    raise CollectionInterrupted(signum, previous_mask)


def spawn_tracked_process(
    command: list[str],
    **kwargs: Any,
) -> subprocess.Popen[str]:
    global ACTIVE_PROCESS
    signals = {
        signal.SIGINT,
        signal.SIGTERM,
        getattr(signal, "SIGHUP", signal.SIGTERM),
    }
    previous_mask = None
    if hasattr(signal, "pthread_sigmask"):
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    try:
        if previous_mask is not None:
            kwargs["preexec_fn"] = lambda: signal.pthread_sigmask(
                signal.SIG_SETMASK, previous_mask
            )
        process = subprocess.Popen(
            command,
            start_new_session=True,
            **kwargs,
        )
        ACTIVE_PROCESS = process
        setattr(process, "_hlo_previous_signal_mask", previous_mask)
        return process
    except BaseException:
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise


def restore_tracked_process_signal_mask(
    process: subprocess.Popen[str],
) -> None:
    previous_mask = getattr(
        process, "_hlo_previous_signal_mask", None
    )
    setattr(process, "_hlo_previous_signal_mask", None)
    if previous_mask is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def finish_tracked_process(
    process: subprocess.Popen[str],
    *,
    send_term: bool,
) -> None:
    previous_mask = None
    if hasattr(signal, "pthread_sigmask"):
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            {
                signal.SIGINT,
                signal.SIGTERM,
                getattr(signal, "SIGHUP", signal.SIGTERM),
            },
        )
    try:
        if send_term:
            signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            process.wait()
    finally:
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def wait_with_heartbeat(
    process: subprocess.Popen[str],
    *,
    progress_label: str,
    log_path: Path,
    heartbeat_sec: int = 30,
) -> int:
    started = time.monotonic()
    while True:
        try:
            return process.wait(timeout=heartbeat_sec)
        except subprocess.TimeoutExpired:
            elapsed = int(time.monotonic() - started)
            log_bytes = log_path.stat().st_size if log_path.exists() else 0
            print(
                f"[{utc_now()}] {progress_label} still running "
                f"({elapsed}s); log={log_path}; log_bytes={log_bytes}",
                flush=True,
            )


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def claim_output_directory(path: Path) -> Path:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError(f"output directory must be absent or empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    lock_path = path / "collection.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise ValueError(f"output directory is already claimed: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "claimed_at": utc_now(),
                "status": "collecting",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return lock_path


def update_collection_lock(path: Path, status: str) -> None:
    write_json(
        path,
        {
            "pid": os.getpid(),
            "updated_at": utc_now(),
            "status": status,
        },
    )


def checkpoint_collection_state(
    *,
    metadata_path: Path,
    lock_path: Path,
    metadata: dict[str, Any],
    status: str,
    error: str | None = None,
) -> None:
    handled_signals = {
        signal.SIGINT,
        signal.SIGTERM,
        getattr(signal, "SIGHUP", signal.SIGTERM),
    }
    previous_mask = None
    if hasattr(signal, "pthread_sigmask"):
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, handled_signals
        )
    try:
        metadata["status"] = status
        timestamp_key = (
            "interrupted_at" if status == "interrupted" else "failed_at"
        )
        if status in {"interrupted", "failed"}:
            metadata[timestamp_key] = utc_now()
        if error is not None:
            metadata["error"] = error
        try:
            write_json(metadata_path, metadata)
        finally:
            update_collection_lock(lock_path, status)
    finally:
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def discover_repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_input(path: Path, base: Path) -> Path:
    value = path.expanduser()
    return (base / value).resolve() if not value.is_absolute() else value.resolve()


def validate_collection_options(args: argparse.Namespace) -> None:
    if args.rounds < 1 or args.minimum_paired_rounds < 1:
        raise ValueError("rounds and minimum paired rounds must be positive")
    if args.runner_bundle is not None and (
        args.bazel_command is not None or args.skip_fetch
    ):
        raise ValueError(
            "--bazel-command and --skip-fetch are valid only with "
            "--xla-source-repo"
        )
    for name in (
        "warmup_cooldown_sec",
        "target_cooldown_sec",
        "round_cooldown_sec",
        "runner_settle_sec",
        "minimum_outlier_percent",
        "temporal_drift_percent",
        "reporting_threshold_percent",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (
        type(args.runner_settle_sec) is not int
        or args.runner_settle_sec < 0
    ):
        raise ValueError("runner_settle_sec must be a nonnegative integer")
    if (
        not math.isfinite(float(args.modified_z_threshold))
        or args.modified_z_threshold <= 0
    ):
        raise ValueError("modified_z_threshold must be finite and positive")


def selected_hlo(path: Path) -> Path:
    if path.is_file():
        files = [path] if path.suffix in {".txt", ".hlo"} else []
    elif path.is_dir():
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix in {".txt", ".hlo"}
        )
    else:
        files = []
    if len(files) != 1:
        raise ValueError(
            f"stability collection requires exactly one .txt/.hlo file; "
            f"found {len(files)} under {path}"
        )
    return files[0]


def validated_orders(
    rounds: int,
    roles: tuple[str, ...],
) -> list[tuple[str, ...]]:
    cycle = order_cycle_for_roles(roles)
    if rounds < len(cycle) or rounds % len(cycle) != 0:
        raise ValueError(
            f"rounds must be a positive multiple of schedule cycle "
            f"{len(cycle)}; found {rounds}"
        )
    return orders_for_rounds(rounds, cycle)


def command_output(command: list[str], *, limit: int | None = None) -> str:
    try:
        output, return_code = xla_runner_bundle.run_capture_result(
            command,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"$ {' '.join(command)}\nERROR: {error}\n"
    lines = output.splitlines()
    if limit is not None:
        lines = lines[:limit]
    output = "\n".join(lines)
    return (
        f"$ {' '.join(command)}\n"
        f"{output}\n"
        f"[exit_code={return_code}]\n"
    )


def write_system_snapshot(path: Path) -> None:
    parts = [
        f"captured_at={utc_now()}\n",
        command_output(["uptime"]),
        command_output(
            [
                "rocm-smi",
                "--showuse",
                "--showmemuse",
                "--showclocks",
                "--showpower",
                "--showtemp",
            ]
        ),
        command_output(["rocm-smi", "--showpids"]),
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def safe_system_snapshot(path: Path) -> None:
    try:
        write_system_snapshot(path)
    except OSError as error:
        try:
            path.write_text(
                f"snapshot_error={error}\n", encoding="utf-8"
            )
        except OSError:
            print(
                "warning: system snapshot could not be written",
                file=sys.stderr,
                flush=True,
            )


def redact_log(path: Path, replacements: dict[str, str]) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if raw:
            text = text.replace(raw, replacement)
    path.write_text(text, encoding="utf-8")


def portable_error(error: BaseException, replacements: dict[str, str]) -> str:
    text = str(error)
    for raw, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if raw:
            text = text.replace(raw, replacement)
    return text


def evaluate_once(
    *,
    role: str,
    target: dict[str, Any],
    output: Path,
    eval_script: Path,
    eval_script_sha256: str,
    hlo_path: Path,
    hlo_sha256: str,
    num_repeats: int,
    runner_settle_sec: int,
    capture_snapshots: bool,
    evaluator_dependencies: dict[Path, str] | None = None,
) -> None:
    global ACTIVE_PROCESS
    csv_dir = output / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_path = output / "eval.log"
    runner = Path(str(target.get("runner", ""))).resolve()
    try:
        if capture_snapshots:
            safe_system_snapshot(output / "system_before.txt")
        runner = validate_runner(target)
        actual_eval_hash = sha256_file(eval_script)
        if actual_eval_hash != eval_script_sha256:
            raise ValueError(
                "evaluation script changed during collection: "
                f"expected={eval_script_sha256}, actual={actual_eval_hash}"
            )
        for dependency, expected_hash in (
            evaluator_dependencies or {}
        ).items():
            actual_hash = sha256_file(dependency)
            if actual_hash != expected_hash:
                raise ValueError(
                    "evaluator dependency changed during collection: "
                    f"{dependency.name}; expected={expected_hash}, "
                    f"actual={actual_hash}"
                )
        actual_hlo_hash = sha256_file(hlo_path)
        if actual_hlo_hash != hlo_sha256:
            raise ValueError(
                f"HLO input changed during collection: {hlo_path.name}; "
                f"expected={hlo_sha256}, actual={actual_hlo_hash}"
            )
        environment = os.environ.copy()
        environment.update(
            {
                "ARG_MODE": "uninitialized",
                "CMD_BUFFER": "off",
                "ORDER": "size",
                "SETTLE_SEC": str(runner_settle_sec),
                "RESUME": "0",
                "PROFILE_OUTPUT_MODE": "auto",
                "XLA_FLAGS": "",
            }
        )
        command = [
            str(eval_script),
            str(runner),
            str(hlo_path),
            str(csv_dir),
            str(num_repeats),
        ]
        print(
            f"[evaluate {role}] {target.get('label', role)}; "
            f"commit={target.get('commit')}; log={log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            log.write(
                f"$ run_hlo_eval.sh <runner:{role}> "
                f"{hlo_path.name} <csv-output> {num_repeats}\n"
            )
            log.flush()
            process = spawn_tracked_process(
                command,
                cwd=eval_script.parent,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                restore_tracked_process_signal_mask(process)
                return_code = wait_with_heartbeat(
                    process,
                    progress_label=f"evaluation {role}",
                    log_path=log_path,
                )
            except CollectionInterrupted:
                finish_tracked_process(process, send_term=False)
                raise
            except BaseException:
                finish_tracked_process(process, send_term=True)
                raise
            finally:
                restore_tracked_process_signal_mask(process)
                ACTIVE_PROCESS = None
            log.write(f"\n[exit_code={return_code}]\n")
        if return_code != 0:
            raise RuntimeError(
                f"{role} evaluation failed with exit code "
                f"{return_code}: {log_path.relative_to(output.parents[1])}"
            )
        files = sorted(csv_dir.glob("*.csv"))
        if len(files) != 1:
            raise ValueError(
                f"{role} evaluation expected one CSV; found {files}"
            )
        read_result(
            files[0],
            expected_module=hlo_path.name,
            require_single_row=True,
        )
        print(
            f"[evaluate {role}] completed; result={files[0]}",
            flush=True,
        )
    finally:
        replacements = {
            str(eval_script): "<eval-script>",
            str(runner): f"<runner:{role}>",
            str(hlo_path): f"<hlo:{hlo_path.name}>",
            str(csv_dir): "<csv-output>",
        }
        replacements.update(
            {
                str(path): f"<evaluator-dependency:{path.name}>"
                for path in (evaluator_dependencies or {})
            }
        )
        redact_log(
            log_path,
            replacements,
        )
        for legacy_log in csv_dir.glob("*.legacy.log"):
            redact_log(legacy_log, replacements)
        if capture_snapshots:
            safe_system_snapshot(output / "system_after.txt")


def collect(args: argparse.Namespace) -> dict[str, Any]:
    validate_collection_options(args)
    repo_root = discover_repository_root()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.is_relative_to(repo_root):
        raise ValueError(
            "output directory must be outside the tool repository"
        )
    hlo_path = selected_hlo(resolve_input(args.hlo_path, repo_root))
    hlo_sha256 = sha256_file(hlo_path)
    eval_script = repo_root / "perf_tools/hlo_eval_tools/run_hlo_eval.sh"
    if not eval_script.is_file() or not os.access(eval_script, os.X_OK):
        raise ValueError(f"evaluation script is missing or not executable: {eval_script}")
    eval_script_sha256 = sha256_file(eval_script)
    legacy_converter = (
        eval_script.parent / "scripts/legacy_profile_to_csv.py"
    )
    if not legacy_converter.is_file():
        raise ValueError(
            f"evaluator dependency is missing: {legacy_converter}"
        )
    evaluator_dependencies = {
        legacy_converter: sha256_file(legacy_converter)
    }
    reference_csv = (
        resolve_input(args.reference_csv, repo_root)
        if args.reference_csv
        else None
    )
    if reference_csv is not None and not reference_csv.is_file():
        raise ValueError(f"historical reference CSV is missing: {reference_csv}")
    if reference_csv is not None:
        read_result(reference_csv, expected_module=hlo_path.name)
    reference_sha256 = sha256_file(reference_csv) if reference_csv else None
    script_dir = Path(__file__).resolve().parent
    stability_root = script_dir.parent
    tooling = {
        name: {
            "relative_path": str(
                (script_dir / name).relative_to(repo_root)
            ),
            "sha256": sha256_file(script_dir / name),
        }
        for name in TOOLING_FILES
    }
    repository_identity = {
        "commit": xla_runner_bundle.git(repo_root, "rev-parse", "HEAD"),
        "dirty": bool(
            xla_runner_bundle.git(
                repo_root,
                "status",
                "--porcelain",
                "--untracked-files=all",
            )
        ),
    }
    build_mode = args.xla_source_repo is not None
    selector = (
        resolve_input(args.targets_file, repo_root)
        if args.targets_file
        else stability_root / "configs/xla_targets.json"
        if build_mode
        else None
    )
    if selector is not None and not selector.is_file():
        raise ValueError(f"target file is missing: {selector}")
    preflight_specs = load_target_specs(selector) if selector else None
    source_repo = (
        args.xla_source_repo.expanduser().resolve()
        if build_mode
        else None
    )
    source_dir = (
        output_dir / "runner_bundle"
        if build_mode
        else resolve_input(args.runner_bundle, repo_root)
    )
    if source_repo is not None and source_repo == repo_root:
        raise ValueError(
            "XLA source repo and tool repo must be different checkouts"
        )
    if not build_mode:
        if output_dir.is_relative_to(source_dir):
            raise ValueError(
                "output directory must be outside the reused runner bundle"
            )
        if not (source_dir / "manifest.json").is_file():
            raise ValueError(
                f"runner bundle has no manifest.json: {source_dir}"
            )
        preflight_manifest = load_json_object(source_dir / "manifest.json")
        preflight_targets = resolve_runner_bundle_targets(
            source_dir, preflight_manifest, preflight_specs
        )
        validated_orders(args.rounds, tuple(preflight_targets))
    elif preflight_specs is not None:
        preflight_roles = tuple(
            ["control"]
            + [
                f"candidate_{index}"
                for index in range(1, len(preflight_specs) + 1)
            ]
        )
        validated_orders(args.rounds, preflight_roles)
    if source_repo is not None and output_dir.is_relative_to(source_repo):
        raise ValueError(
            "output directory must be outside the XLA source repository"
        )
    lock_path = claim_output_directory(output_dir)
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "created_at": utc_now(),
        "status": "preparing_runners" if build_mode else "validating_runners",
        "tooling": tooling,
        "repository": repository_identity,
        "collection": {
            "hlo_input": {
                "file_name": hlo_path.name,
                "sha256": hlo_sha256,
            },
            "evaluation_script": {
                "relative_path": str(eval_script.relative_to(repo_root)),
                "sha256": eval_script_sha256,
            },
            "evaluation_dependencies": [
                {
                    "relative_path": str(path.relative_to(repo_root)),
                    "sha256": digest,
                }
                for path, digest in evaluator_dependencies.items()
            ],
            "reference_csv": (
                {
                    "file_name": reference_csv.name,
                    "sha256": reference_sha256,
                }
                if reference_csv
                else None
            ),
            "rounds": args.rounds,
            "num_repeats": STABILITY_NUM_REPEATS,
            "runner_policy": {
                "reference_aligned": args.runner_settle_sec == 2,
                "fixed_num_repeats": STABILITY_NUM_REPEATS,
                "num_repeats_user_configurable": False,
                "argument_mode": "uninitialized",
                "command_buffer": "off",
                "order": "size",
                "runner_settle_sec": args.runner_settle_sec,
                "runner_settle_user_configurable": True,
            },
            "warmup_cooldown_sec": args.warmup_cooldown_sec,
            "target_cooldown_sec": args.target_cooldown_sec,
            "round_cooldown_sec": args.round_cooldown_sec,
            "runner_settle_sec": args.runner_settle_sec,
            "capture_system_snapshots": args.capture_system_snapshots,
            "system_snapshots_role": (
                "optional point-in-time diagnostic context; not analyzed "
                "and not a profiler"
            ),
            "runner_source_mode": "built" if build_mode else "reused",
            "resume_supported": False,
        },
    }
    metadata_path = output_dir / "experiment_metadata.json"
    try:
        write_json(metadata_path, metadata)
    except BaseException as error:
        checkpoint_collection_state(
            metadata_path=metadata_path,
            lock_path=lock_path,
            metadata=metadata,
            status=(
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed"
            ),
            error=str(error),
        )
        raise

    try:
        if build_mode:
            if source_repo is None or selector is None:
                raise RuntimeError("build mode inputs were not resolved")
            print(
                "Preparing immutable XLA runner bundle...",
                flush=True,
            )
            manifest_path, manifest = xla_runner_bundle.prepare_runner_bundle(
                source_repo=source_repo,
                bundle_dir=source_dir,
                targets_file=selector,
                profile_file=stability_root / "configs/stability_profile.json",
                bazel_command=args.bazel_command,
                skip_fetch=args.skip_fetch,
            )
            deferred_signal = (
                xla_runner_bundle.consume_deferred_finalization_signal()
            )
            if deferred_signal is not None:
                raise CollectionInterrupted(deferred_signal)
        else:
            source_manifest_path = source_dir / "manifest.json"
            manifest_path = output_dir / "runner_source_manifest.json"
            shutil.copy2(source_manifest_path, manifest_path)
            manifest = load_json_object(manifest_path)
        specs = load_target_specs(selector) if selector else None
        if (
            build_mode
            and specs is not None
            and manifest.get("target_specs") != specs
        ):
            raise ValueError("target file changed during runner preparation")
        targets = resolve_runner_bundle_targets(
            source_dir, manifest, specs
        )
        orders = validated_orders(args.rounds, tuple(targets))
        plan = build_stability_plan(
            bundle_dir=source_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            targets=targets,
            rounds=args.rounds,
            target_cooldown_sec=args.target_cooldown_sec,
            round_cooldown_sec=args.round_cooldown_sec,
            selection_file=selector,
            selection_specs=specs,
        )
        manifest_snapshot_path = manifest_path
        metadata = {
            **plan,
            "created_at": metadata["created_at"],
            "status": "collecting",
            "collection": metadata["collection"],
            "tooling": metadata["tooling"],
            "repository": metadata["repository"],
            "collection_environment": (
                xla_runner_bundle.collect_environment()
            ),
        }
        metadata["runner_source"]["mode"] = (
            "built" if build_mode else "reused"
        )
        metadata["runner_source"]["manifest_relative_path"] = str(
            manifest_snapshot_path.relative_to(output_dir)
        )
        write_json(metadata_path, metadata)
        print("Running one unrecorded warmup per target...", flush=True)
        roles = list(targets)
        for index, role in enumerate(roles):
            print(
                f"  warmup {index + 1}/{len(roles)} "
                f"{role}: {targets[role]['label']}",
                flush=True,
            )
            evaluate_once(
                role=role,
                target=targets[role],
                output=output_dir / "warmup" / role,
                eval_script=eval_script,
                eval_script_sha256=eval_script_sha256,
                hlo_path=hlo_path,
                hlo_sha256=hlo_sha256,
                num_repeats=STABILITY_NUM_REPEATS,
                runner_settle_sec=args.runner_settle_sec,
                capture_snapshots=False,
                evaluator_dependencies=evaluator_dependencies,
            )
            if index < len(roles) - 1:
                time.sleep(args.warmup_cooldown_sec)
        if args.round_cooldown_sec:
            time.sleep(args.round_cooldown_sec)

        order_path = output_dir / "round_orders.csv"
        order_path.write_text("round,execution_order\n", encoding="utf-8")
        for round_index, order in enumerate(orders, start=1):
            round_id = f"{round_index:02d}"
            with order_path.open("a", encoding="utf-8") as stream:
                stream.write(f"{round_id},{'>'.join(order)}\n")
            print(
                f"[round {round_id}/{args.rounds}] {'>'.join(order)}",
                flush=True,
            )
            for target_index, role in enumerate(order):
                print(
                    f"  target {target_index + 1}/{len(order)} "
                    f"{role}: {targets[role]['label']}",
                    flush=True,
                )
                evaluate_once(
                    role=role,
                    target=targets[role],
                    output=output_dir / role / f"round_{round_id}",
                    eval_script=eval_script,
                    eval_script_sha256=eval_script_sha256,
                    hlo_path=hlo_path,
                    hlo_sha256=hlo_sha256,
                    num_repeats=STABILITY_NUM_REPEATS,
                    runner_settle_sec=args.runner_settle_sec,
                    capture_snapshots=args.capture_system_snapshots,
                    evaluator_dependencies=evaluator_dependencies,
                )
                if target_index < len(order) - 1:
                    time.sleep(args.target_cooldown_sec)
            if round_index < len(orders):
                time.sleep(args.round_cooldown_sec)

        metadata["status"] = "collected"
        metadata["collected_at"] = utc_now()
        write_json(metadata_path, metadata)
        print("Collection complete; analyzing evidence...", flush=True)
        if reference_csv is not None and (
            sha256_file(reference_csv) != reference_sha256
        ):
            raise ValueError(
                f"historical reference CSV changed during collection: "
                f"{reference_csv.name}"
            )
        result = analyze(
            experiment_dir=output_dir,
            roles=list(targets),
            reference_csv=reference_csv,
            modified_z_threshold=args.modified_z_threshold,
            minimum_outlier_percent=args.minimum_outlier_percent,
            temporal_drift_percent=args.temporal_drift_percent,
            reporting_threshold_percent=args.reporting_threshold_percent,
            minimum_paired_rounds=args.minimum_paired_rounds,
        )
        write_outputs(output_dir, result)
        analysis_files = (
            "stability_analysis.json",
            "stability_summary.csv",
            "raw_rounds_long.csv",
            "paired_deltas.csv",
        )
        metadata["status"] = "analyzed"
        metadata["analysis_artifacts"] = {
            name: sha256_file(output_dir / name) for name in analysis_files
        }
        write_json(metadata_path, metadata)
        print("Analysis complete; rendering HTML report...", flush=True)
        report_path = write_stability_report(output_dir)
        metadata["status"] = "completed"
        metadata["finished_at"] = utc_now()
        metadata["outputs"] = {
            "analysis_json": "stability_analysis.json",
            "summary_csv": "stability_summary.csv",
            "raw_rounds_csv": "raw_rounds_long.csv",
            "paired_deltas_csv": "paired_deltas.csv",
            "html_report": str(report_path.relative_to(output_dir)),
            "html_report_sha256": sha256_file(report_path),
        }
        write_json(metadata_path, metadata)
        update_collection_lock(lock_path, "completed")
        return metadata
    except BaseException as error:
        deferred_signal = (
            xla_runner_bundle.consume_deferred_finalization_signal()
        )
        status = (
            "interrupted"
            if isinstance(error, KeyboardInterrupt)
            or deferred_signal is not None
            else "failed"
        )
        error_text = portable_error(
            error,
            {
                str(repo_root): "<repository-root>",
                str(source_repo) if source_repo else "": "<xla-source-repo>",
                str(source_dir): "<runner-source>",
                str(output_dir): "<output>",
                str(hlo_path): f"<hlo:{hlo_path.name}>",
                str(eval_script): "<eval-script>",
                str(reference_csv) if reference_csv else "": (
                    f"<reference:{reference_csv.name}>"
                    if reference_csv
                    else ""
                ),
            },
        )
        checkpoint_collection_state(
            metadata_path=metadata_path,
            lock_path=lock_path,
            metadata=metadata,
            status=status,
            error=error_text,
        )
        raise


def main() -> int:
    args = parse_args()
    handled_signals = tuple(
        signal_value
        for signal_value in (
            signal.SIGINT,
            signal.SIGTERM,
            getattr(signal, "SIGHUP", None),
        )
        if signal_value is not None
    )
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in handled_signals
    }
    for signum in handled_signals:
        signal.signal(signum, handle_collection_signal)
    interrupted_previous_mask = None
    try:
        metadata = collect(args)
    except CollectionInterrupted as error:
        interrupted_previous_mask = error.previous_mask
        print(
            "interrupted; partial evidence remains in the output directory",
            file=sys.stderr,
            flush=True,
        )
        return 128 + error.signum
    except KeyboardInterrupt:
        print(
            "interrupted; partial evidence remains in the output directory",
            file=sys.stderr,
            flush=True,
        )
        return 130
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
        if interrupted_previous_mask is not None:
            signal.pthread_sigmask(
                signal.SIG_SETMASK, interrupted_previous_mask
            )
    print("HLO stability evidence collection completed.", flush=True)
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    print(f"Status: {metadata['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
