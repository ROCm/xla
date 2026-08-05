#!/usr/bin/env python3
"""Show build and round progress for an HLO stability output directory."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUCCESS_STATUSES = {"completed", "analyzed"}
FAILURE_STATUSES = {"failed", "interrupted", "error"}
FINAL_STATUSES = SUCCESS_STATUSES | FAILURE_STATUSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=5.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def process_running(pid: object, root: Path | None = None) -> bool:
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    command_line = Path(f"/proc/{pid}/cmdline")
    if command_line.is_file():
        arguments = [
            item.decode("utf-8", errors="replace")
            for item in command_line.read_bytes().split(b"\0")
            if item
        ]
        text = " ".join(arguments)
        if "run_hlo_stability.py" not in text:
            return False
        if root is not None and "--output-dir" in arguments:
            index = arguments.index("--output-dir")
            if index + 1 >= len(arguments):
                return False
            recorded_output = Path(arguments[index + 1])
            if not recorded_output.is_absolute():
                process_cwd = Path(f"/proc/{pid}/cwd")
                try:
                    recorded_output = (
                        Path(os.readlink(process_cwd)) / recorded_output
                    )
                except OSError:
                    return False
            if recorded_output.resolve() != root.resolve():
                return False
    return True


def latest_log(root: Path, pattern: str) -> Path | None:
    logs = [path for path in root.glob(pattern) if path.is_file()]
    return max(logs, key=lambda path: path.stat().st_mtime) if logs else None


def status_lines(root: Path) -> tuple[list[str], bool]:
    lock = read_json(root / "collection.lock")
    experiment = read_json(root / "experiment_metadata.json")
    bundle = read_json(root / "runner_bundle/manifest.json")
    pid = lock.get("pid")
    running = process_running(pid, root)
    experiment_status = str(experiment.get("status", "not_started"))
    source_mode = experiment.get("runner_source", {}).get(
        "mode",
        experiment.get("collection", {}).get(
            "runner_source_mode", "unknown"
        ),
    )
    bundle_status = bundle.get("status")
    if bundle_status is None and source_mode == "reused":
        bundle_status = "reused external bundle"
    lines = [
        f"Updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Output: {root}",
        f"Process: pid={pid or '-'} running={running}",
        f"Experiment: {experiment_status}",
        f"Runner bundle: {bundle_status or 'not_started'}",
    ]

    metadata_paths = sorted(
        (root / "runner_bundle").glob("*/metadata.json")
    )
    for index, path in enumerate(metadata_paths, start=1):
        item = read_json(path)
        label = item.get("label", item.get("ref", path.parent.name))
        lines.append(
            f"  runner {index}: {label} — {item.get('status', 'unknown')}"
        )
        build_log = item.get("paths", {}).get("build_log")
        if isinstance(build_log, str) and build_log:
            lines.append(f"    build log: {build_log}")

    roles = experiment.get("design", {}).get("roles", [])
    rounds = experiment.get("design", {}).get("rounds")
    if isinstance(roles, list) and type(rounds) is int:
        for role in roles:
            completed = len(
                {
                    path.parents[1].name
                    for path in root.glob(f"{role}/round_*/csv/*.csv")
                }
            )
            lines.append(f"  rounds {role}: {completed}/{rounds}")

    active_build_log = latest_log(root, "runner_bundle/*/build.log")
    runtime_logs = [
        log
        for pattern in ("warmup/*/eval.log", "*/round_*/eval.log")
        for log in root.glob(pattern)
        if log.is_file()
    ]
    active_eval_log = (
        max(runtime_logs, key=lambda path: path.stat().st_mtime)
        if runtime_logs
        else None
    )
    if active_build_log is not None:
        lines.append(f"Latest build log: {active_build_log}")
    if active_eval_log is not None:
        lines.append(f"Latest runtime log: {active_eval_log}")

    report = root / "stability_report.html"
    lines.append(f"HTML report ready: {report.is_file()}")
    if experiment.get("error"):
        lines.append(f"Error: {experiment['error']}")

    finished = not running
    if finished and experiment_status not in FINAL_STATUSES:
        lines.append(
            "WARNING: collector process stopped before recording a final status"
        )
    return lines, finished


def main() -> int:
    args = parse_args()
    if args.interval_sec <= 0:
        raise SystemExit("--interval-sec must be positive")
    root = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"error: output directory does not exist: {root}", flush=True)
        return 2
    if not (
        (root / "experiment_metadata.json").is_file()
        or (root / "collection.lock").is_file()
    ):
        print(f"error: no stability metadata found under {root}", flush=True)
        return 2
    while True:
        lines, finished = status_lines(root)
        print("\n".join(lines), flush=True)
        if not args.follow or finished:
            experiment = read_json(root / "experiment_metadata.json")
            status = str(experiment.get("status", "not_started"))
            if status in SUCCESS_STATUSES:
                return 0
            if status in FAILURE_STATUSES:
                return 1
            return 2 if finished else 0
        print("-" * 72, flush=True)
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
