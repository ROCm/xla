#!/usr/bin/env python3
"""Convert legacy multihost_hlo_runner timing output to its modern CSV format."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROFILE_RE = re.compile(
    r"^## Execution time, file=(?P<file>.*?) repeat=(?P<repeat>\d+) "
    r"duration=(?P<duration>\d+)ns\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-repeats", required=True, type=int)
    parser.add_argument("hlo_files", nargs="+", type=Path)
    return parser.parse_args()


def read_profiles(log_path: Path) -> dict[str, dict[int, int]]:
    profiles: dict[str, dict[int, int]] = defaultdict(dict)
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = PROFILE_RE.match(line.strip())
            if match:
                profiles[match["file"]][int(match["repeat"])] = int(
                    match["duration"]
                )
    return dict(profiles)


def find_file_profiles(
    profiles: dict[str, dict[int, int]], hlo_file: Path
) -> dict[int, int] | None:
    raw_path = str(hlo_file)
    if raw_path in profiles:
        return profiles[raw_path]

    basename_matches = [
        values for name, values in profiles.items() if Path(name).name == hlo_file.name
    ]
    return basename_matches[0] if len(basename_matches) == 1 else None


def averaged_ms(repeats: dict[int, int]) -> float:
    ordered = [duration for _, duration in sorted(repeats.items())]
    # Match CSVProfileTimeWriter: repeat zero is the warmup when multiple
    # execution profiles are available.
    measured = ordered[1:] if len(ordered) > 1 else ordered
    return sum(measured) / len(measured) / 1e6


def existing_header(path: Path) -> list[str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                return next(csv.reader([line]))
    return None


def append_csv(
    output: Path, hlo_files: list[Path], times_ms: dict[Path, float]
) -> None:
    completed = sorted((path for path in hlo_files if path in times_ms), key=str)
    header = ["Datetime", *(path.name for path in completed)]

    current_header = existing_header(output)
    if current_header is not None and current_header != header:
        raise ValueError(
            f"CSV header mismatch for {output}: "
            f"existing={current_header!r}, new={header!r}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="") as stream:
        if current_header is None:
            stream.write(",".join(header) + "\n")
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        values = "".join(f", {times_ms[path]:.4g}ms" for path in completed)
        stream.write(timestamp + values + "\n")


def main() -> int:
    args = parse_args()
    if args.num_repeats < 1:
        print("error: --num-repeats must be at least 1", file=sys.stderr)
        return 1

    profiles = read_profiles(args.log)
    times_ms: dict[Path, float] = {}
    missing: list[Path] = []
    incomplete: list[tuple[Path, list[int]]] = []
    expected_repeats = set(range(args.num_repeats))

    for hlo_file in args.hlo_files:
        repeats = find_file_profiles(profiles, hlo_file)
        if not repeats:
            missing.append(hlo_file)
            continue
        if set(repeats) != expected_repeats:
            incomplete.append((hlo_file, sorted(repeats)))
            continue
        times_ms[hlo_file] = averaged_ms(repeats)

    if missing or incomplete:
        if missing:
            print(
                f"error: {len(missing)} HLO file(s) have no execution profile "
                f"in {args.log}",
                file=sys.stderr,
            )
            for hlo_file in missing:
                print(f"  missing: {hlo_file}", file=sys.stderr)
        if incomplete:
            print(
                f"error: {len(incomplete)} HLO file(s) have incomplete repeat "
                f"profiles; expected {sorted(expected_repeats)}",
                file=sys.stderr,
            )
            for hlo_file, repeats in incomplete:
                print(f"  incomplete: {hlo_file}: repeats={repeats}", file=sys.stderr)
        return 1

    try:
        append_csv(args.output, args.hlo_files, times_ms)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
