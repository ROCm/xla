#!/usr/bin/env python3
"""Parse JSON, profile CSV, and workload paths for XLA HLO reports."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file and require an object at its root."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_latest_hlo_profile_timings_ms(path: Path) -> dict[str, float]:
    """Read the latest CSV row as HLO-module timings in milliseconds."""
    lines = [
        line
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line and not line.lstrip().startswith("#")
    ]
    if len(lines) < 2:
        return {}
    rows = list(csv.reader(lines))
    header = rows[0]
    values = rows[-1]
    if len(header) != len(values):
        return {}
    result: dict[str, float] = {}
    for module, raw in zip(header[1:], values[1:]):
        match = re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]*)?|[.][0-9]+)"
            r"(?:[eE]([+-]?[0-9]+))?ms\s*",
            raw,
        )
        if not match:
            continue
        value = float(match.group(1))
        if match.group(2):
            value *= 10 ** int(match.group(2))
        result[module] = value
    return result


def parse_workload_hierarchy(leaf: str) -> dict[str, str]:
    """Split a corpus leaf into domain, model, mode, and GPU configuration."""
    parts = leaf.split("/")
    if len(parts) < 3:
        return {
            "category": parts[0] if parts else "unknown",
            "model": parts[1] if len(parts) > 1 else "unknown",
            "mode": parts[2] if len(parts) > 2 else "unknown",
            "gpu": "unknown",
        }
    mode = parts[2]
    gpu = parts[3] if mode == "inference" and len(parts) > 3 else "training"
    return {
        "category": parts[0],
        "model": parts[1],
        "mode": mode,
        "gpu": gpu,
    }
