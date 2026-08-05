#!/usr/bin/env python3
"""Discover and validate checked-in HLO reference result CSVs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


HLO_SUFFIXES = {".txt", ".hlo"}
GPU_LEAF_RE = re.compile(r"^[0-9]+gpu$")
PROVENANCE_RE = re.compile(r"^#\s*([^:]+?)\s*:\s*(.*?)\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_hlo_files(hlo_path: Path) -> list[Path]:
    def is_workload_hlo(path: Path) -> bool:
        return path.suffix in HLO_SUFFIXES and (
            path.parent.name == "training"
            or (
                GPU_LEAF_RE.fullmatch(path.parent.name) is not None
                and path.parent.parent.name == "inference"
            )
        )

    if hlo_path.is_file():
        return [hlo_path] if is_workload_hlo(hlo_path) else []
    return sorted(
        path
        for path in hlo_path.rglob("*")
        if path.is_file() and is_workload_hlo(path)
    )


def workload_leaf(path: Path, tools_root: Path) -> tuple[Path, str]:
    relative = path.resolve().relative_to(tools_root.resolve())
    parts = relative.parts
    if len(parts) >= 4 and parts[2] == "training":
        leaf_parts = parts[:3]
        result_name = "training.csv"
    elif (
        len(parts) >= 5
        and parts[2] == "inference"
        and GPU_LEAF_RE.fullmatch(parts[3])
    ):
        leaf_parts = parts[:4]
        result_name = f"inference_{parts[3]}.csv"
    else:
        raise ValueError(
            f"selected HLO is not in a canonical workload leaf: {relative.as_posix()}"
        )
    return Path(*leaf_parts), result_name


def parse_provenance(path: Path) -> dict[str, str]:
    provenance: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            match = PROVENANCE_RE.match(line.rstrip())
            if match:
                key = re.sub(r"\s+", "_", match.group(1).strip().lower())
                provenance[key] = match.group(2).strip()
    return provenance


def validate_provenance(
    path: Path, provenance: dict[str, str], profile: dict[str, Any]
) -> None:
    reference = profile["reference"]
    runner = profile["runner"]
    required = ("xla_build", "gpu", "docker", "runner")
    missing = [key for key in required if not provenance.get(key)]
    if missing:
        raise ValueError(
            f"reference CSV has incomplete provenance ({', '.join(missing)}): {path}"
        )
    expected_commit = str(reference["xla_commit"])
    if expected_commit not in provenance["xla_build"]:
        raise ValueError(
            f"reference CSV XLA commit does not match {expected_commit}: {path}"
        )
    expected_provenance = {
        "gpu": reference["gpu"],
        "docker": reference["container"],
    }
    for key, expected in expected_provenance.items():
        if provenance[key] != str(expected):
            raise ValueError(
                f"reference CSV {key} is {provenance[key]!r}, expected "
                f"{expected!r}: {path}"
            )
    runner_text = provenance["runner"]
    expected_fragments = [
        f"--hlo_argument_mode={runner['arg_mode']}",
        f"--num_repeats={runner['num_repeats']}",
        "command buffers disabled"
        if runner["cmd_buffer"] == "off"
        else "command buffers enabled",
    ]
    absent = [fragment for fragment in expected_fragments if fragment not in runner_text]
    if absent:
        raise ValueError(
            f"reference CSV runner provenance is missing {absent}: {path}"
        )


def reference_inventory(
    hlo_path: Path, tools_root: Path, profile: dict[str, Any]
) -> dict[str, Any]:
    groups: dict[Path, dict[str, Any]] = {}
    for hlo_file in selected_hlo_files(hlo_path):
        leaf, result_name = workload_leaf(hlo_file, tools_root)
        entry = groups.setdefault(
            leaf,
            {
                "workload": leaf.as_posix().replace("/", "_") + ".csv",
                "leaf": leaf.as_posix(),
                "result_name": result_name,
                "modules": [],
            },
        )
        entry["modules"].append(hlo_file.name)

    digest = hashlib.sha256()
    workloads: list[dict[str, Any]] = []
    for leaf in sorted(groups, key=lambda item: item.as_posix()):
        entry = groups[leaf]
        modules = sorted(set(entry["modules"]))
        reference_path = (
            tools_root
            / leaf.parts[0]
            / leaf.parts[1]
            / "results"
            / entry["result_name"]
        )
        item: dict[str, Any] = {
            "workload": entry["workload"],
            "leaf": entry["leaf"],
            "modules": modules,
            "relative_path": reference_path.relative_to(tools_root).as_posix(),
            "exists": reference_path.is_file(),
        }
        if reference_path.is_file():
            provenance = parse_provenance(reference_path)
            validate_provenance(reference_path, provenance, profile)
            item["sha256"] = sha256_file(reference_path)
            item["provenance"] = provenance
        else:
            item["sha256"] = None
            item["provenance"] = None
        digest.update((item["workload"] + "\0").encode())
        digest.update(("\0".join(modules) + "\0").encode())
        digest.update(((item["sha256"] or "missing") + "\0").encode())
        workloads.append(item)

    if not workloads:
        raise ValueError(f"selected HLO path has no workload leaves: {hlo_path}")
    return {
        "schema_version": 1,
        "selected_hlo_path": (
            hlo_path.resolve().relative_to(tools_root.resolve()).as_posix()
        ),
        "workload_count": len(workloads),
        "available_count": sum(item["exists"] for item in workloads),
        "missing_count": sum(not item["exists"] for item in workloads),
        "sha256": digest.hexdigest(),
        "workloads": workloads,
    }
