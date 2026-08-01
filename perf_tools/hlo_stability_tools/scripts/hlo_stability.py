#!/usr/bin/env python3
"""Shared manifest, schedule, and statistics helpers for HLO stability evidence."""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from file_util import sha256_file


TIME_UNITS_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
MIN_CANDIDATES = 1
MAX_CANDIDATES = 3
REQUIRED_TARGET_FIELDS = (
    "id",
    "role",
    "source_ref",
    "revision",
    "slug",
    "commit",
)
OrderCycle = tuple[tuple[str, ...], ...]


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_target_specs(path: Path) -> list[dict[str, Any]]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"target selector contains duplicate key: {key}")
            result[key] = item
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "targets"}
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or not isinstance(value.get("targets"), list)
        or not value["targets"]
    ):
        raise ValueError(f"invalid schema-v1 target selector: {path}")
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value["targets"]):
        if not isinstance(item, dict) or set(item) - {
            "revision",
            "commit",
            "label",
        }:
            raise ValueError(f"invalid target selector entry {index}: {item}")
        revision = item.get("revision")
        if (
            not isinstance(revision, str)
            or not revision
            or revision != revision.strip()
            or any(character.isspace() for character in revision)
            or revision.startswith("-")
        ):
            raise ValueError(f"target selector entry {index} has no revision")
        if revision in seen:
            raise ValueError(f"duplicate target selector revision: {revision}")
        commit = item.get("commit")
        if commit is not None and (
            not isinstance(commit, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", commit)
        ):
            raise ValueError(
                f"target selector entry {index} has an invalid commit"
            )
        label = item.get("label")
        if label is not None and (
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or len(label) > 128
            or any(ord(character) < 32 for character in label)
        ):
            raise ValueError(f"target selector entry {index} has an invalid label")
        spec: dict[str, Any] = {"revision": revision}
        if "commit" in item:
            spec["commit"] = commit.lower() if isinstance(commit, str) else None
        if isinstance(label, str):
            spec["label"] = label
        specs.append(spec)
        seen.add(revision)
    labels = [spec["label"] for spec in specs if "label" in spec]
    if len(labels) != len(set(labels)):
        raise ValueError("target selector contains duplicate labels")
    if not MIN_CANDIDATES <= len(specs) <= MAX_CANDIDATES:
        raise ValueError(
            "target selector must contain one to three candidates; "
            f"found {len(specs)}"
        )
    return specs


def target_label(target: dict[str, Any], fallback: str) -> str:
    label = target.get("label")
    if isinstance(label, str) and label:
        return label
    source_ref = str(target.get("source_ref", ""))
    for prefix in ("origin/rocm-jaxlib-", "origin/", "upstream/"):
        if source_ref.startswith(prefix):
            return source_ref[len(prefix) :]
    if re.fullmatch(r"[0-9a-fA-F]{40}", source_ref):
        return source_ref[:12]
    return source_ref or fallback


def validate_runner_bundle_manifest(manifest: dict[str, Any]) -> None:
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 2
    ):
        raise ValueError("stability requires a schema-v2 runner bundle")
    if manifest.get("kind") != "hlo_stability_runner_bundle":
        raise ValueError("runner bundle has an unsupported kind")
    if manifest.get("status") not in {"completed", "completed_with_failures"}:
        raise ValueError(
            f"runner bundle is not finalized: {manifest.get('status')!r}"
        )
    if (
        not isinstance(manifest.get("finished_at"), str)
        or not manifest["finished_at"]
    ):
        raise ValueError("runner bundle has no finished_at timestamp")
    source_original_state = manifest.get("source_original_state")
    if (
        not isinstance(source_original_state, dict)
        or not isinstance(source_original_state.get("commit"), str)
        or not isinstance(source_original_state.get("status"), str)
        or (
            source_original_state.get("branch") is not None
            and not isinstance(source_original_state.get("branch"), str)
        )
    ):
        raise ValueError(
            "runner bundle has no valid source_original_state"
        )
    source_restore = manifest.get("source_restore")
    if (
        not isinstance(source_restore, dict)
        or source_restore.get("status") != "restored"
    ):
        raise ValueError(
            "runner bundle source checkout was not successfully restored"
        )
    targets = manifest.get("targets")
    results = manifest.get("results")
    if not isinstance(targets, list) or not isinstance(results, list):
        raise ValueError("runner bundle has no valid target/result lists")
    target_ids = [
        target.get("id") for target in targets if isinstance(target, dict)
    ]
    result_ids = [
        result.get("id") for result in results if isinstance(result, dict)
    ]
    if len(target_ids) != len(targets) or any(
        not isinstance(target_id, str) or not target_id
        for target_id in target_ids
    ):
        raise ValueError("runner bundle contains invalid target IDs")
    if len(result_ids) != len(results) or any(
        not isinstance(result_id, str) or not result_id
        for result_id in result_ids
    ):
        raise ValueError("runner bundle contains invalid result IDs")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("runner bundle contains duplicate target IDs")
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("runner bundle contains duplicate result IDs")
    for index, target in enumerate(targets):
        _validate_target(target, f"runner bundle target {index}")
        if not re.fullmatch(r"[0-9a-f]{40}", target["commit"]):
            raise ValueError(
                f"runner bundle target {index} has an invalid commit"
            )
    live_control_id = manifest.get("live_control_id")
    all_controls = [
        target
        for target in targets
        if isinstance(target, dict)
        and target.get("role") == "live_control"
    ]
    if (
        len(all_controls) != 1
        or all_controls[0].get("id") != live_control_id
    ):
        raise ValueError("runner bundle live_control_id is invalid")


def _validate_target(target: dict[str, Any], label: str) -> None:
    for field in REQUIRED_TARGET_FIELDS:
        value = target.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} target has no valid {field}: {target}")
    slug = target["slug"]
    if Path(slug).name != slug or slug in {".", ".."}:
        raise ValueError(f"{label} target has unsafe slug {slug!r}")


def active_bundle_targets(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_runner_bundle_manifest(manifest)
    targets = manifest.get("targets")
    selected_ids = manifest.get("active_target_ids")
    if not isinstance(targets, list) or not isinstance(selected_ids, list):
        raise ValueError(
            "runner bundle has no valid targets/active_target_ids"
        )
    by_id = {
        target["id"]: target
        for target in targets
        if isinstance(target, dict) and isinstance(target.get("id"), str)
    }
    if any(not isinstance(target_id, str) for target_id in selected_ids):
        raise ValueError("runner bundle target IDs must be strings")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("runner bundle target IDs contain duplicates")
    missing = [target_id for target_id in selected_ids if target_id not in by_id]
    if missing:
        raise ValueError(
            "runner bundle targets are missing: " + ", ".join(missing)
        )
    return _validate_selected_targets([by_id[target_id] for target_id in selected_ids])


def selected_bundle_targets(
    manifest: dict[str, Any],
    target_specs: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if target_specs is None:
        return active_bundle_targets(manifest)
    validate_runner_bundle_manifest(manifest)
    targets = [
        target
        for target in manifest.get("targets", [])
        if isinstance(target, dict)
    ]
    live_control_id = manifest.get("live_control_id")
    controls = [
        target
        for target in targets
        if target.get("id") == live_control_id
        and target.get("role") == "live_control"
    ]
    if len(controls) != 1:
        raise ValueError("runner bundle must identify exactly one live control")
    candidates_by_ref: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        if target.get("role") == "candidate":
            candidates_by_ref.setdefault(
                str(target.get("source_ref", "")), []
            ).append(target)
    selected = []
    for spec in target_specs:
        matches = candidates_by_ref.get(spec["revision"], [])
        if len(matches) != 1:
            raise ValueError(
                f"expected one runner bundle candidate for {spec['revision']!r}; "
                f"found {len(matches)}"
            )
        target = dict(matches[0])
        configured_commit = spec.get("commit")
        if isinstance(configured_commit, str) and (
            target.get("commit") != configured_commit
        ):
            raise ValueError(
                f"runner bundle commit mismatch for {spec['revision']}: "
                f"requested={configured_commit}, recorded={target.get('commit')}"
            )
        if "label" in spec:
            target["label"] = spec["label"]
        selected.append(target)
    return _validate_selected_targets([controls[0], *selected])


def _validate_selected_targets(
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    controls = [target for target in selected if target.get("role") == "live_control"]
    candidates = [target for target in selected if target.get("role") == "candidate"]
    unsupported = [
        target
        for target in selected
        if target.get("role") not in {"live_control", "candidate"}
    ]
    if len(controls) != 1:
        raise ValueError(f"expected one live control; found {len(controls)}")
    if unsupported:
        raise ValueError(f"unsupported target roles: {unsupported}")
    if not MIN_CANDIDATES <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(
            "stability requires one to three candidates; "
            f"found {len(candidates)}"
        )
    return [controls[0], *candidates]


def resolve_runner_bundle_targets(
    bundle_dir: Path,
    manifest: dict[str, Any],
    target_specs: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    bundle_dir = bundle_dir.resolve()
    results_by_id = {
        result["id"]: result
        for result in manifest.get("results", [])
        if isinstance(result, dict) and isinstance(result.get("id"), str)
    }
    resolved: dict[str, dict[str, Any]] = {}
    for index, target in enumerate(
        selected_bundle_targets(manifest, target_specs)
    ):
        role = "control" if target["role"] == "live_control" else f"candidate_{index}"
        label = target_label(target, role)
        _validate_target(target, label)
        result = results_by_id.get(target["id"])
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise ValueError(f"target {target['id']} has no completed result")
        for field in REQUIRED_TARGET_FIELDS:
            if result.get(field) != target.get(field):
                raise ValueError(
                    f"target/result {field} mismatch for {target['id']}"
                )
        recorded_hash = result.get("runner_sha256")
        if not isinstance(recorded_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", recorded_hash
        ):
            raise ValueError(f"target {target['id']} has no valid runner SHA256")
        runner = (
            bundle_dir / target["slug"] / "runner" / "hlo_runner_main"
        ).resolve()
        if not runner.is_relative_to(bundle_dir):
            raise ValueError(f"runner resolves outside bundle: {runner}")
        if not runner.is_file() or not os.access(runner, os.X_OK):
            raise ValueError(f"runner is missing or not executable: {runner}")
        actual_hash = sha256_file(runner)
        if actual_hash != recorded_hash:
            raise ValueError(
                f"runner checksum mismatch for {target['id']}: "
                f"recorded={recorded_hash}, actual={actual_hash}"
            )
        result_paths = result.get("paths")
        recorded_runner = (
            result_paths.get("runner")
            if isinstance(result_paths, dict)
            else None
        )
        resolved[role] = {
            "target_id": target["id"],
            "manifest_role": target["role"],
            "label": label,
            "source_ref": target["source_ref"],
            "revision": target["revision"],
            "configured_commit": target.get("configured_commit"),
            "commit": target["commit"],
            "slug": target["slug"],
            "runner": str(runner),
            "runner_relative_path": str(runner.relative_to(bundle_dir)),
            "recorded_runner_path": recorded_runner,
            "runner_sha256": actual_hash,
        }
    for field in ("target_id", "slug", "runner"):
        values = [str(target[field]) for target in resolved.values()]
        if len(values) != len(set(values)):
            raise ValueError(f"selected targets contain duplicate {field}")
    return resolved


def validate_runner(target: dict[str, Any]) -> Path:
    runner = Path(str(target["runner"])).resolve()
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"runner is missing or not executable: {runner}")
    actual_hash = sha256_file(runner)
    if actual_hash != target.get("runner_sha256"):
        raise ValueError(
            f"runner changed after preflight for {target.get('target_id')}: "
            f"expected={target.get('runner_sha256')}, actual={actual_hash}"
        )
    return runner


def order_cycle_for_roles(roles: tuple[str, ...]) -> OrderCycle:
    if len(roles) == 2:
        first, second = roles
        return ((first, second), (second, first))
    if len(roles) == 3:
        return tuple(itertools.permutations(roles))
    if len(roles) == 4:
        first, second, third, fourth = roles
        return (
            (first, second, fourth, third),
            (second, third, first, fourth),
            (third, fourth, second, first),
            (fourth, first, third, second),
        )
    raise ValueError(f"stability schedule requires two to four roles: {roles}")


def orders_for_rounds(
    rounds: int, order_cycle: OrderCycle
) -> list[tuple[str, ...]]:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    return [
        order_cycle[index % len(order_cycle)] for index in range(rounds)
    ]


def build_stability_plan(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path | None = None,
    targets: dict[str, dict[str, Any]],
    rounds: int,
    target_cooldown_sec: float,
    round_cooldown_sec: float,
    selection_file: Path | None = None,
    selection_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path or bundle_dir / "manifest.json"
    if load_json_object(manifest_path) != manifest:
        raise ValueError(
            "runner bundle manifest does not match on-disk manifest"
        )
    if selection_file is not None and load_target_specs(selection_file) != selection_specs:
        raise ValueError("target selector object does not match on-disk selector")
    cycle = order_cycle_for_roles(tuple(targets))
    if rounds < len(cycle) or rounds % len(cycle) != 0:
        raise ValueError(
            f"rounds must be a positive multiple of schedule cycle "
            f"{len(cycle)}; found {rounds}"
        )
    complete_cycle = True
    portable_targets = {
        role: {
            key: value
            for key, value in target.items()
            if key not in {"runner", "recorded_runner_path"}
        }
        for role, target in targets.items()
    }
    source_target_file = None
    recorded_target_file = manifest.get("inputs", {}).get("targets_file")
    if isinstance(recorded_target_file, dict):
        source_target_file = {
            "file_name": Path(
                str(recorded_target_file.get("path", "targets"))
            ).name,
            "format": "json",
            "sha256": recorded_target_file.get("sha256"),
        }
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runner_source": {
            "kind": "runner_bundle",
            "directory_name": bundle_dir.resolve().name,
            "manifest_sha256": sha256_file(manifest_path),
        },
        "runner_source_manifest_sha256": sha256_file(manifest_path),
        "source_target_file": source_target_file,
        "source_target_specs": manifest.get("target_specs"),
        "stability_target_selection": {
            "source": (
                "explicit_xla_targets_json"
                if selection_file is not None
                else "runner_bundle_active_targets"
            ),
            "file_name": selection_file.name if selection_file else None,
            "sha256": sha256_file(selection_file) if selection_file else None,
            "targets": selection_specs,
        },
        "design": {
            "name": f"{len(targets)}_target_balanced_cycle",
            "roles": list(targets),
            "labels": {
                role: target["label"] for role, target in targets.items()
            },
            "target_count": len(targets),
            "candidate_count": len(targets) - 1,
            "rounds": rounds,
            "order_cycle": [list(order) for order in cycle],
            "order_cycle_length": len(cycle),
            "complete_design_cycles": rounds // len(cycle),
            "position_balanced": complete_cycle,
            "within_round_predecessor_balanced": complete_cycle,
            "cross_round_transitions_in_balance_claim": False,
            "balance_scope": (
                "execution positions and immediate predecessors within each "
                "complete cycle; cross-round transitions are excluded"
            ),
            "target_cooldown_sec": target_cooldown_sec,
            "round_cooldown_sec": round_cooldown_sec,
        },
        "targets": portable_targets,
    }


def parse_time_ms(value: str) -> float:
    text = value.strip()
    for unit, scale in TIME_UNITS_MS.items():
        if text.endswith(unit):
            parsed = float(text[: -len(unit)]) * scale
            if not math.isfinite(parsed) or parsed <= 0:
                raise ValueError(
                    f"timing must be finite and positive: {value!r}"
                )
            return parsed
    raise ValueError(f"unsupported timing value: {value!r}")


def read_result(
    path: Path,
    *,
    expected_module: str | None = None,
    require_single_row: bool = False,
) -> tuple[str, str, float]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    rows = list(csv.reader(lines))
    if len(rows) < 2:
        raise ValueError(f"CSV has no timing row: {path}")
    if require_single_row and len(rows) != 2:
        raise ValueError(
            f"CSV must contain exactly one timing row: {path}; "
            f"found {len(rows) - 1}"
        )
    header, latest = rows[0], rows[-1]
    if len(header) != len(latest):
        raise ValueError(f"CSV column mismatch: {path}")
    modules = header[1:]
    if len(modules) != 1:
        raise ValueError(f"expected one HLO module in {path}: {modules}")
    if expected_module is not None and modules[0] != expected_module:
        raise ValueError(
            f"CSV module differs from selected HLO: "
            f"expected={expected_module!r}, found={modules[0]!r}"
        )
    return modules[0], latest[0].strip(), parse_time_ms(latest[1])


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def basic_stats(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "stdev_ms": stdev,
        "cv_percent": stdev / mean * 100.0 if mean else 0.0,
        "min_ms": min(values),
        "p05_ms": percentile(values, 0.05),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def paired_percent_stats(
    values: list[float],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_percent": None,
            "median_percent": None,
            "stdev_percent": None,
            "mad_percent": None,
            "min_percent": None,
            "p05_percent": None,
            "p95_percent": None,
            "max_percent": None,
        }
    median = statistics.median(values)
    return {
        "count": len(values),
        "mean_percent": statistics.mean(values),
        "median_percent": median,
        "stdev_percent": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "mad_percent": statistics.median(
            abs(value - median) for value in values
        ),
        "min_percent": min(values),
        "p05_percent": percentile(values, 0.05),
        "p95_percent": percentile(values, 0.95),
        "max_percent": max(values),
    }


def round_sort_key(round_id: str) -> tuple[int, int | str]:
    return (0, int(round_id)) if round_id.isdigit() else (1, round_id)


def classify_outliers(
    samples: dict[str, float],
    *,
    modified_z_threshold: float,
    minimum_outlier_percent: float,
) -> tuple[dict[str, bool], dict[str, float]]:
    values = list(samples.values())
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    robust_cutoff = modified_z_threshold * 1.4826 * mad
    percentage_floor = abs(median) * minimum_outlier_percent / 100.0
    cutoff = max(robust_cutoff, percentage_floor)
    return (
        {
            round_id: abs(value - median) > cutoff
            for round_id, value in samples.items()
        },
        {
            "median_ms": median,
            "mad_ms": mad,
            "robust_cutoff_ms": robust_cutoff,
            "percentage_floor_ms": percentage_floor,
            "effective_cutoff_ms": cutoff,
        },
    )


def temporal_trend(
    samples: dict[str, float],
    flags: dict[str, bool],
    common_rounds: list[str],
    *,
    materiality_percent: float,
) -> dict[str, float | int | str | None]:
    split = len(common_rounds) // 2
    early = [
        samples[round_id]
        for round_id in common_rounds[:split]
        if not flags[round_id]
    ]
    late = [
        samples[round_id]
        for round_id in common_rounds[split:]
        if not flags[round_id]
    ]
    points = [
        (index + 1, samples[round_id])
        for index, round_id in enumerate(common_rounds)
        if not flags[round_id]
    ]
    enough_half_samples = len(early) >= 2 and len(late) >= 2
    early_median = (
        statistics.median(early) if enough_half_samples else None
    )
    late_median = (
        statistics.median(late) if enough_half_samples else None
    )
    delta = (
        (late_median / early_median - 1.0) * 100.0
        if early_median and late_median is not None
        else None
    )
    slope = None
    slope_percent = None
    if len(points) >= 2:
        mean_x = statistics.mean(index for index, _ in points)
        mean_y = statistics.mean(value for _, value in points)
        denominator = sum((index - mean_x) ** 2 for index, _ in points)
        if denominator:
            slope = sum(
                (index - mean_x) * (value - mean_y)
                for index, value in points
            ) / denominator
            median = statistics.median(value for _, value in points)
            slope_percent = slope * 10.0 / median * 100.0 if median else None
    if delta is None:
        verdict = "insufficient clean samples per scheduled half"
    elif delta >= materiality_percent:
        verdict = "clean-mode median later rounds slower"
    elif delta <= -materiality_percent:
        verdict = "clean-mode median later rounds faster"
    else:
        verdict = "no material clean-mode median drift"
    return {
        "clean_sample_count": len(points),
        "early_clean_count": len(early),
        "early_clean_median_ms": early_median,
        "late_clean_count": len(late),
        "late_clean_median_ms": late_median,
        "late_vs_early_percent": delta,
        "linear_slope_ms_per_round": slope,
        "linear_slope_percent_per_10_rounds": slope_percent,
        "verdict": verdict,
    }


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def load_orders(
    path: Path,
    roles: list[str],
    round_ids: list[str],
    recorded_cycle: list[list[str]] | None = None,
) -> tuple[dict[str, list[str]], str]:
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        orders: dict[str, list[str]] = {}
        for row in rows:
            round_id = row.get("round", "")
            order_text = row.get("execution_order", "")
            if not round_id or not order_text:
                raise ValueError(f"invalid execution-order row: {row}")
            if round_id in orders:
                raise ValueError(f"duplicate execution-order round: {round_id}")
            order = order_text.split(">")
            if sorted(order) != sorted(roles):
                raise ValueError(
                    f"round {round_id} order does not match roles: {order}"
                )
            orders[round_id] = order
        return orders, "recorded_csv"
    if recorded_cycle:
        if not all(round_id.isdigit() for round_id in round_ids):
            raise ValueError("metadata-cycle inference requires numeric round IDs")
        if any(sorted(order) != sorted(roles) for order in recorded_cycle):
            raise ValueError("recorded order cycle does not match target roles")
        return (
            {
                round_id: list(
                    recorded_cycle[
                        (int(round_id) - 1) % len(recorded_cycle)
                    ]
                )
                for round_id in round_ids
            },
            "inferred_experiment_metadata_cycle",
        )
    raise FileNotFoundError(
        f"execution-order metadata is missing: {path}"
    )


def load_role_samples(
    root: Path, role: str
) -> tuple[str, dict[str, float], dict[str, Path], dict[str, str]]:
    samples: dict[str, float] = {}
    csv_paths: dict[str, Path] = {}
    timestamps: dict[str, str] = {}
    module: str | None = None
    for directory in sorted((root / role).glob("round_*")):
        round_id = directory.name.removeprefix("round_")
        files = sorted((directory / "csv").glob("*.csv"))
        if len(files) != 1:
            raise ValueError(f"expected one CSV in {directory}: {files}")
        current_module, timestamp, value = read_result(files[0])
        if module is None:
            module = current_module
        elif module != current_module:
            raise ValueError(
                f"module changed for {role}: {module!r} vs {current_module!r}"
            )
        samples[round_id] = value
        csv_paths[round_id] = files[0]
        timestamps[round_id] = timestamp
    if not samples or module is None:
        raise ValueError(f"no round results found for role {role!r}")
    return module, samples, csv_paths, timestamps
