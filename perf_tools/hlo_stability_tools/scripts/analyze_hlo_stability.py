#!/usr/bin/env python3
"""Analyze repeated one-HLO measurements as evidence, without deleting samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from file_util import repository_identity, tooling_metadata
from hlo_stability import (
    basic_stats,
    classify_outliers,
    correlation,
    load_json_object,
    load_orders,
    load_role_samples,
    paired_percent_stats,
    read_result,
    round_sort_key,
    sha256_file,
    temporal_trend,
)


ANALYSIS_FILES = (
    "stability_analysis.json",
    "stability_summary.csv",
    "raw_rounds_long.csv",
    "paired_deltas.csv",
)
FREQUENT_OUTLIER_MIN_COUNT = 3
FREQUENT_OUTLIER_RATE_PERCENT = 10.0
RAW_CV_INSTABILITY_PERCENT = 2.0
ANALYSIS_TOOLING_FILES = (
    "analyze_hlo_stability.py",
    "hlo_stability.py",
    "file_util.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument(
        "--modified-z-threshold",
        type=float,
        default=3.5,
        help="robust outlier threshold (default: 3.5)",
    )
    parser.add_argument(
        "--minimum-outlier-percent",
        type=float,
        default=2.0,
        help=(
            "minimum absolute deviation before a sample can be flagged "
            "(default: 2%% of role median)"
        ),
    )
    parser.add_argument(
        "--temporal-drift-percent",
        type=float,
        default=2.0,
        help=(
            "first-half versus last-half change needed to describe a "
            "material temporal pattern (default: 2%%)"
        ),
    )
    parser.add_argument(
        "--reporting-threshold-percent",
        type=float,
        default=2.0,
        help=(
            "candidate/control evidence reporting band, independent of the "
            "outlier policy (default: 2%%)"
        ),
    )
    parser.add_argument(
        "--minimum-paired-rounds",
        type=int,
        default=3,
        help="minimum eligible pairs for candidate/control evidence (default: 3)",
    )
    return parser.parse_args()


def analyze(
    *,
    experiment_dir: Path,
    roles: list[str],
    reference_csv: Path | None,
    modified_z_threshold: float,
    minimum_outlier_percent: float,
    temporal_drift_percent: float = 2.0,
    reporting_threshold_percent: float = 2.0,
    minimum_paired_rounds: int = 3,
) -> dict[str, Any]:
    metadata_path = experiment_dir / "experiment_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"experiment metadata is missing: {metadata_path}")
    metadata = load_json_object(metadata_path)
    if metadata.get("status") not in {
        "collected",
        "analyzed",
        "completed",
    }:
        raise ValueError(
            "experiment is not complete enough for analysis: "
            f"{metadata.get('status')!r}"
        )
    modules: set[str] = set()
    role_samples: dict[str, dict[str, float]] = {}
    role_csv_paths: dict[str, dict[str, Path]] = {}
    role_timestamps: dict[str, dict[str, str]] = {}
    outlier_flags: dict[str, dict[str, bool]] = {}
    role_summaries: dict[str, dict[str, Any]] = {}

    for role in roles:
        module, samples, csv_paths, timestamps = load_role_samples(
            experiment_dir, role
        )
        modules.add(module)
        role_samples[role] = samples
        role_csv_paths[role] = csv_paths
        role_timestamps[role] = timestamps
        flags, outlier_rule = classify_outliers(
            samples,
            modified_z_threshold=modified_z_threshold,
            minimum_outlier_percent=minimum_outlier_percent,
        )
        outlier_flags[role] = flags
        clean_values = [
            value
            for round_id, value in samples.items()
            if not flags[round_id]
        ]
        if not clean_values:
            raise ValueError(f"outlier rule excluded every {role} sample")
        raw_stats = basic_stats(list(samples.values()))
        clean_stats = basic_stats(clean_values)
        outlier_count = sum(flags.values())
        outlier_rate = outlier_count / raw_stats["count"] * 100.0
        high_outlier_count = sum(
            1
            for round_id, flagged in flags.items()
            if flagged and samples[round_id] > clean_stats["median_ms"]
        )
        low_outlier_count = outlier_count - high_outlier_count
        frequent_outliers = (
            outlier_count >= FREQUENT_OUTLIER_MIN_COUNT
            and outlier_rate >= FREQUENT_OUTLIER_RATE_PERCENT
        )
        broad_distribution = (
            raw_stats["cv_percent"] >= RAW_CV_INSTABILITY_PERCENT
        )
        if frequent_outliers:
            direction = (
                "high-latency"
                if high_outlier_count == outlier_count
                else "low-latency"
                if low_outlier_count == outlier_count
                else "mixed"
            )
            stability_evidence = (
                f"frequent {direction} events "
                f"({outlier_count}/{raw_stats['count']}, "
                f"{outlier_rate:.1f}%; raw CV "
                f"{raw_stats['cv_percent']:.2f}%)"
            )
        else:
            stability_evidence = (
                f"broad or multimodal latency distribution "
                f"(raw CV {raw_stats['cv_percent']:.2f}%)"
                if broad_distribution
                else f"no frequent outlier pattern "
                f"({outlier_count}/{raw_stats['count']}, "
                f"{outlier_rate:.1f}%)"
            )
        distribution_instability = frequent_outliers or broad_distribution
        role_summaries[role] = {
            "raw": raw_stats,
            "clean": clean_stats,
            "outlier_rule": outlier_rule,
            "outlier_count": outlier_count,
            "outlier_rate_percent": outlier_rate,
            "high_outlier_count": high_outlier_count,
            "low_outlier_count": low_outlier_count,
            "frequent_outliers": frequent_outliers,
            "broad_distribution": broad_distribution,
            "distribution_instability": distribution_instability,
            "stability_evidence": stability_evidence,
            "outlier_rounds": [
                round_id for round_id, flagged in flags.items() if flagged
            ],
            "outlier_values_ms": [
                samples[round_id]
                for round_id, flagged in flags.items()
                if flagged
            ],
        }

    if len(modules) != 1:
        raise ValueError(f"roles contain different HLO modules: {sorted(modules)}")
    round_sets = {role: set(samples) for role, samples in role_samples.items()}
    expected_rounds = round_sets[roles[0]]
    mismatched_rounds = {
        role: sorted(round_ids, key=round_sort_key)
        for role, round_ids in round_sets.items()
        if round_ids != expected_rounds
    }
    if mismatched_rounds:
        raise ValueError(
            "role round sets differ; refusing to omit unpaired evidence: "
            f"{mismatched_rounds}"
        )
    common_rounds = sorted(
        expected_rounds,
        key=round_sort_key,
    )
    design = metadata.get("design")
    expected_round_count = (
        design.get("rounds") if isinstance(design, dict) else None
    )
    if type(expected_round_count) is not int or expected_round_count < 1:
        raise ValueError("experiment metadata has no valid design.rounds")
    expected_round_ids = [
        f"{index:02d}" for index in range(1, expected_round_count + 1)
    ]
    if common_rounds != expected_round_ids:
        raise ValueError(
            "experiment does not contain every requested round: "
            f"expected={expected_round_ids}, found={common_rounds}"
        )
    recorded_cycle = metadata.get("design", {}).get("order_cycle")
    orders, order_source = load_orders(
        experiment_dir / "round_orders.csv",
        roles,
        common_rounds,
        recorded_cycle if isinstance(recorded_cycle, list) else None,
    )
    if set(common_rounds) != set(orders):
        raise ValueError(
            "round order/result mismatch: "
            f"orders={sorted(orders)}, results={common_rounds}"
        )

    for role in roles:
        role_summaries[role]["temporal_trend"] = temporal_trend(
            role_samples[role],
            outlier_flags[role],
            common_rounds,
            materiality_percent=temporal_drift_percent,
        )
        split = len(common_rounds) // 2
        early_rounds = common_rounds[:split]
        late_rounds = common_rounds[split:]
        early_outliers = sum(
            outlier_flags[role][round_id] for round_id in early_rounds
        )
        late_outliers = sum(
            outlier_flags[role][round_id] for round_id in late_rounds
        )
        early_rate = (
            early_outliers / len(early_rounds) * 100.0
            if early_rounds
            else 0.0
        )
        late_rate = (
            late_outliers / len(late_rounds) * 100.0
            if late_rounds
            else 0.0
        )
        role_summaries[role]["outlier_temporal_pattern"] = {
            "early_count": early_outliers,
            "early_total": len(early_rounds),
            "early_rate_percent": early_rate,
            "late_count": late_outliers,
            "late_total": len(late_rounds),
            "late_rate_percent": late_rate,
            "late_minus_early_percentage_points": late_rate - early_rate,
        }

    module = next(iter(modules))
    long_rows: list[dict[str, Any]] = []
    for round_id in common_rounds:
        execution_order = orders[round_id]
        if sorted(execution_order) != sorted(roles):
            raise ValueError(
                f"round {round_id} order does not contain roles {roles}: "
                f"{execution_order}"
            )
        for position, role in enumerate(execution_order, start=1):
            value = role_samples[role][round_id]
            median = role_summaries[role]["clean"]["median_ms"]
            target = metadata.get("targets", {}).get(role, {})
            long_rows.append(
                {
                    "schema_version": 2,
                    "round": round_id,
                    "execution_order": ">".join(execution_order),
                    "position": position,
                    "timestamp": role_timestamps[role][round_id],
                    "role": role,
                    "label": target.get("label", ""),
                    "source_ref": target.get("source_ref", ""),
                    "commit": target.get("commit", ""),
                    "runner_sha256": target.get("runner_sha256", ""),
                    "module": module,
                    "latency_ms": value,
                    "clean_median_ms": median,
                    "normalized_delta_percent": (
                        value / median - 1.0
                    )
                    * 100.0,
                    "is_outlier": outlier_flags[role][round_id],
                    "analysis_included": not outlier_flags[role][round_id],
                    "csv_path": str(
                        role_csv_paths[role][round_id].relative_to(
                            experiment_dir
                        )
                    ),
                    "system_before_path": (
                        f"{role}/round_{round_id}/system_before.txt"
                        if (
                            experiment_dir
                            / role
                            / f"round_{round_id}"
                            / "system_before.txt"
                        ).is_file()
                        else ""
                    ),
                    "system_after_path": (
                        f"{role}/round_{round_id}/system_after.txt"
                        if (
                            experiment_dir
                            / role
                            / f"round_{round_id}"
                            / "system_after.txt"
                        ).is_file()
                        else ""
                    ),
                }
            )

    control_role = "control" if "control" in roles else roles[0]
    for role in roles:
        if role == control_role:
            continue
        eligible_rounds = [
            round_id
            for round_id in common_rounds
            if not outlier_flags[control_role][round_id]
            and not outlier_flags[role][round_id]
        ]
        paired_deltas = [
            (
                role_samples[role][round_id]
                / role_samples[control_role][round_id]
                - 1.0
            )
            * 100.0
            for round_id in eligible_rounds
        ]
        role_summaries[role]["paired_vs_control"] = {
            **paired_percent_stats(paired_deltas),
            "eligible_rounds": eligible_rounds,
            "policy": "exclude a pair when either target is an outlier",
        }

    pair_rows: list[dict[str, Any]] = []
    for round_id in common_rounds:
        control = role_samples[control_role][round_id]
        row: dict[str, Any] = {
            "schema_version": 2,
            "round": round_id,
            "execution_order": ">".join(orders[round_id]),
            f"{control_role}_ms": control,
            f"{control_role}_timestamp": role_timestamps[control_role][
                round_id
            ],
            f"{control_role}_outlier": outlier_flags[control_role][round_id],
        }
        for role in roles:
            value = role_samples[role][round_id]
            row[f"{role}_ms"] = value
            row[f"{role}_timestamp"] = role_timestamps[role][round_id]
            row[f"{role}_outlier"] = outlier_flags[role][round_id]
            if role != control_role:
                row[f"{role}_vs_{control_role}_percent"] = (
                    value / control - 1.0
                ) * 100.0
                row[f"{role}_pair_analysis_included"] = (
                    not outlier_flags[control_role][round_id]
                    and not outlier_flags[role][round_id]
                )
        pair_rows.append(row)

    normalized_correlations: dict[str, float | None] = {}
    control_clean_median = role_summaries[control_role]["clean"]["median_ms"]
    for role in roles:
        if role == control_role:
            continue
        eligible = [
            round_id
            for round_id in common_rounds
            if not outlier_flags[control_role][round_id]
            and not outlier_flags[role][round_id]
        ]
        left = [
            role_samples[control_role][round_id] / control_clean_median - 1.0
            for round_id in eligible
        ]
        role_clean_median = role_summaries[role]["clean"]["median_ms"]
        right = [
            role_samples[role][round_id] / role_clean_median - 1.0
            for round_id in eligible
        ]
        normalized_correlations[role] = correlation(left, right)

    historical: dict[str, Any] | None = None
    if reference_csv is not None:
        historical_module, timestamp, historical_ms = read_result(reference_csv)
        if historical_module != module:
            raise ValueError(
                f"historical module differs: {historical_module!r}"
            )
        historical = {
            "file_name": reference_csv.name,
            "sha256": sha256_file(reference_csv),
            "timestamp": timestamp,
            "latency_ms": historical_ms,
            "source_commit_verified": False,
            "identity_note": (
                "the timing CSV does not encode a source commit; historical "
                "and live-control commit identity is not programmatically "
                "verified"
            ),
            "control_clean_median_vs_historical_percent": (
                control_clean_median / historical_ms - 1.0
            )
            * 100.0,
        }

    runner_hash_roles: dict[str, list[str]] = {}
    metadata_targets = metadata.get("targets", {})
    if isinstance(metadata_targets, dict):
        for role in roles:
            target = metadata_targets.get(role, {})
            runner_hash = (
                target.get("runner_sha256")
                if isinstance(target, dict)
                else None
            )
            if isinstance(runner_hash, str) and runner_hash:
                runner_hash_roles.setdefault(runner_hash, []).append(role)
    identity_warnings = [
        {
            "type": "identical_runner_binary",
            "runner_sha256": runner_hash,
            "roles": grouped_roles,
            "message": (
                "targets use byte-identical runner binaries; observed "
                "differences cannot be attributed to different runner code"
            ),
        }
        for runner_hash, grouped_roles in runner_hash_roles.items()
        if len(grouped_roles) > 1
    ]

    return {
        "schema_version": 2,
        "experiment_dir": ".",
        "experiment_name": experiment_dir.name,
        "roles": roles,
        "targets": metadata.get("targets", {}),
        "execution_design": metadata.get("design", {}),
        "module": module,
        "round_count": len(common_rounds),
        "execution_order_source": order_source,
        "outlier_policy": {
            "modified_z_threshold": modified_z_threshold,
            "minimum_outlier_percent": minimum_outlier_percent,
            "rule": (
                "flag only when absolute deviation exceeds both the robust "
                "modified-z cutoff and the percentage floor"
            ),
            "percentile_method": "nearest-rank",
            "frequent_outlier_minimum_count": (
                FREQUENT_OUTLIER_MIN_COUNT
            ),
            "frequent_outlier_rate_percent": (
                FREQUENT_OUTLIER_RATE_PERCENT
            ),
            "raw_cv_instability_percent": RAW_CV_INSTABILITY_PERCENT,
        },
        "reporting_policy": {
            "candidate_control_threshold_percent": reporting_threshold_percent,
            "minimum_paired_rounds": minimum_paired_rounds,
            "rule": (
                "classify paired candidate/control median evidence independently "
                "from outlier thresholds"
            ),
        },
        "temporal_evidence_policy": {
            "comparison": (
                "median of the first scheduled half versus the last scheduled "
                "half, excluding flagged outliers"
            ),
            "linear_slope": (
                "ordinary least-squares slope across retained raw rounds; "
                "descriptive only"
            ),
            "materiality_percent": temporal_drift_percent,
        },
        "temporal_trend_policy": {
            "comparison": (
                "median of the first scheduled half versus the last scheduled "
                "half, excluding flagged outliers"
            ),
            "linear_slope": (
                "ordinary least-squares slope across retained raw rounds; "
                "descriptive only"
            ),
            "materiality_percent": temporal_drift_percent,
        },
        "role_summaries": role_summaries,
        "historical_reference": historical,
        "identity_warnings": identity_warnings,
        "normalized_correlations_vs_control": normalized_correlations,
        "long_rows": long_rows,
        "pair_rows": pair_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(root: Path, result: dict[str, Any]) -> None:
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"long_rows", "pair_rows"}
    }
    (root / "stability_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(root / "raw_rounds_long.csv", result["long_rows"])
    write_csv(root / "paired_deltas.csv", result["pair_rows"])

    summary_rows = []
    control_role = "control" if "control" in result["roles"] else result["roles"][0]
    control_median = result["role_summaries"][control_role]["clean"][
        "median_ms"
    ]
    historical = result.get("historical_reference")
    threshold = result["reporting_policy"][
        "candidate_control_threshold_percent"
    ]
    minimum_pairs = result["reporting_policy"]["minimum_paired_rounds"]
    for role in result["roles"]:
        item = result["role_summaries"][role]
        clean = item["clean"]
        temporal = item["temporal_trend"]
        target = result.get("targets", {}).get(role, {})
        delta = (clean["median_ms"] / control_median - 1.0) * 100.0
        paired = item.get("paired_vs_control")
        paired_count = (
            clean["count"] if role == control_role else int(paired["count"])
        )
        paired_delta = (
            0.0
            if role == control_role
            else paired["median_percent"]
        )
        if role == control_role:
            clean_mode_evidence = "live control"
        elif paired_count < minimum_pairs or paired_delta is None:
            clean_mode_evidence = "insufficient paired evidence"
        elif paired_delta <= -threshold:
            clean_mode_evidence = "clean mode observed faster than control"
        elif paired_delta >= threshold:
            clean_mode_evidence = "clean mode observed slower than control"
        else:
            clean_mode_evidence = "clean mode within control reporting band"
        evidence = (
            f"{clean_mode_evidence}; {item['stability_evidence']}"
            if item["distribution_instability"]
            else clean_mode_evidence
        )
        outlier_temporal = item["outlier_temporal_pattern"]
        summary_rows.append(
            {
                "schema_version": 2,
                "role": role,
                "label": target.get("label", ""),
                "source_ref": target.get("source_ref", ""),
                "commit": target.get("commit", ""),
                "runner_sha256": target.get("runner_sha256", ""),
                "raw_count": item["raw"]["count"],
                "raw_median_ms": item["raw"]["median_ms"],
                "raw_cv_percent": item["raw"]["cv_percent"],
                "clean_count": clean["count"],
                "clean_median_ms": clean["median_ms"],
                "clean_cv_percent": clean["cv_percent"],
                "clean_min_ms": clean["min_ms"],
                "clean_p05_ms": clean["p05_ms"],
                "clean_p95_ms": clean["p95_ms"],
                "clean_max_ms": clean["max_ms"],
                "outlier_count": item["outlier_count"],
                "outlier_rate_percent": item["outlier_rate_percent"],
                "high_outlier_count": item["high_outlier_count"],
                "low_outlier_count": item["low_outlier_count"],
                "frequent_outliers": item["frequent_outliers"],
                "broad_distribution": item["broad_distribution"],
                "distribution_instability": item[
                    "distribution_instability"
                ],
                "stability_evidence": item["stability_evidence"],
                "effective_outlier_cutoff_ms": item["outlier_rule"][
                    "effective_cutoff_ms"
                ],
                "median_vs_control_percent": delta,
                "paired_count": paired_count,
                "paired_exclusion_rate_percent": (
                    (item["raw"]["count"] - paired_count)
                    / item["raw"]["count"]
                    * 100.0
                ),
                "paired_median_vs_control_percent": paired_delta,
                "paired_mad_percent": (
                    0.0 if role == control_role else paired["mad_percent"]
                ),
                "paired_p05_percent": (
                    0.0 if role == control_role else paired["p05_percent"]
                ),
                "paired_p95_percent": (
                    0.0 if role == control_role else paired["p95_percent"]
                ),
                "median_vs_historical_percent": (
                    (clean["median_ms"] / historical["latency_ms"] - 1.0)
                    * 100.0
                    if historical
                    else ""
                ),
                "early_clean_median_ms": temporal["early_clean_median_ms"],
                "late_clean_median_ms": temporal["late_clean_median_ms"],
                "late_vs_early_percent": temporal["late_vs_early_percent"],
                "linear_slope_percent_per_10_rounds": temporal[
                    "linear_slope_percent_per_10_rounds"
                ],
                "temporal_evidence": temporal["verdict"],
                "early_outlier_rate_percent": outlier_temporal[
                    "early_rate_percent"
                ],
                "late_outlier_rate_percent": outlier_temporal[
                    "late_rate_percent"
                ],
                "outlier_rate_change_percentage_points": (
                    outlier_temporal[
                        "late_minus_early_percentage_points"
                    ]
                ),
                "clean_mode_evidence": clean_mode_evidence,
                "evidence_summary": evidence,
                "temporal_verdict": temporal["verdict"],
                "verdict": evidence,
            }
        )
    write_csv(root / "stability_summary.csv", summary_rows)


def write_experiment_metadata(
    path: Path,
    metadata: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    numeric_values = {
        "modified-z threshold": args.modified_z_threshold,
        "minimum outlier percent": args.minimum_outlier_percent,
        "temporal drift percent": args.temporal_drift_percent,
        "reporting threshold percent": args.reporting_threshold_percent,
    }
    for label, value in numeric_values.items():
        if not math.isfinite(value):
            raise SystemExit(f"{label} must be finite")
    if args.modified_z_threshold <= 0:
        raise SystemExit("--modified-z-threshold must be positive")
    if args.minimum_outlier_percent < 0:
        raise SystemExit("--minimum-outlier-percent must be nonnegative")
    if args.temporal_drift_percent < 0:
        raise SystemExit("--temporal-drift-percent must be nonnegative")
    if args.reporting_threshold_percent < 0:
        raise SystemExit("--reporting-threshold-percent must be nonnegative")
    if args.minimum_paired_rounds < 1:
        raise SystemExit("--minimum-paired-rounds must be at least 1")
    root = args.experiment_dir.expanduser().resolve()
    metadata_path = root / "experiment_metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(
            f"experiment metadata is missing: {metadata_path}"
        )
    metadata = load_json_object(metadata_path)
    roles = metadata.get("design", {}).get("roles")
    if not isinstance(roles, list) or any(
        not isinstance(role, str) or not role for role in roles
    ):
        raise SystemExit("experiment metadata has no valid design.roles")
    if len(roles) < 2 or len(set(roles)) != len(roles):
        raise SystemExit("--roles must contain at least two unique names")
    reference_csv = (
        args.reference_csv.expanduser().resolve()
        if args.reference_csv
        else None
    )
    result = analyze(
        experiment_dir=root,
        roles=roles,
        reference_csv=reference_csv,
        modified_z_threshold=args.modified_z_threshold,
        minimum_outlier_percent=args.minimum_outlier_percent,
        temporal_drift_percent=args.temporal_drift_percent,
        reporting_threshold_percent=args.reporting_threshold_percent,
        minimum_paired_rounds=args.minimum_paired_rounds,
    )
    write_outputs(root, result)
    metadata["status"] = "analyzed"
    metadata["analyzed_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    metadata["analysis_artifacts"] = {
        name: sha256_file(root / name) for name in ANALYSIS_FILES
    }
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    metadata.setdefault("tooling", {}).update(
        tooling_metadata(
            script_dir,
            repo_root,
            ANALYSIS_TOOLING_FILES,
        )
    )
    metadata["analysis_repository"] = repository_identity(repo_root)
    previous_outputs = metadata.get("outputs")
    report_names = {"stability_report.html"}
    if isinstance(previous_outputs, dict):
        recorded_report = previous_outputs.get("html_report")
        if isinstance(recorded_report, str) and recorded_report:
            report_names.add(recorded_report)
    for report_name in report_names:
        report_path = (root / report_name).resolve()
        if (
            report_path.parent == root
            and report_path.suffix.lower() == ".html"
            and report_path.is_file()
        ):
            report_path.unlink()
    metadata.pop("outputs", None)
    collection = metadata.get("collection")
    if isinstance(collection, dict):
        collection["reference_csv"] = (
            {
                "file_name": reference_csv.name,
                "sha256": sha256_file(reference_csv),
            }
            if reference_csv is not None
            else None
        )
    write_experiment_metadata(metadata_path, metadata)
    print("==== HLO stability evidence ====")
    for role in roles:
        item = result["role_summaries"][role]
        clean = item["clean"]
        print(
            f"{role:12s} median={clean['median_ms']:.4f} ms "
            f"CV={clean['cv_percent']:.2f}% "
            f"P05-P95={clean['p05_ms']:.4f}-{clean['p95_ms']:.4f} ms "
            f"outliers={item['outlier_count']}"
        )
    if result["historical_reference"]:
        print(
            "control vs historical: "
            f"{result['historical_reference']['control_clean_median_vs_historical_percent']:+.2f}%"
        )
    print(f"Outputs: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
