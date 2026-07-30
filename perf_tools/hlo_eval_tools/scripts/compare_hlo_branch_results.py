#!/usr/bin/env python3
"""Compare per-module HLO CSV timings across XLA branch result directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from reference_results import sha256_file


TIME_RE = re.compile(
    r"^\s*(?P<value>[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)"
    r"(?P<unit>ns|us|ms|s)\s*$",
    re.IGNORECASE,
)
TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}


def parse_time_ms(value: str) -> float:
    match = TIME_RE.match(value)
    if not match:
        raise ValueError(f"unsupported timing value: {value!r}")
    return float(match["value"]) * TO_MS[match["unit"].lower()]


def read_latest_row(path: Path) -> dict[str, float]:
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#") and line.strip():
                rows.extend(csv.reader([line]))
    if len(rows) < 2:
        raise ValueError(f"CSV has no timing row: {path}")
    header, latest = rows[0], rows[-1]
    if len(header) != len(latest):
        raise ValueError(
            f"CSV column mismatch in {path}: header={len(header)}, row={len(latest)}"
        )
    modules = header[1:]
    if not modules or any(not module.strip() for module in modules):
        raise ValueError(f"CSV has an empty or missing module header: {path}")
    if len(set(modules)) != len(modules):
        raise ValueError(f"CSV has duplicate module headers: {path}")
    return {
        module: parse_time_ms(value)
        for module, value in zip(modules, latest[1:], strict=True)
    }


def load_branch_csvs(csv_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    workloads: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}
    if not csv_dir.is_dir():
        return workloads, errors
    for path in sorted(csv_dir.glob("*.csv")):
        try:
            workloads[path.name] = read_latest_row(path)
        except ValueError as error:
            errors[path.name] = str(error)
    return workloads, errors


def load_checked_in_reference(
    reference_dataset: dict[str, Any],
) -> tuple[dict[str, dict[str, float | None]], dict[str, str]]:
    workloads: dict[str, dict[str, float | None]] = {}
    errors: dict[str, str] = {}
    inventory = reference_dataset["inventory"]
    tools_root = Path(__file__).resolve().parents[1]
    for item in inventory["workloads"]:
        if not item["exists"]:
            workloads[item["workload"]] = {
                module: None for module in item["modules"]
            }
            continue
        workload = item["workload"]
        path = tools_root / item["relative_path"]
        try:
            if not path.is_file():
                raise ValueError(f"recorded reference CSV is missing: {path}")
            actual_hash = sha256_file(path)
            if actual_hash != item["sha256"]:
                raise ValueError(
                    f"reference CSV checksum changed: {path}; "
                    f"expected {item['sha256']}, found {actual_hash}"
                )
            timings = read_latest_row(path)
            missing_modules = [
                module for module in item["modules"] if module not in timings
            ]
            if missing_modules:
                raise ValueError(
                    f"reference CSV is missing selected module(s): "
                    f"{', '.join(missing_modules)}"
                )
            workloads[workload] = {
                module: timings[module] for module in item["modules"]
            }
        except (OSError, ValueError) as error:
            errors[workload] = str(error)
    return workloads, errors


def format_csv_errors(ref: str, errors: dict[str, str]) -> str:
    details = "; ".join(
        f"{filename}: {message}" for filename, message in sorted(errors.items())
    )
    return f"malformed CSV result(s) for {ref}: {details}"


def comparison_rows(
    *,
    output_dir: Path,
    targets: list[dict[str, str]],
    reference_dataset: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_ref = reference_dataset["id"]
    baseline_commit = reference_dataset["xla_commit"]
    baseline_csvs, baseline_errors = load_checked_in_reference(reference_dataset)
    if baseline_errors:
        raise ValueError(format_csv_errors(baseline_ref, baseline_errors))

    rows: list[dict[str, Any]] = []
    for target in targets:
        candidate_csvs, candidate_errors = load_branch_csvs(
            output_dir / target["slug"] / "csv"
        )
        if candidate_errors:
            raise ValueError(format_csv_errors(target["ref"], candidate_errors))

        for workload in sorted(set(baseline_csvs) | set(candidate_csvs)):
            baseline_modules = baseline_csvs.get(workload, {})
            candidate_modules = candidate_csvs.get(workload, {})
            for module in sorted(set(baseline_modules) | set(candidate_modules)):
                baseline_ms = baseline_modules.get(module)
                candidate_ms = candidate_modules.get(module)
                if baseline_ms is None:
                    status = "missing_baseline"
                    ratio = None
                    delta_ms = None
                    delta_percent = None
                elif candidate_ms is None:
                    status = "missing_candidate"
                    ratio = None
                    delta_ms = None
                    delta_percent = None
                else:
                    ratio = (
                        candidate_ms / baseline_ms if baseline_ms != 0 else None
                    )
                    delta_ms = candidate_ms - baseline_ms
                    delta_percent = (
                        delta_ms / baseline_ms * 100.0 if baseline_ms != 0 else None
                    )
                    if delta_ms > 0:
                        status = "slower"
                    elif delta_ms < 0:
                        status = "faster"
                    else:
                        status = "unchanged"
                row = {
                    "baseline_ref": baseline_ref,
                    "baseline_commit": baseline_commit,
                    "baseline_source": reference_dataset["source"],
                    "candidate_id": target["id"],
                    "candidate_role": target["role"],
                    "candidate_ref": target["ref"],
                    "candidate_commit": target["commit"],
                    "workload": workload,
                    "module": module,
                    "baseline_ms": baseline_ms,
                    "candidate_ms": candidate_ms,
                    "ratio": ratio,
                    "delta_ms": delta_ms,
                    "delta_percent": delta_percent,
                    "status": status,
                }
                if "label" in target:
                    row["candidate_label"] = target["label"]
                rows.append(row)

    counts = Counter(row["status"] for row in rows)
    missing_baseline = counts.get("missing_baseline", 0)
    missing_candidate = counts.get("missing_candidate", 0)
    summary = {
        "baseline_ref": baseline_ref,
        "baseline_commit": baseline_commit,
        "baseline_source": reference_dataset["source"],
        "rows": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "validation": {
            "status": (
                "failed" if missing_baseline or missing_candidate else "passed"
            ),
            "missing_baseline_modules": missing_baseline,
            "missing_candidate_modules": missing_candidate,
        },
        "note": (
            "faster/slower reports the sign of the measured delta only; "
            "apply a noise threshold before declaring a regression or benefit"
        ),
    }
    return rows, summary


def summarize_branches(
    rows: list[dict[str, Any]],
    targets: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_id.setdefault(row["candidate_id"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for target in targets:
        branch_rows = rows_by_id.get(target["id"], [])
        matched = [
            row
            for row in branch_rows
            if row["baseline_ms"] is not None and row["candidate_ms"] is not None
        ]
        baseline_total = sum(row["baseline_ms"] for row in matched)
        candidate_total = sum(row["candidate_ms"] for row in matched)
        ratios = [
            row["candidate_ms"] / row["baseline_ms"]
            for row in matched
            if row["baseline_ms"] > 0 and row["candidate_ms"] > 0
        ]
        deltas = [
            row["delta_percent"]
            for row in matched
            if row["delta_percent"] is not None
        ]
        status_counts = Counter(row["status"] for row in branch_rows)
        suite_delta = (
            (candidate_total / baseline_total - 1.0) * 100.0
            if baseline_total > 0
            else None
        )
        geomean_delta = (
            (math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios)) - 1.0)
            * 100.0
            if ratios
            else None
        )
        summary = {
            "candidate_id": target["id"],
            "candidate_role": target["role"],
            "candidate_ref": target["ref"],
            "candidate_commit": target["commit"],
            "matched_modules": len(matched),
            "faster_modules": status_counts.get("faster", 0),
            "slower_modules": status_counts.get("slower", 0),
            "unchanged_modules": status_counts.get("unchanged", 0),
            "missing_baseline": status_counts.get("missing_baseline", 0),
            "missing_candidate": status_counts.get("missing_candidate", 0),
            "baseline_suite_ms": baseline_total if matched else None,
            "candidate_suite_ms": candidate_total if matched else None,
            "suite_ratio": (
                candidate_total / baseline_total if baseline_total > 0 else None
            ),
            "suite_delta_percent": suite_delta,
            "median_module_ratio": (
                statistics.median(ratios) if ratios else None
            ),
            "median_module_delta_percent": (
                statistics.median(deltas) if deltas else None
            ),
            "geomean_module_ratio": (
                geomean_delta / 100.0 + 1.0
                if geomean_delta is not None
                else None
            ),
            "geomean_module_delta_percent": geomean_delta,
        }
        if "label" in target:
            summary["candidate_label"] = target["label"]
        summaries.append(summary)
    return summaries


def markdown_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def format_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4g} ms"


def format_ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}×"


def format_change(delta_percent: float | None, status: str | None = None) -> str:
    if delta_percent is None:
        return status or "N/A"
    if abs(delta_percent) < 0.005:
        return "unchanged"
    if delta_percent < 0:
        return f"{abs(delta_percent):.2f}% faster"
    if delta_percent > 0:
        return f"{delta_percent:.2f}% slower"
    return "unchanged"


def report_workload_name(workload: str) -> str:
    name = workload.removesuffix(".csv")
    for category in (
        "large_language_models",
        "vision_diffusion",
        "multimodal",
        "science",
    ):
        prefix = f"{category}_"
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        match = re.fullmatch(r"(.+)_(training|inference_(\d+gpu))", remainder)
        if not match:
            break
        model, workload_type, gpu_count = match.groups()
        workload_path = (
            "training"
            if workload_type == "training"
            else f"inference/{gpu_count}"
        )
        return f"{category}/{model}/{workload_path}"
    return name


def report_module_name(module: str) -> str:
    for suffix in (
        ".before_optimizations.txt",
        ".before_optimizations.hlo",
        ".txt",
        ".hlo",
    ):
        if module.endswith(suffix):
            return module[: -len(suffix)]
    return module


def revision_header(ref: str, commit: str, role: str | None = None) -> str:
    header = f"`{markdown_escape(ref)}`<br>`{commit[:12]}`"
    return f"{header}<br>{role}" if role else header


def candidate_display_label(item: dict[str, Any]) -> str:
    label = item.get("candidate_label")
    ref = item["candidate_ref"]
    if isinstance(label, str) and label and label != ref:
        return f"{label} ({ref})"
    return ref


def candidate_matrix_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "N/A<br>not present"
    if row["candidate_ms"] is None:
        return f"N/A<br>{markdown_escape(row['status'].replace('_', ' '))}"
    if row["delta_percent"] is None:
        return (
            f"{format_ms(row['candidate_ms'])}<br>N/A · "
            f"{markdown_escape(row['status'].replace('_', ' '))}"
        )
    return (
        f"{format_ms(row['candidate_ms'])}<br>"
        f"{format_ratio(row['ratio'])} · "
        f"{format_change(row['delta_percent'], row['status'])}"
    )


def write_markdown_report(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    branch_summaries: list[dict[str, Any]],
    baseline_ref: str,
    baseline_commit: str,
    live_control_id: str,
) -> None:
    rows_by_metric: dict[
        tuple[str, str], dict[str, dict[str, Any]]
    ] = {}
    baseline_by_metric: dict[tuple[str, str], float | None] = {}
    for row in rows:
        metric = (row["workload"], row["module"])
        rows_by_metric.setdefault(metric, {})[row["candidate_ref"]] = row
        if metric not in baseline_by_metric or row["baseline_ms"] is not None:
            baseline_by_metric[metric] = row["baseline_ms"]

    lines = [
        "# XLA HLO Performance Comparison",
        "",
        "Lower is better. Ratio is `candidate / baseline`: below 1.0× is "
        "faster; above 1.0× is slower.",
        "",
        "> These are isolated HLO module timings, not end-to-end model latency. "
        "Apply a noise threshold before declaring a regression or benefit.",
        "",
    ]

    lines.extend(
        [
            "## Compared revisions",
            "",
            "| role | XLA ref or dataset | commit |",
            "|---|---|---|",
            (
                f"| Historical reference | `{markdown_escape(baseline_ref)}` | "
                f"`{baseline_commit}` |"
            ),
        ]
    )
    for branch in branch_summaries:
        role = (
            "Live pinned control"
            if branch["candidate_role"] == "live_control"
            else "Live candidate"
        )
        lines.append(
            f"| {role} | `{markdown_escape(candidate_display_label(branch))}` | "
            f"`{branch['candidate_commit']}` |"
        )

    live_control = next(
        (
            branch
            for branch in branch_summaries
            if branch["candidate_id"] == live_control_id
        ),
        None,
    )
    lines.extend(
        [
            "",
            "## Reference reproducibility",
            "",
            "The live control rebuilds the historical reference commit on the "
            "current server. Its gap includes server, environment, and measurement "
            "drift while holding the XLA commit constant.",
            "",
        ]
    )
    if live_control is None:
        lines.append("Live control results are unavailable.")
    else:
        lines.extend(
            [
                "| live control | commit | matched | suite ratio | suite change | "
                "missing control | missing reference |",
                "|---|---|---:|---:|---:|---:|---:|",
                (
                    f"| `{markdown_escape(live_control['candidate_ref'])}` | "
                    f"`{live_control['candidate_commit'][:12]}` | "
                    f"{live_control['matched_modules']} | "
                    f"{format_ratio(live_control['suite_ratio'])} | "
                    f"{format_change(live_control['suite_delta_percent'])} | "
                    f"{live_control['missing_candidate']} | "
                    f"{live_control['missing_baseline']} |"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Branch overview",
            "",
            "| role | candidate | commit | matched | faster | slower | unchanged | "
            "missing candidate | missing baseline |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for branch in branch_summaries:
        role = (
            "control"
            if branch["candidate_role"] == "live_control"
            else "candidate"
        )
        lines.append(
            f"| {role} | `{markdown_escape(candidate_display_label(branch))}` | "
            f"`{branch['candidate_commit'][:12]}` | "
            f"{branch['matched_modules']} | {branch['faster_modules']} | "
            f"{branch['slower_modules']} | {branch['unchanged_modules']} | "
            f"{branch['missing_candidate']} | {branch['missing_baseline']} |"
        )

    lines.extend(
        [
            "",
            "Detailed aggregate statistics remain available in "
            "`branch_summary.csv` and `comparison_summary.json`.",
            "",
            "## Per-HLO comparison",
            "",
            "Candidate cells show `latency / ratio / change` relative to the "
            "baseline.",
            "",
        ]
    )
    candidate_refs = [branch["candidate_ref"] for branch in branch_summaries]
    candidate_commits = {
        branch["candidate_ref"]: branch["candidate_commit"]
        for branch in branch_summaries
    }
    candidate_roles = {
        branch["candidate_ref"]: (
            "Live control"
            if branch["candidate_role"] == "live_control"
            else "Candidate"
        )
        for branch in branch_summaries
    }
    candidate_labels = {
        branch["candidate_ref"]: candidate_display_label(branch)
        for branch in branch_summaries
    }

    def metric_sort_key(metric: tuple[str, str]) -> tuple[Any, ...]:
        metric_rows = rows_by_metric[metric].values()
        deltas = [
            row["delta_percent"]
            for row in metric_rows
            if row["delta_percent"] is not None
        ]
        worst_delta = max(deltas) if deltas else 0.0
        return (not deltas, -worst_delta, metric[0], metric[1])

    sorted_metrics = sorted(rows_by_metric, key=metric_sort_key)
    for group_start in range(0, len(candidate_refs), 3):
        group_refs = candidate_refs[group_start : group_start + 3]
        if len(candidate_refs) > 3:
            lines.extend(
                [
                    (
                        f"### Candidates {group_start + 1}–"
                        f"{group_start + len(group_refs)} of {len(candidate_refs)}"
                    ),
                    "",
                ]
            )
        headers = [
            "workload",
            "HLO module",
            revision_header(
                baseline_ref, baseline_commit, "Historical reference"
            ),
            *[
                revision_header(
                    candidate_labels[ref],
                    candidate_commits[ref],
                    candidate_roles[ref],
                )
                for ref in group_refs
            ],
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|---|---|" + "---:|" * (1 + len(group_refs)))
        for metric in sorted_metrics:
            workload, module = metric
            cells = [
                f"`{markdown_escape(report_workload_name(workload))}`",
                f"`{markdown_escape(report_module_name(module))}`",
                format_ms(baseline_by_metric.get(metric)),
                *[
                    candidate_matrix_cell(rows_by_metric[metric].get(ref))
                    for ref in group_refs
                ],
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison(
    *,
    output_dir: Path,
    targets: list[dict[str, str]],
    reference_dataset: dict[str, Any],
    live_control_id: str,
) -> dict[str, Any]:
    rows, summary = comparison_rows(
        output_dir=output_dir,
        targets=targets,
        reference_dataset=reference_dataset,
    )
    branch_summaries = summarize_branches(rows, targets)
    csv_path = output_dir / "comparison.csv"
    fields = [
        "baseline_ref",
        "baseline_commit",
        "baseline_source",
        "candidate_id",
        "candidate_role",
        "candidate_ref",
        "candidate_commit",
        "workload",
        "module",
        "baseline_ms",
        "candidate_ms",
        "ratio",
        "delta_ms",
        "delta_percent",
        "status",
    ]
    if any("candidate_label" in row for row in rows):
        fields.insert(fields.index("candidate_commit"), "candidate_label")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    branch_summary_path = output_dir / "branch_summary.csv"
    branch_fields = [
        "candidate_id",
        "candidate_role",
        "candidate_ref",
        "candidate_commit",
        "matched_modules",
        "faster_modules",
        "slower_modules",
        "unchanged_modules",
        "missing_baseline",
        "missing_candidate",
        "baseline_suite_ms",
        "candidate_suite_ms",
        "suite_ratio",
        "suite_delta_percent",
        "median_module_ratio",
        "median_module_delta_percent",
        "geomean_module_ratio",
        "geomean_module_delta_percent",
    ]
    if any("candidate_label" in branch for branch in branch_summaries):
        branch_fields.insert(
            branch_fields.index("candidate_commit"), "candidate_label"
        )
    with branch_summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=branch_fields)
        writer.writeheader()
        writer.writerows(branch_summaries)

    summary["branches"] = branch_summaries
    summary["reference_reproducibility"] = next(
        (
            branch
            for branch in branch_summaries
            if branch["candidate_id"] == live_control_id
        ),
        None,
    )
    summary["aggregation_note"] = (
        "suite_delta_percent compares the sum of matched module timings and is "
        "not end-to-end model latency; geomean_module_delta_percent weights each "
        "matched module equally"
    )
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "comparison_report.md"
    write_markdown_report(
        path=report_path,
        rows=rows,
        branch_summaries=branch_summaries,
        baseline_ref=summary["baseline_ref"],
        baseline_commit=summary["baseline_commit"],
        live_control_id=live_control_id,
    )
    return {
        **summary,
        "csv": str(csv_path),
        "branch_summary_csv": str(branch_summary_path),
        "summary_json": str(summary_path),
        "markdown_report": str(report_path),
    }


def select_comparison_targets(
    manifest: dict[str, Any],
) -> list[dict[str, str]]:
    targets = manifest["targets"]
    by_id = {target["id"]: target for target in targets}
    selected_ids = manifest.get("comparison_target_ids")
    if not isinstance(selected_ids, list) or any(
        not isinstance(target_id, str) for target_id in selected_ids
    ):
        raise ValueError("manifest comparison target IDs must be a list of strings")
    selected_ids = list(selected_ids)
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("manifest comparison target IDs contain duplicates")
    missing = [target_id for target_id in selected_ids if target_id not in by_id]
    if missing:
        raise ValueError(
            "manifest comparison target IDs are not campaign targets: "
            + ", ".join(missing)
        )
    return [by_id[target_id] for target_id in selected_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 2:
            raise ValueError("standalone comparison requires a schema-v2 campaign")
        targets = select_comparison_targets(manifest)
        result = write_comparison(
            output_dir=args.output_dir,
            targets=targets,
            reference_dataset=manifest["reference_dataset"],
            live_control_id=manifest["live_control_id"],
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["validation"]["status"] != "passed":
            print("error: comparison validation failed", file=sys.stderr)
            return 1
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
