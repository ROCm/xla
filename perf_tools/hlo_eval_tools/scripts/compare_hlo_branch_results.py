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


def format_csv_errors(ref: str, errors: dict[str, str]) -> str:
    details = "; ".join(
        f"{filename}: {message}" for filename, message in sorted(errors.items())
    )
    return f"malformed CSV result(s) for {ref}: {details}"


def comparison_rows(
    *,
    output_dir: Path,
    targets: list[dict[str, str]],
    baseline_ref: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_ref = {target["ref"]: target for target in targets}
    if baseline_ref not in by_ref:
        raise ValueError(f"baseline ref is not in the campaign target list: {baseline_ref}")

    baseline_target = by_ref[baseline_ref]
    baseline_csvs, baseline_errors = load_branch_csvs(
        output_dir / baseline_target["slug"] / "csv"
    )
    if baseline_errors:
        raise ValueError(format_csv_errors(baseline_ref, baseline_errors))
    if not baseline_csvs:
        raise ValueError(f"baseline produced no readable CSV results: {baseline_ref}")

    rows: list[dict[str, Any]] = []
    for target in targets:
        if target["ref"] == baseline_ref:
            continue
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
                rows.append(
                    {
                        "baseline_ref": baseline_ref,
                        "baseline_commit": baseline_target["commit"],
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
                )

    counts = Counter(row["status"] for row in rows)
    missing_baseline = counts.get("missing_baseline", 0)
    missing_candidate = counts.get("missing_candidate", 0)
    summary = {
        "baseline_ref": baseline_ref,
        "baseline_commit": baseline_target["commit"],
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
    baseline_ref: str,
) -> list[dict[str, Any]]:
    rows_by_ref: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_ref.setdefault(row["candidate_ref"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for target in targets:
        if target["ref"] == baseline_ref:
            continue
        branch_rows = rows_by_ref.get(target["ref"], [])
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
        summaries.append(
            {
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
        )
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


def write_markdown_report(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    branch_summaries: list[dict[str, Any]],
    baseline_ref: str,
    baseline_commit: str,
) -> None:
    rows_by_ref: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_ref.setdefault(row["candidate_ref"], []).append(row)

    lines = [
        "# XLA HLO Performance Comparison",
        "",
        f"Baseline: `{markdown_escape(baseline_ref)}` (`{baseline_commit}`)",
        "",
        "Lower is better. Ratio is `candidate / baseline`: below 1.0× is "
        "faster; above 1.0× is slower.",
        "",
        "> These are isolated HLO module timings, not end-to-end model latency. "
        "Apply a noise threshold before declaring a regression or benefit.",
        "",
    ]

    for branch in branch_summaries:
        candidate_ref = branch["candidate_ref"]
        lines.extend(
            [
                f"## `{markdown_escape(candidate_ref)}`",
                "",
                f"Commit: `{branch['candidate_commit']}`",
                "",
                "| metric | baseline | candidate | ratio | change |",
                "|---|---:|---:|---:|---:|",
                (
                    "| matched module suite | "
                    f"{format_ms(branch['baseline_suite_ms'])} | "
                    f"{format_ms(branch['candidate_suite_ms'])} | "
                    f"{format_ratio(branch['suite_ratio'])} | "
                    f"{format_change(branch['suite_delta_percent'])} |"
                ),
                (
                    "| geometric-mean module latency | 1.000× | "
                    f"{format_ratio(branch['geomean_module_ratio'])} | "
                    f"{format_ratio(branch['geomean_module_ratio'])} | "
                    f"{format_change(branch['geomean_module_delta_percent'])} |"
                ),
                (
                    "| median module latency | 1.000× | "
                    f"{format_ratio(branch['median_module_ratio'])} | "
                    f"{format_ratio(branch['median_module_ratio'])} | "
                    f"{format_change(branch['median_module_delta_percent'])} |"
                ),
                "",
                (
                    f"Coverage: {branch['matched_modules']} matched, "
                    f"{branch['faster_modules']} faster, "
                    f"{branch['slower_modules']} slower, "
                    f"{branch['unchanged_modules']} unchanged, "
                    f"{branch['missing_candidate']} missing candidate, "
                    f"{branch['missing_baseline']} missing baseline."
                ),
                "",
                "### Per-HLO comparison",
                "",
                "| metric | baseline | candidate | ratio | change |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        branch_rows = sorted(
            rows_by_ref.get(candidate_ref, []),
            key=lambda row: (
                row["delta_percent"] is None,
                -(row["delta_percent"] or 0.0),
                row["workload"],
                row["module"],
            ),
        )
        for row in branch_rows:
            metric = markdown_escape(f"{row['workload']} / {row['module']}")
            lines.append(
                f"| {metric} | {format_ms(row['baseline_ms'])} | "
                f"{format_ms(row['candidate_ms'])} | "
                f"{format_ratio(row['ratio'])} | "
                f"{format_change(row['delta_percent'], row['status'])} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison(
    *,
    output_dir: Path,
    targets: list[dict[str, str]],
    baseline_ref: str,
) -> dict[str, Any]:
    rows, summary = comparison_rows(
        output_dir=output_dir, targets=targets, baseline_ref=baseline_ref
    )
    branch_summaries = summarize_branches(rows, targets, baseline_ref)
    csv_path = output_dir / "comparison.csv"
    fields = [
        "baseline_ref",
        "baseline_commit",
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
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    branch_summary_path = output_dir / "branch_summary.csv"
    branch_fields = [
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
    with branch_summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=branch_fields)
        writer.writeheader()
        writer.writerows(branch_summaries)

    summary["branches"] = branch_summaries
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
        baseline_ref=baseline_ref,
        baseline_commit=summary["baseline_commit"],
    )
    return {
        **summary,
        "csv": str(csv_path),
        "branch_summary_csv": str(branch_summary_path),
        "summary_json": str(summary_path),
        "markdown_report": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-ref", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = write_comparison(
            output_dir=args.output_dir,
            targets=manifest["targets"],
            baseline_ref=args.baseline_ref,
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
