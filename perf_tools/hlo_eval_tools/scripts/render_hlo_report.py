#!/usr/bin/env python3
"""Render a self-contained HTML report from a schema-v2 HLO campaign."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="default: <output-dir>/comparison_report.html",
    )
    parser.add_argument(
        "--top-movers",
        type=int,
        default=12,
        help="number of largest matched module changes to show (default: 12)",
    )
    parser.add_argument(
        "--threshold-percent",
        type=float,
        help="live-control reporting threshold (default: 2%%)",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_comparison_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric_fields = (
        "baseline_ms",
        "candidate_ms",
        "ratio",
        "delta_ms",
        "delta_percent",
    )
    parsed: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for field in numeric_fields:
            raw = row.get(field, "")
            item[field] = float(raw) if raw not in {"", None} else None
        parsed.append(item)
    return parsed


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def short_ref(ref: str) -> str:
    for prefix in ("origin/rocm-jaxlib-", "origin/", "upstream/"):
        if ref.startswith(prefix):
            return ref[len(prefix) :]
    return ref


def candidate_label(item: dict[str, Any]) -> str:
    for key in ("candidate_label", "label"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return short_ref(str(item.get("candidate_ref", item.get("ref", ""))))


def workload_name(workload: str) -> str:
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
        match = re.fullmatch(
            r"(.+)_(training|inference_(\d+gpu))", name[len(prefix) :]
        )
        if match is None:
            break
        model, workload_type, gpu_count = match.groups()
        workload_path = (
            "training"
            if workload_type == "training"
            else f"inference/{gpu_count}"
        )
        return f"{category}/{model}/{workload_path}"
    return name


def branch_sort_key(branch: dict[str, Any]) -> tuple[Any, ...]:
    ref = branch["candidate_ref"]
    match = re.search(r"v(\d+)\.(\d+)\.(\d+)", ref)
    if match:
        return (0, *(int(part) for part in match.groups()), ref)
    if ref == "upstream/main":
        return (2, 0, 0, 0, ref)
    return (1, 0, 0, 0, ref)


def format_ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}×"


def format_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4g} ms"


def format_delta(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def relative_performance_ratio(latency_ratio: float | None) -> float | None:
    """Convert live/baseline latency into baseline/live performance."""
    if (
        latency_ratio is None
        or not math.isfinite(latency_ratio)
        or latency_ratio <= 0
    ):
        return None
    return 1.0 / latency_ratio


def relative_performance_delta(latency_ratio: float | None) -> float | None:
    ratio = relative_performance_ratio(latency_ratio)
    return (ratio - 1.0) * 100.0 if ratio is not None else None


def material_class(value: float | None, threshold: float) -> str:
    """Classify latency change, where lower is better."""
    if value is None:
        return "missing"
    if value <= -threshold:
        return "good"
    if value >= threshold:
        return "bad"
    return "neutral"


def performance_class(value: float | None, threshold: float) -> str:
    """Classify relative-performance change, where higher is better."""
    if value is None:
        return "missing"
    if value >= threshold:
        return "good"
    if value <= -threshold:
        return "bad"
    return "neutral"


def live_control_comparisons(
    branches: list[dict[str, Any]],
    control: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    if not isinstance(control, dict):
        return {}
    control_id = control.get("candidate_id")
    if not isinstance(control_id, str):
        return {}

    def row_key(row: dict[str, Any]) -> tuple[str, str]:
        return str(row["workload"]), str(row["module"])

    control_rows = {
        row_key(row): row
        for row in rows
        if row.get("candidate_id") == control_id
        and isinstance(row.get("candidate_ms"), (int, float))
        and row["candidate_ms"] > 0
    }
    comparisons: dict[str, dict[str, float | int | None]] = {}
    for branch in branches:
        candidate_id = branch.get("candidate_id")
        if not isinstance(candidate_id, str):
            continue
        candidate_rows = {
            row_key(row): row
            for row in rows
            if row.get("candidate_id") == candidate_id
            and isinstance(row.get("candidate_ms"), (int, float))
            and row["candidate_ms"] > 0
        }
        common = sorted(set(control_rows) & set(candidate_rows))
        ratios = [
            float(control_rows[key]["candidate_ms"])
            / float(candidate_rows[key]["candidate_ms"])
            for key in common
        ]
        control_total = sum(
            float(control_rows[key]["candidate_ms"]) for key in common
        )
        candidate_total = sum(
            float(candidate_rows[key]["candidate_ms"]) for key in common
        )
        geomean_ratio = (
            math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
            if ratios
            else None
        )
        suite_ratio = (
            control_total / candidate_total
            if common and candidate_total > 0
            else None
        )
        comparisons[candidate_id] = {
            "matched_modules": len(common),
            "geomean_performance_ratio": geomean_ratio,
            "geomean_performance_delta_percent": (
                (geomean_ratio - 1.0) * 100.0
                if geomean_ratio is not None
                else None
            ),
            "suite_performance_ratio": suite_ratio,
            "suite_performance_delta_percent": (
                (suite_ratio - 1.0) * 100.0
                if suite_ratio is not None
                else None
            ),
        }
    return comparisons


def branch_headline(
    latest: dict[str, Any] | None,
    live_comparison: dict[str, float | int | None] | None,
    threshold: float,
) -> tuple[str, str]:
    if latest is None:
        return "No live candidate result is available", "Candidate summary is empty."
    ref = candidate_label(latest)
    historical_ratio = relative_performance_ratio(
        latest.get("geomean_module_ratio")
    )
    historical_delta = relative_performance_delta(
        latest.get("geomean_module_ratio")
    )
    control_ratio = (
        live_comparison.get("geomean_performance_ratio")
        if live_comparison
        else None
    )
    control_delta = (
        live_comparison.get("geomean_performance_delta_percent")
        if live_comparison
        else None
    )
    if historical_ratio is None:
        return f"{ref} has incomplete comparison data", "Review missing modules before interpreting performance."

    details = (
        f"Relative performance is {format_ratio(historical_ratio)} versus "
        f"historical ({format_delta(historical_delta)})"
    )
    if isinstance(control_ratio, (int, float)) and isinstance(
        control_delta, (int, float)
    ):
        details += (
            f" and {format_ratio(float(control_ratio))} versus live control "
            f"({format_delta(float(control_delta))})."
        )
    else:
        details += "; live-control comparison is unavailable."

    if not isinstance(control_delta, (int, float)):
        return f"{ref} historical performance evidence", details
    if control_delta >= threshold:
        return f"{ref} shows higher performance than live control", details
    if control_delta <= -threshold:
        return f"{ref} shows lower performance than live control", details
    return (
        f"{ref} is {format_delta(float(control_delta))} versus live control",
        details
        + f" The observed difference is within the ±{threshold:.2f}% "
        "reporting band.",
    )


def trend_svg(
    branches: list[dict[str, Any]], threshold: float
) -> str:
    points = []
    for branch in branches:
        geomean = relative_performance_ratio(
            branch.get("geomean_module_ratio")
        )
        if geomean is None:
            continue
        points.append(
            (
                # Structured targets may supply a readable label; legacy text
                # refs fall back to the existing shortened ref.
                candidate_label(branch),
                geomean,
                relative_performance_ratio(branch.get("suite_ratio")),
            )
        )
    if not points:
        return '<p class="muted">No matched performance ratios are available.</p>'

    show_suite = any(
        suite is not None
        and not math.isclose(geomean, suite, rel_tol=1e-9, abs_tol=1e-12)
        for _, geomean, suite in points
    )

    width, height = 980, 350
    left, right, top, bottom = 72, 24, 28, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [
        float(value)
        for _, geomean, suite in points
        for value in ((geomean, suite) if show_suite else (geomean,))
        if value is not None
    ]
    values.extend([1.0, 1.0 - threshold / 100, 1.0 + threshold / 100])
    minimum, maximum = min(values), max(values)
    padding = max((maximum - minimum) * 0.12, 0.01)
    y_min = minimum - padding
    y_max = maximum + padding

    def x_position(index: int) -> float:
        if len(points) == 1:
            return left + plot_width / 2
        return left + plot_width * index / (len(points) - 1)

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    tick_values = [
        y_min + (y_max - y_min) * index / 4 for index in range(5)
    ]
    grid = "".join(
        (
            f'<line class="chart-grid" x1="{left}" y1="{y_position(tick):.2f}" '
            f'x2="{width - right}" y2="{y_position(tick):.2f}"/>'
            f'<text class="axis-label" x="{left - 10}" '
            f'y="{y_position(tick) + 4:.2f}" text-anchor="end">{tick:.3f}×</text>'
        )
        for tick in tick_values
    )
    band_top = y_position(1.0 + threshold / 100)
    band_bottom = y_position(1.0 - threshold / 100)
    band = (
        f'<rect class="review-band" x="{left}" y="{band_top:.2f}" '
        f'width="{plot_width}" height="{band_bottom - band_top:.2f}"/>'
        f'<line class="baseline" x1="{left}" y1="{y_position(1.0):.2f}" '
        f'x2="{width - right}" y2="{y_position(1.0):.2f}"/>'
    )

    def series_path(position: int, css_class: str) -> str:
        series = [
            (point[0], x_position(index), float(point[position]))
            for index, point in enumerate(points)
            if point[position] is not None
        ]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x:.2f} {y_position(value):.2f}"
            for index, (_, x, value) in enumerate(series)
        )
        circles = "".join(
            (
                f'<circle class="{css_class}" cx="{x:.2f}" '
                f'cy="{y_position(value):.2f}" r="4">'
                f"<title>{escape(label)}: {value:.4f}×</title></circle>"
            )
            for label, x, value in series
        )
        return f'<path class="{css_class}" d="{path}"/>{circles}'

    labels = "".join(
        (
            f'<text class="x-label" x="{x_position(index):.2f}" '
            f'y="{height - bottom + 24}" text-anchor="end" '
            f'transform="rotate(-32 {x_position(index):.2f} {height - bottom + 24})">'
            f"{escape(label)}</text>"
        )
        for index, (label, _, _) in enumerate(points)
    )
    return f"""
<div class="chart-wrap">
  <svg class="trend-chart" viewBox="0 0 {width} {height}" role="img"
       aria-label="Relative HLO performance by XLA branch; higher is better">
    {band}{grid}
    {series_path(1, "line-primary")}
    {series_path(2, "line-secondary") if show_suite else ""}
    {labels}
    <text class="axis-title" x="16" y="{top + plot_height / 2}"
          text-anchor="middle"
          transform="rotate(-90 16 {top + plot_height / 2})">Relative performance (higher is better)</text>
    <text class="axis-title" x="{left + plot_width / 2}" y="{height - 8}"
          text-anchor="middle">XLA branch (oldest → newest)</text>
  </svg>
  <div class="legend">
    <span><i class="legend-primary"></i>Geometric-mean relative performance</span>
    {('<span><i class="legend-secondary"></i>Summed-suite relative performance</span>' if show_suite else '')}
    <span><i class="legend-band"></i>±{threshold:.2f}% historical reference band</span>
  </div>
</div>"""


def branch_scorecard(
    branches: list[dict[str, Any]],
    live_comparisons: dict[str, dict[str, float | int | None]],
    threshold: float,
) -> str:
    rows = []
    for branch in branches:
        historical_ratio = relative_performance_ratio(
            branch.get("geomean_module_ratio")
        )
        historical_delta = relative_performance_delta(
            branch.get("geomean_module_ratio")
        )
        live = live_comparisons.get(branch["candidate_id"], {})
        live_ratio = live.get("geomean_performance_ratio")
        live_delta = live.get("geomean_performance_delta_percent")
        rows.append(
            f"""<tr>
  <td><strong>{escape(candidate_label(branch))}</strong><br>
      <code>{escape(branch["candidate_commit"][:12])}</code></td>
  <td>{branch.get("matched_modules", 0)}</td>
  <td>{format_ratio(historical_ratio)}</td>
  <td class="{performance_class(historical_delta, threshold)}">{format_delta(historical_delta)}</td>
  <td>{format_ratio(float(live_ratio)) if isinstance(live_ratio, (int, float)) else "N/A"}</td>
  <td class="{performance_class(float(live_delta), threshold) if isinstance(live_delta, (int, float)) else "missing"}">{format_delta(float(live_delta)) if isinstance(live_delta, (int, float)) else "N/A"}</td>
  <td>{branch.get("faster_modules", 0)} / {branch.get("slower_modules", 0)}</td>
  <td>{branch.get("missing_candidate", 0)} / {branch.get("missing_baseline", 0)}</td>
</tr>"""
        )
    return """
<div class="table-wrap"><table>
<thead><tr>
  <th>Live candidate</th><th>Matched HLOs</th>
  <th>Historical performance</th><th>Historical change</th>
  <th>vs live control</th><th>vs control change</th>
  <th>Faster / slower vs historical</th>
  <th>Missing candidate / baseline</th>
</tr></thead>
<tbody>""" + "".join(rows) + "</tbody></table></div>"


def top_movers_table(
    rows: list[dict[str, Any]], limit: int, threshold: float
) -> tuple[str, list[str]]:
    matched = [
        row
        for row in rows
        if row.get("candidate_role") == "candidate"
        and row.get("delta_percent") is not None
    ]
    matched.sort(key=lambda row: abs(row["delta_percent"]), reverse=True)
    selected = matched[:limit]
    candidate_refs = sorted(
        {row["candidate_ref"] for row in matched},
        key=lambda ref: branch_sort_key({"candidate_ref": ref}),
    )
    html_rows = []
    for row in selected:
        delta = row["delta_percent"]
        html_rows.append(
            f"""<tr data-candidate="{escape(row["candidate_ref"])}">
  <td>{escape(workload_name(row["workload"]))}</td>
  <td><code>{escape(row["module"])}</code></td>
  <td>{escape(candidate_label(row))}</td>
  <td>{format_ms(row.get("baseline_ms"))}</td>
  <td>{format_ms(row.get("candidate_ms"))}</td>
  <td class="{material_class(delta, threshold)}">{format_delta(delta)}</td>
</tr>"""
        )
    return (
        """
<div class="table-wrap"><table id="movers-table">
<thead><tr><th>Workload</th><th>HLO module</th><th>Candidate</th>
<th>Historical</th><th>Live</th><th>Latency change</th></tr></thead>
<tbody>"""
        + "".join(html_rows)
        + "</tbody></table></div>",
        candidate_refs,
    )


def matrix_table(
    rows: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    control: dict[str, Any] | None,
    threshold: float,
) -> str:
    ordered = ([control] if control else []) + branches
    ordered = [branch for branch in ordered if branch is not None]
    refs = [branch["candidate_ref"] for branch in ordered]
    labels = {
        branch["candidate_ref"]: candidate_label(branch)
        for branch in ordered
    }
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["workload"], row["module"])
        entry = metrics.setdefault(
            key,
            {"baseline_ms": row.get("baseline_ms"), "candidates": {}},
        )
        if entry["baseline_ms"] is None and row.get("baseline_ms") is not None:
            entry["baseline_ms"] = row["baseline_ms"]
        entry["candidates"][row["candidate_ref"]] = row

    headers = "".join(
        f"<th>{escape(labels[ref])}</th>" for ref in refs
    )
    body_rows = []
    for (workload, module), metric in sorted(metrics.items()):
        cells = []
        for ref in refs:
            row = metric["candidates"].get(ref)
            if row is None or row.get("candidate_ms") is None:
                cells.append('<td class="missing">N/A</td>')
                continue
            delta = row.get("delta_percent")
            cells.append(
                f'<td class="{material_class(delta, threshold)}">'
                f'{format_ms(row["candidate_ms"])}<br>'
                f'<small>latency {format_ratio(row.get("ratio"))} · '
                f'{format_delta(delta)}</small></td>'
            )
        searchable = f"{workload} {module}".lower()
        body_rows.append(
            f"""<tr data-search="{escape(searchable)}">
  <td>{escape(workload.removesuffix(".csv"))}</td>
  <td><code>{escape(module)}</code></td>
  <td>{format_ms(metric["baseline_ms"])}</td>
  {"".join(cells)}
</tr>"""
        )
    return f"""
<div class="table-wrap matrix-wrap"><table id="matrix-table">
<thead><tr><th>Workload</th><th>HLO module</th><th>Historical baseline</th>{headers}</tr></thead>
<tbody>{"".join(body_rows)}</tbody>
</table></div>"""


def health_table(manifest: dict[str, Any]) -> str:
    results_by_id = {
        result["id"]: result for result in manifest.get("results", [])
    }
    rows = []
    for target_id in manifest.get("comparison_target_ids", []):
        result = results_by_id.get(target_id)
        if result is None:
            continue
        status = result.get("status", "unknown")
        tone = "good" if status == "completed" else "bad"
        rows.append(
            f"""<tr><td>{escape(result.get("role", ""))}</td>
<td>{escape(candidate_label(result))}</td>
<td><code>{escape(result.get("source_ref", result.get("ref", "")))}</code></td>
<td><code>{escape(result.get("commit", "")[:12])}</code></td>
<td class="{tone}">{escape(status)}</td>
<td>{escape(result.get("build_exit_code", "N/A"))}</td>
<td>{escape(result.get("evaluation_exit_code", "N/A"))}</td></tr>"""
        )
    return """
<div class="table-wrap"><table><thead><tr>
<th>Role</th><th>Name</th><th>Source revision</th><th>Commit</th><th>Status</th>
<th>Build exit</th><th>Evaluation exit</th></tr></thead>
<tbody>""" + "".join(rows) + "</tbody></table></div>"


def render_report(
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    threshold_percent: float | None,
    top_movers: int,
) -> str:
    if manifest.get("schema_version") != 2:
        raise ValueError("HTML reporting requires a schema-v2 campaign")
    branches = sorted(
        [
            branch
            for branch in summary.get("branches", [])
            if branch.get("candidate_role") == "candidate"
        ],
        key=branch_sort_key,
    )
    control = summary.get("reference_reproducibility")
    control_latency_delta = (
        control.get("geomean_module_delta_percent")
        if isinstance(control, dict)
        else None
    )
    threshold = (
        threshold_percent
        if threshold_percent is not None
        else 2.0
    )
    live_comparisons = live_control_comparisons(
        branches, control, comparison_rows
    )
    latest = branches[-1] if branches else None
    latest_live = (
        live_comparisons.get(latest["candidate_id"]) if latest else None
    )
    headline, headline_detail = branch_headline(
        latest, latest_live, threshold
    )
    validation = summary.get("validation", {})
    validation_status = validation.get("status", "unknown")
    reference = manifest["reference_dataset"]
    inventory = reference["inventory"]
    movers, mover_refs = top_movers_table(
        comparison_rows, top_movers, threshold
    )
    labels_by_ref = {
        branch["candidate_ref"]: candidate_label(branch)
        for branch in branches
    }
    options = '<option value="">All candidates</option>' + "".join(
        f'<option value="{escape(ref)}">'
        f'{escape(labels_by_ref.get(ref, short_ref(ref)))}</option>'
        for ref in mover_refs
    )
    environment = manifest.get("environment", {})
    benchmark = manifest.get("benchmark", {}).get("effective", {})
    source_caption = (
        f"Source: schema-v2 campaign {manifest.get('created_at', 'unknown')} · "
        f"GPU: {manifest.get('profile', {}).get('reference', {}).get('gpu', 'unknown')} · "
        f"container: {manifest.get('profile', {}).get('reference', {}).get('container', 'unknown')}"
    )
    control_ratio = (
        control.get("geomean_module_ratio")
        if isinstance(control, dict)
        else None
    )
    control_performance_ratio = relative_performance_ratio(control_ratio)
    control_performance_delta = relative_performance_delta(control_ratio)
    control_missing = (
        (control.get("missing_candidate", 0) + control.get("missing_baseline", 0))
        if isinstance(control, dict)
        else 0
    )
    latest_performance_ratio = (
        relative_performance_ratio(latest.get("geomean_module_ratio"))
        if latest
        else None
    )
    latest_performance_delta = (
        relative_performance_delta(latest.get("geomean_module_ratio"))
        if latest
        else None
    )
    latest_control_ratio = (
        latest_live.get("geomean_performance_ratio")
        if latest_live
        else None
    )
    latest_control_delta = (
        latest_live.get("geomean_performance_delta_percent")
        if latest_live
        else None
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XLA HLO Performance Trend Report</title>
<style>
:root {{
  --bg: #f5f7fa; --surface: #ffffff; --ink: #172033; --muted: #667085;
  --line: #d8dee8; --accent: #2457d6; --accent-soft: #e9efff;
  --good: #087a55; --good-bg: #e7f7f0; --bad: #b42318; --bad-bg: #fff0ee;
  --warn: #9a6700; --warn-bg: #fff7df; --neutral-bg: #f0f2f5;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
}}
.page {{ max-width: 1240px; margin: 0 auto; padding: 28px 28px 64px; }}
header {{ display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; }}
h1 {{ margin: 0; font-size: 28px; letter-spacing: -0.02em; }}
h2 {{ margin: 0 0 12px; font-size: 20px; }}
h3 {{ margin: 0; font-size: 16px; }}
p {{ margin: 6px 0; }}
.muted {{ color: var(--muted); }}
.eyebrow {{ color: var(--accent); text-transform: uppercase; letter-spacing: .08em; font-weight: 700; font-size: 11px; }}
.tag {{ display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 4px 9px; background: var(--surface); font-size: 12px; }}
.actions {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
button, select, input {{
  border: 1px solid var(--line); background: var(--surface); color: var(--ink);
  border-radius: 6px; padding: 8px 10px; font: inherit;
}}
button {{ cursor: pointer; }}
.hero {{
  margin-top: 22px; padding: 24px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 10px;
  display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(260px, .5fr); gap: 24px;
}}
.hero h2 {{ font-size: 26px; margin: 4px 0; }}
.decision {{ border-left: 3px solid var(--accent); padding-left: 16px; }}
.section {{ margin-top: 26px; }}
.layout-grid {{ display: grid; gap: 14px; }}
.grid-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.grid-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
.stat strong {{ display: block; font-size: 23px; line-height: 1.2; }}
.stat span {{ color: var(--muted); font-size: 12px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }}
.confidence {{ display: grid; grid-template-columns: 1fr 1.3fr; gap: 16px; }}
.good {{ color: var(--good); font-weight: 650; }}
.bad {{ color: var(--bad); font-weight: 650; }}
.neutral {{ color: var(--muted); }}
.missing {{ color: var(--warn); background: var(--warn-bg); }}
.status-pass {{ color: var(--good); background: var(--good-bg); }}
.status-fail {{ color: var(--bad); background: var(--bad-bg); }}
.status-pill {{ border-radius: 999px; padding: 4px 9px; font-weight: 700; font-size: 12px; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; min-width: 760px; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ background: #f8fafc; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; position: sticky; top: 0; z-index: 1; }}
tbody tr:last-child td {{ border-bottom: 0; }}
tbody tr:hover {{ background: #fafcff; }}
code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }}
small {{ color: var(--muted); }}
.chart-wrap {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
.trend-chart {{ width: 100%; height: auto; display: block; }}
.chart-grid {{ stroke: var(--line); stroke-width: 1; }}
.baseline {{ stroke: #6b7280; stroke-width: 1.5; stroke-dasharray: 5 4; }}
.review-band {{ fill: var(--warn-bg); }}
.line-primary, .line-secondary {{ fill: none; stroke-width: 2.4; }}
path.line-primary, circle.line-primary {{ stroke: var(--accent); }}
circle.line-primary {{ fill: var(--surface); }}
path.line-secondary, circle.line-secondary {{ stroke: #7a4bc2; }}
circle.line-secondary {{ fill: var(--surface); }}
.axis-label, .x-label {{ fill: var(--muted); font-size: 11px; }}
.axis-title {{ fill: var(--ink); font-size: 12px; font-weight: 600; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; padding: 4px 8px 8px; color: var(--muted); font-size: 12px; }}
.legend i {{ display: inline-block; width: 16px; height: 3px; margin-right: 6px; vertical-align: middle; }}
.legend-primary {{ background: var(--accent); }} .legend-secondary {{ background: #7a4bc2; }}
.legend-band {{ background: var(--warn-bg); border: 1px solid #e6c761; height: 9px !important; }}
.toolbar {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }}
.matrix-wrap {{ max-height: 680px; }}
.matrix-wrap th {{ top: 0; }}
.caption {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
.two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
@media (max-width: 820px) {{
  .page {{ padding: 18px 14px 40px; }}
  header, .hero {{ display: block; }}
  .hero .decision {{ margin-top: 18px; }}
  .grid-4, .grid-2, .confidence, .two-column {{ grid-template-columns: 1fr; }}
}}
@media print {{
  body {{ background: white; }}
  .page {{ max-width: none; padding: 0; }}
  .actions, .toolbar input, .toolbar select {{ display: none; }}
  .section {{ break-inside: avoid; }}
  .table-wrap {{ overflow: visible; }}
  .matrix-wrap {{ max-height: none; }}
}}
</style>
</head>
<body>
<main class="page">
<header>
  <div>
    <div class="eyebrow">MI350 · schema-v2 campaign</div>
    <h1>XLA HLO Performance Trend Report</h1>
    <p class="muted">{escape(source_caption)}</p>
  </div>
  <div class="actions">
    <span class="tag">Historical: {escape(reference["xla_commit"][:12])}</span>
    <span class="status-pill {'status-pass' if validation_status == 'passed' else 'status-fail'}">
      Data validation {escape(validation_status)}
    </span>
    <button onclick="window.print()">Print / Save PDF</button>
  </div>
</header>

<section class="hero">
  <div>
    <div class="eyebrow">Executive headline</div>
    <h2>{escape(headline)}</h2>
    <p class="muted">{escape(headline_detail)}</p>
  </div>
  <div class="decision">
    <strong>Evidence rule</strong>
    <p>Candidate-to-live-control performance changes outside ±{threshold:.2f}% are highlighted. Historical drift is reported separately and never expands this band.</p>
  </div>
</section>

<section class="section layout-grid grid-4">
  <div class="stat"><strong>{format_ratio(control_performance_ratio)}</strong><span>Live-control performance vs historical ({format_delta(control_performance_delta)})</span></div>
  <div class="stat"><strong>{format_ratio(latest_performance_ratio)}</strong><span>Newest performance vs historical ({format_delta(latest_performance_delta)})</span></div>
  <div class="stat"><strong>{format_ratio(float(latest_control_ratio)) if isinstance(latest_control_ratio, (int, float)) else "N/A"}</strong><span>Newest performance vs live control ({format_delta(float(latest_control_delta)) if isinstance(latest_control_delta, (int, float)) else "N/A"})</span></div>
  <div class="stat"><strong>{len(branches)}</strong><span>Live branch candidates</span></div>
</section>

<section class="section">
  <h2>1. Measurement confidence</h2>
  <div class="confidence">
    <div class="card">
      <h3>Historical reference</h3>
      <p><strong>{escape(reference["id"])}</strong></p>
      <p class="muted">Commit <code>{escape(reference["xla_commit"])}</code></p>
      <p>{inventory.get("available_count", 0)} reference workloads available; {inventory.get("missing_count", 0)} missing.</p>
    </div>
    <div class="card">
      <h3>Live pinned control</h3>
      <p><strong>{format_ratio(control_performance_ratio)}</strong> relative performance</p>
      <p class="muted">{format_delta(control_performance_delta)} performance versus the same historical commit; latency change is {format_delta(float(control_latency_delta)) if isinstance(control_latency_delta, (int, float)) else "N/A"}.</p>
      <p>{control_missing} missing control/reference modules. This gap is historical reproducibility evidence, not an allowable candidate noise band.</p>
    </div>
  </div>
</section>

<section class="section">
  <h2>2. Branch performance trend</h2>
  {trend_svg(branches, threshold)}
  <p class="caption">{escape(source_caption)} · relative performance is historical latency / live latency under a fixed-work assumption · higher is better. The reference band is descriptive, not a release acceptance rule.</p>
</section>

<section class="section">
  <h2>3. Branch scorecard</h2>
  <p class="muted">Historical performance and candidate-to-live-control performance are shown separately; values above 1.0× are faster under the same fixed-work assumption.</p>
  {branch_scorecard(branches, live_comparisons, threshold)}
</section>

<section class="section">
  <div class="toolbar">
    <div><h2>4. Largest HLO movers</h2><p class="muted">Ranked by absolute historical latency change; positive values are slower.</p></div>
    <label>Candidate <select id="candidate-filter">{options}</select></label>
  </div>
  {movers}
</section>

<section class="section">
  <div class="toolbar">
    <div><h2>5. Per-HLO evidence matrix</h2><p class="muted">Live latency, live/historical latency ratio, and latency change for every measured module; lower is better in this table.</p></div>
    <label>Filter <input id="matrix-search" type="search" placeholder="workload or module"></label>
  </div>
  {matrix_table(comparison_rows, branches, control, threshold)}
</section>

<section class="section">
  <h2>Appendix: target health and provenance</h2>
  {health_table(manifest)}
  <div class="two-column" style="margin-top: 16px">
    <div class="card"><h3>Benchmark settings</h3>
      <p><code>num_repeats={escape(benchmark.get("num_repeats", "N/A"))}</code></p>
      <p><code>arg_mode={escape(benchmark.get("arg_mode", "N/A"))}</code></p>
      <p><code>cmd_buffer={escape(benchmark.get("cmd_buffer", "N/A"))}</code></p>
      <p><code>settle_sec={escape(benchmark.get("settle_sec", "N/A"))}</code></p>
    </div>
    <div class="card"><h3>Runtime environment</h3>
      <p>Host: <code>{escape(environment.get("hostname", "N/A"))}</code></p>
      <p>Platform: {escape(environment.get("platform", "N/A"))}</p>
      <p>Python: {escape(environment.get("python", "N/A"))}</p>
      <p>Campaign status: <strong>{escape(manifest.get("status", "N/A"))}</strong></p>
    </div>
  </div>
</section>
</main>
<script>
const candidateFilter = document.getElementById("candidate-filter");
candidateFilter.addEventListener("change", () => {{
  const selected = candidateFilter.value;
  document.querySelectorAll("#movers-table tbody tr").forEach(row => {{
    row.hidden = selected && row.dataset.candidate !== selected;
  }});
}});
const matrixSearch = document.getElementById("matrix-search");
matrixSearch.addEventListener("input", () => {{
  const query = matrixSearch.value.trim().toLowerCase();
  document.querySelectorAll("#matrix-table tbody tr").forEach(row => {{
    row.hidden = query && !row.dataset.search.includes(query);
  }});
}});
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    if args.top_movers < 1:
        raise SystemExit("--top-movers must be at least 1")
    if args.threshold_percent is not None and args.threshold_percent < 0:
        raise SystemExit("--threshold-percent must be nonnegative")
    output_dir = args.output_dir.expanduser().resolve()
    manifest = load_json(output_dir / "manifest.json")
    summary = load_json(output_dir / "comparison_summary.json")
    rows = load_comparison_rows(output_dir / "comparison.csv")
    rendered = render_report(
        manifest=manifest,
        summary=summary,
        comparison_rows=rows,
        threshold_percent=args.threshold_percent,
        top_movers=args.top_movers,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else output_dir / "comparison_report.html"
    )
    output.write_text(rendered, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
