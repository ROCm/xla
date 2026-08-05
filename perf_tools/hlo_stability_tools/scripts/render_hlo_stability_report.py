#!/usr/bin/env python3
"""Render a self-contained HTML report from formal HLO stability outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from file_util import (
    repository_identity,
    sha256_file,
    tooling_metadata,
)


RENDER_TOOLING_FILES = (
    "render_hlo_stability_report.py",
    "file_util.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def truth(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def round_sort_key(round_id: str) -> tuple[int, int | str]:
    return (0, int(round_id)) if round_id.isdigit() else (1, round_id)


def label(row: dict[str, str]) -> str:
    return row.get("label") or row["role"]


def format_optional_percent(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return f"{float(value):+.2f}%" if value else "N/A"


def summary_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        evidence = row["evidence_summary"]
        distribution_instability = (
            row.get("distribution_instability", "").lower() == "true"
        )
        tone = (
            "bad"
            if distribution_instability
            else "good"
            if "faster" in evidence
            else "bad"
            if "slower" in evidence
            else "neutral"
        )
        body.append(
            f"""<tr>
<td><strong>{escape(label(row))}</strong><br>
<code>{escape(row.get("commit", "")[:12])}</code></td>
<td>{number(row, "raw_median_ms"):.4f} ms<br>
<small>CV={number(row, "raw_cv_percent"):.2f}%</small></td>
<td>{number(row, "clean_median_ms"):.4f} ms</td>
<td>{number(row, "clean_cv_percent"):.2f}%</td>
<td>{row["outlier_count"]} / {row["raw_count"]}<br>
<small>{number(row, "outlier_rate_percent"):.1f}% ·
high={row.get("high_outlier_count", "0")} ·
low={row.get("low_outlier_count", "0")}</small></td>
<td>{format_optional_percent(row, "paired_median_vs_control_percent")}<br>
<small>pairs={row.get("paired_count", "N/A")}, excluded={number(row, "paired_exclusion_rate_percent"):.1f}%,
MAD={format_optional_percent(row, "paired_mad_percent")}</small></td>
<td>{format_optional_percent(row, "median_vs_historical_percent")}</td>
<td>{format_optional_percent(row, "late_vs_early_percent")}<br>
<small>{escape(row["temporal_evidence"])} · outliers
{number(row, "early_outlier_rate_percent"):.1f}%→{number(row, "late_outlier_rate_percent"):.1f}%</small></td>
<td class="{tone}">{escape(evidence)}</td>
</tr>"""
        )
    return """<div class="table-wrap"><table>
<thead><tr><th>Target</th><th>Raw latency</th><th>Clean median</th><th>Clean CV</th>
<th>Outliers</th><th>Clean-mode paired vs control</th>
<th>vs historical</th><th>Last vs first half</th><th>Evidence</th></tr></thead>
<tbody>""" + "".join(body) + "</tbody></table></div>"


def trend_svg(
    long_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]
) -> str:
    roles = [row["role"] for row in summary_rows]
    labels = {row["role"]: label(row) for row in summary_rows}
    rounds = sorted({row["round"] for row in long_rows}, key=round_sort_key)
    by_role = {
        role: {
            row["round"]: row
            for row in long_rows
            if row["role"] == role
        }
        for role in roles
    }
    values = [float(row["latency_ms"]) for row in long_rows]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.08, 0.02)
    y_min, y_max = low - padding, high + padding
    width, height = 1080, 390
    left, right, top, bottom = 76, 24, 24, 62
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x(index: int) -> float:
        return (
            left + plot_width / 2
            if len(rounds) == 1
            else left + plot_width * index / (len(rounds) - 1)
        )

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    grid = "".join(
        f'<line class="grid" x1="{left}" y1="{y(tick):.2f}" '
        f'x2="{width-right}" y2="{y(tick):.2f}"/>'
        f'<text class="axis" x="{left-9}" y="{y(tick)+4:.2f}" '
        f'text-anchor="end">{tick:.3f}</text>'
        for tick in [
            y_min + (y_max - y_min) * index / 4 for index in range(5)
        ]
    )
    series = []
    for role_index, role in enumerate(roles):
        coordinates = [
            (
                x(index),
                y(float(by_role[role][round_id]["latency_ms"])),
                by_role[role][round_id],
            )
            for index, round_id in enumerate(rounds)
        ]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {cx:.2f} {cy:.2f}"
            for index, (cx, cy, _) in enumerate(coordinates)
        )

        def marker(cx: float, cy: float, row: dict[str, str]) -> str:
            title = (
                f"<title>{escape(labels[role])} round "
                f"{escape(row['round'])}: "
                f"{float(row['latency_ms']):.4f} ms</title>"
            )
            marker_class = f"series-{role_index}"
            if truth(row["is_outlier"]):
                return (
                    f'<circle class="{marker_class} outlier-dot" '
                    f'cx="{cx:.2f}" cy="{cy:.2f}" r="6">{title}</circle>'
                )
            if role_index == 0:
                return (
                    f'<circle class="{marker_class}" cx="{cx:.2f}" '
                    f'cy="{cy:.2f}" r="4.5">{title}</circle>'
                )
            if role_index == 2:
                return (
                    f'<rect class="{marker_class}" x="{cx-4:.2f}" '
                    f'y="{cy-4:.2f}" width="8" height="8">{title}</rect>'
                )
            if role_index == 3:
                points = (
                    f"{cx:.2f},{cy-5:.2f} {cx+5:.2f},{cy:.2f} "
                    f"{cx:.2f},{cy+5:.2f} {cx-5:.2f},{cy:.2f}"
                )
                return (
                    f'<polygon class="{marker_class}" points="{points}">'
                    f"{title}</polygon>"
                )
            return (
                f'<circle class="{marker_class}" cx="{cx:.2f}" '
                f'cy="{cy:.2f}" r="4">{title}</circle>'
            )

        dots = "".join(
            marker(cx, cy, row) for cx, cy, row in coordinates
        )
        series.append(
            f'<path class="series-{role_index}" d="{path}"/>{dots}'
        )
    x_labels = "".join(
        f'<text class="axis" x="{x(index):.2f}" y="{height-bottom+22}" '
        f'text-anchor="middle">{escape(round_id)}</text>'
        for index, round_id in enumerate(rounds)
        if index % max(1, len(rounds) // 12) == 0
    )
    legend = "".join(
        f'<span><i class="series-key-{index}"></i>{escape(labels[role])}</span>'
        for index, role in enumerate(roles)
    )
    return f"""<div class="chart-wrap">
<svg viewBox="0 0 {width} {height}" role="img"
 aria-label="All retained raw round latencies by target">
{grid}{"".join(series)}{x_labels}
<text class="axis-title" x="18" y="{top+plot_height/2}" text-anchor="middle"
 transform="rotate(-90 18 {top+plot_height/2})">Raw latency (ms)</text>
<text class="axis-title" x="{left+plot_width/2}" y="{height-8}"
 text-anchor="middle">Round</text>
</svg>
<div class="legend">{legend}<span><i class="outlier-key"></i>Flagged outlier</span></div>
</div>"""


def heatmap(
    long_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    threshold: float,
) -> str:
    roles = [row["role"] for row in summary_rows]
    labels = {row["role"]: label(row) for row in summary_rows}
    rounds = sorted({row["round"] for row in long_rows}, key=round_sort_key)
    indexed = {(row["round"], row["role"]): row for row in long_rows}
    body = []
    for round_id in rounds:
        first = indexed[(round_id, roles[0])]
        cells = []
        for role in roles:
            row = indexed[(round_id, role)]
            delta = float(row["normalized_delta_percent"])
            tone = (
                "heat-outlier"
                if truth(row["is_outlier"])
                else "heat-high"
                if delta >= threshold
                else "heat-low"
                if delta <= -threshold
                else "heat-normal"
            )
            cells.append(
                f'<td class="{tone}">{delta:+.2f}%<br>'
                f'<small>{float(row["latency_ms"]):.4f} ms</small></td>'
            )
        body.append(
            f"<tr><td>{escape(round_id)}</td>"
            f"<td><code>{escape(first['execution_order'])}</code></td>"
            + "".join(cells)
            + "</tr>"
        )
    headers = "".join(
        f"<th>{escape(labels[role])}</th>" for role in roles
    )
    return f"""<div class="table-wrap"><table>
<thead><tr><th>Round</th><th>Execution order</th>{headers}</tr></thead>
<tbody>{"".join(body)}</tbody></table></div>"""


def raw_appendix(
    paired_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]
) -> str:
    roles = [row["role"] for row in summary_rows]
    control = "control" if "control" in roles else roles[0]
    labels = {row["role"]: label(row) for row in summary_rows}
    headers = ["Round", "Execution order"]
    for role in roles:
        headers.extend(
            [
                f"{labels[role]} ms",
                f"{labels[role]} timestamp",
                f"{labels[role]} outlier",
            ]
        )
    headers.extend(
        f"{labels[role]} vs control"
        for role in roles
        if role != control
    )
    body = []
    for row in paired_rows:
        cells = [row["round"], row["execution_order"]]
        for role in roles:
            cells.extend(
                [
                    f"{float(row[f'{role}_ms']):.4f}",
                    row[f"{role}_timestamp"],
                    row[f"{role}_outlier"],
                ]
            )
        cells.extend(
            (
                f"{float(row[f'{role}_vs_{control}_percent']):+.2f}% "
                f"({'included' if truth(row[f'{role}_pair_analysis_included']) else 'excluded'})"
            )
            for role in roles
            if role != control
        )
        body.append(
            "<tr>"
            + "".join(f"<td>{escape(cell)}</td>" for cell in cells)
            + "</tr>"
        )
    return (
        '<div class="table-wrap raw-table"><table><thead><tr>'
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def provenance_table(rows: list[dict[str, str]]) -> str:
    body = "".join(
        f"<tr><td>{escape(label(row))}</td>"
        f"<td><code>{escape(row.get('source_ref', ''))}</code></td>"
        f"<td><code>{escape(row.get('commit', ''))}</code></td>"
        f"<td><code>{escape(row.get('runner_sha256', ''))}</code></td></tr>"
        for row in rows
    )
    return """<div class="table-wrap"><table>
<thead><tr><th>Target</th><th>Source revision</th><th>Commit</th>
<th>Runner SHA256</th></tr></thead><tbody>""" + body + "</tbody></table></div>"


def input_provenance(metadata: dict[str, Any]) -> str:
    source = metadata.get("runner_source")
    if not isinstance(source, dict):
        source = {}
    collection = metadata.get("collection", {})
    hlo = collection.get("hlo_input", {})
    evaluator = collection.get("evaluation_script", {})
    reference = collection.get("reference_csv")
    rows = [
        (
            "Runner bundle manifest",
            source.get("directory_name", ""),
            source.get("manifest_sha256", ""),
        ),
        (
            "HLO input",
            hlo.get("file_name", ""),
            hlo.get("sha256", ""),
        ),
        (
            "Evaluation script",
            evaluator.get("relative_path", ""),
            evaluator.get("sha256", ""),
        ),
    ]
    dependencies = collection.get("evaluation_dependencies", [])
    if isinstance(dependencies, list):
        rows.extend(
            (
                "Evaluator dependency",
                dependency.get("relative_path", ""),
                dependency.get("sha256", ""),
            )
            for dependency in dependencies
            if isinstance(dependency, dict)
        )
    if isinstance(reference, dict):
        rows.append(
            (
                "Historical CSV",
                reference.get("file_name", ""),
                reference.get("sha256", ""),
            )
        )
    body = "".join(
        f"<tr><td>{escape(kind)}</td><td><code>{escape(name)}</code></td>"
        f"<td><code>{escape(digest)}</code></td></tr>"
        for kind, name, digest in rows
    )
    table = """<div class="table-wrap"><table>
<thead><tr><th>Input</th><th>Portable name</th><th>SHA256</th></tr></thead>
<tbody>""" + body + "</tbody></table></div>"
    snapshots = bool(collection.get("capture_system_snapshots", False))
    snapshot_text = (
        "enabled; point-in-time ROCm/host context is retained for manual "
        "inspection"
        if snapshots
        else "disabled"
    )
    return (
        table
        + '<p class="note"><strong>Fixed repeat policy:</strong> '
        + f"{escape(collection.get('num_repeats', ''))} repeats "
        + "(first repeat is the runner warm-up), uninitialized arguments, "
        + "HIP command buffers disabled, size order. "
        + "<strong>Requested process settle:</strong> "
        + f"{escape(collection.get('runner_settle_sec', ''))}s. "
        + f"<strong>System snapshots:</strong> {escape(snapshot_text)}. "
        + "Snapshots are coarse diagnostic context, are not used in the "
        + "performance evidence classification, and do not replace rocprofv3 "
        + "or Compute Viewer.</p>"
    )


def render(root: Path, artifact_prefix: str = "") -> str:
    metadata = read_json(root / "experiment_metadata.json")
    analysis = read_json(root / "stability_analysis.json")
    summary_rows = read_csv(root / "stability_summary.csv")
    long_rows = read_csv(root / "raw_rounds_long.csv")
    paired_rows = read_csv(root / "paired_deltas.csv")
    if analysis.get("schema_version") != 2:
        raise ValueError("stability HTML requires schema-v2 analysis")
    if metadata.get("schema_version") != 2 or metadata.get("status") not in {
        "analyzed",
        "completed",
    }:
        raise ValueError("stability HTML requires analyzed experiment metadata")
    artifact_hashes = metadata.get("analysis_artifacts")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("experiment metadata has no analysis artifact hashes")
    artifact_names = (
        "stability_analysis.json",
        "stability_summary.csv",
        "raw_rounds_long.csv",
        "paired_deltas.csv",
    )
    for name in artifact_names:
        if artifact_hashes.get(name) != sha256_file(root / name):
            raise ValueError(f"analysis artifact checksum mismatch: {name}")
    if any(row.get("schema_version") != "2" for row in summary_rows):
        raise ValueError("stability summary CSV schema mismatch")
    if any(row.get("schema_version") != "2" for row in long_rows):
        raise ValueError("raw-round CSV schema mismatch")
    if any(row.get("schema_version") != "2" for row in paired_rows):
        raise ValueError("paired-delta CSV schema mismatch")
    roles = analysis.get("roles")
    if not isinstance(roles, list) or roles != [
        row["role"] for row in summary_rows
    ]:
        raise ValueError("analysis/summary role mismatch")
    round_count = analysis.get("round_count")
    design = metadata.get("design")
    if (
        not isinstance(design, dict)
        or design.get("roles") != roles
        or design.get("rounds") != round_count
    ):
        raise ValueError(
            "analysis does not match the experiment design roles/rounds"
        )
    if type(round_count) is not int or round_count < 1:
        raise ValueError("analysis has no valid round count")
    if len(long_rows) != round_count * len(roles):
        raise ValueError("raw-round row count does not match roles and rounds")
    if len(paired_rows) != round_count:
        raise ValueError("paired row count does not match analysis rounds")
    control = next(
        (row for row in summary_rows if row["role"] == "control"), None
    )
    if control is None:
        raise ValueError("stability report requires a control role")
    candidate_evidence = " ".join(
        f"<strong>{escape(label(row))}</strong>: "
        f"{escape(row['evidence_summary'])} "
        f"({format_optional_percent(row, 'paired_median_vs_control_percent')})."
        for row in summary_rows
        if row["role"] != "control"
    )
    unstable_rows = [
        row
        for row in summary_rows
        if row.get("distribution_instability", "").lower() == "true"
    ]
    stability_warning = (
        "<p class=\"note\"><strong>Distribution instability:</strong> "
        + "; ".join(
            f"{escape(label(row))}: "
            f"{escape(row.get('stability_evidence', 'unstable distribution'))}"
            for row in unstable_rows
        )
        + ". Clean-mode comparisons do not describe this full distribution.</p>"
        if unstable_rows
        else ""
    )
    labels_by_role = {
        row["role"]: label(row) for row in summary_rows
    }
    identity_warnings = analysis.get("identity_warnings", [])
    identity_warning = (
        "<p class=\"note\"><strong>Runner identity warning:</strong> "
        + " ".join(
            f"{escape(', '.join(labels_by_role.get(role, role) for role in warning.get('roles', [])))} "
            f"share runner SHA256 {escape(str(warning.get('runner_sha256', ''))[:12])}; "
            "performance differences cannot be attributed to different runner code."
            for warning in identity_warnings
            if isinstance(warning, dict)
        )
        + "</p>"
        if identity_warnings
        else ""
    )
    historical = analysis.get("historical_reference")
    historical_text = (
        f"Live control clean median is "
        f"{historical['control_clean_median_vs_historical_percent']:+.2f}% "
        "versus the supplied historical timing."
        if historical
        else "No historical timing was supplied."
    )
    historical_identity_note = (
        f" <span class=\"muted\">{escape(historical['identity_note'])}</span>"
        if isinstance(historical, dict)
        and not historical.get("source_commit_verified", False)
        and historical.get("identity_note")
        else ""
    )
    total_outliers = sum(int(row["outlier_count"]) for row in summary_rows)
    heatmap_threshold = float(
        analysis.get("outlier_policy", {}).get(
            "minimum_outlier_percent", 2.0
        )
    )
    normalized_prefix = artifact_prefix.strip("/")

    def artifact_href(name: str) -> str:
        return (
            f"{normalized_prefix}/{name}"
            if normalized_prefix
            else name
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HLO Stability Evidence Report</title>
<style>
:root{{--bg:#f5f7fa;--surface:#fff;--ink:#172033;--muted:#667085;--line:#d8dee8;
--accent:#2457d6;--good:#087a55;--good-bg:#e7f7f0;--bad:#b42318;--bad-bg:#fff0ee;
--warn:#9a6700;--warn-bg:#fff7df}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 Inter,system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:28px}}
h1{{margin:0;font-size:28px}}h2{{margin:0 0 12px;font-size:20px}}p{{margin:6px 0}}
.muted{{color:var(--muted)}}.eyebrow{{color:var(--accent);font-size:11px;font-weight:700;
letter-spacing:.08em;text-transform:uppercase}}.hero{{background:var(--surface);
border:1px solid var(--line);border-radius:9px;padding:20px;margin-top:20px}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}}
.stat{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:15px}}
.stat strong{{display:block;font-size:22px}}.stat span{{color:var(--muted);font-size:12px}}
.section{{margin-top:26px}}.table-wrap{{overflow:auto;border:1px solid var(--line);
border-radius:8px;background:var(--surface)}}table{{border-collapse:collapse;width:100%;
min-width:820px}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}}th{{background:#f8fafc;font-size:11px;
text-transform:uppercase;position:sticky;top:0}}code{{font:12px ui-monospace,monospace}}
small{{color:var(--muted)}}.good{{color:var(--good);font-weight:650}}
.bad{{color:var(--bad);font-weight:650}}.neutral{{color:var(--muted)}}
.chart-wrap{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px}}
svg{{width:100%;height:auto;display:block}}.grid{{stroke:var(--line);stroke-width:1}}
.axis{{fill:var(--muted);font-size:11px}}.axis-title{{fill:var(--ink);font-size:12px;font-weight:600}}
path[class^=series-]{{fill:none;stroke-width:2.4}}
path.series-0{{stroke:#2457d6;stroke-width:3}}path.series-1{{stroke:#087a55}}
path.series-2{{stroke:#7a4bc2;stroke-width:3}}
path.series-3{{stroke:#b54708}}
circle[class^=series-],rect[class^=series-],polygon[class^=series-]{{fill:var(--surface);stroke-width:2}}
circle.series-0{{fill:#2457d6;stroke:#2457d6}}.series-1{{stroke:#087a55}}
.series-2{{stroke:#7a4bc2}}.series-3{{stroke:#b54708}}
.outlier-dot{{fill:var(--bad)!important;stroke:var(--bad)!important}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;padding:6px}}
.legend i{{display:inline-block;width:18px;height:8px;margin-right:5px;vertical-align:middle;background:none}}
.series-key-0{{border-top:4px solid #2457d6}}.series-key-1{{border-top:3px solid #087a55}}
.series-key-2{{border-top:4px solid #7a4bc2}}.series-key-3{{border-top:3px solid #b54708}}
.outlier-key{{background:var(--bad);height:8px!important;border-radius:8px}}
.heat-normal{{background:#f8fafc}}.heat-high,.heat-outlier{{background:var(--bad-bg);color:var(--bad);font-weight:650}}
.heat-low{{background:var(--good-bg);color:var(--good);font-weight:650}}
details{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px}}
summary{{cursor:pointer;font-weight:700}}.downloads{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.raw-table{{max-height:620px}}.raw-table table{{min-width:1500px}}
.note{{border-left:3px solid var(--warn);background:var(--warn-bg);padding:12px 14px}}
@media(max-width:800px){{.stats{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<div class="eyebrow">Repeated one-HLO evidence · no release verdict</div>
<h1>HLO Stability Evidence Report</h1>
<p class="muted">Experiment: {escape(analysis.get("experiment_name", "stability"))} ·
module: {escape(analysis["module"])} · {analysis["round_count"]} rounds ·
order: {escape(analysis["execution_order_source"])}</p>
<section class="hero"><h2>Evidence summary</h2>
<p>{historical_text}{historical_identity_note} {candidate_evidence}</p>
{stability_warning}{identity_warning}
<p class="muted">Outliers remain in raw evidence. Clean-mode paired comparisons exclude a pair when either target is flagged.</p></section>
<section class="stats">
<div class="stat"><strong>{number(control,"clean_median_ms"):.4f} ms</strong><span>Live-control clean median</span></div>
<div class="stat"><strong>{number(control,"clean_cv_percent"):.2f}%</strong><span>Live-control clean CV</span></div>
<div class="stat"><strong>{total_outliers}</strong><span>Flagged samples retained</span></div>
<div class="stat"><strong>{analysis["round_count"]}</strong><span>Original rounds</span></div>
</section>
<section class="section"><h2>1. Summary and paired evidence</h2>
{summary_table(summary_rows)}</section>
<section class="section"><h2>2. All original round measurements</h2>
{trend_svg(long_rows,summary_rows)}
<p class="note">Series use distinct colors and marker shapes so pinned control and candidate HEAD remain distinguishable when values overlap. High points are evidence events, not deleted samples or automatic failures.</p></section>
<section class="section"><h2>3. Normalized round heatmap</h2>
<p class="muted">Each target is normalized to its own clean median, separating temporal behavior from branch speed.</p>
{heatmap(long_rows,summary_rows,heatmap_threshold)}</section>
<section class="section"><h2>4. Target and runner provenance</h2>
{provenance_table(summary_rows)}
<h2 style="margin-top:16px">Input provenance</h2>
{input_provenance(metadata)}</section>
<section class="section"><h2>5. Raw-data appendix</h2>
<div class="downloads"><a href="{escape(artifact_href("stability_summary.csv"))}">Summary CSV</a> ·
<a href="{escape(artifact_href("raw_rounds_long.csv"))}">Long-form raw CSV</a> ·
<a href="{escape(artifact_href("paired_deltas.csv"))}">Paired CSV</a> ·
<a href="{escape(artifact_href("stability_analysis.json"))}">Analysis JSON</a></div>
<details><summary>Show all {analysis["round_count"]} original rounds</summary>
{raw_appendix(paired_rows,summary_rows)}</details></section>
</main></body></html>"""


def write_stability_report(root: Path, output: Path | None = None) -> Path:
    root = root.resolve()
    output_path = (output or root / "stability_report.html").resolve()
    if output_path.parent != root or output_path.suffix.lower() != ".html":
        raise ValueError(
            "stability report output must be an .html file directly inside "
            "the experiment directory"
        )
    protected = {
        root / "experiment_metadata.json",
        root / "stability_analysis.json",
        root / "stability_summary.csv",
        root / "raw_rounds_long.csv",
        root / "paired_deltas.csv",
    }
    if output_path in protected:
        raise ValueError(
            f"stability report cannot overwrite evidence input: {output_path.name}"
        )
    relative_root = os.path.relpath(root, output_path.parent).replace(
        os.sep, "/"
    )
    artifact_prefix = "" if relative_root == "." else relative_root
    output_path.write_text(
        render(root, artifact_prefix=artifact_prefix),
        encoding="utf-8",
    )
    return output_path


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
    root = args.experiment_dir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else None
    report_path = write_stability_report(root, output)
    metadata_path = root / "experiment_metadata.json"
    metadata = read_json(metadata_path)
    metadata["status"] = "completed"
    metadata["rendered_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    metadata["outputs"] = {
        "analysis_json": "stability_analysis.json",
        "summary_csv": "stability_summary.csv",
        "raw_rounds_csv": "raw_rounds_long.csv",
        "paired_deltas_csv": "paired_deltas.csv",
        "html_report": (
            str(report_path.relative_to(root))
            if report_path.is_relative_to(root)
            else report_path.name
        ),
        "html_report_sha256": sha256_file(report_path),
    }
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    metadata.setdefault("tooling", {}).update(
        tooling_metadata(
            script_dir,
            repo_root,
            RENDER_TOOLING_FILES,
        )
    )
    metadata["render_repository"] = repository_identity(repo_root)
    write_experiment_metadata(metadata_path, metadata)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
