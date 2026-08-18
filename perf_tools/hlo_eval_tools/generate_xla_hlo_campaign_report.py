#!/usr/bin/env python3
"""Generate a self-contained XLA HLO full-campaign HTML report."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from xla_hlo_campaign_report_parsing import (
    load_json_object,
    parse_workload_hierarchy,
    read_latest_hlo_profile_timings_ms,
)


SUMMARY_RE = re.compile(
    r"^(profiled|resumed|skipped|failed)\s*:\s*(\d+)"
)
RUNNING_RE = re.compile(r"^\*\* Running (.+) \*\*$")
LEAF_RE = re.compile(r"^leaf:\s+(.+)$")
RUN_RE = re.compile(r"^\s*run:\s*N=(\d+)")
PERFORMANCE_REVIEW_THRESHOLD_PERCENT = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_campaign_manifest(
    manifest: dict[str, Any], manifest_path: Path
) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError(
            f"unsupported campaign manifest schema {schema_version!r}: "
            f"{manifest_path}"
        )
    targets = manifest.get("targets")
    results = manifest.get("results")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"campaign manifest has no targets: {manifest_path}")
    if not isinstance(results, list):
        raise ValueError(f"campaign manifest has no results: {manifest_path}")
    workloads = (
        manifest.get("reference_dataset", {})
        .get("inventory", {})
        .get("workloads")
    )
    if not isinstance(workloads, list):
        raise ValueError(
            f"campaign manifest has no workload inventory: {manifest_path}"
        )
    required = {"id", "role", "commit", "slug"}
    for index, target in enumerate(targets):
        if not isinstance(target, dict) or not required.issubset(target):
            raise ValueError(
                f"campaign target {index} is incomplete: {manifest_path}"
            )
    target_ids = {target["id"] for target in targets}
    control_id = manifest.get("live_control_id")
    if control_id not in target_ids:
        raise ValueError(
            f"campaign live_control_id does not match a target: "
            f"{manifest_path}"
        )


def extract_workload_leaf_from_hlo_path(path: str) -> str:
    marker = "/hlo_eval_tools/"
    return path.split(marker, 1)[1] if marker in path else Path(path).name


def normalize_evaluation_log_line(line: str) -> str:
    line = re.sub(
        r"^[IWEF]\d{4}\s+\S+\s+\d+\s+[^]]+\]\s*", "", line.strip()
    )
    return line.replace("\x00", "")


def failure_evidence_priority(line: str) -> int:
    lowered = line.lower()
    if (
        "double free" in lowered
        or "malloc_consolidate" in lowered
        or "corrupted double-linked list" in lowered
    ):
        return 110
    if "memory access fault" in lowered:
        return 110
    if "resource_exhausted:" in lowered or "out of memory" in lowered:
        return 100
    if "invalid_argument:" in lowered:
        return 100
    if "autotuner could not compile" in lowered:
        return 100
    if "no valid config found" in lowered:
        return 100
    if "segmentation fault" in lowered:
        return 90
    if "hip error" in lowered or "hiperror" in lowered:
        return 90
    if "check failed" in lowered or "fatal" in lowered:
        return 80
    if re.search(
        r"\b(internal|failed_precondition|unimplemented):", lowered
    ):
        return 70
    if "aborted" in lowered or "core dumped" in lowered:
        return 20
    return 0


def classify_failure_signature(raw: str, leaf: str) -> tuple[str, str]:
    cleaned = normalize_evaluation_log_line(raw)
    lowered = cleaned.lower()
    if (
        "autotuner could not compile" in lowered
        or "no valid config found" in lowered
    ):
        detail = (
            " (DEVICE_TYPE_INVALID)"
            if "device_type_invalid" in lowered
            else ""
        )
        return (
            "Triton autotuner",
            f"Autotuner could not compile a valid configuration{detail}",
        )
    if (
        "invalid_argument:" in lowered
        and "expected shape s32" in lowered
        and "incompatible shape f32" in lowered
    ):
        argument = re.search(r"argument\s+(\d+)", cleaned)
        suffix = f" for argument {argument.group(1)}" if argument else ""
        return (
            "Argument dtype mismatch",
            f"Expected s32[N,1] but received f32[N,1]{suffix}",
        )
    if "resource_exhausted:" in lowered or "out of memory" in lowered:
        executable = re.search(r"executable_name='([^']+)'", cleaned)
        suffix = (
            f" in {executable.group(1)}" if executable is not None else ""
        )
        category = "Training OOM" if "/training" in leaf else "Capacity OOM"
        return (
            category,
            f"RESOURCE_EXHAUSTED: device allocation failed{suffix}",
        )
    if "memory access fault" in lowered:
        node = re.search(r"GPU node-?(\d+)", cleaned, re.IGNORECASE)
        suffix = f" on GPU node-{node.group(1)}" if node else ""
        return (
            "GPU memory-access fault",
            f"GPU memory-access fault{suffix}",
        )
    if (
        "double free" in lowered
        or "malloc_consolidate" in lowered
        or "corrupted double-linked list" in lowered
    ):
        return ("Heap corruption", "Heap corruption during runner teardown")
    if "segmentation fault" in lowered:
        return ("Segmentation fault", "Runner segmentation fault")
    if "hip error" in lowered or "hiperror" in lowered:
        code = re.search(r"(?:error|code)\s*[=:]?\s*(\d+)", cleaned, re.I)
        suffix = f" (HIP error {code.group(1)})" if code else ""
        return ("HIP runtime error", f"HIP runtime failure{suffix}")
    status = re.search(
        r"\b(INTERNAL|FAILED_PRECONDITION|UNIMPLEMENTED):\s*([^[]+)",
        cleaned,
    )
    if status:
        message = f"{status.group(1)}: {status.group(2).strip()}"
        return ("Compiler/runtime error", message[:220])
    return ("Runner failure", (cleaned or "Runner exited non-zero")[:220])


def build_focused_reproduction_command(
    *,
    result: dict[str, Any],
    slug: str,
    workload: str,
    module_path: str | None,
    leaf_path: str,
    repeats: int,
    settle_sec: int,
) -> str:
    paths = result.get("paths", {})
    runner = paths.get("runner")
    hlo_input = paths.get("hlo_input", "<perf-tools>/hlo_eval_tools")
    runner_script = f"{str(hlo_input).rstrip('/')}/run_hlo_eval.sh"
    hlo = module_path or leaf_path
    output_dir = "/workspace/debug_space/repro"
    output = f"{output_dir}/{slug}_{workload}.csv"
    runner_argument = str(runner) if runner else "<rebuilt-hlo-runner>"
    command = [
        "bash",
        runner_script,
        runner_argument,
        hlo,
        output,
        str(repeats),
    ]
    prefix = (
        ""
        if runner
        else (
            "# Original runner was not preserved. Rebuild target commit "
            f"{result.get('commit', 'unknown')} and replace "
            "<rebuilt-hlo-runner>.\n"
        )
    )
    return prefix + (
        f"mkdir -p {shlex.quote(output_dir)}\n"
        f"SETTLE_SEC={settle_sec} CMD_BUFFER=off "
        "\\\n  "
        + " \\\n  ".join(shlex.quote(part) for part in command)
    )


def parse_target_evaluation_log(
    path: Path,
    *,
    branch: dict[str, Any],
    result: dict[str, Any],
    benchmark: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract evaluator summary counts and actionable failure evidence."""
    summary: dict[str, Any] = {
        "profiled": None,
        "resumed": None,
        "skipped": None,
        "failed": None,
        "failed_workloads": set(),
        "summary_found": False,
    }
    if not path.is_file():
        return summary, []

    failures: list[dict[str, Any]] = []
    leaf_path = ""
    leaf = ""
    workload = ""
    partitions = 1
    module_path: str | None = None
    candidate_error = ""
    candidate_priority = 0
    candidate_line = 0
    in_summary = False
    collecting_failed_workloads = False

    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if stripped == "==== summary ====":
                in_summary = True
                collecting_failed_workloads = False
                summary.update(
                    {
                        "profiled": 0,
                        "resumed": 0,
                        "skipped": 0,
                        "failed": 0,
                        "failed_workloads": set(),
                        "summary_found": True,
                    }
                )
                continue
            if in_summary:
                summary_match = SUMMARY_RE.match(stripped)
                if summary_match:
                    name = summary_match.group(1)
                    summary[name] = int(summary_match.group(2))
                    collecting_failed_workloads = name == "failed"
                    continue
                if collecting_failed_workloads and stripped.startswith("- "):
                    stem = Path(stripped[2:].strip()).name
                    summary["failed_workloads"].add(
                        stem.removesuffix(".csv")
                    )
                    continue
                if stripped and not stripped.startswith("- "):
                    collecting_failed_workloads = False

            leaf_match = LEAF_RE.match(stripped)
            if leaf_match:
                leaf_path = leaf_match.group(1)
                leaf = extract_workload_leaf_from_hlo_path(leaf_path)
                workload = leaf.replace("/", "_")
                partitions = 1
                module_path = None
                candidate_error = ""
                candidate_priority = 0
                candidate_line = 0
                continue

            run_match = RUN_RE.match(line)
            if run_match:
                partitions = int(run_match.group(1))

            running_match = RUNNING_RE.match(stripped)
            if running_match:
                module_path = running_match.group(1)
                candidate_error = ""
                candidate_priority = 0
                candidate_line = 0
                continue

            priority = failure_evidence_priority(stripped)
            if priority > candidate_priority:
                candidate_error = stripped
                candidate_priority = priority
                candidate_line = line_number

            if "FAIL: runner exited non-zero" not in line:
                continue

            inline = line.split("FAIL: runner exited non-zero", 1)[0].strip()
            inline_priority = failure_evidence_priority(inline)
            if inline_priority >= candidate_priority and inline_priority:
                candidate_error = inline
                candidate_priority = inline_priority
                candidate_line = line_number
            if not leaf:
                output_match = re.search(r"->\s+(.+)$", line)
                if output_match:
                    workload = Path(output_match.group(1)).name
                    leaf = workload
                    leaf_path = leaf

            category, signature = classify_failure_signature(
                candidate_error, leaf
            )
            group = parse_workload_hierarchy(leaf)
            module = (
                PurePosixPath(module_path).name
                if module_path is not None
                else "leaf-level failure"
            )
            failures.append(
                {
                    "id": len(failures),
                    "branch": branch["slug"],
                    "branch_label": branch["label"],
                    "branch_ref": branch["ref"],
                    "commit": branch["commit"],
                    "workload": workload,
                    "leaf": leaf,
                    "leaf_path": leaf_path,
                    "partitions": partitions,
                    "module": module,
                    "module_path": module_path,
                    "category": category,
                    "signature": signature,
                    "raw_error": normalize_evaluation_log_line(candidate_error)
                    or "Runner exited non-zero",
                    "log_path": str(path),
                    "log_uri": path.resolve().as_uri(),
                    "log_line": line_number,
                    "root_error_line": candidate_line or line_number,
                    "repro_command": build_focused_reproduction_command(
                        result=result,
                        slug=branch["slug"],
                        workload=workload,
                        module_path=module_path,
                        leaf_path=leaf_path,
                        repeats=int(benchmark.get("num_repeats", 1)),
                        settle_sec=int(benchmark.get("settle_sec", 0)),
                    ),
                    "domain": group["category"],
                    "model": group["model"],
                    "mode": group["mode"],
                    "gpu": group["gpu"],
                }
            )
    return summary, failures


def target_display_sort_key(target: dict[str, Any]) -> tuple[Any, ...]:
    if target.get("role") in {"control", "live_control"}:
        return (0, 0, 0, 0)
    ref = str(target.get("source_ref") or target.get("ref") or "")
    match = re.search(r"v(\d+)\.(\d+)\.(\d+)", ref)
    if match:
        version = tuple(-int(part) for part in match.groups())
        return (1, *version)
    if ref == "upstream/main":
        return (2, 0, 0, 0)
    return (3, 0, 0, 0)


def calculate_relative_performance(
    reference_ms: float | None,
    candidate_ms: float | None,
) -> tuple[float | None, float | None]:
    """Return reference/candidate performance ratio and percent change."""
    if (
        reference_ms is None
        or candidate_ms is None
        or not math.isfinite(reference_ms)
        or not math.isfinite(candidate_ms)
        or reference_ms <= 0
        or candidate_ms <= 0
    ):
        return None, None
    ratio = reference_ms / candidate_ms
    return ratio, (ratio - 1.0) * 100.0


def classify_live_control_delta(
    delta_percent: float | None,
    threshold_percent: float = PERFORMANCE_REVIEW_THRESHOLD_PERCENT,
) -> str:
    if delta_percent is None:
        return "unavailable"
    if delta_percent > threshold_percent:
        return "higher performance"
    if delta_percent < -threshold_percent:
        return "lower performance"
    return "within review band"


def branch_progression_sort_key(branch: dict[str, Any]) -> tuple[Any, ...]:
    ref = str(branch.get("ref") or "")
    if ref == "upstream/main":
        return (2, 0, 0, 0, 0)
    match = re.search(r"v(\d+)\.(\d+)\.(\d+)", ref)
    if match:
        version = tuple(int(part) for part in match.groups())
        kind = 0 if branch.get("role") == "live_control" else 1
        return (0, *version, kind)
    return (1, 0, 0, 0, 0)


def collect_target_hlo_performance(
    *,
    campaign_dir: Path,
    branch: dict[str, Any],
    inventory: list[dict[str, Any]],
    failed_leaves: set[str],
) -> list[dict[str, Any]]:
    records = []
    csv_dir = campaign_dir / branch["slug"] / "csv"
    for workload in inventory:
        leaf = workload["leaf"]
        csv_path = csv_dir / workload["workload"]
        timings = (
            read_latest_hlo_profile_timings_ms(csv_path)
            if csv_path.is_file()
            else {}
        )
        group = parse_workload_hierarchy(leaf)
        for module in workload.get("modules", []):
            latency = timings.get(module)
            records.append(
                {
                    "branch": branch["slug"],
                    "branch_label": branch["label"],
                    "commit": branch["commit"],
                    "leaf": leaf,
                    "workload": workload["workload"].removesuffix(".csv"),
                    "domain": group["category"],
                    "model": group["model"],
                    "mode": group["mode"],
                    "gpu": group["gpu"],
                    "module": module,
                    "latency_ms": latency,
                    "status": (
                        "fail"
                        if leaf in failed_leaves
                        else "pass"
                        if latency is not None
                        else "missing"
                    ),
                }
            )
    return records


def aggregate_workload_hlo_latencies(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["branch"], record["leaf"])].append(record)

    sums = []
    for rows in grouped.values():
        first = rows[0]
        measured = [
            row
            for row in rows
            if row["status"] == "pass" and row["latency_ms"] is not None
        ]
        failed = any(row["status"] == "fail" for row in rows)
        complete = len(measured) == len(rows) and bool(rows)
        sums.append(
            {
                "branch": first["branch"],
                "branch_label": first["branch_label"],
                "commit": first["commit"],
                "leaf": first["leaf"],
                "workload": first["workload"],
                "domain": first["domain"],
                "model": first["model"],
                "mode": first["mode"],
                "gpu": first["gpu"],
                "module": "All isolated HLO modules",
                "latency_ms": (
                    sum(float(row["latency_ms"]) for row in measured)
                    if complete
                    else None
                ),
                "status": (
                    "fail"
                    if failed
                    else "pass"
                    if complete
                    else "incomplete"
                ),
                "coverage_count": len(measured),
                "module_count": len(rows),
            }
        )
    return sums


def annotate_live_control_performance(
    records: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    threshold_percent: float = PERFORMANCE_REVIEW_THRESHOLD_PERCENT,
) -> None:
    """Mutate records with performance relative to the live control."""
    control = next(
        (branch for branch in branches if branch["role"] == "live_control"),
        None,
    )
    if control is None:
        for record in records:
            record.update(
                {
                    "live_control_ms": None,
                    "relative_performance": None,
                    "live_control_delta_percent": None,
                    "review_direction": "unavailable",
                }
            )
        return
    control_slug = control["slug"]
    control_records = {
        (record["leaf"], record["module"]): record
        for record in records
        if record["branch"] == control_slug
    }
    for record in records:
        control_record = control_records.get(
            (record["leaf"], record["module"])
        )
        live_ms = (
            control_record["latency_ms"]
            if control_record is not None
            and control_record["status"] == "pass"
            else None
        )
        candidate_ms = (
            record["latency_ms"] if record["status"] == "pass" else None
        )
        ratio, delta = calculate_relative_performance(
            live_ms, candidate_ms
        )
        direction = classify_live_control_delta(delta, threshold_percent)
        record.update(
            {
                "live_control_ms": live_ms,
                "relative_performance": ratio,
                "live_control_delta_percent": delta,
                "review_direction": direction,
            }
        )


def summarize_branch_performance_extremes(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] != "pass" or record["latency_ms"] is None:
            continue
        key = (
            record["domain"],
            record["model"],
            record["mode"],
            record["gpu"],
            record["module"],
        )
        grouped[key].append(record)

    extremes = []
    for key, measured in grouped.items():
        fastest = min(measured, key=lambda record: record["latency_ms"])
        slowest = max(measured, key=lambda record: record["latency_ms"])
        extremes.append(
            {
                "domain": key[0],
                "model": key[1],
                "mode": key[2],
                "gpu": key[3],
                "module": key[4],
                "fastest": {
                    "branch": fastest["branch"],
                    "branch_label": fastest["branch_label"],
                    "commit": fastest["commit"],
                    "latency_ms": fastest["latency_ms"],
                    "relative_performance": fastest.get(
                        "relative_performance"
                    ),
                },
                "slowest": {
                    "branch": slowest["branch"],
                    "branch_label": slowest["branch_label"],
                    "commit": slowest["commit"],
                    "latency_ms": slowest["latency_ms"],
                    "relative_performance": slowest.get(
                        "relative_performance"
                    ),
                },
            }
        )
    extremes.sort(
        key=lambda item: (
            item["domain"],
            item["model"],
            item["mode"],
            item["gpu"],
            item["module"],
        )
    )
    return extremes


def build_campaign_report_data(campaign_dir: Path) -> dict[str, Any]:
    """Normalize one completed campaign into the report data model."""
    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    manifest_path = campaign_dir / "manifest.json"
    manifest = load_json_object(manifest_path)
    validate_campaign_manifest(manifest, manifest_path)
    results = {
        result["id"]: result for result in manifest.get("results", [])
    }
    benchmark = manifest.get("benchmark", {}).get("effective") or manifest.get(
        "profile", {}
    ).get("runner", {})
    inventory = manifest["reference_dataset"]["inventory"]["workloads"]
    workload_leaf_by_stem = {
        workload["workload"].removesuffix(".csv"): workload["leaf"]
        for workload in inventory
    }
    workload_stem_by_leaf = {
        leaf: stem for stem, leaf in workload_leaf_by_stem.items()
    }
    control_id = manifest["live_control_id"]
    branches = []
    failures = []
    performance = []

    for target in sorted(
        manifest.get("targets", []), key=target_display_sort_key
    ):
        result = results.get(target["id"], {})
        role = (
            "live_control"
            if target["id"] == control_id
            else target.get("role", "")
        )
        branch = {
            "id": target["id"],
            "slug": target["slug"],
            "label": target.get("label")
            or target.get("source_ref")
            or target.get("ref"),
            "ref": target.get("source_ref") or target.get("ref"),
            "commit": target.get("commit")
            or result.get("revision")
            or result.get("source_head", ""),
            "role": role,
            "source_role": target.get("role", ""),
        }
        eval_log = campaign_dir / branch["slug"] / "eval.log"
        summary, branch_failures = parse_target_evaluation_log(
            eval_log,
            branch=branch,
            result=result,
            benchmark=benchmark,
        )
        failed_leaves = {
            failure["leaf"] for failure in branch_failures
        }
        failed_leaves.update(
            workload_leaf_by_stem[stem]
            for stem in summary["failed_workloads"]
            if stem in workload_leaf_by_stem
        )
        detailed_failure_leaves = {
            failure["leaf"] for failure in branch_failures
        }
        for leaf in sorted(failed_leaves - detailed_failure_leaves):
            group = parse_workload_hierarchy(leaf)
            hlo_root = result.get("paths", {}).get(
                "hlo_input", "<perf-tools>/hlo_eval_tools"
            )
            leaf_path = str(PurePosixPath(str(hlo_root)) / leaf)
            workload = workload_stem_by_leaf.get(
                leaf, leaf.replace("/", "_")
            )
            branch_failures.append(
                {
                    "id": len(branch_failures),
                    "branch": branch["slug"],
                    "branch_label": branch["label"],
                    "branch_ref": branch["ref"],
                    "commit": branch["commit"],
                    "workload": workload,
                    "leaf": leaf,
                    "leaf_path": leaf_path,
                    "partitions": 1,
                    "module": "leaf-level failure",
                    "module_path": None,
                    "category": "Runner failure",
                    "signature": "Runner exited non-zero",
                    "raw_error": (
                        "Workload was listed as failed in the evaluator "
                        "summary without module-level evidence."
                    ),
                    "log_path": str(eval_log),
                    "log_uri": eval_log.resolve().as_uri(),
                    "log_line": 0,
                    "root_error_line": 0,
                    "repro_command": build_focused_reproduction_command(
                        result=result,
                        slug=branch["slug"],
                        workload=workload,
                        module_path=None,
                        leaf_path=leaf_path,
                        repeats=int(benchmark.get("num_repeats", 1)),
                        settle_sec=int(benchmark.get("settle_sec", 0)),
                    ),
                    "domain": group["category"],
                    "model": group["model"],
                    "mode": group["mode"],
                    "gpu": group["gpu"],
                }
            )
        for failure in branch_failures:
            failure["id"] = len(failures)
            failures.append(failure)
        performance.extend(
            collect_target_hlo_performance(
                campaign_dir=campaign_dir,
                branch=branch,
                inventory=inventory,
                failed_leaves=failed_leaves,
            )
        )
        passed = (
            summary["profiled"] + summary["resumed"]
            if summary["summary_found"]
            else None
        )
        failed = summary["failed"]
        denominator = (
            passed + failed
            if passed is not None and failed is not None
            else None
        )
        branches.append(
            {
                **branch,
                "build_exit_code": result.get("build_exit_code"),
                "evaluation_exit_code": result.get("evaluation_exit_code"),
                "result_status": result.get("status", "missing"),
                "pass_count": passed,
                "profiled_count": summary["profiled"],
                "resumed_count": summary["resumed"],
                "failed_count": failed,
                "skipped_count": summary["skipped"],
                "success_percent": (
                    passed / denominator * 100 if denominator else None
                ),
                "log_available": eval_log.is_file(),
                "eval_log": str(eval_log),
                "eval_log_uri": (
                    eval_log.resolve().as_uri()
                    if eval_log.is_file()
                    else None
                ),
            }
        )

    performance_by_workload: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for record in performance:
        performance_by_workload[
            (record["branch"], record["leaf"])
        ].append(record)
    workload_states: dict[tuple[str, str], str] = {}
    for key, records in performance_by_workload.items():
        statuses = {record["status"] for record in records}
        workload_states[key] = (
            "fail"
            if "fail" in statuses
            else "pass"
            if statuses == {"pass"}
            else "missing"
        )

    branch_by_slug = {branch["slug"]: branch for branch in branches}
    failed_by_leaf: dict[str, dict[str, int]] = defaultdict(dict)
    for failure in failures:
        failed_by_leaf[failure["leaf"]][failure["branch"]] = failure["id"]

    matrix = []
    for leaf, branch_failures in failed_by_leaf.items():
        leaf_occurrences = [
            failures[failure_id]
            for failure_id in branch_failures.values()
        ]
        matrix.append(
            {
                "leaf": leaf,
                "category": " / ".join(
                    sorted(
                        {
                            failure["category"]
                            for failure in leaf_occurrences
                        }
                    )
                ),
                "affected_count": len(branch_failures),
                "states": {
                    branch["slug"]: (
                        {
                            "status": "fail",
                            "failure_id": branch_failures[branch["slug"]],
                        }
                        if branch["slug"] in branch_failures
                        else {
                            "status": workload_states.get(
                                (branch["slug"], leaf), "unknown"
                            )
                        }
                    )
                    for branch in branches
                },
            }
        )
    matrix.sort(key=lambda item: (-item["affected_count"], item["leaf"]))

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for failure in failures:
        key = (failure["category"], failure["signature"])
        if key not in grouped:
            grouped[key] = {
                "category": failure["category"],
                "signature": failure["signature"],
                "occurrence_count": 0,
                "workloads": set(),
                "branches": set(),
                "examples": [],
            }
        group = grouped[key]
        group["occurrence_count"] += 1
        group["workloads"].add(failure["leaf"])
        group["branches"].add(failure["branch_label"])
        if len(group["examples"]) < 3:
            group["examples"].append(failure["leaf"])

    signature_groups = []
    for group in grouped.values():
        signature_groups.append(
            {
                **group,
                "workload_count": len(group["workloads"]),
                "workloads": sorted(group["workloads"]),
                "branches": sorted(group["branches"]),
            }
        )
    signature_groups.sort(
        key=lambda item: (-item["occurrence_count"], item["category"])
    )
    performance_sums = aggregate_workload_hlo_latencies(performance)
    annotate_live_control_performance(performance, branches)
    annotate_live_control_performance(performance_sums, branches)
    performance_extremes = summarize_branch_performance_extremes(
        [*performance, *performance_sums]
    )

    environment = manifest.get("environment", {})
    profile = manifest.get("profile", {})
    environment_gpu = environment.get("gpu")
    container = environment.get("container", {})
    return {
        "campaign": {
            "id": campaign_dir.name,
            "status": manifest.get("status"),
            "created_at": manifest.get("created_at"),
            "finished_at": manifest.get("finished_at"),
            "directory": str(campaign_dir),
            "hostname": environment.get("hostname"),
            "platform": environment.get("platform"),
            "gpu": (
                environment_gpu.get("identity")
                if isinstance(environment_gpu, dict)
                else environment_gpu
                if isinstance(environment_gpu, str)
                else profile.get("reference", {}).get("gpu")
            ),
            "container_identity": (
                container.get("identity")
                if isinstance(container, dict)
                else None
            ),
            "container_capture_method": (
                container.get("capture_method")
                if isinstance(container, dict)
                else None
            ),
            "benchmark": benchmark,
        },
        "branches": branches,
        "performance_branch_order": [
            branch["slug"]
            for branch in sorted(
                branches, key=branch_progression_sort_key
            )
        ],
        "performance": performance,
        "performance_sums": performance_sums,
        "performance_review_threshold_percent": (
            PERFORMANCE_REVIEW_THRESHOLD_PERCENT
        ),
        "performance_extremes": performance_extremes,
        "failures": failures,
        "matrix": matrix,
        "signature_groups": signature_groups,
        "summary": {
            "branch_count": len(branches),
            "build_pass_count": sum(
                branch["build_exit_code"] == 0 for branch in branches
            ),
            "pass_count": sum(
                branch["pass_count"] or 0 for branch in branches
            ),
            "failure_count": len(failures),
            "unique_failed_workloads": len(failed_by_leaf),
            "signature_count": len(signature_groups),
        },
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MI350 Full Campaign Report</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--surface:#f6f8fa;--surface2:#eef1f4;--text:#1f2328;--muted:#59636e;--border:#d0d7de;--accent:#0969da;--danger:#cf222e;--success:#1a7f37;--warning:#9a6700;--code:#eff1f3}
@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--surface:#161b22;--surface2:#21262d;--text:#e6edf3;--muted:#9da7b3;--border:#30363d;--accent:#58a6ff;--danger:#ff7b72;--success:#3fb950;--warning:#d29922;--code:#21262d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}main{width:min(1800px,calc(100% - 36px));margin:auto;padding:26px 0 60px}h1{font-size:25px;margin:0}h2{font-size:19px;margin:30px 0 11px}h3{font-size:15px;margin:18px 0 7px}p{margin:6px 0}.muted{color:var(--muted)}code,pre{font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}code{background:var(--code);padding:2px 5px;border-radius:4px}.stats{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:18px 0}.stat{border:1px solid var(--border);border-radius:6px;background:var(--surface);padding:12px}.stat strong{display:block;font-size:22px}.callout{border:1px solid var(--border);border-left:4px solid var(--warning);border-radius:6px;padding:11px 13px;background:var(--surface)}.table-wrap{border:1px solid var(--border);border-radius:6px;overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;min-width:900px}th,td{padding:8px 9px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}th{position:sticky;top:0;background:var(--surface2);z-index:2;font-size:12px}tbody tr:nth-child(even){background:var(--surface)}.pass{color:var(--success)}.fail{color:var(--danger)}.unknown,.warn{color:var(--warning)}.bar{display:flex;width:180px;height:8px;background:var(--surface2);border-radius:4px;overflow:hidden;margin-top:5px}.bar .passed{background:var(--success)}.bar .failed{background:var(--danger)}button{border:1px solid var(--border);border-radius:5px;background:var(--bg);color:var(--accent);padding:4px 7px;cursor:pointer}button:hover{background:var(--surface2)}.matrix td:not(:first-child),.matrix th:not(:first-child){text-align:center;white-space:nowrap}.matrix button.fail-cell{color:var(--danger);border-color:transparent}.controls{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:9px;padding:12px;border:1px solid var(--border);border-radius:6px;background:var(--surface);margin-bottom:10px}label{display:grid;gap:4px;color:var(--muted);font-size:12px}select,input{width:100%;padding:7px;border:1px solid var(--border);border-radius:5px;background:var(--bg);color:var(--text)}.explorer{display:grid;grid-template-columns:minmax(0,3fr) minmax(330px,2fr);gap:12px}.detail{border:1px solid var(--border);border-radius:6px;background:var(--surface);padding:13px;min-width:0}.detail dl{display:grid;grid-template-columns:110px 1fr;gap:5px 9px;margin:8px 0}.detail dt{color:var(--muted)}.detail dd{margin:0;min-width:0;overflow-wrap:anywhere}.detail pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--code);padding:10px;border-radius:5px;max-height:260px;overflow:auto}.detail a{color:var(--accent)}.copy-row{display:flex;align-items:center;gap:8px}.provenance{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.provenance>div{border:1px solid var(--border);border-radius:6px;padding:12px}.signature-example{max-width:420px;overflow-wrap:anywhere}@media(max-width:1150px){.stats{grid-template-columns:repeat(3,1fr)}.explorer{grid-template-columns:1fr}.controls{grid-template-columns:repeat(3,1fr)}}@media(max-width:650px){.stats,.controls,.provenance{grid-template-columns:1fr}}
.performance-controls{grid-template-columns:repeat(6,minmax(130px,1fr))}.performance-chart{border:1px solid var(--border);border-radius:6px;background:var(--surface);overflow:auto;margin-bottom:10px}.chart-head{display:flex;justify-content:space-between;gap:16px;padding:12px 14px 0}.chart-head strong{font-size:15px}.chart-head span{text-align:right}.performance-chart svg{display:block;width:100%;min-width:960px;height:430px}.chart-legend{display:flex;gap:18px;flex-wrap:wrap;padding:0 14px 10px}.legend-mark{display:inline-block;width:11px;height:11px;margin-right:5px;vertical-align:-1px}.legend-line{background:var(--accent);border-radius:50%}.legend-fail{color:var(--danger);font-weight:700}@media(max-width:1150px){.performance-controls{grid-template-columns:repeat(3,1fr)}}@media(max-width:650px){.performance-controls{grid-template-columns:1fr}.chart-head{display:block}.chart-head span{display:block;text-align:left}}
.performance-extremes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:10px 0}.performance-extremes>div{border:1px solid var(--border);border-radius:6px;padding:10px 12px;background:var(--surface)}.performance-extremes strong{display:block}@media(max-width:800px){.performance-extremes{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<h1>MI350 Full Campaign Report</h1>
<p class="muted" id="subtitle"></p>
<div class="stats" id="stats"></div>
<div class="callout"><strong>Interpretation:</strong> “Pass” means the evaluator published a profile CSV for that workload. Failure categories are normalized from the first actionable compiler, runner, or runtime error; downstream “no execution profile” messages are not treated as root signatures.</div>

<h2>System configurations</h2>
<div class="provenance" id="provenance"></div>

<h2>Overall branch status</h2>
<div class="table-wrap"><table><thead><tr><th>Branch</th><th>Commit</th><th>Build</th><th>Pass</th><th>Fail</th><th>Skipped (empty)</th><th>Workload success</th><th>Evidence</th></tr></thead><tbody id="branch-body"></tbody></table></div>

<h2>Model performance across branches</h2>
<p class="muted">Select one model workload and HLO. The graph shows performance relative to the pinned live control (1.0×; higher is faster), while the table preserves absolute latency. The ±2% band is a review threshold—not a confidence interval or proof of regression. Summed latency is the sum of independently executed HLO modules, not end-to-end model latency.</p>
<div class="controls performance-controls">
<label>Domain<select id="perf-domain"></select></label>
<label>Model<select id="perf-model"></select></label>
<label>Mode<select id="perf-mode"></select></label>
<label>GPU / training<select id="perf-gpu"></select></label>
<label>Metric<select id="perf-metric"><option value="module">Selected HLO latency</option><option value="sum">Sum of isolated HLO latencies</option></select></label>
<label>HLO module<select id="perf-module"></select></label>
</div>
<div class="performance-extremes" id="performance-extremes"></div>
<div class="performance-chart">
<div class="chart-head"><strong id="performance-title"></strong><span class="muted" id="performance-caption"></span></div>
<svg id="performance-svg" viewBox="0 0 1120 430" role="img" aria-label="Selected model performance across XLA branches"></svg>
<div class="chart-legend" id="performance-legend"></div>
</div>
<div class="table-wrap" style="max-height:360px"><table><thead><tr><th>Branch</th><th>Commit</th><th>Absolute metric</th><th>vs live control</th><th>Live-control change</th><th>Coverage</th><th>Workload status</th></tr></thead><tbody id="performance-body"></tbody></table></div>

<h2>Failure signatures</h2>
<p class="muted">Grouped branch-workload failures in this campaign. These labels describe observed error signatures, not confirmed root causes.</p>
<div class="table-wrap"><table><thead><tr><th>Category</th><th>Normalized signature</th><th>Occurrences</th><th>Workloads</th><th>Branches</th><th>Examples</th></tr></thead><tbody id="signature-body"></tbody></table></div>

<h2>Where failures occur</h2>
<label style="max-width:520px;margin-bottom:9px">Filter workload matrix<input id="matrix-search" type="search" placeholder="model, workload path, or category"></label>
<div class="table-wrap"><table class="matrix"><thead id="matrix-head"></thead><tbody id="matrix-body"></tbody></table></div>

<h2>Failure explorer and reproduction</h2>
<div class="controls">
<label>Branch<select id="branch-filter"></select></label>
<label>Category<select id="category-filter"></select></label>
<label>Mode<select id="mode-filter"></select></label>
<label>GPU / training<select id="gpu-filter"></select></label>
<label>Search<input id="failure-search" type="search" placeholder="model, HLO, signature"></label>
</div>
<div class="explorer">
<div class="table-wrap"><table><thead><tr><th>Branch</th><th>Workload</th><th>Category</th><th>Failing HLO</th><th>Evidence</th></tr></thead><tbody id="failure-body"></tbody></table></div>
<aside class="detail" id="detail"></aside>
</div>

</main>
<script>
const DATA=__DATA__;
const byId=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const uniq=values=>[...new Set(values.filter(Boolean))].sort();
const option=(value,label=value)=>`<option value="${esc(value)}">${esc(label)}</option>`;
const branchBySlug=Object.fromEntries(DATA.branches.map(branch=>[branch.slug,branch]));
const failureById=Object.fromEntries(DATA.failures.map(failure=>[failure.id,failure]));
const performance=DATA.performance;
const performanceSums=DATA.performance_sums||[];
const performanceExtremes=DATA.performance_extremes||[];
const reviewThreshold=DATA.performance_review_threshold_percent||2;

function renderHeader(){
 const campaign=DATA.campaign,summary=DATA.summary;
 byId("subtitle").textContent=`${campaign.id} · ${campaign.created_at||""} to ${campaign.finished_at||""} · generated __GENERATED__`;
 const stats=[["Campaign",campaign.status],["Branches built",`${summary.build_pass_count}/${summary.branch_count}`],["Pass observations",summary.pass_count],["Failure observations",summary.failure_count],["Unique failed workloads",summary.unique_failed_workloads],["Signature groups",summary.signature_count]];
 byId("stats").innerHTML=stats.map(([label,value])=>`<div class="stat"><strong>${esc(value)}</strong><span class="muted">${esc(label)}</span></div>`).join("");
}
function renderBranches(){
 byId("branch-body").innerHTML=DATA.branches.map(branch=>{const pct=branch.success_percent,build=branch.build_exit_code===0?"Passed":branch.build_exit_code==null?"N/A":`Exit ${esc(branch.build_exit_code)}`,passDetail=branch.pass_count==null?"":`<br><small>${branch.profiled_count} profiled${branch.resumed_count?` + ${branch.resumed_count} resumed`:""}</small>`,success=pct==null?"N/A":`${pct.toFixed(1)}%<div class="bar"><span class="passed" style="width:${pct}%"></span><span class="failed" style="width:${100-pct}%"></span></div>`;return `<tr><td><strong>${esc(branch.label)}</strong><br><span class="muted">${esc(branch.ref)}</span></td><td><code>${esc(branch.commit.slice(0,12))}</code></td><td class="${branch.build_exit_code===0?"pass":branch.build_exit_code==null?"warn":"fail"}">${build}</td><td class="pass">${esc(branch.pass_count??"N/A")}${passDetail}</td><td class="fail">${esc(branch.failed_count??"N/A")}</td><td>${esc(branch.skipped_count??"N/A")}</td><td>${success}</td><td>${branch.log_available?`<a href="${esc(branch.eval_log_uri)}">eval.log</a>`:"Missing log"}</td></tr>`}).join("");
}
function setSelect(id,values,current,preferred){
 const select=byId(id);select.innerHTML=values.map(value=>option(value)).join("");select.value=values.includes(current)?current:values.includes(preferred)?preferred:values[0]||"";
}
function performanceFilters(){
 return {domain:byId("perf-domain").value,model:byId("perf-model").value,mode:byId("perf-mode").value,gpu:byId("perf-gpu").value,metric:byId("perf-metric").value,module:byId("perf-module").value};
}
function cascadePerformance(){
 const previous=performanceFilters();
 setSelect("perf-domain",uniq(performance.map(r=>r.domain)),previous.domain,"vision_diffusion");
 let subset=performance.filter(r=>r.domain===byId("perf-domain").value);
 setSelect("perf-model",uniq(subset.map(r=>r.model)),previous.model,"efficientnet");
 subset=subset.filter(r=>r.model===byId("perf-model").value);
 setSelect("perf-mode",uniq(subset.map(r=>r.mode)),previous.mode,"inference");
 subset=subset.filter(r=>r.mode===byId("perf-mode").value);
 setSelect("perf-gpu",uniq(subset.map(r=>r.gpu)),previous.gpu,"1gpu");
 subset=subset.filter(r=>r.gpu===byId("perf-gpu").value);
 setSelect("perf-module",uniq(subset.map(r=>r.module)),previous.module,"");
 renderPerformance();
}
function performancePoints(){
 const f=performanceFilters(),source=(f.metric==="module"?performance:performanceSums).filter(r=>r.domain===f.domain&&r.model===f.model&&r.mode===f.mode&&r.gpu===f.gpu&&(f.metric!=="module"||r.module===f.module));
 return DATA.performance_branch_order.map(branch=>{const row=source.find(r=>r.branch===branch);if(!row)return {branch,value:null,relative:null,delta:null,status:"missing",coverage:"0/0"};return {...row,value:row.latency_ms,relative:row.relative_performance,delta:row.live_control_delta_percent,coverage:f.metric==="module"?(row.latency_ms==null?"0/1":"1/1"):`${row.coverage_count}/${row.module_count}`}});
}
function shortBranch(branch){
 if(branch.role==="live_control")return branch.label.replace(" pinned live control"," pinned").replace(" pinned baseline"," pinned").replace(" HEAD","");if(branch.ref==="upstream/main")return "upstream/main";const match=branch.ref.match(/v\d+\.\d+\.\d+/);return match?match[0]:branch.label;
}
function formatRatio(value){return value==null?"N/A":value.toFixed(3)+"×"}
function formatPercent(value){return value==null?"N/A":`${value>=0?"+":""}${value.toFixed(2)}%`}
function renderPerformance(){
 const f=performanceFilters(),points=performancePoints(),svg=byId("performance-svg"),width=1120,height=430,left=78,right=30,top=25,bottom=78,order=DATA.performance_branch_order;svg.innerHTML="";
 byId("perf-module").disabled=f.metric==="sum";
 const valid=points.filter(point=>point.relative!=null&&point.relative>0),scaleValues=[.98,1,1.02,...valid.map(point=>point.relative)];let min=Math.min(...scaleValues),max=Math.max(...scaleValues),span=max-min;if(span<.08){const extra=(.08-span)/2;min-=extra;max+=extra;span=max-min}min=Math.max(0,min-span*.12);max+=span*.12;
 const x=branch=>left+(order.indexOf(branch)/Math.max(1,order.length-1))*(width-left-right),y=value=>top+(max-value)/(max-min)*(height-top-bottom),ns="http://www.w3.org/2000/svg";
 const add=(tag,attrs,text)=>{const element=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([key,value])=>element.setAttribute(key,value));if(text!=null)element.textContent=text;svg.appendChild(element);return element};
 const bandTop=y(1+reviewThreshold/100),bandBottom=y(1-reviewThreshold/100);add("rect",{x:left,y:Math.min(bandTop,bandBottom),width:width-left-right,height:Math.abs(bandBottom-bandTop),fill:"var(--warning)","fill-opacity":.12});
 add("line",{x1:left,y1:top,x2:left,y2:height-bottom,stroke:"currentColor","stroke-opacity":.35});add("line",{x1:left,y1:height-bottom,x2:width-right,y2:height-bottom,stroke:"currentColor","stroke-opacity":.35});
 for(let i=0;i<=4;i++){const value=min+(max-min)*i/4,yy=y(value);add("line",{x1:left,y1:yy,x2:width-right,y2:yy,stroke:"currentColor","stroke-opacity":.12});add("text",{x:left-8,y:yy+4,"text-anchor":"end",fill:"currentColor","font-size":11},value.toFixed(3)+"×")}
 add("line",{x1:left,y1:y(1),x2:width-right,y2:y(1),stroke:"currentColor","stroke-width":1.8,"stroke-dasharray":"6 4"});
 order.forEach(branch=>add("text",{x:x(branch),y:height-bottom+23,"text-anchor":"middle",fill:"currentColor","font-size":11},shortBranch(branchBySlug[branch])));
 const segments=[];let current=[];points.forEach(point=>{if(point.relative==null){if(current.length>1)segments.push(current);current=[]}else current.push(point)});if(current.length>1)segments.push(current);segments.forEach(segment=>add("polyline",{points:segment.map(point=>`${x(point.branch)},${y(point.relative)}`).join(" "),fill:"none",stroke:"var(--accent)","stroke-width":2}));
 points.forEach(point=>{if(point.relative!=null){const circle=add("circle",{cx:x(point.branch),cy:y(point.relative),r:5,fill:"var(--accent)"});const title=document.createElementNS(ns,"title");title.textContent=`${branchBySlug[point.branch].label}\n${point.value.toFixed(4)} ms\n${formatRatio(point.relative)} vs live control\n${formatPercent(point.delta)}\ncoverage ${point.coverage}`;circle.appendChild(title)}else if(point.status==="fail"){const xx=x(point.branch),yy=height-bottom-7;add("line",{x1:xx-5,y1:yy-5,x2:xx+5,y2:yy+5,stroke:"var(--danger)","stroke-width":3});add("line",{x1:xx-5,y1:yy+5,x2:xx+5,y2:yy-5,stroke:"var(--danger)","stroke-width":3})}});
 if(!valid.length)add("text",{x:(left+width-right)/2,y:(top+height-bottom)/2,"text-anchor":"middle",fill:"currentColor","font-size":13},"Live-control comparison unavailable for this selection");
 add("text",{x:18,y:(height-bottom+top)/2,transform:`rotate(-90 18 ${(height-bottom+top)/2})`,"text-anchor":"middle",fill:"currentColor","font-size":12},"Relative performance vs live control");add("text",{x:(left+width-right)/2,y:height-12,"text-anchor":"middle",fill:"currentColor","font-size":12},"XLA branch progression");
 const metric=f.metric==="module"?f.module:"isolated HLO sum";byId("performance-title").textContent=`${f.model} · ${f.mode} · ${f.gpu} · ${metric}`;byId("performance-caption").textContent=`Source: ${DATA.campaign.id} live profile CSVs · higher is faster`;
 byId("performance-legend").innerHTML=`<span><i class="legend-mark legend-line"></i>Relative performance</span><span>Dashed line: live control 1.000×</span><span class="warn">±${reviewThreshold.toFixed(2)}% live-control review band</span><span class="legend-fail">× Failed workload</span>`;
 const extremeModule=f.metric==="module"?f.module:"All isolated HLO modules",extreme=performanceExtremes.find(item=>item.domain===f.domain&&item.model===f.model&&item.mode===f.mode&&item.gpu===f.gpu&&item.module===extremeModule);if(extreme){const fastest=extreme.fastest,slowest=extreme.slowest;byId("performance-extremes").innerHTML=`<div><span class="muted">Fastest measured branch</span><strong>${esc(fastest.branch_label)}</strong><code>${esc(fastest.commit.slice(0,12))}</code> · ${fastest.latency_ms.toFixed(4)} ms · ${formatRatio(fastest.relative_performance)}</div><div><span class="muted">Slowest measured branch</span><strong>${esc(slowest.branch_label)}</strong><code>${esc(slowest.commit.slice(0,12))}</code> · ${slowest.latency_ms.toFixed(4)} ms · ${formatRatio(slowest.relative_performance)}</div>`}else byId("performance-extremes").innerHTML=`<div><strong>No complete measurement</strong><span class="muted">Fastest/slowest comparison is unavailable.</span></div>`;
 byId("performance-body").innerHTML=points.map(point=>{const branch=branchBySlug[point.branch],status=point.status==="pass"?"Pass":point.status==="fail"?"Failed":point.status==="incomplete"?"Incomplete coverage":"Missing",direction=point.review_direction==="lower performance"?"fail":point.review_direction==="higher performance"?"pass":"";return `<tr><td>${esc(branch.label)}</td><td><code>${esc(branch.commit.slice(0,12))}</code></td><td>${point.value==null?"N/A":point.value.toFixed(4)+" ms"}</td><td>${formatRatio(point.relative)}</td><td class="${direction}">${formatPercent(point.delta)}</td><td>${esc(point.coverage)}</td><td class="${point.status==="pass"?"pass":point.status==="fail"?"fail":"warn"}">${status}</td></tr>`}).join("");
}
function renderSignatures(){
 byId("signature-body").innerHTML=DATA.signature_groups.map(group=>`<tr><td><button onclick="filterCategory('${esc(group.category)}')">${esc(group.category)}</button></td><td>${esc(group.signature)}</td><td>${group.occurrence_count}</td><td>${group.workload_count}</td><td>${group.branches.length}</td><td class="signature-example">${group.examples.map(esc).join("<br>")}</td></tr>`).join("");
}
function renderMatrix(){
 const query=byId("matrix-search").value.trim().toLowerCase();
 byId("matrix-head").innerHTML=`<tr><th>Workload / signature</th><th>Affected</th>${DATA.branches.map(branch=>`<th title="${esc(branch.label)}">${esc(branch.label.replace(" pinned live control"," pinned").replace(" HEAD",""))}</th>`).join("")}</tr>`;
 const rows=DATA.matrix.filter(row=>(row.leaf+" "+row.category).toLowerCase().includes(query));
 byId("matrix-body").innerHTML=rows.map(row=>`<tr><td><strong>${esc(row.leaf)}</strong><br><span class="muted">${esc(row.category)}</span></td><td>${row.affected_count}/${DATA.branches.length}</td>${DATA.branches.map(branch=>{const state=row.states[branch.slug];if(state.status==="fail")return `<td><button class="fail-cell" onclick="inspectFailure(${state.failure_id})">Fail</button></td>`;return `<td class="${state.status}">${state.status==="pass"?"Pass":"N/A"}</td>`}).join("")}</tr>`).join("");
}
function populateFilters(){
 byId("branch-filter").innerHTML=option("","All branches")+DATA.branches.map(branch=>option(branch.slug,branch.label)).join("");
 byId("category-filter").innerHTML=option("","All categories")+uniq(DATA.failures.map(f=>f.category)).map(value=>option(value)).join("");
 byId("mode-filter").innerHTML=option("","All modes")+uniq(DATA.failures.map(f=>f.mode)).map(value=>option(value)).join("");
 byId("gpu-filter").innerHTML=option("","All GPU counts")+uniq(DATA.failures.map(f=>f.gpu)).map(value=>option(value)).join("");
}
function selectedFailures(){
 const branch=byId("branch-filter").value,category=byId("category-filter").value,mode=byId("mode-filter").value,gpu=byId("gpu-filter").value,query=byId("failure-search").value.trim().toLowerCase();
 return DATA.failures.filter(f=>(!branch||f.branch===branch)&&(!category||f.category===category)&&(!mode||f.mode===mode)&&(!gpu||f.gpu===gpu)&&(!query||(f.leaf+" "+f.module+" "+f.signature+" "+f.raw_error).toLowerCase().includes(query)));
}
function renderFailures(){
 const failures=selectedFailures();
 byId("failure-body").innerHTML=failures.map(f=>`<tr><td>${esc(f.branch_label)}</td><td>${esc(f.leaf)}</td><td>${esc(f.category)}</td><td><code>${esc(f.module)}</code></td><td><button onclick="inspectFailure(${f.id})">Inspect</button></td></tr>`).join("");
 if(failures.length&&!failures.some(f=>String(f.id)===byId("detail").dataset.failureId))inspectFailure(failures[0].id);
}
function inspectFailure(id){
 const f=failureById[id];if(!f)return;byId("detail").dataset.failureId=String(id);
 byId("detail").innerHTML=`<h3>${esc(f.category)}</h3><p>${esc(f.signature)}</p><dl><dt>Branch</dt><dd>${esc(f.branch_label)}<br><code>${esc(f.commit)}</code></dd><dt>Workload</dt><dd><code>${esc(f.leaf)}</code></dd><dt>Failing HLO</dt><dd><code>${esc(f.module_path||f.leaf_path)}</code></dd><dt>Partitions</dt><dd>${f.partitions}</dd><dt>Root evidence</dt><dd>${esc(f.raw_error)}</dd><dt>Source log</dt><dd><a href="${esc(f.log_uri)}">${esc(f.log_path)}</a>:${f.root_error_line}</dd></dl><div class="copy-row"><h3>Focused reproduction</h3><button onclick="copyRepro(${id})">Copy command</button><span class="muted" id="copy-status"></span></div><pre>${esc(f.repro_command)}</pre>`;
}
async function copyRepro(id){const f=failureById[id];try{await navigator.clipboard.writeText(f.repro_command);byId("copy-status").textContent="Copied"}catch(error){byId("copy-status").textContent="Select and copy the command below"}}
function filterCategory(category){byId("category-filter").value=category;renderFailures();byId("failure-search").scrollIntoView({behavior:"smooth",block:"center"})}
function renderProvenance(){
 const c=DATA.campaign,b=c.benchmark||{};byId("provenance").innerHTML=`<div><strong>Execution environment</strong><br>Host: ${esc(c.hostname||"N/A")}<br>Platform: ${esc(c.platform||"N/A")}<br>GPU: ${esc(c.gpu||"Not captured")}<br>Container identity: <code>${esc(c.container_identity||"Not captured")}</code></div><div><strong>Perf-tool configuration</strong><br>Repeats: ${esc(b.num_repeats??"N/A")}<br>Argument mode: <code>${esc(b.arg_mode||"N/A")}</code><br>Command buffer: ${esc(b.cmd_buffer||"N/A")}<br>Settle seconds: ${esc(b.settle_sec??"N/A")}</div>`;
}
renderHeader();renderBranches();cascadePerformance();renderSignatures();renderMatrix();populateFilters();renderFailures();renderProvenance();
byId("matrix-search").addEventListener("input",renderMatrix);["branch-filter","category-filter","mode-filter","gpu-filter"].forEach(id=>byId(id).addEventListener("change",renderFailures));byId("failure-search").addEventListener("input",renderFailures);
["perf-domain","perf-model","perf-mode","perf-gpu"].forEach(id=>byId(id).addEventListener("change",cascadePerformance));["perf-metric","perf-module"].forEach(id=>byId(id).addEventListener("change",renderPerformance));
</script>
</body></html>"""


def render_campaign_report_html(data: dict[str, Any]) -> str:
    """Render normalized campaign data as a self-contained HTML document."""
    payload = json.dumps(data, separators=(",", ":")).replace(
        "</script", "<\\/script"
    )
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return HTML_TEMPLATE.replace("__DATA__", payload).replace(
        "__GENERATED__", generated
    )


def generate_xla_hlo_campaign_html_report(
    campaign_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate the report and return its absolute output path."""
    campaign_dir = campaign_dir.expanduser().resolve(strict=True)
    output = (
        output_path.expanduser().resolve()
        if output_path is not None
        else campaign_dir / "full_campaign_report.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_campaign_report_html(
            build_campaign_report_data(campaign_dir)
        ),
        encoding="utf-8",
    )
    return output


def main() -> int:
    args = parse_args()
    output = generate_xla_hlo_campaign_html_report(
        args.campaign_dir,
        args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
