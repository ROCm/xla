#!/usr/bin/env python3
"""Tests for the self-contained HLO HTML report renderer."""

from __future__ import annotations

import unittest

from render_hlo_report import (
    branch_sort_key,
    relative_performance_delta,
    relative_performance_ratio,
    render_report,
    workload_name,
)


COMMIT = "7b5ecf1c9282fdf1039211e0d45216980058beda"


def branch(
    *,
    target_id: str,
    role: str,
    ref: str,
    ratio: float,
) -> dict[str, object]:
    return {
        "candidate_id": target_id,
        "candidate_role": role,
        "candidate_ref": ref,
        "candidate_commit": COMMIT,
        "matched_modules": 1,
        "faster_modules": int(ratio < 1),
        "slower_modules": int(ratio > 1),
        "unchanged_modules": int(ratio == 1),
        "missing_baseline": 0,
        "missing_candidate": 0,
        "baseline_suite_ms": 3.0,
        "candidate_suite_ms": 3.0 * ratio,
        "suite_ratio": ratio,
        "suite_delta_percent": (ratio - 1) * 100,
        "median_module_ratio": ratio,
        "median_module_delta_percent": (ratio - 1) * 100,
        "geomean_module_ratio": ratio,
        "geomean_module_delta_percent": (ratio - 1) * 100,
    }


class HtmlReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.control = branch(
            target_id=f"live-control:{COMMIT}",
            role="live_control",
            ref="live-control/origin/rocm-jaxlib-v0.10.2",
            ratio=1.087,
        )
        self.v0102 = branch(
            target_id="candidate:origin/rocm-jaxlib-v0.10.2",
            role="candidate",
            ref="origin/rocm-jaxlib-v0.10.2",
            ratio=1.077,
        )
        self.main = branch(
            target_id="candidate:upstream/main",
            role="candidate",
            ref="upstream/main",
            ratio=1.084,
        )
        targets = [
            {
                "id": item["candidate_id"],
                "role": item["candidate_role"],
                "ref": item["candidate_ref"],
                "commit": item["candidate_commit"],
            }
            for item in (self.control, self.v0102, self.main)
        ]
        self.manifest = {
            "schema_version": 2,
            "created_at": "2026-07-28T08:00:00+00:00",
            "status": "completed",
            "profile": {
                "reference": {
                    "gpu": "MI350",
                    "container": "rocm-test-container",
                }
            },
            "benchmark": {
                "effective": {
                    "num_repeats": 2,
                    "arg_mode": "uninitialized",
                    "cmd_buffer": "off",
                    "settle_sec": 2,
                }
            },
            "environment": {
                "hostname": "mi350-test",
                "platform": "Linux",
                "python": "3.11",
            },
            "reference_dataset": {
                "id": "checked-in-v0.10.2-mi350",
                "xla_commit": COMMIT,
                "inventory": {
                    "workload_count": 1,
                    "available_count": 1,
                    "missing_count": 0,
                },
            },
            "comparison_target_ids": [item["id"] for item in targets],
            "targets": targets,
            "results": [
                {
                    **item,
                    "status": "completed",
                    "build_exit_code": 0,
                    "evaluation_exit_code": 0,
                }
                for item in targets
            ],
        }
        self.summary = {
            "baseline_ref": "checked-in-v0.10.2-mi350",
            "baseline_commit": COMMIT,
            "baseline_source": "checked_in",
            "validation": {
                "status": "passed",
                "missing_baseline_modules": 0,
                "missing_candidate_modules": 0,
            },
            "branches": [self.control, self.main, self.v0102],
            "reference_reproducibility": self.control,
        }
        self.rows = [
            {
                "baseline_ref": "checked-in-v0.10.2-mi350",
                "baseline_commit": COMMIT,
                "baseline_source": "checked_in",
                "candidate_id": item["candidate_id"],
                "candidate_role": item["candidate_role"],
                "candidate_ref": item["candidate_ref"],
                "candidate_commit": COMMIT,
                "workload": "vision_diffusion_efficientnet_inference_1gpu.csv",
                "module": "module_0961.jit_predict_step.before_optimizations.txt",
                "baseline_ms": 3.0,
                "candidate_ms": 3.0 * float(item["geomean_module_ratio"]),
                "ratio": item["geomean_module_ratio"],
                "delta_ms": 3.0 * (float(item["geomean_module_ratio"]) - 1),
                "delta_percent": item["geomean_module_delta_percent"],
                "status": (
                    "faster"
                    if float(item["geomean_module_ratio"]) < 1
                    else "slower"
                ),
            }
            for item in (self.control, self.v0102, self.main)
        ]

    def test_report_contains_required_story_sections(self) -> None:
        rendered = render_report(
            manifest=self.manifest,
            summary=self.summary,
            comparison_rows=self.rows,
            threshold_percent=None,
            top_movers=10,
        )
        self.assertIn("<!doctype html>", rendered)
        self.assertIn("1. Measurement confidence", rendered)
        self.assertIn("2. Branch performance trend", rendered)
        self.assertIn("3. Branch scorecard", rendered)
        self.assertIn("4. Largest HLO movers", rendered)
        self.assertIn("5. Per-HLO evidence matrix", rendered)
        self.assertIn("main is +0.28% versus live control", rendered)
        self.assertIn(
            "0.923× versus historical (-7.75%) and 1.003× versus live control (+0.28%)",
            rendered,
        )
        self.assertIn(
            "observed difference is within the ±2.00% reporting band",
            rendered,
        )
        self.assertIn("Data validation passed", rendered)
        self.assertIn("Relative performance (higher is better)", rendered)
        self.assertIn("±2.00% historical reference band", rendered)
        self.assertIn("Historical performance", rendered)
        self.assertIn('<td class="bad">-7.75%</td>', rendered)
        self.assertIn('<td class="neutral">+0.28%</td>', rendered)
        self.assertNotIn("Summed-suite relative performance", rendered)
        self.assertNotIn("larger of 2% and live-control", rendered)
        self.assertIn("live-control/origin/rocm-jaxlib-v0.10.2", rendered)
        self.assertIn("v0.10.2", rendered)
        self.assertNotIn("https://", rendered)

    def test_schema_v1_is_rejected(self) -> None:
        self.manifest["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema-v2"):
            render_report(
                manifest=self.manifest,
                summary=self.summary,
                comparison_rows=self.rows,
                threshold_percent=None,
                top_movers=10,
            )

    def test_workload_name_preserves_model_underscores(self) -> None:
        self.assertEqual(
            workload_name(
                "large_language_models_deepseek2_16b_inference_8gpu.csv"
            ),
            "large_language_models/deepseek2_16b/inference/8gpu",
        )

    def test_branch_order_places_main_last(self) -> None:
        refs = [
            {"candidate_ref": "upstream/main"},
            {"candidate_ref": "origin/rocm-jaxlib-v0.10.2"},
            {"candidate_ref": "origin/rocm-jaxlib-v0.8.0"},
        ]
        ordered = sorted(refs, key=branch_sort_key)
        self.assertEqual(
            [item["candidate_ref"] for item in ordered],
            [
                "origin/rocm-jaxlib-v0.8.0",
                "origin/rocm-jaxlib-v0.10.2",
                "upstream/main",
            ],
        )

    def test_relative_performance_inverts_latency_ratio(self) -> None:
        ratio = relative_performance_ratio(1.084)
        delta = relative_performance_delta(1.084)
        self.assertAlmostEqual(ratio or 0.0, 0.9225092251)
        self.assertAlmostEqual(delta or 0.0, -7.74907749)
        self.assertAlmostEqual(relative_performance_ratio(0.98) or 0.0, 1.0204081633)


if __name__ == "__main__":
    unittest.main()
