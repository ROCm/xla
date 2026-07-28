#!/usr/bin/env python3
"""Focused tests for historical-reference branch evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_xla_branch_eval as orchestrator
from compare_hlo_branch_results import (
    select_comparison_targets,
    write_comparison,
)
from reference_results import reference_inventory


TOOLS_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = TOOLS_ROOT / "configs/benchmark_profile.json"


class ReferenceResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    def test_leaf_inventory_maps_checked_in_result(self) -> None:
        leaf = (
            TOOLS_ROOT
            / "vision_diffusion/efficientnet/inference/1gpu"
        )
        inventory = reference_inventory(leaf, TOOLS_ROOT, self.profile)
        self.assertEqual(inventory["workload_count"], 1)
        self.assertEqual(inventory["available_count"], 1)
        workload = inventory["workloads"][0]
        self.assertEqual(
            workload["workload"],
            "vision_diffusion_efficientnet_inference_1gpu.csv",
        )
        self.assertEqual(
            workload["modules"],
            ["module_0961.jit_predict_step.before_optimizations.txt"],
        )
        self.assertTrue(workload["exists"])

    def test_single_file_inventory_filters_modules(self) -> None:
        hlo_file = next(
            (
                TOOLS_ROOT
                / "large_language_models/qwen3_14b/inference/1gpu"
            ).glob("*.txt")
        )
        inventory = reference_inventory(hlo_file, TOOLS_ROOT, self.profile)
        self.assertEqual(
            inventory["workloads"][0]["modules"], [hlo_file.name]
        )

    def test_known_missing_reference_is_recorded(self) -> None:
        leaf = (
            TOOLS_ROOT
            / "large_language_models/mixtral_8x7b/inference/1gpu"
        )
        inventory = reference_inventory(leaf, TOOLS_ROOT, self.profile)
        self.assertEqual(inventory["available_count"], 0)
        self.assertEqual(inventory["missing_count"], 1)
        self.assertFalse(inventory["workloads"][0]["exists"])

    def test_full_corpus_reference_inventory(self) -> None:
        inventory = reference_inventory(TOOLS_ROOT, TOOLS_ROOT, self.profile)
        self.assertEqual(inventory["workload_count"], 117)
        self.assertEqual(inventory["available_count"], 105)
        self.assertEqual(inventory["missing_count"], 12)


class TargetAndBuildConfigurationTest(unittest.TestCase):
    def test_same_commit_control_and_candidate_have_distinct_ids(self) -> None:
        commit = "a" * 40
        control = {
            "id": f"live-control:{commit}",
            "role": "live_control",
            "ref": "live-control/origin/rocm-jaxlib-v0.10.2",
            "source_ref": "origin/rocm-jaxlib-v0.10.2",
            "revision": commit,
            "commit": commit,
            "slug": "live_control",
        }
        candidate = {
            "id": "candidate:origin/rocm-jaxlib-v0.10.2",
            "role": "candidate",
            "ref": "origin/rocm-jaxlib-v0.10.2",
            "source_ref": "origin/rocm-jaxlib-v0.10.2",
            "revision": "origin/rocm-jaxlib-v0.10.2",
            "commit": commit,
            "slug": "candidate",
        }
        manifest = {
            "targets": [control, candidate],
            "comparison_target_ids": [control["id"], candidate["id"]],
        }
        selected = select_comparison_targets(manifest)
        self.assertEqual([target["role"] for target in selected], [
            "live_control",
            "candidate",
        ])

    def test_container_bazelrc_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            container_bazelrc = root / "rocm.bazelrc"
            container_bazelrc.write_text("build --announce_rc\n", encoding="utf-8")
            with mock.patch.object(
                orchestrator, "CONTAINER_ROCM_BAZELRC", container_bazelrc
            ):
                invocation, metadata = orchestrator.rocm_bazel_configuration(
                    "bazel", root, "v0.8.0"
                )
        self.assertEqual(
            invocation, ["bazel", f"--bazelrc={container_bazelrc}"]
        )
        self.assertEqual(metadata["mode"], "container_ci_fallback")
        self.assertIsNone(metadata["branch_bazelrc"])

    def test_nested_bazel_config_closure(self) -> None:
        text = """
common:rocm --config=rocm_base
common:rocm_base --config=clang_local
common:clang_local --action_env=CLANG_COMPILER_PATH=/usr/lib/llvm-18/bin/clang
"""
        self.assertEqual(
            orchestrator.bazel_config_closure(text, "rocm"),
            ["rocm", "rocm_base", "clang_local"],
        )


class ComparisonReportTest(unittest.TestCase):
    def test_historical_reference_control_and_candidate_report(self) -> None:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        leaf = (
            TOOLS_ROOT
            / "vision_diffusion/efficientnet/inference/1gpu"
        )
        inventory = reference_inventory(leaf, TOOLS_ROOT, profile)
        commit = profile["reference"]["xla_commit"]
        control = {
            "id": f"live-control:{commit}",
            "role": "live_control",
            "ref": "live-control/origin/rocm-jaxlib-v0.10.2",
            "source_ref": "origin/rocm-jaxlib-v0.10.2",
            "revision": commit,
            "commit": commit,
            "slug": "live_control",
        }
        candidate = {
            "id": "candidate:origin/rocm-jaxlib-v0.10.2",
            "role": "candidate",
            "ref": "origin/rocm-jaxlib-v0.10.2",
            "source_ref": "origin/rocm-jaxlib-v0.10.2",
            "revision": "origin/rocm-jaxlib-v0.10.2",
            "commit": commit,
            "slug": "candidate",
        }
        reference_dataset = {
            "id": profile["reference"]["id"],
            "role": "historical_reference",
            "source": "checked_in",
            "xla_ref": profile["reference"]["xla_ref"],
            "xla_commit": commit,
            "inventory": inventory,
        }
        workload = inventory["workloads"][0]["workload"]
        module = inventory["workloads"][0]["modules"][0]
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            for target, timing in ((control, "3.100ms"), (candidate, "3.200ms")):
                csv_dir = output_dir / target["slug"] / "csv"
                csv_dir.mkdir(parents=True)
                (csv_dir / workload).write_text(
                    f"Datetime,{module}\n2026-07-27 00:00:00,{timing}\n",
                    encoding="utf-8",
                )
            result = write_comparison(
                output_dir=output_dir,
                targets=[control, candidate],
                reference_dataset=reference_dataset,
                live_control_id=control["id"],
            )
            report = (output_dir / "comparison_report.md").read_text(
                encoding="utf-8"
            )
        self.assertEqual(result["validation"]["status"], "passed")
        self.assertEqual(len(result["branches"]), 2)
        self.assertEqual(
            result["reference_reproducibility"]["candidate_role"],
            "live_control",
        )
        self.assertIn("## Reference reproducibility", report)
        self.assertIn("origin/rocm-jaxlib-v0.10.2", report)


if __name__ == "__main__":
    unittest.main()
