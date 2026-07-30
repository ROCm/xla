#!/usr/bin/env python3
"""Tests for the formal HLO stability core and analysis contract."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from analyze_hlo_stability import analyze, write_outputs
from hlo_stability import (
    build_stability_plan,
    load_target_specs,
    load_orders,
    order_cycle_for_roles,
    orders_for_rounds,
    resolve_campaign_targets,
    selected_manifest_targets,
)
from reference_results import sha256_file


MODULE = "module_0961.jit_predict_step.before_optimizations.txt"


def write_result(path: Path, value_ms: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Datetime,{MODULE}\n2026-07-30 00:00:00,{value_ms:.4f}ms\n",
        encoding="utf-8",
    )


def create_campaign(
    root: Path, candidate_count: int = 3
) -> dict[str, object]:
    targets: list[dict[str, object]] = [
        {
            "id": "control",
            "role": "live_control",
            "source_ref": "origin/release",
            "revision": "1" * 40,
            "slug": "control",
            "commit": "1" * 40,
        }
    ]
    for index in range(1, 5):
        targets.append(
            {
                "id": f"candidate-{index}",
                "role": "candidate",
                "label": f"candidate {index}",
                "source_ref": f"origin/candidate-{index}",
                "revision": f"origin/candidate-{index}",
                "slug": f"candidate-{index}",
                "commit": str(index + 1) * 40,
            }
        )
    results = []
    for target in targets:
        runner = root / str(target["slug"]) / "runner/hlo_runner_main"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text(
            f"#!/usr/bin/env bash\n# {target['id']}\nexit 0\n",
            encoding="utf-8",
        )
        runner.chmod(0o755)
        results.append(
            {
                **target,
                "status": "completed",
                "runner_sha256": sha256_file(runner),
                "paths": {"runner": str(runner.resolve())},
            }
        )
    selected = [targets[0], *targets[1 : candidate_count + 1]]
    manifest: dict[str, object] = {
        "schema_version": 2,
        "status": "completed",
        "finished_at": "2026-07-30T00:00:00+00:00",
        "targets": targets,
        "results": results,
        "live_control_id": targets[0]["id"],
        "comparison_target_ids": [target["id"] for target in selected],
        "inputs": {
            "refs_file": {
                "path": "configs/xla_targets.json",
                "format": "json",
                "sha256": "a" * 64,
            }
        },
        "target_specs": [
            {
                "revision": target["revision"],
                "label": target["label"],
            }
            for target in selected[1:]
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


class CampaignTargetTest(unittest.TestCase):
    def test_target_selector_uses_strict_product_schema(self) -> None:
        invalid = [
            {"schema_version": True, "targets": [{"revision": "origin/main"}]},
            {
                "schema_version": 1,
                "targets": [{"revision": " bad"}],
            },
            {
                "schema_version": 1,
                "targets": [
                    {"revision": "origin/a", "label": "duplicate"},
                    {"revision": "origin/b", "label": "duplicate"},
                ],
            },
            {
                "schema_version": 1,
                "targets": [{"revision": "origin/a"}],
                "unexpected": True,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.json"
            for value in invalid:
                with self.subTest(value=value):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_target_specs(path)

    def test_selects_subset_and_validates_runner_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            manifest = create_campaign(campaign, candidate_count=4)
            specs = [
                {
                    "revision": "origin/candidate-2",
                    "label": "known good",
                },
                {
                    "revision": "origin/candidate-4",
                    "commit": "5" * 40,
                    "label": "suspected bad",
                },
            ]
            selected = selected_manifest_targets(manifest, specs)
            self.assertEqual(
                [target["id"] for target in selected],
                ["control", "candidate-2", "candidate-4"],
            )
            targets = resolve_campaign_targets(campaign, manifest, specs)
            self.assertEqual(
                list(targets), ["control", "candidate_1", "candidate_2"]
            )
            self.assertEqual(targets["candidate_1"]["label"], "known good")
            self.assertEqual(
                targets["candidate_2"]["runner_sha256"],
                sha256_file(
                    campaign / "candidate-4/runner/hlo_runner_main"
                ),
            )

            selector = campaign / "selector.json"
            selector.write_text(
                json.dumps({"schema_version": 1, "targets": specs}),
                encoding="utf-8",
            )
            plan = build_stability_plan(
                campaign_dir=campaign,
                manifest=manifest,
                targets=targets,
                rounds=12,
                target_cooldown_sec=8,
                round_cooldown_sec=30,
                selection_file=selector,
                selection_specs=specs,
            )
            self.assertEqual(plan["design"]["target_count"], 3)
            self.assertEqual(
                plan["stability_target_selection"]["sha256"],
                sha256_file(selector),
            )

    def test_rejects_runner_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            manifest = create_campaign(campaign, candidate_count=1)
            manifest["results"][1]["runner_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                resolve_campaign_targets(campaign, manifest)

    def test_relocated_campaign_uses_relative_runner_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            original.mkdir()
            create_campaign(original, candidate_count=1)
            relocated = root / "relocated"
            shutil.copytree(original, relocated)
            manifest = json.loads(
                (relocated / "manifest.json").read_text(encoding="utf-8")
            )
            targets = resolve_campaign_targets(relocated, manifest)
            self.assertTrue(
                Path(targets["control"]["runner"]).is_relative_to(relocated)
            )
            self.assertEqual(
                targets["control"]["recorded_runner_path"],
                str(original / "control/runner/hlo_runner_main"),
            )

    def test_rejects_nonfinal_and_identity_inconsistent_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            manifest = create_campaign(campaign, candidate_count=1)
            manifest["schema_version"] = 2.0
            with self.assertRaisesRegex(ValueError, "schema-v2"):
                resolve_campaign_targets(campaign, manifest)

            manifest = create_campaign(campaign, candidate_count=1)
            manifest["status"] = "running"
            with self.assertRaisesRegex(ValueError, "not finalized"):
                resolve_campaign_targets(campaign, manifest)

            manifest = create_campaign(campaign, candidate_count=1)
            manifest["results"][1]["commit"] = "f" * 40
            with self.assertRaisesRegex(ValueError, "commit mismatch"):
                resolve_campaign_targets(campaign, manifest)

            manifest = create_campaign(campaign, candidate_count=1)
            manifest["results"].append(dict(manifest["results"][1]))
            with self.assertRaisesRegex(ValueError, "duplicate result IDs"):
                resolve_campaign_targets(campaign, manifest)

    def test_balances_two_three_and_four_target_schedules(self) -> None:
        for target_count in (2, 3, 4):
            with self.subTest(target_count=target_count):
                roles = tuple(
                    ["control"]
                    + [
                        f"candidate_{index}"
                        for index in range(1, target_count)
                    ]
                )
                cycle = order_cycle_for_roles(roles)
                orders = orders_for_rounds(12, cycle)
                expected = 12 // target_count
                for position in range(target_count):
                    self.assertEqual(
                        Counter(order[position] for order in orders),
                        {role: expected for role in roles},
                    )
                predecessors = Counter(
                    pair
                    for order in orders
                    for pair in zip(order, order[1:])
                )
                self.assertEqual(
                    predecessors,
                    Counter(
                        {
                            (left, right): expected
                            for left in roles
                            for right in roles
                            if left != right
                        }
                    ),
                )

    def test_stability_plan_rejects_partial_schedule_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            manifest = create_campaign(campaign, candidate_count=2)
            targets = resolve_campaign_targets(campaign, manifest)
            with self.assertRaisesRegex(ValueError, "positive multiple"):
                build_stability_plan(
                    campaign_dir=campaign,
                    manifest=manifest,
                    targets=targets,
                    rounds=5,
                    target_cooldown_sec=8,
                    round_cooldown_sec=30,
                )


class StabilityAnalysisTest(unittest.TestCase):
    def test_preserves_raw_rounds_and_emits_evidence_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = ["control", "candidate_1", "candidate_2"]
            cycle = [
                list(order) for order in order_cycle_for_roles(tuple(roles))
            ]
            metadata = {
                "schema_version": 2,
                "design": {"roles": roles, "order_cycle": cycle},
                "targets": {
                    "control": {
                        "label": "Pinned control",
                        "source_ref": "origin/release",
                        "commit": "1" * 40,
                        "runner_sha256": "a" * 64,
                    },
                    "candidate_1": {
                        "label": "Known good",
                        "source_ref": "origin/good",
                        "commit": "2" * 40,
                        "runner_sha256": "b" * 64,
                    },
                    "candidate_2": {
                        "label": "Suspected bad",
                        "source_ref": "origin/bad",
                        "commit": "3" * 40,
                        "runner_sha256": "c" * 64,
                    },
                },
            }
            (root / "experiment_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            orders = orders_for_rounds(6, tuple(tuple(row) for row in cycle))
            with (root / "round_orders.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                stream.write("round,execution_order\n")
                for index, order in enumerate(orders, start=1):
                    stream.write(f"{index:02d},{'>'.join(order)}\n")
            values = {
                "control": [3.20, 3.21, 4.20, 3.19, 3.20, 3.21],
                "candidate_1": [3.00, 3.01, 2.99, 3.00, 3.01, 3.00],
                "candidate_2": [3.30, 3.31, 3.29, 3.30, 3.31, 3.30],
            }
            for role, samples in values.items():
                for index, value in enumerate(samples, start=1):
                    write_result(
                        root
                        / role
                        / f"round_{index:02d}"
                        / "csv/workload.csv",
                        value,
                    )
            historical = root / "historical.csv"
            write_result(historical, 3.0)
            result = analyze(
                experiment_dir=root,
                roles=roles,
                reference_csv=historical,
                modified_z_threshold=3.5,
                minimum_outlier_percent=2.0,
                temporal_drift_percent=2.0,
                reporting_threshold_percent=10.0,
            )
            write_outputs(root, result)

            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["round_count"], 6)
            self.assertEqual(len(result["long_rows"]), 18)
            self.assertEqual(
                result["role_summaries"]["control"]["outlier_count"], 1
            )
            self.assertEqual(
                sum(row["analysis_included"] for row in result["long_rows"]),
                17,
            )
            self.assertEqual(
                result["long_rows"][0]["runner_sha256"], "a" * 64
            )
            self.assertEqual(
                result["role_summaries"]["candidate_1"][
                    "paired_vs_control"
                ]["count"],
                5,
            )
            self.assertEqual(
                result["reporting_policy"][
                    "candidate_control_threshold_percent"
                ],
                10.0,
            )
            self.assertEqual(
                result["long_rows"][0]["system_before_path"], ""
            )
            for name in (
                "stability_analysis.json",
                "stability_summary.csv",
                "raw_rounds_long.csv",
                "paired_deltas.csv",
            ):
                self.assertTrue((root / name).is_file())
            summary = (root / "stability_summary.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("evidence_summary", summary)
            self.assertIn("Known good", summary)
            self.assertIn("paired_median_vs_control_percent", summary)
            self.assertIn("within control reporting band", summary)
            self.assertNotIn("pass", summary.lower())
            result["reporting_policy"]["minimum_paired_rounds"] = 6
            write_outputs(root, result)
            insufficient = (root / "stability_summary.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("insufficient paired evidence", insufficient)

    def test_legacy_order_inference_remains_available(self) -> None:
        orders, source = load_orders(
            Path("missing.csv"),
            ["control", "sentinel", "main"],
            ["01", "02", "03", "04", "05", "06"],
        )
        self.assertEqual(source, "inferred_legacy_three_target_cycle")
        self.assertEqual(orders["01"], ["control", "sentinel", "main"])

    def test_rejects_duplicate_order_rows_and_unpaired_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            order_path = root / "round_orders.csv"
            order_path.write_text(
                "round,execution_order\n"
                "01,control>candidate_1\n"
                "01,candidate_1>control\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_orders(
                    order_path,
                    ["control", "candidate_1"],
                    ["01"],
                )

            metadata = {
                "design": {
                    "roles": ["control", "candidate_1"],
                    "order_cycle": [
                        ["control", "candidate_1"],
                        ["candidate_1", "control"],
                    ],
                },
                "targets": {},
            }
            (root / "experiment_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            order_path.unlink()
            write_result(root / "control/round_01/csv/workload.csv", 3.0)
            write_result(root / "control/round_02/csv/workload.csv", 3.0)
            write_result(
                root / "candidate_1/round_01/csv/workload.csv", 3.1
            )
            with self.assertRaisesRegex(ValueError, "round sets differ"):
                analyze(
                    experiment_dir=root,
                    roles=["control", "candidate_1"],
                    reference_csv=None,
                    modified_z_threshold=3.5,
                    minimum_outlier_percent=2.0,
                )

    def test_metadata_less_legacy_experiment_can_be_analyzed_with_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = ["control", "sentinel", "main"]
            for role_index, role in enumerate(roles):
                for round_index in range(1, 7):
                    write_result(
                        root
                        / role
                        / f"round_{round_index:02d}"
                        / "csv/workload.csv",
                        3.0 + role_index * 0.1,
                    )
            result = analyze(
                experiment_dir=root,
                roles=roles,
                reference_csv=None,
                modified_z_threshold=3.5,
                minimum_outlier_percent=2.0,
            )
            self.assertEqual(
                result["execution_order_source"],
                "inferred_legacy_three_target_cycle",
            )


if __name__ == "__main__":
    unittest.main()
