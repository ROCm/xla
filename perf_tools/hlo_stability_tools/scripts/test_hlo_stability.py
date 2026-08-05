#!/usr/bin/env python3
"""Tests for the formal HLO stability core and analysis contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any
from unittest import mock

import analyze_hlo_stability as analyzer_cli
from analyze_hlo_stability import analyze, write_outputs
import run_hlo_stability as collector
from hlo_stability import (
    build_stability_plan,
    load_target_specs,
    load_orders,
    order_cycle_for_roles,
    orders_for_rounds,
    read_result,
    resolve_runner_bundle_targets,
    selected_bundle_targets,
)
from file_util import sha256_file
import render_hlo_stability_report as renderer_cli
from render_hlo_stability_report import render
import show_hlo_stability_status as status_cli
import xla_runner_bundle


MODULE = "module_0961.jit_predict_step.before_optimizations.txt"


def write_result(path: Path, value_ms: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Datetime,{MODULE}\n2026-07-30 00:00:00,{value_ms:.4f}ms\n",
        encoding="utf-8",
    )


def create_bundle(
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
        "kind": xla_runner_bundle.BUNDLE_KIND,
        "status": "completed",
        "finished_at": "2026-07-30T00:00:00+00:00",
        "source_original_state": {
            "branch": "main",
            "commit": "f" * 40,
            "status": "",
        },
        "source_restore": {
            "status": "restored",
            "branch": "main",
            "commit": "f" * 40,
            "working_tree_status": "",
        },
        "targets": targets,
        "results": results,
        "live_control_id": targets[0]["id"],
        "active_target_ids": [target["id"] for target in selected],
        "inputs": {
            "targets_file": {
                "path": "configs/xla_targets.json",
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


def collector_args(
    root: Path,
    runner_bundle: Path,
    output: Path,
    *,
    rounds: int = 4,
) -> argparse.Namespace:
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    hlo_path = inputs / MODULE
    hlo_path.write_text("HloModule test\n", encoding="utf-8")
    reference = inputs / "reference.csv"
    write_result(reference, 3.0)
    return argparse.Namespace(
        xla_source_repo=None,
        runner_bundle=runner_bundle,
        output_dir=output,
        hlo_path=hlo_path,
        targets_file=None,
        bazel_command=None,
        skip_fetch=False,
        reference_csv=reference,
        rounds=rounds,
        warmup_cooldown_sec=0.0,
        target_cooldown_sec=0.0,
        round_cooldown_sec=0.0,
        runner_settle_sec=0,
        capture_system_snapshots=False,
        modified_z_threshold=3.5,
        minimum_outlier_percent=2.0,
        temporal_drift_percent=2.0,
        reporting_threshold_percent=2.0,
        minimum_paired_rounds=3,
    )


def fake_evaluate_once(**kwargs: Any) -> None:
    role = kwargs["role"]
    output = kwargs["output"]
    role_offset = {
        "control": 0.2,
        "candidate_1": 0.0,
        "candidate_2": 0.3,
        "candidate_3": 0.2,
    }.get(role, 0.0)
    write_result(output / "csv/workload.csv", 3.0 + role_offset)
    (output / "eval.log").write_text("fake evaluation\n", encoding="utf-8")


class BundleTargetTest(unittest.TestCase):
    def test_checked_in_stability_configuration_is_valid(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        profile = xla_runner_bundle.load_stability_profile(
            package_root / "configs/stability_profile.json"
        )
        specs = load_target_specs(
            package_root / "configs/xla_targets.json"
        )
        template_specs = load_target_specs(
            package_root / "configs/xla_targets.template.json"
        )
        self.assertEqual(profile["runner"]["num_repeats"], 2)
        self.assertGreaterEqual(len(specs), 1)
        self.assertLessEqual(len(specs), 3)
        self.assertEqual(len(template_specs), 3)
        self.assertNotIn(
            profile["reference"]["xla_commit"],
            {
                spec.get("commit")
                for spec in specs
                if isinstance(spec.get("commit"), str)
            },
        )

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
            bundle = Path(temporary)
            manifest = create_bundle(bundle, candidate_count=4)
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
            selected = selected_bundle_targets(manifest, specs)
            self.assertEqual(
                [target["id"] for target in selected],
                ["control", "candidate-2", "candidate-4"],
            )
            targets = resolve_runner_bundle_targets(bundle, manifest, specs)
            self.assertEqual(
                list(targets), ["control", "candidate_1", "candidate_2"]
            )
            self.assertEqual(targets["candidate_1"]["label"], "known good")
            self.assertEqual(
                targets["candidate_2"]["runner_sha256"],
                sha256_file(
                    bundle / "candidate-4/runner/hlo_runner_main"
                ),
            )

            selector = bundle / "selector.json"
            selector.write_text(
                json.dumps({"schema_version": 1, "targets": specs}),
                encoding="utf-8",
            )
            plan = build_stability_plan(
                bundle_dir=bundle,
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
            bundle = Path(temporary)
            manifest = create_bundle(bundle, candidate_count=1)
            manifest["results"][1]["runner_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                resolve_runner_bundle_targets(bundle, manifest)

    def test_relocated_bundle_uses_relative_runner_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            original.mkdir()
            create_bundle(original, candidate_count=1)
            relocated = root / "relocated"
            shutil.copytree(original, relocated)
            manifest = json.loads(
                (relocated / "manifest.json").read_text(encoding="utf-8")
            )
            targets = resolve_runner_bundle_targets(relocated, manifest)
            self.assertTrue(
                Path(targets["control"]["runner"]).is_relative_to(relocated)
            )
            self.assertEqual(
                targets["control"]["recorded_runner_path"],
                str(original / "control/runner/hlo_runner_main"),
            )

    def test_rejects_nonfinal_and_identity_inconsistent_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            manifest = create_bundle(bundle, candidate_count=1)
            manifest["schema_version"] = 2.0
            with self.assertRaisesRegex(ValueError, "schema-v2"):
                resolve_runner_bundle_targets(bundle, manifest)

            manifest = create_bundle(bundle, candidate_count=1)
            manifest.pop("kind")
            with self.assertRaisesRegex(ValueError, "unsupported kind"):
                resolve_runner_bundle_targets(bundle, manifest)

            manifest = create_bundle(bundle, candidate_count=1)
            manifest["status"] = "running"
            with self.assertRaisesRegex(ValueError, "not finalized"):
                resolve_runner_bundle_targets(bundle, manifest)

            manifest = create_bundle(bundle, candidate_count=1)
            manifest["status"] = "completed_pending_restore"
            with self.assertRaisesRegex(ValueError, "not finalized"):
                resolve_runner_bundle_targets(bundle, manifest)

            manifest = create_bundle(bundle, candidate_count=1)
            manifest["source_restore"] = {"status": "failed"}
            with self.assertRaisesRegex(ValueError, "not successfully restored"):
                resolve_runner_bundle_targets(bundle, manifest)

            manifest = create_bundle(bundle, candidate_count=1)
            manifest["results"][1]["commit"] = "f" * 40
            with self.assertRaisesRegex(ValueError, "commit mismatch"):
                resolve_runner_bundle_targets(bundle, manifest)

            manifest = create_bundle(bundle, candidate_count=1)
            manifest["results"].append(dict(manifest["results"][1]))
            with self.assertRaisesRegex(ValueError, "duplicate result IDs"):
                resolve_runner_bundle_targets(bundle, manifest)

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
            bundle = Path(temporary)
            manifest = create_bundle(bundle, candidate_count=2)
            targets = resolve_runner_bundle_targets(bundle, manifest)
            with self.assertRaisesRegex(ValueError, "positive multiple"):
                build_stability_plan(
                    bundle_dir=bundle,
                    manifest=manifest,
                    targets=targets,
                    rounds=5,
                    target_cooldown_sec=8,
                    round_cooldown_sec=30,
                )


class RunnerBundleTest(unittest.TestCase):
    def test_logged_build_emits_periodic_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "build.log"
            process = mock.Mock()
            process.wait.side_effect = [
                xla_runner_bundle.subprocess.TimeoutExpired(
                    ["bazel", "build"], 30
                ),
                0,
            ]
            with (
                mock.patch.object(
                    xla_runner_bundle,
                    "_spawn_process",
                    return_value=(process, None),
                ),
                mock.patch.object(
                    xla_runner_bundle.time,
                    "monotonic",
                    side_effect=[0.0, 31.0],
                ),
                mock.patch("builtins.print") as printer,
            ):
                result = xla_runner_bundle.run_logged(
                    ["bazel", "build"],
                    cwd=Path(temporary),
                    log_path=log_path,
                    progress_label="build v0.10.2 HEAD",
                )
            self.assertEqual(result, 0)
            heartbeat = " ".join(
                str(argument)
                for call in printer.call_args_list
                for argument in call.args
            )
            self.assertIn("build v0.10.2 HEAD still running", heartbeat)
            self.assertIn("log_bytes=", heartbeat)

    def test_runner_target_slugs_must_be_unique(self) -> None:
        targets = [
            {"id": "one", "slug": "feature_a"},
            {"id": "two", "slug": "feature_a"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate slug"):
            xla_runner_bundle.validate_target_path_uniqueness(targets)

    def test_restored_checkout_must_match_original_state(self) -> None:
        expected = {
            "branch": "main",
            "commit": "1" * 40,
            "status": "",
        }
        with (
            mock.patch.object(
                xla_runner_bundle,
                "source_checkout_state",
                return_value={
                    "branch": "main",
                    "commit": "2" * 40,
                    "status": "",
                },
            ),
            self.assertRaisesRegex(RuntimeError, "does not match"),
        ):
            xla_runner_bundle.restored_source_checkout_metadata(
                Path("source"), expected
            )

    def test_build_runner_copies_and_hashes_bazel_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / ".bazelversion").write_text(
                "7.4.1\n", encoding="utf-8"
            )
            bazel_bin = root / "bazel-bin"
            built_runner = (
                bazel_bin / xla_runner_bundle.RUNNER_RELATIVE_PATH
            )
            built_runner.parent.mkdir(parents=True)
            built_runner.write_text(
                "#!/bin/sh\nexit 0\n", encoding="utf-8"
            )
            target = {
                "id": "candidate:origin/topic",
                "role": "candidate",
                "ref": "origin/topic",
                "source_ref": "origin/topic",
                "revision": "origin/topic",
                "slug": "origin_topic",
                "commit": "a" * 40,
            }

            def fake_git(_repo: Path, *arguments: str, **_kwargs: Any) -> str:
                if arguments[:2] == ("rev-parse", "HEAD"):
                    return target["commit"]
                return ""

            with (
                mock.patch.object(
                    xla_runner_bundle,
                    "require_clean_source_repo",
                    return_value={
                        "branch": "main",
                        "commit": "f" * 40,
                        "status": "",
                    },
                ),
                mock.patch.object(
                    xla_runner_bundle, "git", side_effect=fake_git
                ),
                mock.patch.object(
                    xla_runner_bundle,
                    "bazel_version",
                    return_value="7.4.1",
                ),
                mock.patch.object(
                    xla_runner_bundle,
                    "rocm_bazel_configuration",
                    return_value=(["bazel"], {"mode": "test"}),
                ),
                mock.patch.object(
                    xla_runner_bundle,
                    "rocm_host_toolchain_metadata",
                    return_value={},
                ),
                mock.patch.object(
                    xla_runner_bundle, "run_logged", return_value=0
                ),
                mock.patch.object(
                    xla_runner_bundle,
                    "run_capture",
                    return_value=str(bazel_bin),
                ),
            ):
                result = xla_runner_bundle.build_runner(
                    target=target,
                    source_repo=source,
                    bundle_dir=root / "bundle",
                    bazel="bazel",
                    reuse=False,
                )
            copied = (
                root
                / "bundle"
                / target["slug"]
                / "runner/hlo_runner_main"
            )
            self.assertEqual(result["status"], "completed")
            self.assertTrue(copied.is_file())
            self.assertTrue(os.access(copied, os.X_OK))
            self.assertEqual(result["runner_sha256"], sha256_file(copied))

    def test_preparation_restores_checkout_for_every_outcome(self) -> None:
        for outcome in (
            "success",
            "failure",
            "interrupt",
            "config_mutation",
        ):
            with self.subTest(outcome=outcome):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    source = root / "source"
                    source.mkdir()
                    bundle = root / "bundle"
                    targets_file = root / "targets.json"
                    targets_file.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "targets": [
                                    {
                                        "revision": "origin/candidate",
                                        "label": "candidate",
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    profile_file = root / "profile.json"
                    profile_file.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "name": "test",
                                "reference": {
                                    "id": "reference",
                                    "source": "checked_in",
                                    "xla_ref": "origin/release",
                                    "xla_commit": "1" * 40,
                                    "gpu": "MI350",
                                    "container": "test",
                                },
                                "runner": {
                                    "num_repeats": 2,
                                    "arg_mode": "uninitialized",
                                    "cmd_buffer": "off",
                                    "order": "size",
                                    "settle_sec": 2,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    control = {
                        "id": "control",
                        "role": "live_control",
                        "ref": "live-control/origin/release",
                        "source_ref": "origin/release",
                        "revision": "1" * 40,
                        "slug": "control",
                        "commit": "1" * 40,
                    }
                    candidate = {
                        "id": "candidate:origin/candidate",
                        "role": "candidate",
                        "ref": "origin/candidate",
                        "source_ref": "origin/candidate",
                        "revision": "origin/candidate",
                        "slug": "candidate",
                        "commit": "2" * 40,
                        "label": "candidate",
                    }

                    def build(**kwargs: Any) -> dict[str, Any]:
                        checkpoint = json.loads(
                            (bundle / "manifest.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        self.assertEqual(
                            checkpoint["source_original_state"], original
                        )
                        if outcome == "interrupt":
                            raise KeyboardInterrupt("synthetic interrupt")
                        target = kwargs["target"]
                        if (
                            outcome == "config_mutation"
                            and target["role"] == "live_control"
                        ):
                            targets_file.write_text(
                                '{"schema_version":1,"targets":[]}',
                                encoding="utf-8",
                            )
                        return {
                            **target,
                            "status": (
                                "completed"
                                if outcome
                                in {"success", "config_mutation"}
                                else "build_failed"
                            ),
                            "runner_sha256": "a" * 64,
                            "paths": {
                                "runner": str(
                                    bundle
                                    / target["slug"]
                                    / "runner/hlo_runner_main"
                                )
                            },
                        }

                    original = {
                        "branch": "main",
                        "commit": "f" * 40,
                        "status": "",
                    }
                    with (
                        mock.patch.object(
                            xla_runner_bundle,
                            "validate_git_root",
                            return_value=source,
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "ensure_and_fetch_remotes",
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "resolve_live_control",
                            return_value=control,
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "resolve_target_specs",
                            return_value=[candidate],
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "choose_bazel",
                            return_value="bazel",
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "collect_environment",
                            return_value={},
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "acquire_source_lock",
                            return_value=10,
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "release_source_lock",
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "require_clean_source_repo",
                            return_value=original,
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "build_runner",
                            side_effect=build,
                        ),
                        mock.patch.object(
                            xla_runner_bundle,
                            "restore_source_checkout",
                        ) as restore,
                        mock.patch.object(
                            xla_runner_bundle,
                            "restored_source_checkout_metadata",
                            return_value={
                                "status": "restored",
                                "branch": "main",
                                "commit": "f" * 40,
                            },
                        ),
                    ):
                        if outcome == "interrupt":
                            with self.assertRaises(KeyboardInterrupt):
                                xla_runner_bundle.prepare_runner_bundle(
                                    source_repo=source,
                                    bundle_dir=bundle,
                                    targets_file=targets_file,
                                    profile_file=profile_file,
                                    bazel_command=None,
                                    skip_fetch=True,
                                )
                        elif outcome == "failure":
                            with self.assertRaisesRegex(
                                RuntimeError, "runner preparation failed"
                            ):
                                xla_runner_bundle.prepare_runner_bundle(
                                    source_repo=source,
                                    bundle_dir=bundle,
                                    targets_file=targets_file,
                                    profile_file=profile_file,
                                    bazel_command=None,
                                    skip_fetch=True,
                                )
                        elif outcome == "config_mutation":
                            with self.assertRaisesRegex(
                                ValueError, "configuration changed"
                            ):
                                xla_runner_bundle.prepare_runner_bundle(
                                    source_repo=source,
                                    bundle_dir=bundle,
                                    targets_file=targets_file,
                                    profile_file=profile_file,
                                    bazel_command=None,
                                    skip_fetch=True,
                                )
                        else:
                            _, manifest = (
                                xla_runner_bundle.prepare_runner_bundle(
                                    source_repo=source,
                                    bundle_dir=bundle,
                                    targets_file=targets_file,
                                    profile_file=profile_file,
                                    bazel_command=None,
                                    skip_fetch=True,
                                )
                            )
                            self.assertEqual(manifest["status"], "completed")
                        restore.assert_called_once_with(source, original)
                    manifest = json.loads(
                        (bundle / "manifest.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        manifest["source_restore"]["status"], "restored"
                    )
                    if outcome == "interrupt":
                        self.assertEqual(manifest["status"], "interrupted")


class StabilityCollectorTest(unittest.TestCase):
    def test_signal_handler_blocks_repeated_finalization_signals(self) -> None:
        previous_mask = {"previous"}
        with (
            mock.patch.object(
                collector.xla_runner_bundle, "signal_active_process"
            ),
            mock.patch.object(
                collector.signal,
                "pthread_sigmask",
                return_value=previous_mask,
                create=True,
            ) as sigmask,
            mock.patch.object(
                collector.signal, "SIG_BLOCK", 0, create=True
            ),
            self.assertRaises(collector.CollectionInterrupted) as raised,
        ):
            collector.handle_collection_signal(
                collector.signal.SIGTERM, None
            )
        self.assertEqual(raised.exception.previous_mask, previous_mask)
        sigmask.assert_called_once()
        with (
            mock.patch.object(
                collector.xla_runner_bundle,
                "bundle_finalization_active",
                return_value=True,
            ),
            mock.patch.object(
                collector.xla_runner_bundle,
                "defer_finalization_signal",
            ) as defer,
            mock.patch.object(
                collector.signal,
                "pthread_sigmask",
                create=True,
            ) as sigmask,
        ):
            collector.handle_collection_signal(
                collector.signal.SIGTERM, None
            )
        defer.assert_called_once_with(collector.signal.SIGTERM)
        sigmask.assert_not_called()

    def test_status_command_reports_build_and_round_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runner_bundle/control").mkdir(parents=True)
            (root / "collection.lock").write_text(
                json.dumps({"pid": os.getpid()}), encoding="utf-8"
            )
            (root / "experiment_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "collecting",
                        "design": {
                            "roles": ["control", "candidate_1"],
                            "rounds": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "runner_bundle/manifest.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (root / "runner_bundle/control/metadata.json").write_text(
                json.dumps(
                    {
                        "label": "Pinned control",
                        "status": "completed",
                        "paths": {
                            "build_log": str(
                                root
                                / "runner_bundle/control/build.log"
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_result(
                root / "control/round_01/csv/workload.csv", 3.0
            )
            with mock.patch.object(
                status_cli, "process_running", return_value=True
            ):
                lines, finished = status_cli.status_lines(root)
            text = "\n".join(lines)
            self.assertIn("Experiment: collecting", text)
            self.assertIn("Pinned control — completed", text)
            self.assertIn("rounds control: 1/2", text)
            self.assertFalse(finished)
            with mock.patch.object(
                status_cli, "process_running", return_value=False
            ):
                dead_lines, dead_finished = status_cli.status_lines(root)
            self.assertTrue(dead_finished)
            self.assertIn(
                "stopped before recording a final status",
                "\n".join(dead_lines),
            )

    def test_public_cli_hides_fixed_runner_repeats(self) -> None:
        argv = [
            "run_hlo_stability.py",
            "--xla-source-repo",
            "source",
            "--output-dir",
            "output",
            "--hlo-path",
            "module.txt",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = collector.parse_args()
        self.assertFalse(hasattr(args, "num_repeats"))
        self.assertFalse(args.capture_system_snapshots)

    @unittest.skipUnless(os.name == "posix", "process-group test requires POSIX")
    def test_spawned_evaluator_process_group_can_be_terminated(self) -> None:
        process = collector.spawn_tracked_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            text=True,
            stdout=collector.subprocess.DEVNULL,
            stderr=collector.subprocess.DEVNULL,
        )
        collector.restore_tracked_process_signal_mask(process)
        try:
            collector.signal_process_group(process, collector.signal.SIGTERM)
            process.wait(timeout=10)
            self.assertIsNotNone(process.returncode)
        finally:
            if process.poll() is None:
                collector.signal_process_group(process, collector.signal.SIGKILL)
                process.wait()
            collector.ACTIVE_PROCESS = None

    def test_collects_analyzes_and_renders_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            create_bundle(bundle, candidate_count=3)
            output = root / "output"
            args = collector_args(root, bundle, output, rounds=4)
            with (
                mock.patch.object(
                    collector, "evaluate_once", side_effect=fake_evaluate_once
                ),
                mock.patch.object(collector.time, "sleep"),
            ):
                metadata = collector.collect(args)
            self.assertEqual(metadata["status"], "completed")
            self.assertNotIn(str(root), json.dumps(metadata))
            self.assertEqual(
                (output / "round_orders.csv").read_text(encoding="utf-8").count("\n"),
                5,
            )
            for name in (
                "experiment_metadata.json",
                "stability_analysis.json",
                "stability_summary.csv",
                "raw_rounds_long.csv",
                "paired_deltas.csv",
                "stability_report.html",
            ):
                self.assertTrue((output / name).is_file(), name)
            report = render(output)
            self.assertIn("HLO Stability Evidence Report", report)
            self.assertIn("candidate 1", report)
            self.assertIn("path.series-2", report)
            self.assertNotIn("stroke-dasharray", report)
            self.assertIn('<rect class="series-2"', report)
            self.assertIn(
                "distinct colors and marker shapes", report
            )
            self.assertIn(
                "Clean-mode paired comparisons exclude a pair", report
            )
            with mock.patch.object(
                sys,
                "argv",
                [
                    "analyze_hlo_stability.py",
                    "--experiment-dir",
                    str(output),
                ],
            ):
                self.assertEqual(analyzer_cli.main(), 0)
            refreshed = json.loads(
                (output / "experiment_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(refreshed["status"], "analyzed")
            self.assertIn(
                "analyze_hlo_stability.py", refreshed["tooling"]
            )
            self.assertIn("analysis_repository", refreshed)
            self.assertFalse((output / "stability_report.html").exists())
            with mock.patch.object(
                sys,
                "argv",
                [
                    "render_hlo_stability_report.py",
                    "--experiment-dir",
                    str(output),
                ],
            ):
                self.assertEqual(renderer_cli.main(), 0)
            refreshed = json.loads(
                (output / "experiment_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(refreshed["status"], "completed")
            self.assertIn(
                "render_hlo_stability_report.py", refreshed["tooling"]
            )
            self.assertIn("render_repository", refreshed)
            with self.assertRaisesRegex(
                ValueError, "must be an .html file"
            ):
                renderer_cli.write_stability_report(
                    output, output / "round_orders.csv"
                )
            metadata_path = output / "experiment_metadata.json"
            valid_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            mismatched_metadata = dict(valid_metadata)
            mismatched_metadata["design"] = dict(
                valid_metadata["design"]
            )
            mismatched_metadata["design"]["rounds"] += 1
            metadata_path.write_text(
                json.dumps(mismatched_metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "does not match the experiment design"
            ):
                render(output)
            metadata_path.write_text(
                json.dumps(valid_metadata), encoding="utf-8"
            )
            (output / "stability_summary.csv").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                render(output)

    def test_reuses_native_runner_bundle_without_building(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "runner_bundle"
            bundle.mkdir()
            manifest = create_bundle(bundle, candidate_count=1)
            manifest["kind"] = xla_runner_bundle.BUNDLE_KIND
            (bundle / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            output = root / "output"
            args = collector_args(root, bundle, output, rounds=2)
            with (
                mock.patch.object(
                    collector, "evaluate_once", side_effect=fake_evaluate_once
                ),
                mock.patch.object(collector.time, "sleep"),
                mock.patch.object(
                    xla_runner_bundle,
                    "prepare_runner_bundle",
                ) as prepare,
            ):
                metadata = collector.collect(args)
            prepare.assert_not_called()
            self.assertEqual(metadata["runner_source"]["kind"], "runner_bundle")
            self.assertEqual(metadata["runner_source"]["mode"], "reused")
            self.assertEqual(
                metadata["runner_source"]["manifest_relative_path"],
                "runner_source_manifest.json",
            )
            self.assertTrue(
                (output / "runner_source_manifest.json").is_file()
            )
            self.assertIn("collection_environment", metadata)
            self.assertIn("run_hlo_stability.py", metadata["tooling"])
            self.assertIn("commit", metadata["repository"])

    def test_build_mode_defaults_to_checked_in_target_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            output = root / "output"
            args = collector_args(root, root / "unused", output, rounds=12)
            args.xla_source_repo = source
            args.runner_bundle = None
            args.targets_file = None
            with (
                mock.patch.object(
                    xla_runner_bundle,
                    "prepare_runner_bundle",
                    side_effect=RuntimeError("stop after preflight"),
                ) as prepare,
                self.assertRaisesRegex(RuntimeError, "stop after preflight"),
            ):
                collector.collect(args)
            selected = prepare.call_args.kwargs["targets_file"]
            self.assertEqual(selected.name, "xla_targets.json")
            self.assertEqual(selected.parent.name, "configs")
            self.assertEqual(selected.parent.parent.name, "hlo_stability_tools")

    def test_invalid_schedule_is_rejected_before_runner_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            output = root / "output"
            selector = root / "targets.json"
            selector.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "targets": [{"revision": "origin/candidate"}],
                    }
                ),
                encoding="utf-8",
            )
            args = collector_args(root, root / "unused", output, rounds=3)
            args.xla_source_repo = source
            args.runner_bundle = None
            args.targets_file = selector
            with (
                mock.patch.object(
                    xla_runner_bundle, "prepare_runner_bundle"
                ) as prepare,
                self.assertRaisesRegex(ValueError, "schedule cycle"),
            ):
                collector.collect(args)
            prepare.assert_not_called()
            self.assertFalse(output.exists())

    def test_build_mode_requires_separate_xla_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            args = collector_args(root, root / "unused", output, rounds=12)
            args.xla_source_repo = collector.discover_repository_root()
            args.runner_bundle = None
            with self.assertRaisesRegex(ValueError, "different checkouts"):
                collector.collect(args)
            self.assertFalse(output.exists())

    def test_historical_reference_is_validated_before_runner_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            output = root / "output"
            args = collector_args(root, root / "unused", output, rounds=12)
            args.xla_source_repo = source
            args.runner_bundle = None
            args.reference_csv.write_text(
                "Datetime,different.txt\n2026-07-30,3.0ms\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    xla_runner_bundle, "prepare_runner_bundle"
                ) as prepare,
                self.assertRaisesRegex(ValueError, "module differs"),
            ):
                collector.collect(args)
            prepare.assert_not_called()
            self.assertFalse(output.exists())

    def test_build_mode_prepares_bundle_and_uses_fixed_runner_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            placeholder = root / "unused"
            output = root / "output"
            selector = root / "targets.json"
            selector.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "targets": [
                            {
                                "revision": "origin/candidate-1",
                                "label": "candidate 1",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = collector_args(root, placeholder, output, rounds=2)
            args.xla_source_repo = source
            args.runner_bundle = None
            args.targets_file = selector
            calls: list[dict[str, Any]] = []

            def prepare(**kwargs: Any) -> tuple[Path, dict[str, Any]]:
                bundle = kwargs["bundle_dir"]
                bundle.mkdir()
                manifest = create_bundle(bundle, candidate_count=1)
                manifest["kind"] = xla_runner_bundle.BUNDLE_KIND
                (bundle / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                return bundle / "manifest.json", manifest

            def evaluate(**kwargs: Any) -> None:
                calls.append(kwargs)
                fake_evaluate_once(**kwargs)

            with (
                mock.patch.object(
                    xla_runner_bundle,
                    "prepare_runner_bundle",
                    side_effect=prepare,
                ) as prepare_mock,
                mock.patch.object(
                    collector, "evaluate_once", side_effect=evaluate
                ),
                mock.patch.object(collector.time, "sleep"),
            ):
                metadata = collector.collect(args)
            prepare_mock.assert_called_once()
            self.assertEqual(
                prepare_mock.call_args.kwargs["profile_file"].name,
                "stability_profile.json",
            )
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["runner_source"]["kind"], "runner_bundle")
            self.assertEqual(metadata["runner_source"]["mode"], "built")
            self.assertEqual(
                metadata["collection"]["num_repeats"],
                collector.STABILITY_NUM_REPEATS,
            )
            self.assertFalse(
                metadata["collection"]["runner_policy"][
                    "num_repeats_user_configurable"
                ]
            )
            self.assertTrue(calls)
            self.assertEqual(
                {call["num_repeats"] for call in calls},
                {collector.STABILITY_NUM_REPEATS},
            )
            report = render(output)
            self.assertIn("Fixed repeat policy", report)
            self.assertIn("do not replace rocprofv3", report)

    def test_snapshot_tool_failure_is_retained_as_optional_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.txt"
            with mock.patch.object(
                xla_runner_bundle,
                "run_capture_result",
                side_effect=FileNotFoundError("tool unavailable"),
            ):
                collector.write_system_snapshot(path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("ERROR: tool unavailable", text)
            with (
                mock.patch.object(
                    collector,
                    "write_system_snapshot",
                    side_effect=PermissionError("read only"),
                ),
                mock.patch.object(
                    Path,
                    "write_text",
                    side_effect=PermissionError("read only"),
                ),
            ):
                collector.safe_system_snapshot(path)

    def test_requires_new_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            create_bundle(bundle, candidate_count=1)
            output = root / "output"
            output.mkdir()
            (output / "existing.txt").write_text("data", encoding="utf-8")
            args = collector_args(root, bundle, output, rounds=2)
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                collector.collect(args)

    def test_output_claim_and_numeric_options_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            collector.claim_output_directory(output)
            with self.assertRaisesRegex(ValueError, "already claimed|absent or empty"):
                collector.claim_output_directory(output)
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    collector.nonnegative_float(value)
        with self.assertRaises(argparse.ArgumentTypeError):
            collector.nonnegative_int("2.5")
        process = mock.Mock()
        process.pid = 123
        process.poll.return_value = None
        with mock.patch.object(collector.os, "killpg", create=True) as killpg:
            collector.signal_process_group(process, collector.signal.SIGTERM)
        killpg.assert_called_once_with(123, collector.signal.SIGTERM)

    def test_collection_failure_preserves_partial_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            create_bundle(bundle, candidate_count=1)
            output = root / "output"
            args = collector_args(root, bundle, output, rounds=2)

            def fail_second_round(**kwargs: Any) -> None:
                if "round_02" in str(kwargs["output"]):
                    raise RuntimeError("synthetic evaluation failure")
                fake_evaluate_once(**kwargs)

            with (
                mock.patch.object(
                    collector,
                    "evaluate_once",
                    side_effect=fail_second_round,
                ),
                mock.patch.object(collector.time, "sleep"),
                self.assertRaisesRegex(RuntimeError, "synthetic"),
            ):
                collector.collect(args)
            metadata = json.loads(
                (output / "experiment_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertTrue((output / "round_orders.csv").is_file())
            self.assertTrue(
                (output / "control/round_01/csv/workload.csv").is_file()
            )

    def test_reference_mutation_is_rejected_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            create_bundle(bundle, candidate_count=1)
            output = root / "output"
            args = collector_args(root, bundle, output, rounds=2)

            def mutate_reference(**kwargs: Any) -> None:
                fake_evaluate_once(**kwargs)
                if "round_02" in str(kwargs["output"]):
                    write_result(args.reference_csv, 9.0)

            with (
                mock.patch.object(
                    collector,
                    "evaluate_once",
                    side_effect=mutate_reference,
                ),
                mock.patch.object(collector.time, "sleep"),
                self.assertRaisesRegex(ValueError, "historical reference"),
            ):
                collector.collect(args)
            metadata = json.loads(
                (output / "experiment_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")

    def test_evaluate_rejects_runner_mutation_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = create_bundle(bundle, candidate_count=1)
            targets = resolve_runner_bundle_targets(bundle, manifest)
            runner = Path(targets["control"]["runner"])
            runner.write_text("changed\n", encoding="utf-8")
            hlo_path = root / "one_hlo.txt"
            hlo_path.write_text("HloModule test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runner changed"):
                collector.evaluate_once(
                    role="control",
                    target=targets["control"],
                    output=root / "measurement",
                    eval_script=Path(sys.executable),
                    eval_script_sha256=sha256_file(Path(sys.executable)),
                    hlo_path=hlo_path,
                    hlo_sha256=sha256_file(hlo_path),
                    num_repeats=2,
                    runner_settle_sec=0,
                    capture_snapshots=False,
                )

    def test_evaluate_uses_trusted_argv_environment_and_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = create_bundle(bundle, candidate_count=1)
            target = resolve_runner_bundle_targets(bundle, manifest)["control"]
            hlo_path = root / MODULE
            hlo_path.write_text("HloModule test\n", encoding="utf-8")
            output = root / "measurement"
            calls: list[tuple[list[str], dict[str, Any]]] = []

            def fake_popen(command: list[str], **kwargs: Any) -> mock.Mock:
                calls.append((command, kwargs))
                kwargs["stdout"].write(
                    f"runner : {target['runner']}\n"
                    f"hlo    : {hlo_path}\n"
                    f"out    : {output / 'csv'}\n"
                )
                write_result(output / "csv/workload.csv", 3.0)
                process = mock.Mock()
                process.pid = 123
                process.wait.return_value = 0
                process.poll.return_value = 0
                return process

            eval_script = root / "run_hlo_eval.sh"
            eval_script.write_text("#!/bin/sh\n", encoding="utf-8")
            with mock.patch.object(
                collector.subprocess, "Popen", side_effect=fake_popen
            ):
                collector.evaluate_once(
                    role="control",
                    target=target,
                    output=output,
                    eval_script=eval_script,
                    eval_script_sha256=sha256_file(eval_script),
                    hlo_path=hlo_path,
                    hlo_sha256=sha256_file(hlo_path),
                    num_repeats=2,
                    runner_settle_sec=1,
                    capture_snapshots=False,
                )
            self.assertEqual(calls[0][0][0], str(eval_script))
            self.assertEqual(calls[0][0][-1], "2")
            self.assertTrue(calls[0][1]["start_new_session"])
            self.assertEqual(calls[0][1]["env"]["RESUME"], "0")
            self.assertEqual(calls[0][1]["env"]["SETTLE_SEC"], "1")
            log = (output / "eval.log").read_text(encoding="utf-8")
            self.assertNotIn(str(root), log)
            self.assertIn("<runner:control>", log)

    def test_hash_validation_occurs_after_before_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = create_bundle(bundle, candidate_count=1)
            target = resolve_runner_bundle_targets(bundle, manifest)["control"]
            runner = Path(target["runner"])
            hlo_path = root / MODULE
            hlo_path.write_text("HloModule test\n", encoding="utf-8")

            def mutate_runner(_path: Path) -> None:
                runner.write_text("changed after snapshot\n", encoding="utf-8")

            with (
                mock.patch.object(
                    collector,
                    "safe_system_snapshot",
                    side_effect=mutate_runner,
                ),
                self.assertRaisesRegex(ValueError, "runner changed"),
            ):
                collector.evaluate_once(
                    role="control",
                    target=target,
                    output=root / "measurement",
                    eval_script=Path(sys.executable),
                    eval_script_sha256=sha256_file(Path(sys.executable)),
                    hlo_path=hlo_path,
                    hlo_sha256=sha256_file(hlo_path),
                    num_repeats=2,
                    runner_settle_sec=0,
                    capture_snapshots=True,
                )

    def test_evaluator_mutation_after_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = create_bundle(bundle, candidate_count=1)
            target = resolve_runner_bundle_targets(bundle, manifest)["control"]
            hlo_path = root / MODULE
            hlo_path.write_text("HloModule test\n", encoding="utf-8")
            eval_script = root / "run_hlo_eval.sh"
            eval_script.write_text("original\n", encoding="utf-8")
            expected_hash = sha256_file(eval_script)

            def mutate_evaluator(_path: Path) -> None:
                eval_script.write_text("changed\n", encoding="utf-8")

            with (
                mock.patch.object(
                    collector,
                    "safe_system_snapshot",
                    side_effect=mutate_evaluator,
                ),
                self.assertRaisesRegex(ValueError, "evaluation script changed"),
            ):
                collector.evaluate_once(
                    role="control",
                    target=target,
                    output=root / "measurement",
                    eval_script=eval_script,
                    eval_script_sha256=expected_hash,
                    hlo_path=hlo_path,
                    hlo_sha256=sha256_file(hlo_path),
                    num_repeats=2,
                    runner_settle_sec=0,
                    capture_snapshots=True,
                )

    def test_evaluator_dependency_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = create_bundle(bundle, candidate_count=1)
            target = resolve_runner_bundle_targets(
                bundle, manifest
            )["control"]
            hlo_path = root / MODULE
            hlo_path.write_text("HloModule test\n", encoding="utf-8")
            dependency = root / "legacy_profile_to_csv.py"
            dependency.write_text("# original\n", encoding="utf-8")
            dependency_hash = sha256_file(dependency)

            def mutate(_path: Path) -> None:
                dependency.write_text("# changed\n", encoding="utf-8")

            with (
                mock.patch.object(
                    collector,
                    "safe_system_snapshot",
                    side_effect=mutate,
                ),
                self.assertRaisesRegex(
                    ValueError, "evaluator dependency changed"
                ),
            ):
                collector.evaluate_once(
                    role="control",
                    target=target,
                    output=root / "measurement",
                    eval_script=Path(sys.executable),
                    eval_script_sha256=sha256_file(Path(sys.executable)),
                    hlo_path=hlo_path,
                    hlo_sha256=sha256_file(hlo_path),
                    num_repeats=2,
                    runner_settle_sec=0,
                    capture_snapshots=True,
                    evaluator_dependencies={
                        dependency: dependency_hash
                    },
                )

    def test_hlo_mutation_after_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = create_bundle(bundle, candidate_count=1)
            target = resolve_runner_bundle_targets(
                bundle, manifest
            )["control"]
            hlo_path = root / MODULE
            hlo_path.write_text("HloModule test\n", encoding="utf-8")
            hlo_hash = sha256_file(hlo_path)

            def mutate(_path: Path) -> None:
                hlo_path.write_text("changed\n", encoding="utf-8")

            with (
                mock.patch.object(
                    collector,
                    "safe_system_snapshot",
                    side_effect=mutate,
                ),
                self.assertRaisesRegex(ValueError, "HLO input changed"),
            ):
                collector.evaluate_once(
                    role="control",
                    target=target,
                    output=root / "measurement",
                    eval_script=Path(sys.executable),
                    eval_script_sha256=sha256_file(Path(sys.executable)),
                    hlo_path=hlo_path,
                    hlo_sha256=hlo_hash,
                    num_repeats=2,
                    runner_settle_sec=0,
                    capture_snapshots=True,
                )


class StabilityAnalysisTest(unittest.TestCase):
    def test_analyzer_rejects_nonfinite_policy_values(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "analyze_hlo_stability.py",
                    "--experiment-dir",
                    "missing",
                    "--modified-z-threshold",
                    "nan",
                ],
            ),
            self.assertRaisesRegex(SystemExit, "must be finite"),
        ):
            analyzer_cli.main()

    def test_collected_csv_must_match_selected_hlo_and_one_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.csv"
            write_result(path, 3.0)
            with self.assertRaisesRegex(ValueError, "module differs"):
                read_result(
                    path,
                    expected_module="different.txt",
                    require_single_row=True,
                )
            path.write_text(
                f"Datetime,{MODULE}\n"
                "2026-07-30 00:00:00,3.0ms\n"
                "2026-07-30 00:00:01,3.1ms\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one timing row"):
                read_result(
                    path,
                    expected_module=MODULE,
                    require_single_row=True,
                )

    def test_partial_or_interrupted_experiment_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_path = root / "experiment_metadata.json"
            metadata = {
                "schema_version": 2,
                "status": "completed",
                "design": {
                    "roles": ["control", "candidate_1"],
                    "rounds": 8,
                    "order_cycle": [
                        ["control", "candidate_1"],
                        ["candidate_1", "control"],
                    ],
                },
                "targets": {},
            }
            metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with (root / "round_orders.csv").open(
                "w", encoding="utf-8"
            ) as stream:
                stream.write("round,execution_order\n")
                for index in range(1, 5):
                    order = (
                        "control>candidate_1"
                        if index % 2
                        else "candidate_1>control"
                    )
                    stream.write(f"{index:02d},{order}\n")
            for role, value in (
                ("control", 3.2),
                ("candidate_1", 3.1),
            ):
                for index in range(1, 5):
                    write_result(
                        root
                        / role
                        / f"round_{index:02d}"
                        / "csv/workload.csv",
                        value,
                    )
            with self.assertRaisesRegex(
                ValueError, "every requested round"
            ):
                analyze(
                    experiment_dir=root,
                    roles=["control", "candidate_1"],
                    reference_csv=None,
                    modified_z_threshold=3.5,
                    minimum_outlier_percent=2.0,
                )
            metadata["status"] = "interrupted"
            metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "not complete enough"
            ):
                analyze(
                    experiment_dir=root,
                    roles=["control", "candidate_1"],
                    reference_csv=None,
                    modified_z_threshold=3.5,
                    minimum_outlier_percent=2.0,
                )

    def test_preserves_raw_rounds_and_emits_evidence_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = ["control", "candidate_1", "candidate_2"]
            cycle = [
                list(order) for order in order_cycle_for_roles(tuple(roles))
            ]
            metadata = {
                "schema_version": 2,
                "status": "completed",
                "design": {
                    "roles": roles,
                    "rounds": 6,
                    "order_cycle": cycle,
                },
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

    def test_frequent_high_mode_and_identical_runners_are_prominent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = ["control", "candidate_1"]
            metadata = {
                "schema_version": 2,
                "status": "analyzed",
                "design": {
                    "roles": roles,
                    "rounds": 8,
                    "order_cycle": [
                        ["control", "candidate_1"],
                        ["candidate_1", "control"],
                    ],
                },
                "targets": {
                    "control": {
                        "label": "Pinned control",
                        "source_ref": "origin/release",
                        "commit": "1" * 40,
                        "runner_sha256": "a" * 64,
                    },
                    "candidate_1": {
                        "label": "Candidate HEAD",
                        "source_ref": "origin/release",
                        "commit": "2" * 40,
                        "runner_sha256": "a" * 64,
                    },
                },
            }
            metadata_path = root / "experiment_metadata.json"
            metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with (root / "round_orders.csv").open(
                "w", encoding="utf-8"
            ) as stream:
                stream.write("round,execution_order\n")
                for index in range(1, 9):
                    order = (
                        "control>candidate_1"
                        if index % 2
                        else "candidate_1>control"
                    )
                    stream.write(f"{index:02d},{order}\n")
            candidate_values = [
                4.30,
                3.20,
                3.20,
                4.20,
                3.20,
                3.20,
                4.40,
                3.20,
            ]
            for index in range(1, 9):
                write_result(
                    root
                    / "control"
                    / f"round_{index:02d}"
                    / "csv/workload.csv",
                    3.20,
                )
                write_result(
                    root
                    / "candidate_1"
                    / f"round_{index:02d}"
                    / "csv/workload.csv",
                    candidate_values[index - 1],
                )
            result = analyze(
                experiment_dir=root,
                roles=roles,
                reference_csv=None,
                modified_z_threshold=3.5,
                minimum_outlier_percent=2.0,
            )
            candidate = result["role_summaries"]["candidate_1"]
            self.assertTrue(candidate["frequent_outliers"])
            self.assertEqual(candidate["high_outlier_count"], 3)
            self.assertGreater(candidate["raw"]["cv_percent"], 10)
            self.assertEqual(len(result["identity_warnings"]), 1)
            write_outputs(root, result)
            artifacts = (
                "stability_analysis.json",
                "stability_summary.csv",
                "raw_rounds_long.csv",
                "paired_deltas.csv",
            )
            metadata["analysis_artifacts"] = {
                name: sha256_file(root / name) for name in artifacts
            }
            metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            report = render(root)
            self.assertIn("Distribution instability", report)
            self.assertIn("37.5%", report)
            self.assertIn("Runner identity warning", report)
            self.assertIn(
                "Clean-mode comparisons do not describe this full distribution",
                report,
            )

    def test_balanced_bimodality_is_not_called_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = ["control", "candidate_1"]
            metadata = {
                "schema_version": 2,
                "status": "completed",
                "design": {
                    "roles": roles,
                    "rounds": 24,
                    "order_cycle": [
                        ["control", "candidate_1"],
                        ["candidate_1", "control"],
                    ],
                },
                "targets": {
                    "control": {"label": "Control"},
                    "candidate_1": {"label": "Bimodal candidate"},
                },
            }
            (root / "experiment_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with (root / "round_orders.csv").open(
                "w", encoding="utf-8"
            ) as stream:
                stream.write("round,execution_order\n")
                for index in range(1, 25):
                    order = (
                        "control>candidate_1"
                        if index % 2
                        else "candidate_1>control"
                    )
                    stream.write(f"{index:02d},{order}\n")
            for index in range(1, 25):
                write_result(
                    root
                    / "control"
                    / f"round_{index:02d}"
                    / "csv/workload.csv",
                    3.70,
                )
                write_result(
                    root
                    / "candidate_1"
                    / f"round_{index:02d}"
                    / "csv/workload.csv",
                    3.20 if index % 2 else 4.20,
                )
            result = analyze(
                experiment_dir=root,
                roles=roles,
                reference_csv=None,
                modified_z_threshold=3.5,
                minimum_outlier_percent=2.0,
            )
            candidate = result["role_summaries"]["candidate_1"]
            self.assertEqual(candidate["outlier_count"], 0)
            self.assertTrue(candidate["broad_distribution"])
            self.assertTrue(candidate["distribution_instability"])
            self.assertIn(
                "broad or multimodal latency distribution",
                candidate["stability_evidence"],
            )

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
                "status": "collected",
                "design": {
                    "roles": ["control", "candidate_1"],
                    "rounds": 2,
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

if __name__ == "__main__":
    unittest.main()
