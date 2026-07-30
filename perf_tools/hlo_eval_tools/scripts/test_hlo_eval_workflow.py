#!/usr/bin/env python3
"""Focused tests for historical-reference branch evaluation."""

from __future__ import annotations

import io
import json
import sys
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
PERF_REPO = TOOLS_ROOT.parents[1]
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

    def test_restore_metadata_keeps_result_and_working_tree_status(self) -> None:
        with mock.patch.object(
            orchestrator,
            "source_checkout_state",
            return_value={
                "branch": "release",
                "commit": "a" * 40,
                "status": "",
            },
        ):
            metadata = orchestrator.restored_source_checkout_metadata(
                Path("source")
            )
        self.assertEqual(metadata["status"], "restored")
        self.assertEqual(metadata["branch"], "release")
        self.assertEqual(metadata["commit"], "a" * 40)
        self.assertEqual(metadata["working_tree_status"], "")

    def test_campaign_html_output_is_recorded_in_manifest(self) -> None:
        manifest = {"comparison": {}}
        expected = Path("output/comparison_report.html")
        with mock.patch.object(
            orchestrator, "write_html_report", return_value=expected
        ) as renderer:
            output = orchestrator.generate_campaign_html_report(
                Path("output"), manifest
            )
        self.assertEqual(output, expected)
        self.assertEqual(
            manifest["comparison"]["html_report"], str(expected)
        )
        renderer.assert_called_once()


class StructuredTargetsTest(unittest.TestCase):
    def test_loads_structured_targets_and_checked_in_example(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets_file = root / "targets.json"
            targets_file.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "targets": [
                            {
                                "revision": "origin/release",
                                "label": "release HEAD",
                            },
                            {"revision": "a" * 40},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                orchestrator.read_structured_targets(targets_file),
                [
                    {
                        "revision": "origin/release",
                        "label": "release HEAD",
                    },
                    {"revision": "a" * 40},
                ],
            )
            checked_in = TOOLS_ROOT / "configs/xla_targets.json"
            checked_targets = orchestrator.read_structured_targets(checked_in)
            self.assertEqual(len(checked_targets), 3)
            self.assertIsNone(checked_targets[0]["commit"])
            self.assertEqual(
                checked_targets[1]["commit"],
                "7b5ecf1c9282fdf1039211e0d45216980058beda",
            )
            self.assertIsNone(checked_targets[2]["commit"])

    def test_rejects_invalid_structured_target_schema(self) -> None:
        invalid_values = [
            {"schema_version": 2, "targets": [{"revision": "abc"}]},
            {"schema_version": True, "targets": [{"revision": "abc"}]},
            {"schema_version": 1.0, "targets": [{"revision": "abc"}]},
            {"schema_version": 1, "targets": []},
            {"schema_version": 1, "targets": [{"revision": "-bad"}]},
            {"schema_version": 1, "targets": [{"revision": "bad ref"}]},
            {
                "schema_version": 1,
                "targets": [{"revision": "abc", "label": None}],
            },
            {
                "schema_version": 1,
                "targets": [{"revision": "abc", "commit": "short"}],
            },
            {
                "schema_version": 1,
                "targets": [{"revision": "abc", "commit": True}],
            },
            {
                "schema_version": 1,
                "targets": [{"revision": "abc", "unknown": True}],
            },
            {
                "schema_version": 1,
                "targets": [
                    {"revision": "abc"},
                    {"revision": "abc"},
                ],
            },
            {
                "schema_version": 1,
                "targets": [
                    {"revision": "abc", "label": "duplicate"},
                    {"revision": "def", "label": "duplicate"},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.json"
            for value in invalid_values:
                with self.subTest(value=value):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        orchestrator.read_structured_targets(path)
            path.write_text(
                '{"schema_version":1,"targets":[],"targets":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                orchestrator.read_structured_targets(path)

    def test_target_resolution_records_label_and_immutable_commit(self) -> None:
        commit = "b" * 40
        with mock.patch.object(orchestrator, "git", return_value=commit):
            target = orchestrator.resolve_target_specs(
                Path("xla"),
                [{"revision": "origin/pr-1234", "label": "release candidate"}],
            )[0]
        self.assertEqual(target["label"], "release candidate")
        self.assertEqual(target["revision"], "origin/pr-1234")
        self.assertEqual(target["commit"], commit)
        self.assertEqual(target["slug"], f"origin_pr-1234_{commit[:12]}")

    def test_configured_commit_overrides_branch_head(self) -> None:
        commit = "c" * 40
        with mock.patch.object(orchestrator, "git", return_value=commit):
            target = orchestrator.resolve_target_specs(
                Path("xla"),
                [
                    {
                        "revision": "upstream/main",
                        "commit": commit,
                        "label": "pinned main",
                    }
                ],
            )[0]
        self.assertEqual(target["revision"], "upstream/main")
        self.assertEqual(target["configured_commit"], commit)
        self.assertEqual(target["commit"], commit)

    def test_remote_revision_is_canonicalized_before_resolution(self) -> None:
        commit = "a" * 40
        with mock.patch.object(
            orchestrator,
            "git",
            side_effect=["origin\nupstream", commit],
        ) as git_mock:
            target = orchestrator.resolve_refs(
                Path("xla"), ["origin/topic"]
            )[0]
        self.assertEqual(target["commit"], commit)
        self.assertIn(
            "refs/remotes/origin/topic^{commit}",
            git_mock.call_args_list[1].args,
        )

    def test_structured_targets_allow_locally_available_revision_names(self) -> None:
        with mock.patch.object(orchestrator, "git", return_value="origin"):
            orchestrator.ensure_and_fetch_remotes(
                Path("xla"),
                ["origin/release", "feature/topic", "refs/pull/123/head"],
                skip_fetch=True,
                allow_local_refs=True,
            )
        with (
            mock.patch.object(orchestrator, "git", return_value="origin"),
            self.assertRaisesRegex(ValueError, "unknown Git remote"),
        ):
            orchestrator.ensure_and_fetch_remotes(
                Path("xla"),
                ["feature/topic"],
                skip_fetch=True,
            )

    def test_target_reader_preserves_legacy_text_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text_path = root / "targets.txt"
            text_path.write_text(
                "# candidates\norigin/release\n" + "a" * 40 + "\n",
                encoding="utf-8",
            )
            target_format, targets = orchestrator.read_target_specs(text_path)
            self.assertEqual(target_format, "text")
            self.assertEqual(
                targets,
                [
                    {"revision": "origin/release"},
                    {"revision": "a" * 40},
                ],
            )

            json_path = root / "targets.json"
            json_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "targets": [
                            {
                                "revision": "origin/release",
                                "label": "release",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            target_format, targets = orchestrator.read_target_specs(json_path)
            self.assertEqual(target_format, "json")
            self.assertEqual(
                targets,
                [{"revision": "origin/release", "label": "release"}],
            )

    def test_resume_reconciles_add_remove_labels_and_frozen_commits(self) -> None:
        control = {
            "id": "live-control:control",
            "role": "live_control",
            "ref": "live-control/origin/release",
            "source_ref": "origin/release",
            "revision": "control",
            "commit": "0" * 40,
            "slug": "control",
        }
        existing = {
            "id": "candidate:origin/existing",
            "role": "candidate",
            "ref": "origin/existing",
            "source_ref": "origin/existing",
            "revision": "origin/existing",
            "commit": "1" * 40,
            "slug": "existing",
            "label": "old label",
        }
        removed = {
            "id": "candidate:origin/removed",
            "role": "candidate",
            "ref": "origin/removed",
            "source_ref": "origin/removed",
            "revision": "origin/removed",
            "commit": "2" * 40,
            "slug": "removed",
        }
        added = {
            "id": "candidate:origin/added",
            "role": "candidate",
            "ref": "origin/added",
            "source_ref": "origin/added",
            "revision": "origin/added",
            "commit": "3" * 40,
            "slug": "added",
        }
        campaign, active = orchestrator.reconcile_campaign_targets(
            [control, existing, removed, added],
            [
                {"revision": "origin/existing", "label": "new label"},
                {"revision": "origin/added"},
            ],
        )
        self.assertEqual(
            [target["source_ref"] for target in active],
            ["origin/existing", "origin/added"],
        )
        self.assertEqual(active[0]["commit"], "1" * 40)
        self.assertEqual(active[0]["label"], "new label")
        self.assertNotIn("label", active[1])
        self.assertIn(
            "origin/removed",
            [target["source_ref"] for target in campaign],
        )
        with self.assertRaisesRegex(ValueError, "cannot change configured commit"):
            orchestrator.reconcile_campaign_targets(
                [control, existing],
                [
                    {
                        "revision": "origin/existing",
                        "commit": "4" * 40,
                    }
                ],
            )

    def test_validates_structured_target_resume_identity(self) -> None:
        commit = "c" * 40
        control = {
            "id": f"live-control:{commit}",
            "role": "live_control",
            "ref": "live-control/origin/release",
            "source_ref": "origin/release",
            "revision": commit,
            "commit": commit,
            "slug": "live_control",
        }
        candidate = {
            "id": "candidate:origin/pr-1234",
            "role": "candidate",
            "ref": "origin/pr-1234",
            "source_ref": "origin/pr-1234",
            "revision": "origin/pr-1234",
            "commit": commit,
            "slug": "candidate_pr",
            "label": "release candidate",
        }
        manifest = {
            "schema_version": 2,
            "inputs": {
                **{
                    name: {}
                    for name in (
                        "perf_tools_repo",
                        "xla_source_repo",
                        "profile_file",
                        "orchestrator_script",
                        "evaluation_script",
                        "comparison_script",
                        "reference_results_script",
                    )
                },
                "refs_file": {
                    "path": "targets.json",
                    "format": "json",
                    "sha256": "abc",
                },
            },
            "benchmark": {"effective": {}},
            "targets": [control, candidate],
            "reference_dataset": {"source": "checked_in", "inventory": {}},
        }
        orchestrator.validate_resume_manifest(manifest)
        manifest["targets"][1]["label"] = ""
        with self.assertRaisesRegex(ValueError, "invalid label"):
            orchestrator.validate_resume_manifest(manifest)

    def test_resume_updates_label_without_rerunning_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            target = {
                "id": "candidate:origin/release",
                "role": "candidate",
                "ref": "origin/release",
                "source_ref": "origin/release",
                "revision": "origin/release",
                "commit": "a" * 40,
                "slug": "release",
                "label": "new label",
            }
            metadata_path = output_dir / "release/metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        **target,
                        "label": "old label",
                        "status": "completed",
                        "evaluation_exit_code": 0,
                    }
                ),
                encoding="utf-8",
            )
            result = orchestrator.evaluate_target(
                target=target,
                source_repo=Path("source"),
                output_dir=output_dir,
                bazel="bazel",
                eval_script=Path("run_hlo_eval.sh"),
                hlo_path=Path("hlo"),
                benchmark={},
                resume=True,
            )
            self.assertEqual(result["label"], "new label")
            self.assertEqual(result["status"], "completed")

    def test_resume_does_not_reuse_finished_result_from_other_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            target = {
                "id": "candidate:origin/release",
                "role": "candidate",
                "ref": "origin/release",
                "source_ref": "origin/release",
                "revision": "origin/release",
                "commit": "a" * 40,
                "slug": "release",
            }
            metadata_path = output_dir / "release/metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        **target,
                        "commit": "b" * 40,
                        "status": "completed",
                        "evaluation_exit_code": 0,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                orchestrator,
                "require_clean_source_repo",
                side_effect=RuntimeError("rebuild required"),
            ) as clean_mock:
                result = orchestrator.evaluate_target(
                    target=target,
                    source_repo=Path("source"),
                    output_dir=output_dir,
                    bazel="bazel",
                    eval_script=Path("run_hlo_eval.sh"),
                    hlo_path=Path("hlo"),
                    benchmark={},
                    resume=True,
                )
            clean_mock.assert_called_once()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["commit"], target["commit"])

    def test_structured_target_cli_dry_run_records_immutable_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_repo = root / "source"
            source_repo.mkdir()
            output_dir = root / "output"
            targets_file = root / "targets.json"
            targets_file.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "targets": [
                            {
                                "revision": "origin/pr-1234",
                                "commit": None,
                                "label": "release candidate",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidate_commit = "e" * 40
            reference_commit = json.loads(
                PROFILE_PATH.read_text(encoding="utf-8")
            )["reference"]["xla_commit"]
            control = {
                "id": f"live-control:{reference_commit}",
                "role": "live_control",
                "ref": "live-control/origin/rocm-jaxlib-v0.10.2",
                "source_ref": "origin/rocm-jaxlib-v0.10.2",
                "revision": reference_commit,
                "commit": reference_commit,
                "slug": f"live_control_{reference_commit[:12]}",
            }

            argv = [
                "run_xla_branch_eval.py",
                "--perf-tools-repo",
                str(PERF_REPO),
                "--xla-source-repo",
                str(source_repo),
                "--output-dir",
                str(output_dir),
                "--refs-file",
                str(targets_file),
                "--hlo-path",
                "perf_tools/hlo_eval_tools/vision_diffusion/efficientnet/inference/1gpu",
                "--skip-fetch",
                "--dry-run",
            ]
            stdout = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    orchestrator,
                    "validate_git_root",
                    side_effect=lambda path, _label: Path(path).resolve(),
                ),
                mock.patch.object(
                    orchestrator, "acquire_source_lock", return_value=10
                ),
                mock.patch.object(orchestrator, "release_source_lock"),
                mock.patch.object(
                    orchestrator,
                    "require_clean_source_repo",
                    return_value={"branch": "main", "commit": "f" * 40, "status": ""},
                ),
                mock.patch.object(orchestrator, "ensure_and_fetch_remotes"),
                mock.patch.object(
                    orchestrator, "resolve_live_control", return_value=control
                ),
                mock.patch.object(
                    orchestrator, "git", return_value=candidate_commit
                ),
                mock.patch.object(
                    orchestrator,
                    "repository_metadata",
                    return_value={
                        "path": "repository",
                        "commit": "f" * 40,
                        "status": "",
                        "hlo_path": "selected-hlo",
                        "hlo_inventory": {"sha256": "inventory"},
                    },
                ),
                mock.patch.object(
                    orchestrator,
                    "reference_inventory",
                    return_value={
                        "workload_count": 1,
                        "available_count": 1,
                        "missing_count": 0,
                        "workloads": [],
                    },
                ),
                mock.patch.object(orchestrator, "choose_bazel", return_value="bazel"),
                mock.patch.object(orchestrator, "collect_environment", return_value={}),
                mock.patch.object(orchestrator.os, "access", return_value=True),
                mock.patch.object(orchestrator.signal, "signal"),
                mock.patch.object(orchestrator.signal, "getsignal"),
                mock.patch.object(
                    orchestrator.signal,
                    "pthread_sigmask",
                    return_value=set(),
                    create=True,
                ),
                mock.patch.object(
                    orchestrator.signal, "SIG_BLOCK", 0, create=True
                ),
                mock.patch.object(
                    orchestrator.signal, "SIG_SETMASK", 0, create=True
                ),
                mock.patch.object(
                    orchestrator.signal, "SIGHUP", 1, create=True
                ),
                mock.patch("sys.stdout", stdout),
            ):
                self.assertEqual(orchestrator.main(), 0)

            manifest = json.loads(stdout.getvalue())
            self.assertEqual(
                manifest["inputs"]["refs_file"]["sha256"],
                orchestrator.sha256_file(targets_file),
            )
            self.assertEqual(manifest["inputs"]["refs_file"]["format"], "json")
            self.assertEqual(
                manifest["target_specs"],
                [
                    {
                        "revision": "origin/pr-1234",
                        "commit": None,
                        "label": "release candidate",
                    }
                ],
            )
            self.assertEqual(
                manifest["targets"][1]["label"], "release candidate"
            )
            self.assertIsNone(manifest["targets"][1]["configured_commit"])
            self.assertEqual(manifest["targets"][1]["commit"], candidate_commit)
            orchestrator.validate_resume_manifest(manifest)


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
            "label": "release-candidate",
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
            comparison_csv = (output_dir / "comparison.csv").read_text(
                encoding="utf-8"
            )
            branch_summary_csv = (
                output_dir / "branch_summary.csv"
            ).read_text(encoding="utf-8")
            unlabeled_candidate = dict(candidate)
            unlabeled_candidate.pop("label")
            write_comparison(
                output_dir=output_dir,
                targets=[control, unlabeled_candidate],
                reference_dataset=reference_dataset,
                live_control_id=control["id"],
            )
            legacy_comparison_header = (
                output_dir / "comparison.csv"
            ).read_text(encoding="utf-8").splitlines()[0]
            legacy_summary_header = (
                output_dir / "branch_summary.csv"
            ).read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(result["validation"]["status"], "passed")
        self.assertEqual(len(result["branches"]), 2)
        self.assertEqual(
            result["reference_reproducibility"]["candidate_role"],
            "live_control",
        )
        self.assertIn("## Reference reproducibility", report)
        self.assertIn("origin/rocm-jaxlib-v0.10.2", report)
        self.assertIn("release-candidate", report)
        self.assertIn("candidate_label", comparison_csv.splitlines()[0])
        self.assertIn("release-candidate", comparison_csv)
        self.assertIn("candidate_label", branch_summary_csv.splitlines()[0])
        self.assertIn("release-candidate", branch_summary_csv)
        self.assertNotIn("candidate_label", legacy_comparison_header)
        self.assertNotIn("candidate_label", legacy_summary_header)
        candidate_summary = next(
            branch
            for branch in result["branches"]
            if branch["candidate_role"] == "candidate"
        )
        self.assertEqual(candidate_summary["candidate_label"], "release-candidate")
        control_summary = next(
            branch
            for branch in result["branches"]
            if branch["candidate_role"] == "live_control"
        )
        self.assertNotIn("candidate_label", control_summary)


if __name__ == "__main__":
    unittest.main()
