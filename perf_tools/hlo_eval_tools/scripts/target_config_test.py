#!/usr/bin/env python3
# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Focused tests for the shared HLO evaluation target configuration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import target_config


def write_config(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8", newline="\n")


def valid_config(targets: list[object]) -> dict[str, object]:
    return {"schema_version": 1, "targets": targets}


class TargetConfigParsingTest(unittest.TestCase):
    def test_loads_checked_in_shared_configuration(self) -> None:
        specs = target_config.load_target_specs()

        self.assertEqual(len(specs), 3)
        self.assertEqual([spec.role for spec in specs], [
            "control",
            "candidate",
            "candidate",
        ])
        self.assertEqual(
            specs[0].commit,
            "7b5ecf1c9282fdf1039211e0d45216980058beda",
        )
        self.assertEqual(specs[0].label, "v0.10.2 pinned")
        self.assertIsNone(specs[1].commit)
        self.assertEqual(specs[0].revision, specs[1].revision)

    def test_rejects_invalid_document_schema(self) -> None:
        invalid_values = [
            [],
            {},
            {"schema_version": True, "targets": []},
            {"schema_version": 2, "targets": []},
            {"schema_version": 1, "targets": []},
            {
                "schema_version": 1,
                "targets": [{"revision": "main", "role": "control"}],
                "unknown": True,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.json"
            for value in invalid_values:
                with self.subTest(value=value):
                    write_config(path, value)
                    with self.assertRaises(ValueError):
                        target_config.load_target_specs(path)

    def test_rejects_invalid_target_fields(self) -> None:
        invalid_targets: list[object] = [
            "not-an-object",
            {"role": "control"},
            {"revision": "-bad", "role": "control"},
            {"revision": "bad ref", "role": "control"},
            {"revision": "main", "commit": "short", "role": "control"},
            {"revision": "main", "commit": True, "role": "control"},
            {"revision": "main", "label": None, "role": "control"},
            {"revision": "main", "label": " bad", "role": "control"},
            {"revision": "main", "role": "unknown"},
            {"revision": "main", "role": "control", "id": "user-id"},
            {"revision": "main", "role": "control", "modes": ["campaign"]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.json"
            for target in invalid_targets:
                with self.subTest(target=target):
                    write_config(path, valid_config([target]))
                    with self.assertRaises(ValueError):
                        target_config.load_target_specs(path)

    def test_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.json"
            path.write_text(
                '{"schema_version":1,"targets":[],"targets":[]}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate key"):
                target_config.load_target_specs(path)

    def test_requires_exactly_one_control(self) -> None:
        invalid_target_lists = [
            [{"revision": "main"}, {"revision": "release"}],
            [
                {"revision": "main", "role": "control"},
                {"revision": "release", "role": "control"},
            ],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.json"
            for targets in invalid_target_lists:
                with self.subTest(targets=targets):
                    write_config(path, valid_config(targets))
                    with self.assertRaisesRegex(ValueError, "exactly one control"):
                        target_config.load_target_specs(path)


class TargetResolutionTest(unittest.TestCase):
    def test_resolves_heads_and_pins_to_deterministic_identity(self) -> None:
        head_commit = "a" * 40
        pinned_commit = "b" * 40
        specs = [
            target_config.TargetSpec(
                revision="origin/release",
                role="control",
                label="release HEAD",
            ),
            target_config.TargetSpec(
                revision="upstream/main",
                commit=pinned_commit,
                label="pinned main",
            ),
        ]

        with mock.patch.object(
            target_config,
            "resolve_revision",
            side_effect=[head_commit, pinned_commit],
        ) as resolver:
            resolved = target_config.resolve_target_specs(Path("xla"), specs)

        self.assertEqual(
            [call.args[1] for call in resolver.call_args_list],
            ["origin/release", pinned_commit],
        )
        self.assertEqual(
            resolved[0].id,
            f"target:origin/release@{head_commit}",
        )
        self.assertEqual(
            resolved[0].slug,
            f"origin_release_{head_commit[:12]}",
        )
        self.assertEqual(resolved[0].role, "control")
        self.assertIsNone(resolved[0].configured_commit)
        self.assertEqual(resolved[1].configured_commit, pinned_commit)
        self.assertEqual(resolved[1].role, "candidate")

    def test_label_does_not_change_target_identity(self) -> None:
        commit = "c" * 40
        first = [
            target_config.TargetSpec(
                revision="origin/release",
                role="control",
                label="first label",
            )
        ]
        second = [
            target_config.TargetSpec(
                revision="origin/release",
                role="control",
                label="changed label",
            )
        ]

        with mock.patch.object(target_config, "resolve_revision", return_value=commit):
            first_target = target_config.resolve_target_specs(
                Path("xla"), first
            )[0]
            second_target = target_config.resolve_target_specs(
                Path("xla"), second
            )[0]

        self.assertEqual(first_target.id, second_target.id)
        self.assertEqual(first_target.slug, second_target.slug)
        self.assertNotEqual(first_target.label, second_target.label)

    def test_rejects_duplicate_resolved_commits(self) -> None:
        commit = "d" * 40
        specs = [
            target_config.TargetSpec("origin/release", role="control"),
            target_config.TargetSpec("upstream/main"),
        ]

        with (
            mock.patch.object(
                target_config,
                "resolve_revision",
                side_effect=[commit, commit],
            ),
            self.assertRaisesRegex(ValueError, "resolve to the same commit"),
        ):
            target_config.resolve_target_specs(Path("xla"), specs)

    def test_remote_branch_is_canonicalized_before_resolution(self) -> None:
        commit = "e" * 40
        with mock.patch.object(
            target_config,
            "_git",
            side_effect=["origin\nupstream", commit],
        ) as git_mock:
            resolved = target_config.resolve_revision(
                Path("xla"), "origin/release"
            )

        self.assertEqual(resolved, commit)
        self.assertIn(
            "refs/remotes/origin/release^{commit}",
            git_mock.call_args_list[1].args,
        )

    def test_resolved_head_remains_immutable_after_branch_moves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "GIT_AUTHOR_NAME": "Target Config Test",
                    "GIT_AUTHOR_EMAIL": "target-config@example.com",
                    "GIT_COMMITTER_NAME": "Target Config Test",
                    "GIT_COMMITTER_EMAIL": "target-config@example.com",
                }
            )
            tracked = repo / "tracked.txt"
            tracked.write_text("first\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-q", "-m", "first"],
                check=True,
                env=environment,
            )
            first_commit = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            resolved = target_config.resolve_target_specs(
                repo,
                [target_config.TargetSpec("HEAD", role="control")],
            )[0]
            tracked.write_text("second\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-q", "-m", "second"],
                check=True,
                env=environment,
            )
            current_head = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        self.assertEqual(resolved.commit, first_commit)
        self.assertNotEqual(current_head, first_commit)


if __name__ == "__main__":
    unittest.main()
