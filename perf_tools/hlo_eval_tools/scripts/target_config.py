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
"""Load and resolve the shared HLO evaluation target configuration."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


TARGET_SCHEMA_VERSION = 1
DEFAULT_TARGET_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "xla_targets.json"
)
_FULL_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")
_TARGET_ROLES = {"control", "candidate"}


@dataclass(frozen=True)
class TargetSpec:
    revision: str
    commit: str | None = None
    role: str = "candidate"
    label: str | None = None


@dataclass(frozen=True)
class ResolvedTarget:
    id: str
    revision: str
    configured_commit: str | None
    commit: str
    role: str
    slug: str
    label: str | None = None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"target configuration contains duplicate key: {key}")
        result[key] = value
    return result


def _validate_revision(path: Path, index: int, value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
        or value.startswith("-")
    ):
        raise ValueError(f"{path}: target {index} has an invalid revision")
    return value


def _validate_commit(path: Path, index: int, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _FULL_SHA_RE.fullmatch(value):
        raise ValueError(
            f"{path}: target {index} commit must be null or a full "
            "40-character SHA"
        )
    return value.lower()


def _validate_label(path: Path, index: int, value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 128
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{path}: target {index} has an invalid label")
    return value


def _require_one_control(specs: Sequence[TargetSpec]) -> None:
    if not specs:
        raise ValueError("target configuration must contain at least one target")
    invalid_roles = sorted({spec.role for spec in specs} - _TARGET_ROLES)
    if invalid_roles:
        raise ValueError(f"unsupported target role(s): {', '.join(invalid_roles)}")
    controls = sum(spec.role == "control" for spec in specs)
    if controls != 1:
        raise ValueError(
            "target configuration must contain exactly one control; "
            f"found {controls}"
        )


def load_target_specs(path: Path = DEFAULT_TARGET_CONFIG) -> list[TargetSpec]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid target configuration JSON in {path}: {error}") from error

    if not isinstance(value, dict):
        raise ValueError(f"target configuration must contain a JSON object: {path}")
    required = {"schema_version", "targets"}
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        raise ValueError(
            "target configuration must contain exactly schema_version and "
            f"targets; missing={missing}, unsupported={extra}: {path}"
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != TARGET_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported target schema: {value['schema_version']!r}; "
            f"expected {TARGET_SCHEMA_VERSION}"
        )

    raw_targets = value["targets"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError(
            f"target configuration must contain a non-empty targets list: {path}"
        )

    specs: list[TargetSpec] = []
    allowed_fields = {"revision", "commit", "role", "label"}
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, dict):
            raise ValueError(f"{path}: target {index} must be a JSON object")
        missing_fields = {"revision"} - set(raw_target)
        extra_fields = set(raw_target) - allowed_fields
        if missing_fields or extra_fields:
            raise ValueError(
                f"{path}: target {index} has missing={sorted(missing_fields)} "
                f"and unsupported={sorted(extra_fields)} fields"
            )

        revision = _validate_revision(path, index, raw_target["revision"])
        commit = _validate_commit(path, index, raw_target.get("commit"))
        role = raw_target.get("role", "candidate")
        if not isinstance(role, str) or role not in _TARGET_ROLES:
            raise ValueError(f"{path}: target {index} has an invalid role")
        label = (
            _validate_label(path, index, raw_target["label"])
            if "label" in raw_target
            else None
        )
        specs.append(
            TargetSpec(
                revision=revision,
                commit=commit,
                role=role,
                label=label,
            )
        )

    _require_one_control(specs)
    return specs


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {repo}: {details}"
        )
    return completed.stdout.strip()


def canonical_revision(repo: Path, revision: str) -> str:
    if revision.startswith("refs/") or "/" not in revision:
        return revision
    remote, _ = revision.split("/", 1)
    remotes = set(_git(repo, "remote").splitlines())
    if remote in remotes:
        return f"refs/remotes/{revision}"
    return revision


def resolve_revision(repo: Path, revision: str) -> str:
    canonical = canonical_revision(repo, revision)
    output = _git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{canonical}^{{commit}}",
    )
    if not _FULL_SHA_RE.fullmatch(output):
        raise RuntimeError(
            f"revision {revision!r} did not resolve to one commit SHA: {output!r}"
        )
    return output.lower()


def _target_slug(revision: str, commit: str) -> str:
    revision_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", revision)
    revision_slug = revision_slug.strip("._-") or "xla"
    return f"{revision_slug}_{commit[:12]}"


def resolve_target_specs(
    repo: Path, specs: Sequence[TargetSpec]
) -> list[ResolvedTarget]:
    _require_one_control(specs)
    resolved: list[ResolvedTarget] = []
    seen_commits: dict[str, str] = {}
    for spec in specs:
        requested = spec.commit if spec.commit is not None else spec.revision
        commit = resolve_revision(repo, requested)
        previous_revision = seen_commits.get(commit)
        if previous_revision is not None:
            raise ValueError(
                f"targets {previous_revision!r} and {spec.revision!r} "
                f"resolve to the same commit: {commit}"
            )
        seen_commits[commit] = spec.revision
        resolved.append(
            ResolvedTarget(
                id=f"target:{spec.revision}@{commit}",
                revision=spec.revision,
                configured_commit=spec.commit,
                commit=commit,
                role=spec.role,
                slug=_target_slug(spec.revision, commit),
                label=spec.label,
            )
        )
    return resolved
