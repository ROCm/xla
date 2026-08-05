#!/usr/bin/env python3
"""Small file helpers owned by the HLO stability tool."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_identity(repo_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(arguments)} failed: "
                f"{completed.stdout.strip()}"
            )
        return completed.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(
            git("status", "--porcelain", "--untracked-files=all")
        ),
    }


def tooling_metadata(
    script_dir: Path,
    repo_root: Path,
    names: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    return {
        name: {
            "relative_path": str(
                (script_dir / name).resolve().relative_to(repo_root)
            ),
            "sha256": sha256_file(script_dir / name),
        }
        for name in names
    }
