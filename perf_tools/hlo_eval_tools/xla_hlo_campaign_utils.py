#!/usr/bin/env python3
"""Shared utility functions for XLA HLO campaigns and reports."""

from __future__ import annotations

import re
from collections.abc import Iterable


ROCM_VERSION_RE = re.compile(r"(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)")
GPU_ARCHITECTURE_RE = re.compile(r"gfx[0-9a-z]+", re.IGNORECASE)


def normalize_rocm_version(value: object) -> str | None:
    """Return one semantic ROCm version from a string-like value."""
    if not isinstance(value, str):
        return None
    match = ROCM_VERSION_RE.search(value)
    return match.group(1) if match else None


def normalize_gpu_architectures(values: object) -> list[str]:
    """Return sorted, unique, normalized AMD GPU architecture names."""
    candidates: Iterable[object]
    if isinstance(values, str):
        candidates = (values,)
    elif isinstance(values, Iterable):
        candidates = values
    else:
        return []
    return sorted(
        {
            value.lower()
            for value in candidates
            if isinstance(value, str)
            and GPU_ARCHITECTURE_RE.fullmatch(value)
        }
    )
