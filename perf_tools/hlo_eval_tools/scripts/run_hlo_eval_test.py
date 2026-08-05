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
"""Focused tests for run_hlo_eval.sh and legacy profile conversion."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import legacy_profile_to_csv


TOOLS_ROOT = SCRIPT_DIR.parent
EVAL_SCRIPT = TOOLS_ROOT / "run_hlo_eval.sh"


def write_hlo(path: Path, partitions: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"HloModule test, num_partitions={partitions}\n\nENTRY main {{}}\n",
        encoding="utf-8",
    )


def write_fake_runner(path: Path, native_csv: bool) -> None:
    if native_csv:
        body = r"""#!/bin/bash
set -uo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "--append_profile_to_csv_file"
  exit 0
fi
base=""
files=()
for arg in "$@"; do
  case "$arg" in
    --append_profile_to_csv_file=*) base=${arg#*=} ;;
    --*) ;;
    *) files+=("$arg") ;;
  esac
done
mapfile -t files < <(printf "%s\n" "${files[@]}" | LC_ALL=C sort)
if [[ -z "$base" ]]; then
  echo "missing CSV base" >&2
  exit 2
fi
if [[ "${FAKE_SKIP_CSV:-0}" != 0 ]]; then
  exit "${FAKE_RUNNER_RC:-0}"
fi
mkdir -p "$(dirname "$base")"
if [[ ! -s "${base}.csv" ]]; then
  {
    printf "Datetime"
    for file in "${files[@]}"; do printf ",%s" "$(basename "$file")"; done
    printf "\n"
  } > "${base}.csv"
fi
if [[ "${FAKE_PARTIAL_CSV:-0}" != 0 ]]; then
  printf "2026-01-01 00:00:00,\n" >> "${base}.csv"
  exit "${FAKE_RUNNER_RC:-0}"
fi
if [[ "${FAKE_NO_FINAL_NEWLINE:-0}" != 0 ]]; then
  printf "2026-01-01 00:00:00" >> "${base}.csv"
  for file in "${files[@]}"; do printf ", 1ms" >> "${base}.csv"; done
  exit "${FAKE_RUNNER_RC:-0}"
fi
{
  printf "2026-01-01 00:00:00"
  for file in "${files[@]}"; do printf ", 1ms"; done
  printf "\n"
} >> "${base}.csv"
if [[ "${FAKE_EXTRA_CSV_ROW:-0}" != 0 ]]; then
  {
    printf "2026-01-01 00:00:01"
    for file in "${files[@]}"; do printf ", 2ms"; done
    printf "\n"
  } >> "${base}.csv"
fi
exit "${FAKE_RUNNER_RC:-0}"
"""
    else:
        body = r"""#!/bin/bash
set -uo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "legacy runner"
  exit 0
fi
repeats=1
files=()
for arg in "$@"; do
  case "$arg" in
    --num_repeats=*) repeats=${arg#*=} ;;
    --*) ;;
    *) files+=("$arg") ;;
  esac
done
for file in "${files[@]}"; do
  limit=$repeats
  if [[ "${FAKE_DROP_LAST_REPEAT:-0}" != 0 ]]; then
    limit=$((repeats - 1))
  fi
  for ((repeat=0; repeat<limit; repeat++)); do
    duration=$(((repeat + 1) * 1000000))
    echo "## Execution time, file=$file repeat=$repeat duration=${duration}ns"
  done
done
exit "${FAKE_RUNNER_RC:-0}"
"""
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def read_data_rows(path: Path) -> list[list[str]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    return list(csv.reader(lines))


class LegacyProfileConverterTest(unittest.TestCase):
    def test_warmup_repeat_is_excluded_from_average(self) -> None:
        self.assertEqual(
            legacy_profile_to_csv.averaged_ms(
                {0: 100_000_000, 1: 2_000_000, 2: 4_000_000}
            ),
            3.0,
        )

    def test_basename_fallback_requires_unique_match(self) -> None:
        profiles = {
            "/first/module.txt": {0: 1},
            "/second/module.txt": {0: 2},
        }
        self.assertIsNone(
            legacy_profile_to_csv.find_file_profiles(
                profiles, Path("/other/module.txt")
            )
        )


class RunHloEvalTest(unittest.TestCase):
    def run_eval(
        self,
        runner: Path,
        hlo_path: Path,
        output: Path,
        repeats: str = "2",
        profile_mode: str = "csv",
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "CMD_BUFFER": "off",
                "PROFILE_OUTPUT_MODE": profile_mode,
                "SETTLE_SEC": "0",
            }
        )
        if extra_env:
            environment.update(extra_env)
        return subprocess.run(
            [
                "bash",
                str(EVAL_SCRIPT),
                str(runner),
                str(hlo_path),
                str(output),
                repeats,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_rejects_invalid_repeat_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            hlo = root / "inference/1gpu/module.txt"
            write_fake_runner(runner, native_csv=True)
            write_hlo(hlo)

            result = self.run_eval(runner, hlo, root / "out", repeats="0")

            self.assertEqual(result.returncode, 2)
            self.assertIn("num_repeats must be a positive integer", result.stderr)

    def test_native_csv_is_published_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            hlo = root / "inference/1gpu/module.txt"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(hlo)

            result = self.run_eval(runner, hlo, output, profile_mode="auto")

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = read_data_rows(output)
            self.assertEqual(rows[0], ["Datetime", "module.txt"])
            self.assertEqual(rows[1][1].strip(), "1ms")
            self.assertFalse((root / ".tmp").exists())
            self.assertNotIn("integer expression expected", result.stderr)

    def test_native_csv_appends_one_complete_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            hlo = root / "inference/1gpu/module.txt"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(hlo)
            output.write_text(
                "Datetime,module.txt\n2025-01-01 00:00:00, 2ms\n",
                encoding="utf-8",
            )

            result = self.run_eval(runner, hlo, output)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = read_data_rows(output)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1][1].strip(), "2ms")
            self.assertEqual(rows[2][1].strip(), "1ms")
            self.assertNotIn("integer expression expected", result.stderr)

    def test_mixed_hlo_extensions_use_stable_header_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            leaf = root / "model/training"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(leaf / "z.txt")
            write_hlo(leaf / "a.hlo")

            result = self.run_eval(runner, leaf, output)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = read_data_rows(output)
            self.assertEqual(rows[0], ["Datetime", "a.hlo", "z.txt"])
            self.assertEqual(len(rows[1]), 3)

    def test_legacy_profiles_are_converted_and_exclude_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "legacy_runner"
            hlo = root / "inference/1gpu/module.txt"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=False)
            write_hlo(hlo)

            result = self.run_eval(
                runner,
                hlo,
                output,
                repeats="3",
                profile_mode="auto",
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = read_data_rows(output)
            self.assertEqual(rows[0], ["Datetime", "module.txt"])
            self.assertEqual(rows[1][1].strip(), "2.5ms")
            self.assertTrue((root / "result.legacy.log").is_file())

    def test_incomplete_legacy_profiles_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "legacy_runner"
            hlo = root / "inference/1gpu/module.txt"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=False)
            write_hlo(hlo)

            result = self.run_eval(
                runner,
                hlo,
                output,
                repeats="3",
                profile_mode="legacy",
                extra_env={"FAKE_DROP_LAST_REPEAT": "1"},
            )

            self.assertEqual(result.returncode, 1)
            self.assertFalse(output.exists())
            self.assertIn("incomplete repeat profiles", result.stdout + result.stderr)

    def test_failed_runner_does_not_replace_existing_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            hlo = root / "inference/1gpu/module.txt"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(hlo)
            original = "Datetime,module.txt\n2025-01-01 00:00:00, 1ms\n"
            output.write_text(original, encoding="utf-8")

            result = self.run_eval(
                runner,
                hlo,
                output,
                extra_env={"FAKE_RUNNER_RC": "17"},
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), original)
            self.assertIn("runner_rc=17", result.stdout)

    def test_successful_runner_without_csv_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            hlo = root / "inference/1gpu/module.txt"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(hlo)

            result = self.run_eval(
                runner,
                hlo,
                output,
                extra_env={"FAKE_SKIP_CSV": "1"},
            )

            self.assertEqual(result.returncode, 1)
            self.assertFalse(output.exists())
            self.assertIn("publish_rc=1", result.stdout)

    def test_partial_native_csv_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            hlo = root / "inference/1gpu/module.txt"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(hlo)

            result = self.run_eval(
                runner,
                hlo,
                output,
                extra_env={"FAKE_PARTIAL_CSV": "1"},
            )

            self.assertEqual(result.returncode, 1)
            self.assertFalse(output.exists())
            self.assertIn("incomplete or unexpected CSV output", result.stdout)

    def test_native_csv_with_extra_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            hlo = root / "inference/1gpu/module.txt"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(hlo)

            result = self.run_eval(
                runner,
                hlo,
                output,
                extra_env={"FAKE_EXTRA_CSV_ROW": "1"},
            )

            self.assertEqual(result.returncode, 1)
            self.assertFalse(output.exists())
            self.assertIn("incomplete or unexpected CSV output", result.stdout)

    def test_native_csv_without_final_newline_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            hlo = root / "inference/1gpu/module.txt"
            output = root / "result.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(hlo)

            result = self.run_eval(
                runner,
                hlo,
                output,
                extra_env={"FAKE_NO_FINAL_NEWLINE": "1"},
            )

            self.assertEqual(result.returncode, 1)
            self.assertFalse(output.exists())
            self.assertIn("incomplete or unexpected CSV output", result.stdout)

    def test_rejects_inconsistent_leaf_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            leaf = root / "model/training"
            write_fake_runner(runner, native_csv=True)
            write_hlo(leaf / "first.txt", partitions=1)
            write_hlo(leaf / "second.txt", partitions=2)

            result = self.run_eval(runner, leaf, root / "out")

            self.assertEqual(result.returncode, 1)
            self.assertIn("inconsistent num_partitions", result.stdout)

    def test_rejects_gpu_leaf_partition_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            leaf = root / "model/inference/2gpu"
            write_fake_runner(runner, native_csv=True)
            write_hlo(leaf / "module.txt", partitions=1)

            result = self.run_eval(runner, leaf, root / "out")

            self.assertEqual(result.returncode, 1)
            self.assertIn("inference path expects 2 partition(s)", result.stdout)

    def test_single_file_rejects_gpu_path_partition_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            hlo = root / "model/inference/2gpu/module.txt"
            write_fake_runner(runner, native_csv=True)
            write_hlo(hlo, partitions=1)

            result = self.run_eval(runner, hlo, root / "out")

            self.assertEqual(result.returncode, 1)
            self.assertIn("inference path expects 2 partition(s)", result.stdout)

    def test_resume_keeps_existing_result_without_invoking_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            leaf = root / "hlo_eval_tools/category/model/inference/1gpu"
            output_dir = root / "out"
            expected = output_dir / "category_model_inference_1gpu.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(leaf / "module.txt")
            expected.parent.mkdir(parents=True)
            original = "Datetime,module.txt\n2025-01-01 00:00:00, 1ms\n"
            expected.write_text(original, encoding="utf-8")

            result = self.run_eval(
                runner,
                leaf,
                output_dir,
                extra_env={
                    "FAKE_RUNNER_RC": "17",
                    "RESUME": "1",
                },
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(expected.read_text(encoding="utf-8"), original)
            self.assertIn("skip (resume, CSV exists)", result.stdout)

    def test_resume_rejects_incomplete_existing_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            leaf = root / "hlo_eval_tools/category/model/inference/1gpu"
            output_dir = root / "out"
            existing = output_dir / "category_model_inference_1gpu.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(leaf / "module.txt")
            existing.parent.mkdir(parents=True)
            existing.write_text("Datetime,module.txt\n", encoding="utf-8")

            result = self.run_eval(
                runner,
                leaf,
                output_dir,
                extra_env={"RESUME": "1"},
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("resume CSV is incomplete", result.stdout)

    def test_resume_rejects_partial_existing_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            leaf = root / "hlo_eval_tools/category/model/inference/1gpu"
            output_dir = root / "out"
            existing = output_dir / "category_model_inference_1gpu.csv"
            write_fake_runner(runner, native_csv=True)
            write_hlo(leaf / "module.txt")
            existing.parent.mkdir(parents=True)
            existing.write_text(
                "Datetime,module.txt\n2025-01-01 00:00:00,\n",
                encoding="utf-8",
            )

            result = self.run_eval(
                runner,
                leaf,
                output_dir,
                extra_env={"RESUME": "1"},
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("resume CSV is incomplete", result.stdout)


if __name__ == "__main__":
    unittest.main()
