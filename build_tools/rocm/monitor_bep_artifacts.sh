#!/usr/bin/env bash
# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
#
# Download ROCm CI BEP artifacts from failed runs.
#
# Usage: monitor_bep_artifacts.sh <run_ids_file> <repository>
#
# Args:
#   run_ids_file: File containing workflow run IDs (one per line)
#   repository: GitHub repository in format "owner/repo"
#
# Exit codes:
#   0: No runs with BEP files found
#   1: Runs with BEP files found
#   2: Script error

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <run_ids_file> <repository>" >&2
    exit 2
fi

RUN_IDS_FILE="$1"
REPOSITORY="$2"

if [ ! -f "$RUN_IDS_FILE" ]; then
    echo "Error: Run IDs file not found: $RUN_IDS_FILE" >&2
    exit 2
fi

mkdir -p bep-files
HAS_BEPS=false

while read -r RUN_ID; do
    [ -z "$RUN_ID" ] && continue

    echo "========================================="
    echo "Downloading BEP artifacts for run: $RUN_ID"
    echo "========================================="

    # Download JAX BEP artifact
    if gh run download "$RUN_ID" -n jax-bep -D "bep-files/run-${RUN_ID}" -R "$REPOSITORY" 2>/dev/null; then
        echo "Downloaded JAX BEP for run $RUN_ID from $REPOSITORY"
        HAS_BEPS=true
    fi

    # Download XLA BEP artifacts
    if gh run download "$RUN_ID" -n xla-bep -D "bep-files/run-${RUN_ID}" -R "$REPOSITORY" 2>/dev/null; then
        echo "Downloaded XLA BEP for run $RUN_ID from $REPOSITORY"
        HAS_BEPS=true
    fi
done < "$RUN_IDS_FILE"

echo "========================================="
if [ "$HAS_BEPS" = true ]; then
    echo "Downloaded BEP files from failed runs"
    echo "========================================="
    exit 1
else
    echo "✓ No BEP files found in the checked runs"
    echo "========================================="
    exit 0
fi
