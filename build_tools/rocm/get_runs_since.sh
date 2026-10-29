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
# List the runs of a pipeline that finished after a given timestamp, one run ID
# per line.
#
# Usage: get_runs_since.sh <repository> <workflow> <cutoff>
#
# Args:
#   repository: GitHub repository, "owner/repo"
#   workflow:   Workflow file name (e.g. "rocm_xla_ci_nightly.yml") or ID
#   cutoff:     ISO8601 UTC timestamp; runs that finished after it are listed
#
# Env:
#   CREATION_BUFFER_HOURS: How long before the cutoff a run may have been
#                          created and still be relevant (default: 48).
#                          "created" is the only date filter the REST API
#                          offers, but a run can take many hours, so one that
#                          finished inside the window may well have been
#                          created long before it.
#
# Output:
#   Run IDs on stdout, newest first; nothing if no run finished since the
#   cutoff. No conclusion filter is applied: a job marked continue-on-error
#   leaves the run's own conclusion green even though it failed, so the
#   caller has to inspect the jobs of every finished run.
#
# Exit codes:
#   0: Success (including "no runs found")
#   1: Error

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <repository> <workflow> <cutoff>" >&2
    exit 1
fi

REPOSITORY="$1"
WORKFLOW="$2"
CUTOFF="$3"

CREATION_BUFFER_HOURS="${CREATION_BUFFER_HOURS:-48}"

CREATED_SINCE=$(date -u -d "$CUTOFF - ${CREATION_BUFFER_HOURS} hours" '+%Y-%m-%dT%H:%M:%SZ')
echo "Listing runs of ${WORKFLOW} created after ${CREATED_SINCE} and finished after ${CUTOFF}" >&2

# --paginate to get ALL runs, not just the first page.
gh api --paginate \
    -H "Accept: application/vnd.github+json" \
    "/repos/${REPOSITORY}/actions/workflows/${WORKFLOW}/runs?status=completed&created=>${CREATED_SINCE}&per_page=100" \
    --jq ".workflow_runs[] | select(.updated_at > \"${CUTOFF}\") | .id"
