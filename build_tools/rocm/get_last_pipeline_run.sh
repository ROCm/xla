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
# Print when a pipeline last ran, as an ISO8601 UTC timestamp.
#
# Usage: get_last_pipeline_run.sh <repository> <workflow> [exclude_run_id]
#
# Args:
#   repository:     GitHub repository, "owner/repo"
#   workflow:       Workflow file name (e.g. "rocm_xla_ci_nightly.yml") or ID
#   exclude_run_id: Run ID to ignore, typically the currently executing run
#
# Env:
#   RUN_STATUS: Run status/conclusion to look for (default: success)
#   RUN_EVENT:  Trigger event to look for, empty for any (default: schedule)
#   RUN_FIELD:  Timestamp to report, "created_at" (when the run started) or
#               "updated_at" (when it finished). Default: created_at, which is
#               the safe choice for a monitoring cutoff: work that finished
#               while the previous monitor run was in flight is not missed.
#
# Output:
#   The timestamp on stdout, or nothing at all if the pipeline has no matching
#   run yet. Callers must handle the empty case.
#
# Exit codes:
#   0: Success (including "no matching run found")
#   1: Error

set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <repository> <workflow> [exclude_run_id]" >&2
    exit 1
fi

REPOSITORY="$1"
WORKFLOW="$2"
EXCLUDE_RUN_ID="${3:-0}"

RUN_STATUS="${RUN_STATUS:-success}"
# No colon: an explicitly empty RUN_EVENT means "any event", only an unset one
# falls back to the default.
RUN_EVENT="${RUN_EVENT-schedule}"
RUN_FIELD="${RUN_FIELD:-created_at}"

QUERY="status=${RUN_STATUS}&per_page=50"
if [ -n "$RUN_EVENT" ]; then
    QUERY="${QUERY}&event=${RUN_EVENT}"
fi

# The runs are returned newest first, but take the max explicitly rather than
# relying on the ordering.
gh api \
    -H "Accept: application/vnd.github+json" \
    "/repos/${REPOSITORY}/actions/workflows/${WORKFLOW}/runs?${QUERY}" \
    --jq "[.workflow_runs[] | select(.id != ${EXCLUDE_RUN_ID}) | .${RUN_FIELD}] | max // empty"
