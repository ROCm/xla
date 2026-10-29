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
# Print the cutoff timestamp a monitoring run should start looking from: the
# moment the monitoring pipeline itself last ran successfully.
#
# Usage: get_monitoring_window.sh <repository> <monitor_workflow> [exclude_run_id]
#
# Args:
#   repository:       GitHub repository hosting the monitor, "owner/repo"
#   monitor_workflow: Monitor workflow file name or ID
#   exclude_run_id:   Run ID to ignore, typically the currently executing run
#
# Env:
#   FALLBACK_HOURS:     Window to use when the monitor has no previous
#                       successful run, e.g. the very first time it runs
#                       (default: 2)
#   MAX_LOOKBACK_HOURS: Never look back further than this, so that a long
#                       monitor outage does not dump days' worth of failures
#                       into a single report (default: 168, one week)
#
# Output:
#   The cutoff as an ISO8601 UTC timestamp on stdout; diagnostics on stderr.
#
# Exit codes:
#   0: Success
#   1: Error

set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <repository> <monitor_workflow> [exclude_run_id]" >&2
    exit 1
fi

REPOSITORY="$1"
MONITOR_WORKFLOW="$2"
EXCLUDE_RUN_ID="${3:-0}"

FALLBACK_HOURS="${FALLBACK_HOURS:-2}"
MAX_LOOKBACK_HOURS="${MAX_LOOKBACK_HOURS:-168}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Current server time, taken from the API response headers so that the result
# does not depend on the runner's clock. Note that rate_limit.reset is *not*
# the current time - it is up to an hour in the future.
NOW_RAW=$(gh api /rate_limit --include --silent 2>/dev/null \
    | grep -i '^date:' | head -1 | cut -d' ' -f2- | tr -d '\r')
if [ -z "$NOW_RAW" ]; then
    echo "Error: could not read current time from the GitHub API" >&2
    exit 1
fi
NOW=$(date -u -d "$NOW_RAW" '+%Y-%m-%dT%H:%M:%SZ')

LAST_RUN=$("${SCRIPT_DIR}/get_last_pipeline_run.sh" \
    "$REPOSITORY" "$MONITOR_WORKFLOW" "$EXCLUDE_RUN_ID")

if [ -n "$LAST_RUN" ]; then
    CUTOFF="$LAST_RUN"
    LAST_RUN_DISPLAY="$LAST_RUN"
else
    CUTOFF=$(date -u -d "$NOW - ${FALLBACK_HOURS} hours" '+%Y-%m-%dT%H:%M:%SZ')
    LAST_RUN_DISPLAY="never (no previous successful run, using a ${FALLBACK_HOURS}h window)"
fi

EARLIEST=$(date -u -d "$NOW - ${MAX_LOOKBACK_HOURS} hours" '+%Y-%m-%dT%H:%M:%SZ')
if [[ "$CUTOFF" < "$EARLIEST" ]]; then
    echo "Cutoff $CUTOFF is older than the ${MAX_LOOKBACK_HOURS}h limit, clamping" >&2
    CUTOFF="$EARLIEST"
fi

{
    echo "Current time (GitHub server): $NOW"
    echo "Last monitor run:             $LAST_RUN_DISPLAY"
    echo "Window starts at:             $CUTOFF"
} >&2

echo "$CUTOFF"
