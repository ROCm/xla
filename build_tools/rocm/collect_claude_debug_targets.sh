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
# For every failure in a JSON Lines file (as produced by
# collect_run_failures.sh), append a detection comment to the single
# tracking issue (see create_infra_error_issue.sh) and queue it for a
# Claude debug run - but only for a failure signature (job_name/step_name)
# that doesn't already have a debug report comment on the issue. A
# recurrence of an already-analyzed signature gets neither: no new
# comment, no new debug run, so a long-standing already-understood failure
# doesn't keep adding noise to the issue every time it shows up again.
# Prints a JSON array of the failures that got queued.
#
# The caller (rocm_infra_monitor.yml) feeds this array straight into a
# `strategy.matrix.include` on its own "debug" job, so every new failure
# signature gets its own isolated Claude Code debug run on real GPU
# hardware - all within the same workflow run, no cross-workflow dispatch
# involved.
#
# Usage: collect_claude_debug_targets.sh <issue_number> <failures_jsonl_file> <workflow_run_url>
#
# Output:
#   A single compact JSON array on stdout (diagnostics on stderr), each
#   element: {issue_number, repository, head_sha, run_id, job_id, job_name,
#   step_name, failure_url, detection_comment_url}. "[]" if there is
#   nothing new to debug.
#
# Exit codes:
#   0: Success
#   1: Error

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <issue_number> <failures_jsonl_file> <workflow_run_url>" >&2
    exit 1
fi

ISSUE_NUMBER="$1"
FAILURES_FILE="$2"
WORKFLOW_URL="$3"

# Repository hosting the monitored runs and the tracking issue.
CI_REPO="ROCm/xla"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGETS="[]"

if [ -s "$FAILURES_FILE" ]; then
    # Fetch existing comments once up front - cheaper than one API call per
    # failure, and the set of *debug report* comments can't change while
    # this loop runs (the debug jobs it queues haven't started yet).
    EXISTING_COMMENTS=$(gh api --paginate "/repos/${CI_REPO}/issues/${ISSUE_NUMBER}/comments")

    # Signatures queued for debug earlier in this very run: two different
    # monitored runs in the same window can share a job/step signature, and
    # EXISTING_COMMENTS can't see a debug job this script itself is about
    # to queue, so that check alone would double-queue it.
    declare -A QUEUED_IN_THIS_RUN

    while IFS= read -r FAILURE; do
        [ -z "$FAILURE" ] && continue

        RUN_ID=$(printf '%s' "$FAILURE" | jq -r '.run_id')
        JOB_NAME=$(printf '%s' "$FAILURE" | jq -r '.job_name')
        JOB_ID=$(printf '%s' "$FAILURE" | jq -r '.job_id')
        JOB_URL=$(printf '%s' "$FAILURE" | jq -r '.job_url')
        STEP_NAME=$(printf '%s' "$FAILURE" | jq -r '.step_name')
        SIGNATURE="${JOB_NAME} / ${STEP_NAME}"

        echo "Processing failure: ${SIGNATURE} (run ${RUN_ID})" >&2

        # Check first, before posting anything: a recurrence of a signature
        # already queued earlier in this same run, or already carrying a
        # debug report comment from a previous run, gets skipped entirely -
        # no detection comment, no debug run. Cheapest check first (no API
        # call) to avoid the EXISTING_COMMENTS scan when it's not needed.
        if [ -n "${QUEUED_IN_THIS_RUN[$SIGNATURE]:-}" ]; then
            echo "  -> already queued for debug earlier in this run, skipping (no new comment)" >&2
            continue
        fi

        DEBUG_MARKER="<!-- rocm-nightly-claude-debug:${SIGNATURE} -->"
        ALREADY_DEBUGGED=$(printf '%s' "$EXISTING_COMMENTS" \
            | jq --arg marker "$DEBUG_MARKER" '[.[] | select(.body | startswith($marker))] | length')

        if [ "${ALREADY_DEBUGGED:-0}" -gt 0 ]; then
            echo "  -> already has a debug report for this signature, skipping (no new comment)" >&2
            continue
        fi

        # Not already analyzed and not queued yet this run - append a
        # detection comment and queue it for debug.
        DETECTION_BODY="<!-- rocm-nightly-failure:${SIGNATURE} -->
## Nightly CI Failure Detected

**Job:** ${JOB_NAME}
**Step:** ${STEP_NAME}

- Failing job: ${JOB_URL}
- Detected by monitoring run: ${WORKFLOW_URL}
- Detected: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"

        DETECTION_COMMENT_URL=$(gh issue comment "$ISSUE_NUMBER" --repo "$CI_REPO" --body "$DETECTION_BODY")
        echo "  -> detection comment posted: ${DETECTION_COMMENT_URL}" >&2

        # What this specific job actually checked out - NOT the run's own
        # head_sha/head_repository, which reflect ROCm/xla's commit at
        # schedule time, not the openxla/xla@main commit the job resolved
        # and tested against. See get_checkout_source.sh for why.
        CHECKOUT_INFO=$("${SCRIPT_DIR}/get_checkout_source.sh" "$CI_REPO" "$JOB_ID")
        HEAD_REPOSITORY=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '1p')
        HEAD_SHA=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '2p')

        echo "  -> queued for debug" >&2
        QUEUED_IN_THIS_RUN[$SIGNATURE]=1
        TARGET=$(jq -nc \
            --arg issue_number "$ISSUE_NUMBER" \
            --arg repository "$HEAD_REPOSITORY" \
            --arg head_sha "$HEAD_SHA" \
            --arg run_id "$RUN_ID" \
            --arg job_id "$JOB_ID" \
            --arg job_name "$JOB_NAME" \
            --arg step_name "$STEP_NAME" \
            --arg failure_url "$JOB_URL" \
            --arg detection_comment_url "$DETECTION_COMMENT_URL" \
            '{issue_number: $issue_number, repository: $repository, head_sha: $head_sha,
              run_id: $run_id, job_id: $job_id, job_name: $job_name,
              step_name: $step_name, failure_url: $failure_url,
              detection_comment_url: $detection_comment_url}')
        TARGETS=$(printf '%s' "$TARGETS" | jq -c --argjson t "$TARGET" '. + [$t]')
    done < "$FAILURES_FILE"
fi

printf '%s\n' "$TARGETS"
