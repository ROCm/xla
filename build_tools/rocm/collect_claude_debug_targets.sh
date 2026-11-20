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
# Two phases, every run:
#
# Phase 1 - report new failures. For every failure in a JSON Lines file (as
# produced by collect_run_failures.sh), append ONE detection comment to the
# single tracking issue (see create_infra_error_issue.sh) - but only for a
# failure signature (job_name/step_name) that doesn't already have ANY
# comment on the issue, analyzed or not. A recurrence of a signature that
# already has a comment gets no new comment here, regardless of whether
# that existing comment has actually been analyzed yet - that's Phase 2's
# job, not this one's. This is what keeps the issue from accumulating
# duplicate comments for the same signature.
#
# Phase 2 - catch up anything still unanalyzed. Separately, and
# unconditionally (even on a run with no new failures at all), scan every
# comment already on the issue and queue a debug run for any that carries a
# `rocm-nightly-failure:` marker but no matching `rocm-nightly-claude-debug:`
# marker yet - i.e. it was posted (by this run's Phase 1, an earlier run's
# Phase 1, or an earlier run's Phase 2 that itself never got dispatched/
# completed) but never actually got analyzed. This is what makes a comment
# self-heal on the next run after something stopped its first attempt
# (a cancelled monitoring run, a PR-validation-skipped debug job, a
# transient infra error) instead of sitting unanalyzed forever.
#
# Both phases queue into the same output array, so both feed the exact same
# `strategy.matrix.include` on the caller's "debug" job - a Phase 2 entry
# behaves identically to a fresh Phase 1 one as far as the debug skill is
# concerned, just with its checkout info recovered from the existing
# comment's own "Failing job:" line instead of from collect_run_failures.sh.
#
# Usage: collect_claude_debug_targets.sh <issue_number> <failures_jsonl_file> <workflow_run_url>
#
# Output:
#   A single compact JSON array on stdout (diagnostics on stderr), each
#   element: {issue_number, repository, head_sha, run_id, job_id, job_name,
#   step_name, failure_url, detection_comment_url}. "[]" if there is
#   nothing to debug.
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

# Fetched once up front and reused by both phases - cheaper than one API
# call per failure/comment, and nothing posted by Phase 1 needs to be
# visible to Phase 2 through a second fetch: Phase 1 already queues
# whatever it posts directly (see QUEUED_IN_THIS_RUN below), so Phase 2
# would just be re-discovering the same signature redundantly.
EXISTING_COMMENTS=$(gh api --paginate "/repos/${CI_REPO}/issues/${ISSUE_NUMBER}/comments")

# Signatures queued in this run, by either phase - two different monitored
# runs in the same window can share a job/step signature (Phase 1), and
# Phase 2 must not re-queue something Phase 1 (or an earlier Phase-2 entry
# this same run) already did.
declare -A QUEUED_IN_THIS_RUN

# ---------------------------------------------------------------------------
# Phase 1 - report failures that don't have a comment yet.
# ---------------------------------------------------------------------------
if [ -s "$FAILURES_FILE" ]; then
    while IFS= read -r FAILURE; do
        [ -z "$FAILURE" ] && continue

        RUN_ID=$(printf '%s' "$FAILURE" | jq -r '.run_id')
        JOB_NAME=$(printf '%s' "$FAILURE" | jq -r '.job_name')
        JOB_ID=$(printf '%s' "$FAILURE" | jq -r '.job_id')
        JOB_URL=$(printf '%s' "$FAILURE" | jq -r '.job_url')
        STEP_NAME=$(printf '%s' "$FAILURE" | jq -r '.step_name')
        SIGNATURE="${JOB_NAME} / ${STEP_NAME}"

        echo "[Phase 1] Processing failure: ${SIGNATURE} (run ${RUN_ID})" >&2

        if [ -n "${QUEUED_IN_THIS_RUN[$SIGNATURE]:-}" ]; then
            echo "  -> already queued earlier in this run, skipping (no new comment)" >&2
            continue
        fi

        # A comment for this signature already existing at all - analyzed
        # or not - is reason enough to skip posting a new one. Whether it
        # still needs analysis is Phase 2's concern, not this check's.
        DETECTION_MARKER="<!-- rocm-nightly-failure:${SIGNATURE} -->"
        DEBUG_MARKER="<!-- rocm-nightly-claude-debug:${SIGNATURE} -->"
        ALREADY_HAS_COMMENT=$(printf '%s' "$EXISTING_COMMENTS" \
            | jq --arg d "$DETECTION_MARKER" --arg g "$DEBUG_MARKER" \
                '[.[] | select((.body | contains($d)) or (.body | contains($g)))] | length')

        if [ "${ALREADY_HAS_COMMENT:-0}" -gt 0 ]; then
            echo "  -> a comment for this signature already exists (analyzed or not), skipping (no new comment)" >&2
            continue
        fi

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

# ---------------------------------------------------------------------------
# Phase 2 - catch up any existing comment that never got analyzed, for
# whatever reason (cancelled monitoring run, PR-validation-skipped debug
# job, expired/aborted earlier attempt, ...). Runs every time, independent
# of Phase 1 finding anything new this run.
# ---------------------------------------------------------------------------
echo "[Phase 2] Scanning the issue for previously-posted, never-analyzed signatures" >&2

mapfile -t COMMENT_IDS < <(printf '%s' "$EXISTING_COMMENTS" | jq -r '.[].id')

for COMMENT_ID in "${COMMENT_IDS[@]}"; do
    BODY=$(printf '%s' "$EXISTING_COMMENTS" | jq -r --argjson id "$COMMENT_ID" '.[] | select(.id == $id) | .body')

    SIGNATURE=$(printf '%s' "$BODY" | grep -oP '(?<=<!-- rocm-nightly-failure:)[^>]*(?= -->)' | head -1)
    if [ -z "$SIGNATURE" ]; then
        continue
    fi

    if [ -n "${QUEUED_IN_THIS_RUN[$SIGNATURE]:-}" ]; then
        continue
    fi

    DEBUG_MARKER="<!-- rocm-nightly-claude-debug:${SIGNATURE} -->"
    if printf '%s' "$BODY" | grep -qF "$DEBUG_MARKER"; then
        continue
    fi

    echo "  -> ${SIGNATURE} (comment ${COMMENT_ID}) was never analyzed" >&2

    FAILURE_URL=$(printf '%s' "$BODY" | grep -oP '(?<=- Failing job: )\S+' | head -1)
    if [ -z "$FAILURE_URL" ]; then
        echo "    -> no 'Failing job:' line found, skipping" >&2
        continue
    fi

    JOB_ID=$(printf '%s' "$FAILURE_URL" | grep -oP '(?<=/job/)[0-9]+')
    RUN_ID=$(printf '%s' "$FAILURE_URL" | grep -oP '(?<=/runs/)[0-9]+')
    if [ -z "$JOB_ID" ]; then
        echo "    -> could not parse a job ID out of ${FAILURE_URL}, skipping" >&2
        continue
    fi

    # The job's log may have aged out of GitHub's retention window by now -
    # don't fail the whole scan over one expired log, just skip this one
    # with a diagnostic; it'll be tried again on the next run. A command
    # used directly as an `if` condition is exempt from `set -e`, so this
    # doesn't need any extra guarding.
    if CHECKOUT_INFO=$("${SCRIPT_DIR}/get_checkout_source.sh" "$CI_REPO" "$JOB_ID" 2>/dev/null); then
        HEAD_REPOSITORY=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '1p')
        HEAD_SHA=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '2p')
    else
        echo "    -> could not resolve the original checkout (log likely expired), skipping" >&2
        continue
    fi

    COMMENT_URL=$(printf '%s' "$EXISTING_COMMENTS" | jq -r --argjson id "$COMMENT_ID" '.[] | select(.id == $id) | .html_url')
    # Split at the LAST " / ", not the first: JOB_NAME itself routinely
    # contains " / " (reusable-workflow-called jobs are named
    # "caller / callee", e.g. "jax-upstream / JAX Upstream Linux x86 ..."),
    # while STEP_NAME never does. Concatenating job_name + " / " +
    # step_name later reproduces the original SIGNATURE either way (the
    # marker text is split-point-invariant), but splitting at the wrong
    # occurrence would still feed Claude a truncated, misleading job_name.
    JOB_NAME="${SIGNATURE% / *}"
    STEP_NAME="${SIGNATURE##* / }"

    echo "    -> queued for debug (catch-up)" >&2
    QUEUED_IN_THIS_RUN[$SIGNATURE]=1
    TARGET=$(jq -nc \
        --arg issue_number "$ISSUE_NUMBER" \
        --arg repository "$HEAD_REPOSITORY" \
        --arg head_sha "$HEAD_SHA" \
        --arg run_id "$RUN_ID" \
        --arg job_id "$JOB_ID" \
        --arg job_name "$JOB_NAME" \
        --arg step_name "$STEP_NAME" \
        --arg failure_url "$FAILURE_URL" \
        --arg detection_comment_url "$COMMENT_URL" \
        '{issue_number: $issue_number, repository: $repository, head_sha: $head_sha,
          run_id: $run_id, job_id: $job_id, job_name: $job_name,
          step_name: $step_name, failure_url: $failure_url,
          detection_comment_url: $detection_comment_url}')
    TARGETS=$(printf '%s' "$TARGETS" | jq -c --argjson t "$TARGET" '. + [$t]')
done

printf '%s\n' "$TARGETS"
