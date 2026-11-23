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
# Each distinct failure signature (job_name/step_name) gets its own
# dedicated GitHub issue, found-or-created here - never a comment on a
# shared tracking issue. The issue BODY is set once, at creation, to the
# static detection text (job/step/links) and never touched again -
# everything debug-nightly-failure adds later (triage, final report) is a
# plain new comment, not an edit to that body.
#
# Two phases, every run:
#
# Phase 1 - report new failures. For every failure in a JSON Lines file (as
# produced by collect_run_failures.sh), find an existing OPEN issue labeled
# rocm-infra-error whose title matches this failure's signature, or create
# one if none exists. If a matching issue already has a comment carrying a
# `rocm-nightly-claude-debug:` marker, it's already been analyzed - skip it
# (no new issue, no re-queue). Otherwise queue it (whether brand new or a
# pre-existing, never-analyzed issue).
#
# Phase 2 - catch up anything still unanalyzed. Separately, and
# unconditionally (even on a run with no new failures at all), scan every
# OPEN `rocm-infra-error` issue and queue a debug run for any with no
# comment carrying a `rocm-nightly-claude-debug:` marker yet - i.e. it was
# created (by this run's Phase 1, an earlier run's Phase 1, or an earlier
# run's Phase 2 that itself never got dispatched/completed) but never
# actually got analyzed. This is what makes an issue self-heal on the next
# run after something stopped its first attempt (a cancelled monitoring
# run, a PR-validation-skipped debug job, a transient infra error) instead
# of sitting unanalyzed forever.
#
# Both phases queue into the same output array, so both feed the exact same
# `strategy.matrix.include` on the caller's "debug" job - a Phase 2 entry
# behaves identically to a fresh Phase 1 one as far as the debug skill is
# concerned, just with its checkout info recovered from the existing
# issue's own "Failing job:" line instead of from collect_run_failures.sh.
#
# Usage: collect_claude_debug_targets.sh <failures_jsonl_file> <workflow_run_url>
#
# Output:
#   A single compact JSON array on stdout (diagnostics on stderr), each
#   element: {issue_number, repository, head_sha, run_id, job_id, job_name,
#   step_name, failure_url}. "[]" if there is nothing to debug.
#
# Exit codes:
#   0: Success
#   1: Error

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <failures_jsonl_file> <workflow_run_url>" >&2
    exit 1
fi

FAILURES_FILE="$1"
WORKFLOW_URL="$2"

# Repository hosting the monitored runs and the per-failure issues.
CI_REPO="ROCm/xla"
LABEL="rocm-infra-error"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGETS="[]"

# `gh issue edit --add-label` fails if the label doesn't exist as a repo
# label object yet - ensure it exists first, every time, since there's
# otherwise no signal that it's missing.
gh label create "$LABEL" --repo "$CI_REPO" --force \
    --color b60205 --description "ROCm CI nightly infra/test failure tracking issue" \
    >/dev/null 2>&1 \
    || echo "Warning: could not create/update label $LABEL" >&2

# Team members to assign to each new per-failure issue.
# Adjust this list as needed.
TEAM_MEMBERS=(
    alekstheod
    i-chaochen
    hsharsha
    charleshofer
    leo-automation
    pelumi1163
    nurmukhametov
)

# Fetched once up front and reused by both phases - cheaper than one API
# call per failure/issue.
OPEN_ISSUES=$(gh issue list --repo "$CI_REPO" --state open --label "$LABEL" --limit 1000 \
    --json number,title,body)

# Issue numbers queued in this run, by either phase - two different
# monitored runs in the same window can share a job/step signature (Phase
# 1), and Phase 2 must not re-queue something Phase 1 (or an earlier
# Phase-2 entry this same run) already did.
declare -A QUEUED_IN_THIS_RUN

# Signature -> issue number, for a signature already resolved earlier in
# Phase 1 of this same run. OPEN_ISSUES above is a one-time snapshot taken
# before any issue this run creates exists, so two failures sharing a
# signature within the same failures.json would otherwise both miss it and
# each create their own duplicate issue - this memoizes the first
# resolution (found-existing or newly-created) so every later occurrence
# of the same signature this run reuses it instead.
declare -A SIGNATURE_TO_ISSUE

issue_title() { echo "Nightly CI Failure: $1"; }

# Prints "true"/"false" on stdout: whether any comment on the given issue
# contains the given marker string. The debug marker lives in a comment
# (debug-nightly-failure's final report), never in the issue body, so this
# is the only place "already analyzed" can be determined from.
issue_has_comment_marker() {
    local ISSUE_NUMBER="$1" MARKER="$2"
    gh api --paginate "/repos/${CI_REPO}/issues/${ISSUE_NUMBER}/comments" \
        | jq --arg m "$MARKER" 'any(.[]; .body | contains($m))'
}

# Creates a brand-new per-signature issue and prints its number on stdout.
# Label and each assignee are added separately, after creation, so one
# invalid assignee (lacking repo access) can't make `gh issue create` fail
# atomically and silently drop the label too. `gh issue edit` prints the
# issue URL to stdout on success - redirect it, or it contaminates this
# function's own stdout, which the caller captures wholesale as the issue
# number.
create_issue() {
    local SIGNATURE="$1" JOB_NAME="$2" STEP_NAME="$3" JOB_URL="$4"
    local TITLE BODY NEW_ISSUE NEW_ISSUE_NUMBER

    TITLE=$(issue_title "$SIGNATURE")
    BODY="<!-- rocm-nightly-failure:${SIGNATURE} -->
## Nightly CI Failure Detected

**Job:** ${JOB_NAME}
**Step:** ${STEP_NAME}

- Failing job: ${JOB_URL}
- Detected by monitoring run: ${WORKFLOW_URL}
- Detected: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

cc @ROCm/ai-fw-openxla"

    NEW_ISSUE=$(gh issue create --repo "$CI_REPO" --title "$TITLE" --body "$BODY")
    echo "  -> issue created: ${NEW_ISSUE}" >&2
    NEW_ISSUE_NUMBER=$(echo "$NEW_ISSUE" | grep -oE '[0-9]+$')

    gh issue edit "$NEW_ISSUE_NUMBER" --repo "$CI_REPO" --add-label "$LABEL" >/dev/null 2>&1 \
        || echo "Warning: could not add label $LABEL" >&2
    for member in "${TEAM_MEMBERS[@]}"; do
        gh issue edit "$NEW_ISSUE_NUMBER" --repo "$CI_REPO" --add-assignee "$member" >/dev/null 2>&1 \
            || echo "Warning: could not assign $member (no repo access?)" >&2
    done

    echo "$NEW_ISSUE_NUMBER"
}

# ---------------------------------------------------------------------------
# Phase 1 - report failures, finding-or-creating one issue per signature.
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
        TITLE=$(issue_title "$SIGNATURE")

        echo "[Phase 1] Processing failure: ${SIGNATURE} (run ${RUN_ID})" >&2

        if [ -n "${SIGNATURE_TO_ISSUE[$SIGNATURE]:-}" ]; then
            ISSUE_NUMBER="${SIGNATURE_TO_ISSUE[$SIGNATURE]}"
            if [ -n "${QUEUED_IN_THIS_RUN[$ISSUE_NUMBER]:-}" ]; then
                echo "  -> already queued earlier in this run (issue #${ISSUE_NUMBER}), skipping" >&2
            else
                echo "  -> already analyzed (issue #${ISSUE_NUMBER}), skipping" >&2
            fi
            continue
        fi

        ISSUE_NUMBER=$(printf '%s' "$OPEN_ISSUES" \
            | jq -r --arg title "$TITLE" 'map(select(.title == $title)) | .[0].number // empty')

        if [ -n "$ISSUE_NUMBER" ]; then
            DEBUG_MARKER="<!-- rocm-nightly-claude-debug:${SIGNATURE} -->"
            if [ "$(issue_has_comment_marker "$ISSUE_NUMBER" "$DEBUG_MARKER")" = "true" ]; then
                echo "  -> issue #${ISSUE_NUMBER} already analyzed, skipping" >&2
                SIGNATURE_TO_ISSUE[$SIGNATURE]="$ISSUE_NUMBER"
                continue
            fi

            echo "  -> reusing existing, never-analyzed issue #${ISSUE_NUMBER}" >&2
        else
            echo "  -> creating new issue for this signature" >&2
            ISSUE_NUMBER=$(create_issue "$SIGNATURE" "$JOB_NAME" "$STEP_NAME" "$JOB_URL")
        fi
        SIGNATURE_TO_ISSUE[$SIGNATURE]="$ISSUE_NUMBER"

        # What this specific job actually checked out - NOT the run's own
        # head_sha/head_repository, which reflect ROCm/xla's commit at
        # schedule time, not the openxla/xla@main commit the job resolved
        # and tested against. See get_checkout_source.sh for why.
        CHECKOUT_INFO=$("${SCRIPT_DIR}/get_checkout_source.sh" "$CI_REPO" "$JOB_ID")
        HEAD_REPOSITORY=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '1p')
        HEAD_SHA=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '2p')

        echo "  -> queued for debug" >&2
        QUEUED_IN_THIS_RUN[$ISSUE_NUMBER]=1
        TARGET=$(jq -nc \
            --arg issue_number "$ISSUE_NUMBER" \
            --arg repository "$HEAD_REPOSITORY" \
            --arg head_sha "$HEAD_SHA" \
            --arg run_id "$RUN_ID" \
            --arg job_id "$JOB_ID" \
            --arg job_name "$JOB_NAME" \
            --arg step_name "$STEP_NAME" \
            --arg failure_url "$JOB_URL" \
            '{issue_number: $issue_number, repository: $repository, head_sha: $head_sha,
              run_id: $run_id, job_id: $job_id, job_name: $job_name,
              step_name: $step_name, failure_url: $failure_url}')
        TARGETS=$(printf '%s' "$TARGETS" | jq -c --argjson t "$TARGET" '. + [$t]')
    done < "$FAILURES_FILE"
fi

# ---------------------------------------------------------------------------
# Phase 2 - catch up any open issue that never got analyzed, for whatever
# reason (cancelled monitoring run, PR-validation-skipped debug job,
# expired/aborted earlier attempt, ...). Runs every time, independent of
# Phase 1 finding anything new this run.
# ---------------------------------------------------------------------------
echo "[Phase 2] Scanning open issues for never-analyzed signatures" >&2

mapfile -t ISSUE_NUMBERS < <(printf '%s' "$OPEN_ISSUES" | jq -r '.[].number')

for ISSUE_NUMBER in "${ISSUE_NUMBERS[@]}"; do
    if [ -n "${QUEUED_IN_THIS_RUN[$ISSUE_NUMBER]:-}" ]; then
        continue
    fi

    TITLE=$(printf '%s' "$OPEN_ISSUES" | jq -r --argjson n "$ISSUE_NUMBER" '.[] | select(.number == $n) | .title')
    BODY=$(printf '%s' "$OPEN_ISSUES" | jq -r --argjson n "$ISSUE_NUMBER" '.[] | select(.number == $n) | .body')

    # Issues created before this script's prefix convention existed (or
    # anything else carrying the label but not this title shape) have no
    # recoverable signature - skip rather than guess.
    SIGNATURE="${TITLE#Nightly CI Failure: }"
    if [ "$SIGNATURE" = "$TITLE" ]; then
        continue
    fi

    DEBUG_MARKER="<!-- rocm-nightly-claude-debug:${SIGNATURE} -->"
    if [ "$(issue_has_comment_marker "$ISSUE_NUMBER" "$DEBUG_MARKER")" = "true" ]; then
        continue
    fi

    echo "  -> issue #${ISSUE_NUMBER} (${SIGNATURE}) was never analyzed" >&2

    # `|| true` matters, not `| head -1` alone: under `set -e -o pipefail`,
    # the pipeline's exit status is the *rightmost command that exited
    # non-zero* - if grep finds nothing (exit 1) and head still exits 0,
    # pipefail still reports 1 for the whole pipe, which `set -e` then
    # treats as this assignment failing and aborts the whole script, with
    # no further diagnostic.
    FAILURE_URL=$(printf '%s' "$BODY" | grep -oP '(?<=- Failing job: )\S+' | head -1 || true)
    if [ -z "$FAILURE_URL" ]; then
        echo "    -> no 'Failing job:' line found, skipping" >&2
        continue
    fi

    JOB_ID=$(printf '%s' "$FAILURE_URL" | grep -oP '(?<=/job/)[0-9]+' | head -1 || true)
    RUN_ID=$(printf '%s' "$FAILURE_URL" | grep -oP '(?<=/runs/)[0-9]+' | head -1 || true)
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
    QUEUED_IN_THIS_RUN[$ISSUE_NUMBER]=1
    TARGET=$(jq -nc \
        --arg issue_number "$ISSUE_NUMBER" \
        --arg repository "$HEAD_REPOSITORY" \
        --arg head_sha "$HEAD_SHA" \
        --arg run_id "$RUN_ID" \
        --arg job_id "$JOB_ID" \
        --arg job_name "$JOB_NAME" \
        --arg step_name "$STEP_NAME" \
        --arg failure_url "$FAILURE_URL" \
        '{issue_number: $issue_number, repository: $repository, head_sha: $head_sha,
          run_id: $run_id, job_id: $job_id, job_name: $job_name,
          step_name: $step_name, failure_url: $failure_url}')
    TARGETS=$(printf '%s' "$TARGETS" | jq -c --argjson t "$TARGET" '. + [$t]')
done

printf '%s\n' "$TARGETS"
