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
# Scan every comment on the tracking issue for ones a human has edited to
# add a `/verify` command, asking Claude to actually try out a proposed fix
# it already posted - and queue each one that qualifies for the
# verify-proposed-fix skill. Unlike collect_claude_debug_targets.sh, this
# doesn't look at "what failed in this monitoring window" - a `/verify` can
# land on a comment analyzed days ago, so every run rescans the whole issue.
#
# A comment qualifies if, and only if:
#   - it carries a `rocm-nightly-failure:` or `rocm-nightly-claude-debug:`
#     marker (so a signature/job/step can be recovered from it), AND
#   - its body contains a line consisting of just `/verify` (whitespace
#     aside) - editing someone else's comment on GitHub requires write/
#     triage access to the repo, so this is naturally restricted to
#     trusted collaborators, unlike a reaction, AND
#   - it does NOT already contain a `rocm-nightly-claude-verify:` marker
#     for that same signature - once attempted, never repeated, regardless
#     of outcome (same "no re-runs" rule debug targets follow).
#
# Usage: collect_claude_verify_targets.sh <issue_number>
#
# Output:
#   A single compact JSON array on stdout (diagnostics on stderr), each
#   element: {issue_number, comment_id, comment_url, job_name, step_name,
#   repository, head_sha, failure_url}. "[]" if there is nothing to verify.
#
# Exit codes:
#   0: Success
#   1: Error

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <issue_number>" >&2
    exit 1
fi

ISSUE_NUMBER="$1"

# Repository hosting the monitored runs and the tracking issue.
CI_REPO="ROCm/xla"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGETS="[]"

COMMENTS=$(gh api --paginate "/repos/${CI_REPO}/issues/${ISSUE_NUMBER}/comments")

# mapfile populates the array fully before the loop starts, so an early
# `continue`/error inside the loop never closes a pipe a producer is still
# writing to (the SIGPIPE-under-pipefail gotcha documented in
# get_checkout_source.sh).
mapfile -t COMMENT_IDS < <(printf '%s' "$COMMENTS" | jq -r '.[].id')

for COMMENT_ID in "${COMMENT_IDS[@]}"; do
    BODY=$(printf '%s' "$COMMENTS" | jq -r --argjson id "$COMMENT_ID" '.[] | select(.id == $id) | .body')
    COMMENT_URL=$(printf '%s' "$COMMENTS" | jq -r --argjson id "$COMMENT_ID" '.[] | select(.id == $id) | .html_url')

    # The signature is whatever job/step marker is present - a comment
    # that debug-nightly-failure hasn't finished analyzing yet only has
    # the `rocm-nightly-failure:` detection marker; one it has finished
    # also carries `rocm-nightly-claude-debug:` further down (now inside
    # Step 7's fold). Either anchors the same signature text.
    SIGNATURE=$(printf '%s' "$BODY" | grep -oP '(?<=<!-- rocm-nightly-failure:)[^>]*(?= -->)' | head -1)
    if [ -z "$SIGNATURE" ]; then
        SIGNATURE=$(printf '%s' "$BODY" | grep -oP '(?<=<!-- rocm-nightly-claude-debug:)[^>]*(?= -->)' | head -1)
    fi
    if [ -z "$SIGNATURE" ]; then
        continue
    fi

    echo "Checking comment ${COMMENT_ID} (${SIGNATURE})" >&2

    VERIFY_MARKER="<!-- rocm-nightly-claude-verify:${SIGNATURE} -->"
    if printf '%s' "$BODY" | grep -qF "$VERIFY_MARKER"; then
        echo "  -> already verified (or verification already attempted), skipping" >&2
        continue
    fi

    if ! printf '%s' "$BODY" | grep -qE '^[[:space:]]*/verify[[:space:]]*$'; then
        echo "  -> no /verify command in the comment body, skipping" >&2
        continue
    fi

    # Split at the LAST " / ", not the first: JOB_NAME itself routinely
    # contains " / " (reusable-workflow-called jobs are named
    # "caller / callee", e.g. "jax-upstream / JAX Upstream Linux x86 ..."),
    # while STEP_NAME never does.
    JOB_NAME="${SIGNATURE% / *}"
    STEP_NAME="${SIGNATURE##* / }"

    FAILURE_URL=$(printf '%s' "$BODY" | grep -oP '(?<=- Failing job: )\S+' | head -1)
    if [ -z "$FAILURE_URL" ]; then
        echo "  -> /verify'd but no 'Failing job:' line found, skipping" >&2
        continue
    fi

    # .../actions/runs/<run_id>/job/<job_id>
    JOB_ID=$(printf '%s' "$FAILURE_URL" | grep -oP '(?<=/job/)[0-9]+')
    RUN_ID=$(printf '%s' "$FAILURE_URL" | grep -oP '(?<=/runs/)[0-9]+')
    if [ -z "$JOB_ID" ]; then
        echo "  -> could not parse a job ID out of ${FAILURE_URL}, skipping" >&2
        continue
    fi

    # The job's log may have aged out of GitHub's retention window by the
    # time someone adds `/verify` - don't fail the whole scan over one expired
    # log, just skip this one with a diagnostic. A command used directly as
    # an `if` condition is exempt from `set -e`, so this doesn't need any
    # extra guarding.
    if CHECKOUT_INFO=$("${SCRIPT_DIR}/get_checkout_source.sh" "$CI_REPO" "$JOB_ID" 2>/dev/null); then
        HEAD_REPOSITORY=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '1p')
        HEAD_SHA=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '2p')
    else
        echo "  -> could not resolve the original checkout (log likely expired)" >&2
        continue
    fi

    echo "  -> queued for verification" >&2
    TARGET=$(jq -nc \
        --arg issue_number "$ISSUE_NUMBER" \
        --arg comment_id "$COMMENT_ID" \
        --arg comment_url "$COMMENT_URL" \
        --arg job_name "$JOB_NAME" \
        --arg step_name "$STEP_NAME" \
        --arg repository "$HEAD_REPOSITORY" \
        --arg head_sha "$HEAD_SHA" \
        --arg failure_url "$FAILURE_URL" \
        --arg run_id "$RUN_ID" \
        --arg job_id "$JOB_ID" \
        '{issue_number: $issue_number, comment_id: $comment_id, comment_url: $comment_url,
          job_name: $job_name, step_name: $step_name, repository: $repository,
          head_sha: $head_sha, failure_url: $failure_url, run_id: $run_id,
          job_id: $job_id}')
    TARGETS=$(printf '%s' "$TARGETS" | jq -c --argjson t "$TARGET" '. + [$t]')
done

printf '%s\n' "$TARGETS"
