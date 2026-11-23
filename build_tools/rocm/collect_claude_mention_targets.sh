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
# Find comments a human has edited to add an `@claude <instructions>` line,
# asking Claude to act on a failure it already reported - verify an
# already-proposed fix, dig further, try a different approach, whatever the
# instructions say - and queue each one that qualifies for the
# respond-to-mention skill.
#
# Two modes:
#   - Full scan (`collect_claude_mention_targets.sh <issue_number>`): checks
#     every comment on the tracking issue. Used by the scheduled run, which
#     has no way to know which comment (if any) changed since last time -
#     an `@claude` line can land on a comment analyzed days ago.
#   - Single comment (`collect_claude_mention_targets.sh <issue_number>
#     --comment <comment_id> <comment_url> <comment_body>`): checks only
#     the one comment an `issue_comment: edited` event already told us
#     changed - no need to page through the rest of the issue. Used by the
#     event-triggered run for an immediate response; the full scan above
#     remains the fallback that catches anything the event-triggered path
#     missed (a run that failed to dispatch, GitHub Actions degraded, etc).
#
# A comment qualifies if, and only if:
#   - it carries a `rocm-nightly-failure:` or `rocm-nightly-claude-debug:`
#     marker (so a signature/job/step can be recovered from it), AND
#   - its body contains a line starting with `@claude` (case-insensitive) -
#     editing someone else's comment on GitHub requires write/triage access
#     to the repo, so this is naturally restricted to trusted
#     collaborators, unlike a reaction, AND
#   - it does NOT already contain a `rocm-nightly-claude-mention:` marker
#     for this same signature *and* this exact instruction text - once a
#     given instruction has been attempted, it is never repeated, but
#     editing the comment with a *new* instruction (different text after
#     `@claude`) queues a fresh run, since that's a new, distinct ask.
#
# Output:
#   A single compact JSON array on stdout (diagnostics on stderr), each
#   element: {issue_number, comment_id, comment_url, job_name, step_name,
#   repository, head_sha, failure_url, instructions, instructions_hash}.
#   "[]" if there is nothing to act on.
#
# Exit codes:
#   0: Success
#   1: Error

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <issue_number> [--comment <comment_id> <comment_url> <comment_body>]" >&2
    exit 1
fi

ISSUE_NUMBER="$1"
shift

# Repository hosting the monitored runs and the tracking issue.
CI_REPO="ROCm/xla"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Checks one comment against every qualifying rule above and, if it
# qualifies, prints its JSON target object on stdout (nothing otherwise).
# Shared by both modes so the qualifying logic only ever lives in one
# place.
process_comment() {
    local COMMENT_ID="$1" BODY="$2" COMMENT_URL="$3"

    # The signature is whatever job/step marker is present - a comment
    # that debug-nightly-failure hasn't finished analyzing yet only has
    # the `rocm-nightly-failure:` detection marker; one it has finished
    # also carries `rocm-nightly-claude-debug:` further down (now inside
    # Step 7's fold). Either anchors the same signature text.
    # `|| true` matters, not `| head -1` alone: under `set -e -o pipefail`,
    # the pipeline's exit status is the *rightmost command that exited
    # non-zero* - if grep finds nothing (exit 1) and head still exits 0,
    # pipefail still reports 1 for the whole pipe, which `set -e` then
    # treats as this assignment failing and aborts the whole script, with
    # no further diagnostic. This is hit by any plain comment with neither
    # marker at all (a human reply, not a detection comment) - `head -1`
    # alone never actually protected against that.
    local SIGNATURE
    SIGNATURE=$(printf '%s' "$BODY" | grep -oP '(?<=<!-- rocm-nightly-failure:)[^>]*(?= -->)' | head -1 || true)
    if [ -z "$SIGNATURE" ]; then
        SIGNATURE=$(printf '%s' "$BODY" | grep -oP '(?<=<!-- rocm-nightly-claude-debug:)[^>]*(?= -->)' | head -1 || true)
    fi
    if [ -z "$SIGNATURE" ]; then
        return
    fi

    echo "Checking comment ${COMMENT_ID} (${SIGNATURE})" >&2

    # Capture the `@claude` line itself plus any immediately following
    # non-blank lines, so a short freeform paragraph works, e.g.:
    #   @claude verify the solution, try to set a breakpoint there and
    #   adjust this code. Inspect if it works.
    # Stops at the first blank line, HTML comment, or <details> tag, so it
    # never swallows a divider or an existing fold below it.
    local INSTRUCTIONS
    INSTRUCTIONS=$(printf '%s' "$BODY" | awk '
        /^[[:space:]]*@[Cc][Ll][Aa][Uu][Dd][Ee]([^a-zA-Z]|$)/ { capture=1 }
        capture {
            if ($0 ~ /^[[:space:]]*$/ || $0 ~ /<!--/ || $0 ~ /<details>/) exit
            print
        }
    ' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')
    if [ -z "$INSTRUCTIONS" ]; then
        echo "  -> no @claude line in the comment body, skipping" >&2
        return
    fi

    local INSTRUCTIONS_HASH MENTION_MARKER
    INSTRUCTIONS_HASH=$(printf '%s' "$INSTRUCTIONS" | sha256sum | cut -c1-12)
    MENTION_MARKER="<!-- rocm-nightly-claude-mention:${SIGNATURE}:${INSTRUCTIONS_HASH} -->"
    if printf '%s' "$BODY" | grep -qF "$MENTION_MARKER"; then
        echo "  -> this exact instruction was already acted on, skipping" >&2
        return
    fi

    # Split at the LAST " / ", not the first: JOB_NAME itself routinely
    # contains " / " (reusable-workflow-called jobs are named
    # "caller / callee", e.g. "jax-upstream / JAX Upstream Linux x86 ..."),
    # while STEP_NAME never does.
    local JOB_NAME="${SIGNATURE% / *}" STEP_NAME="${SIGNATURE##* / }"

    local FAILURE_URL
    FAILURE_URL=$(printf '%s' "$BODY" | grep -oP '(?<=- Failing job: )\S+' | head -1 || true)
    if [ -z "$FAILURE_URL" ]; then
        echo "  -> @claude'd but no 'Failing job:' line found, skipping" >&2
        return
    fi

    # .../actions/runs/<run_id>/job/<job_id>
    # See the SIGNATURE extraction above for why `|| true` (not just
    # `head -1`) is required here too.
    local JOB_ID RUN_ID
    JOB_ID=$(printf '%s' "$FAILURE_URL" | grep -oP '(?<=/job/)[0-9]+' | head -1 || true)
    RUN_ID=$(printf '%s' "$FAILURE_URL" | grep -oP '(?<=/runs/)[0-9]+' | head -1 || true)
    if [ -z "$JOB_ID" ]; then
        echo "  -> could not parse a job ID out of ${FAILURE_URL}, skipping" >&2
        return
    fi

    # The job's log may have aged out of GitHub's retention window by the
    # time someone adds `@claude` - don't fail the whole scan over one
    # expired log, just skip this one with a diagnostic. A command used
    # directly as an `if` condition is exempt from `set -e`, so this
    # doesn't need any extra guarding.
    local HEAD_REPOSITORY HEAD_SHA CHECKOUT_INFO
    if CHECKOUT_INFO=$("${SCRIPT_DIR}/get_checkout_source.sh" "$CI_REPO" "$JOB_ID" 2>/dev/null); then
        HEAD_REPOSITORY=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '1p')
        HEAD_SHA=$(printf '%s\n' "$CHECKOUT_INFO" | sed -n '2p')
    else
        echo "  -> could not resolve the original checkout (log likely expired)" >&2
        return
    fi

    echo "  -> queued: ${INSTRUCTIONS}" >&2
    jq -nc \
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
        --arg instructions "$INSTRUCTIONS" \
        --arg instructions_hash "$INSTRUCTIONS_HASH" \
        '{issue_number: $issue_number, comment_id: $comment_id, comment_url: $comment_url,
          job_name: $job_name, step_name: $step_name, repository: $repository,
          head_sha: $head_sha, failure_url: $failure_url, run_id: $run_id,
          job_id: $job_id, instructions: $instructions,
          instructions_hash: $instructions_hash}'
}

TARGETS="[]"

if [ "${1:-}" = "--comment" ]; then
    if [ $# -ne 4 ]; then
        echo "Usage: $0 <issue_number> --comment <comment_id> <comment_url> <comment_body>" >&2
        exit 1
    fi
    TARGET=$(process_comment "$2" "$4" "$3")
    if [ -n "$TARGET" ]; then
        TARGETS=$(printf '%s' "$TARGETS" | jq -c --argjson t "$TARGET" '. + [$t]')
    fi
else
    COMMENTS=$(gh api --paginate "/repos/${CI_REPO}/issues/${ISSUE_NUMBER}/comments")

    # mapfile populates the array fully before the loop starts, so an
    # early `continue`/error inside the loop never closes a pipe a
    # producer is still writing to (the SIGPIPE-under-pipefail gotcha
    # documented in get_checkout_source.sh).
    mapfile -t COMMENT_IDS < <(printf '%s' "$COMMENTS" | jq -r '.[].id')

    for COMMENT_ID in "${COMMENT_IDS[@]}"; do
        BODY=$(printf '%s' "$COMMENTS" | jq -r --argjson id "$COMMENT_ID" '.[] | select(.id == $id) | .body')
        COMMENT_URL=$(printf '%s' "$COMMENTS" | jq -r --argjson id "$COMMENT_ID" '.[] | select(.id == $id) | .html_url')
        TARGET=$(process_comment "$COMMENT_ID" "$BODY" "$COMMENT_URL")
        if [ -n "$TARGET" ]; then
            TARGETS=$(printf '%s' "$TARGETS" | jq -c --argjson t "$TARGET" '. + [$t]')
        fi
    done
fi

printf '%s\n' "$TARGETS"
