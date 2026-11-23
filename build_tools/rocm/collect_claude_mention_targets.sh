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
# Find new comments starting with `@claude <instructions>` on ANY issue in
# CI_REPO - not just the per-failure issues the nightly monitor creates -
# and queue each one that qualifies. Authorization (who is allowed to
# trigger this at all) is the caller's job, not this script's: see
# rocm_infra_agent.yml's author_association check in the workflow `if:`.
#
# Two modes:
#   - Single comment (`collect_claude_mention_targets.sh --comment
#     <issue_number> <comment_id> <comment_body>`): checks only the one
#     comment an `issue_comment: created` event already told us about - no
#     need to page through anything else. Used by the event-triggered run
#     for an immediate response.
#   - Full scan (`collect_claude_mention_targets.sh`): checks every open
#     issue's comments. Used by the scheduled run, which is the fallback
#     that catches anything the event-triggered path missed (a run that
#     failed to dispatch, GitHub Actions degraded, etc).
#
# A comment qualifies if, and only if:
#   - its body starts with `@claude` (case-insensitive), AND
#   - no comment already on the issue contains the matching mention marker
#     for this exact instruction text (see below) - once a given
#     instruction has been attempted, it is never repeated, but a new
#     comment with different text after `@claude` queues a fresh run,
#     since that's a new, distinct ask.
#
# Each qualifying comment is classified by the issue it's on:
#   - "ci_failure": the issue carries the `rocm-infra-error` label, its
#     title matches the `Nightly CI Failure: <job> / <step>` shape
#     collect_claude_debug_targets.sh creates issues with, and a
#     `- Failing job: <url>` line is present in its body - i.e. it's one
#     of the nightly monitor's own per-failure issues. These get the full
#     treatment: a specific commit to check out, a job/step signature, and
#     the respond-to-mention skill's reproduce/apply/revert protocol.
#     Mention marker: `<!-- rocm-nightly-claude-mention:JOB_NAME /
#     STEP_NAME:INSTRUCTIONS_HASH -->`.
#   - "general": anything else - an ordinary issue with no CI-failure
#     context to reproduce. These get the lighter respond-to-issue skill:
#     no checkout of a specific commit, no build/test, just the repo's own
#     default branch to read code from and a comment in reply. Mention
#     marker: `<!-- claude-mention:INSTRUCTIONS_HASH -->` (no job/step to
#     fold in, since the marker only needs to be unique within this one
#     issue's own comments).
#
# Output:
#   A single compact JSON array on stdout (diagnostics on stderr). Each
#   element has `type: "ci_failure"` and {issue_number, comment_id,
#   job_name, step_name, repository, head_sha, failure_url, run_id,
#   job_id, instructions, instructions_hash}, or `type: "general"` and
#   {issue_number, comment_id, repository, head_sha, instructions,
#   instructions_hash} (repository/head_sha fixed at "ROCm/xla"/"main" -
#   there's no failing job to derive them from - included only so a patch
#   this target's skill run produces can still be applied the same way).
#   "[]" if there is nothing to act on.
#
# Exit codes:
#   0: Success
#   1: Error

set -euo pipefail

# Repository hosting the monitored runs and the per-failure issues.
CI_REPO="ROCm/xla"
LABEL="rocm-infra-error"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Checks one comment against every qualifying rule above and, if it
# qualifies, prints its JSON target object on stdout (nothing otherwise).
# Shared by both modes so the qualifying logic only ever lives in one
# place. ISSUE_TITLE/ISSUE_BODY/ISSUE_HAS_LABEL describe the issue itself
# (the signature and "Failing job:" line, if any, live there, not in any
# comment). ALL_COMMENTS_TEXT is every existing comment's body on the
# issue, newline-joined, searched only for the mention-marker dedup check
# below.
process_comment() {
    local ISSUE_NUMBER="$1" COMMENT_ID="$2" COMMENT_BODY="$3" ISSUE_TITLE="$4" ISSUE_BODY="$5" ISSUE_HAS_LABEL="$6" ALL_COMMENTS_TEXT="$7"

    echo "Checking comment ${COMMENT_ID} on issue #${ISSUE_NUMBER}" >&2

    # Classify: does this look like one of the nightly monitor's own
    # per-failure issues (label + title shape + a parseable "Failing job:"
    # line), or an ordinary issue? Only an issue satisfying all three gets
    # the "ci_failure" treatment - any one of them missing (a human-created
    # issue that merely reuses the label, or a monitor issue whose title
    # was edited) falls back to "general" rather than half-applying the
    # CI-failure protocol to something it doesn't actually describe.
    local IS_CI_FAILURE=false SIGNATURE="" FAILURE_URL=""
    if [ "$ISSUE_HAS_LABEL" = "true" ]; then
        local STRIPPED_TITLE="${ISSUE_TITLE#Nightly CI Failure: }"
        if [ "$STRIPPED_TITLE" != "$ISSUE_TITLE" ]; then
            FAILURE_URL=$(printf '%s' "$ISSUE_BODY" | grep -oP '(?<=- Failing job: )\S+' | head -1 || true)
            if [ -n "$FAILURE_URL" ]; then
                SIGNATURE="$STRIPPED_TITLE"
                IS_CI_FAILURE=true
            fi
        fi
    fi

    # The whole comment *is* the instruction now (it's a brand-new comment,
    # not a line edited into a larger existing one) - just strip the
    # leading `@claude` token and any immediately-following punctuation/
    # whitespace, keeping every line after that.
    if ! printf '%s' "$COMMENT_BODY" | grep -qP '^[[:space:]]*@[Cc][Ll][Aa][Uu][Dd][Ee]([^a-zA-Z]|$)'; then
        echo "  -> comment does not start with @claude, skipping" >&2
        return
    fi
    local INSTRUCTIONS
    INSTRUCTIONS=$(printf '%s' "$COMMENT_BODY" \
        | sed -E '1s/^[[:space:]]*@[Cc][Ll][Aa][Uu][Dd][Ee][[:space:]:,]*//' \
        | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')
    if [ -z "$INSTRUCTIONS" ]; then
        echo "  -> no instructions after @claude, skipping" >&2
        return
    fi

    local INSTRUCTIONS_HASH MENTION_MARKER
    INSTRUCTIONS_HASH=$(printf '%s' "$INSTRUCTIONS" | sha256sum | cut -c1-12)
    if [ "$IS_CI_FAILURE" = true ]; then
        MENTION_MARKER="<!-- rocm-nightly-claude-mention:${SIGNATURE}:${INSTRUCTIONS_HASH} -->"
    else
        MENTION_MARKER="<!-- claude-mention:${INSTRUCTIONS_HASH} -->"
    fi
    if printf '%s' "$ALL_COMMENTS_TEXT" | grep -qF "$MENTION_MARKER"; then
        echo "  -> this exact instruction was already acted on, skipping" >&2
        return
    fi

    if [ "$IS_CI_FAILURE" != true ]; then
        echo "  -> queued (general): ${INSTRUCTIONS}" >&2
        # repository/head_sha are fixed, not derived from any failing job -
        # there isn't one here. Still included so apply_patch_and_open_pr
        # can apply a patch this target's skill run might produce the
        # exact same way it does for a "ci_failure" target, with no
        # type-specific branching of its own.
        jq -nc \
            --arg issue_number "$ISSUE_NUMBER" \
            --arg comment_id "$COMMENT_ID" \
            --arg repository "$CI_REPO" \
            --arg head_sha "main" \
            --arg instructions "$INSTRUCTIONS" \
            --arg instructions_hash "$INSTRUCTIONS_HASH" \
            '{type: "general", issue_number: $issue_number, comment_id: $comment_id,
              repository: $repository, head_sha: $head_sha,
              instructions: $instructions, instructions_hash: $instructions_hash}'
        return
    fi

    # Split at the LAST " / ", not the first: JOB_NAME itself routinely
    # contains " / " (reusable-workflow-called jobs are named
    # "caller / callee", e.g. "jax-upstream / JAX Upstream Linux x86 ..."),
    # while STEP_NAME never does.
    local JOB_NAME="${SIGNATURE% / *}" STEP_NAME="${SIGNATURE##* / }"

    # .../actions/runs/<run_id>/job/<job_id>
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

    echo "  -> queued (ci_failure): ${INSTRUCTIONS}" >&2
    jq -nc \
        --arg issue_number "$ISSUE_NUMBER" \
        --arg comment_id "$COMMENT_ID" \
        --arg job_name "$JOB_NAME" \
        --arg step_name "$STEP_NAME" \
        --arg repository "$HEAD_REPOSITORY" \
        --arg head_sha "$HEAD_SHA" \
        --arg failure_url "$FAILURE_URL" \
        --arg run_id "$RUN_ID" \
        --arg job_id "$JOB_ID" \
        --arg instructions "$INSTRUCTIONS" \
        --arg instructions_hash "$INSTRUCTIONS_HASH" \
        '{type: "ci_failure", issue_number: $issue_number, comment_id: $comment_id,
          job_name: $job_name, step_name: $step_name, repository: $repository,
          head_sha: $head_sha, failure_url: $failure_url, run_id: $run_id,
          job_id: $job_id, instructions: $instructions,
          instructions_hash: $instructions_hash}'
}

TARGETS="[]"

if [ "${1:-}" = "--comment" ]; then
    if [ $# -ne 4 ]; then
        echo "Usage: $0 --comment <issue_number> <comment_id> <comment_body>" >&2
        exit 1
    fi
    ISSUE_NUMBER="$2"
    COMMENT_ID="$3"
    COMMENT_BODY="$4"

    ISSUE_JSON=$(gh issue view "$ISSUE_NUMBER" --repo "$CI_REPO" --json title,body,labels)
    ISSUE_TITLE=$(printf '%s' "$ISSUE_JSON" | jq -r '.title')
    ISSUE_BODY=$(printf '%s' "$ISSUE_JSON" | jq -r '.body')
    ISSUE_HAS_LABEL=$(printf '%s' "$ISSUE_JSON" | jq --arg l "$LABEL" '[.labels[].name] | any(. == $l)')
    ALL_COMMENTS_TEXT=$(gh api --paginate "/repos/${CI_REPO}/issues/${ISSUE_NUMBER}/comments" | jq -r '[.[].body] | join("\n")')
    TARGET=$(process_comment "$ISSUE_NUMBER" "$COMMENT_ID" "$COMMENT_BODY" "$ISSUE_TITLE" "$ISSUE_BODY" "$ISSUE_HAS_LABEL" "$ALL_COMMENTS_TEXT")
    if [ -n "$TARGET" ]; then
        TARGETS=$(printf '%s' "$TARGETS" | jq -c --argjson t "$TARGET" '. + [$t]')
    fi
elif [ $# -eq 0 ]; then
    # No `--label` filter: a qualifying comment can now be on any open
    # issue, not just the nightly monitor's own rocm-infra-error ones.
    OPEN_ISSUES=$(gh issue list --repo "$CI_REPO" --state open --limit 1000 \
        --json number,title,body,labels)

    mapfile -t ISSUE_NUMBERS < <(printf '%s' "$OPEN_ISSUES" | jq -r '.[].number')

    for ISSUE_NUMBER in "${ISSUE_NUMBERS[@]}"; do
        ISSUE_TITLE=$(printf '%s' "$OPEN_ISSUES" | jq -r --argjson n "$ISSUE_NUMBER" '.[] | select(.number == $n) | .title')
        ISSUE_BODY=$(printf '%s' "$OPEN_ISSUES" | jq -r --argjson n "$ISSUE_NUMBER" '.[] | select(.number == $n) | .body')
        ISSUE_HAS_LABEL=$(printf '%s' "$OPEN_ISSUES" | jq --arg l "$LABEL" --argjson n "$ISSUE_NUMBER" \
            '[.[] | select(.number == $n) | .labels[].name] | any(. == $l)')

        COMMENTS=$(gh api --paginate "/repos/${CI_REPO}/issues/${ISSUE_NUMBER}/comments")
        ALL_COMMENTS_TEXT=$(printf '%s' "$COMMENTS" | jq -r '[.[].body] | join("\n")')
        mapfile -t COMMENT_IDS < <(printf '%s' "$COMMENTS" | jq -r '.[].id')

        for COMMENT_ID in "${COMMENT_IDS[@]}"; do
            COMMENT_BODY=$(printf '%s' "$COMMENTS" | jq -r --argjson id "$COMMENT_ID" '.[] | select(.id == $id) | .body')
            TARGET=$(process_comment "$ISSUE_NUMBER" "$COMMENT_ID" "$COMMENT_BODY" "$ISSUE_TITLE" "$ISSUE_BODY" "$ISSUE_HAS_LABEL" "$ALL_COMMENTS_TEXT")
            if [ -n "$TARGET" ]; then
                TARGETS=$(printf '%s' "$TARGETS" | jq -c --argjson t "$TARGET" '. + [$t]')
            fi
        done
    done
else
    echo "Usage: $0 [--comment <issue_number> <comment_id> <comment_body>]" >&2
    exit 1
fi

printf '%s\n' "$TARGETS"
