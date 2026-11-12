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
# Find or create the single, long-lived GitHub issue that tracks ROCm CI
# nightly failures. There is never more than one open issue with this
# title: every individual failure is appended to it as its own comment by
# collect_claude_debug_targets.sh, rather than getting a new issue of its
# own.
#
# Usage: create_infra_error_issue.sh
#
# Output:
#   The issue number on stdout (diagnostics on stderr).
#
# Exit codes:
#   0: Success
#   1: Error

set -euo pipefail

ISSUE_REPO="ROCm/xla"
LABEL="rocm-infra-error"
TITLE="ROCm CI Nightly Failures Detected"

# Exact title match against open issues carrying the label. "gh issue list
# --search" is a fuzzy text search, so match titles exactly in jq instead.
ISSUE_NUMBER=$(gh issue list --repo "$ISSUE_REPO" --state open --label "$LABEL" \
    --json number,title \
    | jq -r --arg title "$TITLE" 'map(select(.title == $title)) | .[0].number // empty')

if [ -n "$ISSUE_NUMBER" ]; then
    echo "Using existing issue #$ISSUE_NUMBER in $ISSUE_REPO" >&2
    echo "$ISSUE_NUMBER"
    exit 0
fi

# Team members to assign to the tracking issue
# Adjust this list as needed
TEAM_MEMBERS=(
    alekstheod
    i-chaochen
    hsharsha
    charleshofer
    leo-automation
    pelumi1163
    nurmukhametov
)

ASSIGNEES=""
for member in "${TEAM_MEMBERS[@]}"; do
    ASSIGNEES="$ASSIGNEES --assignee $member"
done
echo "Will assign to: ${TEAM_MEMBERS[*]}" >&2

BODY="## ROCm CI Nightly Failures

This issue tracks every failure the nightly pipeline monitor detects. Each
failure is appended below as its own comment, with a link to the failing
job. A Claude Code debug run is queued automatically for each failure
signature that hasn't already been debugged, and comments its findings
directly on this issue, quoting the detection comment it addresses.

cc @ROCm/ai-fw-openxla"

echo "Creating new issue in $ISSUE_REPO" >&2
# Try with label and assignees first, fallback if it fails
if NEW_ISSUE=$(gh issue create \
    --repo "$ISSUE_REPO" \
    --title "$TITLE" \
    --body "$BODY" \
    --label "$LABEL" \
    $ASSIGNEES 2>&1); then
    echo "Issue created: $NEW_ISSUE" >&2
else
    echo "Warning: Could not create with label/assignees, trying without label" >&2
    if [ -n "$ASSIGNEES" ]; then
        if NEW_ISSUE=$(gh issue create \
            --repo "$ISSUE_REPO" \
            --title "$TITLE" \
            --body "$BODY" \
            $ASSIGNEES 2>&1); then
            echo "Issue created: $NEW_ISSUE" >&2
        else
            echo "Warning: Could not assign members, creating issue without assignees" >&2
            NEW_ISSUE=$(gh issue create \
                --repo "$ISSUE_REPO" \
                --title "$TITLE" \
                --body "$BODY")
            echo "Issue created: $NEW_ISSUE" >&2
        fi
    else
        NEW_ISSUE=$(gh issue create \
            --repo "$ISSUE_REPO" \
            --title "$TITLE" \
            --body "$BODY")
        echo "Issue created: $NEW_ISSUE" >&2
    fi
fi

echo "$NEW_ISSUE" | grep -oE '[0-9]+$'
