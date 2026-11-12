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

# Exact title match against ALL open issues, not just ones carrying the
# label: gh issue create's label+assignee flags fail atomically if even one
# assignee lacks repo access (see the fallback chain below), which can
# leave the tracking issue without the label - filtering the search on the
# label too would then make it invisible to future runs and recreate a new
# issue every single time. "gh issue list --search" is a fuzzy text search,
# so match titles exactly in jq instead.
# Default --limit is 30 (newest first); without the label filter this now
# scans every open issue in the repo, so raise it well past realistic open
# issue counts rather than risk missing an older, still-open tracking issue.
ISSUE_NUMBER=$(gh issue list --repo "$ISSUE_REPO" --state open --limit 1000 \
    --json number,title \
    | jq -r --arg title "$TITLE" 'map(select(.title == $title)) | .[0].number // empty')

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

# `gh issue edit --add-label` fails if the label doesn't exist as a repo
# label object yet (distinct from it just not being on this issue) - ensure
# it exists first, every time, since there's otherwise no signal that it's
# missing (every add-label call below would silently warn-and-continue
# forever without ever actually labeling anything).
gh label create "$LABEL" --repo "$ISSUE_REPO" --force \
    --color b60205 --description "ROCm CI nightly infra/test failure tracking issue" \
    >/dev/null 2>&1 \
    || echo "Warning: could not create/update label $LABEL" >&2

if [ -n "$ISSUE_NUMBER" ]; then
    echo "Using existing issue #$ISSUE_NUMBER in $ISSUE_REPO" >&2
    # Self-heal: an issue created before this label/assignee decoupling fix
    # may be missing the label (see below) - make sure it has it, since the
    # dedup search above no longer depends on it but other tooling might.
    gh issue edit "$ISSUE_NUMBER" --repo "$ISSUE_REPO" --add-label "$LABEL" >/dev/null 2>&1 \
        || echo "Warning: could not add label $LABEL to #$ISSUE_NUMBER" >&2
    echo "$ISSUE_NUMBER"
    exit 0
fi

BODY="## ROCm CI Nightly Failures

This issue tracks every failure the nightly pipeline monitor detects. Each
failure is appended below as its own comment, with a link to the failing
job. A Claude Code debug run is queued automatically for each failure
signature that hasn't already been debugged, and comments its findings
directly on this issue, quoting the detection comment it addresses.

cc @ROCm/ai-fw-openxla"

echo "Creating new issue in $ISSUE_REPO" >&2
NEW_ISSUE=$(gh issue create --repo "$ISSUE_REPO" --title "$TITLE" --body "$BODY")
echo "Issue created: $NEW_ISSUE" >&2
NEW_ISSUE_NUMBER=$(echo "$NEW_ISSUE" | grep -oE '[0-9]+$')

# Label and each assignee are added separately, after creation, so one
# invalid assignee (lacking repo access) can't make `gh issue create` fail
# atomically and silently drop the label too - which is exactly what used
# to make the dedup search above miss this issue and recreate a new one
# every single run. `gh issue edit` prints the issue URL to stdout on
# success - redirect it, or it contaminates this script's own stdout, which
# the caller captures wholesale as the issue number (confirmed in
# production: the captured "issue number" became the URL repeated once per
# add-label/add-assignee call, which then broke a downstream `gh api` URL).
gh issue edit "$NEW_ISSUE_NUMBER" --repo "$ISSUE_REPO" --add-label "$LABEL" >/dev/null 2>&1 \
    || echo "Warning: could not add label $LABEL" >&2
for member in "${TEAM_MEMBERS[@]}"; do
    gh issue edit "$NEW_ISSUE_NUMBER" --repo "$ISSUE_REPO" --add-assignee "$member" >/dev/null 2>&1 \
        || echo "Warning: could not assign $member (no repo access?)" >&2
done

echo "$NEW_ISSUE_NUMBER"
