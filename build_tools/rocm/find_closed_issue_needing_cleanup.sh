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
# Find the oldest closed tracking issue (title exactly matching
# create_infra_error_issue.sh's TITLE) that the cleanup-closed-nightly-issue
# skill hasn't already processed - i.e. it has no comment starting with the
# "<!-- rocm-nightly-cleanup-done -->" marker yet.
#
# Oldest-first (not "most recently closed"): in steady state there's only
# ever one closed-and-unprocessed issue at a time, so this only matters if
# cleanup is ever down for a while and a backlog accumulates - draining it
# oldest-first keeps migrated signatures from getting jumbled out of order
# across multiple destination issues.
#
# Usage: find_closed_issue_needing_cleanup.sh
#
# Output:
#   The issue number on stdout, empty if none found (diagnostics on stderr).
#
# Exit codes:
#   0: Success (including "none found")
#   1: Error

set -euo pipefail

ISSUE_REPO="ROCm/xla"
TITLE="ROCm CI Nightly Failures Detected"
DONE_MARKER="<!-- rocm-nightly-cleanup-done -->"

# Same --limit 1000 safety margin as create_infra_error_issue.sh's own scan:
# gh issue list defaults to 30 (newest first), which could miss an older
# closed issue in a backlog scenario.
CANDIDATES_JSON=$(gh issue list --repo "$ISSUE_REPO" --state closed --limit 1000 \
    --json number,title,closedAt \
    | jq -c --arg title "$TITLE" \
        'map(select(.title == $title)) | sort_by(.closedAt)')

# mapfile populates the array fully before the loop starts, so an early
# `exit` below never closes a pipe a producer is still writing to (the
# SIGPIPE-under-pipefail gotcha documented in get_checkout_source.sh).
mapfile -t CANDIDATE_NUMBERS < <(printf '%s' "$CANDIDATES_JSON" | jq -r '.[].number')

for NUMBER in "${CANDIDATE_NUMBERS[@]}"; do
    echo "Checking closed issue #${NUMBER} for an existing cleanup marker" >&2
    ALREADY_DONE=$(gh api --paginate "/repos/${ISSUE_REPO}/issues/${NUMBER}/comments" \
        | jq --arg marker "$DONE_MARKER" '[.[] | select(.body | startswith($marker))] | length')

    if [ "${ALREADY_DONE:-0}" -eq 0 ]; then
        echo "  -> needs cleanup" >&2
        echo "$NUMBER"
        exit 0
    fi
    echo "  -> already cleaned up, skipping" >&2
done

echo "No closed tracking issue needs cleanup" >&2
echo ""
