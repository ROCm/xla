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
# Create or update GitHub issue for ROCm CI infrastructure errors.
#
# Usage: create_infra_error_issue.sh <errors_file> <workflow_run_url>
#
# Args:
#   errors_file: File containing error details
#   workflow_run_url: URL to the monitoring workflow run
#
# Exit codes:
#   0: Success
#   1: Error

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <errors_file> <workflow_run_url>" >&2
    exit 1
fi

ERRORS_FILE="$1"
WORKFLOW_URL="$2"

if [ ! -f "$ERRORS_FILE" ]; then
    echo "Error: Errors file not found: $ERRORS_FILE" >&2
    exit 1
fi

TITLE="ROCm CI Infrastructure Errors Detected"

# Repository where issues should be created (ROCm fork, where the team has access)
ISSUE_REPO="ROCm/xla"

# Team members to assign to infrastructure error issues
# Adjust this list as needed
TEAM_MEMBERS=(
    alekstheod
    i-chaochen
    hsharsha
    charleshofer
)

# Build assignee flags
ASSIGNEES=""
for member in "${TEAM_MEMBERS[@]}"; do
    ASSIGNEES="$ASSIGNEES --assignee $member"
done
echo "Will assign to: ${TEAM_MEMBERS[*]}"

# Check if an open issue already exists (search by title, not label)
ISSUE_NUMBER=$(gh issue list --repo "$ISSUE_REPO" --state open --search "in:title $TITLE" --json number --jq '.[0].number')

BODY="## Infrastructure Errors Detected in ROCm CI

Monitoring run: ${WORKFLOW_URL}

The following workflow runs failed with infrastructure errors:

$(cat "$ERRORS_FILE")

### Next Steps
- Click the run links above to view logs and error details
- For runs with BEP artifacts, download and analyze the Build Event Protocol files
- For runs without BEP files, the failure occurred during setup/initialization

cc @ROCm/ai-fw-openxla

---
*Last updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")*"

if [ -n "$ISSUE_NUMBER" ]; then
    echo "Updating existing issue #$ISSUE_NUMBER in $ISSUE_REPO"
    gh issue comment "$ISSUE_NUMBER" --repo "$ISSUE_REPO" --body "$BODY"
    echo "Issue updated: #$ISSUE_NUMBER"
else
    echo "Creating new issue in $ISSUE_REPO"
    # Try with label and assignees first, fallback if it fails
    if NEW_ISSUE=$(gh issue create \
        --repo "$ISSUE_REPO" \
        --title "$TITLE" \
        --body "$BODY" \
        --label "rocm-infra-error" \
        $ASSIGNEES 2>&1); then
        echo "Issue created: $NEW_ISSUE"
    else
        echo "Warning: Could not create with label/assignees, trying without label"
        if [ -n "$ASSIGNEES" ]; then
            if NEW_ISSUE=$(gh issue create \
                --repo "$ISSUE_REPO" \
                --title "$TITLE" \
                --body "$BODY" \
                $ASSIGNEES 2>&1); then
                echo "Issue created: $NEW_ISSUE"
            else
                echo "Warning: Could not assign members, creating issue without assignees"
                NEW_ISSUE=$(gh issue create \
                    --repo "$ISSUE_REPO" \
                    --title "$TITLE" \
                    --body "$BODY")
                echo "Issue created: $NEW_ISSUE"
            fi
        else
            NEW_ISSUE=$(gh issue create \
                --repo "$ISSUE_REPO" \
                --title "$TITLE" \
                --body "$BODY")
            echo "Issue created: $NEW_ISSUE"
        fi
    fi
fi
