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
# List the failures of a single pipeline run, one "job / step" per line.
#
# Every failure is reported: no attempt is made to tell infrastructure failures
# apart from build or test failures, because the monitored pipeline is expected
# to be green.
#
# Usage: collect_run_failures.sh <repository> <run_id>
#
# Args:
#   repository: GitHub repository, "owner/repo"
#   run_id:     Workflow run ID to inspect
#
# Env:
#   IGNORED_STEP_MARKER: Steps whose name contains this marker are known-red
#                        and are not reported. Empty means report everything.
#
# Output:
#   One indented "job / step" line per failure on stdout, nothing if the run
#   has no reportable failure.
#
# Exit codes:
#   0: Success (including "no failures found")
#   1: Error

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <repository> <run_id>" >&2
    exit 1
fi

REPOSITORY="$1"
RUN_ID="$2"

IGNORED_STEP_MARKER="${IGNORED_STEP_MARKER:-}"

JOBS_JSON=$(gh api --paginate \
    -H "Accept: application/vnd.github+json" \
    "/repos/${REPOSITORY}/actions/runs/${RUN_ID}/jobs?per_page=100")

# GitHub truncates job names at 100 characters, and the nightly matrix puts a
# ROCm tarball URL in them, so the name usually ends in a half-cut URL. Keep
# only the first matrix value, which is the one that identifies the job
# ("execute-xla-ut-nightly (therock_release, https://rocm.dev..." becomes
# "execute-xla-ut-nightly (therock_release)"). The job's own URL is appended to
# every line instead, so each failure stays one greppable, clickable line.
SHORTEN_JOB_NAME='def short_job_name: if test("\\(") and test(",")
                                      then sub(",.*$"; ")")
                                      else . end;'

# Failed steps, except the known-red ones. An empty marker reports everything:
# jq's contains("") is true for every name, so it has to be special cased.
STEP_FAILURES=$(printf '%s' "$JOBS_JSON" | jq -r --arg marker "$IGNORED_STEP_MARKER" "
    ${SHORTEN_JOB_NAME}"'
    .jobs[]
    | . as $job
    | ($job.steps // [])[]
    | select(.conclusion == "failure")
    | select($marker == "" or ((.name | contains($marker)) | not))
    | "  - \($job.name | short_job_name) / \(.name) -> \($job.html_url)"
')

# Jobs that failed without any failed step - a startup failure, a lost runner,
# a cancellation. There is no step to point at, so report the job itself.
JOB_FAILURES=$(printf '%s' "$JOBS_JSON" | jq -r "
    ${SHORTEN_JOB_NAME}"'
    .jobs[]
    | select(.conclusion == "failure" or .conclusion == "startup_failure")
    | select([(.steps // [])[] | select(.conclusion == "failure")] | length == 0)
    | "  - \(.name | short_job_name) / job \(.conclusion) with no failed step -> \(.html_url)"
')

printf '%s\n%s\n' "$STEP_FAILURES" "$JOB_FAILURES" | grep -v '^$' | sort -u || true
