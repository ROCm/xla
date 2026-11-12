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
# Find the exact source commit a specific monitored job checked out, by
# parsing its own raw log - not the run's head_sha/head_repository.
#
# The monitored nightly/sanitizer jobs check out a *different* repository
# than the one that triggered the run: rocm_xla_ci_nightly.yml runs in
# ROCm/xla, but its "execute-xla-ut-nightly" (and the sanitizer/jax jobs it
# calls) check out `repository: openxla/xla, ref: main` for the code under
# test. A run's own `head_sha`/`head_repository` reflect ROCm/xla's commit
# at schedule time, which is unrelated to the actual openxla/xla commit
# "main" resolved to when that job ran - using it would make the debug job
# check out the wrong source tree entirely. The job's own actions/checkout
# step, however, always logs both the configured `repository:` input and
# the fully-resolved commit SHA it checked out (`git log -1 --format=%H`),
# so parsing that log is the only reliable way to recover it after the
# fact.
#
# Usage: get_checkout_source.sh <ci_repo> <job_id>
#
# Args:
#   ci_repo: Repository hosting the job (and its log), "owner/repo"
#   job_id:  The job ID to inspect
#
# Output (stdout, two lines; diagnostics on stderr):
#   <repository>
#   <sha>
#
# Exit codes:
#   0: Success
#   1: Error (no resolved checkout commit found in the job's log)

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <ci_repo> <job_id>" >&2
    exit 1
fi

CI_REPO="$1"
JOB_ID="$2"

# Strip the leading ISO-8601 timestamp GitHub prefixes every log line with,
# and any stray carriage returns. --allow-escape-sequences: CI logs routinely
# contain raw ANSI color codes (Bazel, gtest, ...), and gh api otherwise
# refuses to print a response that contains them at all.
LOG=$(gh api --allow-escape-sequences "/repos/${CI_REPO}/actions/jobs/${JOB_ID}/logs" \
    | tr -d '\r' | sed -E 's/^[^ ]+ //')

# The `repository:` input of the *first* actions/checkout step in the job -
# its primary source checkout, always the first step in every monitored job
# observed so far. Here-strings, not a pipe: awk's early `exit` closes its
# stdin before a piped producer finishes writing, and with `pipefail` that
# SIGPIPE aborts the whole script under `set -e`.
REPOSITORY=$(awk '
    /^##\[group\]Run actions\/checkout/ { in_block=1; next }
    in_block && /^##\[endgroup\]/ { exit }
    in_block && /^[[:space:]]*repository:[[:space:]]*/ {
        sub(/^[[:space:]]*repository:[[:space:]]*/, "");
        print;
        exit
    }
' <<< "$LOG")

# actions/checkout logs the fully-resolved commit right after checking out,
# via `git log -1 --format=%H` followed by the SHA on the next line. Take
# the first occurrence - the primary source checkout, same block as above.
SHA=$(awk '
    /^\[command\].*git log -1 --format=%H$/ { getline; print; exit }
' <<< "$LOG")

if [ -z "$SHA" ]; then
    echo "Error: could not find a resolved checkout commit in job ${JOB_ID} (${CI_REPO}) log" >&2
    exit 1
fi

# No repository override found means the job just checked out the repo it
# runs in.
if [ -z "$REPOSITORY" ]; then
    REPOSITORY="$CI_REPO"
fi

echo "$REPOSITORY"
echo "$SHA"
