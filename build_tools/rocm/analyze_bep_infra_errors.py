#!/usr/bin/env python3
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
"""Check BEP JSON for infrastructure error events."""

import json
import re
import sys

if len(sys.argv) != 2:
    sys.exit(2)

bep_file = sys.argv[1]

try:
    with open(bep_file, "r") as f:
        events = [json.loads(line) for line in f if line.strip()]
except:
    sys.exit(0)

# Infrastructure failure reasons in aborted events
INFRA_REASONS = [
    "REMOTE_FAILURE",
    "OUT_OF_MEMORY",
    "INTERNAL",
    "LOADING_FAILURE",
    "NO_ANALYZE",
    "NO_BUILD",
]

# Infrastructure error patterns in stderr/error messages
INFRA_ERROR_PATTERNS = [
    # Remote execution / RBE errors
    r"Failed to query remote execution",
    r"UNAVAILABLE.*Unable to resolve host",
    r"remote cache.*failed",
    r"remote execution.*failed",
    r"Connection refused.*remote",

    # Memory errors
    r"out of memory",
    r"cannot allocate memory",
    r"OOM",
    r"MemoryError",

    # Disk errors
    r"no space left on device",
    r"disk.*full",
    r"I/O error",

    # Network errors
    r"network.*unreachable",
    r"connection.*timed out",
    r"DNS.*failed",

    # BEP upload failures
    r"Build Event Protocol upload failed",
    r"BEP upload.*failed",
    r"All \d+ retry attempts failed",

    # Bazel internal errors
    r"bazel.*internal error",
    r"INTERNAL_ERROR",
    r"IllegalStateException",
    r"NullPointerException",
    r"bazel.*crashed",
]

# Patterns that indicate legitimate build/code/test errors (NOT infrastructure)
BUILD_ERROR_PATTERNS = [
    # Build file errors
    r"no such attribute",
    r"package contains errors",
    r"error loading package",
    r"syntax error",
    r"undefined.*variable",
    r"name.*is not defined",
    r"unexpected.*token",
    r"missing.*argument",

    # Test failures (not infrastructure failures)
    r"FAILED.*test",
    r"test.*failed",
    r"\d+ test.*FAILED",
    r"FAIL:",
    r"Assertion.*failed",
    r"Expected.*but got",
    r"Test.*timed out",
    r"Test case.*failed",

    # Build failures (not infrastructure failures)
    r"FAILED TO BUILD",
    r"fails to build",
]

found = False

for e in events:
    # Skip test summary and test result events (test failures are not infrastructure errors)
    if "testSummary" in e or "testResult" in e:
        continue

    # Check for aborted events with infrastructure failure reasons
    if "aborted" in e:
        reason = e["aborted"].get("reason", "")
        if reason in INFRA_REASONS:
            print(f"INFRA_ERROR: aborted.reason={reason}")  # DISABLE_DEBUG_PRINT_CHECK
            found = True

    # Check finished event for BEP upload errors
    if "finished" in e:
        finished = e["finished"]
        # Check exit code details
        if "exitCode" in finished and "name" in finished.get("exitCode", {}):
            exit_code_name = finished["exitCode"]["name"]
            if exit_code_name and isinstance(exit_code_name, str):
                for pattern in INFRA_ERROR_PATTERNS:
                    if re.search(pattern, exit_code_name, re.IGNORECASE):
                        print(f"INFRA_ERROR: {pattern} in finished.exitCode.name")  # DISABLE_DEBUG_PRINT_CHECK
                        found = True
                        break

    # Check stderr in progress events
    if "progress" in e and "stderr" in e["progress"]:
        stderr = e["progress"]["stderr"]
        if isinstance(stderr, str):
            # Always check for infrastructure errors
            for pattern in INFRA_ERROR_PATTERNS:
                if re.search(pattern, stderr, re.IGNORECASE):
                    print(f"INFRA_ERROR: {pattern} in progress.stderr")  # DISABLE_DEBUG_PRINT_CHECK
                    found = True
                    break

    # Check stdout in progress events
    if "progress" in e and "stdout" in e["progress"]:
        stdout = e["progress"]["stdout"]
        if isinstance(stdout, str):
            # Always check for infrastructure errors
            for pattern in INFRA_ERROR_PATTERNS:
                if re.search(pattern, stdout, re.IGNORECASE):
                    print(f"INFRA_ERROR: {pattern} in progress.stdout")  # DISABLE_DEBUG_PRINT_CHECK
                    found = True
                    break

    # Check action events for failures (but skip test actions - those are test failures, not infra)
    if "action" in e:
        action = e["action"]
        # Skip if this is a test action (test failures are not infrastructure errors)
        action_type = action.get("type", "")
        if action_type == "TestRunner":
            continue

        # Non-zero exit codes from non-test actions
        if "exitCode" in action and action["exitCode"] != 0:
            # Check if failure message indicates infrastructure issue
            if "stderr" in action and isinstance(action["stderr"], str):
                # Always check for infrastructure errors
                for pattern in INFRA_ERROR_PATTERNS:
                    if re.search(pattern, action["stderr"], re.IGNORECASE):
                        print(f"INFRA_ERROR: Action failure - {pattern} in stderr")  # DISABLE_DEBUG_PRINT_CHECK
                        found = True
                        break
            if "stdout" in action and isinstance(action["stdout"], str):
                # Always check for infrastructure errors
                for pattern in INFRA_ERROR_PATTERNS:
                    if re.search(pattern, action["stdout"], re.IGNORECASE):
                        print(f"INFRA_ERROR: Action failure - {pattern} in stdout")  # DISABLE_DEBUG_PRINT_CHECK
                        found = True
                        break

sys.exit(1 if found else 0)
