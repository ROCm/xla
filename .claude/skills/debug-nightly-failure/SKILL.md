---
name: debug-nightly-failure
description: Reproduce and debug a single ROCm XLA nightly CI failure (one build or test job/step) on real GPU hardware via RBE, then report a verified fix or an analysis/callstack directly on the tracking GitHub issue. Use when invoked by the "debug" job of rocm_infra_monitor.yml for one specific failed job/step.
argument-hint: [ci-repo] [run-id] [job-id] [job-name] [step-name] [failure-url] [issue-number] [detection-comment-url]
context: fork
agent: general-purpose
allowed-tools: Bash(gh issue comment *), Bash(gh api *), Bash(gh run view *), Bash(git *), Bash(build_tools/rocm/*), Bash(bazel *), Bash(gdb *), Bash(objdump *), Bash(grep *), Bash(cat *), Bash(timeout *), Read, Grep, Glob, Write, Edit
---

# Debug a ROCm Nightly CI Failure

The commit that produced the failure is already checked out in the current
working directory - that is the repro environment for this whole skill.

## Arguments

The caller passes eight values, space separated: `CI_REPO RUN_ID JOB_ID
JOB_NAME STEP_NAME FAILURE_URL ISSUE_NUMBER DETECTION_COMMENT_URL`.
`JOB_NAME` and `STEP_NAME` can themselves contain spaces and parentheses
(e.g. `execute-xla-ut-nightly (therock_release)`), so do not try to split
`$ARGUMENTS` positionally by whitespace - the invoking prompt also restates
each value on its own labeled line; read the values from there.

- `CI_REPO`: the repository the failing run and tracking issue live in
  (`ROCm/xla`) - use this for every `gh api`/`gh issue` call, never
  `SOURCE REPO`.
- `RUN_ID` / `JOB_ID`: identify the failing job within `CI_REPO`.
- `FAILURE_URL`: link to the failing job, for reference in your report.
- `ISSUE_NUMBER`: the single, long-lived tracking issue to comment on when
  you're done. It is shared by every failure this pipeline has ever
  detected, not just this one - never treat its presence as information
  about this specific failure, and never summarize or react to unrelated
  comments already on it.
- `DETECTION_COMMENT_URL`: the specific comment on that issue reporting
  *this* failure. Quote it in your report (see Step 4) so a reader can tell
  which of the issue's many comments your analysis addresses.

## IMPORTANT: scope and safety

- Debug only this one job/step. Do not try to fix unrelated red steps you
  notice along the way.
- Never `git push`, never create a branch, never open a PR. Report findings
  only via `gh issue comment <ISSUE_NUMBER> --repo <CI_REPO>`.
- Clean up any scratch files (`/tmp/gdb_*`, extra build outputs) before
  finishing - nothing from this debugging session is meant to be committed.
- Never claim a fix works without having re-run the originally-failing
  target and observed it pass.

## Step 1 — Understand what actually failed

Fetch the failing step's raw log and read it before doing anything else:

```
gh api "repos/<CI_REPO>/actions/jobs/<JOB_ID>/logs" > job.log
```

From the log, extract:
- The exact `build_tools/rocm/run_xla_ci_build.sh ...` (or plain `bazel`)
  invocation that was run for the failing step. Reproduce that command
  verbatim rather than guessing matrix-specific flags (`rbe_config`, distro
  URL/hash, `gfx` target) - different nightly matrix legs use different
  values, and the log already has the real ones.
- The specific failing target(s) and, for a test, the specific failing test
  case/parameter.
- The error signature: a crash with a stack trace, a hang/timeout, a wrong
  numerical result, a build error, etc. This determines which debugging
  technique in Step 3 applies.

## Step 2 — Reproduce it, narrowed to one target

Re-run the extracted command narrowed to only the failing target (or
`--test_filter=`/`--test_arg=--gtest_filter=` for one test case), adding:
- `--nocache_test_results` (force re-execution, don't trust the remote
  cache)
- a generous `--test_timeout=<seconds>` if the failure might be a hang
- `--local_test_jobs=1` if not already present, to keep GPU contention out
  of the picture while narrowing down

This mirrors the pattern already explored manually in this repo's
`build.sh` / `build_debug.sh` scratch files (narrow to one target, one
`gtest_filter`, no repeat).

If it does not reproduce at all after a genuine attempt (correct flags,
correct commit, enough timeout), say so plainly in the final report instead
of forcing an inconclusive debugging session.

## Step 3 — Debugging techniques

**`--run_under`**: Bazel's `--run_under` flag accepts either a build-target
label (this repo's own `//build_tools/rocm:parallel_gpu_execute` and
`//build_tools/rocm:sanitizer_wrapper_parallel_gpu_execute`, which is what
the nightly/sanitizer CI itself already runs tests under) **or a plain
command/path string**, which Bazel prepends to the test binary's argv. For
ad hoc debugging, write a small wrapper script and point `--run_under` at
its absolute path - it does not need to be a Bazel target.

**Crash / assertion failure** - run under gdb and capture a backtrace:

```bash
cat > /tmp/gdb_commands.txt <<'EOF'
run
bt full
info registers
quit
EOF
cat > /tmp/gdb_wrapper.sh <<'EOF'
#!/usr/bin/env bash
exec gdb -q -batch -x /tmp/gdb_commands.txt --args "$@"
EOF
chmod +x /tmp/gdb_wrapper.sh

build_tools/rocm/run_xla_ci_build.sh <...same flags as the failing step...> \
  --run_under=/tmp/gdb_wrapper.sh \
  -- //path/to:failing_target
```

**Hang** (process never exits, so a plain `run`/`bt`/`quit` script never
gets to run) - launch in the background and attach after a delay instead:

```bash
cat > /tmp/gdb_hang_wrapper.sh <<'EOF'
#!/usr/bin/env bash
"$@" &
pid=$!
sleep 60   # tune to when the hang is expected to have occurred
gdb -p "$pid" -batch -ex 'thread apply all bt full' -ex 'kill' -ex 'quit'
wait "$pid" 2>/dev/null
EOF
chmod +x /tmp/gdb_hang_wrapper.sh
# use --run_under=/tmp/gdb_hang_wrapper.sh the same way as above
```

**Narrowing where in the pipeline it diverges** - targeted VLOG
instrumentation via `--test_env=TF_CPP_VMODULE="autotuner=2,config_assigner=2,
config_runner=2,codegen_orchestrator=2,hipblaslt=2,triton=2,
fission_backend=2"` (adjust the module list to the subsystem implicated by
the error signature from Step 1), as explored in the scratch
`build_debug.sh` for an autotuner/codegen hang.

**GPU-codegen bugs** (wrong result, miscompiled kernel) - disassemble the
generated kernel and look for structural issues (missing barriers, bad
instruction scheduling):

```bash
find bazel-testlogs -name '*.hsaco' -path '*<failing_target>*'
objdump -d <path-to.hsaco>
```

## Step 4 — Decide the outcome and report

Always report via exactly one comment:
`gh issue comment <ISSUE_NUMBER> --repo <CI_REPO> --body-file <file>`, built
like this:

1. The very first line must be exactly
   `<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->` - substitute
   the literal `JOB_NAME` and `STEP_NAME` values you were given, separated
   by ` / ` with no other change to the text. The caller checks for this
   exact marker, for this exact signature, before starting a debug run, so
   that this specific failure never gets debugged a second time even though
   the tracking issue is shared by many different failures - leaving it
   off, or using the wrong job/step text, would defeat that check.
2. Next, quote the detection comment you were given
   (`DETECTION_COMMENT_URL`): fetch it with
   `gh api repos/<CI_REPO>/issues/comments/<comment-id>` (the numeric ID
   after `#issuecomment-` in the URL), then reproduce its body as a
   markdown blockquote (`>` prefix on every line), followed by a link back
   to it (`[Original detection](DETECTION_COMMENT_URL)`). This is what lets
   a reader tell which of the issue's many comments your analysis
   addresses.
3. Then your actual report, as below.

**Fix found**: before reporting, re-run the exact originally-failing
target/test case one more time and confirm it now passes. Then comment
with:
- A short root-cause summary.
- The fix as a diff, from `git diff`, in a \`\`\`diff fenced code block.
- The command you used to confirm it passes, and its outcome.

**Not resolved**: comment with:
- The failure signature (crash / hang / wrong result / build error) and
  exactly which target/test case.
- What you ruled out and what you tried (be concrete - "ruled out X because
  Y", not "investigated further").
- The most useful evidence you gathered: a backtrace, a VLOG excerpt around
  the divergence point, or a kernel disassembly snippet - whichever applies.
- A concrete, specific suggestion for the next debugging step. Never end
  with a vague "could not reproduce" with no supporting detail.
