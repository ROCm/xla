---
name: debug-nightly-failure
description: Categorize (build/test/config/infra) and label a single ROCm XLA nightly CI failure, post a preliminary root-cause guess, then reproduce and debug it on real GPU hardware via RBE and report a verified fix or an analysis/callstack directly on the tracking GitHub issue. Use when invoked by the "debug" job of rocm_infra_monitor.yml for one specific failed job/step.
argument-hint: [ci-repo] [run-id] [job-id] [job-name] [step-name] [failure-url] [issue-number] [detection-comment-url]
context: fork
agent: general-purpose
allowed-tools: Bash(gh issue comment *), Bash(gh issue edit *), Bash(gh label create *), Bash(gh api *), Bash(gh run view *), Bash(git *), Bash(build_tools/rocm/*), Bash(bazel *), Bash(gdb *), Bash(objdump *), Bash(grep *), Bash(cat *), Bash(timeout *), Read, Grep, Glob, Write, Edit
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
  *this* failure. Quote it in both comments you post (Steps 1 and 5) so a
  reader can tell which of the issue's many comments your analysis
  addresses.

## IMPORTANT: scope and safety

- This runner (`build-only-xla`) has no local GPU - it's a network-access
  node, not a GPU node. Every Bazel/`run_xla_ci_build.sh` invocation you run
  (Steps 3 and 4) **must** include `--strategy=TestRunner=remote,local`, so
  GPU test actions execute on a remote RBE worker from the GPU pool instead
  of trying (and failing) to run on this machine; `local` is kept only as a
  fallback for actions that don't need a GPU. Omitting this flag means your
  test actions will not run at all.
- Debug only this one job/step. Do not try to fix unrelated red steps you
  notice along the way.
- Never `git push`, never create a branch, never open a PR. The only
  GitHub writes you make are: one label (Step 1, corrected in Step 2 if
  needed), one preliminary comment (Step 1), and one final report comment
  (Step 5) - all via `gh issue edit`/`gh issue comment <ISSUE_NUMBER> --repo <CI_REPO>`/`gh label create`.
- Clean up any scratch files before finishing - nothing from this debugging
  session is meant to be committed. This includes `/tmp/gdb_*`, extra build
  outputs, and - unlike `/tmp` - specifically `debug_tools/claude_scratch/`
  (Step 4's run_under wrapper target), since that one lives inside the git
  working tree: `rm -rf debug_tools/claude_scratch`.
- Never claim a fix works without having re-run the originally-failing
  target and observed it pass.

## Step 1 — Triage: categorize and label

Fetch the failing step's raw log once, before anything else - Step 2 reuses
this same file, so don't fetch it twice. CI logs routinely contain raw ANSI
color codes (Bazel, gtest, ...), and `gh api` refuses to print a response
containing them without `--allow-escape-sequences` - always pass it for
this and any other raw-log fetch:

```
gh api --allow-escape-sequences "repos/<CI_REPO>/actions/jobs/<JOB_ID>/logs" > job.log
```

Skim it (you don't need full understanding yet, just enough to classify)
and sort this failure into exactly one category:

- **`nightly-build-failure`** - the build itself didn't compile/link: a
  Bazel `ERROR: ... Compilation of rule '...' failed`, a linker error, a
  missing symbol. Nothing related to test *behavior*.
- **`nightly-test-failure`** - the build succeeded and a test genuinely
  failed: a gtest `[ FAILED ]` assertion/expectation, a wrong numerical
  result, a crash or hang *inside the test binary itself* (segfault,
  assertion, deadlock in the code under test).
- **`nightly-config-failure`** - the pipeline was configured wrong, not a
  bug in XLA/JAX: an undefined Bazel `--config` value, an incompatible flag
  combination, a missing `--repo_env`, a dependency/version mismatch, "no
  such package", a bad BUILD file.
- **`nightly-infra-failure`** - the CI infrastructure itself is at fault,
  not the code or its configuration: runner lost/cancelled, Docker image
  pull failure, network timeout to RBE/a registry, "No space left on
  device", OOM-killed, GPU unavailable/busy, RBE worker pool outage. A
  timeout counts here only if nothing in the log suggests the code itself
  hung - if it does, that's `nightly-test-failure` instead.

If more than one applies (e.g. an infra hiccup masked what might be a real
test failure), pick the one that is the more likely root cause, and say so
explicitly in the preliminary comment below.

Then, before doing any further work:

1. Ensure the label exists and apply it to the tracking issue (idempotent -
   safe to run every time):
   ```
   gh label create "<category>" --repo <CI_REPO> --force \
     --color <hex> --description "<one-line description>"
   gh issue edit <ISSUE_NUMBER> --repo <CI_REPO> --add-label "<category>"
   ```
   Use `d73a4a` for `nightly-build-failure`, `fbca04` for
   `nightly-test-failure`, `0075ca` for `nightly-config-failure`, `5319e7`
   for `nightly-infra-failure`.
2. Post ONE preliminary comment (separate from, and before, your final
   report in Step 5) via
   `gh issue comment <ISSUE_NUMBER> --repo <CI_REPO> --body-file <file>`:
   - First line exactly `<!-- rocm-nightly-claude-triage:JOB_NAME / STEP_NAME -->`
     (substitute the literal values, ` / ` separator, no other change). This
     is a *different* marker from the Step 5 completion marker and is never
     checked for dedup - it only marks this comment as the triage pass for
     this signature, so post it even if you end up unable to finish the
     full analysis.
   - Quote the detection comment (same method as Step 5.2 below).
   - `**Category:** <category>`
   - `**Possible root cause (preliminary, unverified):** <1-3 sentences>` -
     a concrete guess from what the log shows so far (e.g. "the step hit a
     60-minute timeout with no output after the RBE remote_cache handshake,
     which looks like an RBE connectivity issue rather than a hang in the
     test itself"). This is a hypothesis, not a conclusion - say so, and
     don't present it as certain.
   - A closing line that full reproduction is starting and a complete
     report will follow in a separate comment.

## Step 2 — Understand what actually failed

Reread `job.log` (already fetched in Step 1) and extract:
- The exact `build_tools/rocm/run_xla_ci_build.sh ...` (or plain `bazel`)
  invocation that was run for the failing step. Reproduce that command
  verbatim rather than guessing matrix-specific flags (`rbe_config`, distro
  URL/hash, `gfx` target) - different nightly matrix legs use different
  values, and the log already has the real ones.
- The specific failing target(s) and, for a test, the specific failing test
  case/parameter.
- The error signature: a crash with a stack trace, a hang/timeout, a wrong
  numerical result, a build error, etc. This determines which debugging
  technique in Step 4 applies, and whether Step 1's category guess holds up
  - if deeper reading changes your mind about the category, update the
  issue's labels (`gh issue edit --remove-label <old> --add-label <new>`)
  before finishing.

## Step 3 — Reproduce it, narrowed to one target

Re-run the extracted command narrowed to only the failing target (or
`--test_filter=`/`--test_arg=--gtest_filter=` for one test case), adding:
- `--strategy=TestRunner=remote,local` (see above - mandatory, this node has
  no GPU)
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

## Step 4 — Debugging techniques

**`--run_under`**: Bazel's `--run_under` flag accepts either a build-target
label (this repo's own `//build_tools/rocm:parallel_gpu_execute` and
`//build_tools/rocm:sanitizer_wrapper_parallel_gpu_execute`, which is what
the nightly/sanitizer CI itself already runs tests under) **or a plain
command/path string**, which Bazel prepends to the test binary's argv.

With `--strategy=TestRunner=remote,local` in play, the test executes inside
a remote RBE worker's sandbox, which contains only this action's declared
Bazel inputs - **a loose script on this host's `/tmp` is not visible there**
and `--run_under=/tmp/...` will silently do nothing useful remotely. Write
any ad hoc wrapper as a real Bazel target instead, in a throwaway package
inside the checkout - delete it before finishing, same as any other scratch
file:

```bash
mkdir -p debug_tools/claude_scratch
cat > debug_tools/claude_scratch/BUILD <<'EOF'
sh_binary(
    name = "gdb_wrapper",
    srcs = ["gdb_wrapper.sh"],
    data = ["gdb_commands.txt"],
)
EOF
cat > debug_tools/claude_scratch/gdb_commands.txt <<'EOF'
run
bt full
info registers
quit
EOF
cat > debug_tools/claude_scratch/gdb_wrapper.sh <<'EOF'
#!/usr/bin/env bash
exec gdb -q -batch -x "$(dirname "$0")/gdb_commands.txt" --args "$@"
EOF
chmod +x debug_tools/claude_scratch/gdb_wrapper.sh
```

**Crash / assertion failure** - run under gdb and capture a backtrace:

```bash
build_tools/rocm/run_xla_ci_build.sh <...same flags as the failing step...> \
  --strategy=TestRunner=remote,local \
  --run_under=//debug_tools/claude_scratch:gdb_wrapper \
  -- //path/to:failing_target
```

**Hang** (process never exits, so a plain `run`/`bt`/`quit` script never
gets to run) - launch in the background and attach after a delay instead.
Same BUILD target pattern as above, but with a different wrapper script:

```bash
cat > debug_tools/claude_scratch/gdb_wrapper.sh <<'EOF'
#!/usr/bin/env bash
"$@" &
pid=$!
sleep 60   # tune to when the hang is expected to have occurred
gdb -p "$pid" -batch -ex 'thread apply all bt full' -ex 'kill' -ex 'quit'
wait "$pid" 2>/dev/null
EOF
chmod +x debug_tools/claude_scratch/gdb_wrapper.sh
# re-run the same build_tools/rocm/run_xla_ci_build.sh command as above
```

**Narrowing where in the pipeline it diverges** - targeted VLOG
instrumentation via `--test_env=TF_CPP_VMODULE="autotuner=2,config_assigner=2,
config_runner=2,codegen_orchestrator=2,hipblaslt=2,triton=2,
fission_backend=2"` (adjust the module list to the subsystem implicated by
the error signature from Step 2), as explored in the scratch
`build_debug.sh` for an autotuner/codegen hang.

**GPU-codegen bugs** (wrong result, miscompiled kernel) - disassemble the
generated kernel and look for structural issues (missing barriers, bad
instruction scheduling):

```bash
find bazel-testlogs -name '*.hsaco' -path '*<failing_target>*'
objdump -d <path-to.hsaco>
```

## Step 5 — Decide the outcome and report

Always report via exactly one comment (your *final* report - this is in
addition to the Step 1 preliminary comment, not a replacement for it):
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
3. `**Category:** <category>` - the label from Step 1 (or the corrected one
   from Step 2, if you changed it).
4. Then your actual report, as below.

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
