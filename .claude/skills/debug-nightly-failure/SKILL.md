---
name: debug-nightly-failure
description: Hard-capped at 5 minutes - first checks whether this exact failure signature was already analyzed on the tracking issue and, if so, just posts a reference to that instead of repeating the work. Otherwise quickly categorizes (build/test/config/infra) and labels a single ROCm XLA nightly CI failure, checks whether a later upstream commit already fixed it, and does a fast code read for a likely root cause and possible fix. Full RBE reproduction/gdb debugging is optional and rarely fits the time budget. Reports a quick estimate (or an already-found fix, or a callstack/analysis) directly on the tracking GitHub issue. Use when invoked by the "debug" job of rocm_infra_monitor.yml for one specific failed job/step.
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
  *this* failure. Quote it in both comments you post (Steps 1 and 6) so a
  reader can tell which of the issue's many comments your analysis
  addresses.

## IMPORTANT: scope and safety

- **Steps 1-3 (triage and analysis, before any build) are hard-capped at 5
  minutes of wall-clock time** - budget roughly 1 minute each for Steps 1
  and 2, 2-3 minutes for Step 4. Building/testing afterward (Step 5+)
  doesn't count against that 5 minutes, but the whole run still has to fit
  inside this step's own (longer) timeout, so treat Steps 4 and 5 (full RBE
  reproduction, gdb) as something you only start if you're confident, from
  Step 4 alone, that a build+test will complete and you'll still have time
  to post Step 7's report afterward. Default to going straight from Step 4
  to Step 7 with your best estimate.
- This runner (`build-only-xla`) has no local GPU - it's a network-access
  node, not a GPU node. Every Bazel/`run_xla_ci_build.sh` invocation you run,
  anywhere (verifying an already-found fix in Step 4, reproducing in Step 5,
  debugging in Step 6) **must** include `--strategy=TestRunner=remote,local`,
  so GPU test actions execute on a remote RBE worker from the GPU pool
  instead of trying (and failing) to run on this machine; `local` is kept
  only as a fallback for actions that don't need a GPU. Omitting this flag
  means your test actions will not run at all.
- **Never build or test the full `//xla/...` sweep the original CI step
  used.** Always replace that target list with just the single failing
  target (or, for a test binary, `--test_filter=`/`--test_arg=
  --gtest_filter=` narrowed to the one failing test case) - every bazel
  invocation you run, anywhere in this skill, builds/tests exactly one
  target, never the whole tree. This applies even when "reproducing the
  exact command" in Step 3/4: match the flags verbatim, but always swap the
  target list for the one target you're debugging.
- **Never use `--config=rocm_rbe`, even if the failing job's own log did.**
  Always use `--config=rocm_rbe_dev` instead - `rocm_rbe` is not reachable
  from this debug runner (it fails with `Connection timed out:
  internal.actinolite.cluster.engflow.com:443` before running anything,
  which tells you nothing about the failure you're debugging). Applies to
  every bazel invocation anywhere in this skill.
- Debug only this one job/step. Do not try to fix unrelated red steps you
  notice along the way.
- Never `git push`, never create a branch, never open a PR. The only
  GitHub writes you make are: one label (Step 2, corrected in Step 3 if
  needed), one preliminary comment (Step 2), and one final report comment
  (Step 7) - all via `gh issue edit`/`gh issue comment <ISSUE_NUMBER> --repo <CI_REPO>`/`gh label create` - **or**, if Step 1 finds this was already
  analyzed, just the one short reference comment Step 1 describes, nothing
  else.
- This run also has a hard cap on agentic turns, on top of the 5-minute
  wall clock. If you notice you're repeating the same approach without
  learning anything new (re-running the same build, trying the same gdb
  script a third time with no new evidence, going back and forth between
  two hypotheses), stop and go straight to Step 7 with whatever you have
  (a "Possible fix" estimate or a "Not resolved" report) - don't
  posted at all. A concrete partial finding beats silence.
- Clean up any scratch files before finishing - nothing from this debugging
  session is meant to be committed. This includes `/tmp/gdb_*`, extra build
  outputs, and - unlike `/tmp` - specifically `debug_tools/claude_scratch/`
  (Step 6's run_under wrapper target), since that one lives inside the git
  working tree: `rm -rf debug_tools/claude_scratch`.
- Never claim a fix works without having re-run the originally-failing
  target and observed it pass.

## Step 1 — Check whether this was already analyzed

Before anything else: the workflow itself already checks this once before
dispatching you, but check again here too, in case of a race (two runs
queuing the same signature close together) or a manual re-run landing you
here anyway.

```
gh api repos/<CI_REPO>/issues/<ISSUE_NUMBER>/comments --paginate
```

Look for a comment whose body starts with exactly
`<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->` (substitute the
literal `JOB_NAME`/`STEP_NAME` values you were given, ` / ` separator, no
other change). If one exists:
- **Stop here - do not repeat the analysis.** Post one short comment
  instead: quote the detection comment (same method as Step 7.2 below),
  then a line or two noting this exact failure was already analyzed, with
  a link to the existing report comment (`[Previous analysis](<url>)`).
  Do **not** start this comment with the `-claude-debug:` marker - it isn't
  a new completed analysis, just a pointer to one, and giving it the same
  marker would make a future check here ambiguous about which comment is
  the real one.
- Then stop - do not proceed to Step 2.

If no such comment exists, proceed to Step 2.

## Step 2 — Triage: categorize and label

Fetch the failing step's raw log once, before anything else - Step 3 reuses
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
   report in Step 7) via
   `gh issue comment <ISSUE_NUMBER> --repo <CI_REPO> --body-file <file>`:
   - First line exactly `<!-- rocm-nightly-claude-triage:JOB_NAME / STEP_NAME -->`
     (substitute the literal values, ` / ` separator, no other change). This
     is a *different* marker from the Step 7 completion marker and is never
     checked for dedup - it only marks this comment as the triage pass for
     this signature, so post it even if you end up unable to finish the
     full analysis.
   - Quote the detection comment (same method as Step 7.2 below).
   - `**Category:** <category>`
   - `**Possible root cause (preliminary, unverified):** <1-3 sentences>` -
     a concrete guess from what the log shows so far (e.g. "the step hit a
     60-minute timeout with no output after the RBE remote_cache handshake,
     which looks like an RBE connectivity issue rather than a hang in the
     test itself"). This is a hypothesis, not a conclusion - say so, and
     don't present it as certain.
   - A closing line that full reproduction is starting and a complete
     report will follow in a separate comment.

## Step 3 — Understand what actually failed

Reread `job.log` (already fetched in Step 2) and extract:
- The exact `build_tools/rocm/run_xla_ci_build.sh ...` (or plain `bazel`)
  invocation that was run for the failing step. Match its *flags* verbatim
  rather than guessing matrix-specific ones (distro URL/hash, `gfx` target)
  - different nightly matrix legs use different values, and the log
  already has the real ones. Two exceptions:
  - **The target list.** See the "never build the full sweep" rule above:
    that invocation's own `-- //xla/...` covers the whole tree, and you
    always replace it with just the one failing target, in Step 3 or
    anywhere else.
  - **`--config=rocm_rbe`.** Always replace it with
    `--config=rocm_rbe_dev`, even if the failing job's own log used
    `rocm_rbe` - `rocm_rbe` is not reachable from this debug runner
    (confirmed: it fails with `Connection timed out:
    internal.actinolite.cluster.engflow.com:443`, not a real result about
    the failure you're debugging) and the build would never get as far as
    running anything. `rocm_rbe_dev` is. This applies to every bazel
    invocation anywhere in this skill, same as the single-target rule.
- The specific failing target(s) and, for a test, the specific failing test
  case/parameter - this is what you narrow every later bazel invocation to.
- The error signature: a crash with a stack trace, a hang/timeout, a wrong
  numerical result, a build error, etc. This determines which debugging
  technique in Step 6 applies, and whether Step 2's category guess holds up
  - if deeper reading changes your mind about the category, update the
  issue's labels (`gh issue edit --remove-label <old> --add-label <new>`)
  before finishing.

## Step 4 — Check for an existing fix, then analyze the code

Within your remaining time budget (roughly 2-3 minutes - see above), do two
cheap checks using just git and reading code. This is the step that
produces your estimate; by default you go straight from here to Step 7.

**A. Check whether it's already fixed.** The checkout is a shallow clone of
the exact failing commit, so get a bounded amount of extra history first:
`git fetch origin <main-or-relevant-branch> --depth=200`. Do **not** run
`git fetch --unshallow` - on a repo this size that alone can burn the
entire time budget; if 200 isn't enough to reach a commit you need, that's
a sign to stop digging and report with what you have. Then:
- List what changed in the implicated file(s) since the failing commit:
  `git log <HEAD_SHA>..origin/<branch> --oneline -- <file(s)>`.
- Read the diff of any commit in that list that plausibly touches the
  failing area (`git show <sha>`) - a commit message mentioning the same
  test, the same crash, or the same symptom is a strong signal.
- If you find a commit that clearly fixes this exact failure: say so
  explicitly in your report (cite the commit SHA, its one-line summary, and
  the specific lines that fix it) instead of re-deriving the same fix from
  scratch. If time/turn budget allows, confirm it by checking out that
  commit and re-running the originally-failing target to observe it pass;
  state plainly in the report whether you verified this or are reporting it
  from the diff/commit message alone.
- Also compare against **past** occurrences: fetch this issue's existing
  comments (`gh api repos/<CI_REPO>/issues/<ISSUE_NUMBER>/comments`) and
  look for earlier detection comments carrying this exact signature's
  marker (`<!-- rocm-nightly-failure:JOB_NAME / STEP_NAME -->`). A failure
  seen for the first time just now is a fresh regression; one that recurred
  over many prior runs is a standing issue - say which in your report, it
  changes how urgent/novel the finding is.

**B. If nothing already fixes it**, analyze the code directly:
- From the error signature (Step 3): identify the specific file(s)/
  function(s) implicated - a stack trace frame, the HLO pass or emitter
  named in the error, the assertion that fired, or (for a test failure) the
  test's own source and the production code it exercises.
- Read those files. Look for something that would produce *this exact*
  symptom: a logic error (wrong comparison, off-by-one, inverted
  condition), a missing case in a switch/dispatch, a CUDA-only code path
  with no ROCm equivalent, a shape/dtype/type mismatch, a recently-changed
  default the failing configuration doesn't account for.
- A clear culprit found here is still a *hypothesis*, not a verified fix -
  report it as your "possible fix" estimate in Step 7 (see below) rather
  than spending the remaining budget trying to reproduce and verify it.
  Only attempt Steps 4/5 if you're confident, right now, that a build+test
  will complete and you'll still have time left to post Step 7's report -
  this will rarely be true within 5 minutes total.
- If nothing in the code is a clear enough candidate after a genuine read
  (don't force a conclusion from a shallow skim, but don't keep reading
  indefinitely either - see the self-limiting rule above), say so plainly
  in Step 7 instead of reproducing from scratch.

## Step 5 (optional, rarely reached) — Reproduce it, narrowed to one target

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

## Step 6 (optional, rarely reached) — Debugging techniques

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
the error signature from Step 3), as explored in the scratch
`build_debug.sh` for an autotuner/codegen hang.

**GPU-codegen bugs** (wrong result, miscompiled kernel) - disassemble the
generated kernel and look for structural issues (missing barriers, bad
instruction scheduling):

```bash
find bazel-testlogs -name '*.hsaco' -path '*<failing_target>*'
objdump -d <path-to.hsaco>
```

## Step 7 — Decide the outcome and report

Always report via exactly one comment (your *final* report - this is in
addition to the Step 2 preliminary comment, not a replacement for it):
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
3. `**Category:** <category>` - the label from Step 2 (or the corrected one
   from Step 3, if you changed it).
4. Then your actual report, as below.

**Already fixed upstream** (found in Step 4A): comment with:
- The root cause, as understood from the fixing commit.
- The fixing commit's SHA, one-line summary, and the specific lines that
  fix it (quote the relevant part of `git show <sha>`).
- Whether you independently confirmed it (re-ran the originally-failing
  target at that commit and observed it pass) or are reporting this from
  the commit diff/message alone - say which, plainly.
- If this signature has recurred across many prior runs (Step 4A's past-
  occurrences check), mention that too - it means the fix just hasn't
  reached this configuration/branch yet, not that it was never found.

**Possible fix (quick estimate, unverified)** - the expected outcome given
the 5-minute budget, when Step 4B found a plausible culprit but there was
no time to reproduce and confirm it: comment with:
- The file/function you suspect and why - point at the specific lines.
- The fix as a diff, from `git diff` if you wrote one, or described in
  words if you didn't have time to write it out.
- State plainly that this is an estimate from reading the code, not a
  verified fix - nobody should merge it without reproducing first.

**Fix found** (you derived it yourself *and* verified it - only reachable
if Steps 4/5 actually ran and finished within budget): before reporting,
re-run the exact originally-failing target/test case one more time and
confirm it now passes. Then comment with:
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
