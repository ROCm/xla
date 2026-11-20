---
name: debug-nightly-failure
description: Hard-capped at 5 minutes - first checks whether this exact failure signature was already analyzed (by looking inside the detection comment itself) and, if so, does nothing further. Otherwise quickly categorizes (build/test/config/infra) and labels a single ROCm XLA nightly CI failure, checks whether a later upstream commit already fixed it, and does a fast code read for a likely root cause and possible fix. Full RBE reproduction/gdb debugging is optional and rarely fits the time budget. Never posts a new comment - edits the single detection comment in place, folding the triage and final report behind a collapsible "Analysis (click to expand)" section so the comment stays short, and adding a RESOLVED banner above that fold if the failure turned out to be fixed. Use when invoked by the "debug" job of rocm_infra_monitor.yml for one specific failed job/step.
argument-hint: [ci-repo] [run-id] [job-id] [job-name] [step-name] [failure-url] [issue-number] [detection-comment-url]
context: fork
agent: general-purpose
allowed-tools: Bash(gh issue edit *), Bash(gh label create *), Bash(gh api *), Bash(gh run view *), Bash(git *), Bash(build_tools/rocm/*), Bash(bazel *), Bash(gdb *), Bash(objdump *), Bash(grep *), Bash(cat *), Bash(timeout *), Read, Grep, Glob, Write, Edit
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
  *this* failure, and the **only** comment this skill ever writes to. You
  never post a new comment - Step 2 appends a triage section into this
  comment's own body, and Step 7 folds that triage plus the final report
  behind a collapsible "Analysis (click to expand)" section in the same
  body, adding a RESOLVED banner above the fold in Step 8 if the failure
  turned out to be fixed. The numeric comment ID you need for
  `gh api .../issues/comments/<id>` is the part after `#issuecomment-` in
  this URL.

## IMPORTANT: scope and safety

- **This skill itself runs on the bare `build-only-xla` host, not inside any
  container.** A `rocmless` container named `rocmless-debug` is already
  running as a sidecar (the workflow started it) for the ROCm/Bazel
  toolchain - every `build_tools/rocm/run_xla_ci_build.sh`/plain `bazel`
  command you run **must** be wrapped: `docker exec rocmless-debug
  <command>`. Everything else - `git`, `gh`, reading/writing files (BUILD
  files, gdb scripts, reading `bazel-testlogs` output) - runs directly on
  the host; the workspace is bind-mounted into the container at the
  identical path, so files created either way are visible on both sides
  with no extra copying.
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
- **If the failing job used `--config=rocm_rbe`, keep using it, but override
  the EngFlow hostname** to strip the `internal.` prefix - add these four
  flags *after* `--config=rocm_rbe` on the command line (Bazel's
  single-valued flags take the last occurrence, so these override just the
  hostname and leave everything else `rocm_rbe` sets - certs, platform,
  spawn strategy - untouched):
  ```
  --remote_cache="grpcs://actinolite.cluster.engflow.com"
  --remote_executor="grpcs://actinolite.cluster.engflow.com"
  --bes_backend="grpcs://actinolite.cluster.engflow.com"
  --bes_results_url="https://actinolite.cluster.engflow.com/invocation/"
  ```
  `internal.actinolite.cluster.engflow.com` is not reachable from this debug
  runner (confirmed: `Connection timed out:
  internal.actinolite.cluster.engflow.com:443`, before running anything -
  tells you nothing about the failure you're debugging); the same cluster
  without the `internal.` prefix is. If the failing job used
  `--config=rocm_rbe_dev` instead (Buildbarn), leave it alone - this
  override only applies to `rocm_rbe`. See
  `build_tools/rocm/rocm_xla.bazelrc` in `openxla/xla` for the full set of
  flags each config defines, if you need to double check what else a
  config touches.
- Debug only this one job/step. Do not try to fix unrelated red steps you
  notice along the way.
- Never `git push`, never create a branch, never open a PR, and **never
  post a new comment on the tracking issue**. The only GitHub writes you
  make are: one label (Step 2, corrected in Step 3 if needed, via
  `gh issue edit`/`gh label create`), and two or three edits to the single
  existing `DETECTION_COMMENT_URL` comment - Step 2's triage section, Step
  7's fold-and-final-report, and Step 8's RESOLVED banner (only when the
  outcome is resolved) - all via
  `gh api -X PATCH repos/<CI_REPO>/issues/comments/<id> -F body=@<file>`
  (note `-F`, not `-f` - see Step 7 for why) - **or**, if Step 1 finds this
  was already analyzed, no GitHub writes at all.
- This run also has a hard cap on agentic turns, on top of the 5-minute
  wall clock. If you notice you're repeating the same approach without
  learning anything new (re-running the same build, trying the same gdb
  script a third time with no new evidence, going back and forth between
  two hypotheses), stop and go straight to Step 7 with whatever you have
  (a "Possible fix" estimate or a "Not resolved" report) - don't let the
  run end with nothing reported at all. A concrete partial finding beats
  silence.
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
gh api repos/<CI_REPO>/issues/comments/<comment-id-from-DETECTION_COMMENT_URL>
```

Check whether its `body` already **contains**, anywhere (not necessarily as
the first line - Step 2/7 append into this same comment, they don't start a
fresh one),
`<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->` (substitute the
literal `JOB_NAME`/`STEP_NAME` values you were given, ` / ` separator, no
other change). If it does:
- **Stop here - do not repeat the analysis, and make no GitHub writes at
  all.** The full analysis is already sitting in that same comment for
  anyone to read; there's nothing to add and no separate comment to point
  to.

If the marker is absent, proceed to Step 2.

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
explicitly in the triage section below.

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
2. Append a triage section into the existing `DETECTION_COMMENT_URL`
   comment - **never post a new comment**. Fetch its current body first
   (this is the raw detection text posted when the failure was first
   seen), then PATCH it back with your triage section appended:
   ```
   COMMENT_ID=<the numeric id after #issuecomment- in DETECTION_COMMENT_URL>
   gh api "repos/<CI_REPO>/issues/comments/${COMMENT_ID}" --jq .body > /tmp/comment_body.txt
   cat /tmp/comment_body.txt > /tmp/new_body.txt
   # append a divider, then the triage section, to /tmp/new_body.txt
   gh api -X PATCH "repos/<CI_REPO>/issues/comments/${COMMENT_ID}" -F body=@/tmp/new_body.txt
   ```
   Use `-F`, not `-f` (see Step 7 for why), and if `/tmp` isn't writable by
   this skill's sandbox, use a path inside the checkout instead - same
   caveat as Step 7.
   Keep the fetched body byte-for-byte at the top - Step 7 fetches and
   appends again later, so never drop or rewrite what's already there.
   The appended section itself:
   - A blank line, then `---`, then a blank line (visual divider from the
     original detection text).
   - `**Category:** <category>`
   - `**Possible root cause (preliminary, unverified):** <1-3 sentences>` -
     a concrete guess from what the log shows so far (e.g. "the step hit a
     60-minute timeout with no output after the RBE remote_cache handshake,
     which looks like an RBE connectivity issue rather than a hang in the
     test itself"). This is a hypothesis, not a conclusion - say so, and
     don't present it as certain.
   - A closing line that full analysis is in progress and will be appended
     into this same comment when done.

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
  - **`--config=rocm_rbe`.** Keep it, but append the four host-override
    flags from the safety rule above (strip the `internal.` prefix from
    `actinolite.cluster.engflow.com`) - see that rule for the exact flags
    and why. This applies to every bazel invocation anywhere in this
    skill, same as the single-target rule.
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
`--test_filter=`/`--test_arg=--gtest_filter=` for one test case), wrapped in
`docker exec rocmless-debug ...` (see above - this skill runs on the bare
host, the ROCm/Bazel toolchain only exists inside that container), adding:
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

**Crash / assertion failure** - run under gdb and capture a backtrace. The
`cat`/`chmod` steps above run directly on the host (bind-mounted into the
container already); only the actual bazel invocation needs `docker exec`:

```bash
docker exec rocmless-debug \
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
# re-run the same `docker exec rocmless-debug build_tools/rocm/run_xla_ci_build.sh ...` command as above
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

Append your final report into the same `DETECTION_COMMENT_URL` comment
Step 2 already edited once - **never post a new comment**. This is also
where the comment gets folded: the detection header (Job/Step/links)
stays visible, but everything analytical - Step 2's triage plus your
report below - moves behind a single collapsible `<details>` block, so
the comment stays short by default and a reader only expands it if they
want the detail.

Fetch the comment's current body (now the original detection text plus
Step 2's triage section, still plain at this point), then restructure it:

```
COMMENT_ID=<the numeric id after #issuecomment- in DETECTION_COMMENT_URL>
gh api "repos/<CI_REPO>/issues/comments/${COMMENT_ID}" --jq .body > /tmp/comment_body.txt
```

Find the first `\n\n---\n\n` divider in that body - the boundary Step 2
inserted between the original detection text and its own triage section.
Split there: everything **before** it (the `## Nightly CI Failure
Detected` block - Job/Step/links) is the part that stays outside the
fold, byte-for-byte unchanged, so a reader sees at a glance what failed
and where without expanding anything. Everything **from that divider
onward** (Step 2's triage section) moves inside the fold together with
your new report. Build the new body as:

```
<everything before the first "---" divider, byte-for-byte>

<details><summary>Analysis (click to expand)</summary>

<everything from that divider onward - Step 2's triage section, unchanged>

---

<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->
**Category:** <category>
**Outcome:** <outcome>

<your actual report, as below>

</details>
```

then PATCH it back:
```
gh api -X PATCH "repos/<CI_REPO>/issues/comments/${COMMENT_ID}" -F body=@/tmp/new_body.txt
```
Use `-F`, not `-f` - only `-F`/`--field` reads an `@file` argument from
disk; `-f`/`--raw-field` treats `@file` as a literal string and silently
sets the comment body to that text. Also note `/tmp` may not be writable
by this skill's sandbox in every environment - if `gh api -F body=@/tmp/...`
fails to find the file, write the scratch files inside the current
checkout instead (for example `debug_tools/claude_scratch/`, cleaned up
at the end same as Step 6's scratch files) and reference that path
instead of `/tmp`.

For the marker and outcome line specifically:
- `<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->` - substitute
  the literal `JOB_NAME` and `STEP_NAME` values you were given, separated
  by ` / ` with no other change to the text. Step 1 (and the workflow's
  own pre-dispatch check) look for this marker **anywhere** in this
  comment's body - it doesn't need to be the first line, just present
  somewhere - so that this specific failure never gets debugged a second
  time even though the tracking issue is shared by many different
  failures. Leaving it off, or using the wrong job/step text, would
  defeat that check.
- `**Outcome:** <outcome>` - exactly one of the four literal strings
  `Fix found`, `Already fixed upstream`, `Possible fix (unverified)`,
  `Not resolved`, matching which of the four shapes below you're about to
  write. Get the literal string right - Step 8 greps for it to decide
  whether to cross the comment out.

Keep the appended text reasonably tight: GitHub caps a single comment body
at 65536 characters, and this one comment now carries the detection text,
the triage section, and the final report together - trim log excerpts,
diffs, and backtraces to the lines that actually matter rather than
pasting whole blocks.

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

## Step 8 — Cross out the comment if resolved

If Step 7's outcome was `Fix found` or `Already fixed upstream` (the
failure is genuinely fixed, not merely suspected), the comment should
visibly read as resolved so a human can tell at a glance it needs no
attention and drop it from their mental todo list - this step does that
in place, right after Step 7's edit. If the outcome was
`Possible fix (unverified)` or `Not resolved`, skip this step entirely and
leave the comment exactly as Step 7 left it (already folded behind
"Analysis (click to expand)", just with no banner on top).

Do **not** wrap anything in raw `~~strikethrough~~` - GFM strikethrough is
an inline construct and doesn't reliably span headings or fenced code
blocks, and the `Fix found` shape above has a mandatory \`\`\`diff fenced
block that would get garbled by it. Step 7 already folded the analysis
behind a `<details>` block, so this step only needs two small, surgical
edits on top of that - fetch the comment's current full body one more
time and make both:

1. In the part *outside* the fold (the detection header), change
   `## Nightly CI Failure Detected` to
   `## Nightly CI Failure Detected (Resolved)` - the one deliberate edit
   to that otherwise-untouched text, so the resolved state is visible even
   to a reader who expands the `<details>` block without first reading the
   banner below.
2. Immediately above the `<details><summary>Analysis (click to
   expand)</summary>` line Step 7 wrote, insert this banner (don't touch
   the `<details>` tag itself or anything inside it):
   ```
   > **RESOLVED** — <one-line summary of the fix, matching Step 7's report -
   > reuse the report's own outcome heading verbatim where possible, don't
   > write a new one from scratch. If Step 4A found a fixing commit/PR,
   > working a short reference to it into this same line reads better than
   > a separate link line>
   > React 👍 if you want this dropped, or just ignore it - it's done.
   ```

PATCH the result back. This keeps the banner (and the react prompt) always
visible without expanding anything, while the full analysis - detection
triage and report alike - stays one click away behind the same fold
Step 7 already built. That's what actually lets a human drop it from
attention; raw strikethrough would still show all the clutter, just with
a line through it.
