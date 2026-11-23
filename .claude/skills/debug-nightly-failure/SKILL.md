---
name: debug-nightly-failure
description: Hard-capped at 5 minutes - first checks whether this exact failure signature was already analyzed (by looking at the issue's comments) and, if so, does nothing further. Otherwise quickly categorizes (build/test/config/infra) and labels a single ROCm XLA nightly CI failure, checks whether a later upstream commit already fixed it, searches open upstream PRs for one that looks like it already addresses it, and does a fast code read for a likely root cause and possible fix. Full RBE reproduction/gdb debugging is optional and rarely fits the time budget. Never edits the issue body - posts a preliminary triage comment, then a final-report comment (with a RESOLVED banner at the top of that same comment if the failure turned out to be fixed), and rewrites the issue's generic placeholder title into a one-line summary of what actually broke now that it knows. Use when invoked by the "debug" job of rocm_infra_monitor.yml for one specific failed job/step.
argument-hint: "[ci-repo] [run-id] [job-id] [job-name] [step-name] [failure-url] [issue-number]"
context: fork
agent: general-purpose
allowed-tools: Bash(gh issue edit *), Bash(gh issue comment *), Bash(gh label create *), Bash(gh api *), Bash(gh run view *), Bash(gh search prs *), Bash(gh pr diff *), Bash(gh pr view *), Bash(git *), Bash(build_tools/rocm/*), Bash(bazel *), Bash(gdb *), Bash(objdump *), Bash(grep *), Bash(cat *), Bash(timeout *), Read, Grep, Glob, Write, Edit
---

# Debug a ROCm Nightly CI Failure

The commit that produced the failure is already checked out in the current
working directory - that is the repro environment for this whole skill.

## Arguments

The caller passes seven values, space separated: `CI_REPO RUN_ID JOB_ID
JOB_NAME STEP_NAME FAILURE_URL ISSUE_NUMBER`. `JOB_NAME` and `STEP_NAME` can
themselves contain spaces and parentheses (e.g. `execute-xla-ut-nightly
(therock_release)`), so do not try to split `$ARGUMENTS` positionally by
whitespace - the invoking prompt also restates each value on its own
labeled line; read the values from there.

- `CI_REPO`: the repository the failing run and the issue live in
  (`ROCm/xla`) - use this for every `gh api`/`gh issue` call, never
  `SOURCE REPO`.
- `RUN_ID` / `JOB_ID`: identify the failing job within `CI_REPO`.
- `FAILURE_URL`: link to the failing job, for reference in your report.
- `ISSUE_NUMBER`: the GitHub issue dedicated solely to this one failure
  signature (`JOB_NAME / STEP_NAME`), found or created for you before this
  skill was invoked. Its body is the static detection text (Job/Step/
  links, including the `<!-- rocm-nightly-failure:... -->` marker that
  identifies this issue to the pipeline) and you never touch it - Step 2
  posts a preliminary triage comment, Step 7 posts the final report as its
  own comment (prepending a RESOLVED banner to that same comment if the
  failure turned out to be fixed), and Step 7b retitles the issue itself
  (its title starts as a generic placeholder; that's the one thing besides
  comments you do change).

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
  edit the issue body**. The only GitHub writes you make are: one label
  (Step 2, corrected in Step 3 if needed, via `gh issue edit`/
  `gh label create`), two new comments on the `ISSUE_NUMBER` issue - Step
  2's preliminary triage comment and Step 7's final report (with a
  RESOLVED banner at the top of that same comment when the outcome is
  resolved) - both via `gh issue comment <ISSUE_NUMBER> --repo <CI_REPO>
  --body-file <file>`, and one title update (Step 7b, via `gh issue edit
  <ISSUE_NUMBER> --repo <CI_REPO> --title "..."`) - **or**, if Step 1 finds
  this was already analyzed,
  no GitHub writes at all.
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
gh api --paginate repos/<CI_REPO>/issues/<ISSUE_NUMBER>/comments
```

Check whether any comment's `body` already **contains**, anywhere,
`<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->` (substitute the
literal `JOB_NAME`/`STEP_NAME` values you were given, ` / ` separator, no
other change). If one does:
- **Stop here - do not repeat the analysis, and make no GitHub writes at
  all.** The full analysis is already sitting in that comment for anyone
  to read; there's nothing to add.

If the marker is absent from every comment, proceed to Step 2.

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

1. Ensure the label exists and apply it to the issue (idempotent - safe to
   run every time):
   ```
   gh label create "<category>" --repo <CI_REPO> --force \
     --color <hex> --description "<one-line description>"
   gh issue edit <ISSUE_NUMBER> --repo <CI_REPO> --add-label "<category>"
   ```
   Use `d73a4a` for `nightly-build-failure`, `fbca04` for
   `nightly-test-failure`, `0075ca` for `nightly-config-failure`, `5319e7`
   for `nightly-infra-failure`.
2. Post a preliminary triage comment - a plain new comment, nothing to
   fetch or merge first:
   ```
   gh issue comment <ISSUE_NUMBER> --repo <CI_REPO> --body-file /tmp/triage_comment.txt
   ```
   (if `/tmp` isn't writable by this skill's sandbox, write the file inside
   the checkout instead, e.g. `debug_tools/claude_scratch/`, cleaned up at
   the end same as Step 6's scratch files). The comment itself:
   - `**Category:** <category>`
   - `**Possible root cause (preliminary, unverified):** <1-3 sentences>` -
     a concrete guess from what the log shows so far (e.g. "the step hit a
     60-minute timeout with no output after the RBE remote_cache handshake,
     which looks like an RBE connectivity issue rather than a hang in the
     test itself"). This is a hypothesis, not a conclusion - say so, and
     don't present it as certain.
   - A closing line that full analysis is in progress and will follow as a
     separate comment when done.

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

Within your remaining time budget (roughly 2-3 minutes - see above), do
three cheap checks using just git, `gh`, and reading code. This is the
step that produces your estimate; by default you go straight from here to
Step 7.

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

**B. If nothing is merged yet, check for a relevant *open* PR upstream.**
Someone may already be mid-flight fixing this in `openxla/xla` - worth a
quick, targeted search before spending the remaining budget re-deriving
the same fix from scratch:
- Build 1-2 narrow search terms from the error signature (Step 3): the
  specific implicated file's basename, the failing function/class/test
  name, or a short distinctive fragment of the error text. Prefer a
  file/symbol match over free text - PR titles/descriptions don't always
  mention the exact runtime error.
- Search open PRs: `gh search prs --repo openxla/xla --state open
  --limit 5 "<query>"`. If that returns nothing useful, try one narrower
  or differently-worded query - don't iterate endlessly, this is a quick
  check, not an exhaustive one.
- For each candidate, read the title first; only pull the full diff
  (`gh pr diff <number> --repo openxla/xla`) for the one or two most
  promising ones. A good candidate actually touches the exact failing
  file/function/line, not just something nearby or a drive-by refactor.
- If you find one that plausibly fixes this exact failure: note it in your
  report (Step 7) - PR number, link, and a one-line reason it looks
  relevant - and say explicitly that it is **unmerged and unverified**:
  nobody should wait on or rely on it without checking its own CI status
  and actually confirming it resolves this specific failure. This doesn't
  change the category/outcome on its own (an open PR is not "already fixed
  upstream" - that's reserved for a *merged* commit from part A); treat it
  as supporting evidence for a "Possible fix (unverified)" report instead,
  alongside or in place of your own code-read hypothesis from part C.
- If nothing relevant turns up after checking the top few results, say so
  in one line and move on - don't pad the report with a non-finding.

**C. If nothing already fixes it (merged or open)**, analyze the code
directly:
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

## Step 7 — Decide the outcome and post the final report

Post your final report as a **new** comment - a separate comment from
Step 2's triage one, nothing to fetch or merge first:
```
gh issue comment <ISSUE_NUMBER> --repo <CI_REPO> --body-file /tmp/report_comment.txt
```
(same `/tmp`-not-writable fallback as Step 2: write it under
`debug_tools/claude_scratch/` if needed, cleaned up at the end.)

Build that comment as:

```
<RESOLVED banner here, ONLY if outcome is "Fix found" or "Already fixed
upstream" - see below - omit this block entirely otherwise>

<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->
**Category:** <category>
**Outcome:** <outcome>

<your actual report, as below>
```

For the marker, outcome line, and banner specifically:
- `<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->` - substitute
  the literal `JOB_NAME` and `STEP_NAME` values you were given, separated
  by ` / ` with no other change to the text. Step 1 (and the workflow's
  own pre-dispatch check) look for this marker **anywhere** in this
  comment's body - it doesn't need to be the first line, just present
  somewhere - so that this specific failure never gets debugged a second
  time. Leaving it off, or using the wrong job/step text, would defeat
  that check.
- `**Outcome:** <outcome>` - exactly one of the four literal strings
  `Fix found`, `Already fixed upstream`, `Possible fix (unverified)`,
  `Not resolved`, matching which of the four shapes below you're about to
  write. Get the literal string right - it's what a human (or
  `auto_verify`'s `required_marker` check) scans for to tell what
  happened without reading the whole report.
- **RESOLVED banner** - only when the outcome is `Fix found` or `Already
  fixed upstream` (the failure is genuinely fixed, not merely suspected):
  prepend this block, as the very first lines of the comment, so a human
  scanning the issue's comment list sees it without opening anything:
  ```
  > **RESOLVED** — <one-line summary of the fix, matching your report
  > below - reuse its own outcome heading verbatim where possible, don't
  > write a new one from scratch. If Step 4A found a fixing commit,
  > working a short reference to it into this same line reads better than
  > a separate link line>
  > React 👍 if you want this dropped, or just ignore it - it's done.

  ```
  (blank line after, before the marker/category/outcome block). For
  `Possible fix (unverified)` or `Not resolved`, omit this block entirely
  - the comment starts directly with the marker line.

Keep the comment reasonably tight: GitHub caps a single comment body at
65536 characters - trim log excerpts, diffs, and backtraces to the lines
that actually matter rather than pasting whole blocks.

**Already fixed upstream** (found in Step 4A): report with:
- The root cause, as understood from the fixing commit.
- The fixing commit's SHA, one-line summary, and the specific lines that
  fix it (quote the relevant part of `git show <sha>`).
- Whether you independently confirmed it (re-ran the originally-failing
  target at that commit and observed it pass) or are reporting this from
  the commit diff/message alone - say which, plainly.

**Possible fix (quick estimate, unverified)** - the expected outcome given
the 5-minute budget, when Step 4B found a relevant open PR, Step 4C found a
plausible culprit from reading the code, or both, but there was no time to
reproduce and confirm anything: report with:
- If Step 4C found a culprit: the file/function you suspect and why -
  point at the specific lines. The fix as a diff, from `git diff` if you
  wrote one, or described in words if you didn't have time to write it
  out.
- If Step 4B found a relevant open PR instead (or in addition): its number
  and link, a one-line reason it looks relevant, and an explicit note that
  it is unmerged and you did not verify it yourself - quote the specific
  part of its diff that addresses this failure rather than just a link.
- State plainly that this is an estimate - from reading the code, from an
  unverified open PR, or both - not a verified fix. Nobody should merge
  anything or rely on the open PR landing without reproducing/confirming
  first.
- End with a line telling a reviewer this will be verified automatically
  and how to ask for anything beyond that: `This will be verified
  automatically - no action needed for that. For anything else (a
  different approach, a deeper look, a specific thing to inspect), post a
  new comment on this issue starting with \`@claude <what you'd like
  investigated>\`.` rocm_infra_monitor.yml's `auto_verify` job picks up
  this exact "Possible fix (unverified)" outcome on its own, right after
  this report is posted - see that job and respond-to-mention/SKILL.md.
  rocm_infra_agent.yml's `@claude` mechanism (see
  `collect_claude_mention_targets.sh`) stays for anything beyond plain
  verification.

**Fix found** (you derived it yourself *and* verified it - only reachable
if Steps 4/5 actually ran and finished within budget): before reporting,
re-run the exact originally-failing target/test case one more time and
confirm it now passes. Then report with:
- A short root-cause summary.
- The fix as a diff, from `git diff`, in a \`\`\`diff fenced code block.
- The command you used to confirm it passes, and its outcome.

**Not resolved**: report with:
- The failure signature (crash / hang / wrong result / build error) and
  exactly which target/test case.
- What you ruled out and what you tried (be concrete - "ruled out X because
  Y", not "investigated further").
- The most useful evidence you gathered: a backtrace, a VLOG excerpt around
  the divergence point, or a kernel disassembly snippet - whichever applies.
- A concrete, specific suggestion for the next debugging step. Never end
  with a vague "could not reproduce" with no supporting detail.

Do **not** wrap anything in raw `~~strikethrough~~` for a resolved outcome
- GFM strikethrough is an inline construct and doesn't reliably span
headings or fenced code blocks, and the `Fix found` shape above has a
mandatory \`\`\`diff fenced block that would get garbled by it. The
RESOLVED banner above the comment already makes the resolution visible at
a glance in the issue's comment list; nothing else needs to change.

## Step 7b — Retitle the issue now that you know what actually broke

The issue was created with a generic placeholder title
(`Nightly CI Failure: JOB_NAME / STEP_NAME`) because nothing more was known
yet. Now that you've actually analyzed it, replace that with something a
human can recognize at a glance in an issue list, without opening it:

```
gh issue edit <ISSUE_NUMBER> --repo <CI_REPO> --title "<new title>"
```

`<new title>` is `<one-line summary of what actually broke, ≤80
characters> (JOB_NAME / STEP_NAME)` - lead with the summary (that's the
part worth scanning for; the job/step suffix is there for cross-referencing
with CI logs, not for skimming) and keep it concrete and specific to this
failure, e.g. `HSA frame-buffer vs. HIP VRAM size mismatch on gfx950
(execute-xla-ut-sanitizers (asan) / Test XLA [asan] [single_gpu])`, not a
restatement of the category label (`nightly-test-failure`) or something as
generic as "test failure". Base it on whatever you found in Steps 2-4,
regardless of outcome - even a `Not resolved` report usually has a
concrete symptom (a specific crash, a specific wrong value, a specific
timeout) worth naming, which is more useful here than the raw job/step
string alone.

This is always safe to do, however you phrase it: every lookup that needs
to find this issue again (Step 1 above, collect_claude_debug_targets.sh,
collect_claude_mention_targets.sh) keys off the
`<!-- rocm-nightly-failure:JOB_NAME / STEP_NAME -->` marker already sitting
in the issue's body, never the title - so rewriting the title here cannot
break anything downstream, including a later run finding this same issue
again for a recurrence of the same signature.
