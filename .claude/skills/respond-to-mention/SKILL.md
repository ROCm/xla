---
name: respond-to-mention
description: Hard-capped at 30 minutes of non-build decision-making - triggered when a human posts a new `@claude <instructions>` comment on a per-failure issue debug-nightly-failure created, asking Claude to act on that failure - verify the already-proposed fix, dig further (e.g. set a breakpoint and inspect), try a different approach, or devise a new solution, per whatever the instructions literally say. When just asked to verify, reproduces the original failure under heavy repetition (nightly GPU CI can be flaky), applies the proposed fix, confirms it goes green, reverts the fix, and confirms it goes red again - a differential check that the fix, not noise, is responsible. If it ends up with a change it's confident actually fixes the problem, writes it out as a plain patch file (plus a PR title/body) for a deterministic workflow step to apply onto ROCm/xla's own default branch (main) and open a draft "[WIP] ..." PR with openxla/xla's own PR template filled in - never commits, pushes, or opens the PR itself. Posts its findings (noting a verified fix is pending a PR, a plain "not applicable" statement, or whatever was actually investigated) as a new comment on the issue, and never repeats the exact same instruction once attempted. Use when invoked by the "respond_to_mention" job of rocm_infra_monitor.yml for one specific comment carrying an `@claude` instruction.
argument-hint: "[ci-repo] [issue-number] [job-name] [step-name] [repository] [head-sha] [failure-url] [instructions] [instructions-hash]"
context: fork
agent: general-purpose
allowed-tools: Bash(gh issue edit *), Bash(gh issue comment *), Bash(gh label create *), Bash(gh api *), Bash(gh run view *), Bash(git *), Bash(docker exec rocmless-debug *), Bash(build_tools/rocm/*), Bash(bazel *), Bash(gdb *), Bash(objdump *), Bash(grep *), Bash(cat *), Bash(timeout *), Read, Grep, Glob, Write, Edit
---

# Respond to an @claude Follow-Up on a ROCm Nightly CI Failure

The commit that originally failed is already checked out in the current
working directory - that is both the repro environment and the base you
eventually branch from if you end up with a verified fix.

## Arguments

The caller passes nine values, space separated: `CI_REPO ISSUE_NUMBER
JOB_NAME STEP_NAME REPOSITORY HEAD_SHA FAILURE_URL INSTRUCTIONS
INSTRUCTIONS_HASH`. `JOB_NAME`, `STEP_NAME`, and especially `INSTRUCTIONS`
can themselves contain spaces, punctuation, and quotes, so do not split
`$ARGUMENTS` positionally by whitespace - the invoking prompt also restates
each value on its own labeled line; read the values from there.

- `CI_REPO`: `ROCm/xla` - the repo the issue lives in, and the only repo
  the patch you write in Step 5 is ever meant to land in (applied and
  turned into a PR by a later, deterministic workflow step - you don't
  push or open anything yourself), regardless of what `REPOSITORY` says
  (see below).
- `ISSUE_NUMBER`: the issue dedicated to this one failure signature - you
  never edit its body (same rule as `debug-nightly-failure`), only ever
  post a new comment with your findings. The proposed fix you're asked to
  verify usually already lives in one of `debug-nightly-failure`'s own
  comments (its final-report comment, identifiable by the
  `rocm-nightly-claude-debug:` marker and a `**Outcome:**` line) - fetch
  **all** of the issue's comments (Step 1) rather than just its body.
- `JOB_NAME` / `STEP_NAME`: the failure signature, used for the mention
  marker below.
- `REPOSITORY` / `HEAD_SHA`: already checked out in the working directory -
  this is almost always `openxla/xla@<sha>`, **not** `ROCm/xla` (see the
  PR-target rule in Step 5).
- `FAILURE_URL`: the original failing job, for re-fetching its log in Step
  1 if `debug-nightly-failure`'s report comment doesn't already spell out
  the exact command.
- `INSTRUCTIONS`: the literal text a human typed after `@claude` in a new
  comment on the issue - this is what actually governs what you do in Step
  1. Follow it specifically; don't substitute your own idea of what would
  be useful.
- `INSTRUCTIONS_HASH`: a short hash of `INSTRUCTIONS`, already computed for
  you - use it verbatim in the Step 0/Step 6 marker, don't recompute it.

## IMPORTANT: scope and safety

- **This skill itself runs on the bare `build-only-xla` host, not inside any
  container**, exactly like `debug-nightly-failure`. A `rocmless` container
  named `rocmless-debug` is already running as a sidecar - every
  `build_tools/rocm/run_xla_ci_build.sh`/plain `bazel` command you run
  **must** be wrapped: `docker exec rocmless-debug <command>`. Everything
  else (`git`, `gh`, `gdb`, reading/writing files) runs directly on the
  host; the workspace is bind-mounted into the container at the identical
  path.
- **Step 1 (reading the instructions and deciding how to proceed, before
  any build) is hard-capped at 30 minutes of wall-clock time.** The
  build+test cycles in Steps 2-4 (or whatever custom investigation
  `INSTRUCTIONS` calls for) don't count against that, but the whole run
  still has to fit inside this step's own (much longer) job timeout.
- This runner has no local GPU. Every Bazel/`run_xla_ci_build.sh`
  invocation, with no exception, **must** include
  `--strategy=TestRunner=remote,local`, or your test actions won't run at
  all.
- **Never build or test the full `//xla/...` sweep.** Every bazel
  invocation in this skill narrows to the single originally-failing target
  (or `--test_filter=`/`--test_arg=--gtest_filter=` for one test case) -
  same rule as `debug-nightly-failure`.
- **If the failing job used `--config=rocm_rbe`, keep it but override the
  EngFlow hostname** (strip the `internal.` prefix - this debug runner can't
  reach `internal.actinolite.cluster.engflow.com`, confirmed timeout):
  ```
  --remote_cache="grpcs://actinolite.cluster.engflow.com"
  --remote_executor="grpcs://actinolite.cluster.engflow.com"
  --bes_backend="grpcs://actinolite.cluster.engflow.com"
  --bes_results_url="https://actinolite.cluster.engflow.com/invocation/"
  ```
  Leave `--config=rocm_rbe_dev` (Buildbarn) alone - this override only
  applies to `rocm_rbe`.
- Only act on this one signature, per `INSTRUCTIONS` specifically. Do not
  go looking for other red steps or other comments on the issue.
- **You never run `git push`, `git commit`, or `gh pr create` yourself -
  not as a shortcut, not even to "double check" something.** A
  deterministic workflow step does all three, from the patch file (plus
  PR title/body) you write in Step 5 - see that step for why (a hard
  Claude Code permission restriction, not a style choice). That step only
  ever applies your patch to, and opens a PR against, `ROCm/xla`'s own
  default branch (`main`) specifically - never `openxla/xla` directly, even
  though `REPOSITORY` almost always is `openxla/xla` (there's no push
  access upstream at all). Nothing you do in this skill should assume
  otherwise: never try to push anywhere yourself, never force-push, never
  cause more than one PR to be opened, and the resulting PR is always a
  draft, never merged by anything in this pipeline.
- The only other GitHub write is one new comment on the `ISSUE_NUMBER`
  issue (Step 6, your findings) via `gh issue comment <ISSUE_NUMBER>
  --repo <CI_REPO> --body-file <file>`. **Never edit the issue body or an
  existing comment.** If Step 0 finds this exact instruction was already
  acted on, make no GitHub writes at all.
- Use `--runs_per_test=20 --test_arg=--gtest_repeat=50
  --nocache_test_results` (20 independent RBE sandboxes × 50 in-process
  repeats = 1000 total executions) for every reproduce/verify attempt by
  default - this is what rules out "it just happened to pass this run" on
  inherently flaky nightly GPU CI. Only scale it down if the target is slow
  enough that three cycles at this volume won't fit the job's timeout, and
  say so plainly in the report (Step 6) if you do.
- Never claim a fix is verified without having completed the full
  red→green→red cycle yourself in this run (not from the original issue's
  own, earlier, unverified claim), and never write the Step 5 patch file
  (which leads directly to a PR) without having personally observed it go
  green.
- If you bail out partway through (budget exhausted, build infra failure,
  etc.), leave the working tree exactly as Step 1 found it (no code
  changes applied) before reporting - a partial attempt with no code
  applied is always safer than one that silently leaves an untested change
  in place.

## Step 0 — Check whether this exact instruction was already acted on

Before anything else, in case of a race (two runs queuing the same
signature/instruction close together) or a manual re-run:

```
gh api --paginate repos/<CI_REPO>/issues/<ISSUE_NUMBER>/comments
```

Check whether any comment's `body` already **contains**, anywhere,
`<!-- rocm-nightly-claude-mention:JOB_NAME / STEP_NAME:INSTRUCTIONS_HASH -->`
(substitute the literal `JOB_NAME`/`STEP_NAME`/`INSTRUCTIONS_HASH` values you
were given, ` / ` separator before the hash, no other change). If one does:
**stop here, make no GitHub writes at all** - this exact instruction has
already been attempted once, successfully or not, and is never retried. If
a *different* `rocm-nightly-claude-mention:JOB_NAME / STEP_NAME:...` marker
is present instead (a different hash), that's a previous, different
instruction that was already handled - leave its comment alone and proceed
to act on the current one. If no marker for this signature is present at
all, proceed to Step 1. Keep the fetched comments around - Step 1 reads
them too.

## Step 1 — Read the instructions and decide how to proceed

Read `INSTRUCTIONS` carefully - it governs everything that follows. Find
`debug-nightly-failure`'s final-report comment among the ones Step 0 just
fetched (the one with the `<!-- rocm-nightly-claude-debug:... -->` marker
and a `**Outcome:**` line) - that comment's analysis is what Step 1-4
below work from.

- **If `INSTRUCTIONS` is simply asking you to verify/try the fix that
  report already proposes** (e.g. "verify this", "try the fix", "does this
  work?") and doesn't call for anything more specific: find its "Possible
  fix (quick estimate, unverified)" or "Fix found" section (whichever is
  present - `debug-nightly-failure` only ever writes one of the two), then
  follow the default method in Steps 2-4 below.
  - **If it has a \`\`\`diff\`\`\` fenced block**: that's the fix. You'll
    apply it with `git apply` in Step 3.
  - **If it's described in prose only** (the "Possible fix" shape allows
    this when the original analysis ran out of time to write the diff
    out): read the description and the file/function it points at, and
    implement the change yourself via `Edit`.
- **If `INSTRUCTIONS` asks for something else** - setting a breakpoint and
  inspecting specific state, digging into a particular hypothesis, trying
  a different approach than what's in that report, re-deriving root cause,
  or anything else a human might reasonably ask for - follow those literal
  instructions instead, not the default reproduce/apply/revert protocol.
  `gdb`/`objdump` are available for this (e.g. `gdb --batch -ex 'break
  <file>:<line>' -ex run -ex bt -ex quit --args <binary> <args>` inside
  `docker exec rocmless-debug ...`, or against a host-built binary if the
  investigation doesn't need the ROCm toolchain). This may end with no code
  change at all (pure investigation - skip straight to Step 6 with your
  findings), with a fix you then still need to validate causally (continue
  into Steps 2-4, adapted to whatever you changed), or with something in
  between. Use your judgment on how much of Steps 2-4's rigor still applies
  - the heavy-repetition causality check matters most when you're about to
  write a Step 5 patch and claim it's verified; it matters less for a
  request that's purely investigative.

Also recover the exact original invocation: the specific failing
target/test case and the full `build_tools/rocm/run_xla_ci_build.sh`/
`bazel` command and flags. That report comment usually already states this
plainly; if it doesn't spell out the flags verbatim, re-fetch the original
job's log instead of guessing:
```
gh api --allow-escape-sequences "repos/<CI_REPO>/actions/jobs/<job-id-from-FAILURE_URL>/logs" > job.log
```
(the numeric job ID is the part after `/job/` in `FAILURE_URL`).

Do not touch the working tree until you've decided your approach - Step 2
(if you follow the default method) needs to reproduce the *original*,
unmodified failure first.

## Step 2 — Reproduce red (baseline, fix not yet applied)

*(Default method, used when Step 1 determined `INSTRUCTIONS` is a plain
verify ask. If you're following a custom investigation instead, adapt or
skip this and the next two steps as appropriate, then go to Step 5/6.)*

Run the extracted command, narrowed to the single failing target, wrapped
in `docker exec rocmless-debug ...`, with the heavy-repetition flags from
the safety rules above added:
```
docker exec rocmless-debug \
  build_tools/rocm/run_xla_ci_build.sh <...same flags as the failing step, with the rocm_rbe host override if applicable...> \
  --strategy=TestRunner=remote,local \
  --runs_per_test=20 --test_arg=--gtest_repeat=50 --nocache_test_results \
  -- //path/to:failing_target
```

- **If it reproduces red** (at least one failure across all those
  repeats): proceed to Step 3.
- **If it does not reproduce at all**, even at this volume: there's
  nothing to differentially validate a fix against - the original failure
  may already be gone, or was pure flakiness to begin with. Stop here, skip
  straight to Step 6, and report "Not applicable - could not reproduce the
  original failure even without the fix, across N repeats."

## Step 3 — Apply the fix, confirm green

Apply the diff (`git apply`) or the code change you wrote in Step 1, then
re-run the **exact same** heavily-repeated command from Step 2.

- **If it comes back fully green** (zero failures across every repeat):
  proceed to Step 4.
- **If it's still red at all** (even partially): the fix doesn't resolve
  it. Revert whatever you applied (`git checkout -- <file>` or `git apply
  -R`) so the tree is clean, then stop here, skip straight to Step 6, and
  report "Not applicable - the fix was applied but the target still
  failed (<M> of <N> repeats) under the same repeated run."

## Step 4 — Revert the fix, confirm red again

This is the causality check: a single green run could just be RBE
worker/cache variance, not the fix. Revert exactly what Step 3 applied
(`git checkout -- <file>` for a hand-written change, `git apply -R` for a
diff), then re-run the same heavily-repeated command once more.

- **If it comes back red again** (at least one failure): the fix is
  verified. Re-apply it (undo this revert) and proceed to Step 5.
- **If it stays green even without the fix**: the earlier green in Step 3
  wasn't actually caused by the fix - don't trust it. Leave the fix
  unapplied (working tree clean), stop here, skip straight to Step 6, and
  report "Not applicable - still green even after reverting the fix, so
  the green in Step 3 wasn't caused by it (worker/cache noise, or the
  original failure itself was flaky)."

## Step 5 — If you have a verified fix: write it out as a patch for the workflow to apply

Only reached if you end up with a code change you're confident actually
fixes the problem - via the default Step 2-4 cycle confirming red→green→red,
or via whatever causality check made sense for a custom investigation. Not
reached at all for a purely investigative ask with no code change (go
straight to Step 6 instead). The fix is currently applied in the working
tree, and nothing has been committed. **You do not commit, push, or run
`gh pr create` yourself** - that is a hard rule, not a simplification of
convenience. The first time this skill tried to push/open a PR directly,
Claude Code's own Bash permission check rejected the command outright ("A
variable in braces in this command can't be checked before it runs") the
moment it needed to reference a token, and there is no variable-free way
to write an authenticated `git push`/`gh pr create` yourself. Instead, a
deterministic `apply_patch_and_open_pr` job the workflow runs afterward
does all of that - plain shell, no agent, no permission check to fight,
working from a patch file you leave behind. That job runs separately from
this skill's own job (a reusable-workflow invocation can't share a runner
filesystem with whatever calls it), so the patch file doesn't just sit
there waiting to be read - the calling workflow uploads whatever you leave
at the fixed paths below as an artifact, which `apply_patch_and_open_pr`
then downloads. None of that is this skill's concern; it only needs to
write the files:

1. **Capture the fix as a patch**, nothing else:
   ```
   git diff > /tmp/claude_fix.patch
   ```
   The working tree should have exactly the fix applied and nothing else
   touched at this point, so this is exactly the fix, no more - don't
   `git add`/`git commit`/`git checkout -b` anything.
2. **Write the PR title and body to two more plain files** - fixed,
   literal paths, no variables needed to produce them:
   ```
   cat > /tmp/claude_pr_title.txt <<'EOF'
   [WIP] <one-line summary of the fix>
   EOF
   cat > /tmp/claude_pr_body.txt <<'EOF'
   <written with the xla-pr-description skill - see below>
   EOF
   ```
   The workflow treats **the mere existence of a non-empty
   `/tmp/claude_fix.patch`** as "there is a verified fix to apply" - if you
   reach this step, always write all three files; if you stop earlier,
   never create any of them. If `/tmp` isn't writable in this sandbox,
   write them under `debug_tools/claude_scratch/` instead (same fallback
   as elsewhere in this skill) and say so plainly in Step 6's report,
   since in that case the workflow's upload step won't find them at all -
   no patch will reach `apply_patch_and_open_pr`, so mention that
   explicitly rather than letting the report imply a PR is coming.

   PR body - write it with the `xla-pr-description` skill: invoke it with
   the Skill tool (if that fails, Read
   `.claude/skills/xla-pr-description/SKILL.md`), and follow its Gather
   and Rules sections. Don't restate its rules here, so the two can't
   drift. Overrides for this setting:
   - The change is the uncommitted `git diff` you just captured, not a
     branch against a merge base.
   - The issue and its comments are the conversation: take the problem,
     the failing test and the root cause from there. Nobody can answer a
     question here, so leave out anything you can't back with what you
     found instead of asking.
   - Skip its Output section: write the body to `/tmp/claude_pr_body.txt`
     (not `/tmp/pr_desc.md`), and don't touch the clipboard or the PR.
   - A fix that makes an existing test pass adds no tests, so the test
     sections are usually dropped; the closing line below names the
     target instead.

   After the template's sections, add one closing line saying how the fix
   was verified, with a link back to the issue:
   ```
   Automatically verified with `<test target>`: reproduced the failure under
   heavy repetition, green with this fix, red again on revert - see
   https://github.com/<CI_REPO>/issues/<ISSUE_NUMBER>.
   ```
3. **The workflow applies this patch directly on top of `REPOSITORY@HEAD_SHA`
   itself** - the exact commit this very skill run just verified the fix
   against, fetched fresh rather than reused from any branch tip. Rebasing
   onto a later tip (`ROCm/xla`'s own `main`, or even `openxla/xla`'s)
   would ship something that was never actually put through the
   red->green->red cycle, so the workflow deliberately doesn't do that. It
   then pushes the resulting commit as a new branch into `ROCm/xla` (the
   only repo it can push to - `ROCm/xla` being a registered GitHub fork of
   `openxla/xla` is what makes that branch's history acceptable there) and
   opens the PR against `ROCm/xla`'s own `main`. Not your call to make
   here, it's baked into the workflow step itself - should essentially
   never fail to apply, since it's the same patch against the same commit
   it was generated from.

## Step 6 — Report, as a new comment

Whether you end up with a verified fix, "not applicable", or just the
findings from a custom investigation: post your report as a **new**
comment - nothing to fetch or merge first:
```
gh issue comment <ISSUE_NUMBER> --repo <CI_REPO> --body-file /tmp/response_comment.txt
```
(same `/tmp`-not-writable fallback as `debug-nightly-failure`: write it
inside the checkout instead, e.g. `debug_tools/claude_scratch/`, and clean
it up before finishing.)

Build that comment as:
```
<!-- rocm-nightly-claude-mention:JOB_NAME / STEP_NAME:INSTRUCTIONS_HASH -->
**You asked:** <quote INSTRUCTIONS verbatim, or a tight paraphrase if it was long>

**What I did:** <one or two sentences - e.g. "reproduced the original
failure under heavy repetition, applied the proposed fix (green), reverted
it (red again)", or "set a breakpoint at <file>:<line> and inspected
<what>", or whatever actually happened>

<exactly one of:>
- ✅ **Verified** — opening a draft PR with the fix now. A follow-up
  comment will link to it once opened.
- ❌ **Not applicable** — <the specific reason: never reproduced red /
  still red with the fix applied / stayed green after reverting it>
- 🔍 **Investigated, no fix proposed** — <findings from the custom
  investigation you were asked to do, with specific evidence: variable
  values at the breakpoint, a backtrace excerpt, whatever directly answers
  what was asked>
```
The marker is what Step 0 (and `claude_issue_triage.yml`'s pre-dispatch
check) scan the issue's comments for to know this exact instruction was
already attempted - get `JOB_NAME`/`STEP_NAME`/`INSTRUCTIONS_HASH` right,
same substitution rule as Step 0. For a verified fix, you don't need to
(and can't yet) include the real PR link or number - a separate,
deterministic workflow step posts its own follow-up comment with that link
once the PR actually exists, entirely outside your own execution.
