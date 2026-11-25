---
name: respond-to-issue
description: Triggered when an authorized ROCm/xla collaborator posts a new `@claude <instructions>` comment on an ordinary issue - not one of the nightly monitor's own per-failure issues (those go through respond-to-mention instead, with a specific commit to reproduce). Reads the issue's title, body, and every existing comment for context, then does whatever INSTRUCTIONS literally asks - answer a question, investigate something in the code/git history, build/run/debug something using the same rocmless/bazel/gdb toolchain debug-nightly-failure and respond-to-mention use, or make a code change. A code change is written out as a plain patch file (plus a PR title/body) for a deterministic workflow step to apply and open as a draft PR - the report says plainly whether it was actually built/tested or not. Posts its response as a single new comment, and never repeats the exact same instruction once attempted. Use when invoked by the "respond_to_issue_mention" job of rocm_infra_agent.yml for one specific comment carrying an `@claude` instruction on a non-CI-failure issue.
argument-hint: "[ci-repo] [issue-number] [instructions] [instructions-hash]"
context: fork
agent: general-purpose
allowed-tools: Bash(gh issue comment *), Bash(gh api *), Bash(gh search prs *), Bash(gh pr view *), Bash(gh pr diff *), Bash(gh run view *), Bash(git log *), Bash(git show *), Bash(git diff *), Bash(git blame *), Bash(docker exec rocmless-debug *), Bash(build_tools/rocm/*), Bash(bazel *), Bash(gdb *), Bash(objdump *), Bash(grep *), Bash(cat *), Bash(timeout *), Read, Grep, Glob, Write, Edit
---

# Respond to an @claude Mention on an Ordinary Issue

ROCm/xla's own default branch is already checked out in the current
working directory - there is no specific failing commit to reproduce here
(that's respond-to-mention's job, for the nightly monitor's own per-failure
issues), but the same `rocmless` toolchain sidecar `debug-nightly-failure`
and `respond-to-mention` use is already running too, if `INSTRUCTIONS`
actually calls for building, running, or debugging something. This is the
generic path: read the issue, do what it asks.

## Arguments

The caller passes four values, space separated: `CI_REPO ISSUE_NUMBER
INSTRUCTIONS INSTRUCTIONS_HASH`. `INSTRUCTIONS` can itself contain spaces,
punctuation, and quotes, so do not split `$ARGUMENTS` positionally by
whitespace - the invoking prompt also restates each value on its own
labeled line; read the values from there.

- `CI_REPO`: `ROCm/xla` - the repo the issue lives in, and the only repo
  any patch you write in Step 3 is ever meant to land in.
- `ISSUE_NUMBER`: the issue the `@claude` comment was posted on. Read
  **everything** already on it (Step 1) before doing anything else - the
  title, the body, and every existing comment are your only context; there
  is no job/step/failing-commit metadata to fall back on the way
  respond-to-mention has.
- `INSTRUCTIONS`: the literal text a human typed after `@claude` in a new
  comment on the issue - this is what actually governs what you do. Follow
  it specifically; don't substitute your own idea of what would be useful.
- `INSTRUCTIONS_HASH`: a short hash of `INSTRUCTIONS`, already computed for
  you - use it verbatim in the Step 0/4 marker, don't recompute it.

## IMPORTANT: scope and safety

- **This skill itself runs on the bare `build-only-xla` host, not inside
  any container** - same as `debug-nightly-failure`/`respond-to-mention`.
  A `rocmless` container named `rocmless-debug` is already running as a
  sidecar for the ROCm/Bazel toolchain - every
  `build_tools/rocm/run_xla_ci_build.sh`/plain `bazel` command you run
  **must** be wrapped: `docker exec rocmless-debug <command>`. Everything
  else (`git`, `gh`, reading/writing files) runs directly on the host.
- **Decision-making before any build (Steps 0-2) is hard-capped at 30
  minutes of wall-clock time**, matching respond-to-mention. Time spent
  actually building/testing (if `INSTRUCTIONS` calls for it) doesn't count
  against that, but the whole run still has to fit this step's own
  (longer) job timeout.
- **If you do build or run anything**, the same rules
  `debug-nightly-failure`/`respond-to-mention` follow apply here too -
  this skill doesn't get a lighter version of them just because the ask is
  more open-ended:
  - This runner has no local GPU - every Bazel/`run_xla_ci_build.sh`
    invocation **must** include `--strategy=TestRunner=remote,local`, or
    GPU test actions won't run at all.
  - Narrow every invocation to the specific target(s) `INSTRUCTIONS`
    actually concerns - never the whole `//xla/...` tree.
  - If a `--config=rocm_rbe` build is involved, override the EngFlow
    hostname (strip the `internal.` prefix - see
    `debug-nightly-failure/SKILL.md`'s safety rules for the exact four
    flags and why; this debug runner can't reach
    `internal.actinolite.cluster.engflow.com`). Leave
    `--config=rocm_rbe_dev` alone.
  - `gdb`/`objdump`/VLOG techniques from `debug-nightly-failure/SKILL.md`
    Step 6 apply unchanged if `INSTRUCTIONS` calls for that kind of
    debugging.
- Only act on this one issue, per `INSTRUCTIONS` specifically. Do not go
  looking at other issues or unrelated comments.
- **You never run `git push`, `git commit`, or `gh pr create` yourself -**
  same hard rule as respond-to-mention (a Claude Code Bash permission
  restriction, not a style choice). If you end up with a code change
  (Step 2), write it out as a patch file (Step 3) for a deterministic
  workflow step to apply and open as a draft PR against `ROCm/xla`'s own
  `main` - never commit, branch, or push anything yourself.
- **State plainly, in both the PR body (if you open one) and your comment
  (Step 4), whether a change was actually built/tested or only read
  through** - never claim something "works" or "fixes" a problem without
  having actually run it. A reasonable default when `INSTRUCTIONS` is
  purely "fix this" with no explicit ask to verify: attempt a build/test
  if there's an obvious, narrow one to run and time/turns allow; otherwise
  say so and let the PR go up as unverified for human/CI review.
- The only GitHub write is one new comment on the `ISSUE_NUMBER` issue
  (Step 4) via `gh issue comment <ISSUE_NUMBER> --repo <CI_REPO>
  --body-file <file>`. **Never edit the issue body or an existing
  comment.** If Step 0 finds this exact instruction was already acted on,
  make no GitHub writes at all.
- This run also has a hard cap on agentic turns. If you notice you're
  going in circles, stop and post Step 4's report with whatever you have -
  a partial answer beats silence.
- Clean up any scratch files before finishing (`/tmp/gdb_*`, extra build
  outputs, and `debug_tools/claude_scratch/` if you used it - see
  `debug-nightly-failure/SKILL.md` for why that one needs `rm -rf`
  specifically, not just `/tmp` cleanup).

## Step 0 — Check whether this exact instruction was already acted on

Before anything else, in case of a race or a manual re-run:
```
gh api --paginate repos/<CI_REPO>/issues/<ISSUE_NUMBER>/comments
```
Check whether any comment's `body` already **contains**, anywhere,
`<!-- claude-mention:INSTRUCTIONS_HASH -->` (substitute the literal
`INSTRUCTIONS_HASH` value you were given, no other change). If one does:
**stop here, make no GitHub writes at all** - this exact instruction has
already been attempted once, successfully or not, and is never retried. If
a *different* `claude-mention:...` marker is present instead (a different
hash), that's a previous, different instruction that was already handled -
leave its comment alone and proceed to act on the current one. Keep the
fetched comments around - Step 1 reads them too.

## Step 1 — Read the issue for context

Fetch the issue's title and body:
```
gh issue view <ISSUE_NUMBER> --repo <CI_REPO> --json title,body
```
Read it together with every comment already fetched in Step 0, in order -
this is your entire picture of what's being discussed. There is no
job/step/failing-commit context to fall back on here, so don't assume any;
if `INSTRUCTIONS` references something ("the bug above", "this", "it")
that only makes sense in light of the rest of the issue, resolve it from
what you just read.

## Step 2 — Do what INSTRUCTIONS asks

Use your judgment on what's actually being asked, and answer it directly:

- **A question** ("why does X do Y", "what's the status of Z", "is this
  still true upstream") - investigate with `Read`/`Grep`/`Glob` on the
  checked-out code, `git log`/`git show`/`git blame` for history, and
  `gh search prs`/`gh pr view`/`gh pr diff`/`gh run view` for anything
  that needs looking up on GitHub - then answer from what you actually
  found, not from a guess.
- **An investigation** (find where something is defined, trace a
  behavior, check whether a PR/commit addresses something) - same tools,
  report concrete findings (file/line references, commit SHAs, PR
  numbers) rather than vague impressions.
- **A build/run/debug request** ("does this build", "run this test",
  "reproduce X", "what does this print") - use the `rocmless` toolchain
  exactly as `debug-nightly-failure`/`respond-to-mention` do: `docker exec
  rocmless-debug build_tools/rocm/run_xla_ci_build.sh ...` (or plain
  `bazel`) narrowed to the specific target(s), with
  `--strategy=TestRunner=remote,local` and (if `--config=rocm_rbe`) the
  EngFlow host override - see the safety rules above. `gdb`/`objdump`/VLOG
  techniques apply the same as `debug-nightly-failure/SKILL.md` Step 6 if
  the ask calls for them.
- **A code change** ("fix this", "add X", "update Y") - make the change
  directly in the checked-out working tree via `Edit`/`Write`. Keep it
  scoped to exactly what was asked; don't refactor or touch anything
  unrelated. If there's an obvious, narrow build/test to confirm it with
  (and time/turns allow), run it the same way as the bullet above before
  writing Step 3's patch - otherwise write the patch from a careful code
  read alone and say so plainly in Step 4.

If `INSTRUCTIONS` doesn't actually require a code change, skip straight to
Step 4 - don't write a patch file for a purely investigative,
question-answering, or build/debug-only ask.

## Step 3 — If you made a code change: write it out as a patch for the workflow to apply

Only reached if Step 2 ended with an actual code change in the working
tree. Nothing has been committed. **You do not commit, push, or run
`gh pr create` yourself** - see respond-to-mention/SKILL.md's Step 5 for
why (a hard Claude Code permission restriction). A deterministic
`apply_patch_and_open_pr` job the workflow runs afterward does all of
that, working from a patch file you leave behind:

1. **Capture the change as a patch**, nothing else:
   ```
   git diff > /tmp/claude_fix.patch
   ```
   The working tree should have exactly your change applied and nothing
   else touched - don't `git add`/`git commit`/`git checkout -b` anything.
2. **Write the PR title and body to two more plain files**:
   ```
   cat > /tmp/claude_pr_title.txt <<'EOF'
   [WIP] <one-line summary of the change>
   EOF
   cat > /tmp/claude_pr_body.txt <<'EOF'
   <openxla/xla's own PR template, filled in - see shape below>
   EOF
   ```
   The workflow treats the mere existence of a non-empty
   `/tmp/claude_fix.patch` as "there is a change to apply" - if you reach
   this step, always write all three files; if Step 2 made no change,
   never create any of them. If `/tmp` isn't writable in this sandbox,
   write them under `debug_tools/claude_scratch/` instead and say so
   plainly in Step 4's comment, since in that case the upload step won't
   find them and no PR will follow.

   PR body shape - fill in `openxla/xla`'s own pull request template with
   real, specific content under every heading it asks for, never leaving
   one of its placeholder prompts in place and never substituting a
   different shape of your own. Read the template itself rather than
   recalling it from memory - it can change upstream, and a stale copy in
   this skill would silently drift from it. `CI_REPO`'s own default branch
   (already checked out) doesn't carry this file, so fetch it directly:
   ```
   gh api repos/openxla/xla/contents/.github/pull_request_template.md \
     --jq '.content' | base64 -d
   ```
   **Keep every section short - one or two sentences, not paragraphs.**
   This is a PR description a reviewer skims to decide whether to look
   closer, not a lab notebook: no tables, no flag-by-flag command dumps, no
   restating the whole investigation. That detail already lives in the
   issue the closing line below links back to.

   Two things the template itself doesn't spell out, so apply them
   yourself: for its "Kind of Contribution" section, pick exactly the one
   that applies and delete the rest (its own instruction, easy to skip by
   accident) rather than leaving all the options listed; and omit its
   "Benchmark" section entirely unless the kind is a Performance
   Improvement - it doesn't apply otherwise, the same way a human filling
   this out by hand would leave it out rather than writing "N/A". For
   "Unit Tests"/"Execution Tests" specifically, name the target and the
   one-line result, not the full command or every flag used. After the
   template's own sections, add one closing line:
   ```
   <exactly one of:>
   Verified by running <the specific target/command>, which <passed /
   confirmed the fix>. Requested in https://github.com/<CI_REPO>/issues/<ISSUE_NUMBER>.
   <or, if you did not build/test it:>
   Not built or tested - written from a direct reading of the surrounding
   code only. Requested in https://github.com/<CI_REPO>/issues/<ISSUE_NUMBER>.
   ```
3. **The workflow applies this patch directly on top of `CI_REPO`'s own
   `main`**, pushes the resulting commit as a new branch, and opens a
   draft PR against `main`. Not your call to make here, it's baked into
   the workflow step itself.

## Step 4 — Report, as a new comment

Post your response as a **new** comment - nothing to fetch or merge first:
```
gh issue comment <ISSUE_NUMBER> --repo <CI_REPO> --body-file /tmp/response_comment.txt
```
(if `/tmp` isn't writable, write it under `debug_tools/claude_scratch/`
instead.)

Build that comment as:
```
<!-- claude-mention:INSTRUCTIONS_HASH -->
**You asked:** <quote INSTRUCTIONS verbatim, or a tight paraphrase if it was long>

<exactly one of:>
<your actual answer/findings - for a question, investigation, or
build/run/debug request with no code change>
<or, if Step 3 ran:>
✅ Opening a draft PR with this change now - verified by <what you ran>,
see the PR description.
<or, if Step 3 ran but you didn't build/test it:>
✅ Opening a draft PR with this change now - **not built or tested**, see
the PR description.
```
The marker is what Step 0 (and `claude_issue_triage.yml`'s pre-dispatch
check) scan the issue's comments for to know this exact instruction was
already attempted - substitute the literal `INSTRUCTIONS_HASH` value, no
other change.
