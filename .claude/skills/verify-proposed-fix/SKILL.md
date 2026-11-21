---
name: verify-proposed-fix
description: Hard-capped at 30 minutes of non-build decision-making - triggered when a human reacts 🚀 on a debug-nightly-failure comment's proposed fix, asking Claude to actually try it out. Reproduces the original failure under heavy repetition (nightly GPU CI can be flaky), applies the proposed fix, confirms it goes green, reverts the fix, and confirms it goes red again - a differential check that the fix, not noise, is responsible. If verified, opens a draft "[WIP] ..." PR against ROCm/xla with a Google-style CL description. Edits its verdict (a PR link, or a plain "not applicable" statement) back into the same comment, folded, and never repeats once attempted. Use when invoked by the "verify" job of rocm_infra_monitor.yml for one specific 🚀-reacted comment.
argument-hint: [ci-repo] [issue-number] [comment-url] [job-name] [step-name] [repository] [head-sha] [failure-url]
context: fork
agent: general-purpose
allowed-tools: Bash(gh issue edit *), Bash(gh label create *), Bash(gh api *), Bash(gh run view *), Bash(gh pr create *), Bash(gh pr edit *), Bash(git *), Bash(docker exec rocmless-debug *), Bash(build_tools/rocm/*), Bash(bazel *), Bash(grep *), Bash(cat *), Bash(timeout *), Read, Grep, Glob, Write, Edit
---

# Verify a Proposed Fix for a ROCm Nightly CI Failure

The commit that originally failed is already checked out in the current
working directory - that is both the repro environment and the base you
eventually branch from if the fix checks out.

## Arguments

The caller passes eight values, space separated: `CI_REPO ISSUE_NUMBER
COMMENT_URL JOB_NAME STEP_NAME REPOSITORY HEAD_SHA FAILURE_URL`. `JOB_NAME`
and `STEP_NAME` can themselves contain spaces and parentheses, so do not
split `$ARGUMENTS` positionally by whitespace - the invoking prompt also
restates each value on its own labeled line; read the values from there.

- `CI_REPO`: `ROCm/xla` - the repo the tracking issue lives in, and the
  **only** repo you ever push a branch or open a PR against, regardless of
  what `REPOSITORY` says (see below).
- `ISSUE_NUMBER`: the tracking issue (shared by many unrelated failures -
  don't react to or summarize anything else on it).
- `COMMENT_URL`: the single comment carrying the proposed fix, and the
  **only** comment this skill ever writes to - same "edit in place, never
  post a new comment" rule as `debug-nightly-failure`. Its full analysis
  may be sitting behind a `<details>` fold from that skill's Step 7 - that
  doesn't hide anything from `gh api`, only from rendering, so read the
  raw body for the proposal regardless of fold state.
- `JOB_NAME` / `STEP_NAME`: the failure signature, used for the verify
  marker below.
- `REPOSITORY` / `HEAD_SHA`: already checked out in the working directory -
  this is almost always `openxla/xla@<sha>`, **not** `ROCm/xla` (see the
  PR-target rule in Step 5).
- `FAILURE_URL`: the original failing job, for re-fetching its log in Step
  1 if the comment doesn't already spell out the exact command.

## IMPORTANT: scope and safety

- **This skill itself runs on the bare `build-only-xla` host, not inside any
  container**, exactly like `debug-nightly-failure`. A `rocmless` container
  named `rocmless-debug` is already running as a sidecar - every
  `build_tools/rocm/run_xla_ci_build.sh`/plain `bazel` command you run
  **must** be wrapped: `docker exec rocmless-debug <command>`. Everything
  else (`git`, `gh`, reading/writing files) runs directly on the host; the
  workspace is bind-mounted into the container at the identical path.
- **Step 1 (extracting/understanding the proposal, before any build) is
  hard-capped at 30 minutes of wall-clock time.** The three build+test
  cycles in Steps 2-4 don't count against that, but the whole run still has
  to fit inside this step's own (much longer) job timeout - budget those
  30 minutes roughly as: 10 minutes to read the proposal and recover the
  original failing command, up to 20 more only if the fix is prose-only and
  you need to actually write the code change yourself.
- This runner has no local GPU. Every Bazel/`run_xla_ci_build.sh`
  invocation in Steps 2-4, with no exception, **must** include
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
- Verify only this one signature's proposed fix. Do not go looking for
  other red steps or other proposed fixes along the way.
- **This is the one skill in the whole monitoring pipeline allowed to
  `git push` and `gh pr create`.** Even so: only ever push to, and open a
  PR against, `ROCm/xla` - never `openxla/xla`, even when `REPOSITORY` is
  `openxla/xla` (the common case for nightly failures). We have no push
  access upstream, so the PR's base is always `ROCm/xla`'s `main-mirror`
  branch specifically (see Step 5) - never `openxla/xla` directly, and
  never any other `ROCm/xla` branch either, even its own default one.
  Pushing to or opening PRs on the separately-governed upstream project is
  out of scope for this automation regardless of where the failing commit
  came from. Never push anywhere else, never force-push, never open more
  than one PR, never mark it ready-for-review (always a draft), never
  merge anything.
- The only other GitHub write is one edit to the single `COMMENT_URL`
  comment (Step 6, your final verdict) via
  `gh api -X PATCH repos/<CI_REPO>/issues/comments/<id> -F body=@<file>`
  (note `-F`, not `-f` - `-f`/`--raw-field` treats `@file` as a literal
  string and silently sets the body to that text; only `-F`/`--field`
  actually reads the file). **Never post a new comment.** If Step 0 finds
  this was already verified, make no GitHub writes at all.
- Use `--runs_per_test=20 --test_arg=--gtest_repeat=50
  --nocache_test_results` (20 independent RBE sandboxes × 50 in-process
  repeats = 1000 total executions) for every reproduce attempt in Steps
  2-4, by default - this is what rules out "it just happened to pass this
  run" on inherently flaky nightly GPU CI. Only scale it down if the target
  is slow enough that three cycles at this volume won't fit the job's
  timeout, and say so plainly in the report (Step 6) if you do.
- Never claim "not applicable" without having completed the full
  red→green→red cycle for that verdict, and never open a PR without having
  personally observed it go green in this same run (not from the original
  comment's own, earlier, unverified claim).
- If you bail out partway through (budget exhausted, build infra failure,
  etc.), leave the working tree exactly as Step 1 found it (fix not
  applied) before reporting - a partial attempt with no code applied is
  always safer than one that silently leaves an untested change in place.

## Step 0 — Check whether this was already verified

Before anything else, in case of a race (two runs queuing the same
signature close together) or a manual re-run:

```
gh api repos/<CI_REPO>/issues/comments/<comment-id-from-COMMENT_URL>
```

Check whether its `body` already **contains**, anywhere,
`<!-- rocm-nightly-claude-verify:JOB_NAME / STEP_NAME -->` (substitute the
literal `JOB_NAME`/`STEP_NAME` values you were given, ` / ` separator, no
other change). If it does: **stop here, make no GitHub writes at all** -
this has already been attempted once, successfully or not, and is never
retried. If the marker is absent, proceed to Step 1.

## Step 1 — Extract the proposed fix and the original failing command

Fetch the comment's current body and find its "Possible fix (quick
estimate, unverified)" or "Fix found" section (whichever is present -
`debug-nightly-failure` only ever writes one of the two).

- **If it has a \`\`\`diff\`\`\` fenced block**: that's the fix. You'll
  apply it with `git apply` in Step 3.
- **If it's described in prose only** (the "Possible fix" shape allows
  this when the original analysis ran out of time to write the diff out):
  read the description and the file/function it points at, and implement
  the change yourself via `Edit`. This is the one place in this entire
  pipeline this skill writes production code - and only because a human
  explicitly reacted 🚀 asking for this exact proposal to be tried, not a
  new one of your own devising.

Also recover the exact original invocation: the specific failing
target/test case and the full `build_tools/rocm/run_xla_ci_build.sh`/
`bazel` command and flags. The comment's analysis usually already states
this plainly; if it doesn't spell out the flags verbatim, re-fetch the
original job's log instead of guessing:
```
gh api --allow-escape-sequences "repos/<CI_REPO>/actions/jobs/<job-id-from-FAILURE_URL>/logs" > job.log
```
(the numeric job ID is the part after `/job/` in `FAILURE_URL`).

Do not touch the working tree in this step - don't apply anything yet.
Step 2 needs to reproduce the *original*, unmodified failure first.

## Step 2 — Reproduce red (baseline, fix not yet applied)

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

## Step 5 — If verified: open the draft PR against `ROCm/xla`

Only reached if Step 4 confirmed red→green→red. The fix is currently
re-applied in the working tree (you undid the revert at the end of Step
4). Do this **before** Step 6's report, so the report can link the real
PR instead of a placeholder.

1. **Commit** it:
   ```
   git config user.name "github-actions[bot]"
   git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
   git checkout -b claude/verify-fix/<slug>
   git add -A
   git commit -m "<one-line summary, same text as the PR title below>"
   ```
   `<slug>`: `JOB_NAME`/`STEP_NAME` lowercased, non-alphanumeric runs
   collapsed to a single `-`, trimmed, plus a short prefix of `HEAD_SHA` for
   uniqueness (e.g. `claude/verify-fix/gemm-rewriter-group-gemm-test-4a02f02`).
2. **Push it to `ROCm/xla`** through an explicit token-embedded remote,
   not the checkout's own ambient credential (this checkout used
   `persist-credentials: false`, same precedent as
   `rocm_xla_jax_metrics_nightly.yml`'s own pushes). Use `${PUSH_TOKEN}`
   here, **not** `${GITHUB_TOKEN}`:
   ```
   git remote add verify-target "https://x-access-token:${PUSH_TOKEN}@github.com/ROCm/xla.git"
   git push verify-target claude/verify-fix/<slug>
   ```
   `PUSH_TOKEN` is a separate, scoped GitHub App token the workflow creates
   specifically for this push - the default `GITHUB_TOKEN` cannot push a
   brand-new branch into this repo at all, because creating a ref makes
   GitHub check whether it touches `.github/workflows/*`, which needs a
   "workflows" permission scope the default token can never carry no
   matter what the job's `permissions:` block grants it. This isn't
   theoretical - it's exactly what failed the first time this skill ran
   for real (`POST /git/refs` → 403 "Resource not accessible by
   integration" / "workflows scope may be required"), even though the fix
   itself never touched a workflow file - that check applies to any
   brand-new ref, not just to the commit's own diff. Use `${GITHUB_TOKEN}`
   (the default) for everything else in this skill, including `gh pr
   create` below - opening a PR against an already-pushed branch needs
   nothing but `pull-requests: write`, which the job's own `permissions:`
   already grants the default token; it has no workflows-scope involved
   and doesn't need `PUSH_TOKEN`.
3. **Base the PR on `main-mirror`, not `ROCm/xla`'s own default branch.**
   `REPOSITORY`/`HEAD_SHA` are almost always `openxla/xla@<sha>` - a
   different history from `ROCm/xla`'s own default branch (which carries
   ROCm-specific patches and has long since diverged from upstream). Using
   that as `--base` would make GitHub diff your one-commit fix against the
   wrong ancestry and show a huge, nonsensical PR. `main-mirror` is kept a
   byte-for-byte copy of `openxla/xla`'s own `main` by the
   `sync_main_mirror.yml` pipeline (resynced every 2 hours), so it shares
   real ancestry with `HEAD_SHA` - that's what keeps the PR diff to just
   your fix commit. Don't look up or use any other branch name here, even
   if it looks like the "main" one for this repo.
4. **Open the draft PR**:
   ```
   gh pr create --repo ROCm/xla --draft \
     --title "[WIP] <one-line summary of the fix>" \
     --body "<Google-style CL description - see shape below>" \
     --base main-mirror \
     --head claude/verify-fix/<slug>
   ```
   Body shape - a Google-style CL description, **not** a Summary/Test-plan
   bullet template: the one-line subject is already in `--title`, so the
   body is a blank line followed by prose paragraphs - what was broken,
   the root cause (reuse the analysis already in the comment rather than
   re-deriving it), and what this change does about it - then a closing
   line disclosing how it was produced and verified, with a link back to
   the comment:
   ```
   <One or two prose paragraphs: what broke and why, what this change
   does differently. No bullet lists, no "## Summary"/"## Test plan"
   headings.>

   Automatically verified by reproducing the original failure under heavy
   repetition, applying this fix (green), and reverting it (red again) to
   confirm causality - see the verification writeup on
   <COMMENT_URL>.
   ```
   `gh pr create` prints the new PR's URL on success - keep it for Step 6.

## Step 6 — Report, folded, back into the same comment

Whether verified, not applicable, or stopped early in Steps 2-4: fetch the
comment's current body, append your verdict, and PATCH it back - **never
post a new comment**:
```
COMMENT_ID=<the numeric id after #issuecomment- in COMMENT_URL>
gh api "repos/<CI_REPO>/issues/comments/${COMMENT_ID}" --jq .body > /tmp/comment_body.txt
# append a divider, then the verdict below, to /tmp/comment_body.txt -> /tmp/new_body.txt
gh api -X PATCH "repos/<CI_REPO>/issues/comments/${COMMENT_ID}" -F body=@/tmp/new_body.txt
```
(same `/tmp`-not-writable fallback as `debug-nightly-failure`: if `gh api`
can't find the file there, write it inside the checkout instead, e.g.
`debug_tools/claude_scratch/`, and clean it up before finishing.)

Append, as a *nested* fold (the comment may already be wrapped in its own
outer `<details>` from `debug-nightly-failure` Step 7 - GFM nests these
fine):
```
---

<!-- rocm-nightly-claude-verify:JOB_NAME / STEP_NAME -->
<details><summary>Fix verification (click to expand)</summary>

**Method:** reproduced the original failure under heavy repetition
(`--runs_per_test=20 --test_arg=--gtest_repeat=50`, or the scaled-down
values actually used - say which), applied the proposed fix (green),
reverted it (red again) to confirm the fix - not noise - is responsible.

<exactly one of:>
- ✅ **Verified** — [draft PR #<n>](<the real URL from Step 5>) opened
  with the fix.
- ❌ **Not applicable** — <the specific reason from whichever step you
  stopped at: never reproduced red / still red with the fix applied /
  stayed green after reverting it>

</details>
```
