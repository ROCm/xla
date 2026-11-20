---
name: cleanup-closed-nightly-issue
description: Hard-capped at 20 minutes - classifies every failure signature on a just-closed ROCm CI nightly tracking issue as finalized-resolved (crossed out AND 👍'd), tentatively-resolved (Claude thinks it's fixed but nobody has confirmed yet), or still-pending. Marks the first two resolved in place (never closing or deleting anything) and migrates everything except finalized-resolved signatures forward onto the current open tracking issue, preserving dedup markers so debug-nightly-failure never re-analyzes them. Use when invoked by the "cleanup" job of rocm_infra_monitor.yml for one specific closed tracking issue.
argument-hint: [ci-repo] [closed-issue-number]
context: fork
agent: general-purpose
allowed-tools: Bash(gh issue comment *), Bash(gh api *), Bash(gh issue view *), Bash(gh issue list *), Bash(gh issue edit *), Bash(gh label create *), Bash(build_tools/rocm/create_infra_error_issue.sh), Read, Grep, Glob
---

# Clean Up a Closed ROCm Nightly Tracking Issue

A human just closed the long-lived `ROCm CI Nightly Failures Detected`
tracking issue (this is, and remains, a fully manual action - nothing
about this skill closes or reopens issues). Your job is to go through
every failure signature that accumulated on it, mark the ones that are
actually done, and carry everything else forward so nothing gets lost.

## Arguments

Two values, space separated: `CI_REPO CLOSED_ISSUE_NUMBER`.

- `CI_REPO`: the repository the closed issue lives in (`ROCm/xla`) - use
  this for every `gh` call, never guess a different repo.
- `CLOSED_ISSUE_NUMBER`: the issue to clean up. It was found by the
  caller because it matches the exact tracking-issue title and has no
  `<!-- rocm-nightly-cleanup-done -->` marker comment yet - don't
  re-derive that, just double check it in Step 1.

## IMPORTANT: scope and safety

- **Never `gh issue close`, never `gh issue reopen`, never delete a
  comment.** The only writes you make are: `gh issue comment` (migrated
  comments onto the destination issue, plus the summary comment on the
  closed issue), `gh api -X PATCH .../issues/comments/<id>` (in-place
  crossout edits), and `gh label create` / `gh issue edit --add-label`
  (re-applying category labels on the destination issue for migrated
  signatures).
- **Hard-capped at 20 minutes wall-clock.** A closed issue can carry 15+
  signatures (unlike `debug-nightly-failure`'s single-signature 5-minute
  cap) - budget roughly 3 minutes of fixed overhead (Steps 1-2, plus
  finding/creating the destination issue once, plus the final summary)
  and ~1 minute per signature for classification/crossout/migration. If
  you're going to run out of budget, stop starting new signatures,
  finish or cleanly abandon (never half-post) whatever signature is
  in-flight, then post an **interim** progress comment on the closed
  issue - same content as Step 6's summary, but **without** the
  `rocm-nightly-cleanup-done` marker line - so the next hourly run picks
  this issue back up safely. Every write below is idempotency-checked
  specifically so a resumed partial run never double-edits or
  double-migrates anything.
- Never claim a signature was migrated or marked resolved without having
  actually posted/edited the corresponding comment.
- This issue is shared by many unrelated failure signatures accumulated
  over its whole lifetime - handle every one you find, don't skip ones
  that look old or unimportant.

## Step 1 — Sanity check and already-cleaned-up guard

```
gh issue view <CLOSED_ISSUE_NUMBER> --repo <CI_REPO> --json number,title,state,closedAt,body
```

Confirm `state == "CLOSED"` and the title is exactly
`ROCm CI Nightly Failures Detected`. If either doesn't hold, stop - this
is a defensive re-check, the caller already verified it, but a bad
argument or a race (someone reopened it) should not make you proceed.

Fetch every comment once - you'll need the full set repeatedly below, and
it can't change out from under you mid-run on a closed issue:

```
gh api --paginate "repos/<CI_REPO>/issues/<CLOSED_ISSUE_NUMBER>/comments"
```

If any comment's body starts with exactly
`<!-- rocm-nightly-cleanup-done -->`, **stop here** - this issue was
already fully processed by an earlier run.

## Step 2 — Group comments into signatures

Every comment this pipeline posts starts its body with exactly one of
three markers, ` / ` separator between job name and step name, no other
change to the literal text:

- `<!-- rocm-nightly-failure:JOB_NAME / STEP_NAME -->` - detection
  comment (posted by `github-actions[bot]`)
- `<!-- rocm-nightly-claude-triage:JOB_NAME / STEP_NAME -->` - triage
  comment (posted by `claude[bot]`)
- `<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->` - final
  report comment (posted by `claude[bot]`)

Group all comments by their `JOB_NAME / STEP_NAME` signature into
`detection` / `triage` (optional) / `final_report` (optional). A
signature with only a detection (and maybe a triage) comment but no
final report means a debug run never finished - handle it, it's just
missing information, not an error.

Ignore comments with no marker at all (e.g. a short "already analyzed,
see [Previous analysis](...)" pointer comment, or a human's own free-form
remark that isn't attached to a specific signature via crossout - see
Step 3) - they don't represent a signature of their own.

## Step 3 — Classify each signature

For each signature, pick the single most informative comment:
`final_report`, else `triage`, else `detection`. Determine two
independent facts about it:

1. **`crossed_out`** - does this comment visibly read as struck
   through / marked resolved? This includes Claude's own
   `**RESOLVED (auto-detected fix, awaiting confirmation)**` banner from
   an earlier pass of this same skill (Step 4), **or** a human's own
   manual edit of the comment (markdown `~~strikethrough~~` somewhere in
   it, a line saying "RESOLVED"/"resolved", or similar) - judge this by
   actually reading the comment, not a single fixed regex.
2. **`thumbed_up`** - fetch
   `gh api "repos/<CI_REPO>/issues/comments/<id>/reactions"` for that
   same comment and check whether any entry has `"content": "+1"`. If
   so, capture that reactor's `user.login` (if several, use the first).

Then classify:

- **Finalized resolved** (`crossed_out` AND `thumbed_up`, regardless of
  who did the crossing-out): fully done. This is the *only* outcome that
  does not get migrated in Step 5 - everything else does. This
  deliberately requires both signals together: a crossout with no 👍
  (Claude merely proposing a fix) or a 👍 with no crossout (a stray
  reaction for some unrelated reason) never drops a signature from
  tracking on its own.
- **Tentatively resolved**: not finalized-resolved, but either already
  `crossed_out` from an earlier pass, or `final_report` exists and its
  prose reads as "Already fixed upstream" or a verified "Fix found" (look
  for a literal `**Outcome:** Fix found` / `**Outcome:** Already fixed
  upstream` line if present - `debug-nightly-failure` is expected to
  write one - otherwise read the report's own prose; the four possible
  shapes are documented in that skill's Step 7). Gets crossed out in
  Step 4 if not already, and still migrated forward in Step 5 so a human
  can confirm it with 👍 later.
- **Still pending**: everything else, including a signature with no
  `final_report` at all. Migrated forward untouched.

## Step 4 — Cross out tentatively-resolved signatures in place

Skip (idempotent) if the comment already contains `**RESOLVED` or you
judged it `crossed_out` in Step 3 for any other reason (e.g. a human's
own manual edit - never overwrite a human's own wording).

Otherwise, build a new body for that comment, preserving its first line
(the marker) byte-for-byte, and edit it via:

```
gh api -X PATCH "repos/<CI_REPO>/issues/comments/<id>" -f body=@<tmpfile>
```

Shape:

```
<!-- rocm-nightly-claude-debug:JOB_NAME / STEP_NAME -->

> **RESOLVED (auto-detected fix, awaiting confirmation)** — <sha>
> (<link to the PR that merged it, if found> / <plain commit link
> otherwise>): <one-line summary of the fix>
> React 👍 on this comment once confirmed, to stop it being carried
> forward.

<details><summary>Original analysis (click to expand)</summary>

<everything from the original body's line 2 onward, byte-for-byte>

</details>
```

Don't use raw `~~strikethrough~~` across the whole body - GFM
strikethrough is an inline construct and doesn't reliably span headings
or fenced code blocks, which a "Fix found" report's mandatory ```` ```diff
```` block would hit and garble. The bold banner plus collapsed
`<details>` gives the same "visibly resolved, but still fully
inspectable" effect without that risk.

**Finding the fixing commit's PR**, for the banner's link: the commit is
almost always in `openxla/xla` (the monitored nightly jobs check out
`openxla/xla@main`), not necessarily `CI_REPO` itself, so try in this
order, using the first that returns a result:

```
gh api "repos/openxla/xla/commits/<sha>/pulls"
gh api "repos/<CI_REPO>/commits/<sha>/pulls"
```

This is the real GitHub "list pull requests associated with a commit"
endpoint - do **not** use `gh pr list --search <sha>`, which is a
free-text search over titles/bodies and can miss or false-match. If
neither call finds a PR, just link the plain commit
(`https://github.com/openxla/xla/commit/<sha>`), don't guess.

## Step 5 — Migrate everything except finalized-resolved signatures forward

If there is at least one tentatively-resolved or still-pending signature,
get or create the single current open tracking issue by shelling out
directly to the existing helper (don't reimplement its find-or-create
logic):

```
build_tools/rocm/create_infra_error_issue.sh
```

prints the issue number on stdout. **Skip this call entirely if every
signature on the closed issue turned out to be finalized-resolved** -
don't manufacture a new empty tracking issue when there's nothing to
carry forward.

For each signature to migrate, repost its `detection`, then `triage` (if
present), then `final_report` (if present) comments, in that order, onto
the destination issue - each one idempotency-checked first: does the
destination issue already have a comment starting with this exact marker
line? If so, skip reposting it (already migrated by an earlier partial
run), but still count it in Step 6's summary. Otherwise:

```
gh issue comment <destination-issue> --repo <CI_REPO> --body-file <file>
```

where the body is: the original marker line, byte-for-byte, unchanged +
blank line + `_Carried over from #<CLOSED_ISSUE_NUMBER> (originally
posted <original comment's created_at>)_` + blank line + the rest of the
original body unchanged (for a tentatively-resolved signature's
final_report, that means the Step-4 crossout banner travels with it
as-is - don't re-cross it out again, it's already idempotency-checked
not to need it).

Re-apply the category label to the destination issue so migrated
signatures don't silently lose label coverage - read the category from
the signature's `**Category:** <category>` line (present on `triage` and
`final_report`), then:

```
gh label create "<category>" --repo <CI_REPO> --force \
  --color <hex> --description "<one-line description>"
gh issue edit <destination-issue> --repo <CI_REPO> --add-label "<category>"
```

using the same four category/color/description pairs
`debug-nightly-failure/SKILL.md` Step 2 defines (`nightly-build-failure`
`d73a4a`, `nightly-test-failure` `fbca04`, `nightly-config-failure`
`0075ca`, `nightly-infra-failure` `5319e7`) - don't invent a second,
driftable copy of that table, just reuse these exact values.

## Step 6 — Summary comment on the closed issue

Post exactly one comment on `CLOSED_ISSUE_NUMBER`:

```
gh issue comment <CLOSED_ISSUE_NUMBER> --repo <CI_REPO> --body-file <file>
```

First line exactly `<!-- rocm-nightly-cleanup-done -->` - but **only**
include this marker line if you fully classified and handled every
signature found in Step 2 this run (see the budget note above - if you
ran out of time partway through, post the same content without this
marker line instead, as an interim progress comment, so the next hourly
run resumes safely rather than treating this issue as done).

Content: a short list of every signature found, grouped by outcome -
finalized resolved (note `@<login>` of whoever reacted 👍), tentatively
resolved (awaiting confirmation), and migrated as still pending - and, if
anything was migrated at all, a closing line `Continued tracking:
#<destination-issue>`.
