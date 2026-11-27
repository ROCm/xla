---
name: xla-pr-description
description: Write a short PR description for an openxla/xla upstream PR by filling in its pull request template (Summary, Justification, Kind of Contribution, Benchmark, Unit/Execution Tests) from the branch's commits and diff. Use when asked to write, fill in or draft an XLA PR description or PR body.
argument-hint: "[git ref or range, default: branch vs merge base with main]"
---

# Fill in the openxla/xla PR template

## Gather

1. Find the change: `git log` and `git diff` of the branch against its merge
   base with upstream `main` (or the ref/range given). Ignore upstream files
   a stale merge base drags in.
2. Read the template from the checkout, `.github/pull_request_template.md`.
   If it is missing: `gh api repos/openxla/xla/contents/.github/pull_request_template.md --jq .content | base64 -d`.
3. Look for the facts the description needs and the diff doesn't show: the
   problem or workload behind the change, issue links, benchmark numbers. Take
   them from the conversation; if a section needs one and you don't have it,
   ask rather than invent it.

## Rules

- On merge the PR description is squashed into the commit message, after
  the title. Don't restate the title or the commit message.
- Keep only the sections that apply. Drop Benchmark unless it is a perf
  change, and drop the test sections when no tests were added. Under Kind of
  Contribution, pick the one kind that fits best; add a second only if the
  change really is two things.
- Say why, not how. The diff shows how. Give the problem, who hits it, and
  why this fix. Don't explain anything a reviewer can read off the diff,
  including config values, tag filters, flag lists, or which options do
  or don't overlap.
- Be short: at most two sentences per section, one is better. Name the
  change by its identifier (`--config=foo`, `FooPass`) instead of
  describing it. No intro, no recap, no "This PR...", no lists of changed
  files, no marketing adjectives ("robust", "seamless", "comprehensive"),
  no filler.
- After drafting, cut every clause a reviewer wouldn't miss, then cut
  again.
- Replace every template prompt with real content or remove its section.
  Keep the template's emoji headings for the sections you keep.
- Benchmark: real measured numbers on a public HLO from
  `xla/tools/benchmarks/hlo/` (or what the user measured), with GPU and
  baseline. Report Δ time as (new − base)/base, so negative means faster.
  Never make up numbers.
- Tests: name the test targets or cases added and what each checks, in one
  line each.

## Output

The terminal indents fenced code blocks, so a pasted fenced block picks up
leading spaces. Instead:

1. Write the description to `/tmp/pr_desc.md`.
2. Copy it to the clipboard with whichever tool exists:

   ```bash
   f=/tmp/pr_desc.md
   if command -v clip.exe >/dev/null; then   # WSL, Git Bash; clip.exe wants UTF-16LE for emoji
     iconv -f utf-8 -t utf-16le "$f" | clip.exe
   elif command -v pbcopy >/dev/null; then pbcopy < "$f"
   elif [ -n "$WAYLAND_DISPLAY" ] && command -v wl-copy >/dev/null; then wl-copy < "$f"
   elif command -v xclip >/dev/null; then xclip -selection clipboard < "$f"
   elif command -v xsel >/dev/null; then xsel --clipboard --input < "$f"
   else echo "no clipboard tool"; fi
   ```

3. Also print the description as plain text, not in a code fence, and say
   whether it was copied to the clipboard or only saved to the file.

Don't open or edit the PR unless asked; if asked, use
`gh pr edit <n> --body-file /tmp/pr_desc.md`.

Example of the expected density:

```markdown
📝 Summary of Changes
Fall back to hipBLASLt when the Triton GEMM autotuner finds no valid config
on gfx950, instead of failing compilation.

🎯 Justification
Small-K bf16 GEMMs in Llama-3 8B prefill have no valid Triton tiling on
gfx950, so compilation fails. cuBLAS has always been the fallback on CUDA.

🚀 Kind of Contribution
🐛 Bug Fix

🧪 Unit Tests:
`gemm_fusion_autotuner_test`: `NoValidConfigFallsBackToLibrary` checks the
fusion is rewritten to a hipBLASLt custom call.
```
