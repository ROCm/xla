# Phase II: Specific Candidate Evaluation

Status: historical design record; the standalone stability design supersedes
the manifest-only proposal below.

Implementation update (2026-07-30): the formal independent
`perf_tools/hlo_stability_tools` package builds its own immutable runner bundle
by default from a clean XLA checkout and optionally reuses only a native bundle
created by that package. It shares no campaign schema or orchestrator code with
`hlo_eval_tools`; the older discussion below is retained to document how the
evidence policy and runner-provenance requirements evolved.

## Debug-only stability prototype

The MI350 schema-v2 smoke exposed two related questions that are not yet formal
Phase II requirements:

1. Does a candidate differ from the live pinned control after current-session
   jitter is accounted for?
2. Does the live pinned control reproduce the checked-in historical result?

A debug-only prototype currently addresses these questions with balanced
repeated measurements, system snapshots, robust statistics, outlier flags, and
complete raw-round preservation:

```text
investigations/xla-mi350-hlo-drift/scripts/
  run_hlo_stability_experiment.sh
  resolve_hlo_stability_targets.py
  analyze_hlo_stability.py
  render_hlo_stability_report.py
```

This prototype deliberately remains outside `perf_tools/hlo_eval_tools`:

- It reads runners and metadata from an existing schema-v2 campaign.
- It never modifies checked-in HLOs, historical CSVs, campaign artifacts, or
  product source.
- It writes new evidence only under a user-selected debug output directory.
- It preserves all raw samples and marks outliers instead of deleting them.
- It emits summary, wide paired, and long-form raw CSVs plus a self-contained
  HTML report with a collapsible raw-round appendix.
- It is currently scoped to one smoke HLO and selectable four-target debug
  profiles: broad stability (pinned control, v0.8.2, v0.10.1, main) and the
  v0.10.2 transition (pinned control, v0.10.1, campaign v0.10.2 candidate,
  main). In the final run that candidate resolved to the same commit as the
  pinned control, not a newer HEAD. Complete four-round Williams cycles balance
  execution positions and within-round predecessor effects; cross-round
  transitions are excluded from the balance claim after the fixed round
  cooldown.
- It rejects missing or aliased targets before evaluation by requiring four
  distinct target IDs, slugs, and resolved runner paths.
- It reports descriptive first-half/last-half temporal changes alongside the
  raw trends; this is evidence for review, not a statistical-significance
  claim.

During Phase II, discuss whether to refine this workflow into a formal feature.
Do not promote it automatically. Before promotion, decide:

1. Whether repeated stability evaluation is a recurring PR-review requirement.
2. Whether target selection should come from the candidate file, a separate
   stability configuration, or the completed campaign manifest.
3. Whether collection remains Bash plus Python analysis or moves into the Python
   orchestrator.
4. Which statistical policy is authoritative: round count, balanced ordering,
   cooldowns, percentile method, outlier policy, and materiality threshold.
5. Whether HTML reporting should ingest optional stability outputs and expose
   raw rounds, system snapshots, and normalized candidate/control deltas.
6. How multi-HLO and multi-GPU repetitions should control total runtime.

Formalization requires a separate design review and explicit acceptance criteria.
The current prototype is evidence-gathering tooling, not part of the Phase II
candidate-mode contract below.

## Stability evidence design checkpoint - 2026-07-29

This is a provisional brainstorm record for continued Phase II discussion. It
does not authorize implementation or promotion of the debug scripts.

### Staged direction

- Treat the workflow as **stability evidence gathering**, not release
  certification, CI gating, or a replacement for verification-team processes.
- Use evidence-only language: observed median differences, outlier events,
  temporal patterns, limitations, and suggested next experiments.
- Present trend/score views as higher-is-better relative performance
  (`reference latency / live latency`) under a fixed-work assumption. Keep raw
  latency and latency deltas explicitly labeled in evidence tables.
- Never use historical-to-live drift as an allowable candidate noise band;
  compare candidates to the live control with a separate reporting threshold.
- Optimize for focused investigations rather than broad branch sweeps.
- Start with one user-selected HLO. Multi-HLO use is a later question for
  determining whether an issue generalizes beyond one workload.
- Support two primary workflows:
  - characterize a branch near branch creation or before release;
  - iteratively narrow a suspected regression using known-good, intermediate,
    and known-bad refs or commits.

### Proposed target interface

Use the active target selection already frozen in a completed schema-v2
campaign manifest. The user narrows branches/tags/commits through
`xla_targets.json` before building the campaign; the stability collector does
not maintain a second target-selection format. When reusing a broader archived
campaign as a runner pool, an optional `xla_targets.json` may select a focused
subset of its already-built candidates.

Provisional rules:

- Accept one to three active candidates plus exactly one live control from
  `manifest.comparison_target_ids`, or select one to three matching candidates
  from the broader manifest target inventory using the optional selector.
- Use only manifest-recorded immutable commits and runner paths; display labels,
  source revisions, commits, and runner SHA256 values in preflight and reports.
- Use the completed schema-v2 manifest as the authority for runner identity.
- Never re-resolve moving branches during stability collection. A different
  HEAD requires a new campaign/build.
- Verify each executable runner against the SHA256 recorded by the campaign.
- Do not support arbitrary runner-path overrides or build runners inside the
  stability collector. A future product build-only mode is a separate proposal.
- Snapshot the source campaign manifest hash and structured-target provenance
  into every stability experiment.

### Scheduling direction

- One candidate plus live control: balanced two-target alternation.
- Two candidates plus live control: balanced three-target schedule.
- Three candidates plus live control: four-target Williams schedule.
- More than three candidates should initially be split into focused
  experiments instead of increasing runtime and interpretive complexity.
- Twelve rounds is a useful default across these schedule sizes; twenty-four
  rounds is a stronger confirmation when noise or small deltas require it.

### Regression-analysis refinement

The formal report may need two distinct anchors:

- **live control**: an environmental and historical-reproducibility sentinel;
- **comparison anchor**: an optional user-selected known-good branch or commit
  used to quantify a regression interval.

The first formal version can remain human-guided. Automatic commit bisection is
a possible later layer after target preparation, build reuse, and statistical
policy are proven.

### Topics to continue

1. Whether a later formal version should generalize beyond the debug decision
   of two-to-four total targets.
2. Whether a separate product build-only mode should prepare manifest runners
   without evaluating HLOs; collection itself remains manifest-only.
3. Whether `--anchor-ref` is needed in the first version or pairwise reporting
   is sufficient.
4. Whether one-HLO scope is the formal initial contract and how a later HLO
   list avoids hiding per-HLO behavior behind aggregates.
5. Which statistics are descriptive defaults versus user-configurable policy.
6. How to export one complete evidence bundle so HTML links cannot reference
   missing or stale CSV/JSON artifacts.
7. Maintenance ownership and the boundary with existing CI/release
   qualification.

## Problem

The current workflow is optimized for evaluating the branch list in
`configs/xla_refs.txt`. Users can technically place a commit SHA in that file,
but the workflow has no explicit, user-friendly mode for the common PR-review
case:

> Compare one exact candidate branch or commit against the pinned baseline.

Long commit IDs are inconvenient and error-prone as CLI arguments. Extending the
plain refs file with labels, roles, and commit metadata would also make its
currently simple format harder to understand.

## Proposed user model

Keep the existing default workflow unchanged:

```text
no candidate file
  -> benchmark_profile.json + xla_refs.txt
```

Add an optional runtime candidate mode:

```text
--candidate-file <path>
  -> baseline from benchmark_profile.json
  -> one candidate from the runtime file
  -> xla_refs.txt is not used for target selection
```

Example:

```bash
python3 run_xla_branch_eval.py \
  --xla-source-repo /workspace/codeRepo/xla_debug/xla \
  --candidate-file /tmp/pr-1234.json \
  --output-dir /workspace/debug_space/pr-1234-results
```

## Candidate file

Provide a checked-in template:

```text
configs/candidate.template.json
```

Proposed minimal schema:

```json
{
  "schema_version": 1,
  "name": "pr-1234",
  "revision": "0123456789abcdef0123456789abcdef01234567"
}
```

`revision` may be an exact commit, branch, or tag, but it must resolve to an
immutable commit before the build begins. The manifest records both the
requested revision and resolved commit.

## Configuration ownership

- `benchmark_profile.json`: stable benchmark protocol and baseline.
- `xla_refs.txt`: simple default multi-branch target list.
- Candidate file: runtime description of one specific PR/commit candidate.
- CLI: machine-specific source, output, and candidate-file paths.

The benchmark profile should change only when the baseline or benchmark protocol
changes.

## Resume behavior

- The manifest stores the candidate configuration and its hash.
- Resume reuses the recorded candidate commit.
- The user does not need to repeat the candidate revision on the CLI.
- A different candidate requires a new output directory.
- Supplying a conflicting candidate file during resume must fail clearly.

## Alternatives considered

### `--candidate <commit>`

Simple implementation, but forces users to copy long commit IDs into commands
and makes repeatable PR-review instructions less convenient.

### `xla_targets.json`

Supports structured baseline and multiple candidates, but is more general and
complex than the currently identified one-candidate requirement.

### Extending `xla_refs.txt`

Avoids another file, but would require a custom syntax for labels, roles, and
commit pins. This weakens the simplicity of the default workflow.

## Review questions

Before implementation, confirm:

1. Is baseline-versus-one-specific-commit a recurring user workflow?
2. Must the tool fetch PR refs, or can users make the candidate available
   locally first?
3. Is a readable candidate name required in generated reports?
4. Is one candidate sufficient, or is multi-candidate support needed?
5. Should the candidate file be checked into a PR or normally created under
   `/tmp`?

## Phase II acceptance criteria

If approved:

- Add `--candidate-file`.
- Keep default `xla_refs.txt` behavior unchanged.
- Add and document `candidate.template.json`.
- Validate schema and reject unedited placeholders.
- Resolve and record an immutable candidate SHA.
- Use the profile baseline automatically.
- Include the candidate name and commit in reports.
- Preserve candidate identity across resume.
- Reject ambiguous combinations such as `--candidate-file` with `--refs-file`.
