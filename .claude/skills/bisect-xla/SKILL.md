---
name: bisect-xla
description: Find the XLA commit that broke a unit test or regressed performance. Covers unit-test bisects with `git bisect run bazel test` and performance bisects with the bisect_run.sh helper. Use when asked which commit broke a test, caused a crash, or made a model/HLO module slower, or when asked to run or set up a git bisect in XLA.
argument-hint: [ut|perf] [test-target-or-module] [good-rev] [bad-rev]
---

# Bisect an XLA regression (ROCm)

Two flows. Pick by symptom, not by habit:

| Symptom | Flow | Verdict comes from |
|---|---|---|
| Test fails, crashes, hangs, CHECK-fails | **UT** | `bazel test` exit code |
| Model/HLO module got slower (or faster) | **perf** | `bisect_run.sh` runtime vs threshold |
| An HLO module crashes or emits a given error | **perf** | `bisect_run.sh --mode status` / `--mode grep --pattern <re>` |

Both flows cost `log2(range) × build_time`. A 400-commit range is ~9 builds. Say
that estimate out loud before starting — if each build is 20 minutes, the user is
committing to three hours and may prefer a narrower range.

## Locate the pieces first

Nothing below assumes a fixed layout. Resolve these once, then use the variables:

- **`$BISECT_TREE`** — the XLA checkout to bisect. `git bisect` checks out
  arbitrary revisions and leaves a detached HEAD, so it must not be a checkout
  the user is working in. **Ask which tree to use** unless the user already said:
  clones and worktrees can live anywhere, and guessing by directory name picks up
  look-alike scratch dirs (`bisect-logs/`, `bisect-bins/`). Confirm the answer is
  an XLA checkout — `git -C <dir> rev-parse --show-toplevel` and an `xla/`
  directory. A git worktree is fine; `git worktree list` enumerates them from
  inside any checkout of the same repo.
- **`$MODULES`** — the HLO/StableHLO corpus, `perf_tools/hlo_eval_tools` in this
  repo, laid out as `<family>/<model>/<inference|training>/<N>gpu/module_*.mlirbc`.
- **`$BISECT_RUN`** — the perf driver, `$MODULES/bisect_run.sh`, documented in
  that directory's `README.md` ("Bisecting to a commit") and by `--help`.
  Resolve it from this repo's root (`git rev-parse --show-toplevel` while inside
  it), never by a relative hop from wherever you happen to be. It locates the
  tree it bisects from the working directory or `--repo-dir`, so it does not
  care where it is stored.
- **`$DISK_CACHE`** — the bazel disk cache the target tree already builds with
  (`$BAZEL_DISK_CACHE`, or recover it from that tree's build history). Reusing it
  is what keeps a bisect to minutes per step instead of hours.

Derive, never hardcode: `GFX=$(rocminfo | grep -om1 'gfx[0-9a-f]*')` for the GPU
target, `$(bazel info output_base)` for anything under the output base.

## Before `git bisect start`

1. `git -C "$BISECT_TREE" status --porcelain` — **modified tracked files block
   the checkouts.** Stash them and say so. Untracked scratch files are fine and
   survive checkouts.
2. Confirm the range: `git log --oneline <good>..<bad> | wc -l`. Modern XLA
   history is linear copybara imports with no merges, so there is a clean line
   to walk. (Only pre-2023 TF-era history has merges; `git bisect start
   --first-parent` handles it.)
3. **Do one manual build at HEAD with the exact flags the bisect will use, and
   confirm it is a cache hit.** This is the single most common way to waste an
   afternoon: if the ROCm source does not match how that tree's `bazel-bin` was
   last built, the first revision re-fetches `local_config_rocm` and rebuilds
   everything ROCm-dependent.

A build line of the usual shape:

```bash
bazel build --config=rocm_clang_hermetic \
  --repo_env=TF_ROCM_AMDGPU_TARGETS="$GFX" \
  --disk_cache="$DISK_CACHE" --dynamic_mode=off \
  //xla/tools/multihost_hlo_runner:hlo_runner_main
```

Add `--repo_env=ROCM_PATH= --repo_env=ROCM_DISTRO_URL=<tarball>
--repo_env=ROCM_DISTRO_HASH=no_hash` if the tree builds against a hermetic
distro rather than a local ROCm install (`cat "$(bazel info
output_base)"/external/*local_config_rocm/rocm/rocm_dist/.info/version` shows
which it holds). Recover the rest of the flags from that tree's shell history or
helper scripts rather than inventing them, and if a team remote cache is in
play, check it is reachable before passing `--remote_cache` — an unreachable one
fails every build, so every revision looks "bad".

## Launch it so it survives

A bisect runs for hours, far past any foreground command timeout. Give each run
its own scratch directory — the TSV is appended to, so reusing a path silently
interleaves this run's rows with an older one's — then start it detached and
record the pid:

```bash
OUT=$(mktemp -d /tmp/bisect-XXXXXX)
nohup git bisect run <cmd...> > "$OUT/progress.log" 2>&1 & echo $! > "$OUT/pid"
```

**Wait on the pid, not on new log output:** `tail --pid=$(cat "$OUT/pid") -f
/dev/null` blocks until it exits and returns immediately if it already has.
Two ways to miss the ending, both observed in practice: `tail -f -n 0` only
shows lines written *after* it attaches, so a result written seconds earlier
never appears; and `pgrep -f 'git bisect run'` matches your own shell command
containing that string, reporting RUNNING forever.

If you poll text instead, grep the whole file, never a live tail. The run is
over when it matches:
`is the first bad commit|cannot continue any more|bisect run failed`
(found / only skips left / the script returned ≥128).

State lives in `.git/BISECT_*` and survives a dropped session (`git bisect log`
to resume). Never start a second bisect in the same tree while one is live.

## Flow 1 — unit test

Start with the one-liner. No script needed:

```bash
cd "$BISECT_TREE"
git bisect start <bad-rev> <good-rev>
git bisect run bazel test --config=rocm_clang_hermetic \
    --repo_env=TF_ROCM_AMDGPU_TARGETS="$GFX" --dynamic_mode=off \
    --disk_cache="$DISK_CACHE" --test_output=errors \
    //xla/service/gpu:some_test
```

This is correct **only if every revision in the range builds and the test target
exists throughout**. Otherwise you must map bazel's exit codes, because
`git bisect` calls everything non-zero "bad":

| bazel rc | meaning | bisect should |
|---|---|---|
| 0 | passed | good (0) |
| 3 | test failed, build fine | **bad** (1) |
| 1 | build / analysis error | **skip** (125) |
| 4 | no such test target | **skip** (125) — the test was added mid-range |
| other | interrupted, internal | skip (125) |

When the range contains build breaks or the test was added partway, write this
wrapper to a scratch dir **outside** the repo (`$CLAUDE_JOB_DIR/tmp` if set, else
`/tmp`) so bisect checkouts cannot touch it, and bisect with that:

```bash
#!/usr/bin/env bash
# bisect_ut.sh <bazel-test-args...>  -- maps bazel exit codes to git-bisect ones.
bazel test "$@"; rc=$?
case $rc in
  0) exit 0 ;;    # passed  -> good
  3) exit 1 ;;    # failed  -> bad
  *) exit 125 ;;  # unbuildable / missing target -> skip
esac
```

Narrow what runs. A whole test binary is slow and invites unrelated failures:

- `--test_filter=SuiteName.TestName` — bisect the one failing case.
- `--runs_per_test=10` for a flaky failure; bazel reports FAILED if any run fails,
  which is usually what you want when hunting the commit that introduced a flake.
- `--nocache_test_results` if a cached PASS might be masking a real run.
- To bisect a *specific* failure rather than any failure, grep the test log and
  skip on anything else — an unrelated breakage mid-range otherwise sends the
  bisect down the wrong half. The log lands under
  `bazel-testlogs/<package-path>/<test-name>/test.log`.

## Flow 2 — performance

Driver: **`$BISECT_RUN`**. Its `--help` and README are complete — read them
rather than improvising flags, and rather than expecting them restated here. It
builds `hlo_runner_main` per revision, times the module(s), and classifies
against a threshold. HLO-runner only by design: do not bisect a unit test with
it.

It decides on its own when a revision is untestable (build failure, missing or
partial timings, a runner that failed after timing everything, one too old for
the module format) and emits a skip. If *every* revision skips because a
non-zero exit is systematic in your range, `--judge-on-run-error` judges them
anyway rather than abandoning the bisect. You
do not need to pre-check any of that — but a range reaching back before July
2026 will skip every revision if you feed it `.mlirbc`, so use pre-lowered
`.txt` HLO for ranges that old.

### Set the threshold from measurement, never from a guess

A perf bisect is only as good as the gap between the endpoints. Measure both
first, with the flags you will bisect with:

```bash
git checkout <good-rev> && "$BISECT_RUN" --module <module> --threshold 0 ...
git checkout <bad-rev>  && "$BISECT_RUN" --module <module> --threshold 0 ...
```

`--threshold 0` makes every result BAD, so the script exits 1 — expected here,
you are only reading the printed metric. Add `--skip-build` once that revision is
already built. Put the threshold **midway between the two medians**. If the
endpoint gap is smaller than the run-to-run spread, stop and say so: the bisect
cannot resolve it, and you need more repeats, a quieter machine, or a module with
a bigger delta.

### Run it

```bash
git bisect start <bad-rev> <good-rev>
git bisect run "$BISECT_RUN" \
    --module "$MODULES/large_language_models/flan_t5_large/inference/1gpu/module_0199.jit_predict_step.before_optimizations.mlirbc" \
    --threshold 12 --trials 3 \
    --autotune-cache <merged-cache>.textproto \
    --tsv bisect-log.tsv --keep-logs bisect-logs
```

Things that matter, in order of how often they bite:

- **`--tsv`, not `--log`.** `git bisect run` eats a bare `--log` and passes only
  its value through, so the script aborts at the first revision.
- **`--autotune-cache` is close to mandatory.** Without a pinned cache, autotune
  noise and any commit that merely perturbs autotuning drive the verdict instead
  of the change you are hunting. Build one with
  `--xla_gpu_dump_autotune_results_to=`, chained across the modules you care
  about.
- **`--trials 3`** suppresses one-sided process noise. Check the per-trial spread
  in the TSV; on a quiet host it should be well under 1%.
- **Hunting a speedup or a fix?** `--invert`, and start the bisect the other way
  round (`git bisect start <fast-rev> <slow-rev>`).
- `--dry-run` prints the exact commands without touching anything. Use it before
  committing hours.

Watch the TSV while it runs: `rev / metric_ms / samples / trials / build_s /
verdict / detail`. A column of medians that steps cleanly between two levels is a
trustworthy bisect; one that jitters across the threshold is not — raise
`--num-repeats`/`--trials`, or pick a module with a bigger delta.

## When it finishes

1. `git bisect log` — keep it; it records what was tested, and `git bisect replay`
   re-runs it if you need to redo the range with a different threshold.
2. **Verify the culprit directly.** Build and measure the first-bad commit and its
   parent back to back. A bisect fed a noisy threshold can land one commit off,
   and that pair is the cheapest possible check.
3. Read the commit: `git show --stat <rev>`; the `PR #NNNNN` in the subject gives
   you the upstream PR to cite.
4. `git bisect reset` when done — leaving the tree on a detached HEAD is how the
   next session gets confused.

Report: the first bad commit (sha + PR number + one-line subject), the metric
before and after (or the failure signature), which module/test and flags produced
it, and where the TSV and logs live.

## Pitfalls

- Skip often enough and bisect reports a *range* of candidates rather than one
  commit — that is a real answer, not a failure; report the range.
- If `MODULE.bazel` / `MODULE.bazel.lock` changes inside the range, expect repo
  re-fetches at those revisions and a slower bisect. Keep one bazel output base
  for the whole run so the disk cache keeps paying off.
- Timing runs compete for the GPU. Check nothing else is running (`amd-smi list`,
  stray `hlo_runner_main` processes) before trusting a perf verdict, and never run
  two bisects on one GPU.
- Don't bisect XLA and a ROCm distro change at once. If the suspect is a ROCm
  version rather than an XLA commit, hold XLA fixed and sweep the distro instead.
