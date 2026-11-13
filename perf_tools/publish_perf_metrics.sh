#!/usr/bin/env bash
#
# publish_perf_metrics.sh - the publish half of the ROCm perf nightly.
#
# The matrix jobs upload one artifact per mode; this turns them into a single
# commit on the data branch and then deletes the artifacts that carried them.
# Driven by .github/workflows/rocm_xla_perf_nightly.yml, one subcommand per
# step so that each stage still shows up separately in the Actions UI:
#
#   publish_perf_metrics.sh assemble   # artifacts -> a run directory
#   publish_perf_metrics.sh push       # run directory -> the data branch
#   publish_perf_metrics.sh cleanup    # drop the transport artifacts
#
# Environment, all provided by the workflow:
#   RUNNER_TEMP         Scratch dir; holds both the downloaded artifacts and
#                       the assembled payload.
#   STAMP, SHA6         Run stamp and short XLA commit; name the run directory.
#   ISO, XLA_SHA        Full timestamp and commit, recorded in run.json.
#   MODES_JSON          The benchmark matrix as JSON, exactly as rocm-config
#                       built it, recorded in run.json. Optional: unset outside
#                       the workflow, where the matrix is not knowable.
#   THEROCK_VERSION     The nightly TheRock version rocm-config detected, also
#                       recorded in run.json. Optional, as above.
#   GITHUB_OUTPUT       Step outputs; `assemble` writes `empty` here.
#   GITHUB_REPOSITORY   owner/repo, for the push URL and the artifact API.
#   GITHUB_RUN_ID       Recorded in run.json; also selects the artifacts to
#                       delete.
#   DATA_BRANCH         Branch the results are published to.
#   PUSH_TOKEN          Token for the push (`push` only).
#   GH_TOKEN            Token for the artifact API, read by gh (`cleanup` only).

set -euo pipefail

die() { echo "error: $*" >&2; exit 2; }

# Where the artifacts land, and where the commit is staged from. Kept here so
# the subcommands cannot disagree about either path.
artifacts_dir() { echo "${RUNNER_TEMP}/artifacts"; }
payload_dir()   { echo "${RUNNER_TEMP}/publish-payload"; }
run_dir_rel()   { echo "data/${STAMP}-${SHA6}"; }

# Copy every downloaded artifact into data/<stamp>-<sha6>/<mode>/ and write the
# run manifest. Sets the `empty` step output so the later steps can skip when
# no mode produced anything.
cmd_assemble() {
  : "${RUNNER_TEMP:?}" "${STAMP:?}" "${SHA6:?}" "${ISO:?}" "${XLA_SHA:?}"
  : "${GITHUB_OUTPUT:?}" "${GITHUB_RUN_ID:?}"

  local run_dir="$(payload_dir)/$(run_dir_rel)"
  mkdir -p "$run_dir"

  local modes=() d mode
  for d in "$(artifacts_dir)"/perf-results-*/; do
    [ -d "$d" ] || continue
    mode="$(basename "$d")"
    mode="${mode#perf-results-}"
    mkdir -p "$run_dir/$mode"
    cp -a "$d." "$run_dir/$mode/"
    modes+=("$mode")
  done

  if [ "${#modes[@]}" -eq 0 ]; then
    echo "no result artifacts to publish"
    echo "empty=true" >> "$GITHUB_OUTPUT"
    return 0
  fi

  # Per-run manifest, so the directory is self-describing too. Written to one
  # side and moved into place, so a run directory never holds a half-written
  # manifest: `> run.json` truncates before python runs, and a manifest that
  # exists but does not parse is worse than one that is missing.
  printf '%s\n' "${modes[@]}" |
    python3 -I -c '
import json, os, sys

# The benchmark matrix, as rocm-config built it: every mode with the TheRock
# release it pulled and the digest that pinned it. Recorded here because the
# results are only meaningful against the ROCm build that produced them, and
# nothing in the TSV or CSV rows says which that was.
#
# This is the *whole* matrix, including any mode that failed before it
# published anything -- that is what the run set out to measure. `modes` below
# is the subset that actually landed, so the two together say which failed.
#
# Absent when assemble is run by hand outside the workflow; an unparseable
# value is a bug in the workflow and fails rather than publishing a run whose
# provenance is silently empty.
raw = os.environ.get("MODES_JSON")
matrix = json.loads(raw) if raw else []

print(json.dumps({
    "timestamp": os.environ["ISO"],
    "commit": os.environ["SHA6"],
    "xla_sha": os.environ["XLA_SHA"],
    "repository": "openxla/xla",
    "workflow_run": os.environ["GITHUB_RUN_ID"],
    "therock_version": os.environ.get("THEROCK_VERSION") or None,
    "modes": sorted(l.strip() for l in sys.stdin if l.strip()),
    "matrix": matrix,
}, indent=2))
' > "$run_dir/.run.json.tmp"
  mv "$run_dir/.run.json.tmp" "$run_dir/run.json"

  echo "empty=false" >> "$GITHUB_OUTPUT"
  find "$run_dir" -type f | sort
}

# Commit the assembled run directory onto the data branch.
cmd_push() {
  : "${RUNNER_TEMP:?}" "${STAMP:?}" "${SHA6:?}" "${DATA_BRANCH:?}"
  : "${PUSH_TOKEN:?}" "${GITHUB_REPOSITORY:?}"

  [ -d "$(payload_dir)/data" ] || die "nothing assembled; run 'assemble' first"

  local work="${RUNNER_TEMP}/publish-data"
  rm -rf "$work" && mkdir -p "$work" && cd "$work"
  git init -q
  git config user.name "github-actions[bot]"
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git remote add origin \
    "https://x-access-token:${PUSH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
  if git fetch -q --depth 1 origin "$DATA_BRANCH"; then
    git checkout -q -b "$DATA_BRANCH" FETCH_HEAD
  else
    echo "$DATA_BRANCH does not exist yet; creating it"
    git checkout -q --orphan "$DATA_BRANCH"
    git rm -rqf --ignore-unmatch .
  fi
  cp -a "$(payload_dir)/data" .
  git add -A data
  if git diff --cached --quiet; then
    echo "nothing to commit"
  else
    git commit -q -m "perf-nightly results ${STAMP} (xla ${SHA6})"
    git push -q origin "$DATA_BRANCH"
    echo "published $(run_dir_rel) to $DATA_BRANCH"
  fi
}

# Delete the per-mode artifacts of this run; they were only the transport.
cmd_cleanup() {
  : "${GITHUB_REPOSITORY:?}" "${GITHUB_RUN_ID:?}" "${GH_TOKEN:?}"

  gh api --paginate \
    "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/artifacts" \
    --jq '.artifacts[] | select(.name | startswith("perf-results-")) | "\(.id) \(.name)"' |
    while read -r id name; do
      # Best effort: the results are already published, so a failure to clean
      # up must not fail the run -- retention-days expires it instead.
      if gh api --silent -X DELETE \
           "repos/${GITHUB_REPOSITORY}/actions/artifacts/${id}"; then
        echo "deleted artifact $name"
      else
        echo "warning: could not delete artifact $name ($id)" >&2
      fi
    done
}

case "${1:-}" in
  assemble) cmd_assemble ;;
  push)     cmd_push ;;
  cleanup)  cmd_cleanup ;;
  *) die "usage: $(basename "$0") assemble|push|cleanup" ;;
esac
