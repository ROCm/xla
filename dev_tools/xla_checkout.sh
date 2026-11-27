#!/usr/bin/env bash
# Create or reuse a worktree of this repo for an XLA ref and print its path.
# See dev_tools/README.md.
set -euo pipefail
die() { echo "xla_checkout.sh: error: $*" >&2; exit 2; }

USAGE="usage: xla_checkout.sh <ref> [<name>]"
case "${1:-}" in -h|--help) echo "$USAGE"; exit 0 ;; esac
[ "$#" -ge 1 ] && [ "$#" -le 2 ] || die "$USAGE"
ref=$1

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
if git -C "$repo" show-ref -q --verify "refs/heads/$ref"; then
  branch=$ref name=${2:-${ref##*/}}
else
  branch= name=${2:-${ref//\//-}}
  if [[ "$ref" == */* ]] && git -C "$repo" remote | grep -qx "${ref%%/*}"; then
    git -C "$repo" fetch -q "${ref%%/*}" "${ref#*/}" || die "$ref: fetch failed"
  fi
fi
sha=$(git -C "$repo" rev-parse -q --verify "$ref^{commit}") || die "$ref: no such git ref"
wt=$repo/worktrees/$name

if [ ! -d "$wt" ]; then
  if [ -n "$branch" ]; then
    git -C "$repo" worktree add -q "$wt" "$branch" >&2
  else
    git -C "$repo" worktree add -q --detach "$wt" "$sha" >&2
  fi
elif cur=$(git -C "$wt" symbolic-ref -q --short HEAD); then
  # On a branch: use it only if it is the branch asked for; never move it.
  [ "$cur" = "$ref" ] || die "$wt has branch $cur checked out, not $ref; pass another <name>"
elif [ -n "$branch" ]; then
  die "$wt is detached, not on branch $ref; pass another <name>"
elif [ "$(git -C "$wt" rev-parse HEAD)" != "$sha" ]; then
  # Same lock file as run_test.sh and rbe_hlo_eval.sh.
  if [ -e "$wt/_hlo_eval_rbe/.lock" ]; then
    exec 9>"$wt/_hlo_eval_rbe/.lock"
    flock -n 9 || die "a run is using $wt"
  fi
  git -C "$wt" diff --quiet HEAD || die "$wt has local changes"
  git -C "$wt" checkout -q --detach "$sha"
fi
echo "$wt"
