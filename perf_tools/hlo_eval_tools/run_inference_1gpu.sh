#!/usr/bin/env bash
#
# run_inference_1gpu.sh - profile only the single-GPU inference leaves
# (`*/*/inference/1gpu`) for every model, via run_hlo_eval.sh.
#
# This is the fast, bounded first pass: single-device only, so no leaf can fail
# on GPU count and nothing needs the Shardy / multi-GPU flags.
#
# Usage:
#   run_inference_1gpu.sh <hlo_runner_main> <out> [num_repeats]
#
#   <hlo_runner_main>  Path to the built multihost_hlo_runner binary.
#   <out>              Output dir (one CSV per model) or a .csv path.
#   [num_repeats]      Executions per module (default 5); forwarded to run_hlo_eval.sh.
#
# RESUME=1 (default) skips leaves whose CSV already exists, so the pass is
# restartable. Leaves are run smallest-HLO first so SDXL / SD1.5 land last.

set -uo pipefail

die() { echo "error: $*" >&2; exit 2; }

[ "$#" -ge 2 ] || die "usage: $(basename "$0") <hlo_runner_main> <out> [num_repeats]"
RUNNER=$1
OUT=$2
REPEATS=${3:-5}

[ -x "$RUNNER" ] || die "runner not found or not executable: $RUNNER"

# Resolve this script's directory so the HLO tree is found regardless of CWD.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Collect every single-GPU inference leaf, skipping empty ones.
leaves=()
for leaf in "$HERE"/*/*/inference/1gpu; do
  files=("$leaf"/*.txt)
  [ -e "${files[0]}" ] && leaves+=("$leaf")
done

[ "${#leaves[@]}" -gt 0 ] || die "no inference/1gpu leaves found under $HERE"

# Sort leaves by total HLO byte size (ascending) so the biggest run last.
mapfile -t leaves < <(
  for leaf in "${leaves[@]}"; do
    sz=$(du -sb "$leaf" | cut -f1)
    printf '%s\t%s\n' "$sz" "$leaf"
  done | sort -n | cut -f2-
)

echo "profiling ${#leaves[@]} single-GPU inference leaves -> $OUT"

total=${#leaves[@]}
n=0
fail=0
for leaf in "${leaves[@]}"; do
  n=$((n + 1))
  echo "=== [$n/$total] $leaf ==="
  # Drop the runner's glog / allocator noise; keep only the per-repeat timing
  # lines ("## Execution time, file=... duration=...ns"). Combine stderr+stdout
  # so both streams are filtered; PIPESTATUS[0] preserves the run's exit code.
  RESUME=${RESUME:-1} \
  "$HERE/run_hlo_eval.sh" "$RUNNER" "$leaf" "$OUT" "$REPEATS" 2>&1 \
    | grep --line-buffered -E 'Execution time'
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
done

exit "$fail"
