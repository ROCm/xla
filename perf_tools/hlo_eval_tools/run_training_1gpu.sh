#!/usr/bin/env bash
#
# run_training_1gpu.sh - profile only the single-GPU training leaves
# (`*/*/training` with num_partitions=1) for every model, via run_hlo_eval.sh.
#
# This is the fast, bounded first pass: single-device only, so no leaf needs the
# Shardy / multi-GPU flags. The six MaxText LLMs dump training on 8 GPUs
# (num_partitions=8, FSDP) and are skipped here; run them with the multi-GPU
# path. Empty training leaves (alphafold3, colabfold - no public training path)
# are skipped too.
#
# Usage:
#   run_training_1gpu.sh <hlo_runner_main> <out> [num_repeats]
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

# num_partitions of a module file (default 1 if the header lacks the token).
file_parts() {
  local n
  n=$(grep -oE 'num_partitions=[0-9]+' "$1" | head -1 | cut -d= -f2)
  echo "${n:-1}"
}

# Collect every single-GPU training leaf, skipping empty ones and any leaf whose
# modules were lowered for >1 partition (the 8-GPU MaxText LLM training dumps).
leaves=()
for leaf in "$HERE"/*/*/training; do
  files=("$leaf"/*.txt)
  [ -e "${files[0]}" ] || continue
  [ "$(file_parts "${files[0]}")" -eq 1 ] || continue
  leaves+=("$leaf")
done

[ "${#leaves[@]}" -gt 0 ] || die "no single-GPU training leaves found under $HERE"

# Sort leaves by total HLO byte size (ascending) so the biggest run last.
mapfile -t leaves < <(
  for leaf in "${leaves[@]}"; do
    sz=$(du -sb "$leaf" | cut -f1)
    printf '%s\t%s\n' "$sz" "$leaf"
  done | sort -n | cut -f2-
)

echo "profiling ${#leaves[@]} single-GPU training leaves -> $OUT"

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
