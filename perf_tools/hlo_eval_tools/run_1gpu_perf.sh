#!/usr/bin/env bash
#
# run_1gpu_perf.sh - run the 1-GPU inference + training perf passes
# (run_inference_1gpu.sh + run_training_1gpu.sh), tar up the resulting CSV
# dir, and print a TSV summary.
#
# Usage:
#   run_1gpu_perf.sh <hlo_runner_main> <results-dir> [num_repeats]
#
#   <hlo_runner_main>  Path to the built multihost_hlo_runner binary.
#   <results-dir>      Output dir for the per-model CSVs (created if missing).
#   [num_repeats]       Executions per module (default 5); forwarded as-is.
#
# Produces:
#   <results-dir>/*.csv   one CSV per model/leaf
#   <results-dir>.tar     tar of <results-dir>, for copying between machines
#   <results-dir>.tsv     TSV: model  leaf  module  duration_ms (also on stdout)
#
# RESUME=1 (default, set by the underlying tools) skips leaves whose CSV
# already exists, so re-running after a partial/failed pass is safe.

set -uo pipefail

die() { echo "error: $*" >&2; exit 2; }

[ "$#" -ge 2 ] || die "usage: $(basename "$0") <hlo_runner_main> <results-dir> [num_repeats]"
RUNNER=$1
OUT=$2
REPEATS=${3:-5}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

[ -x "$RUNNER" ] || die "runner not found or not executable: $RUNNER"

mkdir -p "$OUT"

fail=0
"$HERE/run_inference_1gpu.sh" "$RUNNER" "$OUT" "$REPEATS" || fail=1
"$HERE/run_training_1gpu.sh" "$RUNNER" "$OUT" "$REPEATS" || fail=1

TAR="${OUT%/}.tar"
tar -cf "$TAR" "$OUT"
echo "packed $OUT -> $TAR" >&2

# Emit a TSV summary: model  leaf  module  duration_ms
# Each CSV is named <model>_<leaf>.csv (leaf = inference_1gpu | training) and
# holds one header row of module names plus one data row of durations, e.g.:
#   Datetime,module_0087.jit__prefill_jit....txt,module_2266.jit__generate_jit....txt
#   2026-07-31 16:28:51, 24.52ms, 25.89ms
{
  printf 'model\tleaf\tmodule\tduration_ms\n'
  for csv in "$OUT"/*.csv; do
    [ -e "$csv" ] || continue
    base=$(basename "$csv" .csv)
    if [[ "$base" == *_inference_1gpu ]]; then
      leaf=inference_1gpu
      model=${base%_inference_1gpu}
    elif [[ "$base" == *_training ]]; then
      leaf=training
      model=${base%_training}
    else
      leaf=unknown
      model=$base
    fi

    IFS=',' read -r -a headers < <(head -1 "$csv")
    IFS=',' read -r -a values  < <(sed -n '2p' "$csv")
    for i in "${!headers[@]}"; do
      [ "$i" -eq 0 ] && continue   # skip Datetime column
      module=$(echo "${headers[$i]}" | xargs | grep -oE '^module_[0-9]+')
      duration=$(echo "${values[$i]:-}" | xargs | sed 's/ms$//')
      [ -n "$duration" ] || continue
      printf '%s\t%s\t%s\t%s\n' "$model" "$leaf" "$module" "$duration"
    done
  done
} | tee "$OUT.tsv"

exit "$fail"
