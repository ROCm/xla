#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <clean-xla-source-repo> [new-output-dir]" >&2
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || {
  usage
  exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STABILITY_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(git -C "$STABILITY_ROOT" rev-parse --show-toplevel)
SOURCE=$(realpath "$1")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT=${OUTPUT_ROOT:-/tmp}
OUT=${2:-"$OUTPUT_ROOT/hlo-stability-4target-24round-$STAMP"}

TARGETS="$STABILITY_ROOT/configs/xla_targets.template.json"
HLO="$REPO_ROOT/perf_tools/hlo_eval_tools/vision_diffusion/efficientnet/inference/1gpu/module_0961.jit_predict_step.before_optimizations.txt"
REFERENCE="$REPO_ROOT/perf_tools/hlo_eval_tools/vision_diffusion/efficientnet/results/inference_1gpu.csv"
LOG="${OUT}.console.log"
exec > >(tee "$LOG") 2>&1

test -f "$TARGETS"
test -f "$HLO"
test -f "$REFERENCE"
test ! -e "$OUT"

echo "Source: $SOURCE"
echo "Targets: $TARGETS"
echo "Output: $OUT"
echo "Console log: $LOG"
echo "Status command:"
echo "  python3 $STABILITY_ROOT/scripts/show_hlo_stability_status.py --output-dir $OUT --follow"

collector_pid=
forward_signal() {
  signal_name=$1
  trap - INT TERM HUP
  if [ -n "$collector_pid" ] && kill -0 "$collector_pid" 2>/dev/null; then
    kill -s "$signal_name" "$collector_pid"
    wait "$collector_pid"
    exit $?
  fi
  exit 128
}
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM
trap 'forward_signal HUP' HUP

set +e
PYTHONUNBUFFERED=1 \
python3 "$STABILITY_ROOT/scripts/run_hlo_stability.py" \
  --xla-source-repo "$SOURCE" \
  --output-dir "$OUT" \
  --hlo-path "$HLO" \
  --targets-file "$TARGETS" \
  --reference-csv "$REFERENCE" \
  --rounds 24 \
  --warmup-cooldown-sec 8 \
  --target-cooldown-sec 8 \
  --round-cooldown-sec 30 \
  --runner-settle-sec 2 \
  --capture-system-snapshots &
collector_pid=$!
wait "$collector_pid"
collector_rc=$?
set -e
trap - INT TERM HUP

if [ "$collector_rc" -ne 0 ]; then
  echo "stability collection failed with exit code $collector_rc" >&2
  exit "$collector_rc"
fi

echo "HTML report: $OUT/stability_report.html"
