#!/usr/bin/env bash
#
# validate.sh - regression-validation harness for rocm_latency_check_tool.
#
# Runs the SAME tool binary under several environments/configs and prints a
# compact QPS + tail-latency table, so a regression shows up as a gap between
# rows. HIP *runtime* flags (e.g. DEBUG_HIP_DYNAMIC_QUEUES) are set in front of
# the binary as an env prefix, never as tool flags -- the tool stays a simple,
# generic benchmark and you pick the runtime behavior on the command line.
#
# Usage:
#   ./validate.sh [path-to-binary]      # default: ./rocm_latency_check_tool
#   DURATION=20 ./validate.sh ./tool    # override per-run duration (seconds)
#
# To compare across ROCm versions, build one binary per version (see README's
# "Building" section) and run this script once per binary.
#
set -uo pipefail

BIN="${1:-./rocm_latency_check_tool}"
DURATION="${DURATION:-10}"

if [[ ! -x "$BIN" ]]; then
  echo "error: '$BIN' not found or not executable" >&2
  echo "usage: $0 [path-to-rocm_latency_check_tool]" >&2
  exit 1
fi

# Pull QPS / mean / p99 / p99.9 / count out of a tool run (last report wins).
summarize() {
  awk '
    /count =/  {c=$3}
    /mean =/   {m=$3}
    / 99% <=/  {p99=$3}
    /99.9% <=/ {p999=$3}
    /QPS =/    {q=$3}
    END { printf "QPS=%-11s mean_us=%-9s p99_us=%-9s p999_us=%-9s (n=%s)\n", q, m, p99, p999, c }
  '
}

# run <row-label> <env-prefix> <tool-args...>
run() {
  local label="$1"; shift
  local envp="$1"; shift
  printf "  %-30s : " "$label"
  env $envp "$BIN" "$@" \
      --duration-sec "$DURATION" --report-interval-sec "$DURATION" 2>/dev/null \
    | summarize
}

echo "binary : $BIN"
echo "config : duration=${DURATION}s/run"
echo

# ---------------------------------------------------------------------------
# Usage case: HIP dynamic hardware-queue regression.
#
# The HIP runtime maps streams onto a small per-device pool of hardware queues
# (GPU_MAX_HW_QUEUES, default 4). With dynamic queue management on
# (DEBUG_HIP_DYNAMIC_QUEUES=1, the runtime default) an idle stream releases its
# HW queue and re-selects one on its next submit; when many streams
# oversubscribe the 4-queue pool this "moving around" serializes dispatch.
#
# Put the tool in a dispatch-bound regime (near-nop op, copies skipped) with 64
# streams on 1 GPU (16x oversubscription) so per-dispatch queue management
# dominates, then compare the runtime flag ON vs OFF.
# ---------------------------------------------------------------------------
echo "== HIP dynamic hardware-queue regression (1 GPU, 64 streams, near-nop dispatch) =="
QCFG=(--num-gpus 1 --streams-per-gpu 64 --consumer-threads 64
      --skip-h2d --skip-d2d --skip-d2h --busy-kernel-ns 0 --warmup-iters 10)
run "DEBUG_HIP_DYNAMIC_QUEUES=1 (default)" "DEBUG_HIP_DYNAMIC_QUEUES=1" "${QCFG[@]}"
run "DEBUG_HIP_DYNAMIC_QUEUES=0 (static)"  "DEBUG_HIP_DYNAMIC_QUEUES=0" "${QCFG[@]}"
echo
echo "A large QPS drop / p99.9 blow-up on the =1 row vs the =0 row is the dynamic"
echo "hardware-queue regression. The default pipeline (run without the flags above)"
echo "hides it, because compute + copy time dominates each request."

# ---------------------------------------------------------------------------
# Add more usage cases below by copying the block above, e.g. a different env
# prefix ("" for none), a different tool config, and two or more run rows to
# compare. Keep runtime flags as the env prefix, not as tool flags.
# ---------------------------------------------------------------------------
