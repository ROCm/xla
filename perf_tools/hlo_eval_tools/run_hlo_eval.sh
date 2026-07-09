#!/usr/bin/env bash
#
# run_hlo_eval.sh - profile the collected `before_optimizations` HLO modules with
# `multihost_hlo_runner` and accumulate per-module timings into CSV files.
#
# Usage:
#   run_hlo_eval.sh <hlo_runner_main> <hlo_path> <out> [num_repeats]
#
#   <hlo_runner_main>  Path to the built multihost_hlo_runner binary, e.g.
#                      .../bazel-bin/xla/tools/multihost_hlo_runner/hlo_runner_main
#   <hlo_path>         What to profile. One of:
#                        - the hlo_eval_tools root (or any subtree, e.g. a single
#                          category or model dir): every `training/` and
#                          `inference/<N>gpu/` leaf beneath it is profiled;
#                        - a single leaf dir (.../training or .../inference/8gpu):
#                          all of its modules are profiled into one CSV;
#                        - a single .txt / .hlo module file.
#   <out>              Where to write CSVs:
#                        - a directory: one CSV per leaf, named
#                          <category>_<model>_<workload>[_<N>gpu].csv;
#                        - a path ending in .csv: all results append into that one
#                          file (only sensible when <hlo_path> is a single leaf/file).
#   [num_repeats]      Executions per module (default 5; the warmup repeat is
#                      skipped in the averaged CSV timing).
#
# For each leaf the device/partition count N is read from the first module's
# header (`num_partitions=N`, default 1). Modules with N>1 were lowered by JAX with
# Shardy, so they are run with `--num_partitions=N --use_shardy_partitioner` and
# HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES=0..N-1 (N devices must be visible).
# `--hlo_argument_mode=uninitialized` is always passed to skip random-input
# generation (much faster startup for the large LLM / diffusion modules). Empty
# leaves (e.g. alphafold3/training) are skipped.
#
# Environment variables:
#   RESUME=1       Skip leaves whose target CSV already exists (resume an
#                  interrupted run without re-profiling finished leaves).
#   SETTLE_SEC=N   Seconds to pause between runner processes (default 2).

set -uo pipefail

die() { echo "error: $*" >&2; exit 2; }

[ "$#" -ge 3 ] || die "usage: $(basename "$0") <hlo_runner_main> <hlo_path> <out> [num_repeats]"
RUNNER=$1
HLO=$2
OUT=$3
REPEATS=${4:-5}

[ -x "$RUNNER" ] || die "runner not found or not executable: $RUNNER"
[ -e "$HLO" ]    || die "hlo path not found: $HLO"

# Seconds to pause between runner processes so a finished run's GPU memory is
# reclaimed before the next launches (back-to-back multi-GPU runs can otherwise
# hit transient allocation/collective-init failures). Override with SETTLE_SEC=0.
SETTLE_SEC=${SETTLE_SEC:-2}

# RESUME=1 skips any leaf whose target CSV already exists and is non-empty, so an
# interrupted suite run can be continued without re-profiling finished leaves.
# Only meaningful in per-leaf (directory) output mode. Default off (append).
RESUME=${RESUME:-0}

# ORDER controls the order leaves are profiled when walking a tree:
#   size (default) - ascending total HLO byte size, so the small/fast models run
#                    first and the biggest (slow-to-compile) models run last;
#   path           - alphabetical by directory path.
ORDER=${ORDER:-size}

# ARG_MODE is the runner's --hlo_argument_mode. Default "uninitialized" (fastest:
# no input generation).
ARG_MODE=${ARG_MODE:-uninitialized}

# CMD_BUFFER=off (default) adds XLA_FLAGS=--xla_gpu_enable_command_buffer= to every
# run, disabling XLA's HIP command buffers (graphs). Required on this ROCm build:
# ~half the models otherwise SIGSEGV inside libamdhip64's graph launch
# (RocmCommandBuffer::LaunchGraph) at execution. Set CMD_BUFFER=on to re-enable.
CMD_BUFFER=${CMD_BUFFER:-off}

# Output mode: a single .csv file vs. a directory of per-leaf CSVs.
SINGLE_CSV=""
if [[ "$OUT" == *.csv ]]; then
  SINGLE_CSV="${OUT%.csv}"                 # the runner appends the .csv itself
  mkdir -p "$(dirname -- "$SINGLE_CSV")"
else
  mkdir -p "$OUT"
fi

# Best-effort count of visible GPUs, only used for a friendly warning (non-fatal).
avail_gpus() {
  local n
  n=$(ls -d /dev/dri/renderD* 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && { echo "$n"; return; }
  echo 8
}
NGPU=$(avail_gpus)

# CSV base name (no extension) for a leaf directory: the path relative to
# hlo_eval_tools with '/' replaced by '_' (matches the README naming, e.g.
# large_language_models_llama3_8b_inference_8gpu). The leaf is resolved to an
# absolute path first so relative inputs (run from inside the tree) still get the
# full <category>_<model>_<workload>[_<N>gpu] name.
csv_base() {
  local p=${1%/}
  p=$(realpath -m -- "$p" 2>/dev/null || printf '%s' "$p")
  case "$p" in
    */hlo_eval_tools/*) p=${p##*/hlo_eval_tools/} ;;
    # Fallback for a tree not living under a 'hlo_eval_tools' dir: keep the
    # workload tail (4 components for inference/<N>gpu, 3 for training).
    */training) p=$(printf '%s\n' "$p" | rev | cut -d/ -f1-3 | rev) ;;
    *)          p=$(printf '%s\n' "$p" | rev | cut -d/ -f1-4 | rev) ;;
  esac
  printf '%s' "${p//\//_}"
}

# num_partitions of a module file (default 1 if the header lacks the token).
file_parts() {
  local n
  n=$(grep -oE 'num_partitions=[0-9]+' "$1" | head -1 | cut -d= -f2)
  echo "${n:-1}"
}

# comma-separated device list 0..N-1 (just "0" for N<=1).
dev_list() { if [ "$1" -gt 1 ]; then seq -s, 0 $(($1 - 1)); else echo 0; fi; }

RAN=(); SKIPPED=(); FAILED=(); RESUMED=()

# Invoke the runner on one or more module files that share the same partition
# count, writing into $csv (path without the .csv extension).
invoke() {
  local n=$1 csv=$2; shift 2
  local devs; devs=$(dev_list "$n")
  local -a args=(--device_type=gpu)
  if [ "$n" -gt 1 ]; then
    args+=(--num_partitions="$n" --use_shardy_partitioner)
    [ "$n" -le "$NGPU" ] || echo "  WARN: needs $n GPUs but only ~$NGPU visible"
  fi
  args+=(--hlo_argument_mode="$ARG_MODE"
         --num_repeats="$REPEATS"
         --profile_execution=true
         --append_profile_to_csv_file="$csv"
         "$@")
  echo "  run: N=$n, $# module(s), devices=[$devs] -> ${csv}.csv"
  local xf="${XLA_FLAGS:-}"
  [ "$CMD_BUFFER" = off ] && xf="--xla_gpu_enable_command_buffer= $xf"
  if HIP_VISIBLE_DEVICES="$devs" CUDA_VISIBLE_DEVICES="$devs" XLA_FLAGS="$xf" "$RUNNER" "${args[@]}"; then
    RAN+=("${csv}.csv")
  else
    FAILED+=("$csv"); echo "  FAIL: runner exited non-zero -> $csv"
  fi
  [ "$SETTLE_SEC" -gt 0 ] 2>/dev/null && sleep "$SETTLE_SEC" || true
}

# Emit the training/ and inference/<N>gpu/ leaves under $1, ordered per $ORDER:
# "size" (default) sorts by ascending total HLO byte size (small models first,
# biggest last); "path" sorts alphabetically.
discover_leaves() {
  local base=$1 leaves
  leaves=$(find "$base" -type d \( -name training -o -name '[0-9]*gpu' \))
  [ -n "$leaves" ] || return 0
  if [ "$ORDER" = path ]; then
    printf '%s\n' "$leaves" | sort
  else
    # Zero-pad the size so a plain lexical sort orders numerically.
    printf '%s\n' "$leaves" | while IFS= read -r d; do
      [ -n "$d" ] || continue
      local sz
      sz=$(stat -c %s "$d"/*.txt "$d"/*.hlo 2>/dev/null | awk '{s+=$1} END{printf "%d", s+0}')
      printf '%015d\t%s\n' "$sz" "$d"
    done | sort | cut -f2-
  fi
}

# Profile every module in one leaf directory into a single CSV.
run_leaf() {
  local leaf=$1
  local -a files=()
  local f
  for f in "$leaf"/*.txt "$leaf"/*.hlo; do [ -e "$f" ] && files+=("$f"); done
  if [ "${#files[@]}" -eq 0 ]; then
    SKIPPED+=("$leaf"); echo "  skip (empty): $leaf"; return
  fi
  local n; n=$(file_parts "${files[0]}")
  local csv=${SINGLE_CSV:-"$OUT/$(csv_base "$leaf")"}
  if [ -z "$SINGLE_CSV" ] && [ "$RESUME" != 0 ] && [ -s "${csv}.csv" ]; then
    RESUMED+=("$leaf"); echo "leaf: $leaf"; echo "  skip (resume, CSV exists): ${csv}.csv"; return
  fi
  echo "leaf: $leaf"
  invoke "$n" "$csv" "${files[@]}"
}

echo "runner : $RUNNER"
echo "hlo    : $HLO"
if [ -n "$SINGLE_CSV" ]; then echo "out    : ${SINGLE_CSV}.csv (single file)"; else echo "out    : $OUT/ (one CSV per leaf)"; fi
echo "repeats: $REPEATS   order: $ORDER   arg_mode: $ARG_MODE   cmd_buffer: $CMD_BUFFER   visible GPUs (approx): $NGPU"
echo

if [ -f "$HLO" ]; then
  n=$(file_parts "$HLO")
  csv=${SINGLE_CSV:-"$OUT/$(csv_base "$(dirname "$HLO")")"}
  echo "file: $HLO"
  invoke "$n" "$csv" "$HLO"
else
  # Directory: a leaf (modules directly inside) or a tree to walk.
  shopt -s nullglob
  direct=("$HLO"/*.txt "$HLO"/*.hlo)
  shopt -u nullglob
  if [ "${#direct[@]}" -gt 0 ]; then
    run_leaf "$HLO"
  else
    # Discover every training/ and inference/<N>gpu/ leaf beneath $HLO,
    # ordered per $ORDER (small models first by default).
    while IFS= read -r leaf; do
      run_leaf "$leaf"
    done < <(discover_leaves "$HLO")
  fi
fi

echo
echo "==== summary ===="
echo "profiled : ${#RAN[@]} CSV(s)"
[ "${#RESUMED[@]}" -gt 0 ] && echo "resumed  : ${#RESUMED[@]} leaf(s) skipped (CSV already present)"
[ "${#SKIPPED[@]}" -gt 0 ] && echo "skipped  : ${#SKIPPED[@]} empty leaf(s)"
if [ "${#FAILED[@]}" -gt 0 ]; then echo "failed   : ${#FAILED[@]}"; printf '  - %s\n' "${FAILED[@]}"; fi
if [ "${#RAN[@]}" -gt 0 ]; then echo "CSV files:"; printf '%s\n' "${RAN[@]}" | sort -u | sed 's/^/  /'; fi

# Exit non-zero if any leaf failed, so the suite can be scripted / used in CI.
[ "${#FAILED[@]}" -eq 0 ]
