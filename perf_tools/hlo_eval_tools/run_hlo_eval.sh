#!/usr/bin/env bash
#
# run_hlo_eval.sh - profile the collected `before_optimizations` HLO modules with
# `multihost_hlo_runner` and accumulate per-module timings into CSV files.
#
# Usage:
#   run_hlo_eval.sh <hlo_runner_main> <hlo_path> <out> [num_repeats]
#   run_hlo_eval.sh --branches --xla-source-repo <repo> --output-dir <dir>
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
#   STAT=mean|median
#                  Which statistic over the timed repeats the runner writes to
#                  the CSV (default: the runner's own default, i.e. mean).
#                  median needs a runner built with --profile_csv_statistic.
#   INPUT_FORMAT=hlo|stablehlo
#                  Which modules a directory walk picks up: the HLO text
#                  `*.txt`/`*.hlo` dumps (default), or the StableHLO portable
#                  artifacts `*.mlirbc` written by convert_hlo_to_stablehlo.py.
#                  `.mlirbc` modules are run with --input_format=stablehlo
#                  if the runner has it (openxla/xla@49468c7ea9); otherwise
#                  they are first lowered to HLO with XLA_TRANSLATE and
#                  STABLEHLO_TRANSLATE (default: next to the runner in the
#                  same bazel-bin; build them at the runner's revision).
#
# Runners without --append_profile_to_csv_file (or without
# --profile_csv_statistic while STAT is set) get their CSV row written by this
# script from the runner's "## Execution time" output (see write_csv_row).

set -uo pipefail

die() { echo "error: $*" >&2; exit 2; }

if [ "${1:-}" = "--branches" ]; then
  shift
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  campaign="$script_dir/run_xla_branch_eval.py"
  [ -f "$campaign" ] || die "branch evaluator not found: $campaign"
  exec python3 "$campaign" "$@"
fi

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

# STAT is the runner's --profile_csv_statistic ("mean" or "median"). Left empty
# by default so the flag is not passed at all: the runner then falls back to
# HLO_RUNNER_PROFILE_CSV_STATISTIC if it is exported, and to the mean otherwise.
# With the default 5 repeats the median is taken over 4 timed samples, so it is
# the mean of the middle two - raise the repeat count to make it meaningful.
STAT=${STAT:-}

# CMD_BUFFER=off (default) adds XLA_FLAGS=--xla_gpu_enable_command_buffer= to every
# run, disabling XLA's HIP command buffers (graphs). Required on this ROCm build:
# ~half the models otherwise SIGSEGV inside libamdhip64's graph launch
# (RocmCommandBuffer::LaunchGraph) at execution. Set CMD_BUFFER=on to re-enable.
CMD_BUFFER=${CMD_BUFFER:-off}

# CAPTURE_RESOLVED_XLA_FLAGS=true enables gpu_compiler VLOG(1) for only the
# first runner process, then restores normal logging. The default "false"
# preserves normal direct-run logging for every invocation.
CAPTURE_RESOLVED_XLA_FLAGS=${CAPTURE_RESOLVED_XLA_FLAGS:-false}
case "$CAPTURE_RESOLVED_XLA_FLAGS" in
  true|false) ;;
  *) die "CAPTURE_RESOLVED_XLA_FLAGS must be true or false" ;;
esac
RESOLVED_XLA_FLAGS_CAPTURED=0

INPUT_FORMAT=${INPUT_FORMAT:-hlo}
case "$INPUT_FORMAT" in
  hlo|stablehlo) ;;
  *) die "INPUT_FORMAT must be hlo or stablehlo" ;;
esac

# Older runners (e.g. rocm-jaxlib-v0.8.0) predate --append_profile_to_csv_file
# (openxla/xla#42696), --profile_csv_statistic and --input_format=stablehlo
# (openxla/xla@49468c7ea9). The runner has no version flag, so probe for the
# features themselves; neither probe initializes a device.
RUNNER_HELP=$("$RUNNER" --help 2>&1)
grep -q -- '--num_repeats=' <<<"$RUNNER_HELP" ||
  die "could not list the runner's flags; '$RUNNER --help' printed:
$(head -5 <<<"$RUNNER_HELP")"
runner_has_flag() { grep -q -- "--$1=" <<<"$RUNNER_HELP"; }
RUNNER_HAS_CSV=0;      runner_has_flag append_profile_to_csv_file && RUNNER_HAS_CSV=1
RUNNER_HAS_CSV_STAT=0; runner_has_flag profile_csv_statistic && RUNNER_HAS_CSV_STAT=1
# --input_format is validated before the "No HLO file specified" check, so a
# run without any module tells the two kinds of runner apart.
RUNNER_HAS_STABLEHLO=0
grep -q 'No HLO file specified' <<<"$("$RUNNER" --input_format=stablehlo 2>&1)" &&
  RUNNER_HAS_STABLEHLO=1

OWN_CSV=1
if [ "$RUNNER_HAS_CSV" -eq 1 ] && { [ -z "$STAT" ] || [ "$RUNNER_HAS_CSV_STAT" -eq 1 ]; }; then
  OWN_CSV=0
fi

# Without --input_format=stablehlo, .mlirbc modules are lowered to HLO first,
# with tools from the runner's revision (by default, from its bazel-bin; the
# runner's symlinks are resolved first). RUNNER_BIN_ROOT is empty for a runner
# outside a bazel-bin, which has no default tools.
RUNNER_REAL=$(readlink -f -- "$RUNNER")
RUNNER_BIN_ROOT=${RUNNER_REAL%/xla/tools/multihost_hlo_runner/hlo_runner_main}
[ "$RUNNER_BIN_ROOT" != "$RUNNER_REAL" ] || RUNNER_BIN_ROOT=""
XLA_TRANSLATE=${XLA_TRANSLATE:-${RUNNER_BIN_ROOT:+$RUNNER_BIN_ROOT/xla/hlo/translate/xla-translate}}
STABLEHLO_TRANSLATE=${STABLEHLO_TRANSLATE:-}
# The external repo is `stablehlo` under WORKSPACE, `+third_party_ext+stablehlo`
# under bzlmod.
STABLEHLO_TRANSLATE_FOUND=()
if [ -z "$STABLEHLO_TRANSLATE" ] && [ -n "$RUNNER_BIN_ROOT" ]; then
  for tool in "$RUNNER_BIN_ROOT"/external/{,*+}stablehlo/stablehlo-translate; do
    [ -x "$tool" ] && STABLEHLO_TRANSLATE_FOUND+=("$tool")
  done
  STABLEHLO_TRANSLATE=${STABLEHLO_TRANSLATE_FOUND[0]:-}
fi
if [ "$RUNNER_HAS_STABLEHLO" -eq 0 ] &&
   { [ "$INPUT_FORMAT" = stablehlo ] || [[ "$HLO" == *.mlirbc ]]; }; then
  build_hint="build //xla/hlo/translate:xla-translate and @stablehlo//:stablehlo-translate
at the runner's revision, or point XLA_TRANSLATE / STABLEHLO_TRANSLATE at them"
  [ -n "$RUNNER_BIN_ROOT" ] || { [ -n "$XLA_TRANSLATE" ] && [ -n "$STABLEHLO_TRANSLATE" ]; } ||
    die "runner lacks --input_format=stablehlo and is not in a bazel-bin
(.../xla/tools/multihost_hlo_runner/hlo_runner_main), so there are no default
lowering tools; set XLA_TRANSLATE and STABLEHLO_TRANSLATE to xla-translate and
stablehlo-translate built at the runner's revision"
  [ -x "$XLA_TRANSLATE" ] ||
    die "runner lacks --input_format=stablehlo and $XLA_TRANSLATE is missing;
$build_hint"
  [ -n "$STABLEHLO_TRANSLATE" ] ||
    die "runner lacks --input_format=stablehlo and no stablehlo-translate is in
$RUNNER_BIN_ROOT/external/stablehlo/ or $RUNNER_BIN_ROOT/external/*+stablehlo/;
$build_hint"
  [ -x "$STABLEHLO_TRANSLATE" ] ||
    die "runner lacks --input_format=stablehlo and $STABLEHLO_TRANSLATE is missing;
$build_hint"
  [ "${#STABLEHLO_TRANSLATE_FOUND[@]}" -le 1 ] ||
    echo "warning: ${#STABLEHLO_TRANSLATE_FOUND[@]} stablehlo-translate builds in $RUNNER_BIN_ROOT/external, using $STABLEHLO_TRANSLATE; set STABLEHLO_TRANSLATE to pick another" >&2
fi

STAGE_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/run_hlo_eval.XXXXXX")
trap 'rm -rf "$STAGE_ROOT"' EXIT

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

# Module files directly inside directory $1 for the selected INPUT_FORMAT.
leaf_modules() {
  local f
  if [ "$INPUT_FORMAT" = stablehlo ]; then
    set -- "$1"/*.mlirbc
  else
    set -- "$1"/*.txt "$1"/*.hlo
  fi
  for f in "$@"; do [ -e "$f" ] && printf '%s\n' "$f"; done
}

# num_partitions of a module file (default 1 if the header lacks the token).
# StableHLO bytecode cannot be grepped, so it comes from the .meta sidecar.
file_parts() {
  local n src=$1
  [[ "$src" == *.mlirbc ]] && src="$src.meta"
  n=$(grep -oE 'num_partitions=[0-9]+' "$src" | head -1 | cut -d= -f2)
  echo "${n:-1}"
}

# comma-separated device list 0..N-1 (just "0" for N<=1).
dev_list() { if [ "$1" -gt 1 ]; then seq -s, 0 $(($1 - 1)); else echo 0; fi; }

RAN=(); SKIPPED=(); FAILED=(); RESUMED=()

# Lower the StableHLO artifacts $2.. to HloProto text protos under directory $1
# (run with --input_format=proto_text) and print the lowered paths. They keep
# their basenames, so the CSV columns match a native --input_format=stablehlo
# run. Not --stablehlo-to-hlo-text: it skips the legalization that turns
# `custom_call @mhlo.topk` back into TopK (and that convert_hlo_to_stablehlo.py
# verifies). The intermediate keeps its locations, so the HLO keeps the
# op_name/source metadata. Modules that fail to lower are left out.
stage_stablehlo_as_hlo() {
  local dir=$1 f out; shift
  for f in "$@"; do
    out="$dir/$(basename -- "$f")"
    if "$STABLEHLO_TRANSLATE" --deserialize --mlir-print-debuginfo "$f" \
         -o "$out.shlo.mlir" >&2 &&
       "$XLA_TRANSLATE" --stablehlo-to-hlo "$out.shlo.mlir" -o "$out" >&2; then
      printf '%s\n' "$out"
    else
      echo "  WARN: could not lower $f to HLO, skipping it" >&2
    fi
    rm -f "$out.shlo.mlir"
  done
}

# Copy stdin to stdout with staging directory $1 replaced by the modules'
# original directory $2, so the log and the campaign report's reproduction
# commands name the .mlirbc artifacts rather than deleted temp files.
unstage_paths() {
  FROM="$1/" TO="$2/" awk '{
      out = ""; s = $0
      while ((i = index(s, ENVIRON["FROM"])) > 0) {
        out = out substr(s, 1, i - 1) ENVIRON["TO"]
        s = substr(s, i + length(ENVIRON["FROM"]))
      }
      print out s; fflush()
    }'
}

# Append a row to CSV file $1 in the format of the runner's
# --append_profile_to_csv_file, from the "## Execution time, file=F repeat=I
# duration=Nns" lines on stdin. $2 is the timestamp, $3 a file listing every
# module of the invocation. Like the runner, drop the warmup repeat unless it
# is the only one and reduce with $STAT (default mean). Unlike it, match cells
# to the header by name: a new header lists every module in $3, and modules
# without timings get empty cells. Returns 1 (writing nothing) if there are no
# timings, 3 if a timed module has no column in an existing header (awk itself
# exits with 2 on I/O errors).
write_csv_row() {
  local csv=$1 run_time=$2 modules=$3
  LC_ALL=C awk -v csv="$csv" -v run_time="$run_time" \
      -v stat="${STAT:-mean}" '
    FNR == NR {
      if (!($0 in listed)) { listed[$0] = 1; files[++nfiles] = $0 }
      next
    }
    /^## Execution time, file=.* repeat=[0-9]+ duration=[0-9]+ns$/ {
      ns = $NF; sub(/^duration=/, "", ns); sub(/ns$/, "", ns)
      file = substr($0, length("## Execution time, file=") + 1)
      file = substr(file, 1, length(file) - length($(NF-1)) - length($NF) - 2)
      if (!(file in listed)) { listed[file] = 1; files[++nfiles] = file }
      times[file, ++count[file]] = ns + 0
      ntimes++
    }
    END {
      if (ntimes == 0) exit 1
      for (i = 2; i <= nfiles; i++) {
        f = files[i]
        for (j = i - 1; j >= 1 && files[j] > f; j--) files[j + 1] = files[j]
        files[j + 1] = f
      }
      prefix = files[1]
      for (i = 2; i <= nfiles; i++)
        while (index(files[i], prefix) != 1)
          prefix = substr(prefix, 1, length(prefix) - 1)
      sub(/[^\/]*$/, "", prefix)
      for (i = 1; i <= nfiles; i++) {
        f = files[i]
        if (!(f in count)) continue
        first = count[f] > 1 ? 2 : 1; n = 0
        for (k = first; k <= count[f]; k++) {
          v = times[f, k]
          for (j = n; j >= 1 && sorted[j] > v; j--) sorted[j + 1] = sorted[j]
          sorted[j + 1] = v; n++
        }
        if (stat == "median") {
          ns = n % 2 ? sorted[(n + 1) / 2] : 0.5 * (sorted[n / 2] + sorted[n / 2 + 1])
        } else {
          ns = 0; for (k = 1; k <= n; k++) ns += sorted[k]; ns /= n
        }
        cell[substr(f, length(prefix) + 1)] = sprintf(" %.4gms", ns / 1e6)
      }
      # Skip the `#` provenance block before the header.
      while ((got = (getline line < csv)) > 0)
        if (line !~ /^[ \t]*(#|$)/) break
      if (got > 0) {
        ncols = split(line, cols, ",")
      } else {
        ncols = 1
        printf "Datetime" >> csv
        for (i = 1; i <= nfiles; i++) {
          cols[++ncols] = substr(files[i], length(prefix) + 1)
          printf ",%s", cols[ncols] >> csv
        }
        printf "\n" >> csv
      }
      close(csv)
      printf "%s", run_time >> csv
      for (c = 2; c <= ncols; c++) {
        printf ",%s", ((cols[c] in cell) ? cell[cols[c]] : "") >> csv
        delete cell[cols[c]]
      }
      printf "\n" >> csv
      rc = 0
      for (name in cell) {
        printf "  WARN: %s has no column in %s; its timing is not recorded\n",
            name, csv > "/dev/stderr"
        rc = 3
      }
      exit rc
    }' "$modules" -
}

# Whether the last row of CSV file $1 has as many cells as its header (the
# runner matches them by position). `#` and blank lines are skipped.
csv_row_matches_header() {
  [ -s "$1" ] || return 0
  awk -F, '/^[ \t]*(#|$)/ { next }
    !n { n = NF; next }
    { rows++; last = NF }
    END { exit !(rows == 0 || last == n) }' "$1"
}

# Whether the header of CSV file $1 only has columns of module kind $2
# (`mlirbc` or `hlo`), so the two kinds never share a CSV. `#` and blank lines
# are skipped.
csv_columns_match_kind() {
  [ -s "$1" ] || return 0
  awk -F, -v kind="$2" '/^[ \t]*(#|$)/ { next }
    {
      for (i = 2; i <= NF; i++)
        if ((kind == "mlirbc") != ($i ~ /\.mlirbc$/)) exit 1
      exit 0
    }' "$1"
}

# Invoke the runner on one or more module files that share the same partition
# count, writing into $csv (path without the .csv extension).
invoke() {
  local n=$1 csv=$2; shift 2
  local devs; devs=$(dev_list "$n")
  local -a args=(--device_type=gpu)
  local stage="" orig_dir="" unlowered=0 own_csv=$OWN_CSV
  local -a modules=("$@")
  local kind=hlo
  [[ "$1" == *.mlirbc ]] && kind=mlirbc
  if ! csv_columns_match_kind "${csv}.csv" "$kind"; then
    FAILED+=("$csv")
    echo "  FAIL: ${csv}.csv has columns of the other module kind (.txt/.hlo vs .mlirbc) -> $csv"
    return
  fi
  if [[ "$1" == *.mlirbc ]]; then
    if [ "$RUNNER_HAS_STABLEHLO" -eq 1 ]; then
      args+=(--input_format=stablehlo)
    else
      stage=$(mktemp -d "$STAGE_ROOT/hlo.XXXXXX")
      local -a lowered=()
      mapfile -t lowered < <(stage_stablehlo_as_hlo "$stage" "$@")
      if [ "${#lowered[@]}" -eq 0 ]; then
        FAILED+=("$csv"); echo "  FAIL: no module could be lowered to HLO -> $csv"
        rm -rf "$stage"; return
      fi
      # Profile the rest but fail the leaf below. Only this script knows the
      # modules left out, so it writes the CSV, keeping a column for each.
      unlowered=$(($# - ${#lowered[@]}))
      local f; modules=()
      for f in "$@"; do modules+=("$stage/$(basename -- "$f")"); done
      [ "$unlowered" -eq 0 ] || own_csv=1
      orig_dir=$(dirname -- "$1")
      set -- "${lowered[@]}"
      args+=(--input_format=proto_text)
    fi
  fi
  if [ "$n" -gt 1 ]; then
    args+=(--num_partitions="$n" --use_shardy_partitioner)
    [ "$n" -le "$NGPU" ] || echo "  WARN: needs $n GPUs but only ~$NGPU visible"
  fi
  args+=(--hlo_argument_mode="$ARG_MODE"
         --num_repeats="$REPEATS"
         --profile_execution=true)
  if [ "$own_csv" -eq 0 ]; then
    args+=(--append_profile_to_csv_file="$csv")
    [ -n "$STAT" ] && args+=(--profile_csv_statistic="$STAT")
  fi
  args+=("$@")
  echo "  run: N=$n, $# module(s), devices=[$devs] -> ${csv}.csv"
  local inherited_xf="${XLA_FLAGS:-}"
  local xf="$inherited_xf"
  [ "$CMD_BUFFER" = off ] && xf="--xla_gpu_enable_command_buffer= $xf"
  local capture_resolved_flags=0
  local vmodule="${TF_CPP_VMODULE:-}"
  if [ "$CAPTURE_RESOLVED_XLA_FLAGS" = true ] &&
     [ "$RESOLVED_XLA_FLAGS_CAPTURED" -eq 0 ]; then
    capture_resolved_flags=1
    RESOLVED_XLA_FLAGS_CAPTURED=1
    if ! [[ ",$vmodule," =~ ,gpu_compiler=[1-9][0-9]*, ]]; then
      vmodule="${vmodule:+$vmodule,}gpu_compiler=1"
    fi
  fi
  printf '  inherited XLA_FLAGS=%q\n' "$inherited_xf"
  printf '  effective invocation: env '
  local name
  for name in ROCR_VISIBLE_DEVICES TF_XLA_FLAGS; do
    [[ -v $name ]] && printf '%s=%q ' "$name" "${!name}"
  done
  [ "$capture_resolved_flags" -eq 0 ] ||
    printf 'TF_CPP_VMODULE=%q ' "$vmodule"
  printf 'HIP_VISIBLE_DEVICES=%q CUDA_VISIBLE_DEVICES=%q XLA_FLAGS=%q ' \
    "$devs" "$devs" "$xf"
  printf '%q ' "$RUNNER" "${args[@]}"
  printf '\n'
  local -a run_env=(HIP_VISIBLE_DEVICES="$devs" CUDA_VISIBLE_DEVICES="$devs"
                    XLA_FLAGS="$xf")
  [ "$capture_resolved_flags" -eq 0 ] ||
    run_env=(TF_CPP_VMODULE="$vmodule" "${run_env[@]}")
  local runner_rc=0 no_timings=0 csv_misaligned=0 csv_write_failed=0
  local -a show=(cat)
  [ -z "$stage" ] || show=(unstage_paths "$stage" "$orig_dir")
  if [ "$own_csv" -eq 1 ]; then
    # The runner stamps its row when it starts, before compiling.
    local run_time; run_time=$(date '+%Y-%m-%d %H:%M:%S')
    local log; log=$(mktemp "$STAGE_ROOT/runner.XXXXXX.log")
    env "${run_env[@]}" "$RUNNER" "${args[@]}" | tee "$log" | "${show[@]}"
    runner_rc=${PIPESTATUS[0]}
    local csv_rc=0
    write_csv_row "${csv}.csv" "$run_time" <(printf '%s\n' "${modules[@]}") \
      < "$log" || csv_rc=$?
    case $csv_rc in
      0) ;;
      1) no_timings=1 ;;
      3) csv_misaligned=1 ;;
      *) csv_write_failed=1 ;;
    esac
    rm -f "$log"
  elif [ -n "$stage" ]; then
    env "${run_env[@]}" "$RUNNER" "${args[@]}" | "${show[@]}"
    runner_rc=${PIPESTATUS[0]}
    csv_row_matches_header "${csv}.csv" || csv_misaligned=1
  else
    env "${run_env[@]}" "$RUNNER" "${args[@]}" || runner_rc=$?
    csv_row_matches_header "${csv}.csv" || csv_misaligned=1
  fi
  [ -z "$stage" ] || rm -rf "$stage"
  if [ "$runner_rc" -ne 0 ]; then
    FAILED+=("$csv"); echo "  FAIL: runner exited non-zero -> $csv"
  elif [ "$unlowered" -gt 0 ]; then
    FAILED+=("$csv"); echo "  FAIL: $unlowered module(s) could not be lowered to HLO -> $csv"
  elif [ "$no_timings" -eq 1 ]; then
    FAILED+=("$csv"); echo "  FAIL: no '## Execution time' lines in the runner's output -> $csv"
  elif [ "$csv_write_failed" -eq 1 ]; then
    FAILED+=("$csv"); echo "  FAIL: could not write ${csv}.csv -> $csv"
  elif [ "$csv_misaligned" -eq 1 ]; then
    FAILED+=("$csv"); echo "  FAIL: the new row does not line up with the header of ${csv}.csv -> $csv"
  else
    RAN+=("${csv}.csv")
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
      sz=$(leaf_modules "$d" | xargs -r -d '\n' stat -c %s | awk '{s+=$1} END{printf "%d", s+0}')
      printf '%015d\t%s\n' "$sz" "$d"
    done | sort | cut -f2-
  fi
}

# Profile every module in one leaf directory into a single CSV.
run_leaf() {
  local leaf=$1
  local -a files=()
  mapfile -t files < <(leaf_modules "$leaf")
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
echo "         csv: $([ "$OWN_CSV" -eq 0 ] && echo runner || echo 'written by this script')   stablehlo: $([ "$RUNNER_HAS_STABLEHLO" -eq 1 ] && echo native || echo 'lowered to HLO first')"
echo "hlo    : $HLO"
if [ -n "$SINGLE_CSV" ]; then echo "out    : ${SINGLE_CSV}.csv (single file)"; else echo "out    : $OUT/ (one CSV per leaf)"; fi
echo "repeats: $REPEATS   order: $ORDER   arg_mode: $ARG_MODE   cmd_buffer: $CMD_BUFFER   stat: ${STAT:-runner default}   visible GPUs (approx): $NGPU"
echo

if [ -f "$HLO" ]; then
  n=$(file_parts "$HLO")
  csv=${SINGLE_CSV:-"$OUT/$(csv_base "$(dirname "$HLO")")"}
  echo "file: $HLO"
  invoke "$n" "$csv" "$HLO"
else
  # Directory: a leaf (modules directly inside) or a tree to walk.
  mapfile -t direct < <(leaf_modules "$HLO")
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
