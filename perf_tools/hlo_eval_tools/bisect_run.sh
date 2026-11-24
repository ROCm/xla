#!/usr/bin/env bash
#
# bisect_run.sh - a `git bisect run` script that finds the XLA commit behind an
# HLO-level performance regression (or a runner crash) on ROCm.
#
# Per revision of the XLA checkout being bisected it builds hlo_runner_main,
# times the given module(s), and classifies the revision: slower than a
# threshold, crashing, or printing a given pattern. It runs bazel and the runner
# natively -- inside the dev container, not through docker.
#
# Modules are the pre-optimization dumps beside this script, or any other
# HLO / StableHLO file; .mlirbc is detected and run as StableHLO. See the
# "Bisecting to a commit" section of README.md.
#
# Usage:
#   cd <your XLA checkout>            # the tree being bisected, not this repo
#   git bisect start <bad> <good>
#   git bisect run /path/to/bisect_run.sh --module <hlo-or-mlirbc> \
#       --threshold <ms> --tsv bisect-log.tsv
#
# The XLA checkout defaults to the git repository containing the current working
# directory -- which is where `git bisect run` starts the script -- so the
# script itself can live anywhere. Use --repo-dir to point elsewhere.
#
# Use --tsv, not --log: `git bisect run` eats a bare --log itself and passes
# only its value through, which this script then rejects as a stray argument.
#
# Modes (--mode):
#   time    (default) BAD = slow: metric >= --threshold (ms). Requires
#           --threshold. The metric is the sum over modules of the per-module
#           median runtime, aggregated over --trials with --trial-agg.
#   status  BAD = the runner exited non-zero (crash, compile error, check
#           failure). No threshold needed.
#   grep    BAD = the runner output matches --pattern (extended regex).
#
# --invert flips which side of the verdict is the bisect "bad" one, for every
# mode. Use it to hunt a speedup or a fix rather than a regression, and start
# the bisect the other way round:
#   git bisect start <fast-or-fixed-new-rev> <slow-or-broken-old-rev>
#   git bisect run /path/to/bisect_run.sh --module m.mlirbc --threshold 45 --invert
#
# Exit codes follow the `git bisect run` convention:
#   0   good
#   1   bad
#   125 skip (untestable: the build failed, or the run did not produce a
#       complete, clean set of timings -- see --judge-on-run-error)
#   128 abort (bad usage / missing module / missing runner)
#   130/143 interrupted -- stops the bisect loop rather than judging the rev

set -u -o pipefail

# The XLA checkout to bisect. Empty means "the repository containing the cwd",
# where `git bisect run` starts us -- never this script's own directory, which
# lives in a different repository.
REPO_DIR=""

MODE="time"
THRESHOLD=""
PATTERN=""
MODULES=()
INPUT_FORMAT=""            # auto: stablehlo for .mlirbc/.mlir, else text
NUM_REPEATS=10
WARMUP=1                   # repeats dropped before taking the median
TRIALS=1                   # timed processes per revision
TRIAL_AGG="min"            # min | median over the per-trial totals

# Constants, not flags: timing a GPU replay is the whole point of the tool.
ARGUMENT_MODE="uninitialized"
DEVICE_TYPE="gpu"
BUILD_CONFIG="rocm_clang_hermetic"

TARGET="//xla/tools/multihost_hlo_runner:hlo_runner_main"
RUNNER=""                  # defaults to bazel-bin/.../hlo_runner_main
SKIP_BUILD=0
AMDGPU_TARGETS=""          # auto-detected from rocminfo unless --targets is given
DISK_CACHE="${BAZEL_DISK_CACHE:-}"   # strongly recommended: a bisect rebuilds
                           # the same revisions repeatedly, and the disk cache is
                           # what keeps each step to minutes
REMOTE_CACHE=""            # a shared bazel cache, if your host can reach it --
                           # verify first, since an unreachable cache fails every
                           # build and then every revision looks "bad"
ROCM_DISTRO_URL="${ROCM_DISTRO_URL:-}"
ROCM_PATH_FLAG=""          # --rocm-path: build against a local ROCm install
# Niche build knobs, settable from the environment rather than as flags.
ROCM_DISTRO_HASH="${ROCM_DISTRO_HASH:-no_hash}"
SYSROOT_DIST="${SYSROOT_DIST:-}"
EXTRA_BAZEL_FLAGS=""
DRY_RUN=0

# Command buffers off, so the timings reflect the kernels rather than the
# capture/replay path. --xla-flags is appended after this.
BASE_XLA_FLAGS="--xla_gpu_enable_command_buffer="
EXTRA_XLA_FLAGS=""
AUTOTUNE_CACHE=""
REQUIRE_COMPLETE_AUTOTUNE=0
AUTOTUNE_LEVEL=""
JUDGE_ON_RUN_ERROR=0       # time mode: judge a failed-but-complete run anyway

BUILD_TIMEOUT=5400
RUN_TIMEOUT=1800
LOG_TSV=""
ON_BUILD_FAIL="skip"       # skip | bad
KEEP_LOGS_DIR=""
INVERT=0

usage() {
  sed -n '/^# Usage:/,/^#   128/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  cat <<'EOF'

Options:
  --module <path>        HLO / StableHLO module; repeat for several modules.
  --mode time|status|grep
  --threshold <ms>       time mode: BAD when the metric is >= this.
  --pattern <regex>      grep mode: BAD when the run output matches.
  --invert               flip which side of the verdict is bisect's 'bad'.

  --num-repeats <n>      runner --num_repeats (default 10).
  --warmup <n>           repeats dropped before the median (default 1).
  --trials <n>           timed processes per revision (default 1).
  --trial-agg min|median aggregate the per-trial totals (default min).
  --input-format <f>     override the format auto-detected from the module
                         extension. One format applies to every module, so a
                         mixed set is rejected unless you pass this explicitly.

  --xla-flags <flags>    appended to XLA_FLAGS, which always starts with
                         --xla_gpu_enable_command_buffer= .
  --autotune-cache <f>   --xla_gpu_load_autotune_results_from=<f>; strongly
                         recommended for perf bisects, otherwise autotune
                         noise and autotune-affecting commits drive the result.
  --require-complete-autotune
                         also pass --xla_gpu_require_complete_aot_autotune_results=true.
                         Note: a revision whose fusions miss from the cache
                         then fails the run and is SKIPped, so prefer leaving
                         this off across a wide revision range.
  --autotune-level <n>   --xla_gpu_autotune_level=<n>.
  --judge-on-run-error   judge a revision even though the runner exited
                         non-zero: in time mode when every module still timed,
                         in grep mode when the pattern simply did not match.
                         Default is to skip both. Use when such a failure is
                         systematic across the range and skipping would abandon
                         every revision. (status mode is unaffected: there the
                         exit code *is* the verdict.)

  --target <label>       bazel target (default multihost_hlo_runner).
  --runner <path>        runner binary (default derived from --target).
  --skip-build           do not build; use the runner as it is on disk.
  --targets <gfx,...>    TF_ROCM_AMDGPU_TARGETS (default: this host's gfx).
  --repo-dir <path>      the XLA checkout to bisect (default: the repository
                         containing the current working directory).
  --disk-cache <path>    bazel --disk_cache, also read from $BAZEL_DISK_CACHE
                         ("" disables). Use one: it keeps each bisect step to
                         minutes instead of a full rebuild.
  --remote-cache <url>   bazel --remote_cache ("" disables, the default).
                         Check it is reachable before using it.
  --rocm-distro-url <u>  hermetic ROCm tarball (ROCM_PATH is then cleared);
                         also read from $ROCM_DISTRO_URL.
  --rocm-path <path>     build against a local ROCm install instead.
  --bazel-flags <flags>  appended verbatim to the bazel build command; the
                         build always passes --config=rocm_clang_hermetic.
  --dry-run              print the build and run commands and exit 0.

  $ROCM_DISTRO_HASH (default no_hash) and $SYSROOT_DIST are read from the
  environment for the rarer hermetic-build cases.

  Pick the ROCm source that matches how this tree's bazel-bin was last built.
  With neither --rocm-distro-url nor --rocm-path the script passes no ROCm
  repo_env at all and behaves like a bare `bazel build` in your shell. A
  mismatch re-fetches local_config_rocm and rebuilds everything ROCm-dependent
  at the first revision, so do one manual build first and confirm it is a
  cache hit before starting a bisect.

  --tsv <file>           append one row per revision. Not --log: git bisect
                         swallows that one before it reaches this script.
  --keep-logs <dir>      keep per-revision build/run logs there.
  --on-build-fail skip|bad   what a build failure means (default skip).
  --build-timeout <s>    default 5400.   --run-timeout <s>  default 1800.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --module)         MODULES+=("$2"); shift 2 ;;
    --mode)           MODE="$2"; shift 2 ;;
    --threshold)      THRESHOLD="$2"; shift 2 ;;
    --pattern)        PATTERN="$2"; shift 2 ;;
    --invert)         INVERT=1; shift ;;

    --num-repeats)    NUM_REPEATS="$2"; shift 2 ;;
    --warmup)         WARMUP="$2"; shift 2 ;;
    --trials)         TRIALS="$2"; shift 2 ;;
    --trial-agg)      TRIAL_AGG="$2"; shift 2 ;;
    --input-format)   INPUT_FORMAT="$2"; shift 2 ;;

    --xla-flags)      EXTRA_XLA_FLAGS="$2"; shift 2 ;;
    --autotune-cache) AUTOTUNE_CACHE="$2"; shift 2 ;;
    --require-complete-autotune) REQUIRE_COMPLETE_AUTOTUNE=1; shift ;;
    --autotune-level) AUTOTUNE_LEVEL="$2"; shift 2 ;;
    --judge-on-run-error) JUDGE_ON_RUN_ERROR=1; shift ;;

    --target)         TARGET="$2"; shift 2 ;;
    --runner)         RUNNER="$2"; shift 2 ;;
    --skip-build)     SKIP_BUILD=1; shift ;;
    --targets)        AMDGPU_TARGETS="$2"; shift 2 ;;
    --disk-cache)     DISK_CACHE="$2"; shift 2 ;;
    --remote-cache)   REMOTE_CACHE="$2"; shift 2 ;;
    --rocm-distro-url) ROCM_DISTRO_URL="$2"; shift 2 ;;
    --rocm-path)      ROCM_PATH_FLAG="$2"; shift 2 ;;
    --bazel-flags)    EXTRA_BAZEL_FLAGS="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;

    --repo-dir)       REPO_DIR="$2"; shift 2 ;;
    --tsv)            LOG_TSV="$2"; shift 2 ;;
    --keep-logs)      KEEP_LOGS_DIR="$2"; shift 2 ;;
    --on-build-fail)  ON_BUILD_FAIL="$2"; shift 2 ;;
    --build-timeout)  BUILD_TIMEOUT="$2"; shift 2 ;;
    --run-timeout)    RUN_TIMEOUT="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *)  # A stray non-flag argument nearly always means --log was used for
        # --tsv: git bisect eats the flag and passes only its value on.
      echo "bisect_run: unknown argument '$1'" >&2
      [[ "$1" == -* ]] || echo "bisect_run: (--log is swallowed by git bisect; use --tsv)" >&2
      usage >&2; exit 128 ;;
  esac
done

# ------------------------------------------------------------ validation ----
die() { echo "bisect_run: $*" >&2; exit 128; }

fmt_of() {  # fmt_of <module path> -> the input format implied by its extension
  case "$1" in
    *.mlirbc|*.mlir) echo stablehlo ;;
    *.pb)            echo proto_binary ;;
    *.pbtxt)         echo proto_text ;;
    *)               echo text ;;
  esac
}

[[ ${#MODULES[@]} -gt 0 ]] || die "--module <path> is required (repeatable)"
for m in "${MODULES[@]}"; do
  [[ -f "$m" ]] || die "module not found: $m"
done
case "$MODE" in
  time)   [[ -n "$THRESHOLD" ]] || die "--threshold <ms> is required in time mode"
          [[ "$THRESHOLD" =~ ^[0-9]+(\.[0-9]+)?$ ]] ||
            die "--threshold must be a number in ms (got '$THRESHOLD')" ;;
  status) : ;;
  grep)   [[ -n "$PATTERN" ]] || die "--pattern <regex> is required in grep mode" ;;
  *) die "--mode must be time, status or grep (got '$MODE')" ;;
esac
# A zero/garbage trial count would leave no run logs at all, which later reads
# as an unbound array (or a grep with no files, which blocks on stdin).
[[ "$TRIALS" =~ ^[1-9][0-9]*$ ]] || die "--trials must be a positive integer (got '$TRIALS')"
[[ "$NUM_REPEATS" =~ ^[1-9][0-9]*$ ]] || die "--num-repeats must be a positive integer"
[[ "$WARMUP" =~ ^[0-9]+$ ]] || die "--warmup must be a non-negative integer"
[[ "$WARMUP" -lt "$NUM_REPEATS" ]] || die "--warmup ($WARMUP) must be below --num-repeats ($NUM_REPEATS)"
[[ "$TRIAL_AGG" == min || "$TRIAL_AGG" == median ]] || die "--trial-agg must be min or median"
[[ "$ON_BUILD_FAIL" == skip || "$ON_BUILD_FAIL" == bad ]] || die "--on-build-fail must be skip or bad"
[[ -z "$AUTOTUNE_CACHE" || -f "$AUTOTUNE_CACHE" ]] || die "autotune cache not found: $AUTOTUNE_CACHE"
command -v python3 >/dev/null || die "python3 is required to parse timings"

# Modules and caches are passed to a future checkout of this repo; resolve them
# now so a relative path keeps pointing at the same file.
for i in "${!MODULES[@]}"; do MODULES[$i]="$(realpath "${MODULES[$i]}")"; done
[[ -n "$AUTOTUNE_CACHE" ]] && AUTOTUNE_CACHE="$(realpath "$AUTOTUNE_CACHE")"

# One --input_format covers every module, so a mixed set would parse the first
# and fail on the rest: a partial run, not a clean error. Refuse it up front.
if [[ -z "$INPUT_FORMAT" ]]; then
  INPUT_FORMAT="$(fmt_of "${MODULES[0]}")"
  for m in "${MODULES[@]}"; do
    [[ "$(fmt_of "$m")" == "$INPUT_FORMAT" ]] || die \
      "modules imply different input formats ($INPUT_FORMAT from ${MODULES[0]##*/}," \
      "$(fmt_of "$m") from ${m##*/}); a single --input_format applies to all of" \
      "them, so pass --input-format explicitly or bisect them separately"
  done
fi

if [[ -z "$AMDGPU_TARGETS" ]]; then
  AMDGPU_TARGETS="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)"
  # Guessing an architecture here would build the whole bisect for the wrong
  # GPU, so say so instead.
  [[ -n "$AMDGPU_TARGETS" ]] ||
    die "could not detect the GPU architecture from rocminfo; pass --targets <gfx...>"
fi

# Resolve the XLA checkout being bisected. `git bisect run` starts us at its
# root; from anywhere else, --repo-dir says which tree to use.
if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null)" ||
    die "not inside a git repository -- run from the XLA checkout, or pass --repo-dir"
  [[ -n "$REPO_DIR" ]] ||
    die "not inside a git repository -- run from the XLA checkout, or pass --repo-dir"
fi
REPO_DIR="$(realpath "$REPO_DIR")"
# Guard against the easy mistake of running this from the tools repo instead of
# from the XLA tree it is meant to bisect.
[[ -d "$REPO_DIR/xla" ]] ||
  die "$REPO_DIR does not look like an XLA checkout (no xla/ directory); pass --repo-dir"

if [[ -z "$RUNNER" ]]; then
  # //a/b:c -> bazel-bin/a/b/c
  _t="${TARGET#//}"; RUNNER="$REPO_DIR/bazel-bin/${_t%:*}/${_t##*:}"
fi

REV="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
SHORT="${REV:0:12}"
say() { echo "[bisect $SHORT] $*" >&2; }

# Where to stash build/run logs for this revision.
if [[ -n "$KEEP_LOGS_DIR" ]]; then
  mkdir -p "$KEEP_LOGS_DIR"
  KEEP_LOGS_DIR="$(realpath "$KEEP_LOGS_DIR")"
  LOG_PREFIX="$KEEP_LOGS_DIR/$SHORT"
  CLEANUP=""
else
  TMPD="$(mktemp -d)"
  LOG_PREFIX="$TMPD/$SHORT"
  CLEANUP="$TMPD"
fi
BUILD_LOG="$LOG_PREFIX.build.log"
trap '[[ -n "$CLEANUP" ]] && rm -rf "$CLEANUP"' EXIT

[[ -n "$LOG_TSV" ]] && LOG_TSV="$(realpath -m "$LOG_TSV")"

# Paths are all resolved against the caller's cwd by now, so move to the repo
# root for a deterministic bazel cwd.
cd "$REPO_DIR" || die "cannot cd to $REPO_DIR"

# ------------------------------------------------------ verdict reporting ----
build_s=0        # filled in by the build step
n_trials=1       # filled in once the mode is known

tsv_row() {  # tsv_row <metric> <samples> <verdict> <detail>
  [[ -n "$LOG_TSV" ]] || return 0
  [[ -s "$LOG_TSV" ]] || printf 'rev\tmetric_ms\tsamples\ttrials\tbuild_s\tverdict\tdetail\n' >"$LOG_TSV"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$REV" "$1" "$2" "$n_trials" "$build_s" "$3" "$4" >>"$LOG_TSV"
}

# Physical verdict: BAD always means "the symptom is present" (slow / failing /
# pattern matched). That is what lands in the TSV, so a log means the same
# thing regardless of the bisect direction. --invert only moves the exit code.
finish() {  # finish <verdict BAD|GOOD> <message>
  local verdict="$1" msg="$2" rc term
  if [[ "$verdict" == "BAD" ]]; then rc=$(( INVERT ? 0 : 1 )); else rc=$(( INVERT ? 1 : 0 )); fi
  [[ $rc -eq 0 ]] && term="good" || term="bad"
  say "$msg -> $verdict -> bisect '$term'"
  exit $rc
}

# A revision that cannot answer the question -- as opposed to one that answers
# it badly -- is untestable, and must leave the bisect undecided.
skip_revision() {  # skip_revision <reason> <detail-for-tsv> [samples]
  say "verdict: SKIP ($1)"
  tsv_row NA "${3:-0}" SKIP "$2"
  exit 125
}

# ----------------------------------------------------- signal / children ----
CHILD_PID=""
IN_BUILD=0                   # 1 only while bazel is running

killtree() {  # killtree <pid> <signal>
  local pid="$1" sig="$2" child
  for child in $(ps -eo pid=,ppid= 2>/dev/null | awk -v p="$pid" '$2==p{print $1}'); do
    killtree "$child" "$sig"
  done
  kill -"$sig" "$pid" 2>/dev/null
}

on_signal() {
  local sig="$1"
  trap - INT TERM HUP          # a second Ctrl-C kills us outright
  say "caught SIG$sig -- stopping the build/run..."
  if [[ -n "$CHILD_PID" ]]; then
    killtree "$CHILD_PID" TERM
    for _ in 1 2 3 4 5; do kill -0 "$CHILD_PID" 2>/dev/null || break; sleep 1; done
    killtree "$CHILD_PID" KILL
  fi
  # A build killed mid-flight leaves the bazel server holding the output base
  # lock. Only on an interrupted build: a Ctrl-C during a timing run must not
  # shut down a server something else in this tree is using.
  if [[ $IN_BUILD -eq 1 ]]; then
    (cd "$REPO_DIR" && timeout 30 bazel shutdown >/dev/null 2>&1) || true
  fi
  say "aborted"
  # >=128 tells `git bisect run` to stop the loop rather than judge this rev.
  case "$sig" in
    INT)  exit 130 ;;
    TERM) exit 143 ;;
    *)    exit 129 ;;
  esac
}
trap 'on_signal INT'  INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP'  HUP

run_logged() {  # run_logged <timeout-seconds> <logfile> <cmd...>
  local t="$1" log="$2" rc=0; shift 2
  # Backgrounded + `wait` so that traps fire immediately instead of being
  # deferred until the foreground command returns.
  timeout --signal=TERM --kill-after=60 "$t" "$@" >"$log" 2>&1 &
  CHILD_PID=$!
  wait "$CHILD_PID" || rc=$?
  CHILD_PID=""
  return $rc
}

# ---------------------------------------------------------------- build ----
build_s=0
if [[ $SKIP_BUILD -eq 1 ]]; then
  say "skipping build (--skip-build)"
else
  BAZEL_ARGS=(build "--config=$BUILD_CONFIG"
              "--repo_env=TF_ROCM_AMDGPU_TARGETS=$AMDGPU_TARGETS"
              --dynamic_mode=off "--test_env=HOME=$HOME")
  [[ -n "$DISK_CACHE"   ]] && BAZEL_ARGS+=("--disk_cache=$DISK_CACHE")
  [[ -n "$REMOTE_CACHE" ]] && BAZEL_ARGS+=("--remote_cache=$REMOTE_CACHE")
  [[ -n "$SYSROOT_DIST" ]] && BAZEL_ARGS+=("--repo_env=SYSROOT_DIST=$SYSROOT_DIST")
  # Must match how this tree's bazel-bin was last built, or the first
  # revision re-fetches local_config_rocm and rebuilds everything ROCm-
  # dependent. Default passes no ROCm repo_env: a bare `bazel build`.
  if [[ -n "$ROCM_DISTRO_URL" ]]; then
    rocm_src="hermetic distro $ROCM_DISTRO_URL"
    BAZEL_ARGS+=("--repo_env=ROCM_PATH="
                 "--repo_env=ROCM_DISTRO_URL=$ROCM_DISTRO_URL"
                 "--repo_env=ROCM_DISTRO_HASH=$ROCM_DISTRO_HASH")
  elif [[ -n "$ROCM_PATH_FLAG" ]]; then
    rocm_src="local ROCm $ROCM_PATH_FLAG"
    BAZEL_ARGS+=("--repo_env=ROCM_PATH=$ROCM_PATH_FLAG")
  else
    rocm_src="inherited from the environment (ROCM_PATH=${ROCM_PATH:-unset})"
  fi
  # Split user flags on whitespace, but without filename globbing.
  if [[ -n "$EXTRA_BAZEL_FLAGS" ]]; then
    read -ra _extra_bazel <<<"$EXTRA_BAZEL_FLAGS"
    BAZEL_ARGS+=("${_extra_bazel[@]}")
  fi
  BAZEL_ARGS+=("$TARGET")

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "# ROCm source: $rocm_src"
    echo "# build:"; printf '  bazel'; printf ' %q' "${BAZEL_ARGS[@]}"; echo
  fi
  build_start=$(date +%s)
  build_rc=0
  if [[ $DRY_RUN -eq 0 ]]; then
    say "building $TARGET (ROCm: $rocm_src) ..."
    IN_BUILD=1
    run_logged "$BUILD_TIMEOUT" "$BUILD_LOG" bazel "${BAZEL_ARGS[@]}"
    build_rc=$?
    IN_BUILD=0
  fi
  build_s=$(( $(date +%s) - build_start ))

  if [[ $build_rc -ne 0 ]]; then
    say "build FAILED (rc=$build_rc, ${build_s}s). Tail:"
    tail -n 25 "$BUILD_LOG" >&2
    if [[ "$ON_BUILD_FAIL" == "bad" ]]; then
      # Expressed in bisect terms, not runtime terms -- unaffected by --invert.
      tsv_row NA 0 BAD "build-failed rc=$build_rc"
      say "verdict: bisect 'bad' (build failure)"
      exit 1
    fi
    skip_revision "build failure" "build-failed rc=$build_rc"
  fi
  [[ $DRY_RUN -eq 1 ]] || say "build ok (${build_s}s)"
fi

[[ -x "$RUNNER" || $DRY_RUN -eq 1 ]] || { say "runner not executable: $RUNNER"; exit 128; }

# ------------------------------------------------------------------ run ----
# The runner's RUNPATH to rocm_dist is cwd relative, so point at its runfiles
# copy. Glob the repo dir: bzlmod has spelled it `+rocm_configure_ext+...`,
# `~rocm_configure_ext~...` and plain `local_config_rocm`, and a range can
# straddle that. A non-hermetic build has no such dir, which is fine.
ROCM_LIB=""
for _cand in "$RUNNER.runfiles"/*local_config_rocm/rocm/rocm_dist/lib; do
  [[ -d "$_cand" ]] && { ROCM_LIB="$(realpath "$_cand")"; break; }
done
if [[ -n "$ROCM_LIB" ]]; then
  export LD_LIBRARY_PATH="$ROCM_LIB:$ROCM_LIB/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCPROFILER_REGISTER_ENABLED="${ROCPROFILER_REGISTER_ENABLED:-0}"

XLA_FLAGS_VALUE="$BASE_XLA_FLAGS"
[[ -n "$AUTOTUNE_CACHE" ]] && XLA_FLAGS_VALUE+=" --xla_gpu_load_autotune_results_from=$AUTOTUNE_CACHE"
[[ $REQUIRE_COMPLETE_AUTOTUNE -eq 1 ]] && XLA_FLAGS_VALUE+=" --xla_gpu_require_complete_aot_autotune_results=true"
[[ -n "$AUTOTUNE_LEVEL" ]] && XLA_FLAGS_VALUE+=" --xla_gpu_autotune_level=$AUTOTUNE_LEVEL"
[[ -n "$EXTRA_XLA_FLAGS" ]] && XLA_FLAGS_VALUE+=" $EXTRA_XLA_FLAGS"

RUN_ARGS=("--device_type=$DEVICE_TYPE" "--input_format=$INPUT_FORMAT"
          "--hlo_argument_mode=$ARGUMENT_MODE")
if [[ "$MODE" == "time" ]]; then
  RUN_ARGS+=("--num_repeats=$NUM_REPEATS" --profile_execution=true)
  n_trials="$TRIALS"
else
  n_trials=1
fi
RUN_ARGS+=("${MODULES[@]}")

if [[ $DRY_RUN -eq 1 ]]; then
  echo "# run (x$n_trials), mode=$MODE:"
  printf '  XLA_FLAGS=%q' "$XLA_FLAGS_VALUE"
  printf ' %q' "$RUNNER" "${RUN_ARGS[@]}"; echo
  echo "# LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  case "$MODE" in
    time) echo "# verdict: BAD when the ${TRIAL_AGG} trial total >= ${THRESHOLD}ms" ;;
    status) echo "# verdict: BAD when the run exits non-zero" ;;
    grep) echo "# verdict: BAD when the output matches /$PATTERN/" ;;
  esac
  echo "# bisect exit code: BAD -> $([[ $INVERT -eq 1 ]] && echo "0 (good, --invert)" || echo "1 (bad)")"
  exit 0
fi

# Runners before openxla/xla@49468c7ea9 (2026-07-27) have no
# --input_format=stablehlo. Skip those revisions rather than let a rejected flag
# pass for the crash being hunted in status/grep mode. --input_format is checked
# before "No HLO file specified", so a module-less run tells the generations
# apart in ~0.2s without touching a device (as in run_hlo_eval.sh).
if [[ "$INPUT_FORMAT" == "stablehlo" ]]; then
  probe_out="$("$RUNNER" --input_format=stablehlo 2>&1)"
  if ! grep -q 'No HLO file specified' <<<"$probe_out"; then
    say "this revision's runner does not accept --input_format=stablehlo" \
        "(it predates openxla/xla@49468c7ea9), so these modules cannot be" \
        "replayed here. Pass pre-lowered HLO to reach this far back. Probe said:"
    head -3 <<<"$probe_out" >&2
    skip_revision "runner has no stablehlo input support" "no-stablehlo-input-format"
  fi
fi

# The runner keys its timing lines by file path, so the same module passed twice
# collapses into one entry. Count distinct paths, or a repeated module would
# look like a half-finished trial.
N_EXPECTED_MODULES="$(printf '%s\n' "${MODULES[@]}" | sort -u | wc -l)"

RUN_LOGS=()
worst_rc=0
for i in $(seq 1 "$n_trials"); do
  log="$LOG_PREFIX.run$i.log"
  RUN_LOGS+=("$log")
  say "run $i/$n_trials ..."
  XLA_FLAGS="$XLA_FLAGS_VALUE" run_logged "$RUN_TIMEOUT" "$log" "$RUNNER" "${RUN_ARGS[@]}"
  rc=$?
  [[ $rc -ne 0 ]] && worst_rc=$rc
done

if [[ "$MODE" == "status" ]]; then
  if [[ $worst_rc -ne 0 ]]; then
    tail -n 15 "${RUN_LOGS[0]}" >&2
    tsv_row NA 0 BAD "rc=$worst_rc"
    finish BAD "run exited rc=$worst_rc"
  fi
  tsv_row NA 0 GOOD "rc=0"
  finish GOOD "run exited rc=0"
fi

if [[ "$MODE" == "grep" ]]; then
  if grep -qE -- "$PATTERN" "${RUN_LOGS[@]}"; then
    tsv_row NA 0 BAD "matched"
    finish BAD "output matched /$PATTERN/ (rc=$worst_rc)"
  fi
  # No match, but a failed run never gave a clean look at what we grep for:
  # "absent" and "never reached" are different answers.
  if [[ $worst_rc -ne 0 && $JUDGE_ON_RUN_ERROR -eq 0 ]]; then
    tail -n 15 "${RUN_LOGS[-1]}" >&2
    skip_revision \
      "no match, but the run exited rc=$worst_rc; pass --judge-on-run-error to call that good" \
      "no-match rc=$worst_rc"
  fi
  tsv_row NA 0 GOOD "no-match"
  finish GOOD "output did not match /$PATTERN/ (rc=$worst_rc)"
fi

# ------------------------------------------------------------ time mode ----
# hlo_runner_main prints, per module per repeat:
#   ## Execution time, file=<f> repeat=<i> duration=<ns>ns
# Per-module median over the post-warmup repeats, summed into one total per
# trial. Trials that timed fewer modules than expected are counted as
# incomplete, never summed: a partial total reads as a confident, wrong "good".
PARSE_PY='
import os, re, statistics as st, sys
warm, agg, expect = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
pat = re.compile(r"## Execution time, file=(\S+) repeat=(\d+) duration=(\d+)ns")
agg_fn = min if agg == "min" else st.median
totals, n_samples, detail, incomplete = [], 0, {}, 0
for log in sys.argv[4:]:
    per = {}
    try:
        text = open(log, errors="replace").read()
    except OSError:
        pass
    else:
        for f, r, d in pat.findall(text):
            if int(r) >= warm:
                per.setdefault(f, []).append(int(d) / 1e6)
    if len(per) < expect:
        incomplete += 1
    if not per:
        continue
    n_samples += sum(len(v) for v in per.values())
    meds = {f: st.median(v) for f, v in per.items()}
    totals.append(sum(meds.values()))
    for f, m in meds.items():
        detail.setdefault(f, []).append(m)
if not totals:
    print("0\t0\t0\t%d\tno-timings" % incomplete)
    sys.exit(0)
metric = agg_fn(totals)
per_mod = ",".join("%s=%.3f" % (os.path.basename(f), agg_fn(v))
                   for f, v in detail.items())
trials = "/".join("%.3f" % t for t in totals)
print("%d\t%.3f\t%d\t%d\t%s;trials=%s"
      % (n_samples, metric, len(totals), incomplete, per_mod, trials))
'
IFS=$'\t' read -r n_samples metric_ms n_ok n_incomplete detail < <(
  python3 -I -c "$PARSE_PY" "$WARMUP" "$TRIAL_AGG" "$N_EXPECTED_MODULES" "${RUN_LOGS[@]}")

if [[ "${n_samples:-0}" -eq 0 ]]; then
  say "run produced no timings (rc=$worst_rc). Tail:"
  tail -n 25 "${RUN_LOGS[0]}" >&2
  skip_revision "untestable" "run-failed rc=$worst_rc"
fi

# Partial data is worse than no data: it still produces a number, just a wrong
# one. Refuse to judge the revision on it.
if [[ "${n_incomplete:-0}" -gt 0 ]]; then
  say "$n_incomplete of $n_trials trial(s) timed fewer than $N_EXPECTED_MODULES module(s)" \
      "(rc=$worst_rc) -- the metric would be summed over survivors only. Tail:"
  tail -n 25 "${RUN_LOGS[-1]}" >&2
  skip_revision "incomplete run" \
    "incomplete-run rc=$worst_rc trials_incomplete=$n_incomplete" "$n_samples"
fi

# Every module timed, yet the runner failed anyway (late crash, teardown bug).
# "Probably fine" is not enough to pin a regression on a commit, so skip --
# unless such failures are systematic across the range, hence
# --judge-on-run-error.
if [[ $worst_rc -ne 0 ]]; then
  if [[ $JUDGE_ON_RUN_ERROR -eq 0 ]]; then
    say "all $N_EXPECTED_MODULES module(s) timed, but the runner exited rc=$worst_rc. Tail:"
    tail -n 15 "${RUN_LOGS[-1]}" >&2
    skip_revision "run error; pass --judge-on-run-error to judge anyway" \
      "run-error rc=$worst_rc" "$n_samples"
  fi
  say "WARNING: all modules timed but the runner exited rc=$worst_rc (judging anyway)"
fi

verdict=$(awk -v m="$metric_ms" -v t="$THRESHOLD" 'BEGIN{print (m >= t) ? "BAD" : "GOOD"}')
tsv_row "$metric_ms" "$n_samples" "$verdict" "$detail"
finish "$verdict" \
  "$TRIAL_AGG of $n_ok trial total(s) = ${metric_ms}ms over $n_samples samples (threshold=${THRESHOLD}ms, BAD=slow) [$detail]"
