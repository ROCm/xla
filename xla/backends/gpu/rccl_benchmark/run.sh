#!/usr/bin/env bash
#
# Driver for the XLA-driven RCCL benchmark suite.
#
# Builds the cases inside a container and runs them one arm per process against
# a chosen librccl. The library is the variable under test, so it is injected
# rather than assumed: every result records which library was actually loaded.
#
# Why one process per arm: RCCL caches each NCCL_/RCCL_ parameter the first time
# it reads one, so two arms that differ only in the environment cannot share a
# process. Separate processes also keep a GPU memory fault in one arm from
# taking the rest of the run with it, which matters because a fault is one of
# the outcomes these cases are looking for.
#
# The bazel ci_multi_gpu config is deliberately not used anywhere here. It pins
# NCCL_MAX_NCHANNELS=1, exposes four GPUs and retries failures three times; any
# one of those would hide a channel-assignment defect.
#
# Usage:
#   run.sh build
#   run.sh list
#   run.sh case <arm> [repeats]
#   run.sh matrix [repeats]

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKSPACE="$(cd "$ROOT/../../../.." && pwd)"

IMAGE="${IMAGE:-rocm/jax-training:maxtext-v26.5}"
HOST_MOUNT="${HOST_MOUNT:-$HOME}"
CONTAINER_MOUNT="${CONTAINER_MOUNT:-/xuefjian}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/xla_build}"
LOG_DIR="${LOG_DIR:-$ROOT/results}"
# Same directory as seen from inside the container, so the library can write its
# log somewhere that survives the container exiting.
LOG_DIR_IN_CONTAINER="${LOG_DIR_IN_CONTAINER:-$CONTAINER_MOUNT/repos/xla/xla/backends/gpu/rccl_benchmark/results}"
JOBS="${JOBS:-64}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"

# xla_test expands into one binary per backend; on ROCm that is the
# "_amdgpu_any" variant. The bare name is a test_suite, not something runnable.
BACKEND_SUFFIX="${BACKEND_SUFFIX:-amdgpu_any}"
readonly CASE_NAME="grouped_all_gather_test_${BACKEND_SUFFIX}"
readonly TARGET="//xla/backends/gpu/rccl_benchmark/warp_speed:${CASE_NAME}"
readonly BINARY_DIR_IN_CONTAINER="$CONTAINER_MOUNT/repos/xla/bazel-bin/xla/backends/gpu/rccl_benchmark/warp_speed"
readonly BINARY_IN_CONTAINER="$BINARY_DIR_IN_CONTAINER/$CASE_NAME"
# XLA's own libraries are linked statically (--dynamic_mode=off), but the ROCm
# libraries stay shared and part of the binary's RUNPATH is relative to the
# working directory, so it still has to start from its runfiles workspace root.
readonly RUNFILES_IN_CONTAINER="$BINARY_IN_CONTAINER.runfiles/xla"

# An RCCL prefix to inject. Empty means the library that ships in the image,
# which for the reference image is a build with WarpSpeed compiled in.
RCCL_PREFIX="${RCCL_PREFIX:-}"

# ---------------------------------------------------------------------------
# Arms.
#
# Each arm is "gtest_filter|per_rank_bytes|expect_warp_speed|extra env...".
#
# Sizes are set so that arms compared against each other carry the *same
# aggregate traffic per kernel plan*, because that - not the size of one
# operand - is what the library weighs against its activation threshold. The
# reference library declines at 64 MiB aggregate and activates at 128 MiB, so:
#
#   two buffers x 8 MiB per rank x 8 ranks  = 128 MiB aggregate, two tasks
#   one  buffer x 16 MiB per rank x 8 ranks = 128 MiB aggregate, one task
#
# Those two differ only in how many tasks share the plan, which is exactly the
# variable of interest. Matching on per-rank size instead would have left the
# single-buffer arm below the threshold, so it would have passed by never
# entering the branch - a control that controls for nothing.
# ---------------------------------------------------------------------------
declare -A ARMS=(
  # The case under investigation: two operations in one plan, feature active.
  [grouped_two_on]="WarpSpeedGroupedAllGatherTest.TwoBuffersInOneGroup|8388608|1|"

  # Same aggregate traffic, same feature state, one task instead of two.
  [single_on]="WarpSpeedGroupedAllGatherTest.SingleBuffer|16777216|1|"

  # Same two operations and the same per-plan traffic, submitted as two plans.
  [separate_groups_on]="WarpSpeedGroupedAllGatherTest.TwoBuffersInSeparateGroups|16777216|1|"

  # Same shape, below the activation threshold: isolates traffic from grouping.
  [grouped_two_below_threshold]="WarpSpeedGroupedAllGatherTest.TwoBuffersInOneGroup|4194304|0|"

  # Same shape and traffic, feature explicitly disabled: isolates the feature.
  [grouped_two_feature_off]="WarpSpeedGroupedAllGatherTest.TwoBuffersInOneGroup|8388608|0|RCCL_WARP_SPEED_AUTO=0"

  # More tasks per plan, to see whether severity tracks task count.
  [grouped_four_on]="WarpSpeedGroupedAllGatherTest.FourBuffersInOneGroup|8388608|1|"

  # Small transfers with the threshold lowered. Reaches the same branch for a
  # fraction of the memory and the time, which makes it suitable for a gating
  # lane. It supplements the default-configuration arms above and does not
  # replace them: a lane built only on overridden thresholds keeps passing after
  # the library changes the thresholds it ships with.
  [grouped_two_forced_small]="WarpSpeedGroupedAllGatherTest.TwoBuffersInOneGroup|1048576|1|RCCL_WARP_SPEED_AG_THRESHOLD=1048576"
)

# Order matters for reading the report: the target case first, then controls.
readonly ARM_ORDER=(
  grouped_two_on
  single_on
  separate_groups_on
  grouped_two_below_threshold
  grouped_two_feature_off
  grouped_four_on
  grouped_two_forced_small
)

usage() {
  sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
}

docker_common_args() {
  printf '%s\n' \
    --rm \
    --entrypoint bash \
    --user "$(id -u):$(id -g)" \
    -e "HOME=/cache/home" \
    -v "$HOST_MOUNT:$CONTAINER_MOUNT" \
    -v "$CACHE_DIR:/cache"
}

# GPU access for an unprivileged container user.
#
# `--group-add render` resolves against the *container's* group table, which
# rarely agrees with the host's, so the numeric owners of the device nodes are
# passed instead. Getting this wrong does not produce an error: the runtime
# reports zero devices, and a suite that skips on missing hardware then reports
# a pass. The cases here fail instead, but the environment should still be
# right.
gpu_args() {
  printf '%s\n' \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size=16g
  local node gid
  local -A seen=()
  for node in /dev/kfd /dev/dri/render* /dev/dri/card*; do
    [[ -e "$node" ]] || continue
    gid="$(stat -c '%g' "$node")"
    [[ -n "${seen[$gid]:-}" ]] && continue
    seen[$gid]=1
    printf -- '--group-add\n%s\n' "$gid"
  done
}

# xla_configure.bazelrc is generated and git-ignored, so it is not carried with
# the suite - but the wrong value in it produces a binary that segfaults during
# executor initialization, with a backtrace that points at hipBLASLt and gives
# no hint that the build configuration is at fault. Worth one check here.
check_build_config() {
  local rc="$WORKSPACE/xla_configure.bazelrc"
  if [[ ! -f "$rc" ]]; then
    echo "warning: $rc is missing; run configure.py inside the image first" >&2
    return
  fi
  if grep -qE '^build --config rocm$' "$rc"; then
    cat >&2 <<'MSG'
warning: xla_configure.bazelrc selects "--config rocm", which builds against a
         hermetic ROCm. The resulting binary mixes two C++ standard libraries
         with the container's ROCm and crashes inside std::filesystem before any
         collective runs. Change that line to:

             build --config rocm_clang_local

MSG
  fi
}

do_build() {
  check_build_config
  mkdir -p "$CACHE_DIR/home" "$CACHE_DIR/bazel"
  local -a args
  mapfile -t args < <(docker_common_args)
  echo "building $TARGET in $IMAGE"
  docker run "${args[@]}" --network host "$IMAGE" -lc "
    set -euo pipefail
    cd $CONTAINER_MOUNT/repos/xla
    bazel --output_user_root=/cache/bazel build --jobs=$JOBS --dynamic_mode=off --verbose_failures $TARGET
  "
}

# Resolves what the run will actually load, so a result can never be attributed
# to a library that was not present. Checking that a file exists is not enough;
# the build id ties the log to one specific artifact.
describe_rccl() {
  local prefix="$1"
  local library
  if [[ -n "$prefix" ]]; then
    if [[ -e "$prefix/lib/librccl.so" ]]; then
      library="$prefix/lib/librccl.so"
    elif [[ -e "$prefix/lib64/librccl.so" ]]; then
      library="$prefix/lib64/librccl.so"
    else
      echo "librccl.so not found below $prefix/lib or $prefix/lib64" >&2
      exit 2
    fi
    echo "rccl_source=injected"
    echo "rccl_library=$library"
    echo "rccl_sha256=$(sha256sum "$(readlink -f "$library")" | awk '{print $1}')"
    echo "rccl_build_id=$(readelf -n "$(readlink -f "$library")" 2>/dev/null | awk '/Build ID/ {print $3}' | head -1)"
  else
    echo "rccl_source=image"
  fi
}

classify() {
  local exit_code="$1" log="$2"
  if grep -Eq 'Memory access fault by GPU|Memory Fault Error|HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION' "$log"; then
    echo fault
  elif [[ $exit_code -eq 124 || $exit_code -eq 137 ]]; then
    echo timeout
  elif grep -q 'guard was overwritten' "$log"; then
    echo out_of_bounds_write
  elif grep -q 'payload words are wrong' "$log"; then
    echo mismatch
  elif grep -Eq 'WarpSpeed did not activate|no RCCL debug log was found|cannot be confirmed' "$log"; then
    echo inconclusive
  elif [[ $exit_code -eq 0 ]]; then
    echo pass
  else
    echo failed
  fi
}

do_case() {
  local arm="$1"
  local repeats="${2:-1}"
  local spec="${ARMS[$arm]:-}"
  if [[ -z "$spec" ]]; then
    echo "unknown arm: $arm" >&2
    echo "known arms: ${ARM_ORDER[*]}" >&2
    exit 2
  fi

  IFS='|' read -r filter per_rank_bytes expect_warp_speed extra_env <<<"$spec"
  mkdir -p "$LOG_DIR"
  local log="$LOG_DIR/${arm}.log"
  local container="rccl-bench-${arm}-$$"

  local -a args gpu
  mapfile -t args < <(docker_common_args)
  mapfile -t gpu < <(gpu_args)
  args+=("${gpu[@]}")
  args+=(
    --name "$container"
    -e "RCCL_BENCHMARK_PER_RANK_BYTES=$per_rank_bytes"
    -e "RCCL_BENCHMARK_EXPECT_WARP_SPEED=$expect_warp_speed"
    -e "RCCL_BENCHMARK_REPEATS=$repeats"
    # Keep the library's own log next to the case log. It is the only record of
    # which branch ran, and a container-local path would take it with the
    # container - including in exactly the runs that crash.
    -e "RCCL_BENCHMARK_LOG_DIR=$LOG_DIR_IN_CONTAINER/$arm.rccl"
    # Auto mode ships disabled in some builds, so the arms that target the
    # feature ask for it explicitly rather than hoping for a default.
    -e "RCCL_WARP_SPEED_AUTO=1"
    -e "HSA_NO_SCRATCH_RECLAIM=1"
  )
  if [[ -n "$extra_env" ]]; then
    # Placed after the defaults so an arm can override them.
    args+=(-e "$extra_env")
  fi

  # RUNPATH loses to LD_LIBRARY_PATH, which is what makes injection work at all.
  #
  # Inject only the library under test. Putting a whole foreign ROCm prefix
  # ahead of the one the binary was built against mixes two C++ standard
  # libraries: hipBLASLt passes std::filesystem::path across that boundary
  # during executor initialization and segfaults there, long before any
  # collective runs. RCCL is safe to swap because its interface is C.
  local library_path="/opt/rocm/lib:/opt/rocm/lib64"
  if [[ -n "$RCCL_PREFIX" ]]; then
    args+=(-v "$RCCL_PREFIX:/work/rccl:ro")
    library_path="/work/rccl/lib:/work/rccl/lib64:$library_path"
  fi
  args+=(-e "LD_LIBRARY_PATH=$library_path")

  {
    echo "arm=$arm"
    echo "gtest_filter=$filter"
    echo "per_rank_bytes=$per_rank_bytes"
    echo "expect_warp_speed=$expect_warp_speed"
    echo "extra_env=${extra_env:-none}"
    echo "repeats=$repeats"
    echo "image=$IMAGE"
    echo "image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")"
    echo "xla_commit=$(git -C "$WORKSPACE" rev-parse HEAD)"
    echo "xla_dirty_files=$(git -C "$WORKSPACE" status --porcelain | wc -l)"
    # Read from inside the image: the host ROCm is irrelevant to what runs.
    echo "rocm_version=$(docker run --rm --entrypoint bash "$IMAGE" -lc \
      'cat /opt/rocm/.info/version 2>/dev/null || echo unknown' 2>/dev/null | tr -d '\r')"
    echo "gpus=$(rocm-smi --showid 2>/dev/null | grep -oE 'GPU\[[0-9]+\]' | sort -u | wc -l)"
    describe_rccl "$RCCL_PREFIX"
    echo "timeout_seconds=$TIMEOUT_SECONDS"
    echo "--- output ---"
  } >"$log"

  # Progress goes to stderr so that stdout carries only the tab-separated row,
  # which the matrix collects verbatim.
  echo "[$arm] filter=$filter per_rank=$per_rank_bytes expect_warp_speed=$expect_warp_speed" >&2
  set +e
  timeout --signal=TERM --kill-after=15s "${TIMEOUT_SECONDS}s" \
    docker run "${args[@]}" "$IMAGE" -lc "
      set -uo pipefail
      cd $RUNFILES_IN_CONTAINER
      # Record the library the process actually mapped, not the one that was
      # offered to it. A result attributed to the wrong library is worse than
      # no result.
      echo '--- resolved librccl ---'
      ( LD_DEBUG=libs $BINARY_IN_CONTAINER --gtest_list_tests >/dev/null ) 2>&1 |
        grep -E 'calling init.*librccl|librccl.so.*\\[0\\]' | head -3 || true
      readlink -f \$(ldd $BINARY_IN_CONTAINER 2>/dev/null |
        awk '/librccl/ {print \$3}' | head -1) || true
      echo '--- case ---'
      exec stdbuf -oL -eL $BINARY_IN_CONTAINER --gtest_filter='$filter'
    " >>"$log" 2>&1
  local exit_code=$?
  docker rm -f "$container" >/dev/null 2>&1 || true
  set -e

  local observed
  observed="$(classify "$exit_code" "$log")"
  {
    echo "--- result ---"
    echo "exit_code=$exit_code"
    echo "observed=$observed"
  } >>"$log"
  echo "[$arm] exit=$exit_code observed=$observed log=$log" >&2
  printf '%s\t%s\t%s\n' "$arm" "$exit_code" "$observed"
}

do_matrix() {
  local repeats="${1:-1}"
  mkdir -p "$LOG_DIR"
  local summary="$LOG_DIR/summary.tsv"
  printf 'arm\texit_code\tobserved\n' >"$summary"
  local arm
  for arm in "${ARM_ORDER[@]}"; do
    # An arm that faults must not stop the sweep; the controls are what make
    # the target case interpretable.
    do_case "$arm" "$repeats" >>"$summary" || true
  done
  echo
  echo "summary written to $summary"
  column -t "$summary" 2>/dev/null || cat "$summary"
}

main() {
  local command="${1:-}"
  case "$command" in
    build) do_build ;;
    list) printf '%s\n' "${ARM_ORDER[@]}" ;;
    case) shift; [[ $# -ge 1 ]] || usage; do_case "$@" ;;
    matrix) shift; do_matrix "${1:-1}" ;;
    *) usage ;;
  esac
}

main "$@"
