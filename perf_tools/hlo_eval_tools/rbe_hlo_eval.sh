#!/usr/bin/env bash
# Run hlo_eval_tools models on the ROCm RBE GPU pools with a hermetic ROCm.
#   rbe_hlo_eval.sh [--xla <dir|ref>] <all|model[,model...]> [<leaf>...] [-- <bazel args>]
#   rbe_hlo_eval.sh --xla rocm/main all     # every 1-GPU workload (40 leaves)
#   rbe_hlo_eval.sh sdxl,vit inference/1gpu # from the root of an XLA checkout
# --xla is the XLA checkout to build and measure (default: current directory),
# or a git ref, checked out detached into worktrees/<ref> of this repo.
# Environment variables and output: README.md, "On ROCm RBE".
# The body is one { } block, so bash parses it fully before running and editing
# this file during a run cannot break that run.
{
set -euo pipefail
shopt -s nullglob
die() { echo "rbe_hlo_eval.sh: error: $*" >&2; exit 2; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USAGE="usage: rbe_hlo_eval.sh [--xla <dir|ref>] <all|model[,model...]> [<leaf>...] [-- <bazel args>]"
XLA=${XLA_DIR:-.}
case "${1:-}" in
  --xla) [ "$#" -ge 2 ] || die "$USAGE"; XLA=$2; shift 2 ;;
  --xla=*) XLA=${1#*=}; shift ;;
esac
if [ ! -d "$XLA" ]; then
  # A git ref: (re)use a detached worktree at worktrees/<ref> of this repo.
  repo=$(git -C "$HERE" rev-parse --show-toplevel)
  if [[ "$XLA" == */* ]] && git -C "$repo" remote | grep -qx "${XLA%%/*}"; then
    git -C "$repo" fetch -q "${XLA%%/*}" "${XLA#*/}"
  fi
  sha=$(git -C "$repo" rev-parse -q --verify "$XLA^{commit}") || die "--xla $XLA: no such directory or git ref"
  wt=$repo/worktrees/${XLA//\//-}
  if [ -d "$wt" ]; then
    git -C "$wt" diff --quiet HEAD || die "$wt has local changes"
    git -C "$wt" checkout -q --detach "$sha"
  else
    git -C "$repo" worktree add -q --detach "$wt" "$sha"
  fi
  XLA=$wt
fi
XLA_DIR=$(realpath "$XLA")
[ -f "$XLA_DIR/build_tools/rocm/rocm_xla_ci.bazelrc" ] ||
  die "$XLA_DIR is not an XLA checkout; run from one or pass --xla <dir|ref>"
XLA_SHA=$(git -C "$XLA_DIR" rev-parse --short=10 HEAD 2>/dev/null || echo nogit)
git -C "$XLA_DIR" diff --quiet HEAD 2>/dev/null || XLA_SHA+=-dirty
REPEATS=${REPEATS:-20} STAT=${STAT:-median} MAX_GPUS=${MAX_GPUS:-1}
NOISY_PCT=${NOISY_PCT:-2} RETRY_NOISY=${RETRY_NOISY:-1}
MULTI_GPU_POOL=${MULTI_GPU_POOL:-linux_x64_multi4}

# Defaults come from the XLA checkout: what its ROCm CI pins in rocm_ci.yml,
# and for the tarball the rocm_redist.bzl entry the build itself falls back to.
CI_YML=$XLA_DIR/.github/workflows/rocm_ci.yml
ci_env() { sed -n "s/^  $1: *[\"']\{0,1\}\([^\"' #]*\).*/\1/p" "$CI_YML" 2>/dev/null | head -1 || true; }
redist() {
  local v
  v=$(sed -n 's/^_DEFAULT_ROCM_DISTRO_VERSION *= *"\(.*\)".*/\1/p' \
    "$XLA_DIR/third_party/gpus/rocm_configure.bzl" 2>/dev/null || true)
  [ -n "$v" ] && sed -n "/\"$v\"/,\$s/.*\"$1\": *\"\([^\"]*\)\".*/\1/p" \
    "$XLA_DIR/third_party/gpus/rocm/rocm_redist.bzl" | head -1 || true
}
if [ -n "${ROCM_DISTRO_URL:-}${ROCM_DISTRO_HASH:-}" ]; then
  [ -n "${ROCM_DISTRO_URL:-}" ] && [ -n "${ROCM_DISTRO_HASH:-}" ] ||
    die "set both ROCM_DISTRO_URL and ROCM_DISTRO_HASH"
  DISTRO_URL=$ROCM_DISTRO_URL DISTRO_HASH=$ROCM_DISTRO_HASH
else
  DISTRO_URL=$(ci_env ROCM_DISTRO_URL) DISTRO_HASH=$(ci_env ROCM_DISTRO_HASH)
  [ -n "$DISTRO_URL" ] || DISTRO_URL=$(redist url) DISTRO_HASH=$(redist sha256)
fi
DOCKER_IMAGE=${ROCM_DOCKER_IMAGE:-$(sed -n 's/.*docker-image=\([^"]*\)".*/\1/p' "$CI_YML" 2>/dev/null | head -1 || true)}
GPU_TARGET=${ROCM_GPU_TARGET:-$(ci_env ROCM_AMD_GPU_TARGETS)}
GPU_POOL=$(ci_env ROCM_SINGLE_GPU_RBE_POOL)
for v in DISTRO_URL DISTRO_HASH DOCKER_IMAGE GPU_TARGET GPU_POOL; do
  [ -n "${!v}" ] || die "$v not found in $CI_YML (or rocm_redist.bzl)"
done
OB_TAG=$(printf '%s\n%s' "$DISTRO_URL" "$DISTRO_HASH" | sha256sum | cut -c1-12)
OUTPUT_BASE=${XLA_OUTPUT_BASE:-$HOME/.cache/bazel/xla-rocm-$OB_TAG}

[ "$#" -ge 1 ] || die "$USAGE"
MODELS=$1; shift
LEAVES=()
while [ "$#" -gt 0 ] && [ "$1" != -- ]; do LEAVES+=("${1%/}"); shift; done
[ "${1:-}" = -- ] && shift

MODEL_DIRS=()
if [ "$MODELS" = all ]; then
  for d in "$HERE"/*/*/; do
    if [ -d "$d/inference" ] || [ -d "$d/training" ]; then MODEL_DIRS+=("${d%/}"); fi
  done
else
  for m in ${MODELS//,/ }; do
    hits=("$HERE"/*/"$m")
    [ "${#hits[@]}" -eq 1 ] && [ -d "${hits[0]}" ] || die "unknown or ambiguous model: $m"
    MODEL_DIRS+=("${hits[0]}")
  done
fi

# One throwaway package per model, one sh_test per leaf. The modules are staged
# under hlo_eval_tools/ so run_hlo_eval.sh names the CSVs as it does locally.
RUNNER=//xla/tools/multihost_hlo_runner:hlo_runner_main
TARGETS=() skipped=0
for mdir in "${MODEL_DIRS[@]}"; do
  model=${mdir##*/} rel=${mdir#"$HERE"/}
  pkg=_hlo_eval_rbe/$model
  rm -rf "${XLA_DIR:?}/$pkg"; mkdir -p "$XLA_DIR/$pkg"
  ln -s "$HERE/run_hlo_eval.sh" "$XLA_DIR/$pkg/"
  cat > "$XLA_DIR/$pkg/test.sh" <<'EOF'
#!/usr/bin/env bash
echo "worker: $(cat /proc/sys/kernel/hostname)"
export TMPDIR=$TEST_TMPDIR
exec bash "$PWD/$2" "$PWD/$1" "$PWD/$3" "$TEST_UNDECLARED_OUTPUTS_DIR" "$HLO_EVAL_REPEATS"
EOF
  chmod +x "$XLA_DIR/$pkg/test.sh"
  printf '%s\n' 'load("@rules_shell//shell:sh_test.bzl", "sh_test")' \
    'load("//xla/tsl/platform:build_config_root.bzl", "tf_exec_properties")' > "$XLA_DIR/$pkg/BUILD"
  leaves=("${LEAVES[@]}")
  [ "${#leaves[@]}" -gt 0 ] || leaves=($(cd "$mdir" && echo inference/*gpu training))
  for leaf in "${leaves[@]}"; do
    files=("$mdir/$leaf"/*.txt "$mdir/$leaf"/*.hlo)
    n=$( ((${#files[@]})) && grep -om1 'num_partitions=[0-9]*' "${files[0]}" | cut -d= -f2 || true)
    if [ "${#files[@]}" -eq 0 ] || [ "${n:-1}" -gt "$MAX_GPUS" ]; then skipped=$((skipped + 1)); continue; fi
    mkdir -p "$XLA_DIR/$pkg/hlo_eval_tools/$rel/$leaf"
    ln -s "${files[@]}" "$XLA_DIR/$pkg/hlo_eval_tools/$rel/$leaf/"
    tags='"external", "gpu", "requires-gpu-amd", "requires-gpu-rocm", "manual"'
    [ "${n:-1}" -eq 1 ] || tags="$tags, \"multi_gpu\""
    cat >> "$XLA_DIR/$pkg/BUILD" <<EOF

sh_test(
    name = "${leaf//\//_}",
    size = "enormous",
    srcs = ["test.sh"],
    args = ["\$(rootpath $RUNNER)", "\$(rootpath run_hlo_eval.sh)", "$pkg/hlo_eval_tools/$rel/$leaf"],
    data = ["run_hlo_eval.sh", "$RUNNER"] + glob(["hlo_eval_tools/$rel/$leaf/*"]),
    exec_properties = tf_exec_properties({"tags": [$tags]}),
    tags = [$tags],
)
EOF
    TARGETS+=("//$pkg:${leaf//\//_}")
  done
done
[ "${#TARGETS[@]}" -gt 0 ] || die "nothing to run"

TEST_ENV=(--test_env=HLO_EVAL_REPEATS="$REPEATS" --test_env=STAT="$STAT")
for v in CMD_BUFFER ARG_MODE SETTLE_SEC XLA_FLAGS; do
  [ -z "${!v+x}" ] || TEST_ENV+=(--test_env="$v=${!v}")
done
RESULTS_DIR=$(realpath -m "${RESULTS_DIR:-hlo_eval_results/$XLA_SHA-${MODELS//,/+}-$(date +%Y%m%d-%H%M%S)}")
mkdir -p "$RESULTS_DIR"
printf '%s\n' "xla: $XLA_DIR @ $XLA_SHA ($(git -C "$XLA_DIR" log -1 --format='%cd %s' --date=short 2>/dev/null || true))" \
  "rocm: $DISTRO_URL $DISTRO_HASH" "image: $DOCKER_IMAGE" "models: $MODELS ${LEAVES[*]}" \
  "repeats: $REPEATS stat: $STAT max_gpus: $MAX_GPUS bazel args: $*" > "$RESULTS_DIR/run_info.txt"
printf 'rbe_hlo_eval.sh: %s leaves (%s skipped: no modules or > MAX_GPUS=%s GPUs), repeats=%s, stat=%s\n  xla %s @ %s\n  distro %s\n  output_base %s\n' \
  "${#TARGETS[@]}" "$skipped" "$MAX_GPUS" "$REPEATS" "$STAT" "$XLA_DIR" "$XLA_SHA" "$DISTRO_URL" "$OUTPUT_BASE" >&2

cd "$XLA_DIR"
run_bazel() {
  bazel --output_base="$OUTPUT_BASE" --bazelrc=build_tools/rocm/rocm_xla_ci.bazelrc test \
    --config=rocm_ci_hermetic --config=ci_single_gpu --config="${RBE:-rocm_rbe_dev}" \
    --keep_going --test_output=errors --color=yes \
    --sandbox_add_mount_pair=/dev/null:/etc/ld.so.cache \
    --remote_download_minimal --remote_download_regex='.*/testlogs/_hlo_eval_rbe/.*' \
    --nozip_undeclared_test_outputs --nocache_test_results --test_sharding_strategy=disabled \
    "${TEST_ENV[@]}" \
    --repo_env=ROCM_PATH= --repo_env=ROCM_DISTRO_URL="$DISTRO_URL" \
    --repo_env=ROCM_DISTRO_HASH="$DISTRO_HASH" \
    --repo_env=TF_ROCM_RBE_DOCKER_IMAGE="$DOCKER_IMAGE" \
    --repo_env=TF_ROCM_AMDGPU_TARGETS="$GPU_TARGET" \
    --repo_env=TF_ROCM_RBE_SINGLE_GPU_POOL="$GPU_POOL" \
    --repo_env=TF_ROCM_RBE_MULTI_GPU_POOL="$MULTI_GPU_POOL" \
    "$@" "${TARGETS_TO_RUN[@]}"
}
# Copy the targets' test logs to $RESULTS_DIR/<model>/<leaf>/attempts/<round>/.
# Only logs written by this round count: Bazel keeps the previous run's logs of
# tests it did not run (e.g. after a build failure), and those leaves must show
# up as NO_DATA, not as stale timings.
collect() {
  local round=$1 t m l; shift
  for t; do
    m=${t#//_hlo_eval_rbe/} l=${t#*:}; m=${m%%:*}
    mkdir -p "$RESULTS_DIR/$m/$l/attempts/$round"
    [ "bazel-testlogs/_hlo_eval_rbe/$m/$l/test.log" -nt "$RESULTS_DIR/.round_start" ] || continue
    cp -rf "bazel-testlogs/_hlo_eval_rbe/$m/$l/test.log" "bazel-testlogs/_hlo_eval_rbe/$m/$l/test.outputs/." \
      "$RESULTS_DIR/$m/$l/attempts/$round/" 2>/dev/null || true
  done
}

rc=0
TARGETS_TO_RUN=("${TARGETS[@]}")
touch "$RESULTS_DIR/.round_start"
run_bazel "$@" || rc=$?
collect 1 "${TARGETS[@]}"
# Rerun leaves whose spread is too wide (failed leaves have no timings).
for ((round = 2; round <= RETRY_NOISY + 1; round++)); do
  python3 -I "$HERE/hlo_eval_summarize.py" "$RESULTS_DIR" "$NOISY_PCT" > /dev/null
  mapfile -t TARGETS_TO_RUN < "$RESULTS_DIR/noisy.txt"
  [ "${#TARGETS_TO_RUN[@]}" -gt 0 ] || break
  echo "rbe_hlo_eval.sh: rerunning noisy leaves: ${TARGETS_TO_RUN[*]}" >&2
  touch "$RESULTS_DIR/.round_start"
  run_bazel "$@" || true
  collect "$round" "${TARGETS_TO_RUN[@]}"
done
python3 -I "$HERE/hlo_eval_summarize.py" "$RESULTS_DIR" "$NOISY_PCT"
echo "results: $RESULTS_DIR" >&2
[ "$rc" -eq 0 ] || echo "rbe_hlo_eval.sh: warning: bazel test failed (exit $rc); leaves without new results are NO_DATA" >&2
exit "$rc"
}
