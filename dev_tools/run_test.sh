#!/usr/bin/env bash
# Run XLA tests the way ROCm CI does, locally or on RBE. See dev_tools/README.md.
{
set -euo pipefail
die() { echo "run_test.sh: error: $*" >&2; exit 2; }

USAGE="usage: run_test.sh [--xla <dir|ref>] [--config=rocm_rbe_dev|rocm_rbe] [--config=ci_*] <targets/bazel args...>"
XLA=.
case "${1:-}" in
  --xla) [ "$#" -ge 2 ] || die "$USAGE"; XLA=$2; shift 2 ;;
  --xla=*) XLA=${1#*=}; shift ;;
  -h|--help) echo "$USAGE"; exit 0 ;;
esac
[ "$#" -ge 1 ] || die "$USAGE"
# A git ref is tested in the worktree xla_checkout.sh creates or reuses for it.
[ -d "$XLA" ] || XLA=$("$(dirname "${BASH_SOURCE[0]}")/xla_checkout.sh" "$XLA") || exit
cd "$XLA"
XLA_DIR=$PWD
RUN_CI=build_tools/rocm/run_xla_ci_build.sh
[ -f "$RUN_CI" ] || die "$XLA_DIR is not an XLA checkout; run from one or pass --xla <dir|ref>"

# One bazel command per worktree; worktrees run in parallel. rbe_hlo_eval.sh
# and xla_checkout.sh honor the same lock. It is fd 9, closed for bazel so its
# server, which outlives the run, does not keep it.
mkdir -p _hlo_eval_rbe; exec 9>_hlo_eval_rbe/.lock
flock -n 9 || die "another run is using the XLA checkout $XLA_DIR"

# The CI config selects the tests (run_xla_ci_build.sh derives the tag filters
# from it; default ci_rocm_cpu). The backend config selects RBE (default: this
# machine). At most one of each: two backends would merge into a broken mix.
CI_CONFIG= RBE=
for a in "$@"; do
  case "$a" in
    --config=ci_rocm_cpu|--config=ci_rocm_untagged|--config=ci_single_gpu|--config=ci_multi_gpu)
      [ -z "$CI_CONFIG" ] || die "pass one CI config, not both $CI_CONFIG and ${a#--config=}"
      CI_CONFIG=${a#--config=} ;;
    --config=rocm_rbe|--config=rocm_rbe_dev)
      [ -z "$RBE" ] || die "pass one RBE backend, not both $RBE and ${a#--config=}"
      RBE=${a#--config=} ;;
  esac
done
ARGS=(--config=rocm_ci_hermetic --config=no_flaky)
[ -n "$CI_CONFIG" ] || { CI_CONFIG=ci_rocm_cpu; ARGS+=(--config=ci_rocm_cpu); }
[ -n "$RBE" ] && ARGS+=(--remote_download_minimal) || RBE=local
# GPU tests that run on this machine: one at a time.
case "$CI_CONFIG" in ci_single_gpu|ci_multi_gpu) ARGS+=(--local_test_jobs=1) ;; esac

# The ROCm settings the checkout's CI pins: the ROCM_* keys of rocm_ci.yml's env
# block and its docker image, under their local_config_rocm names (see
# third_party/gpus/rocm_configure.bzl). Unpinned ones keep that file's default.
CI_YML=.github/workflows/rocm_ci.yml
declare -A PIN=()
while read -r k v; do PIN[$k]=$v; done < <(
  sed -n "s/^  \(ROCM_[A-Z_]*\): *[\"']\{0,1\}\([^\"' #]*\).*/\1 \2/p" "$CI_YML" 2>/dev/null)
PIN[DOCKER_IMAGE]=$(sed -n 's/.*docker-image=\([^"]*\)".*/\1/p' "$CI_YML" 2>/dev/null | head -1)
ARGS+=(--repo_env=ROCM_PATH=)
for m in ROCM_DISTRO_URL=ROCM_DISTRO_URL ROCM_DISTRO_HASH=ROCM_DISTRO_HASH \
         ROCM_AMD_GPU_TARGETS=TF_ROCM_AMDGPU_TARGETS \
         ROCM_SINGLE_GPU_RBE_POOL=TF_ROCM_RBE_SINGLE_GPU_POOL \
         DOCKER_IMAGE=TF_ROCM_RBE_DOCKER_IMAGE; do
  v=${PIN[${m%%=*}]:-}
  [ -z "$v" ] || ARGS+=("--repo_env=${m#*=}=$v")
done
ARGS+=(--repo_env=TF_ROCM_RBE_MULTI_GPU_POOL=linux_x64_multi4)  # not in rocm_ci.yml

SHA=$(git rev-parse --short=10 HEAD 2>/dev/null || echo nogit)
git diff --quiet HEAD 2>/dev/null || SHA+=-dirty
ROCM=${PIN[ROCM_DISTRO_URL]:-rocm_configure.bzl default}
for a in "$@"; do case "$a" in --repo_env=ROCM_DISTRO_URL=*) ROCM=${a#*_URL=} ;; esac; done
printf 'run_test.sh: xla %s @ %s, %s %s, rocm %s\n' "$XLA_DIR" "$SHA" "$CI_CONFIG" "$RBE" "$ROCM" >&2

# Run the checkout's CI script; older copies' `mkdir -p /tf/pkg` needs root, so
# drop it (in memory).
unset XLA_ROCM_TARGET_PATTERN_FILE  # would switch run_xla_ci_build.sh to bazel-diff mode
bash -c "$(sed '/^mkdir -p \/tf\/pkg$/d' "$RUN_CI")" "$PWD/$RUN_CI" "${ARGS[@]}" "$@" 9>&-
exit
}
