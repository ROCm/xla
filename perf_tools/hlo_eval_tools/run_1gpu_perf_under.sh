#!/usr/bin/env bash
#
# run_1gpu_perf_under.sh - `bazel run --run_under` shim for run_1gpu_perf.sh.
#
# Usage:
#   bazel run -c opt <flags> \
#     --run_under=/abs/path/to/run_1gpu_perf_under.sh \
#     //xla/tools/multihost_hlo_runner:hlo_runner_main
#
# Bazel appends the binary it just built *after* the --run_under command's own
# arguments, so the runner arrives here as $1 and run_1gpu_perf.sh's
# <runner> <results-dir> [num_repeats] order cannot be expressed directly.
# This shim reorders them, which also keeps the hlo_runner_main path out of the
# workflow entirely: no reconstructing it from `bazel info output_base`.
#
# The results dir and repeat count come from the environment rather than from
# the --run_under string, because Bazel splits that string on whitespace.
#
#   HLO_EVAL_OUT       Output dir for the per-model CSVs. Required, and must be
#                      absolute: `bazel run` starts the command in the runfiles
#                      tree, not in the directory bazel was invoked from.
#   HLO_EVAL_REPEATS   Executions per module (default 5).

set -uo pipefail

die() { echo "error: $*" >&2; exit 2; }

[ "$#" -ge 1 ] || die "no binary passed; use this only as a --run_under command"
[ -n "${HLO_EVAL_OUT:-}" ] || die "HLO_EVAL_OUT is not set"
case "$HLO_EVAL_OUT" in
  /*) ;;
  *) die "HLO_EVAL_OUT must be an absolute path: $HLO_EVAL_OUT" ;;
esac

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec "$HERE/run_1gpu_perf.sh" "$1" "$HLO_EVAL_OUT" "${HLO_EVAL_REPEATS:-5}"
