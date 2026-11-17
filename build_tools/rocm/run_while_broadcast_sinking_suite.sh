#!/usr/bin/env bash
# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# run_while_broadcast_sinking_suite.sh runs the hlo_eval_tools 1-GPU perf suite
# (run_1gpu_perf.sh) with WhileLoopConstantSinking keeping broadcasts of
# constants in the while loop state (off) and sinking them into the loop (on),
# one results dir per setting.
# Local experiment helper, not for upstream.
#
# Usage
#   run_while_broadcast_sinking_suite.sh [settings] [runner] [prefix]
#
#   settings  Space separated subset of "off on" (default "off on").
#               off  --xla_gpu_experimental_while_loop_sink_broadcast_constants=false
#               on   --xla_gpu_experimental_while_loop_sink_broadcast_constants=true
#   runner    hlo_runner_main to use
#             (default bazel-bin/xla/tools/multihost_hlo_runner/hlo_runner_main).
#   prefix    Results dir prefix (default results-while-bcast-sink-<format>),
#             created in the current directory as <prefix>-<setting>.
#
# INPUT_FORMAT selects the modules (default stablehlo, the .mlirbc modules run
# natively with --input_format=stablehlo). Set INPUT_FORMAT=hlo for the HLO
# text dumps. The default prefix includes the format because the suite refuses
# to mix StableHLO and HLO text CSVs in one dir.
# HLO_EVAL_TOOLS points at the hlo_eval_tools checkout
# (default /mnt/anurmukh/xla-dev-infra/perf_tools/hlo_eval_tools).
# Any XLA_FLAGS already in the environment are kept and the per setting flags
# are appended. RESUME=1 (the suite default) skips models whose CSV already
# exists, so a rerun into the same dir only fills in what is missing.

set -uo pipefail

export INPUT_FORMAT=${INPUT_FORMAT:-stablehlo}
SETTINGS=${1:-off on}
RUNNER=${2:-bazel-bin/xla/tools/multihost_hlo_runner/hlo_runner_main}
PREFIX=${3:-results-while-bcast-sink-$INPUT_FORMAT}
HLO_EVAL_TOOLS=${HLO_EVAL_TOOLS:-/mnt/anurmukh/xla-dev-infra/perf_tools/hlo_eval_tools}
SUITE=$HLO_EVAL_TOOLS/run_1gpu_perf.sh

SWITCH=--xla_gpu_experimental_while_loop_sink_broadcast_constants

[ -x "$RUNNER" ] || { echo "error: runner not executable: $RUNNER" >&2; exit 2; }
[ -x "$SUITE" ] || { echo "error: suite script not found: $SUITE" >&2; exit 2; }

# Record the partition mode next to the results. Timings only compare within
# one mode.
amd-smi partition 2>/dev/null | grep -E "NPS[12] +[SDQC]PX" | head -1

fail=0
for s in $SETTINGS; do
  case $s in
    off) flags="$SWITCH=false" ;;
    on)  flags="$SWITCH=true" ;;
    *) echo "error: unknown setting '$s' (use off or on)" >&2; exit 2 ;;
  esac
  out="$PREFIX-$s"
  echo "=== setting $s ($INPUT_FORMAT) -> $out  XLA_FLAGS+='$flags'" >&2
  ROCPROFILER_REGISTER_ENABLED=0 \
  XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }$flags" \
    "$SUITE" "$RUNNER" "$out" || fail=1
done
exit "$fail"
