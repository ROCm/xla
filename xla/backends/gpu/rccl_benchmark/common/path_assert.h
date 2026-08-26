/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_PATH_ASSERT_H_
#define XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_PATH_ASSERT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla::gpu::rccl_benchmark {

// Evidence that the RCCL code path a case is aiming at actually ran.
//
// "The result is correct" is not evidence that anything was tested. The feature
// these cases target is compiled out of most RCCL release builds and disabled
// by default even when it is compiled in, so a case can pass every correctness
// check while exercising nothing at all. Worse, the default flipped between
// RCCL versions, so the same case can be meaningful against one build and
// vacuous against the next with no visible difference in the result.
//
// Each case therefore declares what it expects the library to do and fails when
// the log disagrees, rather than reporting a pass it has not earned.
struct RcclPathObservation {
  // False when no RCCL debug log was produced at all, which usually means
  // NCCL_DEBUG was not in the environment before the library initialized.
  bool log_available = false;
  int scanned_lines = 0;

  // Whether a collective actually executed under WarpSpeed.
  //
  // The distinction matters and is easy to get wrong. The library logs
  // "WarpSpeed enabled: ..." once per communicator to report that the feature
  // is *available*, and it logs that even for transfers it then declines to
  // use it for. The per-collective decision shows up as
  // "RCCL Warp Speed Channels set to %d. Warps per block is set to %d", which
  // is emitted only on the path taken when the feature is really engaged.
  //
  // Reading the communicator-level line as activation makes a case that never
  // entered the branch look like a case that entered it and behaved.
  bool warp_speed_active = false;
  // "RCCL Warp Speed Channels set to %d. Warps per block is set to %d"
  std::optional<int> warp_speed_channels;
  std::optional<int> warp_speed_warps_per_block;

  // "WarpSpeed enabled: ..." - the feature is compiled in and enabled for the
  // communicator. Necessary but not sufficient for activation.
  bool warp_speed_available = false;
  // "RCCL WarpSpeed not enabled for %s at %zu bytes as it below the warpSpeed
  // threshold" - the library declined for this transfer size. The size it
  // reports is the aggregated traffic of the kernel plan, not one operand.
  bool warp_speed_below_threshold = false;
  std::vector<int64_t> warp_speed_declined_bytes;
  // "Overriding %s algorithm with RING ... as WarpSpeed is requested"
  bool warp_speed_forced_ring = false;

  // "RCCL Tuning index:%d" - selects which tuning table supplies the protocol
  // and channel-count thresholds, so it belongs in every result record.
  std::optional<int> tuning_index;

  // Channel counts from "post-adjustment based on threadThreshold:... nc:%i".
  std::vector<int> channel_counts;

  // Version banner, recorded so a result can be attributed to a library build.
  std::optional<std::string> version_line;

  // Lines that produced the fields above, quoted back in failure messages so a
  // disagreement can be judged without re-running.
  std::vector<std::string> evidence;

  std::string DebugString() const;
};

// Directory this process told RCCL to write its debug log into.
std::string RcclDebugLogDir();

// Puts NCCL_DEBUG, NCCL_DEBUG_SUBSYS and NCCL_DEBUG_FILE into the environment
// unless the caller already set them, and creates the log directory.
//
// Must run before the first RCCL entry point: the library caches every
// parameter it reads on first use, so anything set later is ignored. In
// practice that means the process main, before any test body.
absl::Status ConfigureRcclDebugLogging();

// Parses every log the process produced. Call after the collective ran.
absl::StatusOr<RcclPathObservation> ObserveRcclPath();

// Fails unless WarpSpeed activation matches `expected`. The error explains
// which signal was found instead, because "the feature never turned on" and
// "the feature turned on and behaved" are indistinguishable from the result
// alone.
absl::Status ExpectWarpSpeed(const RcclPathObservation& observation,
                             bool expected);

}  // namespace xla::gpu::rccl_benchmark

#endif  // XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_PATH_ASSERT_H_
