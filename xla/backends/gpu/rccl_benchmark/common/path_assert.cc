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

#include "xla/backends/gpu/rccl_benchmark/common/path_assert.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_macros.h"
#include "tsl/platform/path.h"

namespace xla::gpu::rccl_benchmark {
namespace {

// Markers emitted by RCCL builds that have WarpSpeed compiled in. They are
// matched as substrings of the formatted line rather than reconstructed from
// the format strings, so a wording change shows up as "path could not be
// confirmed" instead of a silent pass.
constexpr absl::string_view kWarpSpeedEnabled = "WarpSpeed enabled:";
constexpr absl::string_view kWarpSpeedBelowThreshold =
    "below the warpSpeed threshold";
constexpr absl::string_view kWarpSpeedForcedRing =
    "as WarpSpeed is requested and only supports RING";
constexpr absl::string_view kWarpSpeedChannels = "RCCL Warp Speed Channels set to";
constexpr absl::string_view kTuningIndex = "RCCL Tuning index:";
constexpr absl::string_view kPostAdjustment = "post-adjustment based on threadThreshold:";
constexpr absl::string_view kVersionBanner = "RCCL version";

constexpr int kMaxEvidenceLines = 24;

// Returns the integer that follows `key` in `line`, if any. Handles both
// "key:123" and "key 123" spellings, which RCCL uses interchangeably.
std::optional<int> IntAfter(absl::string_view line, absl::string_view key) {
  const size_t pos = line.find(key);
  if (pos == absl::string_view::npos) {
    return std::nullopt;
  }
  absl::string_view rest = line.substr(pos + key.size());
  while (!rest.empty() && (rest.front() == ' ' || rest.front() == ':')) {
    rest.remove_prefix(1);
  }
  size_t end = 0;
  while (end < rest.size() && absl::ascii_isdigit(rest[end])) {
    ++end;
  }
  if (end == 0) {
    return std::nullopt;
  }
  int value = 0;
  if (!absl::SimpleAtoi(rest.substr(0, end), &value)) {
    return std::nullopt;
  }
  return value;
}

// Returns the integer that appears between `before` and `after` in `line`.
// Used for the transfer size the library reports when it declines to use a
// feature, which is the number that explains the decision.
std::optional<int64_t> Int64Between(absl::string_view line,
                                    absl::string_view before,
                                    absl::string_view after) {
  const size_t start = line.find(before);
  if (start == absl::string_view::npos) {
    return std::nullopt;
  }
  absl::string_view rest = line.substr(start + before.size());
  const size_t end = rest.find(after);
  if (end == absl::string_view::npos) {
    return std::nullopt;
  }
  int64_t value = 0;
  if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(rest.substr(0, end)),
                        &value)) {
    return std::nullopt;
  }
  return value;
}

void RecordEvidence(RcclPathObservation& observation, absl::string_view line) {
  if (observation.evidence.size() >= kMaxEvidenceLines) {
    return;
  }
  observation.evidence.emplace_back(absl::StripAsciiWhitespace(line));
}

void ScanLine(absl::string_view line, RcclPathObservation& observation) {
  ++observation.scanned_lines;

  if (absl::StrContains(line, kWarpSpeedEnabled)) {
    observation.warp_speed_available = true;
    RecordEvidence(observation, line);
  }
  if (absl::StrContains(line, kWarpSpeedBelowThreshold)) {
    observation.warp_speed_below_threshold = true;
    if (std::optional<int64_t> bytes = Int64Between(line, " at ", " bytes");
        bytes.has_value()) {
      observation.warp_speed_declined_bytes.push_back(*bytes);
    }
    RecordEvidence(observation, line);
  }
  if (absl::StrContains(line, kWarpSpeedForcedRing)) {
    observation.warp_speed_forced_ring = true;
    RecordEvidence(observation, line);
  }
  if (absl::StrContains(line, kWarpSpeedChannels)) {
    // The only message the library emits from the path it takes when a
    // collective really runs under WarpSpeed.
    observation.warp_speed_active = true;
    observation.warp_speed_channels = IntAfter(line, kWarpSpeedChannels);
    observation.warp_speed_warps_per_block =
        IntAfter(line, "Warps per block is set to");
    RecordEvidence(observation, line);
  }
  if (!observation.tuning_index.has_value() &&
      absl::StrContains(line, kTuningIndex)) {
    observation.tuning_index = IntAfter(line, kTuningIndex);
    RecordEvidence(observation, line);
  }
  if (absl::StrContains(line, kPostAdjustment)) {
    if (std::optional<int> nc = IntAfter(line, "nc:"); nc.has_value()) {
      observation.channel_counts.push_back(*nc);
    }
  }
  if (!observation.version_line.has_value() &&
      absl::StrContains(line, kVersionBanner)) {
    observation.version_line = std::string(absl::StripAsciiWhitespace(line));
    RecordEvidence(observation, line);
  }
}

}  // namespace

std::string RcclDebugLogDir() {
  // Deliberately dependency-free so the same helper works from a gtest binary
  // and from a standalone performance binary.
  for (const char* name : {"RCCL_BENCHMARK_LOG_DIR", "TEST_TMPDIR", "TMPDIR"}) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
      continue;
    }
    if (absl::string_view(name) == "RCCL_BENCHMARK_LOG_DIR") {
      return std::string(value);
    }
    return tsl::io::JoinPath(value, "rccl_benchmark_logs");
  }
  return tsl::io::JoinPath("/tmp", "rccl_benchmark_logs");
}

absl::Status ConfigureRcclDebugLogging() {
  const std::string dir = RcclDebugLogDir();
  RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(dir));

  // Values a caller supplied are left alone, so the runner can redirect the log
  // or raise verbosity without editing a case.
  setenv("RCCL_BENCHMARK_LOG_DIR", dir.c_str(), /*overwrite=*/0);

  // NCCL_DEBUG is the exception: it is overwritten unless it is already at a
  // level that produces the messages the assertions read.
  //
  // Container images commonly ship NCCL_DEBUG=VERSION, which prints a banner
  // and nothing else. Respecting that would leave every case unable to confirm
  // its path - the correct outcome being reported for the wrong reason, and one
  // that looks identical to a genuinely missing feature.
  const char* debug_level = std::getenv("NCCL_DEBUG");
  const bool level_is_usable =
      debug_level != nullptr && (absl::EqualsIgnoreCase(debug_level, "INFO") ||
                                 absl::EqualsIgnoreCase(debug_level, "TRACE"));
  if (!level_is_usable) {
    setenv("NCCL_DEBUG", "INFO", /*overwrite=*/1);
  }

  // The subsystems carrying those messages. "ALL" looks like the safer choice
  // and is not: the reference library emits nothing under it.
  setenv("NCCL_DEBUG_SUBSYS", "INIT,TUNING,COLL", /*overwrite=*/0);
  const std::string log_file = tsl::io::JoinPath(dir, "rccl.%h.%p.log");
  setenv("NCCL_DEBUG_FILE", log_file.c_str(), /*overwrite=*/0);
  return absl::OkStatus();
}

absl::StatusOr<RcclPathObservation> ObserveRcclPath() {
  RcclPathObservation observation;
  const std::string dir = RcclDebugLogDir();

  tsl::Env* env = tsl::Env::Default();
  std::vector<std::string> children;
  if (!env->GetChildren(dir, &children).ok()) {
    return observation;
  }

  for (const std::string& child : children) {
    const std::string path = tsl::io::JoinPath(dir, child);
    std::string contents;
    if (!tsl::ReadFileToString(env, path, &contents).ok()) {
      continue;
    }
    observation.log_available = true;
    for (absl::string_view line : absl::StrSplit(contents, '\n')) {
      if (!line.empty()) {
        ScanLine(line, observation);
      }
    }
  }
  return observation;
}

std::string RcclPathObservation::DebugString() const {
  auto yes_no = [](bool value) { return value ? "yes" : "no"; };
  std::string out = absl::StrFormat(
      "log_available=%s scanned_lines=%d warp_speed_active=%s "
      "warp_speed_available=%s warp_speed_below_threshold=%s "
      "warp_speed_forced_ring=%s",
      yes_no(log_available), scanned_lines, yes_no(warp_speed_active),
      yes_no(warp_speed_available), yes_no(warp_speed_below_threshold),
      yes_no(warp_speed_forced_ring));
  if (!warp_speed_declined_bytes.empty()) {
    absl::StrAppendFormat(&out, " declined_at_bytes=[%s]",
                          absl::StrJoin(warp_speed_declined_bytes, ","));
  }
  if (warp_speed_channels.has_value()) {
    absl::StrAppendFormat(&out, " warp_speed_channels=%d",
                          *warp_speed_channels);
  }
  if (warp_speed_warps_per_block.has_value()) {
    absl::StrAppendFormat(&out, " warps_per_block=%d",
                          *warp_speed_warps_per_block);
  }
  if (tuning_index.has_value()) {
    absl::StrAppendFormat(&out, " tuning_index=%d", *tuning_index);
  }
  if (!channel_counts.empty()) {
    absl::StrAppendFormat(&out, " channel_counts=[%s]",
                          absl::StrJoin(channel_counts, ","));
  }
  if (version_line.has_value()) {
    absl::StrAppendFormat(&out, " version=\"%s\"", *version_line);
  }
  if (!evidence.empty()) {
    absl::StrAppendFormat(&out, "\nevidence:\n  %s",
                          absl::StrJoin(evidence, "\n  "));
  }
  return out;
}

absl::Status ExpectWarpSpeed(const RcclPathObservation& observation,
                             bool expected) {
  if (!observation.log_available) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "No RCCL debug log was found under %s, so it cannot be confirmed that "
        "the intended code path ran. The case is inconclusive, not passing. "
        "NCCL_DEBUG must be set before the library initializes.",
        RcclDebugLogDir()));
  }

  if (observation.warp_speed_active == expected) {
    return absl::OkStatus();
  }

  if (expected) {
    std::string reason;
    if (observation.warp_speed_below_threshold) {
      reason = absl::StrFormat(
          "the library declined it for this transfer (reported %s bytes, "
          "aggregated over the kernel plan) as below the WarpSpeed threshold. "
          "Raise the transfer size, add operations to the group, or lower "
          "RCCL_WARP_SPEED_AG_THRESHOLD",
          observation.warp_speed_declined_bytes.empty()
              ? "an unparsed number of"
              : absl::StrJoin(observation.warp_speed_declined_bytes, "/"));
    } else if (observation.warp_speed_available) {
      reason =
          "the communicator reports the feature as available but no collective "
          "took the path; availability is not activation";
    } else {
      reason =
          "no WarpSpeed message of any kind was logged, which is what a build "
          "without ENABLE_WARP_SPEED, or one with RCCL_WARP_SPEED_AUTO=0, "
          "looks like";
    }
    return absl::FailedPreconditionError(absl::StrFormat(
        "This case targets WarpSpeed but no collective ran under it: %s. "
        "Passing here would mean nothing, so the case fails instead.\n%s",
        reason, observation.DebugString()));
  }

  return absl::FailedPreconditionError(absl::StrFormat(
      "This case is a control arm that requires WarpSpeed to stay off, but the "
      "library activated it. The arm no longer isolates what it was meant "
      "to.\n%s",
      observation.DebugString()));
}

}  // namespace xla::gpu::rccl_benchmark
