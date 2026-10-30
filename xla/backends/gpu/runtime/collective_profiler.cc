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

#include "xla/backends/gpu/runtime/collective_profiler.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {
namespace {

// Upper bound on in-flight timers per device. Above it the host waits for the
// oldest timer, which keeps the number of live GPU events bounded.
constexpr size_t kMaxPendingPerDevice = 1 << 16;

thread_local bool in_nested_scope = false;

std::string FormatBytes(uint64_t bytes) {
  static constexpr const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  int unit = 0;
  while (unit < 4 && bytes >= 1024 && bytes % 1024 == 0) {
    bytes /= 1024;
    ++unit;
  }
  return absl::StrCat(bytes, " ", kUnits[unit]);
}

std::string FormatBucket(int bucket) {
  if (bucket < 0) return "0 B";
  return absl::StrCat("[", FormatBytes(uint64_t{1} << bucket), ", ",
                      FormatBytes(uint64_t{1} << (bucket + 1)), ")");
}

double ToMs(absl::Duration d) { return absl::ToDoubleMilliseconds(d); }
double ToUs(absl::Duration d) { return absl::ToDoubleMicroseconds(d); }

CollectiveProfiler* CreateFromEnv() {
  const char* env = std::getenv("XLA_GPU_COLLECTIVE_TIMING");
  if (env == nullptr || *env == '\0') return nullptr;
  int64_t skip_rounds = 0;
  if (!absl::SimpleAtoi(env, &skip_rounds) || skip_rounds < 0) {
    LOG(WARNING) << "Ignoring invalid XLA_GPU_COLLECTIVE_TIMING=" << env
                 << "; expected a non-negative number of rounds to skip";
    return nullptr;
  }
  auto* profiler = new CollectiveProfiler(skip_rounds);
  std::atexit([] { CollectiveProfiler::Get()->Print(); });
  LOG(INFO) << "Collective timing enabled, skipping the first " << skip_rounds
            << " executable runs per device";
  return profiler;
}

}  // namespace

void CollectiveProfiler::Stats::Add(absl::Duration d) {
  ++count;
  total += d;
  min = std::min(min, d);
  max = std::max(max, d);
}

CollectiveProfiler* CollectiveProfiler::Get() {
  static CollectiveProfiler* const profiler = CreateFromEnv();
  return profiler;
}

CollectiveProfiler::CollectiveProfiler(int64_t skip_rounds)
    : skip_rounds_(skip_rounds) {}

CollectiveProfiler::NestedScope::NestedScope() : prev_(in_nested_scope) {
  in_nested_scope = true;
}

CollectiveProfiler::NestedScope::~NestedScope() { in_nested_scope = prev_; }

bool CollectiveProfiler::InNestedScope() { return in_nested_scope; }

int CollectiveProfiler::SizeBucket(int64_t bytes) {
  if (bytes <= 0) return -1;
  return absl::bit_width(static_cast<uint64_t>(bytes)) - 1;
}

CollectiveProfiler::DeviceState& CollectiveProfiler::GetDeviceState(
    int device_ordinal) {
  absl::MutexLock lock(devices_mu_);
  return devices_[device_ordinal];
}

void CollectiveProfiler::BeginRound(int device_ordinal) {
  DeviceState& state = GetDeviceState(device_ordinal);
  state.rounds.fetch_add(1, std::memory_order_relaxed);
  Harvest(state, /*block_oldest=*/false);
}

bool CollectiveProfiler::IsActive(int device_ordinal) {
  if (in_nested_scope) return false;
  return GetDeviceState(device_ordinal)
             .rounds.load(std::memory_order_relaxed) > skip_rounds_;
}

absl::Status CollectiveProfiler::Time(stream_executor::Stream& stream, std::string name,
                                      int64_t bytes,
                                      absl::FunctionRef<absl::Status()> fn) {
  int device_ordinal = stream.parent()->device_ordinal();
  if (!IsActive(device_ordinal)) return fn();

  absl::StatusOr<std::unique_ptr<stream_executor::EventBasedTimer>> timer =
      stream.CreateEventBasedTimer(/*use_delay_kernel=*/false);
  if (!timer.ok()) {
    LOG_FIRST_N(WARNING, 1) << "Collective timing: failed to create timer: "
                            << timer.status();
    return fn();
  }

  absl::Status status = fn();
  if (!status.ok()) return status;

  if (absl::Status stopped = (*timer)->Stop(); !stopped.ok()) {
    LOG_FIRST_N(WARNING, 1)
        << "Collective timing: failed to stop timer: " << stopped;
    return status;
  }

  DeviceState& state = GetDeviceState(device_ordinal);
  bool over_limit;
  {
    absl::MutexLock lock(state.mu);
    state.pending.push_back(
        Pending{*std::move(timer), std::move(name), bytes});
    over_limit = state.pending.size() > kMaxPendingPerDevice;
  }
  if (over_limit) Harvest(state, /*block_oldest=*/true);
  return status;
}

void CollectiveProfiler::Accumulate(const Pending& pending,
                                    absl::Duration duration) {
  absl::MutexLock lock(aggregates_mu_);
  aggregates_[pending.name][SizeBucket(pending.bytes)].Add(duration);
}

void CollectiveProfiler::Harvest(DeviceState& state, bool block_oldest) {
  absl::MutexLock lock(state.mu);
  bool first = true;
  for (auto it = state.pending.begin(); it != state.pending.end();) {
    bool wait = block_oldest && first;
    first = false;
    if (!wait) {
      absl::StatusOr<bool> done = it->timer->IsDone();
      if (done.ok() && !*done) {
        ++it;
        continue;
      }
      if (!done.ok()) {
        LOG_FIRST_N(WARNING, 1)
            << "Collective timing: failed to query timer: " << done.status();
        it = state.pending.erase(it);
        continue;
      }
    }
    absl::StatusOr<absl::Duration> duration = it->timer->GetElapsedDuration();
    if (duration.ok()) {
      Accumulate(*it, *duration);
    } else {
      LOG_FIRST_N(WARNING, 1)
          << "Collective timing: failed to read timer: " << duration.status();
    }
    it = state.pending.erase(it);
  }
}

CollectiveProfiler::Aggregates CollectiveProfiler::Snapshot() {
  std::vector<DeviceState*> states;
  {
    absl::MutexLock lock(devices_mu_);
    for (auto& [ordinal, state] : devices_) states.push_back(&state);
  }
  for (DeviceState* state : states) Harvest(*state, /*block_oldest=*/false);
  absl::MutexLock lock(aggregates_mu_);
  return aggregates_;
}

std::string CollectiveProfiler::Format(const Aggregates& aggregates,
                                       int64_t skip_rounds, int num_devices,
                                       int64_t pending,
                                       int64_t command_buffer_records) {
  std::string out = absl::StrFormat(
      "\n===== XLA GPU collective timing (devices: %d, skipped first %d "
      "executable runs per device; times are summed over devices) =====\n",
      num_devices, skip_rounds);

  absl::Duration grand_total = absl::ZeroDuration();
  for (const auto& [name, buckets] : aggregates) {
    for (const auto& [bucket, stats] : buckets) grand_total += stats.total;
  }

  for (const auto& [name, buckets] : aggregates) {
    Stats total;
    for (const auto& [bucket, stats] : buckets) {
      total.count += stats.count;
      total.total += stats.total;
      total.min = std::min(total.min, stats.min);
      total.max = std::max(total.max, stats.max);
    }
    double share = grand_total > absl::ZeroDuration()
                       ? 100.0 * absl::FDivDuration(total.total, grand_total)
                       : 0.0;
    absl::StrAppendFormat(
        &out, "%s: calls=%d total=%.3f ms (%.1f%%) avg=%.1f us\n", name,
        total.count, ToMs(total.total), share,
        ToUs(total.total) / std::max<int64_t>(total.count, 1));
    for (const auto& [bucket, stats] : buckets) {
      absl::StrAppendFormat(
          &out,
          "  %-22s calls=%-8d total=%10.3f ms  avg=%9.1f us  min=%9.1f us  "
          "max=%9.1f us\n",
          FormatBucket(bucket), stats.count, ToMs(stats.total),
          ToUs(stats.total) / std::max<int64_t>(stats.count, 1),
          ToUs(stats.min), ToUs(stats.max));
    }
  }
  absl::StrAppendFormat(&out, "TOTAL: %.3f ms\n", ToMs(grand_total));
  if (pending > 0) {
    absl::StrAppendFormat(
        &out, "(%d collectives still in flight at exit were not counted)\n",
        pending);
  }
  if (command_buffer_records > 0) {
    absl::StrAppendFormat(
        &out,
        "(%d collectives were recorded into command buffers and are not "
        "timed; disable command buffers for collectives to time them)\n",
        command_buffer_records);
  }
  absl::StrAppend(&out, "=====\n");
  return out;
}

void CollectiveProfiler::Print() {
  Aggregates aggregates = Snapshot();
  int num_devices = 0;
  int64_t pending = 0;
  {
    absl::MutexLock lock(devices_mu_);
    num_devices = devices_.size();
    for (auto& [ordinal, state] : devices_) {
      absl::MutexLock state_lock(state.mu);
      pending += state.pending.size();
    }
  }
  std::string out =
      Format(aggregates, skip_rounds_, num_devices, pending,
             command_buffer_records_.load(std::memory_order_relaxed));
  std::fputs(out.c_str(), stderr);
  std::fflush(stderr);
}

}  // namespace xla::gpu
