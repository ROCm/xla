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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PROFILER_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PROFILER_H_

#include <atomic>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// Process-wide profiler that accumulates device-side execution time of
// collective thunks per collective kind and per power-of-two message size
// bucket, and prints a summary table to stderr at process exit.
//
// Enabled by the environment variable XLA_GPU_COLLECTIVE_TIMING=<N>, where N is
// the number of GPU executable runs (per device) to skip before recording
// starts, e.g. to exclude warmup and autotuning steps.
//
// Timing is done with GPU events recorded around the collective on the stream
// it runs on, so it includes time spent waiting for peers. Events are
// harvested without blocking the host at the start of each executable run.
// Collectives recorded into command buffers are not timed.
class CollectiveProfiler {
 public:
  struct Stats {
    int64_t count = 0;
    absl::Duration total = absl::ZeroDuration();
    absl::Duration min = absl::InfiniteDuration();
    absl::Duration max = absl::ZeroDuration();

    void Add(absl::Duration d);
  };

  // Collective name -> size bucket (-1 for zero bytes, otherwise k for sizes
  // in [2^k, 2^(k+1)) bytes) -> stats.
  using Aggregates = std::map<std::string, std::map<int, Stats>>;

  // Returns the process-wide profiler, or nullptr if profiling is disabled.
  static CollectiveProfiler* Get();

  explicit CollectiveProfiler(int64_t skip_rounds);

  // Marks the start of a GPU executable run on `device_ordinal` and harvests
  // completed timers without blocking.
  void BeginRound(int device_ordinal);

  // Returns true if collectives on `device_ordinal` should be timed now.
  bool IsActive(int device_ordinal);

  // Times `fn` enqueuing a collective on `stream`. Records into the bucket
  // `name` / `bytes`. If the profiler is inactive or timer creation fails,
  // simply runs `fn`.
  absl::Status Time(stream_executor::Stream& stream, std::string name, int64_t bytes,
                    absl::FunctionRef<absl::Status()> fn);

  // Called when a collective is recorded into a command buffer.
  void NoteCommandBufferRecord() {
    command_buffer_records_.fetch_add(1, std::memory_order_relaxed);
  }

  // Harvests completed timers (non-blocking) and returns the aggregates.
  Aggregates Snapshot();

  // Renders aggregates as a human-readable table.
  static std::string Format(const Aggregates& aggregates, int64_t skip_rounds,
                            int num_devices, int64_t pending,
                            int64_t command_buffer_records);

  // Harvests completed timers and prints the table to stderr.
  void Print();

  static int SizeBucket(int64_t bytes);

  // Scoped marker that disables timing of nested collectives on the current
  // thread, e.g. inside a collective group where collectives are only
  // launched at group end.
  class NestedScope {
   public:
    NestedScope();
    ~NestedScope();
    NestedScope(const NestedScope&) = delete;
    NestedScope& operator=(const NestedScope&) = delete;

   private:
    bool prev_;
  };
  static bool InNestedScope();

 private:
  struct Pending {
    std::unique_ptr<stream_executor::EventBasedTimer> timer;
    std::string name;
    int64_t bytes;
  };

  struct DeviceState {
    std::atomic<int64_t> rounds{0};
    absl::Mutex mu;
    std::list<Pending> pending ABSL_GUARDED_BY(mu);
  };

  DeviceState& GetDeviceState(int device_ordinal);

  // Moves finished timers from `state` into aggregates. If `block_oldest` is
  // set, waits for the oldest pending timer first.
  void Harvest(DeviceState& state, bool block_oldest);

  void Accumulate(const Pending& pending, absl::Duration duration);

  const int64_t skip_rounds_;
  std::atomic<int64_t> command_buffer_records_{0};

  absl::Mutex devices_mu_;
  absl::node_hash_map<int, DeviceState> devices_ ABSL_GUARDED_BY(devices_mu_);

  absl::Mutex aggregates_mu_;
  Aggregates aggregates_ ABSL_GUARDED_BY(aggregates_mu_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PROFILER_H_
