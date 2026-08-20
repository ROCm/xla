/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_AUTOTUNER_PROFILER_H_
#define XLA_BACKENDS_AUTOTUNER_PROFILER_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/service/executable.h"
#include "xla/service/shaped_buffer.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

struct ProfileOptions {
  // Padding around the buffers to check for out-of-bounds reads/writes.
  int redzone_padding_bytes = 0;
  // Whether to initialize the buffers with random data or leave them
  // uninitialized.
  bool should_init_buffers = false;
  // Total size of the rotating input buffer pool. The profiler allocates as
  // many identical copies of the input set as fit in this budget and cycles
  // through them, so that data touched by one run is evicted from the last
  // level cache before the next run reads it. Zero disables rotation and
  // reuses a single set, which is the historical behaviour.
  int64_t rotating_buffer_bytes = 0;
  // Size of a scratch buffer the profiler streams a read over between a
  // candidate's warm-up run and its timed run, so that the timed run starts
  // from a cold cache. Should exceed the last level cache. Unlike
  // rotating_buffer_bytes this also evicts the output and scratch buffers, and
  // costs no extra allocation per instruction. Zero disables flushing.
  int64_t cache_flush_bytes = 0;
  // Number of timed runs per candidate. The reported duration is their median,
  // which is robust to a single clock or preemption outlier in a way the mean
  // is not. One reproduces the historical single-shot measurement.
  int num_timed_runs = 1;
};

struct ProfileResult {
  // The duration of the executable run.
  absl::Duration duration = absl::ZeroDuration();
  // The output buffer of the executable.
  std::optional<ScopedShapedBuffer> output_buffer = std::nullopt;
  // The scratch bytes used by the executable, if any.
  int scratch_bytes = 0;
};

struct InputBuffers {
  virtual ~InputBuffers() = default;
};

// Interface to run and profile XLA compiled executables for autotuning,
// also manages buffers for the executables.
class Profiler {
 public:
  virtual ~Profiler() = default;

  // Profiles a single executable.
  virtual absl::StatusOr<ProfileResult> Profile(
      std::unique_ptr<Executable> executable) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<InputBuffers> buffers,
                     CreateInputBuffers(executable.get()));
    return Profile(executable.get(), *buffers);
  }

  // Creates Input buffers for a given executable on the device. The buffers
  // are created with the same shape as the input parameters of the executable.
  // The optional instruction which was extracted to a module to create the
  // executable, can be provided to enable operation-specific buffer
  // initialization.
  virtual absl::StatusOr<std::unique_ptr<InputBuffers>> CreateInputBuffers(
      const Executable* executable, const HloInstruction* instr = nullptr) = 0;

  // Profiles a single executable with the provided buffers. The buffers
  // must be created by calling CreateInputBuffers from the same profiler.
  virtual absl::StatusOr<ProfileResult> Profile(
      Executable* executable, const InputBuffers& buffers) = 0;

  // Check for red-zone errors i.e out-of-bounds reads/writes to InputBuffers.
  // This is a no-op if redzone padding bytes are not set in the options.
  // The buffers are refreshed if needed, so that the buffers can be reused
  // for subsequent profiling.
  virtual absl::Status CheckInputBuffers(InputBuffers& buffers) = 0;

  // Compare the on device output buffer against the reference output buffer.
  // Returns ok if the buffers have the same content, otherwise returns an
  // error.
  virtual absl::Status CheckOutputBuffer(ScopedShapedBuffer& output,
                                         ScopedShapedBuffer& reference,
                                         float rtol) = 0;
};
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_PROFILER_H_
