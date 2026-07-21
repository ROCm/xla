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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_PM_SAMPLER_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_PM_SAMPLER_H_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/backends/profiler/gpu/rocm_pm_samples.h"

// Minimal interface for ROCm PM (hardware-counter) sampling, kept free of any
// rocprofiler-sdk types so it compiles without vendor headers. The concrete
// implementation lives in rocm_pm_sampler_impl.h.

namespace xla {
namespace profiler {

// Configuration for the ROCm PM sampler.
//
// Unlike the CUDA CuptiPmSamplerOptions this has no hw_buf_size (ROCm 7.2.4 has
// no hardware ring buffer on the synchronous sampling path -- the sample call
// returns records directly) and no devs_per_decode_thd (there is no separate
// decode thread; each device drives its own sampling loop -- see the impl).
struct RocmPmSamplerOptions {
  // Whether to enable PM sampling.
  bool enable = false;
  // Counter names to collect (must match rocprofiler agent counter names).
  std::vector<std::string> metrics{};
  // Host sampling period. 500,000ns = 2kHz, matching the CUDA default.
  size_t sample_interval_ns = 500'000;
  // How often the per-device thread hands a batch of samples to
  // process_samples. Larger batches amortize the callback + XPlane locking.
  absl::Duration flush_period = absl::Milliseconds(100);
  // Reserve size for the per-batch sampler-range vector.
  size_t max_samples = 500;
  // What to do with samples once gathered.
  // Note: must be thread-safe - may be called by multiple device threads
  // simultaneously (with different samples data per thread).
  std::function<void(RocmPmSamples* samples)> process_samples;
};

class RocmPmSampler {
 public:
  RocmPmSampler() = default;

  // Not copyable or movable.
  RocmPmSampler(const RocmPmSampler&) = delete;
  RocmPmSampler(RocmPmSampler&&) = delete;
  RocmPmSampler& operator=(const RocmPmSampler&) = delete;
  RocmPmSampler& operator=(RocmPmSampler&&) = delete;

  virtual ~RocmPmSampler() = default;

  // Start sampling (starts the counting context and per-device threads).
  virtual absl::Status StartSampler() = 0;

  // Stop sampling (pauses per-device threads, drains a final batch).
  virtual absl::Status StopSampler() = 0;

  // Deinitialize the PM sampler (stops the context, joins threads, frees).
  virtual absl::Status Deinitialize() = 0;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_PM_SAMPLER_H_
