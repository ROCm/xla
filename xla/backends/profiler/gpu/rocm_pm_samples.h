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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_PM_SAMPLES_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_PM_SAMPLES_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"

// Vendor-neutral container for time-sampled hardware-counter data and the code
// that writes it into an XPlane counter line. This is a deliberate duplicate of
// the CUDA-side SamplerRange/PmSamples in cupti_collector.h: that logic has no
// CUPTI dependency, but it lives in a cuda-only bazel target, and a ROCm target
// must not depend on it. Keep the counter-line output byte-for-byte compatible
// with the CUDA version so the same frontend renders both.

namespace xla {
namespace profiler {

// One time-sampled sweep of all requested metrics on one device.
struct RocmSamplerRange {
  size_t range_index;
  uint64_t start_timestamp_ns;
  uint64_t end_timestamp_ns;
  // One value per metric, in the same order as PmSamples::metrics_.
  std::vector<double> metric_values;
};

// Holds many sampler ranges plus the metric names, for one device.
class RocmPmSamples {
 public:
  RocmPmSamples(std::vector<std::string> metrics,
                std::vector<RocmSamplerRange> sampler_ranges, int device_id)
      : metrics_(std::move(metrics)),
        sampler_ranges_(std::move(sampler_ranges)),
        device_id_(device_id) {}

  // Writes each metric value as a zero-duration event on the plane's counter
  // line ("_counters_"), rebasing timestamps to start_gpu_time_ns. NaN values
  // are skipped and tallied.
  void PopulateCounterLine(tsl::profiler::XPlaneBuilder* plane,
                           uint64_t start_gpu_time_ns);

  size_t GetNumSamples() const;
  int64_t GetDeviceId() const;
  const std::vector<std::string>& GetMetrics() const;
  const std::vector<RocmSamplerRange>& GetSamplerRanges() const;

 private:
  std::vector<std::string> metrics_;
  std::vector<RocmSamplerRange> sampler_ranges_;
  int device_id_;
};

// Returns a human-readable display name for a ROCm counter, or the counter name
// unchanged if there is no mapping. Mirrors the CUDA GetGpuProfileMetricName.
std::string GetRocmProfileMetricName(absl::string_view metric_name);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_PM_SAMPLES_H_
