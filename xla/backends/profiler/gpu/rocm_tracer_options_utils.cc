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

#include "xla/backends/profiler/gpu/rocm_tracer_options_utils.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/profiler/utils/profiler_options_util.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace xla {
namespace profiler {

using tsl::profiler::SetValue;

absl::Status UpdateRocmTracerOptionsFromProfilerOptions(
    const tensorflow::ProfileOptions& profile_options,
    RocmTracerOptions& tracer_options) {
  // Only the PM-sampling keys are consumed here. Unlike the CUDA path we do not
  // error on unrecognized keys: the ROCm collector reads its sizing options
  // from debug flags, not advanced_configuration, so other keys may legitimately
  // be present.
  absl::flat_hash_set<absl::string_view> input_keys;
  for (const auto& [key, _] : profile_options.advanced_configuration()) {
    input_keys.insert(key);
  }

  RETURN_IF_ERROR(SetValue<std::string>(
      profile_options, "gpu_pm_sample_counters", input_keys,
      [&](const std::string& value) {
        std::vector<std::string> metrics;
        for (absl::string_view metric :
             absl::StrSplit(value, ',', absl::SkipEmpty())) {
          metrics.push_back(std::string(absl::StripAsciiWhitespace(metric)));
        }
        tracer_options.pm_sampler_options.metrics = metrics;
        tracer_options.pm_sampler_options.enable = !metrics.empty();
      }));

  RETURN_IF_ERROR(SetValue<int64_t>(
      profile_options, "gpu_pm_sample_interval_us", input_keys,
      [&](int64_t value) {
        tracer_options.pm_sampler_options.sample_interval_ns = value * 1000;
      }));

  // gpu_pm_sample_buffer_size_per_gpu_mb is accepted for CLI parity with the
  // CUDA path but has no hardware-buffer meaning on ROCm 7.2.4's synchronous
  // sampling path; it is parsed and ignored.
  RETURN_IF_ERROR(SetValue<int64_t>(
      profile_options, "gpu_pm_sample_buffer_size_per_gpu_mb", input_keys,
      [&](int64_t /*value*/) {}));

  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace xla
