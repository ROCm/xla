/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_PROFILER_CONFIG_H_
#define XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_PROFILER_CONFIG_H_

#include <string>
#include "absl/status/statusor.h"

namespace xla {
namespace profiler {

// Configuration structure for distributed profiling
struct DistributedProfilerConfig {
  bool enabled = false;
  int probe_cadence_us = 800;
  int probe_window_s = 4;
  int packet_spacing_us = 100;
  int snapshot_period_ms = 100;
  std::string output_dir = "/tmp/xla_dist_prof";
  
  // Load configuration with precedence: env vars > config file > defaults
  static DistributedProfilerConfig Load();
  
 private:
  static DistributedProfilerConfig LoadFromEnvVars();
  static absl::StatusOr<DistributedProfilerConfig> LoadFromFile(
      const std::string& path);
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_PROFILER_CONFIG_H_

