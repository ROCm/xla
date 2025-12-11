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

#ifndef XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_PROFILER_PLUGIN_H_
#define XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_PROFILER_PLUGIN_H_

#include "xla/backends/profiler/gpu/profiler_plugin_interface.h"
#include "xla/backends/profiler/gpu/distributed_profiler_config.h"

namespace xla {
namespace profiler {

// Plugin implementation for distributed profiling functionality
// This plugin handles distributed timestamp synchronization, network probing,
// and probe data export.
class DistributedProfilerPlugin : public ProfilerPlugin {
 public:
  DistributedProfilerPlugin() = default;
  ~DistributedProfilerPlugin() override = default;

  absl::Status Initialize() override;
  absl::Status OnProfilingStart(RocmTraceCollector* collector) override;
  absl::Status OnProfilingStop() override;
  absl::Status ExportData(tensorflow::profiler::XSpace* space) override;
  bool IsEnabled() const override;

 private:
  DistributedProfilerConfig config_;
  bool enabled_ = false;
};

// Static registration helper (call this from a static initializer)
void RegisterDistributedProfilerPlugin();

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_DISTRIBUTED_PROFILER_PLUGIN_H_

