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

#include "xla/backends/profiler/gpu/distributed_profiler_plugin.h"

#include "tsl/platform/logging.h"

namespace xla {
namespace profiler {

absl::Status DistributedProfilerPlugin::Initialize() {
  config_ = DistributedProfilerConfig::Load();
  enabled_ = config_.enabled;
  
  if (!enabled_) {
    VLOG(1) << "Distributed profiling disabled";
    return absl::OkStatus();
  }
  
  LOG(INFO) << "Distributed profiling plugin initialized with config:";
  LOG(INFO) << "  probe_cadence_us: " << config_.probe_cadence_us;
  LOG(INFO) << "  probe_window_s: " << config_.probe_window_s;
  LOG(INFO) << "  packet_spacing_us: " << config_.packet_spacing_us;
  LOG(INFO) << "  snapshot_period_ms: " << config_.snapshot_period_ms;
  LOG(INFO) << "  output_dir: " << config_.output_dir;
  
  return absl::OkStatus();
}

absl::Status DistributedProfilerPlugin::OnProfilingStart(RocmTraceCollector* collector) {
  if (!enabled_) {
    return absl::OkStatus();
  }
  
  VLOG(1) << "DistributedProfilerPlugin::OnProfilingStart";
  // Future: additional logic can be added here for distributed profiling
  
  return absl::OkStatus();
}

absl::Status DistributedProfilerPlugin::OnProfilingStop() {
  if (!enabled_) {
    return absl::OkStatus();
  }
  
  VLOG(1) << "DistributedProfilerPlugin::OnProfilingStop";
  // Future: cleanup logic can be added here
  
  return absl::OkStatus();
}

absl::Status DistributedProfilerPlugin::ExportData(
    tensorflow::profiler::XSpace* space) {
  if (!enabled_) {
    return absl::OkStatus();
  }
  
  VLOG(1) << "DistributedProfilerPlugin::ExportData";
  // Future: add distributed profiling data to the XSpace
  // For now, the distributed timestamp sync is still handled by rocm_collector
  // This plugin serves as a hook point for future extensions
  
  return absl::OkStatus();
}

bool DistributedProfilerPlugin::IsEnabled() const {
  return enabled_;
}

// Static registration
void RegisterDistributedProfilerPlugin() {
  ProfilerPluginRegistry::Get().RegisterPlugin(
      std::make_unique<DistributedProfilerPlugin>());
}

// Auto-registration at static initialization time
namespace {
struct PluginRegistrar {
  PluginRegistrar() {
    RegisterDistributedProfilerPlugin();
  }
};
static PluginRegistrar registrar;
}  // namespace

}  // namespace profiler
}  // namespace xla

