#include "xla/backends/profiler/gpu/profiler_plugin_interface.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace profiler {

ProfilerPluginRegistry& ProfilerPluginRegistry::Get() {
  static ProfilerPluginRegistry* registry = new ProfilerPluginRegistry();
  return *registry;
}

void ProfilerPluginRegistry::RegisterPlugin(std::unique_ptr<ProfilerPlugin> plugin) {
  plugins_.push_back(std::move(plugin));
}

void ProfilerPluginRegistry::InitializePlugins() {
  for (const auto& plugin : plugins_) {
    if (plugin->IsEnabled()) {
      absl::Status status = plugin->Initialize();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to initialize profiler plugin: " << status;
      }
    }
  }
}

void ProfilerPluginRegistry::OnProfilingStart(RocmTraceCollector* collector) {
  for (const auto& plugin : plugins_) {
    if (plugin->IsEnabled()) {
      absl::Status status = plugin->OnProfilingStart(collector);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to start profiler plugin: " << status;
      }
    }
  }
}

void ProfilerPluginRegistry::OnProfilingStop() {
  for (const auto& plugin : plugins_) {
    if (plugin->IsEnabled()) {
      absl::Status status = plugin->OnProfilingStop();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to stop profiler plugin: " << status;
      }
    }
  }
}

void ProfilerPluginRegistry::ExportPluginData(
    tensorflow::profiler::XSpace* space) {
  for (const auto& plugin : plugins_) {
    if (plugin->IsEnabled()) {
      absl::Status status = plugin->ExportData(space);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to export profiler plugin data: " << status;
      }
    }
  }
}

}  // namespace profiler
}  // namespace xla

