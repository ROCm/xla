#ifndef XLA_BACKENDS_PROFILER_GPU_PROFILER_PLUGIN_INTERFACE_H_
#define XLA_BACKENDS_PROFILER_GPU_PROFILER_PLUGIN_INTERFACE_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

// Forward declaration
class RocmTraceCollector;

// Abstract plugin interface for extending profiler functionality
class ProfilerPlugin {
 public:
  virtual ~ProfilerPlugin() = default;

  // Called during profiler initialization
  virtual absl::Status Initialize() = 0;

  // Called when profiling starts
  virtual absl::Status OnProfilingStart(RocmTraceCollector* collector) = 0;

  // Called when profiling stops
  virtual absl::Status OnProfilingStop() = 0;

  // Called during trace export
  virtual absl::Status ExportData(tensorflow::profiler::XSpace* space) = 0;

  // Check if plugin is enabled
  virtual bool IsEnabled() const = 0;
};

// Plugin registry (singleton)
class ProfilerPluginRegistry {
 public:
  static ProfilerPluginRegistry& Get();

  void RegisterPlugin(std::unique_ptr<ProfilerPlugin> plugin);
  void InitializePlugins();
  void OnProfilingStart(RocmTraceCollector* collector);
  void OnProfilingStop();
  void ExportPluginData(tensorflow::profiler::XSpace* space);

 private:
  std::vector<std::unique_ptr<ProfilerPlugin>> plugins_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_PROFILER_PLUGIN_INTERFACE_H_

