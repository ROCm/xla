#include "xla/backends/profiler/gpu/profiler_plugin_interface.h"

#include <memory>
#include <utility>

#include "tsl/platform/test.h"
#include "absl/status/status.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

// Simple manual mock
class TestProfilerPlugin : public ProfilerPlugin {
 public:
  bool initialized = false;
  bool started = false;
  bool stopped = false;
  bool exported = false;
  bool enabled = true;

  absl::Status Initialize() override {
    initialized = true;
    return absl::OkStatus();
  }
  
  absl::Status OnProfilingStart(RocmTraceCollector* collector) override {
    started = true;
    return absl::OkStatus();
  }
  
  absl::Status OnProfilingStop() override {
    stopped = true;
    return absl::OkStatus();
  }
  
  absl::Status ExportData(tensorflow::profiler::XSpace* space) override {
    exported = true;
    return absl::OkStatus();
  }
  
  bool IsEnabled() const override {
    return enabled;
  }
};

// We need a way to reset the singleton for testing, or just register new plugins and ignore old ones.
// The current implementation accumulates plugins. For unit tests, this is suboptimal if tests run in same process.
// But bazel tests usually run in isolation.
// Let's assume isolation for now.

TEST(ProfilerPluginTest, Lifecycle) {
  auto plugin_ptr = std::make_unique<TestProfilerPlugin>();
  TestProfilerPlugin* plugin = plugin_ptr.get();
  
  ProfilerPluginRegistry::Get().RegisterPlugin(std::move(plugin_ptr));
  
  ProfilerPluginRegistry::Get().InitializePlugins();
  EXPECT_TRUE(plugin->initialized);
  
  ProfilerPluginRegistry::Get().OnProfilingStart(nullptr);
  EXPECT_TRUE(plugin->started);
  
  ProfilerPluginRegistry::Get().OnProfilingStop();
  EXPECT_TRUE(plugin->stopped);
  
  tensorflow::profiler::XSpace space;
  ProfilerPluginRegistry::Get().ExportPluginData(&space);
  EXPECT_TRUE(plugin->exported);
}

TEST(ProfilerPluginTest, DisabledPlugin) {
  auto plugin_ptr = std::make_unique<TestProfilerPlugin>();
  plugin_ptr->enabled = false;
  TestProfilerPlugin* plugin = plugin_ptr.get();
  
  ProfilerPluginRegistry::Get().RegisterPlugin(std::move(plugin_ptr));
  
  ProfilerPluginRegistry::Get().InitializePlugins();
  EXPECT_FALSE(plugin->initialized);
}

}  // namespace
}  // namespace profiler
}  // namespace xla

