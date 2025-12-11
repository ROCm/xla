# Distributed Profiling Refactoring Plan

## Goal
Minimize in-place modifications to original XLA files by using:
- **Plugin architecture** for injecting distributed profiling functionality
- **Extension/wrapper classes** to avoid modifying core classes
- **Configuration-based enablement** via environment variables or flags
- **Dependency injection** instead of tight coupling

---

## Current State Analysis

### Modified Original XLA Files (13 files)
1. **`xla/pjrt/gpu/se_gpu_pjrt_client.cc`** (+358 lines)
   - Added graph generation, port assignment functions
   - Direct insertion of distributed profiling logic
   
2. **`xla/backends/profiler/gpu/rocm_collector.cc`** (+97 lines)
   - Added distributed sync initialization
   - Added snapshot thread management
   - Added probe data export
   
3. **`xla/backends/profiler/gpu/rocm_tracer_v1.cc`** (~50 lines modified)
   - Field name changes (block_x → workgroup_x, etc.), this is a good fix. 

   
4. **`xla/backends/profiler/gpu/device_tracer_rocm.cc`** (+36 lines)
   - Added ProfileOptions parameter
   - Added snapshot period parsing
   
5. **`xla/backends/profiler/gpu/rocm_profiler_sdk.cc`** (need to check)
   
6. **`xla/backends/profiler/gpu/rocm_tracer_utils.h`** (need to check)
   
7. **`xla/tsl/distributed_runtime/coordination/coordination_service_agent.cc`** (+106 lines)
   - Added IP address discovery and KV store insertion
   - **CONCERN**: Direct modification of coordination service
   
8. **`xla/pjrt/distributed/client.cc`** (+7 lines)
   - Minor changes (headers, commented fields)
   
9. **`third_party/tsl/tsl/profiler/protobuf/xplane.proto`** (need to check)
   
10. **`WORKSPACE`** (build dependencies)
   
11. **`xla/pjrt/gpu/BUILD`** (build target changes)
   
12. **`xla/backends/profiler/gpu/BUILD`** (build target changes)

### New Files Added (21+ files)
All distributed profiling implementation:
- `distributed_timestamp_sync.h/cc`
- `network_probe.h/cc`
- `svm_wrapper.h/cc`
- `probe_utils.h/cc`
- Test files, tools, documentation

---

## Refactoring Strategy

### Phase 1: Extract Distributed Profiling as a Plugin

#### 1.1 Create Plugin Interface
**New file**: `xla/backends/profiler/gpu/profiler_plugin_interface.h`

```cpp
namespace xla {
namespace profiler {

// Abstract plugin interface for extending profiler functionality
class ProfilerPlugin {
 public:
  virtual ~ProfilerPlugin() = default;
  
  // Called during profiler initialization
  virtual absl::Status Initialize(const ProfileOptions& options) = 0;
  
  // Called when profiling starts
  virtual absl::Status OnProfilingStart(RocmTraceCollector* collector) = 0;
  
  // Called when profiling stops
  virtual absl::Status OnProfilingStop() = 0;
  
  // Called during trace export
  virtual absl::Status ExportData(XSpace* space) = 0;
  
  // Check if plugin is enabled
  virtual bool IsEnabled() const = 0;
};

// Plugin registry (singleton)
class ProfilerPluginRegistry {
 public:
  static ProfilerPluginRegistry& Get();
  
  void RegisterPlugin(std::unique_ptr<ProfilerPlugin> plugin);
  void InitializePlugins(const ProfileOptions& options);
  void OnProfilingStart(RocmTraceCollector* collector);
  void OnProfilingStop();
  void ExportPluginData(XSpace* space);
  
 private:
  std::vector<std::unique_ptr<ProfilerPlugin>> plugins_;
};

}  // namespace profiler
}  // namespace xla
```

#### 1.2 Implement Distributed Profiling Plugin
**New file**: `xla/backends/profiler/gpu/distributed_profiler_plugin.h/cc`

Move all distributed profiling logic into this plugin:
- `InitializeDistributedSync()` → `Initialize()`
- `StartProbing()` → `OnProfilingStart()`
- `ExportProbeData()` → `ExportData()`
- Snapshot thread management

#### 1.3 Modify `rocm_collector.cc` to Use Plugin System

**Changes to `rocm_collector.cc`**:
```cpp
// BEFORE (current):
absl::Status RocmTraceCollectorImpl::InitializeDistributedSync() {
  auto dist_ctx_opt = DistributedProfilerContextManager::Get().GetDistributedContext();
  // ... 30+ lines of distributed profiling code
}

// AFTER (refactored):
absl::Status RocmTraceCollectorImpl::InitializePlugins() {
  ProfilerPluginRegistry::Get().InitializePlugins(options_);
  ProfilerPluginRegistry::Get().OnProfilingStart(this);
  return absl::OkStatus();
}

void RocmTraceCollectorImpl::Export(XSpace* space) {
  // ... existing export code ...
  
  // Export plugin data
  ProfilerPluginRegistry::Get().ExportPluginData(space);
  
  // ... rest of export code ...
}
```

**Impact**: Reduces `rocm_collector.cc` modifications from +97 lines to ~10 lines

---

### Phase 2: Extract PJRT Client Extensions

#### 2.1 Create Distributed Context Setup Utility
**New file**: `xla/pjrt/gpu/distributed_context_setup.h/cc`

Move all graph generation and port assignment functions:
- `ExchangeNodeAddresses()`
- `GenerateDirectedNeighbors()`
- Port assignment logic

#### 2.2 Use Wrapper/Extension Pattern for PJRT Client

**Option A: Factory Wrapper**
```cpp
// New file: xla/pjrt/gpu/se_gpu_pjrt_client_distributed.h

namespace xla {

// Wrapper that extends PJRT client initialization with distributed profiling
class DistributedGpuClientFactory {
 public:
  static absl::StatusOr<std::unique_ptr<PjRtClient>> CreateWithDistributedProfiling(
      GpuClientOptions options,
      KeyValueStoreInterface* kv_store);
  
 private:
  static absl::Status SetupDistributedContext(
      const GpuClientOptions& options,
      KeyValueStoreInterface* kv_store);
};

}  // namespace xla
```

**Option B: Initialization Hook**
```cpp
// Add to existing se_gpu_pjrt_client.cc (minimal change):

// At the end of GetStreamExecutorGpuClient():
#ifdef TENSORFLOW_USE_ROCM
  if (auto* dist_setup = GetDistributedProfilingSetup()) {
    TF_RETURN_IF_ERROR(dist_setup->Initialize(node_id, num_nodes, kv_store));
  }
#endif
```

**Impact**: Reduces `se_gpu_pjrt_client.cc` modifications from +358 lines to ~20 lines

---

### Phase 3: Extract Coordination Service Extensions

#### 3.1 Create Network Configuration Plugin
**New file**: `xla/tsl/distributed_runtime/coordination/network_config_plugin.h/cc`

Move IP address discovery logic into a separate utility:
```cpp
namespace tensorflow {

class NetworkConfigPlugin {
 public:
  // Discover local IP address and register in KV store
  static absl::Status RegisterNodeAddress(
      CoordinationServiceAgent* agent,
      int task_id);
  
 private:
  static std::string GetLocalIP(const std::string& interface);
  static std::string GetHostname();
};

}  // namespace tensorflow
```

#### 3.2 Modify `coordination_service_agent.cc` Minimally
```cpp
// Add single hook at end of Connect():
absl::Status CoordinationServiceAgent::Connect() {
  // ... existing connect logic ...
  
  LOG(INFO) << "Coordination agent has successfully connected.";
  
  // Plugin hook for extensions
  #ifdef XLA_ENABLE_DISTRIBUTED_PROFILING
    TF_RETURN_IF_ERROR(NetworkConfigPlugin::RegisterNodeAddress(this, task_.task_id()));
  #endif
  
  // ... rest of existing code ...
}
```

**Impact**: Reduces `coordination_service_agent.cc` modifications from +106 lines to ~5 lines

---

### Phase 4: Handle Field Name Changes

#### 4.1 Assess `rocm_tracer_v1.cc` Changes

**Current changes**:
- `block_x/y/z` → `workgroup_x/y/z`
- `dynamic_shared_memory_usage` → `group_segment_size`
Keep these changes (separate from refactoring)

---

### Phase 5: Configuration-Based Enablement

#### 5.1 Runtime Configuration (Hybrid Approach)

**Design**: Support both individual env vars and JSON config file, with precedence: env vars > config file > defaults.

##### 5.1.1 Configuration Structure
```cpp
// New file: xla/backends/profiler/gpu/distributed_profiler_config.h
struct DistributedProfilerConfig {
  bool enabled = false;
  int probe_cadence_us = 800;
  int probe_window_s = 4;
  int packet_spacing_us = 100;
  int snapshot_period_ms = 100;
  std::string output_dir = "/tmp/xla_dist_prof";
  
  // Load from environment and config file
  static DistributedProfilerConfig Load();
  
 private:
  static DistributedProfilerConfig LoadFromEnvVars();
  static absl::StatusOr<DistributedProfilerConfig> LoadFromFile(
      const std::string& path);
};
```

##### 5.1.2 Configuration Loading Logic
```cpp
// Implementation in distributed_profiler_config.cc
DistributedProfilerConfig DistributedProfilerConfig::Load() {
  DistributedProfilerConfig config;
  
  // Step 1: Load defaults (already set in struct)
  
  // Step 2: Load from config file if specified
  char* config_path = std::getenv("XLA_DIST_PROF_CONFIG");
  if (config_path != nullptr) {
    auto file_config = LoadFromFile(config_path);
    if (file_config.ok()) {
      config = *file_config;
      VLOG(1) << "Loaded distributed profiling config from: " << config_path;
    } else {
      LOG(WARNING) << "Failed to load config from " << config_path 
                   << ": " << file_config.status();
    }
  }
  
  // Step 3: Override with individual env vars (highest precedence)
  bool enabled;
  if (tsl::ReadBoolFromEnvVar("XLA_ENABLE_DISTRIBUTED_PROFILING", 
                               config.enabled, &enabled).ok()) {
    config.enabled = enabled;
  }
  
  int64_t cadence;
  if (tsl::ReadInt64FromEnvVar("XLA_PROBE_CADENCE_US", 
                                config.probe_cadence_us, &cadence).ok()) {
    config.probe_cadence_us = cadence;
  }
  
  int64_t window;
  if (tsl::ReadInt64FromEnvVar("XLA_PROBE_WINDOW_S", 
                                config.probe_window_s, &window).ok()) {
    config.probe_window_s = window;
  }
  
  int64_t spacing;
  if (tsl::ReadInt64FromEnvVar("XLA_PACKET_SPACING_US", 
                                config.packet_spacing_us, &spacing).ok()) {
    config.packet_spacing_us = spacing;
  }
  
  std::string output_dir;
  if (tsl::ReadStringFromEnvVar("XLA_DIST_PROF_OUTPUT_DIR", 
                                 config.output_dir, &output_dir).ok()) {
    config.output_dir = output_dir;
  }
  
  return config;
}

absl::StatusOr<DistributedProfilerConfig> 
DistributedProfilerConfig::LoadFromFile(const std::string& path) {
  // Read file
  std::string contents;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), path, &contents));
  
  // Parse JSON (using nlohmann/json or absl's JSON parser)
  // Note: XLA already has JSON dependencies, no need for YAML
  auto json = nlohmann::json::parse(contents);
  
  DistributedProfilerConfig config;
  if (json.contains("enabled")) config.enabled = json["enabled"];
  if (json.contains("probe_cadence_us")) config.probe_cadence_us = json["probe_cadence_us"];
  if (json.contains("probe_window_s")) config.probe_window_s = json["probe_window_s"];
  if (json.contains("packet_spacing_us")) config.packet_spacing_us = json["packet_spacing_us"];
  if (json.contains("snapshot_period_ms")) config.snapshot_period_ms = json["snapshot_period_ms"];
  if (json.contains("output_dir")) config.output_dir = json["output_dir"];
  
  return config;
}
```

##### 5.1.3 Usage Examples

**Simple case** (just enable with defaults):
```bash
XLA_ENABLE_DISTRIBUTED_PROFILING=1 python train.py
```

**Override specific parameters**:
```bash
XLA_ENABLE_DISTRIBUTED_PROFILING=1 \
XLA_PROBE_CADENCE_US=1000 \
XLA_PROBE_WINDOW_S=8 \
python train.py
```

**Use config file**:
```bash
# Create config.json:
cat > dist_prof_config.json <<EOF
{
  "enabled": true,
  "probe_cadence_us": 800,
  "probe_window_s": 4,
  "packet_spacing_us": 100,
  "snapshot_period_ms": 100,
  "output_dir": "/tmp/my_prof_data"
}
EOF

# Run with config:
XLA_DIST_PROF_CONFIG=dist_prof_config.json python train.py
```

**Config file with overrides**:
```bash
# Use config but override cadence for this run
XLA_DIST_PROF_CONFIG=dist_prof_config.json \
XLA_PROBE_CADENCE_US=2000 \
python train.py
```

##### 5.1.4 Integration with Plugin
```cpp
// In DistributedProfilerPlugin::Initialize()
absl::Status DistributedProfilerPlugin::Initialize(const ProfileOptions& options) {
  config_ = DistributedProfilerConfig::Load();
  
  if (!config_.enabled) {
    VLOG(1) << "Distributed profiling disabled";
    return absl::OkStatus();
  }
  
  LOG(INFO) << "Distributed profiling enabled with config:";
  LOG(INFO) << "  probe_cadence_us: " << config_.probe_cadence_us;
  LOG(INFO) << "  probe_window_s: " << config_.probe_window_s;
  LOG(INFO) << "  packet_spacing_us: " << config_.packet_spacing_us;
  LOG(INFO) << "  output_dir: " << config_.output_dir;
  
  // Use config values in initialization...
  // ...
}

bool DistributedProfilerPlugin::IsEnabled() const {
  return config_.enabled;
}
```

#### 5.2 Build-Time Flags (Optional)

For users who want to completely exclude distributed profiling code:

```python
# In BUILD file:
config_setting(
    name = "enable_distributed_profiling",
    define_values = {"enable_distributed_profiling": "true"},
)

cc_library(
    name = "distributed_profiler_plugin",
    srcs = ["distributed_profiler_plugin.cc"],
    deps = select({
        ":enable_distributed_profiling": [
            ":network_probe",
            ":distributed_timestamp_sync",
            # ... other deps
        ],
        "//conditions:default": [],
    }),
)
```

**Note**: Most users should use runtime config (5.1), not build flags.

---

## Refactoring Summary

### Before Refactoring
| File | Lines Modified | Type |
|------|---------------|------|
| `se_gpu_pjrt_client.cc` | +358 | In-place addition |
| `rocm_collector.cc` | +97 | In-place addition |
| `coordination_service_agent.cc` | +106 | In-place addition |
| `device_tracer_rocm.cc` | +36 | In-place addition |
| `rocm_tracer_v1.cc` | ~50 | In-place modification |
| **Total** | **~647 lines** | **5 core files modified** |

### After Refactoring
| File | Lines Modified | Type |
|------|---------------|------|
| `se_gpu_pjrt_client.cc` | +20 | Hook insertion |
| `rocm_collector.cc` | +15 | Plugin system integration |
| `coordination_service_agent.cc` | +5 | Plugin hook |
| `device_tracer_rocm.cc` | +10 | Plugin system integration |
| `rocm_tracer_v1.cc` | 0 (or keep if correctness fix) | - |
| **Total** | **~50 lines** | **4 core files modified** |

**New plugin files**:
- `profiler_plugin_interface.h/cc` (~150 lines)
- `distributed_profiler_plugin.h/cc` (~300 lines, moved from rocm_collector.cc)
- `distributed_context_setup.h/cc` (~400 lines, moved from se_gpu_pjrt_client.cc)
- `network_config_plugin.h/cc` (~150 lines, moved from coordination_service_agent.cc)

---

## Implementation Plan

### Step 1: Create Plugin Infrastructure (Week 1)
- [ ] Implement `ProfilerPluginInterface`
- [ ] Implement `ProfilerPluginRegistry`
- [ ] Add plugin hooks to `rocm_collector.cc`
- [ ] Test plugin system with empty plugin

### Step 2: Extract Distributed Profiler (Week 2)
- [ ] Create `DistributedProfilerPlugin`
- [ ] Move initialization logic from `rocm_collector.cc`
- [ ] Move export logic
- [ ] Test distributed profiling still works

### Step 3: Extract PJRT Extensions (Week 2)
- [ ] Create `DistributedContextSetup` utility
- [ ] Move graph generation from `se_gpu_pjrt_client.cc`
- [ ] Add initialization hook
- [ ] Test multi-node setup

### Step 4: Extract Coordination Extensions (Week 3)
- [ ] Create `NetworkConfigPlugin`
- [ ] Move IP discovery from `coordination_service_agent.cc`
- [ ] Add coordination hook
- [ ] Test node address exchange

### Step 5: Add Configuration Controls (Week 3)
- [ ] Add environment variable checks
- [ ] Add build flags
- [ ] Test enable/disable functionality
- [ ] Document configuration options

### Step 6: Testing & Validation (Week 4)
- [ ] Run all existing tests
- [ ] Test with distributed profiling enabled
- [ ] Test with distributed profiling disabled
- [ ] Performance benchmarking
- [ ] Update documentation

---

## Benefits

### Maintainability
- **Separation of concerns**: Distributed profiling isolated from core XLA
- **Easier updates**: Can update XLA without conflicts
- **Clear boundaries**: Plugin API defines integration points

### Testing
- **Unit testable**: Plugins can be tested independently
- **Integration testable**: Can test with/without plugins
- **Gradual rollout**: Enable for specific workloads

### Upstream Compatibility
- **Minimal diffs**: Only ~50 lines changed in core files vs ~647
- **Plugin architecture**: Can be proposed upstream as general solution
- **Clean separation**: Easier to rebase on upstream changes

### Code Quality
- **Single Responsibility**: Each component has one job
- **Open/Closed Principle**: Extend without modifying
- **Dependency Inversion**: Core depends on interfaces, not implementations

---

## Risks & Mitigations

### Risk 1: Performance Overhead
**Risk**: Plugin indirection adds overhead  
**Mitigation**: 
- Use static initialization (no runtime lookup)
- Inline plugin calls in release builds
- Benchmark before/after

### Risk 2: Breaking Existing Functionality
**Risk**: Refactoring breaks distributed profiling  
**Mitigation**:
- Incremental refactoring with tests at each step
- Keep old code until new code validated
- Extensive testing on multi-node clusters

### Risk 3: Incomplete Abstraction
**Risk**: Plugin interface doesn't support all needs  
**Mitigation**:
- Start with current requirements
- Design for extensibility
- Iterate on interface as needed

---

## Alternative Approaches

### Alternative 1: Conditional Compilation Only
Use `#ifdef XLA_ENABLE_DISTRIBUTED_PROFILING` everywhere
- **Pro**: Simplest approach
- **Con**: Still modifies original files extensively
- **Con**: Hard to maintain both paths

### Alternative 2: Fork & Vendor
Maintain separate fork of XLA
- **Pro**: No restrictions on changes
- **Con**: Hard to keep up with upstream
- **Con**: Duplicates maintenance effort

### Alternative 3: Dynamic Library Plugin
Load distributed profiling as separate `.so`
- **Pro**: Completely separate build
- **Con**: Complex loading mechanism
- **Con**: Version compatibility issues

**Recommendation**: Use plugin interface approach (original plan)

---

## Next Steps

1. **Review this plan** with team
2. **Prioritize** which files to refactor first
3. **Create tracking issues** for each phase
4. **Start with Phase 1** (plugin infrastructure)
5. **Iterate** based on learnings

---

## Questions for Discussion

2. Should we target upstream contribution of the plugin system, or keep it as an internal refactoring?

3. What's the priority: minimize changed lines or maximize code clarity?

4. Are there other XLA extension points we should leverage (e.g., existing plugin systems)?

5. Should we refactor incrementally (ship after each phase) or all at once?

---

## References

- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Current architecture
- [DIRECTED_PROBE_SPEC.md](xla/backends/profiler/gpu/context/DIRECTED_PROBE_SPEC.md) - Probe protocol
- [OPTION_B_IMPLEMENTATION.md](xla/backends/profiler/gpu/context/OPTION_B_IMPLEMENTATION.md) - Socket architecture

---

**Document Status**: Draft for review  
**Last Updated**: 2025-12-08  
**Author**: Refactoring Analysis

