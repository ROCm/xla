# Refactoring Architecture: Before & After

## Current Architecture (Before Refactoring)

```
┌─────────────────────────────────────────────────────────────────┐
│                        XLA Core Files                            │
│                    (HEAVILY MODIFIED)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ se_gpu_pjrt      │  │ rocm_collector   │  │ coordination     │
│ _client.cc       │  │ .cc              │  │ _service_agent   │
│                  │  │                  │  │ .cc              │
│ +358 lines       │  │ +97 lines        │  │ +106 lines       │
│ ─────────────    │  │ ─────────────    │  │ ─────────────    │
│ • Graph gen      │  │ • Dist sync      │  │ • IP discovery   │
│ • Port assign    │  │ • Snapshot       │  │ • KV store       │
│ • Neighbor calc  │  │ • Probe export   │  │ • Address reg    │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │  Distributed         │
                  │  Profiling           │
                  │  Implementation      │
                  │                      │
                  │ • network_probe.cc   │
                  │ • timestamp_sync.cc  │
                  │ • svm_wrapper.cc     │
                  └──────────────────────┘
```

**Problems**:
- ❌ Core XLA files heavily modified (+647 lines across 5 files)
- ❌ Tight coupling between distributed profiling and core
- ❌ Hard to maintain/upgrade XLA
- ❌ Hard to disable distributed profiling
- ❌ Difficult to upstream

---

## Proposed Architecture (After Refactoring)

```
┌─────────────────────────────────────────────────────────────────┐
│                        XLA Core Files                            │
│                  (MINIMAL MODIFICATIONS)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ se_gpu_pjrt      │  │ rocm_collector   │  │ coordination     │
│ _client.cc       │  │ .cc              │  │ _service_agent   │
│                  │  │                  │  │ .cc              │
│ +20 lines        │  │ +15 lines        │  │ +5 lines         │
│ ─────────────    │  │ ─────────────    │  │ ─────────────    │
│ • Hook only      │  │ • Plugin hook    │  │ • Plugin hook    │
│   #ifdef         │  │ • Export call    │  │   #ifdef         │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                     │                     │
        │                     │ calls               │
        │                     ▼                     │
        │           ┌──────────────────┐            │
        │           │ Plugin Registry  │            │
        │           │                  │            │
        │           │ • Initialize()   │            │
        │           │ • Export()       │            │
        │           │ • Lifecycle      │            │
        │           └──────────────────┘            │
        │                     │                     │
        │                     │ manages             │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                  ┌──────────────────────────────────┐
                  │     Plugin Implementation        │
                  │                                  │
                  │  ┌────────────────────────────┐  │
                  │  │ ProfilerPlugin Interface   │  │
                  │  └────────────────────────────┘  │
                  │              │                   │
                  │              ├─ implements       │
                  │              ▼                   │
                  │  ┌────────────────────────────┐  │
                  │  │ DistributedProfilerPlugin  │  │
                  │  │                            │  │
                  │  │ • Initialize()             │  │
                  │  │ • OnProfilingStart()       │  │
                  │  │ • ExportData()             │  │
                  │  │ • IsEnabled()              │  │
                  │  └────────────────────────────┘  │
                  │              │                   │
                  │              │ uses              │
                  │              ▼                   │
                  │  ┌────────────────────────────┐  │
                  │  │ Utility Modules            │  │
                  │  │                            │  │
                  │  │ • DistributedContextSetup  │  │
                  │  │   (graph gen, ports)       │  │
                  │  │                            │  │
                  │  │ • NetworkConfigPlugin      │  │
                  │  │   (IP discovery, KV)       │  │
                  │  │                            │  │
                  │  │ • NetworkProbe             │  │
                  │  │ • TimestampSync            │  │
                  │  │ • SVMWrapper               │  │
                  │  └────────────────────────────┘  │
                  └──────────────────────────────────┘
```

**Benefits**:
- ✅ Core XLA files minimally modified (~50 lines vs ~647)
- ✅ Clean separation via plugin interface
- ✅ Easy to enable/disable at runtime or build-time
- ✅ Easier to maintain and upstream
- ✅ Testable in isolation

---

## Key Design Patterns Applied

### 1. Plugin Pattern
```cpp
// Core code stays clean
ProfilerPluginRegistry::Get().ExportPluginData(space);

// Plugin implements the interface
class DistributedProfilerPlugin : public ProfilerPlugin {
  absl::Status ExportData(XSpace* space) override {
    // All distributed profiling logic here
  }
};
```

### 2. Dependency Injection
```cpp
// Instead of tight coupling:
// rocm_collector.cc directly calls distributed_timestamp_sync

// Use injection:
class RocmTraceCollectorImpl {
  void RegisterPlugin(ProfilerPlugin* plugin) {
    plugins_.push_back(plugin);
  }
};
```

### 3. Factory Pattern
```cpp
// Encapsulate complex initialization
class DistributedContextSetup {
  static absl::Status Initialize(int node_id, int num_nodes, 
                                   KeyValueStoreInterface* kv);
};

// Call from minimal hook in se_gpu_pjrt_client.cc
TF_RETURN_IF_ERROR(DistributedContextSetup::Initialize(...));
```

### 4. Strategy Pattern
```cpp
// Different strategies for network discovery
class NetworkConfigStrategy {
  virtual std::string GetLocalIP() = 0;
};

class NCCLConfigStrategy : public NetworkConfigStrategy {
  std::string GetLocalIP() override {
    // Use NCCL_SOCKET_IFNAME
  }
};
```

---

## File Organization

### Before
```
xla/
├── pjrt/gpu/
│   └── se_gpu_pjrt_client.cc  (358 lines of dist profiling mixed in)
├── backends/profiler/gpu/
│   ├── rocm_collector.cc  (97 lines of dist profiling mixed in)
│   ├── network_probe.cc  (new, standalone)
│   └── distributed_timestamp_sync.cc  (new, standalone)
└── tsl/distributed_runtime/coordination/
    └── coordination_service_agent.cc  (106 lines mixed in)
```

### After
```
xla/
├── pjrt/gpu/
│   ├── se_gpu_pjrt_client.cc  (20 lines hook)
│   └── distributed_context_setup.cc  (NEW - extracted)
│
├── backends/profiler/gpu/
│   ├── rocm_collector.cc  (15 lines plugin integration)
│   ├── profiler_plugin_interface.h  (NEW - interface)
│   ├── distributed_profiler_plugin.cc  (NEW - extracted)
│   ├── network_probe.cc  (existing, unchanged)
│   └── distributed_timestamp_sync.cc  (existing, unchanged)
│
└── tsl/distributed_runtime/coordination/
    ├── coordination_service_agent.cc  (5 lines hook)
    └── network_config_plugin.cc  (NEW - extracted)
```

---

## Data Flow Comparison

### Before: Tightly Coupled
```
User calls JAX
    ↓
PJRT Client Init
    ↓
se_gpu_pjrt_client.cc
    ↓
[358 LINES] Graph generation, port assignment
    ↓
KV Store
    ↓
Profiler Start
    ↓
rocm_collector.cc
    ↓
[97 LINES] Dist sync, snapshots, probes
    ↓
Network Probe
    ↓
Export Traces
```

### After: Plugin-Based
```
User calls JAX
    ↓
PJRT Client Init
    ↓
se_gpu_pjrt_client.cc [20 lines hook]
    ├─ if (DistProfiling enabled) ──→ DistributedContextSetup::Initialize()
    │                                      ↓
    │                                 [358 lines] Graph gen, ports
    │                                      ↓
    │                                 KV Store
    └─ Continue normal init
    ↓
Profiler Start
    ↓
rocm_collector.cc [15 lines]
    ├─ PluginRegistry::InitializePlugins()
    │       ↓
    │   DistributedProfilerPlugin::Initialize()
    │       ↓
    │   [97 lines] Dist sync, snapshots
    │       ↓
    │   Network Probe
    │
    └─ Continue normal profiling
    ↓
Export Traces
    ├─ Normal trace export
    └─ PluginRegistry::ExportPluginData()
            ↓
        DistributedProfilerPlugin::ExportData()
```

---

## Configuration & Control Flow

### Runtime Control (Primary Method)

**Three ways to configure** (in order of precedence):

#### 1. Environment Variables (Highest Precedence)
```bash
# Simple enable/disable
export XLA_ENABLE_DISTRIBUTED_PROFILING=1

# Fine-grained control
export XLA_ENABLE_DISTRIBUTED_PROFILING=1
export XLA_PROBE_CADENCE_US=1000
export XLA_PROBE_WINDOW_S=8
export XLA_PACKET_SPACING_US=100
export XLA_DIST_PROF_OUTPUT_DIR=/tmp/my_prof_data
```

#### 2. JSON Config File
```bash
# Create config file
cat > dist_prof_config.json <<EOF
{
  "enabled": true,
  "probe_cadence_us": 800,
  "probe_window_s": 4,
  "packet_spacing_us": 100,
  "snapshot_period_ms": 100,
  "output_dir": "/tmp/xla_dist_prof"
}
EOF

# Use config file
export XLA_DIST_PROF_CONFIG=dist_prof_config.json
python train.py
```

#### 3. Defaults (Lowest Precedence)
```cpp
struct DistributedProfilerConfig {
  bool enabled = false;              // Default: disabled
  int probe_cadence_us = 800;        // Default: 800μs
  int probe_window_s = 4;            // Default: 4s
  int packet_spacing_us = 100;       // Default: 100μs
  int snapshot_period_ms = 100;      // Default: 100ms
  std::string output_dir = "/tmp/xla_dist_prof";
};
```

### Implementation
```cpp
// Configuration loading (hybrid approach)
DistributedProfilerConfig DistributedProfilerConfig::Load() {
  DistributedProfilerConfig config;  // Start with defaults
  
  // Load from JSON file if specified
  if (auto* path = std::getenv("XLA_DIST_PROF_CONFIG")) {
    auto file_config = LoadFromFile(path);
    if (file_config.ok()) {
      config = *file_config;  // Override defaults
    }
  }
  
  // Override with individual env vars (highest precedence)
  if (auto* val = std::getenv("XLA_ENABLE_DISTRIBUTED_PROFILING")) {
    config.enabled = (std::string(val) == "1" || std::string(val) == "true");
  }
  if (auto* val = std::getenv("XLA_PROBE_CADENCE_US")) {
    config.probe_cadence_us = std::stoi(val);
  }
  // ... etc for other variables
  
  return config;
}

// Plugin uses loaded config
bool DistributedProfilerPlugin::IsEnabled() const {
  return config_.enabled;
}
```

### Compile-Time Control (Optional)

For users who want to completely exclude code:

```cpp
// In BUILD file
cc_library(
    name = "distributed_profiler_plugin",
    srcs = select({
        ":enable_distributed_profiling": ["distributed_profiler_plugin.cc"],
        "//conditions:default": ["empty_plugin.cc"],
    }),
)

// Build with/without
bazel build //xla/...  # Default: included but disabled at runtime
bazel build //xla/... --define enable_distributed_profiling=false  # Excluded
```

---

## Testing Strategy

### Unit Tests
```cpp
// Test plugin interface
TEST(ProfilerPluginTest, InterfaceContract) {
  MockProfilerPlugin plugin;
  EXPECT_CALL(plugin, Initialize(_)).WillOnce(Return(OkStatus()));
  // ...
}

// Test plugin registry
TEST(PluginRegistryTest, RegisterAndExecute) {
  auto plugin = std::make_unique<DistributedProfilerPlugin>();
  ProfilerPluginRegistry::Get().RegisterPlugin(std::move(plugin));
  // ...
}

// Test distributed logic in isolation
TEST(DistributedContextSetupTest, GraphGeneration) {
  // No need to initialize full PJRT client
  auto result = DistributedContextSetup::GenerateProbeGraph(4, mock_kv);
  EXPECT_OK(result);
}
```

### Integration Tests
```cpp
// Test with plugin enabled
TEST(RocmCollectorTest, WithDistributedProfiling) {
  setenv("XLA_ENABLE_DISTRIBUTED_PROFILING", "1", 1);
  // ... run profiling test
}

// Test with plugin disabled
TEST(RocmCollectorTest, WithoutDistributedProfiling) {
  setenv("XLA_ENABLE_DISTRIBUTED_PROFILING", "0", 1);
  // ... ensure no performance impact
}
```

---

## Migration Path

### Phase 1: Add Plugin Infrastructure (No Functional Changes)
- Add plugin interface
- Add plugin registry
- Add empty plugin implementation
- Core code calls plugin (but plugin does nothing)
- **Verify**: All tests pass, no behavior change

### Phase 2: Move rocm_collector.cc Logic
- Implement `DistributedProfilerPlugin`
- Move 97 lines from `rocm_collector.cc` to plugin
- Update `rocm_collector.cc` to call plugin
- **Verify**: Distributed profiling still works

### Phase 3: Move se_gpu_pjrt_client.cc Logic
- Implement `DistributedContextSetup`
- Move 358 lines from `se_gpu_pjrt_client.cc` to utility
- Update client to call utility
- **Verify**: Multi-node setup still works

### Phase 4: Move coordination_service_agent.cc Logic
- Implement `NetworkConfigPlugin`
- Move 106 lines to plugin
- Update agent to call plugin
- **Verify**: Node address discovery still works

### Phase 5: Add Configuration Controls
- Add environment variable checks
- Add build flags
- Test enable/disable
- **Verify**: Works both enabled and disabled

### Phase 6: Cleanup & Documentation
- Remove debug code
- Update documentation
- Add migration guide
- **Verify**: Everything documented

---

## Metrics for Success

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Lines modified in core files | 647 | 50 | < 100 |
| Files with 100+ line changes | 3 | 0 | 0 |
| Plugin interface complexity | N/A | 5 methods | < 10 |
| Runtime overhead (disabled) | 0% | 0% | < 0.1% |
| Runtime overhead (enabled) | baseline | baseline | < 1% |
| Test coverage | ~60% | 90% | > 85% |
| Build time impact | 0s | +5s | < 10s |

---

## Long-Term Vision

### Potential Future Plugins
Once infrastructure is in place, can add:
- **GPU metrics plugin**: Vendor-specific GPU stats
- **Network topology plugin**: Custom network configs
- **Custom tracing plugin**: Application-specific events
- **Performance analysis plugin**: Real-time anomaly detection

### Upstream Contribution Path
1. **Step 1**: Propose plugin interface to XLA team
2. **Step 2**: Upstream plugin infrastructure (generic)
3. **Step 3**: Optionally upstream distributed profiling as example plugin
4. **Step 4**: Let community develop more plugins

---

## Conclusion

The plugin-based refactoring provides:
- **90% reduction** in modifications to core XLA files
- **Clear separation** between core and extensions
- **Easy maintenance** of both XLA updates and distributed profiling
- **Path to upstream** contribution
- **Foundation** for future extensions

**Recommended**: Proceed with refactoring in 6 phases over 4 weeks.

---

**Document Status**: Proposal  
**Related**: [REFACTORING_PLAN.md](REFACTORING_PLAN.md)  
**Last Updated**: 2025-12-08

