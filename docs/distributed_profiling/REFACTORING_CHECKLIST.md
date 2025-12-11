# Distributed Profiling Refactoring Checklist

This is a practical, step-by-step checklist for implementing the refactoring plan.

---

## Pre-Refactoring

- [ ] Create feature branch: `git checkout -b refactor/plugin-architecture`
- [ ] Run full test suite and record baseline: `./run_tests.sh > baseline_tests.log`
- [ ] Document current behavior for verification later
- [ ] Back up modified files (just in case)
- [ ] Review REFACTORING_PLAN.md and REFACTORING_ARCHITECTURE.md

---

## Phase 1: Plugin Infrastructure

### 1.1 Create Plugin Interface
- [ ] Create `xla/backends/profiler/gpu/profiler_plugin_interface.h`
  ```cpp
  class ProfilerPlugin {
    virtual ~ProfilerPlugin() = default;
    virtual absl::Status Initialize(const ProfileOptions&) = 0;
    virtual absl::Status OnProfilingStart(RocmTraceCollector*) = 0;
    virtual absl::Status OnProfilingStop() = 0;
    virtual absl::Status ExportData(XSpace*) = 0;
    virtual bool IsEnabled() const = 0;
  };
  ```

- [ ] Create `xla/backends/profiler/gpu/profiler_plugin_interface.cc`
  - Implement `ProfilerPluginRegistry` singleton
  - Implement `RegisterPlugin()`
  - Implement `InitializePlugins()`
  - Implement `OnProfilingStart()`
  - Implement `OnProfilingStop()`
  - Implement `ExportPluginData()`

- [ ] Update `xla/backends/profiler/gpu/BUILD`
  ```python
  cc_library(
      name = "profiler_plugin_interface",
      srcs = ["profiler_plugin_interface.cc"],
      hdrs = ["profiler_plugin_interface.h"],
      deps = [
          "//xla:status",
          "@com_google_absl//absl/status",
          # ...
      ],
  )
  ```

- [ ] Write unit tests: `profiler_plugin_interface_test.cc`
  - Test plugin registration
  - Test lifecycle calls
  - Test with mock plugin
  
- [ ] Run tests: `bazel test //xla/backends/profiler/gpu:profiler_plugin_interface_test`

### 1.2 Integrate Plugin System into rocm_collector.cc
- [ ] Add include to `rocm_collector.h`:
  ```cpp
  #include "xla/backends/profiler/gpu/profiler_plugin_interface.h"
  ```

- [ ] Add plugin initialization call in `RocmTraceCollectorImpl::Start()` (or appropriate init method):
  ```cpp
  absl::Status RocmTraceCollectorImpl::InitializePlugins() {
    ProfilerPluginRegistry::Get().InitializePlugins(options_);
    ProfilerPluginRegistry::Get().OnProfilingStart(this);
    return absl::OkStatus();
  }
  ```

- [ ] Add plugin export call in `RocmTraceCollectorImpl::Export()`:
  ```cpp
  void RocmTraceCollectorImpl::Export(XSpace* space) {
    // ... existing export code ...
    
    // Export plugin data
    TF_CHECK_OK(ProfilerPluginRegistry::Get().ExportPluginData(space));
    
    // ... rest of code ...
  }
  ```

- [ ] Add plugin stop call in appropriate destructor/stop method:
  ```cpp
  TF_CHECK_OK(ProfilerPluginRegistry::Get().OnProfilingStop());
  ```

- [ ] Update BUILD file to add dependency on `profiler_plugin_interface`

- [ ] Compile and run tests: `bazel test //xla/backends/profiler/gpu:rocm_collector_test`

### 1.3 Verify Phase 1
- [ ] All existing tests pass
- [ ] No behavior changes (plugin registry is empty, so no-op)
- [ ] Code compiles without warnings
- [ ] Commit: `git commit -m "Add profiler plugin infrastructure (no-op)"`

---

## Phase 2: Extract Distributed Profiler Plugin

### 2.1 Create Empty Plugin
- [ ] Create `xla/backends/profiler/gpu/distributed_profiler_plugin.h`
  ```cpp
  class DistributedProfilerPlugin : public ProfilerPlugin {
   public:
    DistributedProfilerPlugin() = default;
    ~DistributedProfilerPlugin() override = default;
    
    absl::Status Initialize(const ProfileOptions& options) override;
    absl::Status OnProfilingStart(RocmTraceCollector* collector) override;
    absl::Status OnProfilingStop() override;
    absl::Status ExportData(XSpace* space) override;
    bool IsEnabled() const override;
    
   private:
    std::unique_ptr<DistributedTimestampSynchronizer> ts_sync_;
    // ... other members
  };
  ```

- [ ] Create `xla/backends/profiler/gpu/distributed_profiler_plugin.cc` (empty implementations)

- [ ] Update BUILD file

### 2.2 Move Initialization Logic
- [ ] Copy `InitializeDistributedSync()` from `rocm_collector.cc` to `DistributedProfilerPlugin::Initialize()`
  - Original location: `rocm_collector.cc` lines ~549-588
  - New location: `distributed_profiler_plugin.cc`

- [ ] Move snapshot thread logic:
  - `StartSnapshotThread()` → plugin
  - `StopSnapshotThread()` → plugin
  - `snapshot_thread_`, `snapshot_mutex_`, `snapshots_` → plugin members

- [ ] Update plugin to store collector reference for snapshot thread

### 2.3 Move Export Logic
- [ ] Copy probe data export from `rocm_collector.cc::Export()` to `DistributedProfilerPlugin::ExportData()`
  - Lines ~600-610 (probe data export)
  - Lines ~620-634 (snapshot export)

- [ ] Remove moved code from `rocm_collector.cc`

### 2.4 Add IsEnabled() Check
- [ ] Implement environment variable check:
  ```cpp
  bool DistributedProfilerPlugin::IsEnabled() const {
    static bool enabled = []() {
      bool val = false;
      TF_CHECK_OK(tsl::ReadBoolFromEnvVar(
          "XLA_ENABLE_DISTRIBUTED_PROFILING",
          /*default_val=*/false,
          &val));
      return val;
    }();
    return enabled;
  }
  ```

- [ ] Add early return in each method if not enabled

### 2.5 Register Plugin
- [ ] Add plugin registration in appropriate initialization point
  - Option A: In `rocm_collector.cc` constructor/factory
  - Option B: Static registration in `distributed_profiler_plugin.cc`
  
  ```cpp
  // Static registration approach:
  namespace {
  struct PluginRegistrar {
    PluginRegistrar() {
      ProfilerPluginRegistry::Get().RegisterPlugin(
          std::make_unique<DistributedProfilerPlugin>());
    }
  };
  static PluginRegistrar registrar;
  }
  ```

### 2.6 Clean up rocm_collector.cc
- [ ] Remove `InitializeDistributedSync()` method
- [ ] Remove snapshot thread methods
- [ ] Remove distributed sync members
- [ ] Remove direct includes of `distributed_timestamp_sync.h`
- [ ] Remove direct includes of `network_probe.h`
- [ ] Keep only plugin system calls

### 2.7 Verify Phase 2
- [ ] Set `XLA_ENABLE_DISTRIBUTED_PROFILING=1`
- [ ] Run distributed profiling tests
- [ ] Verify probe data still exported
- [ ] Verify snapshots still captured
- [ ] Set `XLA_ENABLE_DISTRIBUTED_PROFILING=0`
- [ ] Verify no overhead when disabled
- [ ] Commit: `git commit -m "Extract distributed profiling to plugin"`

---

## Phase 3: Extract PJRT Client Extensions

### 3.1 Create Distributed Context Setup Utility
- [ ] Create `xla/pjrt/gpu/distributed_context_setup.h`
  ```cpp
  class DistributedContextSetup {
   public:
    static absl::Status Initialize(
        int node_id,
        int num_nodes,
        KeyValueStoreInterface* kv_store);
    
   private:
    static absl::StatusOr<std::vector<std::string>> 
        ExchangeNodeAddresses(int node_id, int num_nodes, 
                             KeyValueStoreInterface* kv_store);
    
    static absl::StatusOr<std::pair<std::vector<int>, std::vector<int>>>
        GenerateDirectedNeighbors(int node_id, int num_nodes,
                                 KeyValueStoreInterface* kv_store);
  };
  ```

- [ ] Create `xla/pjrt/gpu/distributed_context_setup.cc`

### 3.2 Move Functions from se_gpu_pjrt_client.cc
- [ ] Move `ExchangeNodeAddresses()` (lines ~167-185)
- [ ] Move `GenerateDirectedNeighbors()` (lines ~189-272)
- [ ] Move any helper functions related to graph generation
- [ ] Update to use class method signatures

### 3.3 Add Minimal Hook in se_gpu_pjrt_client.cc
- [ ] Find where distributed context setup is currently called
- [ ] Replace with:
  ```cpp
  #if defined(TENSORFLOW_USE_ROCM) && defined(XLA_ENABLE_DISTRIBUTED_PROFILING)
  if (num_nodes > 1) {
    TF_RETURN_IF_ERROR(
        DistributedContextSetup::Initialize(node_id, num_nodes, kv_store));
  }
  #endif
  ```

### 3.4 Clean up se_gpu_pjrt_client.cc
- [ ] Remove `ExchangeNodeAddresses()` definition
- [ ] Remove `GenerateDirectedNeighbors()` definition
- [ ] Remove unnecessary includes (if any)
- [ ] Add include for `distributed_context_setup.h`

### 3.5 Update BUILD Files
- [ ] Create `distributed_context_setup` build target
- [ ] Add dependency to `se_gpu_pjrt_client`
- [ ] Use `select()` for conditional compilation:
  ```python
  cc_library(
      name = "se_gpu_pjrt_client",
      deps = select({
          ":enable_distributed_profiling": [
              ":distributed_context_setup",
          ],
          "//conditions:default": [],
      }),
  )
  ```

### 3.6 Verify Phase 3
- [ ] Run multi-node setup tests
- [ ] Verify graph generation still works
- [ ] Verify port assignment still works
- [ ] Verify neighbor calculation correct
- [ ] Commit: `git commit -m "Extract PJRT distributed setup to utility"`

---

## Phase 4: Extract Coordination Service Extensions

### 4.1 Create Network Config Plugin
- [ ] Create `xla/tsl/distributed_runtime/coordination/network_config_plugin.h`
  ```cpp
  class NetworkConfigPlugin {
   public:
    static absl::Status RegisterNodeAddress(
        CoordinationServiceAgent* agent,
        int task_id);
    
   private:
    static std::string GetLocalIP(const std::string& interface);
    static std::string GetHostname();
    static std::vector<std::string> GetNetworkInterfaces();
  };
  ```

- [ ] Create `xla/tsl/distributed_runtime/coordination/network_config_plugin.cc`

### 4.2 Move IP Discovery Logic
- [ ] Move `get_hostname()` lambda (lines ~253-259)
- [ ] Move `get_local_ip()` lambda (lines ~261-284)
- [ ] Move IP registration logic (lines ~324-354)
- [ ] Convert to class methods

### 4.3 Add Minimal Hook in coordination_service_agent.cc
- [ ] Find location in `Connect()` method (after "successfully connected" log)
- [ ] Add hook:
  ```cpp
  #ifdef XLA_ENABLE_DISTRIBUTED_PROFILING
  TF_RETURN_IF_ERROR(
      NetworkConfigPlugin::RegisterNodeAddress(this, task_.task_id()));
  #endif
  ```

### 4.4 Clean up coordination_service_agent.cc
- [ ] Remove `get_hostname()` lambda
- [ ] Remove `get_local_ip()` lambda
- [ ] Remove IP registration code
- [ ] Remove unnecessary includes (arpa/inet.h, ifaddrs.h, etc.)
- [ ] Add include for `network_config_plugin.h`

### 4.5 Update BUILD Files
- [ ] Create `network_config_plugin` target
- [ ] Add conditional dependency

### 4.6 Verify Phase 4
- [ ] Run coordination service tests
- [ ] Verify node addresses still exchanged
- [ ] Verify KV store populated correctly
- [ ] Commit: `git commit -m "Extract network config to plugin"`

---

## Phase 5: Add Configuration Controls

### 5.1 Create Configuration Structure
- [ ] Create `xla/backends/profiler/gpu/distributed_profiler_config.h`
  ```cpp
  struct DistributedProfilerConfig {
    bool enabled = false;
    int probe_cadence_us = 800;
    int probe_window_s = 4;
    int packet_spacing_us = 100;
    int snapshot_period_ms = 100;
    std::string output_dir = "/tmp/xla_dist_prof";
    
    static DistributedProfilerConfig Load();
   private:
    static DistributedProfilerConfig LoadFromEnvVars();
    static absl::StatusOr<DistributedProfilerConfig> LoadFromFile(
        const std::string& path);
  };
  ```

- [ ] Create `xla/backends/profiler/gpu/distributed_profiler_config.cc`
- [ ] Update BUILD file to add `distributed_profiler_config` target

### 5.2 Implement Configuration Loading
- [ ] Implement `LoadFromEnvVars()`:
  - Read `XLA_ENABLE_DISTRIBUTED_PROFILING`
  - Read `XLA_PROBE_CADENCE_US`
  - Read `XLA_PROBE_WINDOW_S`
  - Read `XLA_PACKET_SPACING_US`
  - Read `XLA_DIST_PROF_OUTPUT_DIR`

- [ ] Implement `LoadFromFile()`:
  - Check if `XLA_DIST_PROF_CONFIG` env var is set
  - Read and parse JSON file (use nlohmann/json or absl JSON)
  - Handle file not found gracefully
  - Handle parse errors gracefully
  - Log loaded config values

- [ ] Implement `Load()` with precedence:
  ```cpp
  DistributedProfilerConfig Load() {
    // 1. Start with defaults
    DistributedProfilerConfig config;
    
    // 2. Override with config file (if specified)
    if (auto* path = getenv("XLA_DIST_PROF_CONFIG")) {
      auto file_config = LoadFromFile(path);
      if (file_config.ok()) config = *file_config;
    }
    
    // 3. Override with individual env vars (highest precedence)
    bool enabled;
    if (ReadBoolFromEnvVar("XLA_ENABLE_DISTRIBUTED_PROFILING", 
                           config.enabled, &enabled).ok()) {
      config.enabled = enabled;
    }
    // ... similar for other vars
    
    return config;
  }
  ```

### 5.3 Example Config Files
- [ ] Create example config: `examples/dist_prof_config.json`
  ```json
  {
    "enabled": true,
    "probe_cadence_us": 800,
    "probe_window_s": 4,
    "packet_spacing_us": 100,
    "snapshot_period_ms": 100,
    "output_dir": "/tmp/xla_dist_prof"
  }
  ```

- [ ] Create aggressive config: `examples/dist_prof_aggressive.json`
  ```json
  {
    "enabled": true,
    "probe_cadence_us": 400,
    "probe_window_s": 2,
    "packet_spacing_us": 50,
    "snapshot_period_ms": 50,
    "output_dir": "/tmp/xla_dist_prof"
  }
  ```

- [ ] Create conservative config: `examples/dist_prof_conservative.json`
  ```json
  {
    "enabled": true,
    "probe_cadence_us": 2000,
    "probe_window_s": 10,
    "packet_spacing_us": 200,
    "snapshot_period_ms": 200,
    "output_dir": "/tmp/xla_dist_prof"
  }
  ```

### 5.4 Integrate Configuration into Plugin
- [ ] Update `DistributedProfilerPlugin` to use config:
  ```cpp
  class DistributedProfilerPlugin : public ProfilerPlugin {
   private:
    DistributedProfilerConfig config_;
  };
  ```

- [ ] Load config in `Initialize()`:
  ```cpp
  absl::Status Initialize(const ProfileOptions& options) override {
    config_ = DistributedProfilerConfig::Load();
    
    if (!config_.enabled) {
      VLOG(1) << "Distributed profiling disabled";
      return absl::OkStatus();
    }
    
    LOG(INFO) << "Distributed profiling config:";
    LOG(INFO) << "  probe_cadence_us: " << config_.probe_cadence_us;
    LOG(INFO) << "  probe_window_s: " << config_.probe_window_s;
    // ... log other values
    
    // Use config values for initialization
    // ...
  }
  ```

- [ ] Update `IsEnabled()`:
  ```cpp
  bool IsEnabled() const override {
    return config_.enabled;
  }
  ```

### 5.5 Update Documentation
- [ ] Document environment variables in README:
  ```markdown
  ## Configuration
  
  ### Method 1: Simple Enable/Disable
  export XLA_ENABLE_DISTRIBUTED_PROFILING=1
  python train.py
  
  ### Method 2: Individual Variables
  export XLA_ENABLE_DISTRIBUTED_PROFILING=1
  export XLA_PROBE_CADENCE_US=1000
  export XLA_PROBE_WINDOW_S=8
  python train.py
  
  ### Method 3: Config File
  XLA_DIST_PROF_CONFIG=config.json python train.py
  
  ### Precedence
  Env vars > Config file > Defaults
  ```

- [ ] Create configuration guide: `docs/CONFIGURATION_GUIDE.md`
- [ ] Add config examples to documentation

### 5.6 Test Configuration
- [ ] Test with simple enable: 
  ```bash
  XLA_ENABLE_DISTRIBUTED_PROFILING=1 ./test
  ```

- [ ] Test with disabled (default):
  ```bash
  XLA_ENABLE_DISTRIBUTED_PROFILING=0 ./test
  ```

- [ ] Test with individual env vars:
  ```bash
  XLA_ENABLE_DISTRIBUTED_PROFILING=1 \
  XLA_PROBE_CADENCE_US=1000 \
  XLA_PROBE_WINDOW_S=8 \
  ./test
  ```

- [ ] Test with config file:
  ```bash
  XLA_DIST_PROF_CONFIG=examples/dist_prof_config.json ./test
  ```

- [ ] Test config file with overrides:
  ```bash
  XLA_DIST_PROF_CONFIG=examples/dist_prof_config.json \
  XLA_PROBE_CADENCE_US=2000 \
  ./test
  ```

- [ ] Test config file not found (should fall back to defaults)
- [ ] Test invalid JSON (should log warning and use defaults)
- [ ] Verify precedence order is correct

### 5.7 Optional: Build Flags
- [ ] Create `.bazelrc` config (optional):
  ```
  build:dist_prof --define enable_distributed_profiling=true
  ```

- [ ] Add config_setting in BUILD file (optional)
- [ ] Test build with/without flag (optional)

### 5.8 Verify Phase 5
- [ ] Configuration loads correctly from all sources
- [ ] Precedence order works as expected
- [ ] Invalid config handled gracefully
- [ ] Can enable/disable at runtime
- [ ] Config values propagate to probe system
- [ ] No overhead when disabled
- [ ] Documentation complete
- [ ] Commit: `git commit -m "Add runtime configuration system"`

---

## Phase 6: Testing & Validation

### 6.1 Unit Tests
- [ ] Write tests for `ProfilerPluginInterface`
- [ ] Write tests for `DistributedProfilerPlugin` (mocked)
- [ ] Write tests for `DistributedContextSetup`
- [ ] Write tests for `NetworkConfigPlugin`
- [ ] Run all unit tests: `bazel test //xla/backends/profiler/gpu:all`

### 6.2 Integration Tests
- [ ] Run standalone test: `./network_probe_standalone_test`
- [ ] Run 2-node distributed test
- [ ] Run 4-node distributed test
- [ ] Run 8-node distributed test

### 6.3 Performance Testing
- [ ] Benchmark with distributed profiling enabled
- [ ] Benchmark with distributed profiling disabled
- [ ] Compare against baseline (Phase 0)
- [ ] Verify < 1% overhead when enabled
- [ ] Verify < 0.1% overhead when disabled

### 6.4 Regression Testing
- [ ] Run full test suite: `bazel test //xla/...`
- [ ] Compare against baseline_tests.log
- [ ] Investigate any failures
- [ ] Fix any regressions

### 6.5 Documentation Testing
- [ ] Follow STANDALONE_TEST_README.md
- [ ] Follow REFACTORING_PLAN.md
- [ ] Verify all examples work
- [ ] Update docs if needed

### 6.6 Code Quality
- [ ] Run linter: `./lint.sh`
- [ ] Run clang-format: `clang-format -i **/*.cc **/*.h`
- [ ] Check for TODOs and address them
- [ ] Check for commented-out code and remove
- [ ] Review code for clarity

### 6.7 Verify Phase 6
- [ ] All tests pass
- [ ] Performance acceptable
- [ ] Code quality high
- [ ] Commit: `git commit -m "Add tests and validation"`

---

## Phase 7: Cleanup & Documentation

### 7.1 Code Cleanup
- [ ] Remove any temporary/debug code
- [ ] Remove commented-out code
- [ ] Consolidate headers
- [ ] Remove unused includes
- [ ] Fix any linter warnings

### 7.2 Update Documentation
- [ ] Update DOCUMENTATION_INDEX.md
- [ ] Update README.md with configuration instructions
- [ ] Add plugin development guide
- [ ] Add migration guide for upgrading
- [ ] Document environment variables
- [ ] Document build flags

### 7.3 Add Examples
- [ ] Add example of using plugin system
- [ ] Add example of disabling distributed profiling
- [ ] Add example of custom plugin (future work)

### 7.4 Final Review
- [ ] Review all changed files
- [ ] Verify minimal changes to core files
- [ ] Count lines modified (should be ~50 in core files)
- [ ] Generate final diff: `git diff rocm-jaxlib-v0.7.1 > refactoring.diff`

### 7.5 Verify Phase 7
- [ ] Documentation complete and accurate
- [ ] Code clean and well-commented
- [ ] Ready for review
- [ ] Commit: `git commit -m "Cleanup and documentation"`

---

## Post-Refactoring

### Merge & Deploy
- [ ] Create pull request
- [ ] Request code review
- [ ] Address review comments
- [ ] Get approval
- [ ] Merge to main branch
- [ ] Tag release: `git tag v2.0.0-refactored`
- [ ] Deploy to staging
- [ ] Deploy to production

### Verification in Production
- [ ] Monitor performance
- [ ] Check for errors/crashes
- [ ] Verify traces exported correctly
- [ ] Verify probe data accurate
- [ ] Monitor for regressions

### Celebration! 🎉
- [ ] Document lessons learned
- [ ] Share refactoring experience with team
- [ ] Update project status

---

## Rollback Plan (If Needed)

If something goes wrong:
1. `git revert` commits in reverse order
2. Or `git reset --hard <commit-before-refactoring>`
3. Re-deploy previous version
4. Investigate issues
5. Fix and try again

---

## Metrics Tracking

Track these metrics before and after:

| Metric | Baseline | Target | Actual |
|--------|----------|--------|--------|
| Lines in se_gpu_pjrt_client.cc | +358 | +20 | _____ |
| Lines in rocm_collector.cc | +97 | +15 | _____ |
| Lines in coordination_service_agent.cc | +106 | +5 | _____ |
| Total lines in core files | 647 | 50 | _____ |
| Build time (no dist prof) | ___s | < +10s | _____ |
| Build time (with dist prof) | ___s | < +30s | _____ |
| Runtime overhead (disabled) | 0% | < 0.1% | _____ |
| Runtime overhead (enabled) | baseline | < 1% | _____ |
| Test coverage | ___% | > 85% | _____ |
| Number of tests | ___ | ___ | _____ |

---

## Common Issues & Solutions

### Issue: Plugin not getting called
**Solution**: Check registration is happening, add debug logs

### Issue: Build errors with conditional compilation
**Solution**: Check `#ifdef` guards match build flags

### Issue: Tests failing after Phase X
**Solution**: Verify moved code matches exactly, check dependencies

### Issue: Performance regression
**Solution**: Profile code, check for unintended overhead in hot path

### Issue: Distributed profiling not working
**Solution**: Check environment variable set, verify plugin enabled

---

## Notes & Lessons Learned

_(Fill in as you go)_

---

**Status**: Ready to start  
**Estimated Time**: 4 weeks (1 week per main phase)  
**Last Updated**: 2025-12-08

