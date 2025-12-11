# Refactoring Summary: Plugin-Based Architecture

## 🎯 Goal
Minimize in-place modifications to original XLA files by extracting distributed profiling into a plugin architecture.

## 📊 Impact

### Before Refactoring
```
Modified Files:        13 files
Core File Changes:     ~647 lines added/modified across 5 files
Largest Change:        se_gpu_pjrt_client.cc (+358 lines)
Maintainability:       ❌ Poor (tight coupling)
Upgradability:         ❌ Difficult (conflicts with upstream)
Testability:          ⚠️  Mixed (hard to isolate)
```

### After Refactoring
```
Modified Files:        13 files (same)
Core File Changes:     ~50 lines hooks/calls across 4 files
Largest Change:        se_gpu_pjrt_client.cc (+20 lines)
Maintainability:       ✅ Good (clean separation)
Upgradability:         ✅ Easy (minimal core changes)
Testability:          ✅ Excellent (plugins isolated)
```

**Result**: **93% reduction** in core file modifications (647 → 50 lines)

---

## 📁 Key Documents

1. **[REFACTORING_PLAN.md](REFACTORING_PLAN.md)** - Detailed strategy and analysis
2. **[REFACTORING_ARCHITECTURE.md](REFACTORING_ARCHITECTURE.md)** - Before/after architecture diagrams
3. **[REFACTORING_CHECKLIST.md](REFACTORING_CHECKLIST.md)** - Step-by-step implementation guide
4. **This Document** - Quick reference and overview

---

## 🏗️ Architecture Overview

### Plugin System

```
Core XLA Code (~50 lines of hooks)
        ↓
  Plugin Registry
        ↓
Distributed Profiler Plugin
        ↓
Implementation Modules
(network_probe, timestamp_sync, etc.)
```

### Key Components

| Component | Purpose | Lines |
|-----------|---------|-------|
| `ProfilerPluginInterface` | Abstract plugin interface | ~100 |
| `ProfilerPluginRegistry` | Plugin management singleton | ~50 |
| `DistributedProfilerPlugin` | Distributed profiling implementation | ~300 |
| `DistributedContextSetup` | Graph generation & port assignment | ~400 |
| `NetworkConfigPlugin` | IP discovery & KV store | ~150 |

---

## 🔑 Core File Changes

### 1. `se_gpu_pjrt_client.cc` (358 → 20 lines)

**Before**:
```cpp
// 358 lines of graph generation, port assignment, neighbor calculation
absl::StatusOr<std::vector<std::string>> ExchangeNodeAddresses(...) { ... }
absl::StatusOr<std::pair<...>> GenerateDirectedNeighbors(...) { ... }
// ... many more functions ...
```

**After**:
```cpp
#if defined(TENSORFLOW_USE_ROCM) && defined(XLA_ENABLE_DISTRIBUTED_PROFILING)
if (num_nodes > 1) {
  TF_RETURN_IF_ERROR(
      DistributedContextSetup::Initialize(node_id, num_nodes, kv_store));
}
#endif
```

### 2. `rocm_collector.cc` (97 → 15 lines)

**Before**:
```cpp
// 97 lines of distributed sync, snapshot threads, probe export
absl::Status RocmTraceCollectorImpl::InitializeDistributedSync() { ... }
void RocmTraceCollectorImpl::StartSnapshotThread(...) { ... }
void RocmTraceCollectorImpl::Export(XSpace* space) {
  // ... lots of distributed profiling code ...
}
```

**After**:
```cpp
void RocmTraceCollectorImpl::InitializePlugins() {
  ProfilerPluginRegistry::Get().InitializePlugins(options_);
  ProfilerPluginRegistry::Get().OnProfilingStart(this);
}

void RocmTraceCollectorImpl::Export(XSpace* space) {
  // ... existing code ...
  ProfilerPluginRegistry::Get().ExportPluginData(space);
  // ... existing code ...
}
```

### 3. `coordination_service_agent.cc` (106 → 5 lines)

**Before**:
```cpp
// 106 lines of IP discovery, hostname lookup, interface enumeration
auto get_hostname = []() -> std::string { ... };
auto get_local_ip = [](const std::string& ifname) -> std::string { ... };
// ... lots more code ...
```

**After**:
```cpp
#ifdef XLA_ENABLE_DISTRIBUTED_PROFILING
TF_RETURN_IF_ERROR(
    NetworkConfigPlugin::RegisterNodeAddress(this, task_.task_id()));
#endif
```

### 4. `device_tracer_rocm.cc` (36 → 10 lines)

Minor changes for options passing and parsing - stays similar.

---

## ⚙️ Configuration

### Runtime Control (Recommended)

**Three ways to configure** (env vars > config file > defaults):

#### Option 1: Simple Enable/Disable
```bash
# Enable with defaults
export XLA_ENABLE_DISTRIBUTED_PROFILING=1
python train.py

# Disable
export XLA_ENABLE_DISTRIBUTED_PROFILING=0
python train.py
```

#### Option 2: Individual Environment Variables
```bash
export XLA_ENABLE_DISTRIBUTED_PROFILING=1
export XLA_PROBE_CADENCE_US=1000
export XLA_PROBE_WINDOW_S=8
export XLA_PACKET_SPACING_US=100
export XLA_DIST_PROF_OUTPUT_DIR=/tmp/my_prof_data
python train.py
```

#### Option 3: JSON Config File
```bash
# Create config.json
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
XLA_DIST_PROF_CONFIG=dist_prof_config.json python train.py

# Or use config file with overrides
XLA_DIST_PROF_CONFIG=dist_prof_config.json \
XLA_PROBE_CADENCE_US=2000 \
python train.py
```

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `XLA_ENABLE_DISTRIBUTED_PROFILING` | `0` | Enable/disable distributed profiling |
| `XLA_DIST_PROF_CONFIG` | (none) | Path to JSON config file |
| `XLA_PROBE_CADENCE_US` | `800` | Probe cadence in microseconds |
| `XLA_PROBE_WINDOW_S` | `4` | Probe window size in seconds |
| `XLA_PACKET_SPACING_US` | `100` | Packet spacing in microseconds |
| `XLA_DIST_PROF_OUTPUT_DIR` | `/tmp/xla_dist_prof` | Output directory for probe data |
| `NCCL_SOCKET_IFNAME` | `eth0` | Network interface for probing |

**Precedence**: Individual env vars > Config file values > Defaults

### Build-Time Control (Optional)

For advanced users who want to exclude code entirely:

```bash
# Build without distributed profiling support
bazel build //xla/... --define enable_distributed_profiling=false

# Default build (included but disabled at runtime)
bazel build //xla/...
```

---

## 📋 Implementation Phases

| Phase | Duration | Key Tasks | Output |
|-------|----------|-----------|--------|
| **Phase 1** | 1 week | Plugin infrastructure | Interface + Registry |
| **Phase 2** | 1 week | Extract rocm_collector logic | DistributedProfilerPlugin |
| **Phase 3** | 1 week | Extract PJRT logic | DistributedContextSetup |
| **Phase 4** | 3 days | Extract coordination logic | NetworkConfigPlugin |
| **Phase 5** | 2 days | Configuration controls | Env vars + build flags |
| **Phase 6** | 2 days | Testing & validation | Tests + benchmarks |
| **Phase 7** | 2 days | Cleanup & docs | Final documentation |

**Total**: ~4 weeks

---

## ✅ Verification Steps

After each phase, verify:

```bash
# 1. Code compiles
bazel build //xla/backends/profiler/gpu/...

# 2. Unit tests pass
bazel test //xla/backends/profiler/gpu:all

# 3. Integration tests pass
bazel test //xla/backends/profiler/gpu:network_probe_standalone_test

# 4. Distributed profiling still works
XLA_ENABLE_DISTRIBUTED_PROFILING=1 ./run_distributed_test.sh

# 5. No overhead when disabled
XLA_ENABLE_DISTRIBUTED_PROFILING=0 ./benchmark.sh

# 6. Changes minimized
git diff rocm-jaxlib-v0.7.1 --stat
```

---

## 🚀 Getting Started

### 1. Review Documentation
```bash
# Read in this order:
cat REFACTORING_SUMMARY.md        # This file (overview)
cat REFACTORING_PLAN.md           # Detailed strategy
cat REFACTORING_ARCHITECTURE.md   # Architecture diagrams
cat REFACTORING_CHECKLIST.md      # Step-by-step guide
```

### 2. Set Up Branch
```bash
git checkout -b refactor/plugin-architecture
```

### 3. Baseline Testing
```bash
./run_tests.sh > baseline_tests.log
./benchmark.sh > baseline_bench.log
```

### 4. Start Phase 1
```bash
# Follow REFACTORING_CHECKLIST.md Phase 1
# Create plugin interface files
# Write tests
# Integrate into rocm_collector
```

### 5. Iterate Through Phases
Continue with Phase 2, 3, 4, etc., testing after each phase.

---

## 📈 Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Lines in core files | < 100 | `git diff --stat` |
| Files with 100+ line changes | 0 | `git diff --stat` |
| Runtime overhead (disabled) | < 0.1% | Benchmarks |
| Runtime overhead (enabled) | < 1% | Benchmarks |
| Test coverage | > 85% | Coverage tools |
| Build time increase | < 10s | Time bazel build |

---

## 🛠️ Troubleshooting

### Build Errors
```bash
# Check for missing dependencies
bazel query 'deps(//xla/backends/profiler/gpu:distributed_profiler_plugin)'

# Check for circular dependencies
bazel query 'somepath(//xla/..., //xla/...)'
```

### Tests Failing
```bash
# Run specific test with verbose output
bazel test //xla/backends/profiler/gpu:profiler_plugin_interface_test --test_output=all

# Check test logs
cat bazel-testlogs/.../test.log
```

### Plugin Not Loading
```bash
# Add debug logging
VLOG(1) << "Plugin registered: " << typeid(*plugin).name();

# Check environment variable
echo $XLA_ENABLE_DISTRIBUTED_PROFILING
```

### Performance Regression
```bash
# Profile before and after
perf record -g ./benchmark
perf report

# Check for unintended calls in hot path
```

---

## 📚 References

### Internal Docs
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Main documentation index
- [DIRECTED_PROBE_SPEC.md](xla/backends/profiler/gpu/context/DIRECTED_PROBE_SPEC.md) - Probe protocol
- [STANDALONE_TEST_README.md](STANDALONE_TEST_README.md) - Testing guide

### Design Patterns Used
- **Plugin Pattern**: For extensibility
- **Singleton Pattern**: For plugin registry
- **Factory Pattern**: For context setup
- **Dependency Injection**: For loose coupling
- **Strategy Pattern**: For configuration

### Code Locations
```
New Plugin Infrastructure:
  xla/backends/profiler/gpu/profiler_plugin_interface.{h,cc}
  xla/backends/profiler/gpu/distributed_profiler_plugin.{h,cc}
  
New Utilities:
  xla/pjrt/gpu/distributed_context_setup.{h,cc}
  xla/tsl/distributed_runtime/coordination/network_config_plugin.{h,cc}
  
Modified Core Files (minimal):
  xla/pjrt/gpu/se_gpu_pjrt_client.cc (~20 lines)
  xla/backends/profiler/gpu/rocm_collector.cc (~15 lines)
  xla/tsl/distributed_runtime/coordination/coordination_service_agent.cc (~5 lines)
  xla/backends/profiler/gpu/device_tracer_rocm.cc (~10 lines)
```

---

## 💡 Key Takeaways

1. **Separation of Concerns**: Distributed profiling isolated from core XLA
2. **Minimal Core Changes**: 93% reduction in modifications (647 → 50 lines)
3. **Easy to Maintain**: Update XLA or distributed profiling independently
4. **Configuration Control**: Enable/disable at build-time or runtime
5. **Testable**: Plugins can be tested in isolation
6. **Extensible**: Plugin system supports future additions

---

## 🎯 Next Steps

1. **Now**: Review this summary and related documents
2. **Today**: Set up branch and run baseline tests
3. **This Week**: Start Phase 1 (plugin infrastructure)
4. **Next 4 Weeks**: Complete all phases
5. **After**: Deploy, monitor, and celebrate! 🎉

---

## ❓ Questions?

If you have questions:
- Review detailed docs: REFACTORING_PLAN.md, REFACTORING_ARCHITECTURE.md
- Check implementation guide: REFACTORING_CHECKLIST.md
- Review existing code: `git diff rocm-jaxlib-v0.7.1`
- Ask the team!

---

**Document Status**: Ready for implementation  
**Estimated Effort**: 4 weeks  
**Last Updated**: 2025-12-08  
**Dependencies**: None (can start immediately)

