# Documentation Index

This document helps you navigate the distributed profiling documentation.

---

## 📚 **Core Documentation**

### Essential Reading

1. **`README.md`** - Project overview and quick start
2. **`docs/distributed_profiling/CRITICAL_PATH_ANALYSIS.md`** - Performance impact analysis
3. **`docs/distributed_profiling/WINDOW_MANAGER_IMPLEMENTATION.md`** - Window-based statistics export
4. **`docs/distributed_profiling/HANDSHAKE_IMPLEMENTATION.md`** - Connection establishment protocol
5. **`docs/distributed_profiling/STANDALONE_TEST_README.md`** - Testing guide
6. **`docs/distributed_profiling/master_alpha_sync.md`** - Master-controlled alpha/beta synchronization plan (socket Variant B + Gloo roadmap)
7. **`docs/distributed_profiling/snapshot_pair.md`** - Snapshot sampling/export plan (distributed-only)
8. **`docs/distributed_profiling/trace_algo.md`** - Timeline alignment algorithm (R↔S conversion)
9. **`docs/distributed_profiling/trace_combiner.md`** - Multi-node trace merge and metadata spec

### Refactoring Documentation

10. **`docs/distributed_profiling/REFACTORING_SUMMARY.md`** - Quick overview of plugin-based refactoring
11. **`docs/distributed_profiling/REFACTORING_PLAN.md`** - Detailed refactoring strategy and analysis
12. **`docs/distributed_profiling/REFACTORING_ARCHITECTURE.md`** - Before/after architecture diagrams
13. **`docs/distributed_profiling/REFACTORING_CHECKLIST.md`** - Step-by-step implementation guide
14. **`docs/distributed_profiling/CONFIGURATION_GUIDE.md`** - Complete configuration reference (env vars, JSON, tuning)

---

## 📁 **Technical Specifications** (`xla/backends/profiler/gpu/context/`)

### Implementation Details

1. **`DIRECTED_PROBE_SPEC.md`** 
   - Complete specification for directed network probing
   - Packet protocol (Pt1/Pt2/Pt3, Pr1/Pr2/Pr3)
   - SVM-based clock skew estimation
   - **Start here for technical details**

2. **`OPTION_B_IMPLEMENTATION.md`**
   - (2xO + 2xI) socket architecture
   - Port assignment strategy
   - JSONL export format
   - **Reference for socket implementation**

3. **`CENTRALIZED_PORT_ASSIGNMENT.md`**
   - Master node port assignment
   - KV store protocol for port distribution
   - Port conflict resolution

4. **Other context docs** (historical/design notes)

---

## 🎯 **Quick Reference by Task**

### "I want to understand the system architecture"
→ Start with `DIRECTED_PROBE_SPEC.md` Section 1-2

### "I want to understand performance impact"
→ Read `docs/distributed_profiling/CRITICAL_PATH_ANALYSIS.md`

### "I want to test the network probing"
→ Follow `docs/distributed_profiling/STANDALONE_TEST_README.md`

### "I want to understand the window-based export"
→ Read `docs/distributed_profiling/WINDOW_MANAGER_IMPLEMENTATION.md`

### "I want to understand socket setup"
→ Read `OPTION_B_IMPLEMENTATION.md` + `CENTRALIZED_PORT_ASSIGNMENT.md`

### "I want to modify the probe protocol"
→ See `DIRECTED_PROBE_SPEC.md` Phase 1-3

### "I want to analyze the JSONL output"
→ See `docs/distributed_profiling/WINDOW_MANAGER_IMPLEMENTATION.md` Section on "Analysis Tools"

### "I want to understand master-based synchronization"
→ Read `docs/distributed_profiling/master_alpha_sync.md` (Gloo barriers, cross-node aggregation)

### "I want to know how snapshot pairs are captured"
→ Read `docs/distributed_profiling/snapshot_pair.md`

### "I want to align per-node traces onto a global timeline"
→ Read `docs/distributed_profiling/trace_algo.md`

### "I want to merge multiple traces into one view"
→ Read `docs/distributed_profiling/trace_combiner.md`

### "I want to minimize changes to original XLA files"
→ Read `docs/distributed_profiling/REFACTORING_SUMMARY.md` (start here)
→ Then `docs/distributed_profiling/REFACTORING_PLAN.md` for detailed strategy
→ Follow `docs/distributed_profiling/REFACTORING_CHECKLIST.md` for step-by-step guide

### "I want to configure distributed profiling"
→ Read `docs/distributed_profiling/CONFIGURATION_GUIDE.md` (runtime config, JSON files, tuning)

---

## 📊 **Key Implementation Files**

| File | Purpose | Lines |
|------|---------|-------|
| `network_probe.h` | NetworkProbeManager class definition | ~220 |
| `network_probe.cc` | Core probing logic, WindowManager | ~1400 |
| `distributed_timestamp_sync.h/cc` | High-level synchronizer | ~200 |
| `probe_utils.h/cc` | Probe pair filtering and conversion | ~80 |
| `svm_wrapper.h/cc` | SVM training for α/β estimation | ~170 |
| `rocm_collector.cc` | Integration with ROCm profiler | ~600 |
| `se_gpu_pjrt_client.cc` | Graph generation and port assignment | ~800 |

---

## 🔧 **Configuration**

All configuration is in `DistributedProfilerContext` (defined in `xla/backends/profiler/gpu/distributed_profiler_context.h`):

```cpp
struct DistributedProfilerContext {
  // Node identity
  int node_id = -1;                     // This node's ID (0 to N-1)
  int num_nodes = 1;                    // Total number of nodes
  std::vector<std::string> node_addresses;  // IP addresses
  std::vector<int> neighbors;           // Out-neighbor IDs
  std::vector<int> in_neighbors;        // In-neighbor IDs
  
  // Probing parameters
  uint64_t probe_cadence_us = 800;      // 800µs between probes
  uint64_t probe_window_s = 4;          // 4-second windows
  
  // Output configuration
  std::string output_dir = "/tmp/xla_dist_prof";  // Export directory
  
  // Port assignments (centrally allocated by master)
  std::map<std::string, std::pair<uint16_t, uint16_t>> edge_ports;
};
```

**Environment Variables:**
- `XLA_DIST_PROF_OUTPUT_DIR`: Override output directory
- `XLA_PROBE_CADENCE_US`: Override probe rate (µs)
- `XLA_PROBE_WINDOW_S`: Override window duration (s)
- `XLA_ENABLE_PROBE_EXPORT`: Enable/disable probe data export (0/1)

---

## 🧪 **Testing**

### Unit Tests
- `network_probe_test.cc` (if exists)
- `probe_utils_test.cc` (if exists)

### Integration Tests
- `network_probe_standalone_test.cc` - 2-node standalone test
- See `docs/distributed_profiling/STANDALONE_TEST_README.md` for usage

### SkyPilot Deployment
- `skypilot/tasks/standalone.yaml` - Multi-node cluster test

---

## 📝 **Document Status**

| Document | Status | Last Updated |
|----------|--------|--------------|
| `DIRECTED_PROBE_SPEC.md` | ✅ Current | Final implementation |
| `OPTION_B_IMPLEMENTATION.md` | ✅ Current | Final socket architecture |
| `CENTRALIZED_PORT_ASSIGNMENT.md` | ✅ Current | Final port strategy |
| `docs/distributed_profiling/CRITICAL_PATH_ANALYSIS.md` | ✅ Current | Performance evaluation |
| `docs/distributed_profiling/WINDOW_MANAGER_IMPLEMENTATION.md` | ✅ Current | Barrier-based with absl::Barrier |
| `docs/distributed_profiling/HANDSHAKE_IMPLEMENTATION.md` | ✅ Current | Connection protocol |
| `docs/distributed_profiling/STANDALONE_TEST_README.md` | ✅ Current | Testing guide |
| `docs/distributed_profiling/master_alpha_sync.md` | 📋 Planning | Master-based synchronization design |
| `docs/distributed_profiling/snapshot_pair.md` | ✅ Implemented | Distributed snapshot capture (embedded plane) |
| `docs/distributed_profiling/trace_algo.md` | ✅ Implemented | Timeline alignment algorithm (Python) |
| `docs/distributed_profiling/trace_combiner.md` | ✅ Implemented | Multi-node trace merging (Python) |

---

## 🗑️ **Removed Documentation**

The following obsolete docs have been cleaned up:
- `TIMESTAMP_EXCHANGE_DESIGN.md` (replaced by DIRECTED_PROBE_SPEC.md)
- `ARCHITECTURE_DIAGRAM.md` (integrated into spec docs)
- `DISTRIBUTED_PROFILING_DESIGN.md` (early design, superseded)
- `SOCKET_TIMESTAMP_SYNC_SUMMARY.md` (interim summary)
- `IMPLEMENTATION_REVIEW.md` (review notes)
- `SOLUTION_SUMMARY.md` (generic summary)
- `FAST_ITERATION_SUMMARY.md` (build optimization notes)
- `FAST_DEBUG_GUIDE.md` (temporary debugging guide)
- `CONTRIBUTING.md` (empty stub)

---

## 💡 **Tips**

1. **Start with specs** - Read `DIRECTED_PROBE_SPEC.md` first
2. **Check performance** - `docs/distributed_profiling/CRITICAL_PATH_ANALYSIS.md` shows it's safe
3. **Test early** - Use `docs/distributed_profiling/STANDALONE_TEST_README.md` to validate
4. **Analyze output** - JSONL format is easy to parse with `jq` or Python

---

## 📞 **Contact / Support**

For questions about:
- **Architecture**: See `DIRECTED_PROBE_SPEC.md`
- **Performance**: See `docs/distributed_profiling/CRITICAL_PATH_ANALYSIS.md`
- **Testing**: See `docs/distributed_profiling/STANDALONE_TEST_README.md`
- **Implementation**: See source code comments in `network_probe.cc`
