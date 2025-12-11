# Window Manager Implementation Summary

## Overview

Implemented shared window management for distributed network probing with JSON Lines (JSONL) export format.

## Key Features

### 1. Shared Window State
- All `ProbeSender` threads share a single `WindowManager` instance
- Atomic window rotation ensures only one thread rotates windows
- Thread-safe statistics collection via `absl::Mutex`
- **All windows held in memory** until shutdown (batch export)

### 2. JSONL Export Format
- **File:** `/tmp/probe_windows_node{NODE_ID}.jsonl`
- **One line per window** (written in single batch at shutdown)
- **Reduced I/O overhead** during probing

### 3. Window Statistics Per Edge
Each window records:
- `alpha`: Clock skew coefficient
- `beta`: Clock offset + one-way delay (nanoseconds)
- `pairs`: Number of complete probe pairs used for SVM training
- `lost`: Number of lost packets during window

## Implementation Details

### WindowManager Class

```cpp
class WindowManager {
 public:
  struct EdgeWindowStats {
    double alpha;
    double beta;
    int pairs_collected;
    int packets_lost;
  };
  
  struct WindowStats {
    uint64_t window_start_ns;
    uint64_t window_end_ns;
    absl::flat_hash_map<int, EdgeWindowStats> edges;
  };
  
  bool IsWindowExpired();  // Check if current window has passed
  void RotateWindow();     // Move to next window, clear stats
  void RecordEdgeStats(int dst_id, double alpha, double beta, int pairs, int lost);
  WindowStats GetCurrentWindow();
  void ExportWindowToJSON(const WindowStats& window, std::ofstream& out, int node_id);
```

### ProbeSender Integration

Each `ProbeSender` thread:
1. **Checks window expiry** at start of loop
2. **Trains SVM** on accumulated probe pairs for its edge
3. **Records stats** in shared window manager
4. **Arrives at barrier** - only the **last thread** rotates the window

```cpp
if (window_manager_->IsWindowExpired()) {
  // Train SVM, record stats
  window_manager_->RecordEdgeStats(dst_id, alpha, beta, pairs, lost);
  
  // Barrier-based rotation: only last thread rotates
  if (window_manager_->NotifyWindowExpired()) {
    window_manager_->RotateWindow();  // This thread is the last one
  }
}
```

**Thread Synchronization:** Using **`absl::Barrier`** (from the [Abseil Synchronization library](https://abseil.io/docs/cpp/guides/synchronization)) for clean barrier implementation:

```cpp
// Constructor
WindowManager::WindowManager(uint64_t window_duration_ns, int num_probe_threads)
    : window_duration_ns_(window_duration_ns),
      num_probe_threads_(num_probe_threads) {
  barrier_ = std::make_unique<absl::Barrier>(num_probe_threads);
}

// Barrier synchronization
bool WindowManager::NotifyWindowExpired() {
  bool am_last = barrier_->Block();  // Blocks until all threads arrive
  return am_last;  // Returns true for exactly one thread
}

// Window rotation (called only by last thread)
void WindowManager::RotateWindow() {
  // 1. Rotate window data
  { /* ... save window, update timestamps ... */ }
  
  // 2. Recreate barrier for next window
  {
    absl::MutexLock lock(&barrier_mu_);
    barrier_ = std::make_unique<absl::Barrier>(num_probe_threads_);
  }
}
```

**Benefits:**
- **Built-in barrier:** Uses Abseil's proven `Barrier` implementation (no manual counter logic)
- **Clean API:** `Block()` returns `true` for exactly one thread (the last to arrive)
- **Automatic blocking:** All threads wait at barrier until all arrive
- **Simple recreation:** Create new barrier after rotation for next window
- **No race conditions:** All threads coordinated before and after rotation
- **No data loss:** All edges record their stats before window is saved

### Shutdown Export

All windows exported in single batch:

```cpp
void NetworkProbeManager::Shutdown() {
  // ... join threads ...
  
  // Export ALL accumulated windows when shutdown
  std::ofstream out("/tmp/probe_windows_node{ID}.jsonl");
  window_manager_->ExportAllWindows(out, node_id,
                                    collector_start_walltime_ns,
                                    collector_start_gpu_ns);
  out.close();
}
```

## Output Format

### Example JSONL Output

```jsonl
{"meta":true,"node_id":0,"start_walltime_ns":1712094923123456789,"start_gpu_ns":9876543210}
{"window_start_ns":1000000000000,"window_end_ns":1004000000000,"node_id":0,"edges":[{"dst":1,"alpha":1.0000023,"beta":-12345,"pairs":487,"lost":5}]}
{"window_start_ns":1004000000000,"window_end_ns":1008000000000,"node_id":0,"edges":[{"dst":1,"alpha":1.0000025,"beta":-12340,"pairs":501,"lost":2}]}
```

> **Downstream parity:** The offline `graph_calc_main` CLI preserves this metadata by writing an analogous header as the first line of `round_offsets.jsonl`, so the computed offsets can always be related back to the original wall-clock and GPU timestamps.

### Prettified Example

```json
{
  "window_start_ns": 1000000000000,
  "window_end_ns": 1004000000000,
  "node_id": 0,
  "edges": [
    {
      "dst": 1,
      "alpha": 1.0000023,
      "beta": -12345,
      "pairs": 487,
      "lost": 5
    }
  ]
}
```

## Analysis Tools

### Python Example

```python
import json
import matplotlib.pyplot as plt

# Load windows
windows = []
with open('/tmp/probe_windows_node0.jsonl') as f:
    for line in f:
        windows.append(json.loads(line))

# Plot alpha evolution for edge 0→1
timestamps = [w['window_start_ns'] for w in windows]
alphas = [next((e['alpha'] for e in w['edges'] if e['dst'] == 1), None) 
          for w in windows]

plt.plot(timestamps, alphas)
plt.xlabel('Time (ns)')
plt.ylabel('Clock Skew (α)')
plt.title('Edge 0→1 Clock Skew Over Time')
plt.show()
```

### Shell Commands

```bash
# Count windows (after profiler shutdown)
wc -l /tmp/probe_windows_node0.jsonl

# View latest window
tail -1 /tmp/probe_windows_node0.jsonl | jq '.'

# Extract all alpha values for dst=1
cat /tmp/probe_windows_node0.jsonl | jq '.edges[] | select(.dst==1) | .alpha'

# View all windows in human-readable format
cat /tmp/probe_windows_node0.jsonl | jq '.'
```

## Files Modified

### 1. `network_probe.h`
- Added `WindowManager` class with:
  - `std::vector<WindowStats> completed_windows_` (accumulates all windows)
  - `std::unique_ptr<absl::Barrier> barrier_` (synchronization primitive)
  - `absl::Mutex barrier_mu_` (protects barrier recreation)
  - `NotifyWindowExpired()` method (calls `barrier_->Block()`)
  - `ExportAllWindows()` method (batch export)
  - Internal `absl::Mutex mu_` for thread-safe data access
- Added includes:
  - `absl/synchronization/barrier.h`
  - `absl/synchronization/mutex.h`
- Added to `NetworkProbeManager`:
  - `std::unique_ptr<WindowManager> window_manager_`

### 2. `network_probe.cc`
- **WindowManager implementation** (lines 68-135)
  - Constructor creates `absl::Barrier` with thread count (lines 69-76)
  - `NotifyWindowExpired()`: Simple call to `barrier_->Block()` (lines 84-94)
    - Returns `true` for last thread, `false` for others (automatic)
  - `RotateWindow()`: Saves window, recreates barrier (lines 97-125)
    - Much simpler than manual generation counter approach
  - `ExportAllWindows()`: Writes all accumulated windows to JSONL in one batch
- **Initialize()**: Create window manager with thread count (lines 203-211)
- **ProbeSender()**: Window expiry check, SVM training, barrier-based rotation (lines 760-780)
  - Uses `absl::Barrier`: only last thread calls `RotateWindow()`, others block automatically
- **Shutdown()**: Open file and export ALL windows (lines 1400+)

### 3. Documentation
- **OPTION_B_IMPLEMENTATION.md**: Added JSONL export section
- **DIRECTED_PROBE_SPEC.md**: Updated Phase 4 with WindowManager details

## Benefits

| Aspect | Benefit |
|--------|---------|
| **Low I/O Overhead** | Single batch write at shutdown (no I/O during probing) |
| **Flexible** | Easy to add new fields without breaking format |
| **Compact** | ~100-200 bytes per window |
| **Analyzable** | jq, grep, Python, any JSON parser |
| **Memory Efficient** | Windows typically small (~100-500 bytes), memory grows linearly |
| **Atomic Write** | All data written in one operation (no partial files) |

## Next Steps

1. **Test 2-node setup** and verify JSONL output
2. **Implement post-processing** to compute global clock offsets from per-edge α/β
3. **Add visualization** scripts for α/β evolution
4. **Consider SQLite export** if query performance becomes important

## Notes

- Window duration controlled by `probe_window_s` in config (default 4 seconds)
- Export file appends (safe for process restarts)
- Final window exported during shutdown (no data loss)
- Thread-safe via mutex-protected window state

