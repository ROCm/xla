# Distributed Profiling Configuration Guide

Complete guide to configuring distributed profiling in XLA.

---

## 🎯 Quick Start

### Minimal Configuration (Just Enable)
```bash
XLA_ENABLE_DISTRIBUTED_PROFILING=1 python train.py
```

This enables distributed profiling with all default settings.

---

## 📋 Configuration Methods

### Method 1: Environment Variables (Recommended for Quick Tests)

**Pros**: Simple, no extra files, easy to override  
**Cons**: Many variables can be cluttered  
**Best for**: Quick experiments, CI/CD, container overrides

```bash
# Enable and configure
export XLA_ENABLE_DISTRIBUTED_PROFILING=1
export XLA_PROBE_CADENCE_US=800
export XLA_PROBE_WINDOW_S=4
export XLA_PACKET_SPACING_US=100
export XLA_DIST_PROF_OUTPUT_DIR=/tmp/my_prof_data

python train.py
```

### Method 2: JSON Config File (Recommended for Production)

**Pros**: Clean, version-controllable, easy to share  
**Cons**: Extra file to manage  
**Best for**: Production runs, reproducible experiments, team sharing

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

# Use it
XLA_DIST_PROF_CONFIG=dist_prof_config.json python train.py
```

### Method 3: Hybrid (Best of Both Worlds)

**Pros**: Reusable base config + per-run overrides  
**Best for**: Production with occasional tweaks

```bash
# Base config in file, override specific values
XLA_DIST_PROF_CONFIG=dist_prof_config.json \
XLA_PROBE_CADENCE_US=2000 \
python train.py
```

---

## ⚙️ Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Master switch for distributed profiling |
| `probe_cadence_us` | int | `800` | Microseconds between probes (lower = more accurate, higher overhead) |
| `probe_window_s` | int | `4` | Seconds per probe window before computing α/β |
| `packet_spacing_us` | int | `100` | Microseconds between Pt1/Pt2/Pt3 packets |
| `snapshot_period_ms` | int | `100` | Milliseconds between clock snapshot samples |
| `output_dir` | string | `/tmp/xla_dist_prof` | Directory for probe data output (JSONL files) |

### Environment Variable Mapping

| Environment Variable | JSON Field | Type |
|---------------------|------------|------|
| `XLA_ENABLE_DISTRIBUTED_PROFILING` | `enabled` | bool (0/1) |
| `XLA_PROBE_CADENCE_US` | `probe_cadence_us` | int |
| `XLA_PROBE_WINDOW_S` | `probe_window_s` | int |
| `XLA_PACKET_SPACING_US` | `packet_spacing_us` | int |
| `XLA_DIST_PROF_OUTPUT_DIR` | `output_dir` | string |
| `XLA_DIST_PROF_CONFIG` | (file path) | string |

### Network Parameters (Separate)

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NCCL_SOCKET_IFNAME` | `eth0` | Network interface(s) for probing (comma-separated) |

---

## 📊 Recommended Configurations

### Default Configuration (Balanced)
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

**Use case**: Most workloads  
**Overhead**: ~0.5-1%  
**Accuracy**: ±50-100μs

### High-Accuracy Configuration
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

**Use case**: Latency-sensitive analysis, debugging timing issues  
**Overhead**: ~1-2%  
**Accuracy**: ±20-50μs

### Low-Overhead Configuration
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

**Use case**: Production workloads where overhead must be minimal  
**Overhead**: ~0.1-0.3%  
**Accuracy**: ±200-500μs

### Debug Configuration
```json
{
  "enabled": true,
  "probe_cadence_us": 1000,
  "probe_window_s": 2,
  "packet_spacing_us": 100,
  "snapshot_period_ms": 100,
  "output_dir": "/tmp/xla_dist_prof_debug"
}
```

**Use case**: Development, testing, debugging  
**Overhead**: Don't care  
**Accuracy**: Good enough

---

## 🔧 Tuning Guide

### Probe Cadence (`probe_cadence_us`)

**What it controls**: How often probe packets are sent

```
Lower value (e.g., 200μs):
  ✅ More data points → better α/β estimation
  ✅ More responsive to clock drift changes
  ❌ Higher network overhead
  ❌ More CPU overhead

Higher value (e.g., 5000μs):
  ✅ Lower overhead
  ❌ Fewer data points → worse estimation
  ❌ Slower to detect drift changes
```

**Recommended range**: 400-2000μs  
**Sweet spot**: 800μs (balance accuracy and overhead)

### Probe Window (`probe_window_s`)

**What it controls**: How long we collect probes before computing α/β

```
Shorter window (e.g., 2s):
  ✅ More frequent α/β updates
  ✅ Better for variable clock drift
  ❌ Fewer samples per window → higher variance

Longer window (e.g., 10s):
  ✅ More samples → more stable α/β estimates
  ✅ Better for consistent clocks
  ❌ Slower to adapt to drift changes
```

**Recommended range**: 2-10s  
**Sweet spot**: 4s (good stability, decent responsiveness)

### Packet Spacing (`packet_spacing_us`)

**What it controls**: Delay between Pt1→Pt2→Pt3 packets in a probe

```
Lower value (e.g., 50μs):
  ✅ Faster probe completion
  ✅ More samples per second
  ❌ Potential for network congestion

Higher value (e.g., 200μs):
  ✅ Less bursty network traffic
  ❌ Slower probe completion
```

**Recommended range**: 50-200μs  
**Sweet spot**: 100μs (avoid burstiness)

### Snapshot Period (`snapshot_period_ms`)

**What it controls**: How often we sample system clock vs GPU clock

```
Lower value (e.g., 50ms):
  ✅ More snapshot pairs for trace alignment
  ✅ Better for short traces
  ❌ Slightly higher overhead

Higher value (e.g., 500ms):
  ✅ Lower overhead
  ❌ Fewer snapshots for alignment
```

**Recommended range**: 50-200ms  
**Sweet spot**: 100ms (good coverage)

---

## 🎭 Configuration Precedence

Configuration sources are merged with the following precedence (highest to lowest):

1. **Individual environment variables** (highest)
2. **JSON config file** (via `XLA_DIST_PROF_CONFIG`)
3. **Defaults** (lowest)

### Example: Precedence in Action

**Config file** (`base_config.json`):
```json
{
  "enabled": true,
  "probe_cadence_us": 800,
  "probe_window_s": 4
}
```

**Command**:
```bash
XLA_DIST_PROF_CONFIG=base_config.json \
XLA_PROBE_CADENCE_US=2000 \
python train.py
```

**Result**:
- `enabled`: `true` (from file)
- `probe_cadence_us`: `2000` (from env var - overrides file!)
- `probe_window_s`: `4` (from file)
- `packet_spacing_us`: `100` (default)
- `output_dir`: `/tmp/xla_dist_prof` (default)

---

## 🚨 Common Pitfalls

### Pitfall 1: Typos in Environment Variables
```bash
# ❌ Wrong (typo in variable name)
export XLA_PROBE_CADANCE_US=1000  # Note: CADANCE vs CADENCE

# ✅ Correct
export XLA_PROBE_CADENCE_US=1000
```

**Solution**: Check logs for "Distributed profiling config:" to see loaded values

### Pitfall 2: Config File Not Found
```bash
# ❌ Wrong (relative path may not work)
XLA_DIST_PROF_CONFIG=config.json python train.py

# ✅ Correct (use absolute path)
XLA_DIST_PROF_CONFIG=/home/user/configs/config.json python train.py

# ✅ Or (use path relative to PWD)
XLA_DIST_PROF_CONFIG=./config.json python train.py
```

### Pitfall 3: Forgetting to Enable
```bash
# ❌ Wrong (has config but enabled=false)
cat > config.json <<EOF
{
  "enabled": false,
  "probe_cadence_us": 800
}
EOF
XLA_DIST_PROF_CONFIG=config.json python train.py  # Won't run!

# ✅ Correct
cat > config.json <<EOF
{
  "enabled": true,
  "probe_cadence_us": 800
}
EOF
```

### Pitfall 4: Invalid JSON
```json
// ❌ Wrong (trailing comma)
{
  "enabled": true,
  "probe_cadence_us": 800,
}

// ✅ Correct
{
  "enabled": true,
  "probe_cadence_us": 800
}
```

**Solution**: Validate JSON with `jq`:
```bash
jq . config.json  # Should pretty-print if valid
```

---

## 📝 Configuration Validation

Add logging to verify loaded configuration:

```cpp
// In DistributedProfilerPlugin::Initialize()
LOG(INFO) << "=== Distributed Profiling Configuration ===";
LOG(INFO) << "  enabled: " << (config_.enabled ? "YES" : "NO");
if (config_.enabled) {
  LOG(INFO) << "  probe_cadence_us: " << config_.probe_cadence_us;
  LOG(INFO) << "  probe_window_s: " << config_.probe_window_s;
  LOG(INFO) << "  packet_spacing_us: " << config_.packet_spacing_us;
  LOG(INFO) << "  snapshot_period_ms: " << config_.snapshot_period_ms;
  LOG(INFO) << "  output_dir: " << config_.output_dir;
}
LOG(INFO) << "===========================================";
```

Check logs after running to confirm values:
```bash
python train.py 2>&1 | grep "Distributed Profiling Configuration" -A 10
```

---

## 🧪 Testing Configuration

### Test 1: Verify Enable/Disable
```bash
# Should NOT see profiling logs
XLA_ENABLE_DISTRIBUTED_PROFILING=0 python train.py 2>&1 | grep "Distributed Profiling"

# Should see profiling logs
XLA_ENABLE_DISTRIBUTED_PROFILING=1 python train.py 2>&1 | grep "Distributed Profiling"
```

### Test 2: Verify Config File Loading
```bash
cat > test_config.json <<EOF
{
  "enabled": true,
  "probe_cadence_us": 9999
}
EOF

# Should see "probe_cadence_us: 9999" in logs
XLA_DIST_PROF_CONFIG=test_config.json python train.py 2>&1 | grep "probe_cadence_us"
```

### Test 3: Verify Precedence
```bash
cat > test_config.json <<EOF
{
  "enabled": true,
  "probe_cadence_us": 1000
}
EOF

# Should see "probe_cadence_us: 8888" (env var overrides file)
XLA_DIST_PROF_CONFIG=test_config.json \
XLA_PROBE_CADENCE_US=8888 \
python train.py 2>&1 | grep "probe_cadence_us"
```

---

## 📦 Example Configs for Different Scenarios

### For JAX Training (Large Models)
```json
{
  "enabled": true,
  "probe_cadence_us": 1000,
  "probe_window_s": 5,
  "packet_spacing_us": 100,
  "snapshot_period_ms": 100,
  "output_dir": "/data/profiling/training_run_001"
}
```

### For PyTorch/XLA Inference
```json
{
  "enabled": true,
  "probe_cadence_us": 500,
  "probe_window_s": 3,
  "packet_spacing_us": 75,
  "snapshot_period_ms": 75,
  "output_dir": "/data/profiling/inference"
}
```

### For Benchmarking (Minimal Overhead)
```json
{
  "enabled": true,
  "probe_cadence_us": 5000,
  "probe_window_s": 20,
  "packet_spacing_us": 200,
  "snapshot_period_ms": 500,
  "output_dir": "/tmp/benchmark_prof"
}
```

### For Debugging Synchronization Issues
```json
{
  "enabled": true,
  "probe_cadence_us": 200,
  "probe_window_s": 1,
  "packet_spacing_us": 50,
  "snapshot_period_ms": 25,
  "output_dir": "/tmp/debug_sync"
}
```

---

## 🔍 Monitoring Configuration Effectiveness

After a run, check probe data quality:

```bash
# Check how many probe windows were completed
grep "Window completed" /tmp/xla_dist_prof/*.jsonl | wc -l

# Check alpha/beta values over time
jq -r 'select(.type=="window_stats") | "\(.window_id) \(.alpha) \(.beta)"' \
  /tmp/xla_dist_prof/*.jsonl

# Check for probe failures
jq -r 'select(.status=="failed")' /tmp/xla_dist_prof/*.jsonl | wc -l
```

**Good indicators**:
- Alpha close to 1.0 (within 0.0001)
- Beta relatively stable (not jumping around)
- Low probe failure rate (< 1%)

**Bad indicators**:
- Alpha far from 1.0 (clocks drifting fast)
- Beta changing rapidly (unstable)
- High probe failure rate (network issues)

---

## 📚 Related Documentation

- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Refactoring overview
- [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - Implementation details
- [DIRECTED_PROBE_SPEC.md](xla/backends/profiler/gpu/context/DIRECTED_PROBE_SPEC.md) - Probe protocol
- [WINDOW_MANAGER_IMPLEMENTATION.md](WINDOW_MANAGER_IMPLEMENTATION.md) - Window statistics

---

**Document Status**: Draft for implementation  
**Last Updated**: 2025-12-08  
**Next**: Implement DistributedProfilerConfig class

