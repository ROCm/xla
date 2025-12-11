# Standalone Network Probe Test

## Quick Start

### 1. Build the test (30 seconds - 2 minutes)
```bash
./build_standalone_test.sh
```

### 2. Run the test (multiple options)

**Option A: Automatic (uses tmux)**
```bash
./run_2node_test.sh
```

**Option B: Quick debug cycle**
```bash
./debug_probe.sh   # Build + run + show results in one command
```

**Option C: Manual 2-terminal setup**
```bash
# Terminal 1 (Node 0 - prober)
bazel-bin/xla/backends/profiler/gpu/network_probe_standalone_test \
  --node_id=0 --num_nodes=2 --duration_sec=10 --vmodule=network_probe=2

# Terminal 2 (Node 1 - listener)
bazel-bin/xla/backends/profiler/gpu/network_probe_standalone_test \
  --node_id=1 --num_nodes=2 --duration_sec=10 --vmodule=network_probe=2
```

---

## What This Tests

This standalone test validates the **complete network probe system**:

✅ Port assignment and socket creation  
✅ Listener threads receiving probes  
✅ Prober threads sending probes  
✅ Timestamp collection (TX/RX)  
✅ Packet loss handling  
✅ SVM training and α/β calculation  
✅ CSV export  

**WITHOUT** requiring:
- Full XLA build
- JAX/Python
- Distributed PJRT client
- Real GPU devices

---

## Test Configuration

### Default 2-Node Setup

**Node 0 (Prober):**
- Sends probes to Node 1
- Listens on `127.0.0.1:12345`
- Probe socket sends to `127.0.0.1:20100` (Node 1's listen port)
- Response socket binds to port `20000`

**Node 1 (Listener):**
- Receives probes from Node 0
- Listens on `127.0.0.1:12346`
- Listen socket binds to port `20100`
- Response socket sends to `127.0.0.1:20000` (Node 0's response port)

**Graph:** `0 → 1` (directed edge, Node 0 probes Node 1)

### Hardcoded Edge Ports
```cpp
config.edge_ports["probe_edge:0->1"] = {20100, 20000};
//                                       ^       ^
//                                       |       └─ Node 0 receives responses
//                                       └───────── Node 1 listens for probes
```

---

## Command Line Options

```bash
--node_id=N              # Node ID (0 or 1)
--num_nodes=N            # Total nodes (default: 2)
--duration_sec=N         # Test duration (default: 10)
--probe_cadence_us=N     # Probe interval (default: 800)
--probe_window_s=N       # SVM window size (default: 4)
--node0_addr=ADDR        # Node 0 address (default: 127.0.0.1:12345)
--node1_addr=ADDR        # Node 1 address (default: 127.0.0.1:12346)
--vmodule=MODULE=LEVEL   # Verbose logging (e.g., network_probe=2)
```

### Examples

**Test with high verbosity:**
```bash
./debug_probe.sh --vmodule=network_probe=2
```

**Test for 30 seconds:**
```bash
bazel-bin/.../network_probe_standalone_test --node_id=0 --duration_sec=30
```

**Test across real network:**
```bash
# On machine A (192.168.1.10):
./network_probe_standalone_test --node_id=0 \
  --node0_addr=192.168.1.10:12345 \
  --node1_addr=192.168.1.11:12345

# On machine B (192.168.1.11):
./network_probe_standalone_test --node_id=1 \
  --node0_addr=192.168.1.10:12345 \
  --node1_addr=192.168.1.11:12345
```

---

## Expected Output

### During Test
```
I0000 Starting standalone network probe test
I0000 Node ID: 0 / 2
I0000 === Test Configuration ===
I0000 Node ID: 0
I0000 OUT-neighbors: 1
I0000 IN-neighbors: 
I0000 Edge ports:
I0000   probe_edge:0->1: listen=20100, response=20000
I0000 ==========================
I0000 
I0000 Initializing NetworkProbeManager...
I0000 SetupSockets: node=0 out_neighbors=1 in_neighbors=0 edge_ports=1
I0000 Created OUT sockets for neighbor 1 probe_sock=3 probe_response_sock=4
I0000 ✅ NetworkProbeManager initialized
I0000 
I0000 Starting probe threads...
I0000 ✅ Probe threads started
I0000 
I0000 🔬 Running test for 10 seconds...
I0000 Probe thread started for OUT-neighbor 1
I0000 Test progress: 5/10 seconds
I0000 Edge 0->1: α=1.000023, β=-12345 ns (487 pairs)
I0000 Test progress: 10/10 seconds
I0000 
I0000 📊 Exporting data...
I0000 Exported probe data to /tmp/alpha_beta_node0.csv
I0000 ✅ Data exported
```

### Results File
```bash
$ cat /tmp/alpha_beta_node0.csv
src,dst,alpha,beta_ns,num_pairs,lost_packets
0,1,1.000023,-12345,487,14
```

**Interpretation:**
- `alpha ≈ 1.0`: Clocks running at same speed (good!)
- `beta`: Clock offset + network delay (in nanoseconds)
- `num_pairs`: Number of successful probe pairs collected
- `lost_packets`: Packets lost due to timeout

---

## Debugging Common Issues

### Issue: "No port assignment found for OUT-edge"
**Cause:** Edge not in `config.edge_ports`  
**Fix:** Check that edge key matches: `"probe_edge:SRC->DST"`

### Issue: "Failed to bind listen socket to port 20100"
**Cause:** Port already in use or permission denied  
**Fix:**
```bash
# Check if port is in use
sudo lsof -i :20100

# Kill conflicting process or change port in test
```

### Issue: "num_pairs=0, lost_packets=large"
**Causes:**
1. Nodes not started simultaneously (listener not ready)
2. Firewall blocking UDP
3. Wrong address configuration

**Debugging:**
```bash
# Run with verbose logging
--vmodule=network_probe=2

# Check for "sendmsg failed" or "recvmsg failed" messages
# Check errno output in logs
```

### Issue: Build takes too long
**Fix:** This test only rebuilds `network_probe.cc` and dependencies
- First build: 1-2 minutes
- Incremental: 10-30 seconds

---

## Development Workflow

### Fast Iteration (recommended)

1. **Edit** `network_probe.cc`

2. **Build + Run + See results:**
   ```bash
   ./debug_probe.sh
   ```

3. **Repeat** - no full XLA rebuild needed!

### Time Comparison

| Method | Time | What it tests |
|--------|------|---------------|
| Standalone test | 30 sec build + 10 sec run | Full network_probe.cc logic |
| Full XLA build | 10-30 min build + setup | Everything (overkill for debugging) |

**Speed-up: ~20-40x faster** 🚀

---

## Testing Specific Features

### Test Port Assignment
```cpp
// Edit network_probe_standalone_test.cc
config.edge_ports["probe_edge:0->1"] = {20100, 20000};  // Try different ports
config.edge_ports["probe_edge:1->0"] = {20001, 20101};  // Add bidirectional
```

### Test Packet Loss
```bash
# Use tc (traffic control) to simulate loss
sudo tc qdisc add dev lo root netem loss 10%

# Run test
./debug_probe.sh

# Clean up
sudo tc qdisc del dev lo root
```

### Test with Real Network
Update addresses and run on separate machines.

---

## Extending the Test

### Add 3rd Node

Edit `network_probe_standalone_test.cc`:

```cpp
// Update CreateTestConfig():
config.node_addresses = {"127.0.0.1:12345", "127.0.0.1:12346", "127.0.0.1:12347"};

if (node_id == 0) {
  config.neighbors = {1};
} else if (node_id == 1) {
  config.neighbors = {2};
  config.in_neighbors = {0};
} else if (node_id == 2) {
  config.in_neighbors = {1};
}

// Add edge ports
config.edge_ports["probe_edge:0->1"] = {20100, 20000};
config.edge_ports["probe_edge:1->2"] = {20200, 20101};
```

### Test Bidirectional Graph

```cpp
config.edge_ports["probe_edge:0->1"] = {20100, 20000};
config.edge_ports["probe_edge:1->0"] = {20001, 20101};

if (node_id == 0) {
  config.neighbors = {1};
  config.in_neighbors = {1};
} else {
  config.neighbors = {0};
  config.in_neighbors = {0};
}
```

---

## Next Steps

Once standalone test passes:
1. ✅ Network probe logic is correct
2. ⏭️ Integrate with full PJRT client
3. ⏭️ Test in real distributed environment

---

## Troubleshooting

### Build Errors

**"Cannot find network_probe.h"**
- Make sure you're in the XLA root directory
- Check that `xla/backends/profiler/gpu/network_probe.h` exists

**Linker errors**
- Clean and rebuild: `bazel clean && ./build_standalone_test.sh`

### Runtime Errors

**Segfault in NetworkProbeManager**
- Run with debugger: `gdb --args bazel-bin/.../network_probe_standalone_test --node_id=0`
- Check that `config.edge_ports` is populated

**No CSV output**
- Check write permissions: `ls -la /tmp/`
- Verify test ran long enough (> probe_window_s)

---

## Summary

This standalone test lets you:
- ✅ Test network probe logic in isolation
- ✅ Iterate 20-40x faster than full XLA builds
- ✅ Debug socket/port issues easily
- ✅ Validate centralized port assignment
- ✅ See α/β results immediately

**Perfect for rapid development!** 🎯





