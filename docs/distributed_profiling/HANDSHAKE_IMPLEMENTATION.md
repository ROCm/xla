# Network Probe Handshake Implementation

## Overview
Implemented a SYN/ACK handshake protocol to synchronize prober and listener nodes before high-frequency probing begins. This prevents stale packet issues at startup.

## Changes Made

### 1. Protocol Extension (`network_probe.h`)
Added two new message types to `ProbeMessageType` enum:
- `kSyn = 7`: Handshake SYN packet (prober → listener)
- `kAck = 8`: Handshake ACK packet (listener → prober)

### 2. Handshake Method (`network_probe.h` + `network_probe.cc`)
```cpp
bool PerformHandshake(int neighbor_id, bool is_prober);
```

**Prober Mode (`is_prober=true`):**
1. Sends SYN to listener
2. Waits for ACK response
3. Retries up to 5 times with 200ms delay
4. Returns `true` if ACK received, `false` otherwise

**Listener Mode (`is_prober=false`):**
1. Waits for SYN from prober
2. Sends ACK back
3. Retries up to 5 times with 200ms delay
4. Returns `true` if SYN received and ACK sent, `false` otherwise

### 3. Integration

**In `ListenerLoop()`:**
```cpp
// Perform handshake before starting probe loop
if (!PerformHandshake(src_neighbor_id, false)) {
  LOG(ERROR) << "Handshake failed, aborting listener";
  return;
}
LOG(INFO) << "Handshake complete, entering probe loop";
```

**In `ProbeNeighbor()`:**
```cpp
// Perform handshake before starting probe loop
if (!PerformHandshake(dst_node_id, true)) {
  LOG(ERROR) << "Handshake failed, aborting prober";
  return;
}
LOG(INFO) << "Handshake complete, entering probe loop";
```

## Benefits

1. **Startup Synchronization**: Both nodes confirm they're ready before probing
2. **Buffer Clearing**: Ensures no initialization packets remain in buffers
3. **Failure Detection**: Early detection if nodes can't communicate
4. **Clean State**: Probe loop starts with clean socket buffers

## Testing

Build and run the standalone test:
```bash
cd /home/howwang/work/ml101/rocmxla

# Build
bazel build //xla/backends/profiler/gpu:network_probe_standalone_test

# Run on both nodes
# Node 0:
bazel-bin/xla/backends/profiler/gpu/network_probe_standalone_test \
  --node_id=0 --num_nodes=2 --duration_sec=10

# Node 1:
bazel-bin/xla/backends/profiler/gpu/network_probe_standalone_test \
  --node_id=1 --num_nodes=2 --duration_sec=10
```

Expected output:
```
I1110 ... Handshake: Sending SYN to neighbor X
I1110 ... Handshake: Received ACK from neighbor X
I1110 ... Handshake complete with neighbor X, entering probe loop
```

## Configuration

- **Retries**: 5 attempts per handshake
- **Timeout**: 200ms between retries
- **Total wait**: Up to 1 second per edge

For a node with 10 out-neighbors, handshake completes in ~1-10 seconds depending on network.

## Edge Cases Handled

1. **Packet loss**: Retries handle transient failures
2. **Wrong neighbor ID**: Validates `src_node_id` in packets
3. **Socket errors**: Fails gracefully if send/recv fails
4. **Timeout**: Returns `false` after max retries, thread exits cleanly

## Future Enhancements

- Make retry count and timeout configurable
- Add handshake version negotiation
- Support capability exchange (e.g., hardware timestamping support)




