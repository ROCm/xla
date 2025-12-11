# Master Alpha/Beta Synchronization Plan

## Executive Summary

**Goal:** Aggregate alpha/beta values from all nodes at the end of each probing window, perform centralized calculations on a master node, and synchronize the next probing round.

**Three Implementation Options:**

**Socket-Based:**
- **Variant A (Time-Based):** Fixed windows, relies on aligned clocks
- **Variant B (Master-Controlled):** Master drives start/stop, tolerates clock drift (recommended)

**Gloo-Based:** For large clusters (>32 nodes) with complex collectives

**Quick Comparison:**

| Aspect | Variant A (Time-Based) | Variant B (Master-Controlled) | Gloo |
|--------|----------------------|------------------------------|------|
| **Window Control** | Time-based (4s fixed) | Master controls (adaptive) | Master controls |
| **Clock Drift Handling** | ❌ Sensitive (needs aligned clocks) | ✅ Tolerant (master dictates events) | ✅ Tolerant |
| **Latency** | **16ms** | 22ms | 20-30ms |
| **Complexity** | **Low** | Medium | High |
| **Development** | **6-7 days** | 10-11 days | 14-23 days |
| **Control Thread** | No | Yes (master_sync) | Yes (Gloo) |
| **Best For** | Small testbeds with tight clock sync | **Production, adaptive control, drift tolerance** | Large clusters |

**Recommendation:** Start with **Variant B (Master-Controlled)** to avoid clock-drift stalls, use Variant A only for tightly synchronized MVP environments, and migrate to Gloo when node count exceeds 32 or collectives must be shared with PJRT.

---

## Overview

This document describes the architecture for **master-based synchronization** of alpha/beta coefficients across distributed nodes. The goal is to aggregate alpha/beta values from all probing threads on all nodes at the end of each window, perform centralized calculation on the master node, and synchronize a new probing round via sequence numbers.

Two implementation approaches are documented, with **Socket Variant B** serving as the default baseline due to its tolerance to clock drift:
1. **Socket-Based (Recommended):** Reliable TCP sockets for guaranteed delivery, reuses existing infrastructure
2. **Gloo-Based:** Collective communication library for large-scale deployments

---

## Motivation

**Current State:**
- Each `ProbeSender` thread calculates alpha/beta independently per edge
- Threads synchronize locally using `absl::Barrier` for window rotation
- No cross-node synchronization of alpha/beta values
- Each node makes independent decisions based on local data

**Desired State:**
- At the end of each window, all nodes send their alpha/beta values to the master
- Master performs global calculations (e.g., outlier detection, consensus estimation)
- Master announces a sequence number for the next probing round
- All nodes wait at Gloo barrier before proceeding to next window

---

## Design Options

We have two architectural approaches for master-based synchronization:

1. **Socket-Based (Recommended)** - Reliable TCP sockets, reuses existing infrastructure
2. **Gloo-Based** - Collective communication library, more scalable for large clusters

### Trade-offs Comparison

| Aspect | Socket-Based (TCP) | Gloo-Based |
|--------|-------------------|------------|
| **Latency** | **15-20ms** (TCP gather + send back) | 20-30ms (AllGather + Broadcast) |
| **Reliability** | **TCP guarantees** (no packet loss) | **Gloo guarantees** (retries + checksums) |
| **Code Complexity** | **~250 lines** | ~500 lines + library integration |
| **Infrastructure Reuse** | **High** (reuse socket patterns) | Low (new Gloo context, store) |
| **Dependencies** | **None** (standard TCP sockets) | Gloo library (~10MB), transport layer |
| **Debugging** | **Easy** (tcpdump, strace, netstat) | Harder (library internals, exceptions) |
| **Scalability** | Good up to 32 nodes | Excellent for 100+ nodes |
| **Fault Tolerance** | **Connection-level** (detect failures) | Built-in (automatic retries) |
| **Collective Patterns** | Manual (gather + send back) | **Built-in** (AllGather, Reduce, etc.) |
| **Setup Time** | **<1 day** | 2-3 days (learning curve) |
| **Network Overhead** | **N × data_size** (sequential) | Optimized tree/ring algorithms |
| **Error Handling** | Explicit (connection errors, errno) | Exceptions (try/catch) |
| **Integration** | **Standalone** | Can reuse PJRT collectives |
| **Future Extensions** | Manual hierarchies or persistent conns | **Tree broadcast**, hierarchical reduce |

### Recommendation

**Use Socket-Based Variant **B** (TCP + master control) for initial implementation** because:
- ✅ **Reliable delivery** (TCP guarantees no packet loss)
- ✅ Your existing socket infrastructure is already production-tested
- ✅ Comparable latency to Gloo (22ms vs 20-30ms) while avoiding clock-drift stalls
- ✅ Master-issued round boundaries remove dependency on synchronized node clocks
- ✅ Simpler debugging during development (tcpdump, netstat)
- ✅ No new dependencies or learning curve
- ✅ Sufficient for typical cluster sizes (≤32 nodes)

**Consider Variant A only** if:
- Clocks are already tightly synchronized (≤10 µs skew)
- You need a minimal MVP or offline replay

**Consider migrating to Gloo later** if:
- You scale beyond 32 nodes (need hierarchical aggregation)
- You add complex collective patterns (AllReduce, consensus voting)
- You want to integrate with PJRT's collective infrastructure
- You need automatic fault recovery across node failures

---

## Architecture Option 1: Socket-Based (Recommended)

We have two variants for socket-based synchronization:

### Variant A: Time-Based Windows (Simpler)

**Flow:**
```
[Window N expires after 4s]
    ↓
[All threads on each node]
    → Train SVM, compute alpha/beta for their edges
    → Record stats in local WindowManager
    → Arrive at local absl::Barrier
    ↓
[Last thread on each node]
    → Serialize local alpha/beta data
    → IF master: Accept N-1 TCP connections from workers
    → IF worker: Connect to master via TCP
    ↓
[Master node only]
    → Accept() N-1 worker connections
    → Receive data from each worker (reliable TCP)
    → Perform global calculation (placeholder)
    → Generate sequence number for next round
    → Send sequence back to each worker on their TCP connection
    → Close all connections
    ↓
[All worker nodes]
    → Send data to master (reliable TCP)
    → Block on recv() waiting for sequence number
    → Receive sequence number
    → Close connection
    → Rotate to Window N+1
    → Continue probing
```

### Variant B: Master-Controlled Rounds (More Flexible)

**Flow:**
```
[Master decides to end round N]
    ↓
[Master node]
    → Send "ROUND_END" packet to all N-1 workers (broadcast or individual)
    ↓
[master_sync thread on each node]
    → Receives "ROUND_END" signal
    → Notifies all probe threads to stop probing
    ↓
[All probe threads on each node]
    → Stop probing, calculate alpha/beta for accumulated pairs
    → Each thread sends (alpha, beta) to master_sync thread via queue
    → Block waiting for next round
    ↓
[master_sync thread collects from queue]
    → Receives alpha/beta from all local threads
    → Sends all data to master (multiple packets, one per thread)
    ↓
[Master node]
    → Collects alpha/beta from all nodes (N-1 workers × M threads each)
    → Performs global calculation
    → Generates sequence number for next round
    → Broadcasts "ROUND_START(seq_num)" to all workers
    ↓
[master_sync thread on each node]
    → Receives "ROUND_START(seq_num)"
    → Notifies all probe threads to resume probing
    → Probe threads continue
```

> **Why Variant B is recommended now:** In Variant A the master must wait for every node to naturally rotate its local 4-second window. If any node's clock drifts or a probe thread lags, the master gather stalls. Variant B eliminates this risk by letting the master explicitly signal *when* all nodes stop and start, so drift only affects the probe timestamps—not the round orchestration.

### Comparison: Variant A vs Variant B

| Aspect | Variant A (Time-Based) | Variant B (Master-Controlled) |
|--------|------------------------|-------------------------------|
| **Control** | Distributed (each node decides) | Centralized (master decides) |
| **Complexity** | **Simpler** (no master_sync thread) | More complex (extra thread + queue) |
| **Packets** | **1 per node** (8 for 8 nodes) | **M×N packets** (32 for 8 nodes × 4 threads) |
| **Latency** | 16ms (time-based sync) | 20-25ms (master signals + sync) |
| **Flexibility** | Fixed 4s windows | **Master can adapt window size** |
| **Probe Interruption** | No (wait for window end) | **Yes** (master can stop early) |
| **Clock Sync Required** | Yes (for window alignment) | **No** (master controls timing) |
| **State Management** | **Minimal** (barrier only) | Complex (start/stop signals, queues) |
| **Network Overhead** | **Low** (1 data packet + 1 seq per node) | High (M data packets + 2 control per node) |
| **Implementation** | **6-7 days** | 10-12 days (thread, queue, signaling) |

### Recommendation

**Default to Variant B (Master-Controlled)** because:
- ✅ Eliminates dependence on synchronized clocks (master dictates round boundaries)
- ✅ Allows emergency stop/start and adaptive window lengths
- ✅ Probe threads spend zero time waiting for straggling nodes; master_sync thread controls flow
- ✅ Still uses the same TCP data plane as Variant A

**Use Variant A (Time-Based) only when:**
- Cluster clocks are known to stay within tight skew bounds (≤10 µs)
- You need the fastest possible MVP without master control logic
- Losing up to several hundred milliseconds on drift-induced gather waits is acceptable

### Improved Variant B (Recommended if you need master control)

To implement the queue-based `master_sync` flow (per-thread packets to the master) while keeping master control:

```
[Master decides to end round N]
    ↓
[Master broadcasts "ROUND_END" to all workers]
    ↓
[master_sync thread on each node]
    → Receives "ROUND_END"
    → Sets atomic flag: stop_probing_ = true
    ↓
[All probe threads]
    → Check stop_probing_ in their loop
    → Stop probing, calculate alpha/beta
    → Push stats into a lock-free queue owned by `master_sync`
    → Hit barrier to ensure all packets are queued
    ↓
[master_sync thread]
    → Drains queue entries (one per probe thread)
    → Sends `num_probe_threads` TCP packets to master (one per entry)
    ↓
[Master node]
    → Collects aggregated data from all N-1 workers
    → Performs calculation
    → Broadcasts "ROUND_START(seq_num)" to all workers
    ↓
[master_sync thread]
    → Sets stop_probing_ = false
    → Probe threads resume
```

**This combines the best of both:**
- ✅ Master control (adaptive windows)
- ✅ Explicit per-thread reporting (queue decouples ProbeSender from network I/O)
- ✅ Local queuing ensures master_sync serializes packets without blocking probe threads
- ⚠️ Still needs master_sync thread and control packets

### Data Structures (Common to Both Options)

```cpp
// network_probe.h

// Per-edge alpha/beta for a single window
struct EdgeAlphaBeta {
  int src_node_id;
  int dst_node_id;
  double alpha;
  double beta;
  int pairs_count;
  int lost_count;
};

// All edges from one node in one window
struct NodeWindowData {
  int node_id;
  uint64_t window_id;          // Monotonic window counter
  uint64_t round_id;           // Master-issued round identifier
  uint64_t window_start_ns;
  uint64_t window_end_ns;
  std::vector<EdgeAlphaBeta> edges;
};

// Master's aggregated view of all nodes
struct GlobalWindowData {
  uint64_t window_id;
  uint64_t round_id;             // Matches the master's round broadcast
  uint64_t window_start_ns;
  uint64_t window_end_ns;
  std::vector<NodeWindowData> all_nodes;  // Index by node_id
  uint64_t sequence_number;  // Master-assigned sequence for next round
};
```

**JSONL export requirement:** every line emitted by `WindowManager::ExportWindowToJSON()` and the master’s aggregated JSONL dump must include the new `round_id` so offline analysis can correlate alphas/betas with master-controlled rounds even if `window_start_ns` drifts. This field is in addition to the existing start/end timestamps.

### Socket-Based Implementation

#### NetworkProbeManager Extensions

```cpp
class NetworkProbeManager {
 public:
  // Existing methods...
  absl::Status Initialize();
  absl::Status Start();
  absl::Status Sync();
  
  // NEW: TCP-based master synchronization
  absl::Status InitializeMasterSyncSockets();
  absl::Status PerformMasterSync(const NodeWindowData& local_data);
  
 private:
  DistributedProfilerContext config_;
  std::unique_ptr<WindowManager> window_manager_;
  
  // NEW: Master sync sockets (TCP for reliability)
  int master_listen_sock_ = -1;      // Master TCP listening socket
  std::vector<int> worker_conns_;    // Master's accepted worker connections
  
  uint16_t master_sync_port_ = 0;    // Master TCP listening port (base + 1000)
  
  int master_node_id_ = 0;           // Configurable master (default: node 0)
  std::atomic<uint64_t> current_sequence_{0};
  
  // Helper methods
  void PerformMasterCollection(const NodeWindowData& local_data);
  void PerformWorkerSync(const NodeWindowData& local_data);
  
  // TCP helpers
  absl::StatusOr<int> ConnectToMaster();  // Worker connects to master
  ssize_t SendAll(int sock, const void* data, size_t len);  // Reliable send
  ssize_t RecvAll(int sock, void* data, size_t len);        // Reliable recv
};
```

#### 1. Socket Setup (in `Initialize()`)

```cpp
absl::Status NetworkProbeManager::InitializeMasterSyncSockets() {
  // Port allocation: use base_port + offset to avoid conflicts with probe ports
  // Probing uses: base_port + 2*node_id and base_port + 2*node_id + 1
  // Master sync uses: base_port + 1000 (TCP listening port)
  
  master_sync_port_ = config_.base_port + 1000;
  
  if (config_.node_id == master_node_id_) {
    // ===== MASTER SETUP (TCP Listening Socket) =====
    
    // Create TCP listening socket
    master_listen_sock_ = socket(AF_INET, SOCK_STREAM, 0);
    if (master_listen_sock_ < 0) {
      return absl::InternalError("Failed to create master TCP listening socket");
    }
    
    // Allow port reuse (important for quick restarts)
    int reuse = 1;
    setsockopt(master_listen_sock_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    sockaddr_in bind_addr{};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = INADDR_ANY;
    bind_addr.sin_port = htons(master_sync_port_);
    
    if (bind(master_listen_sock_, (struct sockaddr*)&bind_addr, 
             sizeof(bind_addr)) < 0) {
      close(master_listen_sock_);
      return absl::InternalError(
          absl::StrCat("Failed to bind master listening socket to port ", 
                       master_sync_port_, " (errno=", errno, ")"));
    }
    
    // Listen with backlog = num_nodes (one connection per worker)
    if (listen(master_listen_sock_, config_.num_nodes) < 0) {
      close(master_listen_sock_);
      return absl::InternalError("Failed to listen on master socket");
    }
    
    // Set accept timeout (window_duration - 1s safety margin)
    struct timeval tv = {
      .tv_sec = static_cast<long>(config_.probe_window_s - 1), 
      .tv_usec = 0
    };
    setsockopt(master_listen_sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
    LOG(INFO) << "Master TCP listening socket initialized on port " 
              << master_sync_port_;
              
  } else {
    // ===== WORKER SETUP (No pre-connection needed) =====
    // Workers will connect on-demand during window sync
    LOG(INFO) << "Worker node " << config_.node_id 
              << " will connect to master at " << config_.node_addresses[master_node_id_]
              << ":" << master_sync_port_;
  }
  
  return absl::OkStatus();
}

// TCP Helper: Reliable send (handles partial writes)
ssize_t NetworkProbeManager::SendAll(int sock, const void* data, size_t len) {
  size_t total_sent = 0;
  const char* ptr = static_cast<const char*>(data);
  
  while (total_sent < len) {
    ssize_t sent = send(sock, ptr + total_sent, len - total_sent, 0);
    if (sent < 0) {
      if (errno == EINTR) continue;  // Interrupted, retry
      return -1;  // Error
    }
    if (sent == 0) return -1;  // Connection closed
    total_sent += sent;
  }
  
  return total_sent;
}

// TCP Helper: Reliable receive (handles partial reads)
ssize_t NetworkProbeManager::RecvAll(int sock, void* data, size_t len) {
  size_t total_recv = 0;
  char* ptr = static_cast<char*>(data);
  
  while (total_recv < len) {
    ssize_t received = recv(sock, ptr + total_recv, len - total_recv, 0);
    if (received < 0) {
      if (errno == EINTR) continue;  // Interrupted, retry
      return -1;  // Error
    }
    if (received == 0) return -1;  // Connection closed
    total_recv += received;
  }
  
  return total_recv;
}

// TCP Helper: Worker connects to master
absl::StatusOr<int> NetworkProbeManager::ConnectToMaster() {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    return absl::InternalError("Failed to create TCP socket for master connection");
  }
  
  // Set connection timeout (5 seconds)
  struct timeval tv = {.tv_sec = 5, .tv_usec = 0};
  setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  
  sockaddr_in master_addr{};
  master_addr.sin_family = AF_INET;
  master_addr.sin_addr.s_addr = inet_addr(
      config_.node_addresses[master_node_id_].c_str());
  master_addr.sin_port = htons(master_sync_port_);
  
  if (connect(sock, (struct sockaddr*)&master_addr, sizeof(master_addr)) < 0) {
    close(sock);
    return absl::UnavailableError(
        absl::StrCat("Failed to connect to master at ", 
                     config_.node_addresses[master_node_id_], ":", 
                     master_sync_port_, " (errno=", errno, ")"));
  }
  
  LOG(INFO) << "Worker " << config_.node_id << " connected to master";
  return sock;
}
```

#### 2. Window Sync Orchestration

```cpp
absl::Status NetworkProbeManager::PerformMasterSync(
    const NodeWindowData& local_data) {
  
  if (config_.node_id == master_node_id_) {
    PerformMasterCollection(local_data);
  } else {
    PerformWorkerSync(local_data);
  }
  
  return absl::OkStatus();
}
```

#### 3. Master Collection (TCP with accept/recv/send)

```cpp
void NetworkProbeManager::PerformMasterCollection(const NodeWindowData& local_data) {
  std::vector<NodeWindowData> all_data;
  all_data.reserve(config_.num_nodes);
  all_data.push_back(local_data);  // Master's own data
  
  LOG(INFO) << "Master collecting alpha/beta from " << (config_.num_nodes - 1) 
            << " workers for window " << local_data.window_id;
  
  // Accept connections and receive data from N-1 workers
  worker_conns_.clear();
  worker_conns_.reserve(config_.num_nodes - 1);
  
  for (int i = 1; i < config_.num_nodes; ++i) {
    // Accept connection from worker
    sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    int conn = accept(master_listen_sock_, (struct sockaddr*)&client_addr, &addr_len);
    
    if (conn < 0) {
      LOG(ERROR) << "Master failed to accept connection from worker " << i 
                 << " (errno=" << errno << ")";
      continue;
    }
    
    worker_conns_.push_back(conn);
    
    // First, receive the data size (4 bytes)
    uint32_t data_size;
    if (RecvAll(conn, &data_size, sizeof(data_size)) < 0) {
      LOG(ERROR) << "Master failed to receive data size from worker (errno=" << errno << ")";
      close(conn);
      continue;
    }
    
    // Sanity check on data size (max 64KB)
    if (data_size > 65536) {
      LOG(ERROR) << "Master received invalid data size " << data_size << " from worker";
      close(conn);
      continue;
    }
    
    // Receive the actual data
    std::vector<char> buffer(data_size);
    if (RecvAll(conn, buffer.data(), data_size) < 0) {
      LOG(ERROR) << "Master failed to receive data from worker (errno=" << errno << ")";
      close(conn);
      continue;
    }
    
    NodeWindowData worker_data = DeserializeNodeWindowData(buffer.data(), data_size);
    all_data.push_back(worker_data);
    LOG(INFO) << "Master received data from worker " << worker_data.node_id
              << " (" << worker_data.edges.size() << " edges, " 
              << data_size << " bytes)";
  }
  
  // Perform master calculation
  GlobalWindowData global = MasterCalculation(all_data);
  
  // Send sequence number back to all workers on their connections
  uint64_t seq = global.sequence_number;
  current_sequence_ = seq;
  
  for (size_t i = 0; i < worker_conns_.size(); ++i) {
    if (SendAll(worker_conns_[i], &seq, sizeof(seq)) < 0) {
      LOG(ERROR) << "Master failed to send sequence to worker connection " << i 
                 << " (errno=" << errno << ")";
    } else {
      LOG(INFO) << "Master sent sequence " << seq << " to worker connection " << i;
    }
    close(worker_conns_[i]);  // Close connection after sending
  }
  
  worker_conns_.clear();
  
  LOG(INFO) << "Master completed window " << global.window_id 
            << " sync with sequence " << seq;
}
```

Only nodes listed in `DistributedProfilerContext::probe_participants` are
expected to connect during a given gather. Nodes that never spawn `ProbeSender`
threads (no out-neighbors) skip `PerformMasterSync()` entirely, which prevents
the master from blocking on listener-only workers.

#### 4. Worker Sync (TCP with connect/send/recv)

```cpp
void NetworkProbeManager::PerformWorkerSync(const NodeWindowData& local_data) {
  // Connect to master
  auto conn_result = ConnectToMaster();
  if (!conn_result.ok()) {
    LOG(ERROR) << "Worker " << config_.node_id 
               << " failed to connect to master: " << conn_result.status();
    return;
  }
  int conn = *conn_result;
  
  // Serialize data
  std::string serialized = SerializeNodeWindowData(local_data);
  uint32_t data_size = static_cast<uint32_t>(serialized.size());
  
  // Send data size first (4 bytes)
  if (SendAll(conn, &data_size, sizeof(data_size)) < 0) {
    LOG(ERROR) << "Worker " << config_.node_id 
               << " failed to send data size to master (errno=" << errno << ")";
    close(conn);
    return;
  }
  
  // Send actual data
  if (SendAll(conn, serialized.data(), serialized.size()) < 0) {
    LOG(ERROR) << "Worker " << config_.node_id 
               << " failed to send data to master (errno=" << errno << ")";
    close(conn);
    return;
  }
  
  LOG(INFO) << "Worker " << config_.node_id << " sent " << serialized.size() 
            << " bytes to master for window " << local_data.window_id;
  
  // Receive sequence number from master
  uint64_t seq;
  if (RecvAll(conn, &seq, sizeof(seq)) < 0) {
    LOG(ERROR) << "Worker " << config_.node_id 
               << " failed to receive sequence from master (errno=" << errno << ")";
    close(conn);
    return;
  }
  
  current_sequence_ = seq;
  LOG(INFO) << "Worker " << config_.node_id << " received sequence " << seq;
  
  // Close connection
  close(conn);
}
```

#### 5. Integration with ProbeSender

```cpp
void NetworkProbeManager::ProbeSender(int dst_node_id) {
  LOG(INFO) << "ProbeSender thread started for edge " 
            << config_.node_id << " -> " << dst_node_id;
  
  while (running_.load()) {
    // Check shared window expiry (local check)
    if (window_manager_->IsWindowExpired()) {
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[dst_node_id]);
      
      // Train SVM, record stats (existing code)
      // ... [existing SVM training code] ...
      
      // Local barrier: wait for all threads on this node
      bool is_last_thread = window_manager_->NotifyWindowExpired();
      
      if (is_last_thread) {
        // This thread is responsible for cross-node TCP sync
        
        // 1. Collect all alpha/beta from this node
        NodeWindowData local_data = window_manager_->CollectWindowData(
            config_.node_id);
        
        // 2. Perform TCP-based master synchronization
        // - Master: accepts N-1 connections, receives data, sends sequence back
        // - Worker: connects to master, sends data, receives sequence
        auto status = PerformMasterSync(local_data);
        if (!status.ok()) {
          LOG(ERROR) << "Master sync failed: " << status;
        }
        
        // 3. Rotate window after receiving sequence number
        window_manager_->RotateWindow();
      }
      // Other threads are blocked at NotifyWindowExpired() barrier
      // They resume after last thread completes rotation
    }
    
    // Continue normal probing (existing code)
    SendProbeTriple(dst_node_id);
    std::this_thread::sleep_for(
        std::chrono::microseconds(config_.probe_cadence_us));
  }
}
```

---

## Variant B Implementation (Master-Controlled Rounds)

If you choose the master-controlled approach, here's the implementation:

### Additional Data Structures

```cpp
// network_probe.h

// Control messages between master and workers
enum class SyncCommand : uint8_t {
  ROUND_END = 1,     // Master tells workers to stop and sync
  ROUND_START = 2,   // Master tells workers to start next round
};

struct SyncMessage {
  SyncCommand command;
  uint64_t sequence_number;  // For ROUND_START
  uint64_t timestamp_ns;     // When message was sent
};

class NetworkProbeManager {
 public:
  // NEW: Variant B methods
  void MasterSyncThread();  // Dedicated thread for master communication
  
 private:
  // NEW: Variant B state
  std::atomic<bool> stop_probing_{false};  // Signal to stop probing
  std::thread master_sync_thread_;         // Dedicated sync thread
  int control_sock_ = -1;                  // Socket for master control messages
  uint16_t control_port_ = 0;              // Port for control messages (base + 2000)
  
  std::condition_variable round_start_cv_;  // Wake probe threads for new round
  std::mutex round_mutex_;
};
```

### Master Sync Thread

```cpp
void NetworkProbeManager::MasterSyncThread() {
  LOG(INFO) << "MasterSyncThread started on node " << config_.node_id;
  
  if (config_.node_id == master_node_id_) {
    // ===== MASTER BEHAVIOR =====
    while (running_.load()) {
      // Wait for window duration
      std::this_thread::sleep_for(std::chrono::seconds(config_.probe_window_s));
      
      // Send ROUND_END to all workers
      SyncMessage msg;
      msg.command = SyncCommand::ROUND_END;
      msg.timestamp_ns = GetMonotonicNs();
      
      for (int i = 1; i < config_.num_nodes; ++i) {
        sockaddr_in worker_addr = ParseAddress(
            config_.node_addresses[i], control_port_);
        sendto(control_sock_, &msg, sizeof(msg), 0,
               (struct sockaddr*)&worker_addr, sizeof(worker_addr));
      }
      
      LOG(INFO) << "Master sent ROUND_END to all workers";
      
      // Set own stop flag
      stop_probing_ = true;
      
      // Wait for probe threads to finish via barrier
      // (probe threads will hit barrier in IsWindowExpired() check)
      
      // Collect data from self
      NodeWindowData local_data = window_manager_->CollectWindowData(config_.node_id);
      
      // Collect from all workers (same as Variant A)
      PerformMasterCollection(local_data);
      
      // Generate new sequence
      uint64_t new_seq = current_sequence_ + 1;
      current_sequence_ = new_seq;
      
      // Send ROUND_START to all workers
      msg.command = SyncCommand::ROUND_START;
      msg.sequence_number = new_seq;
      msg.timestamp_ns = GetMonotonicNs();
      
      for (int i = 1; i < config_.num_nodes; ++i) {
        sockaddr_in worker_addr = ParseAddress(
            config_.node_addresses[i], control_port_);
        sendto(control_sock_, &msg, sizeof(msg), 0,
               (struct sockaddr*)&worker_addr, sizeof(worker_addr));
      }
      
      LOG(INFO) << "Master sent ROUND_START(" << new_seq << ") to all workers";
      
      // Reset own flag
      stop_probing_ = false;
      window_manager_->RotateWindow();
      
      // Wake probe threads
      {
        std::lock_guard<std::mutex> lock(round_mutex_);
        round_start_cv_.notify_all();
      }
    }
    
  } else {
    // ===== WORKER BEHAVIOR =====
    while (running_.load()) {
      // Wait for ROUND_END from master
      SyncMessage msg;
      ssize_t n = recvfrom(control_sock_, &msg, sizeof(msg), 0, nullptr, nullptr);
      
      if (n != sizeof(msg)) {
        LOG(ERROR) << "Worker failed to receive control message (errno=" << errno << ")";
        continue;
      }
      
      if (msg.command == SyncCommand::ROUND_END) {
        LOG(INFO) << "Worker received ROUND_END from master";
        
        // Signal probe threads to stop
        stop_probing_ = true;
        
        // Wait for probe threads to finish and collect data
        // (they will hit barrier and last thread will collect)
        NodeWindowData local_data = window_manager_->CollectWindowData(config_.node_id);
        
        // Send to master (same as Variant A)
        PerformWorkerSync(local_data);
        
      } else if (msg.command == SyncCommand::ROUND_START) {
        LOG(INFO) << "Worker received ROUND_START(" << msg.sequence_number << ") from master";
        
        // Update sequence
        current_sequence_ = msg.sequence_number;
        
        // Reset flag
        stop_probing_ = false;
        window_manager_->RotateWindow();
        
        // Wake probe threads
        {
          std::lock_guard<std::mutex> lock(round_mutex_);
          round_start_cv_.notify_all();
        }
      }
    }
  }
}
```

### Modified ProbeSender (Variant B)

```cpp
void NetworkProbeManager::ProbeSender(int dst_node_id) {
  LOG(INFO) << "ProbeSender thread started for edge " 
            << config_.node_id << " -> " << dst_node_id;
  
  while (running_.load()) {
    // Check if master told us to stop
    if (stop_probing_.load()) {
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[dst_node_id]);
      
      // Train SVM, record stats (same as Variant A)
      auto& pairs_map = probe_pairs_[dst_node_id];
      if (!pairs_map.empty()) {
        // ... [existing SVM training code] ...
        window_manager_->RecordEdgeStats(dst_node_id, alpha, beta, 
                                         complete_pairs.size(), lost_count);
      }
      
      probe_pairs_[dst_node_id].clear();
      
      // Local barrier: wait for all threads
      bool is_last_thread = window_manager_->NotifyWindowExpired();
      
      if (is_last_thread) {
        // Note: master_sync thread will handle communication with master
        // This thread just waits at barrier
      }
      
      // Wait for ROUND_START signal
      {
        std::unique_lock<std::mutex> lock(round_mutex_);
        round_start_cv_.wait(lock, [this] { return !stop_probing_.load(); });
      }
      
      LOG(INFO) << "ProbeSender resumed after ROUND_START";
      continue;
    }
    
    // Normal probing (same as before)
    SendProbeTriple(dst_node_id);
    std::this_thread::sleep_for(
        std::chrono::microseconds(config_.probe_cadence_us));
  }
}
```

### Variant B: Performance Analysis

**Latency Breakdown (8 nodes, LAN):**
```
1. Master sends ROUND_END:    1ms (UDP to 7 workers)
2. Workers stop probing:      ~1ms (check atomic flag)
3. SVM calculation:           ~1ms (per thread)
4. Barrier synchronization:   <1ms (local threads)
5. TCP data transfer:         16ms (same as Variant A)
6. Master calculation:        1ms
7. Master sends ROUND_START:  1ms (UDP to 7 workers)
Total:                        ~22ms (vs 16ms for Variant A)
```

**Advantages over Variant A:**
- ✅ Master has full control (can adjust window size dynamically)
- ✅ Can trigger emergency sync (e.g., detected anomaly)
- ✅ No clock sync needed (master controls timing)

**Disadvantages:**
- ⚠️ +6ms overhead (control messages + signaling)
- ⚠️ More complex (extra thread, atomic flags, condition variables)
- ⚠️ More failure modes (control message loss, thread coordination)

### When to Use Variant B

Choose Variant B if:
1. **Adaptive window sizing** - Master adjusts windows based on stability
2. **Coordinated experiments** - Need to sync multiple profiling phases
3. **No clock sync** - Nodes have significant clock drift
4. **Emergency response** - Master needs to interrupt probing immediately

Otherwise, **stick with Variant A** (time-based) for simplicity.

---

### Socket-Based: Performance Analysis (Variant A)

**Latency Breakdown (8 nodes, LAN):**
```
1. Serialize data:          0.1ms per node
2. TCP connection setup:    0.5ms per worker (3-way handshake)
3. Workers send to master:  7 × 1ms = 7ms (sequential accept/recv)
4. Master calculation:      1ms
5. Master send to workers:  7 × 1ms = 7ms (sequential send on each conn)
6. TCP teardown:            negligible (RST packets)
Total:                      ~16ms (8 nodes)
```

**Why TCP Instead of UDP:**
- ✅ **Reliable delivery** (no packet loss of critical alpha/beta data)
- ✅ **Ordered delivery** (data + sequence number arrive in order)
- ✅ **Connection state** (easy to detect failed workers)
- ✅ **Backpressure** (TCP flow control prevents buffer overflow)
- ⚠️ Slightly higher latency (+6ms) due to connection setup/teardown
- ⚠️ 3-way handshake overhead per window

**TCP vs UDP Trade-off:**
- UDP would be ~10ms (no connection setup)
- TCP is ~16ms (with connection overhead)
- **+6ms overhead is acceptable** for guaranteed delivery
- **Alternative**: Keep persistent connections (reduce to ~10ms, but adds state)

**Advantages:**
- ✅ **No packet loss** (critical for sync correctness)
- ✅ 2-3x lower latency than Gloo
- ✅ Simple failure mode: connection refused/timeout
- ✅ Reuse existing socket helpers
- ✅ Easy debugging: `tcpdump -i eth0 port 48000`
- ✅ Built-in flow control

**Disadvantages:**
- ⚠️ Sequential gather (not optimized for >32 nodes)
- ⚠️ Connection overhead per window (+6ms)
- ⚠️ No hierarchical patterns (need custom code)

---

## Architecture Option 2: Gloo-Based

### High-Level Flow

```
[Window N expires]
    ↓
[All threads on each node]
    → Train SVM, compute alpha/beta for their edges
    → Record stats in local WindowManager
    → Arrive at local absl::Barrier
    ↓
[Last thread on each node]
    → Collect all alpha/beta from current window
    → Send aggregated data to master via Gloo AllGather
    → Arrive at Gloo barrier (cross-node sync)
    ↓
[Master node only]
    → Receive alpha/beta from all nodes
    → Perform global calculation (placeholder: average, outlier filtering, etc.)
    → Generate sequence number for next round
    → Broadcast sequence number via Gloo Broadcast
    ↓
[All nodes]
    → Receive sequence number from master
    → Release from Gloo barrier
    → Rotate to Window N+1
    → Continue probing
```

---

## Data Structures

### 1. Per-Window Alpha/Beta Aggregation

```cpp
// network_probe.h

// Per-edge alpha/beta for a single window
struct EdgeAlphaBeta {
  int src_node_id;
  int dst_node_id;
  double alpha;
  double beta;
  int pairs_count;
  int lost_count;
};

// All edges from one node in one window
struct NodeWindowData {
  int node_id;
  uint64_t window_id;          // Monotonic window counter
  uint64_t window_start_ns;
  uint64_t window_end_ns;
  std::vector<EdgeAlphaBeta> edges;
};

// Master's aggregated view of all nodes
struct GlobalWindowData {
  uint64_t window_id;
  uint64_t window_start_ns;
  uint64_t window_end_ns;
  std::vector<NodeWindowData> all_nodes;  // Index by node_id
  uint64_t sequence_number;  // Master-assigned sequence for next round
};
```

### 2. WindowManager Extensions

```cpp
class WindowManager {
 public:
  // Existing methods...
  bool IsWindowExpired();
  bool NotifyWindowExpired();  // Local absl::Barrier
  void RotateWindow();
  void RecordEdgeStats(int dst_id, double alpha, double beta, int pairs, int lost);
  
  // NEW: Collect all edges for current window
  NodeWindowData CollectWindowData(int node_id);
  
  // NEW: Apply master's decision to local state (optional)
  void ApplyMasterDecision(const GlobalWindowData& global_data);
  
 private:
  uint64_t window_duration_ns_;
  std::atomic<uint64_t> window_id_{0};  // NEW: Monotonic counter
  std::atomic<uint64_t> window_start_ns_;
  std::atomic<uint64_t> window_end_ns_;
  
  absl::Mutex mu_;
  WindowStats current_window_ ABSL_GUARDED_BY(mu_);
  std::vector<WindowStats> completed_windows_ ABSL_GUARDED_BY(mu_);
  
  std::unique_ptr<absl::Barrier> barrier_;  // Local thread barrier
  absl::Mutex barrier_mu_;
};
```

### 3. NetworkProbeManager Extensions

```cpp
class NetworkProbeManager {
 public:
  // Existing methods...
  absl::Status Initialize();
  absl::Status Start();
  absl::Status Sync();
  
  // NEW: Gloo-based master synchronization
  absl::Status InitializeGlooContext();
  absl::Status PerformMasterSync(const NodeWindowData& local_data);
  
 private:
  DistributedProfilerContext config_;
  std::unique_ptr<WindowManager> window_manager_;
  
  // NEW: Gloo integration
  std::shared_ptr<gloo::Context> gloo_context_;
  std::shared_ptr<gloo::rendezvous::Store> gloo_store_;
  std::shared_ptr<gloo::transport::Device> gloo_device_;
  
  int master_node_id_ = 0;  // Configurable master (default: node 0)
  std::atomic<uint64_t> current_sequence_{0};
};
```

---

## Gloo Integration

### 1. Initialization

```cpp
absl::Status NetworkProbeManager::InitializeGlooContext() {
  // Create Gloo transport device
  gloo::transport::tcp::attr device_attr;
  device_attr.iface = "eth0";  // TODO: Read from config
  gloo_device_ = gloo::transport::tcp::CreateDevice(device_attr);
  
  // Create KV store for rendezvous
  // Option A: Use existing KV store (if available via PJRT client)
  // Option B: Create dedicated KV store for profiling
  auto kv_client = GetDistributedKeyValueClient();  // From PJRT context
  gloo_store_ = std::make_unique<GlooKVStore>(kv_client);
  
  // Create Gloo context
  gloo_context_ = std::make_shared<gloo::rendezvous::Context>(
      config_.node_id, config_.num_nodes);
  
  auto prefix_store = std::make_shared<gloo::rendezvous::PrefixStore>(
      "profiler/master_sync", *gloo_store_);
  
  try {
    gloo_context_->connectFullMesh(*prefix_store, gloo_device_);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo context initialization failed: ", e.what()));
  }
  
  LOG(INFO) << "Gloo context initialized for node " << config_.node_id;
  return absl::OkStatus();
}
```

### 2. AllGather for Alpha/Beta Collection

```cpp
absl::Status NetworkProbeManager::PerformMasterSync(
    const NodeWindowData& local_data) {
  
  // Serialize local data
  std::string serialized_local = SerializeNodeWindowData(local_data);
  
  // Prepare buffers for AllGather
  std::vector<std::string> all_node_data(config_.num_nodes);
  std::vector<std::vector<char>> send_bufs(config_.num_nodes);
  std::vector<std::vector<char>> recv_bufs(config_.num_nodes);
  
  // Each node sends its data to all other nodes
  for (int i = 0; i < config_.num_nodes; ++i) {
    if (i == config_.node_id) {
      send_bufs[i].assign(serialized_local.begin(), serialized_local.end());
    } else {
      send_bufs[i].resize(0);  // Empty for non-self
    }
    recv_bufs[i].resize(MAX_SERIALIZED_SIZE);  // Pre-allocate
  }
  
  // Gloo AllGather
  try {
    // Use Gloo's allgatherv (variable-length gather)
    gloo::AllgathervOptions opts(gloo_context_);
    opts.setTag(window_manager_->GetCurrentWindowId());  // Unique tag per window
    
    // TODO: Implement proper AllGather with Gloo primitives
    // This is a high-level sketch
    
    for (int i = 0; i < config_.num_nodes; ++i) {
      // Each node broadcasts its data
      if (i == config_.node_id) {
        // Send local data
        opts.setInput(send_bufs[i].data(), send_bufs[i].size());
      }
      // All nodes receive
      opts.setOutput(recv_bufs[i].data(), recv_bufs[i].size());
    }
    
    gloo::allgatherv(opts);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo AllGather failed: ", e.what()));
  }
  
  // Deserialize all received data
  std::vector<NodeWindowData> all_nodes_data;
  for (int i = 0; i < config_.num_nodes; ++i) {
    all_nodes_data.push_back(DeserializeNodeWindowData(recv_bufs[i]));
  }
  
  // Master performs calculation
  if (config_.node_id == master_node_id_) {
    GlobalWindowData global_data = MasterCalculation(all_nodes_data);
    
    // Broadcast sequence number
    TF_RETURN_IF_ERROR(BroadcastSequenceNumber(global_data.sequence_number));
  } else {
    // Non-master: receive sequence number
    TF_RETURN_IF_ERROR(ReceiveSequenceNumber());
  }
  
  return absl::OkStatus();
}
```

### 3. Broadcast Sequence Number

```cpp
absl::Status NetworkProbeManager::BroadcastSequenceNumber(uint64_t seq_num) {
  try {
    gloo::BroadcastOptions opts(gloo_context_);
    opts.setRoot(master_node_id_);
    opts.setTag(0xDEADBEEF);  // Fixed tag for sequence numbers
    
    std::vector<uint64_t> buffer(1);
    if (config_.node_id == master_node_id_) {
      buffer[0] = seq_num;
      opts.setOutput(&buffer[0], 1);
      LOG(INFO) << "Master broadcasting sequence number: " << seq_num;
    } else {
      opts.setOutput(&buffer[0], 1);
    }
    
    gloo::broadcast(opts);
    
    if (config_.node_id != master_node_id_) {
      current_sequence_ = buffer[0];
      LOG(INFO) << "Node " << config_.node_id 
                << " received sequence number: " << buffer[0];
    }
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo Broadcast failed: ", e.what()));
  }
  
  return absl::OkStatus();
}

absl::Status NetworkProbeManager::ReceiveSequenceNumber() {
  return BroadcastSequenceNumber(0);  // Non-master just receives
}
```

---

## Master Calculation (Placeholder)

```cpp
GlobalWindowData NetworkProbeManager::MasterCalculation(
    const std::vector<NodeWindowData>& all_nodes_data) {
  
  GlobalWindowData global;
  global.window_id = all_nodes_data[0].window_id;
  global.round_id = all_nodes_data[0].round_id;
  global.window_start_ns = all_nodes_data[0].window_start_ns;
  global.window_end_ns = all_nodes_data[0].window_end_ns;
  global.all_nodes = all_nodes_data;
  
  // ============================================================
  // PLACEHOLDER: Future global calculations
  // ============================================================
  
  // Option 1: Consensus alpha/beta per edge
  // - For each edge (i -> j), average alpha/beta across multiple windows
  // - Detect outliers using MAD (Median Absolute Deviation)
  // - Update global clock skew model
  
  // Option 2: Global clock synchronization
  // - Build spanning tree of alpha/beta relationships
  // - Compute transitive clock offsets: offset[i][j] = offset[i][k] + offset[k][j]
  // - Detect cycles and inconsistencies
  
  // Option 3: Adaptive probing control
  // - If alpha/beta stable across windows, reduce probe cadence
  // - If high variance, increase probe cadence or packet spacing
  
  // Example: Simple averaging (for demonstration)
  absl::flat_hash_map<std::pair<int, int>, std::vector<double>> alpha_map;
  absl::flat_hash_map<std::pair<int, int>, std::vector<double>> beta_map;
  
  for (const auto& node_data : all_nodes_data) {
    for (const auto& edge : node_data.edges) {
      auto key = std::make_pair(edge.src_node_id, edge.dst_node_id);
      alpha_map[key].push_back(edge.alpha);
      beta_map[key].push_back(edge.beta);
    }
  }
  
  LOG(INFO) << "Master aggregated data from " << all_nodes_data.size() 
            << " nodes for window " << global.window_id;
  
  // TODO: Store aggregated alpha/beta in global state
  // TODO: Export to centralized JSONL file (with round_id)
  
  // ============================================================
  // Generate sequence number for next round
  // ============================================================
  global.sequence_number = global.window_id + 1;
  
  LOG(INFO) << "Master assigned sequence number " << global.sequence_number 
            << " for next window";
  
  return global;
}
```

---

## GraphCalc Utility (New)

To keep `NetworkProbeManager` lean and enable richer master-side analytics, introduce a dedicated `GraphCalc` utility:

| Item | Description |
|------|-------------|
| **Files** | `xla/backends/profiler/gpu/graph_calc.h`, `xla/backends/profiler/gpu/graph_calc.cc` |
| **Design Doc** | `docs/graph_calc.md` (captures math, graph inputs, aggregation algorithms) |
| **Inputs** | `GlobalWindowData` (per-round alphas/betas), topology/edge metadata |
| **Outputs** | Aggregated skew models, consensus offsets, diagnostics for JSONL export |
| **API Sketch** | `absl::StatusOr<GraphCalc::RoundSummary> GraphCalc::ProcessRound(const GlobalWindowData& data);` |

Responsibilities:
1. Normalize and validate per-edge α/β pairs before exporting.
2. Optionally compute higher-level metrics (e.g., spanning-tree offsets, outlier detection) without bloating `NetworkProbeManager`.
3. Emit a `RoundSummary` struct consumed by both logging and JSONL export; includes `round_id` so downstream tools can join on the same identifier.

`MasterCalculation()` should evolve into a thin wrapper that populates `GlobalWindowData`, hands it to `GraphCalc`, then merges the resulting summary back into its control flow (sequence numbers, exports, dashboards).

---

## Integration with ProbeSender Loop

### Modified ProbeSender Thread

```cpp
void NetworkProbeManager::ProbeSender(int dst_node_id) {
  LOG(INFO) << "ProbeSender thread started for edge " 
            << config_.node_id << " -> " << dst_node_id;
  
  while (running_.load()) {
    // Check shared window expiry (local check)
    if (window_manager_->IsWindowExpired()) {
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[dst_node_id]);
      
      // Train SVM, record stats (existing code)
      auto& pairs_map = probe_pairs_[dst_node_id];
      if (!pairs_map.empty()) {
        // ... (existing SVM training code) ...
        
        if (complete_pairs.size() >= 10) {
          auto prob_info = probe_utils::convert_probe_pairs_to_xy_pairs(
              complete_pairs, 0.2, true);
          if (!prob_info.points.empty()) {
            SVMModel svm_model;
            if (svm_model.train(prob_info)) {
              double alpha = svm_model.getAlpha();
              double beta = svm_model.getBeta();
              
              // Record stats in shared window manager
              window_manager_->RecordEdgeStats(dst_node_id, alpha, beta, 
                                               complete_pairs.size(), lost_count);
            }
          }
        }
      }
      
      probe_pairs_[dst_node_id].clear();
      
      // ===================================================
      // NEW: Cross-node synchronization
      // ===================================================
      
      // Local barrier: wait for all threads on this node
      bool is_last_thread = window_manager_->NotifyWindowExpired();
      
      if (is_last_thread) {
        // This thread is responsible for cross-node sync
        
        // 1. Collect all alpha/beta from this node
        NodeWindowData local_data = window_manager_->CollectWindowData(
            config_.node_id);
        
        // 2. Perform Gloo-based master synchronization
        auto status = PerformMasterSync(local_data);
        if (!status.ok()) {
          LOG(ERROR) << "Master sync failed: " << status;
        }
        
        // 3. Rotate window after receiving sequence number
        window_manager_->RotateWindow();
      }
      // Other threads are blocked at NotifyWindowExpired() barrier
      // They resume after last thread completes rotation
    }
    
    // Continue normal probing (existing code)
    SendProbeTriple(dst_node_id);
    std::this_thread::sleep_for(
        std::chrono::microseconds(config_.probe_cadence_us));
  }
}
```

---

## Serialization

### Protobuf Schema (Recommended)

```protobuf
// probe_sync.proto

syntax = "proto3";

package xla.profiler;

message EdgeAlphaBeta {
  int32 src_node_id = 1;
  int32 dst_node_id = 2;
  double alpha = 3;
  double beta = 4;
  int32 pairs_count = 5;
  int32 lost_count = 6;
}

message NodeWindowData {
  int32 node_id = 1;
  uint64 window_id = 2;
  uint64 round_id = 3;
  uint64 window_start_ns = 4;
  uint64 window_end_ns = 5;
  repeated EdgeAlphaBeta edges = 6;
}

message GlobalWindowData {
  uint64 window_id = 1;
  uint64 round_id = 2;
  uint64 window_start_ns = 3;
  uint64 window_end_ns = 4;
  repeated NodeWindowData all_nodes = 5;
  uint64 sequence_number = 6;
}
```

### Alternative: Manual Binary Serialization

```cpp
std::string SerializeNodeWindowData(const NodeWindowData& data) {
  std::ostringstream oss;
  oss.write(reinterpret_cast<const char*>(&data.node_id), sizeof(data.node_id));
  oss.write(reinterpret_cast<const char*>(&data.window_id), sizeof(data.window_id));
  oss.write(reinterpret_cast<const char*>(&data.round_id), sizeof(data.round_id));
  oss.write(reinterpret_cast<const char*>(&data.window_start_ns), sizeof(data.window_start_ns));
  oss.write(reinterpret_cast<const char*>(&data.window_end_ns), sizeof(data.window_end_ns));
  
  size_t num_edges = data.edges.size();
  oss.write(reinterpret_cast<const char*>(&num_edges), sizeof(num_edges));
  
  for (const auto& edge : data.edges) {
    oss.write(reinterpret_cast<const char*>(&edge), sizeof(EdgeAlphaBeta));
  }
  
  return oss.str();
}

NodeWindowData DeserializeNodeWindowData(const std::vector<char>& buffer) {
  std::istringstream iss(std::string(buffer.begin(), buffer.end()));
  NodeWindowData data;
  
  iss.read(reinterpret_cast<char*>(&data.node_id), sizeof(data.node_id));
  iss.read(reinterpret_cast<char*>(&data.window_id), sizeof(data.window_id));
  iss.read(reinterpret_cast<char*>(&data.round_id), sizeof(data.round_id));
  iss.read(reinterpret_cast<char*>(&data.window_start_ns), sizeof(data.window_start_ns));
  iss.read(reinterpret_cast<char*>(&data.window_end_ns), sizeof(data.window_end_ns));
  
  size_t num_edges;
  iss.read(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));
  
  data.edges.resize(num_edges);
  for (size_t i = 0; i < num_edges; ++i) {
    iss.read(reinterpret_cast<char*>(&data.edges[i]), sizeof(EdgeAlphaBeta));
  }
  
  return data;
}
```

---

## Configuration

### DistributedProfilerContext Extensions

```cpp
struct DistributedProfilerContext {
  // Existing fields...
  int node_id;
  int num_nodes;
  std::vector<std::string> node_addresses;
  std::vector<int> neighbors;
  std::vector<int> in_neighbors;
  int probe_cadence_us = 800;
  int probe_window_s = 4;
  
  // NEW: Master synchronization config
  bool enable_master_sync = true;          // Enable cross-node sync
  int master_node_id = 0;                  // Master node (default: 0)
  uint16_t master_control_port = 36000;    // Base UDP port for ROUND_END/START
  uint16_t master_sync_port = 37000;       // TCP port for per-window payloads
  std::vector<int> probe_participants;     // Nodes expected to run ProbeSender
  bool has_probe_senders = false;          // Convenience flag for zero out-degree nodes
  std::string gloo_interface = "eth0";     // Network interface for Gloo
  int gloo_timeout_ms = 30000;             // 30s timeout for Gloo ops
  
  // NEW: KV store for Gloo rendezvous
  std::shared_ptr<DistributedKeyValueClient> kv_client;
};
```

> **Note:** During graph generation the master now persists a `probe_participants`
> KV entry listing every node with at least one `ProbeSender` thread. This list
> feeds `DistributedProfilerContext::probe_participants` and allows the master to
> skip idle listener-only nodes during the TCP gather phase.

---

## Implementation Phases

### Socket-Based Implementation (Recommended Path)

#### Phase 1: Data Structures and Serialization
**Files:** `xla/backends/profiler/gpu/network_probe.h`

- [ ] Add `EdgeAlphaBeta`, `NodeWindowData`, `GlobalWindowData` structs
- [ ] Add `window_id_` counter to `WindowManager`
- [ ] Implement `WindowManager::CollectWindowData()`
- [ ] Implement serialization/deserialization functions (binary)
- [ ] Add unit tests for serialization

**Estimated effort:** 1 day

---

#### Phase 2: Socket Setup
**Files:** `xla/backends/profiler/gpu/network_probe.{h,cc}`

- [ ] Add master sync socket members to `NetworkProbeManager`
- [ ] Implement `InitializeMasterSyncSockets()`
- [ ] Port allocation: `base_port + 1000/1001`
- [ ] Test socket creation (master + workers)

**Estimated effort:** 0.5 days

---

#### Phase 3: Master Collection & Broadcast
**Files:** `xla/backends/profiler/gpu/network_probe.cc`

- [ ] Implement `PerformMasterCollection()` with recvfrom loop
- [ ] Implement `PerformWorkerSync()` with sendto + recvfrom
- [ ] Add timeout handling (window_duration - 1s)
- [ ] Test 2-node, 4-node sync

**Estimated effort:** 1 day

---

#### Phase 4: Master Calculation Placeholder
**Files:** `xla/backends/profiler/gpu/network_probe.cc`, `xla/backends/profiler/gpu/graph_calc.{h,cc}`, `docs/graph_calc.md`

- [ ] Create `docs/graph_calc.md` design doc detailing GraphCalc responsibilities, inputs (alphas, betas, topology), and APIs
- [ ] Add `graph_calc.h/.cc` implementing `GraphCalc` class (ingests node graph + per-round stats)
- [ ] Refactor `MasterCalculation()` to delegate to `GraphCalc` utilities
- [ ] Log aggregated alpha/beta on master and return `round_id`
- [ ] Export master's view to separate JSONL file (including new `round_id` field)

**Estimated effort:** 1 day

---

#### Phase 5: Integration with ProbeSender
**Files:** `xla/backends/profiler/gpu/network_probe.cc`

- [ ] Call `PerformMasterSync()` after local barrier
- [ ] Ensure only last thread performs sync
- [ ] Test end-to-end flow with probing active

**Estimated effort:** 0.5 days

---

#### Phase 6: Testing and Validation
**Files:** `xla/backends/profiler/gpu/network_probe_standalone_test.cc`, SkyPilot tasks

- [ ] Unit tests for serialization, socket ops
- [ ] Standalone test: 2-node master sync
- [ ] SkyPilot: 4-node, 8-node master sync
- [ ] Verify window alignment across nodes
- [ ] Measure latency impact (expect <10ms)
- [ ] Test failure scenarios (worker timeout, master crash)

**Estimated effort:** 2 days

---

#### Phase 7: Documentation
**Files:** `master_alpha_sync.md`, `DOCUMENTATION_INDEX.md`

- [ ] Document socket-based architecture
- [ ] Add examples of future use cases (outlier detection, adaptive probing)
- [ ] Update `DOCUMENTATION_INDEX.md`

**Estimated effort:** 0.5 days

---

### Socket-Based: Total Estimated Effort

**6-7 days** (single developer, no major blockers)

---

### Gloo-Based Implementation (Future Upgrade Path)

If you later decide to migrate to Gloo (for >32 nodes or complex collectives):

#### Phase 1: Gloo Context Setup
- [ ] Add Gloo dependencies to `BUILD` file
- [ ] Implement `InitializeGlooContext()` using `GlooKVStore`
- [ ] Test connection establishment (2-node)

**Estimated effort:** 2 days

---

#### Phase 2: AllGather + Broadcast
- [ ] Replace socket gather with `gloo::AllgathervOptions`
- [ ] Replace socket broadcast with `gloo::BroadcastOptions`
- [ ] Test with 8-node, 16-node setups

**Estimated effort:** 2-3 days

---

#### Phase 3: Hierarchical Extensions
- [ ] Implement tree-based broadcast for >32 nodes
- [ ] Add regional masters for WAN deployments

**Estimated effort:** 3-4 days

---

### Gloo-Based: Total Estimated Effort

**7-9 days** (additional effort on top of socket-based)

---

## Performance Considerations

### Latency Comparison

**Current (no master sync):**
- Window rotation: ~1ms (local `absl::Barrier` only)

**Socket-Based Master Sync (TCP, Variant B default):**
- Serialize data: ~0.1ms per node
- TCP connection setup: ~0.5ms per worker (3-way handshake)
- Workers → Master (TCP): 1-2ms per worker (sequential accept/recv)
- Master calculation: ~1ms
- Master → Workers (TCP): 1-2ms per worker (send on established conns)
- TCP teardown: negligible
- **Total: 15-20ms** (8 nodes on LAN)
  - (Add +2ms for ROUND_END/ROUND_START control packets → ~22ms total)

**Gloo-Based Master Sync:**
- Serialize data: ~0.1ms per node
- Gloo AllGather: ~10-15ms (tree algorithm, 8 nodes)
- Master calculation: ~1ms
- Gloo Broadcast: ~5-10ms (tree algorithm)
- **Total: 15-25ms** (8 nodes on LAN)

### Scalability Comparison

| # Nodes | Socket (TCP) Latency | Gloo Latency | Winner |
|---------|---------------------|--------------|--------|
| 2       | ~8ms                | ~8ms         | **Tie** |
| 4       | ~12ms               | ~12ms        | **Tie** |
| 8       | ~16ms               | ~15ms        | **Gloo (1.07x faster)** |
| 16      | ~24ms               | ~18ms        | **Gloo (1.3x faster)** |
| 32      | ~40ms               | ~22ms        | **Gloo (1.8x faster)** |
| 64      | ~72ms               | ~25ms        | **Gloo (2.9x faster)** |
| 128     | ~136ms              | ~28ms        | **Gloo (4.9x faster)** |

**Crossover point: ~6-8 nodes** (where Gloo's tree algorithms become faster)

**Note:** TCP's sequential nature means latency grows linearly (2ms per node), while Gloo uses tree algorithms (log growth). For typical ML clusters (8-16 nodes), the difference is small (~5ms), making TCP acceptable for simplicity.

### Impact on Window Duration

For 4-second windows:
- **Socket-based (TCP):** 16ms / 4000ms = **0.4% overhead** (negligible)
- **Gloo-based:** 20ms / 4000ms = **0.5% overhead** (negligible)

Both approaches are acceptable. **Socket-based TCP is preferred for typical cluster sizes** (≤16 nodes) due to:
- Simplicity (no library dependencies)
- Reliable delivery (guaranteed correctness)
- Easy debugging
- Comparable latency to Gloo

---

## Future Extensions

### 0. Persistent TCP Connections (Performance Optimization)

To reduce the +6ms TCP connection overhead per window, keep connections alive:

```cpp
class NetworkProbeManager {
 private:
  // NEW: Persistent connections for master sync
  std::vector<int> persistent_worker_conns_;  // Master's persistent connections
  int persistent_master_conn_ = -1;           // Worker's persistent connection
};

// Initialize persistent connections once during startup
absl::Status NetworkProbeManager::InitializePersistentConnections() {
  if (config_.node_id == master_node_id_) {
    // Master: accept N-1 persistent connections
    for (int i = 1; i < config_.num_nodes; ++i) {
      int conn = accept(master_listen_sock_, nullptr, nullptr);
      persistent_worker_conns_.push_back(conn);
    }
  } else {
    // Worker: establish persistent connection to master
    auto result = ConnectToMaster();
    if (result.ok()) {
      persistent_master_conn_ = *result;
    }
  }
}

// Modified sync: reuse connections
void NetworkProbeManager::PerformMasterCollection(const NodeWindowData& local_data) {
  // No accept() needed - use persistent_worker_conns_
  for (int conn : persistent_worker_conns_) {
    // Receive data, send sequence back
    // ...
  }
  // DON'T close connections
}
```

**Benefits:**
- Reduces latency from ~16ms to ~10ms (removes 3-way handshake)
- Comparable to UDP performance
- Still maintains TCP reliability

**Trade-offs:**
- ⚠️ Adds state management (connection keep-alive)
- ⚠️ Need heartbeats to detect dead connections
- ⚠️ Connection recovery on failure

**Recommendation:** Implement persistent connections if profiling shows >1% overhead from sync.

---

### 1. Outlier Detection

```cpp
GlobalWindowData NetworkProbeManager::MasterCalculation(
    const std::vector<NodeWindowData>& all_nodes_data) {
  
  // For each edge, compute median and MAD
  for (auto& [edge_key, alphas] : alpha_map) {
    double median_alpha = ComputeMedian(alphas);
    double mad_alpha = ComputeMAD(alphas, median_alpha);
    
    // Filter outliers (> 3 MAD from median)
    std::vector<double> filtered_alphas;
    for (double a : alphas) {
      if (std::abs(a - median_alpha) < 3 * mad_alpha) {
        filtered_alphas.push_back(a);
      }
    }
    
    double consensus_alpha = ComputeMean(filtered_alphas);
    // Store consensus_alpha for broadcast
  }
}
```

### 2. Adaptive Probing

```cpp
GlobalWindowData NetworkProbeManager::MasterCalculation(
    const std::vector<NodeWindowData>& all_nodes_data) {
  
  // Check stability of alpha/beta across last N windows
  bool all_stable = true;
  for (auto& [edge_key, alphas] : alpha_map) {
    double variance = ComputeVariance(alphas);
    if (variance > STABILITY_THRESHOLD) {
      all_stable = false;
      break;
    }
  }
  
  if (all_stable) {
    // Reduce probe cadence
    global.next_probe_cadence_us = 1600;  // 2x slower
  } else {
    // Increase probe cadence
    global.next_probe_cadence_us = 400;   // 2x faster
  }
  
  // Broadcast new cadence to all nodes
}
```

### 3. Global Clock Model

```cpp
// Build transitive clock offset graph
// offset[i][j] = alpha_ij * t_i + beta_ij
// Solve for global time reference using least-squares
Eigen::MatrixXd BuildClockOffsetMatrix(const GlobalWindowData& global);
Eigen::VectorXd SolveGlobalClockOffsets(const Eigen::MatrixXd& A);
```

---

## References

- **Gloo Documentation:** https://github.com/facebookincubator/gloo
- **Existing Gloo Integration:** `xla/backends/cpu/collectives/gloo_communicator.{h,cc}`
- **Window Manager:** `WINDOW_MANAGER_IMPLEMENTATION.md`
- **Probe Protocol:** `xla/backends/profiler/gpu/context/DIRECTED_PROBE_SPEC.md`
- **Port Assignment:** `xla/backends/profiler/gpu/context/CENTRALIZED_PORT_ASSIGNMENT.md`

---

## Summary

This plan adds **master-based synchronization** to the existing distributed profiling system with **two implementation options**:

### Key Components (Common to Both)

1. **Local synchronization:** `absl::Barrier` for threads within a node (existing)
2. **Cross-node synchronization:** Master-worker gather + broadcast pattern (new)
3. **Master calculation:** Placeholder for future global algorithms (outlier detection, consensus, adaptive control)
4. **Sequence-based rounds:** Master assigns sequence numbers to ensure all nodes proceed together

### Implementation Decision Matrix

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| **Cluster size ≤ 32 nodes** | **Socket-Based** | Lower latency, simpler code, reuses infrastructure |
| **Cluster size > 32 nodes** | **Gloo-Based** | Better scalability with tree algorithms |
| **Initial MVP** | **Socket-Based** | Faster development (6-7 days vs 14-23 days) |
| **Need AllReduce, consensus** | **Gloo-Based** | Built-in collective patterns |
| **Debugging/iteration speed** | **Socket-Based** | Easier to debug with tcpdump, strace |
| **Integration with PJRT** | **Gloo-Based** | Can share Gloo context with CPU collectives |

### Recommended Path

**Phase 1:** Implement **Socket-Based Variant B** (Master-Controlled Rounds) - 10-11 days
- Default for production because it tolerates clock drift and gives master control
- Includes data/serialization layer, control channel, and TCP data transfers
- Adds metrics for ROUND_END/START latency and queue depth

**Phase 2 (Optional A):** Backport **Variant A** (Time-Based) for MVPs - 2-3 additional days
- Only if you need the simplest possible setup for lab experiments
- Requires that clocks remain tightly synchronized

**Phase 2 (Optional B):** Migrate to **Gloo-Based** if needed - 7-9 additional days
- Only if cluster scales beyond 32 nodes
- Or if complex collective patterns are needed (AllReduce, consensus)
- Can reuse data structures and master calculation logic

### Next Steps

1. **Immediate:** Implement socket-based data/serialization layer (Phase 1a)
2. **Week 1:** Build master_sync control path + TCP data plane (Variant B Phases 2-5)
3. **Week 2:** Integration, testing, and validation (Phase 6)
4. **Evaluate:** After production deployment, measure actual latency and decide if Gloo upgrade is needed

---

**Status:** Planning Phase (all options documented)  
**Recommended:** Start with Socket-Based Variant B (Master-Controlled)  
**Next Step:** Phase 1 - Data Structures and Serialization

---

## Decision Tree

Use this decision tree to choose the right approach:

```
START: Do you need master-based synchronization?
  │
  ├─ NO → Use existing local-only window rotation (no cross-node sync)
  │
  └─ YES → 
      │
      ├─ Q1: What's your cluster size?
      │   ├─ ≤16 nodes → Socket-Based (Variant A or B)
      │   └─ >16 nodes → Consider Gloo-Based
      │
      ├─ Q2: Are node clocks tightly synchronized (≤10 µs skew)?
      │   ├─ YES → Variant A (Time-Based) ✔ (only if you accept drift risk)
      │   └─ NO / UNSURE → **Variant B (Master-Controlled)** ✅ DEFAULT
      │
      ├─ Q3: Do you need adaptive window sizing?
      │   ├─ NO → Variant A
      │   └─ YES → Variant B or Gloo
      │
      ├─ Q4: Do you need complex collectives (AllReduce, etc.)?
      │   ├─ NO → Socket-Based (A or B)
      │   └─ YES → Gloo-Based
      │
      └─ Q5: How important is development speed?
          ├─ Very important (MVP + aligned clocks) → Variant A (6-7 days)
          ├─ Moderate / clocks drift → **Variant B** (10-11 days)
          └─ Not critical → Gloo-Based (14-23 days)
```

### Quick Comparison Matrix

| Feature | Variant A | Variant B | Gloo |
|---------|-----------|-----------|------|
| **Window Control** | Time-based (fixed 4s) | Master-controlled (adaptive) | Master-controlled |
| **Latency** | 16ms | 22ms | 20-30ms |
| **Complexity** | Low | Medium | High |
| **Threads** | Probe threads only | + master_sync thread | + Gloo threads |
| **Control Messages** | None | ROUND_END + ROUND_START | Gloo protocol |
| **Development Time** | **6-7 days** | 10-11 days | 14-23 days |
| **Debugging Ease** | Easy | Medium | Hard |
| **Scalability** | Good (≤32 nodes) | Good (≤32 nodes) | Excellent (100+ nodes) |
| **Dependencies** | None | None | Gloo library |
| **Best For** | **MVP, fixed windows** | Adaptive control | Large clusters, complex collectives |

### Final Recommendation

For **most use cases**, start with **Variant B (Master-Controlled)**:
- ✅ Eliminates drift-induced stalls and guarantees coordinated rounds
- ✅ Still uses the same TCP data plane and serialization layer
- ✅ Enables emergency stop/start and adaptive window sizing
- ✅ Adds only moderate complexity (one extra thread) relative to Variant A

Use **Variant A** only for tightly controlled MVPs, and switch to **Gloo** when scaling past 32 nodes or when you must reuse PJRT collectives.

Only use **Gloo** if your cluster scales beyond 32 nodes.

---

## Recommended Implementation

**Evaluation snapshot**
- Document now defaults to Variant B to avoid drift-induced stalls; Variant A is relegated to tightly synchronized MVPs, while Gloo remains optional for >32 nodes.
- Data-plane primitives (`EdgeAlphaBeta`, `NodeWindowData`, serialization) — now including `round_id` — are universally required and pose the highest dependency risk; build them first and ensure JSONL export captures the new field.
- Transport-layer choices hinge on node count: ≤16 nodes favor the existing TCP stack, while the effort to integrate Gloo only pays off if >32 nodes or PJRT collectives reuse is mandatory.
- Test coverage is defined but not yet automated; the standalone 2-node test should be promoted to a gating check before expanding to SkyPilot.
- Master-side analytics should graduate into a dedicated `GraphCalc` utility so `NetworkProbeManager` stays focused on orchestration.

**Implementation plan (Variant B default)**
1. Land the shared data/serialization layer plus `WindowManager::CollectWindowData()` (Phase 1) — make `round_id` a first-class field in structs, serialization, and JSONL export.
2. Author `docs/graph_calc.md` and add `graph_calc.{h,cc}` with a `GraphCalc` class that ingests per-round data and encapsulates master-side math; refactor `MasterCalculation()` to call it.
3. Implement Socket Variant **B** end-to-end (Phases 2-5) by introducing the `master_sync` thread, control channel, and stop/start signalling while still using reliable TCP for data transfers; target 4-second nominal rounds with ≤25 ms sync budget and add metrics for ROUND_END/ROUND_START latency.
4. Ship the validation matrix in Phase 6 (unit + standalone + SkyPilot) and capture latency, failure-mode, JSONL size, and throughput numbers—these measurements gate any decision to keep connections persistent or switch to Gloo.
5. Define upgrade criteria in documentation: fall back to Variant A only for tightly synchronized MVPs; adopt Gloo only when (a) node count ≥32 or (b) PJRT collectives must share infrastructure.

**Exit criteria**
- Socket Variant **B** passes automated 2-node tests, 4-node SkyPilot, and meets <1% round-overhead target even with ±1 ms clock drift injected.
- Operational dashboards expose sequence skew, TCP retry counts, control-message latency, GraphCalc processing duration, and serialization sizes for three consecutive runs.
- Decision review scheduled after first production soak to reassess need for Variant A (for simpler deployments) or Gloo (for larger clusters) based on collected metrics.

