#ifndef XLA_BACKENDS_PROFILER_GPU_NETWORK_PROBE_H_
#define XLA_BACKENDS_PROFILER_GPU_NETWORK_PROBE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <condition_variable>
#include <deque>

#include <netinet/in.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/container/flat_hash_map.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/backends/profiler/gpu/probe_utils.h"

namespace xla::profiler {

struct SenderSession;

enum class ProbeMessageType : uint8_t {
  kPt1 = 0,
  kPt2 = 1,
  kPt3 = 2,
  kPr1 = 3,
  kPr2 = 4,
  kPd1 = 5,
  kPd2 = 6,
  kSyn = 7,  // Handshake: prober → listener
  kAck = 8,  // Handshake: listener → prober
};

struct ProbePacket {
  uint64_t embed1;
  uint64_t embed2;
  uint8_t type;
  uint8_t version;
  uint16_t src_node_id;      // Sender's node ID
  uint16_t dst_node_id;      // Receiver's node ID (for validation)
  uint32_t sequence_id;      // Triplet sequence number
  char marker[8];
};

struct EdgeStats {
  int successful_pairs = 0;
  int failed_pairs = 0;
  int probe_attempts = 0;
  int lost_packets = 0;
  double last_alpha = 0.0;
  double last_beta = 0.0;
};

struct ProbeThreads {
  std::thread fw_thread;
  std::thread bw_thread;
  bool direction = true; // true: forward, false: backward
};

struct ProbeQueueEntry {
  uint64_t pt1_rx;
  uint64_t pt2_rx;
  uint32_t seq_id;
};

struct ProbeQueue {
  std::deque<ProbeQueueEntry> queue;
  std::mutex mutex;
  std::condition_variable cond;
};

class NetworkProbeManager {
 public:
  explicit NetworkProbeManager(const DistributedProfilerContext& config);
  ~NetworkProbeManager();

  // Initialize sockets and graph based on config
  absl::Status Initialize();

  // Start the background probing loop
  absl::Status Start();

  // Perform one sync cycle (probe window)
  absl::Status Sync();

  // Export results to files (samples, alpha/beta, topology)
  absl::Status ExportData();

  // Shutdown and cleanup
  void Shutdown();

 private:
  struct NeighborSockets {
    // For OUT-neighbors (I probe them):
    int probe_sock = -1;          // Send Pt1/Pt2/Pt3
    int probe_response_sock = -1; // Recv Pr1/Pr2/Pr3
    uint16_t dst_listen_port = 0;
    sockaddr_in dst_addr{};
    
    // For IN-neighbors (They probe me):
    int listen_sock = -1;          // Recv Pt1/Pt2/Pt3
    int listen_response_sock = -1; // Send Pr1/Pr2/Pr3
    uint16_t my_listen_port = 0;
    sockaddr_in src_response_addr{};
  };

  DistributedProfilerContext config_;
  bool hw_timestamp_enabled_ = false;
  std::atomic<bool> running_{false};
  // std::map<int, std::thread> probe_threads_;    // Key: out-neighbor id
  // std::map<int, std::thread> listener_threads_; // Key: in-neighbor id
  absl::flat_hash_map<int, ProbeThreads> probe_threads_;    // Key: out-neighbor id
  absl::flat_hash_map<int, ProbeThreads> listener_threads_; // Key: in-neighbor id

  absl::flat_hash_map<int, std::unique_ptr<std::mutex>> probing_mutex_map_;  // Key: out-neighbor id
  absl::flat_hash_map<int, absl::flat_hash_map<uint32_t, probe_info::ProbePair>> probe_pairs_;    // Key: out-neighbor id, Key: sequence id, Value: probe pair

  absl::flat_hash_map<int, std::unique_ptr<std::mutex>> probing_handshake_mutex_map_;  // Key: out-neighbor id
  absl::flat_hash_map<int, std::unique_ptr<std::condition_variable>> probing_handshake_cond_map_;  // Key: out-neighbor id

  absl::flat_hash_map<int, bool> probing_handshake_done_map_;  // Key: out-neighbor id
  absl::flat_hash_map<int, bool> listener_handshake_done_map_;  // Key: in-neighbor id
  absl::flat_hash_map<int, std::unique_ptr<std::mutex>> listener_handshake_mutex_map_;  // Key: in-neighbor id
  absl::flat_hash_map<int, std::unique_ptr<std::condition_variable>> listener_handshake_cond_map_;  // Key: in-neighbor id

  std::vector<int> out_neighbors_;  // Who I probe
  std::vector<int> in_neighbors_;   // Who probes me
  absl::flat_hash_map<int, NeighborSockets> neighbor_socks_;  // Key: neighbor_id
  // std::atomic<uint32_t> seq_counter_{0};
  absl::flat_hash_map<int, std::unique_ptr<ProbeQueue>> probe_queue_map_;  // Key: out-neighbor id
  absl::flat_hash_map<int, std::unique_ptr<ProbeQueue>> listener_queue_map_; // Key: in-neighbor id
  absl::flat_hash_map<int, std::unique_ptr<std::atomic<uint32_t>>> seq_counter_map_;  // Key: out-neighbor id

  absl::flat_hash_map<std::pair<int, int>, EdgeStats> edge_stats_;
  absl::Mutex stats_mu_;

  absl::Status SetupSockets();
  absl::Status BuildGraph();
  absl::Status ComputeInNeighbors();  // Discover who probes me

  void ListenerLoop(int src_neighbor_id);  // Listen for probes FROM src (legacy)
  void ProbedListener(int src_neighbor_id);  // Listen for probes FROM src
  void ProbedResponder(int src_neighbor_id); // Send responses TO src
  void ProbeNeighbor(int dst_neighbor_id); // Send probes TO dst (legacy)
  void ProbeSender(int dst_neighbor_id); // Send probes TO dst
  void ProbeRespListener(int dst_neighbor_id); // Receive responses FROM dst
  void TrainAndStoreSVM(int dst_node_id, const absl::flat_hash_map<uint32_t, probe_info::ProbePair>& pairs);
  void UpdateEdgeStat(int dst_node_id, int lost, int success);
  
  // Handshake to synchronize prober and listener before starting
  bool PerformHandshake(int neighbor_id, bool is_prober);

  bool SendPacket(int sockfd, const sockaddr_in& dest, ProbeMessageType type,
                  uint64_t embed1, uint64_t embed2, uint32_t sequence_id,
                  uint64_t* send_ts_ns);
  bool RecvPacket(int sockfd, ProbePacket* pkt, uint64_t* recv_ts_ns,
                  sockaddr_in* sender);
};

}  // namespace xla::profiler

#endif  // XLA_BACKENDS_PROFILER_GPU_NETWORK_PROBE_H_
