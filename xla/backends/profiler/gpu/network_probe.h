#ifndef XLA_BACKENDS_PROFILER_GPU_NETWORK_PROBE_H_
#define XLA_BACKENDS_PROFILER_GPU_NETWORK_PROBE_H_

#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <condition_variable>
#include <deque>
#include <optional>

#include <netinet/in.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/profiler/gpu/probe_data_types.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/backends/profiler/gpu/probe_utils.h"

namespace xla::profiler {

class GraphCalc;

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

enum class SyncCommand : uint8_t {
  kRoundEnd = 1,
  kRoundStart = 2,
};

struct SyncMessage {
  SyncCommand command = SyncCommand::kRoundEnd;
  uint64_t sequence_number = 0;
  uint64_t timestamp_ns = 0;
};

// Window manager for shared window state across all probe threads
class WindowManager {
 public:
  struct EdgeWindowStats {
    double alpha = 0.0;
    double beta = 0.0;
    int pairs_collected = 0;
    int packets_lost = 0;
  };
  
  struct WindowStats {
    uint64_t window_start_ns = 0;
    uint64_t window_end_ns = 0;
    uint64_t window_id = 0;
    uint64_t round_id = 0;
    absl::flat_hash_map<int, EdgeWindowStats> edges;  // dst_id -> stats
  };
  
  explicit WindowManager(uint64_t window_duration_ns, int num_probe_threads);
  
  bool IsWindowExpired();
  
  // Barrier-based rotation: returns true if this thread is the last one (should rotate)
  bool NotifyWindowExpired();
  void RotateWindow();
  
  void RecordEdgeStats(int dst_id, double alpha, double beta, int pairs, int lost);
  WindowStats GetCurrentWindow();
  NodeWindowData CollectWindowData(int node_id);
  void SetCurrentRoundId(uint64_t round_id);
  uint64_t GetCurrentRoundId() const;
  uint64_t GetCurrentWindowId() const;
  
  // Export all accumulated windows to JSONL
  void ExportAllWindows(std::ofstream& out, int node_id,
                        const std::optional<uint64_t>& start_walltime_ns,
                        const std::optional<uint64_t>& start_gpu_ns);
  
 private:
  uint64_t window_duration_ns_;
  int num_probe_threads_;
  std::atomic<uint64_t> window_id_{0};
  std::atomic<uint64_t> window_start_ns_;
  std::atomic<uint64_t> window_end_ns_;
  std::atomic<uint64_t> current_round_id_{0};
  
  absl::Mutex mu_;
  WindowStats current_window_ ABSL_GUARDED_BY(mu_);
  std::vector<WindowStats> completed_windows_ ABSL_GUARDED_BY(mu_);
  
  // Barrier for window rotation
  std::shared_ptr<absl::Barrier> barrier_;
  absl::Mutex barrier_mu_;  // Protects barrier recreation
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
  std::vector<int> worker_participants_;
  absl::flat_hash_map<int, NeighborSockets> neighbor_socks_;  // Key: neighbor_id
  // std::atomic<uint32_t> seq_counter_{0};
  absl::flat_hash_map<int, std::unique_ptr<ProbeQueue>> probe_queue_map_;  // Key: out-neighbor id
  absl::flat_hash_map<int, std::unique_ptr<ProbeQueue>> listener_queue_map_; // Key: in-neighbor id
  absl::flat_hash_map<int, std::unique_ptr<std::atomic<uint32_t>>> seq_counter_map_;  // Key: out-neighbor id

  absl::flat_hash_map<std::pair<int, int>, EdgeStats> edge_stats_;
  absl::Mutex stats_mu_;
  
  // Shared window manager (holds all windows in memory until shutdown)
  // Thread-safe: multiple threads can call RotateWindow(), only one actually rotates
  std::unique_ptr<WindowManager> window_manager_;

  // Master synchronization (Variant B)
  bool enable_master_sync_ = false;
  int master_node_id_ = 0;
  uint16_t control_port_ = 0;
  uint16_t master_sync_port_ = 0;
  int control_sock_ = -1;
  int master_listen_sock_ = -1;
  std::vector<int> worker_sync_conns_;
  std::thread master_sync_thread_;
  std::atomic<bool> stop_probing_{false};
  std::atomic<uint64_t> current_sequence_{0};
  std::mutex round_mutex_;
  std::condition_variable round_start_cv_;
  std::mutex sync_mutex_;
  std::condition_variable sync_cv_;
  bool round_sync_pending_ = false;
  bool has_probe_senders_ = false;
  int expected_worker_reports_ = 0;

  std::unique_ptr<GraphCalc> graph_calc_;

  absl::Status SetupSockets();
  absl::Status BuildGraph();
  absl::Status ComputeInNeighbors();  // Discover who probes me
  absl::Status InitializeMasterSyncSockets();
  absl::Status PerformMasterSync(const NodeWindowData& local_data);
  void MasterSyncThread();
  void PerformMasterCollection(const NodeWindowData& local_data);
  void PerformWorkerSync(const NodeWindowData& local_data);
  absl::StatusOr<int> ConnectToMaster() const;
  ssize_t SendAll(int sockfd, const void* data, size_t len) const;
  ssize_t RecvAll(int sockfd, void* data, size_t len) const;

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

  std::string SerializeNodeWindowData(const NodeWindowData& data) const;
  absl::StatusOr<NodeWindowData> DeserializeNodeWindowData(const char* data,
                                                           size_t len) const;
};

}  // namespace xla::profiler

#endif  // XLA_BACKENDS_PROFILER_GPU_NETWORK_PROBE_H_
