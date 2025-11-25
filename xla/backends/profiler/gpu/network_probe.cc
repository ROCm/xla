#include "xla/backends/profiler/gpu/network_probe.h"

#include <arpa/inet.h>
#include <linux/net_tstamp.h>
#include <linux/sockios.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/profiler/gpu/probe_utils.h"
#include "xla/backends/profiler/gpu/graph_calc.h"
#include "xla/backends/profiler/gpu/svm_wrapper.h"
#include "xla/tsl/platform/logging.h"

namespace xla::profiler {

namespace {

constexpr uint16_t kBasePort = 20000;
constexpr uint16_t kPortsPerNode = 100;
constexpr uint64_t kPacketSpacingNs = 800000;  // 800 µs
constexpr int kRecvTimeoutMs = 10;  // 10ms timeout for missing packets
constexpr char kMarker[8] = "HUYGENS";

// Definition for kernel timestamp structure
struct scm_timestamping {
  struct timespec ts[3];  // [0]: software, [1]: deprecated, [2]: hardware
};

uint64_t GetSystemNs() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
         static_cast<uint64_t>(ts.tv_nsec);
}

sockaddr_in ParseAddress(const std::string& addr_str, uint16_t port) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  
  size_t colon = addr_str.find(':');
  std::string host = (colon != std::string::npos) ? addr_str.substr(0, colon) : addr_str;
  
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
    // Fallback to any
    addr.sin_addr.s_addr = INADDR_ANY;
  }
  
  return addr;
}

}  // namespace

// WindowManager implementation
WindowManager::WindowManager(uint64_t window_duration_ns, int num_probe_threads)
    : window_duration_ns_(window_duration_ns),
      num_probe_threads_(num_probe_threads) {
  window_start_ns_ = GetSystemNs();
  window_end_ns_ = window_start_ns_.load() + window_duration_ns_;
  current_window_.window_start_ns = window_start_ns_.load();
  current_window_.window_end_ns = window_end_ns_.load();
  current_window_.window_id = window_id_.load();
  current_window_.round_id = current_round_id_.load();
  
  // Create barrier for synchronizing window rotation
  barrier_ = std::make_unique<absl::Barrier>(num_probe_threads);
}

bool WindowManager::IsWindowExpired() {
  uint64_t now = GetSystemNs();
  return now > window_end_ns_.load();
}

bool WindowManager::NotifyWindowExpired() {
  // Block at barrier until all threads arrive
  bool am_last = barrier_->Block();
  
  if (am_last) {
    LOG(ERROR) << "Last thread at barrier, will rotate window";
  } else {
    VLOG(2) << "Thread released from barrier";
  }
  
  return am_last;
}

void WindowManager::RotateWindow() {
  // Rotate the window data
  {
    absl::MutexLock lock(&mu_);
    current_window_.window_start_ns = window_start_ns_.load();
    current_window_.window_end_ns = window_end_ns_.load();
    current_window_.window_id = window_id_.load();
    current_window_.round_id = current_round_id_.load();
    
    // Save current window to history if it has data
    if (!current_window_.edges.empty()) {
      WindowStats snapshot = current_window_;
      snapshot.window_start_ns = window_start_ns_.load();
      snapshot.window_end_ns = window_end_ns_.load();
      completed_windows_.push_back(std::move(snapshot));
      LOG(ERROR) << "Rotated window: saved window with " 
                << snapshot.edges.size() << " edges";
    }
    
    // Start new window with ATOMIC update of timestamps
    uint64_t now = GetSystemNs();
    window_start_ns_ = window_end_ns_.load();
    window_end_ns_ = now + window_duration_ns_;
    window_id_.fetch_add(1);
    current_window_.edges.clear();
    current_window_.window_start_ns = window_start_ns_.load();
    current_window_.window_end_ns = window_end_ns_.load();
    current_window_.window_id = window_id_.load();
    current_window_.round_id = current_round_id_.load();
  }
  
  // Recreate barrier for next window
  {
    absl::MutexLock lock(&barrier_mu_);
    barrier_ = std::make_unique<absl::Barrier>(num_probe_threads_);
    LOG(ERROR) << "Barrier recreated for next window";
  }
}

void WindowManager::RecordEdgeStats(int dst_id, double alpha, double beta, 
                                     int pairs, int lost) {
  absl::MutexLock lock(&mu_);
  current_window_.window_id = window_id_.load();
  current_window_.round_id = current_round_id_.load();
  current_window_.window_start_ns = window_start_ns_.load();
  current_window_.window_end_ns = window_end_ns_.load();
  current_window_.edges[dst_id] = {alpha, beta, pairs, lost};
}

WindowManager::WindowStats WindowManager::GetCurrentWindow() {
  absl::MutexLock lock(&mu_);
  WindowStats snapshot = current_window_;
  snapshot.window_start_ns = window_start_ns_.load();
  snapshot.window_end_ns = window_end_ns_.load();
  snapshot.window_id = current_window_.window_id;
  snapshot.round_id = current_window_.round_id;
  return snapshot;
}

NodeWindowData WindowManager::CollectWindowData(int node_id) {
  absl::MutexLock lock(&mu_);
  NodeWindowData data;
  data.node_id = node_id;
  data.window_id = current_window_.window_id;
  data.round_id = current_window_.round_id;
  data.window_start_ns = current_window_.window_start_ns;
  data.window_end_ns = current_window_.window_end_ns;
  data.edges.reserve(current_window_.edges.size());
  for (const auto& [dst_id, stats] : current_window_.edges) {
    EdgeAlphaBeta edge;
    edge.src_node_id = node_id;
    edge.dst_node_id = dst_id;
    edge.alpha = stats.alpha;
    edge.beta = stats.beta;
    edge.pairs_count = stats.pairs_collected;
    edge.lost_count = stats.packets_lost;
    data.edges.push_back(edge);
  }
  return data;
}

void WindowManager::SetCurrentRoundId(uint64_t round_id) {
  current_round_id_.store(round_id, std::memory_order_relaxed);
  absl::MutexLock lock(&mu_);
  current_window_.round_id = round_id;
}

uint64_t WindowManager::GetCurrentRoundId() const {
  return current_round_id_.load(std::memory_order_relaxed);
}

uint64_t WindowManager::GetCurrentWindowId() const {
  return window_id_.load(std::memory_order_relaxed);
}

void WindowManager::ExportAllWindows(
    std::ofstream& out, int node_id,
    const std::optional<uint64_t>& start_walltime_ns,
    const std::optional<uint64_t>& start_gpu_ns) {
  absl::MutexLock lock(&mu_);
  
  LOG(ERROR) << "Exporting " << completed_windows_.size() << " completed windows";

  if (start_walltime_ns.has_value() || start_gpu_ns.has_value()) {
    out << "{\"meta\":true,\"node_id\":" << node_id;
    if (start_walltime_ns.has_value()) {
      out << ",\"start_walltime_ns\":" << *start_walltime_ns;
    }
    if (start_gpu_ns.has_value()) {
      out << ",\"start_gpu_ns\":" << *start_gpu_ns;
    }
    out << "}\n";
  }
  
  // Export all completed windows
  for (const auto& window : completed_windows_) {
    out << "{\"window_start_ns\":" << window.window_start_ns
        << ",\"window_end_ns\":" << window.window_end_ns
        << ",\"window_id\":" << window.window_id
        << ",\"round_id\":" << window.round_id
        << ",\"node_id\":" << node_id
        << ",\"edges\":[";
    
    bool first = true;
    for (const auto& [dst_id, stats] : window.edges) {
      if (!first) out << ",";
      out << "{\"dst\":" << dst_id
          << ",\"alpha\":" << std::fixed << std::setprecision(10) << stats.alpha
          << ",\"beta\":" << static_cast<int64_t>(stats.beta)
          << ",\"pairs\":" << stats.pairs_collected
          << ",\"lost\":" << stats.packets_lost
          << "}";
      first = false;
    }
    out << "]}\n";  // Newline for JSONL format
  }
  
  // Export current window if it has data
  if (!current_window_.edges.empty()) {
    WindowStats final_window = current_window_;
    final_window.window_start_ns = window_start_ns_.load();
    final_window.window_end_ns = window_end_ns_.load();
    final_window.window_id = current_window_.window_id;
    final_window.round_id = current_window_.round_id;
    
    out << "{\"window_start_ns\":" << final_window.window_start_ns
        << ",\"window_end_ns\":" << final_window.window_end_ns
        << ",\"window_id\":" << final_window.window_id
        << ",\"round_id\":" << final_window.round_id
        << ",\"node_id\":" << node_id
        << ",\"edges\":[";
    
    bool first = true;
    for (const auto& [dst_id, stats] : final_window.edges) {
      if (!first) out << ",";
      out << "{\"dst\":" << dst_id
          << ",\"alpha\":" << std::fixed << std::setprecision(10) << stats.alpha
          << ",\"beta\":" << static_cast<int64_t>(stats.beta)
          << ",\"pairs\":" << stats.pairs_collected
          << ",\"lost\":" << stats.packets_lost
          << "}";
      first = false;
    }
    out << "]}\n";
  }
  
  out.flush();
  LOG(ERROR) << "Exported total " << (completed_windows_.size() + (current_window_.edges.empty() ? 0 : 1)) << " windows";
}

NetworkProbeManager::NetworkProbeManager(const DistributedProfilerContext& config)
    : config_(config) {}

NetworkProbeManager::~NetworkProbeManager() {
  Shutdown();
}

absl::Status NetworkProbeManager::Initialize() {
  LOG(ERROR) << "Initializing NetworkProbeManager for node " << config_.node_id;
  
  TF_RETURN_IF_ERROR(BuildGraph());
  TF_RETURN_IF_ERROR(ComputeInNeighbors());
  TF_RETURN_IF_ERROR(SetupSockets());
  
  // Create shared window manager with barrier for all probe threads
  int num_probe_threads = out_neighbors_.size();
  window_manager_ = std::make_unique<WindowManager>(
      config_.probe_window_s * 1'000'000'000ULL,
      num_probe_threads);
  
  LOG(ERROR) << "Window manager initialized with " << num_probe_threads 
            << " probe threads (will export to /tmp/probe_windows_node" 
            << config_.node_id << ".jsonl on shutdown)";

  has_probe_senders_ = config_.has_probe_senders || !out_neighbors_.empty();
  if (!has_probe_senders_ && !config_.probe_participants.empty()) {
    has_probe_senders_ = absl::c_linear_search(config_.probe_participants,
                                               config_.node_id);
  }
  
  if (config_.enable_master_sync) {
    TF_RETURN_IF_ERROR(InitializeMasterSyncSockets());
  }

  if (enable_master_sync_ && config_.node_id == master_node_id_) {
    GraphCalc::Config calc_config;
    calc_config.reference_node_id = master_node_id_;
    calc_config.num_nodes = config_.num_nodes;
    calc_config.min_pairs = 12;
    calc_config.max_loss_ratio = 0.6;
    graph_calc_ = std::make_unique<GraphCalc>(calc_config);
    LOG(INFO) << "GraphCalc initialized for master node " << master_node_id_
              << " (num_nodes=" << config_.num_nodes << ")";
  }
  
  // Initialize queues, counters, and mutexes for each neighbor
  for (int src : in_neighbors_) {
    listener_queue_map_[src] = std::make_unique<ProbeQueue>();
    probing_mutex_map_[src] = std::make_unique<std::mutex>();
    listener_handshake_cond_map_[src] = std::make_unique<std::condition_variable>();
    listener_handshake_mutex_map_[src] = std::make_unique<std::mutex>();
    listener_handshake_done_map_[src] = false;
  }
  for (int dst : out_neighbors_) {
    probe_queue_map_[dst] = std::make_unique<ProbeQueue>();
    seq_counter_map_[dst] = std::make_unique<std::atomic<uint32_t>>(0);
    probing_mutex_map_[dst] = std::make_unique<std::mutex>();
    probing_handshake_mutex_map_[dst] = std::make_unique<std::mutex>();
    probing_handshake_cond_map_[dst] = std::make_unique<std::condition_variable>();
    probing_handshake_done_map_[dst] = false;
  }
  
  // Start listener threads (one per IN-neighbor)
  running_ = true;
  
  LOG(ERROR) << "NetworkProbeManager initialized with " 
            << out_neighbors_.size() << " out-neighbors, " 
            << in_neighbors_.size() << " in-neighbors (" 
            << listener_threads_.size() << " listener threads, "
            << neighbor_socks_.size() << " socket entries)";
  
  return absl::OkStatus();
}

absl::Status NetworkProbeManager::Start() {
  if (!running_) {
    return absl::FailedPreconditionError("Manager not initialized");
  }
  
  // Start probe threads for each OUT-neighbor
  // TODO: make this loop async
  for (int dst : out_neighbors_) {
    // probe_threads_[dst] = std::thread(&NetworkProbeManager::ProbeNeighbor, this, dst);
    probe_threads_[dst].fw_thread = std::thread(&NetworkProbeManager::ProbeSender, this, dst);
  }
  for (int src : in_neighbors_) {
    // listener_threads_[src] = std::thread(&NetworkProbeManager::ListenerLoop, this, src);
    listener_threads_[src].fw_thread = std::thread(&NetworkProbeManager::ProbedListener, this, src);
    
  }
  for (int src: in_neighbors_){
    {
      std::unique_lock<std::mutex> lock(*listener_handshake_mutex_map_[src]);
      listener_handshake_cond_map_[src]->wait(lock, [this, src] { return listener_handshake_done_map_[src]; });
    }
    listener_threads_[src].bw_thread = std::thread(&NetworkProbeManager::ProbedResponder, this, src);
  }
  for(int dst : out_neighbors_){
    {
      std::unique_lock<std::mutex> lock(*probing_handshake_mutex_map_[dst]);
      probing_handshake_cond_map_[dst]->wait(lock, [this, dst] { return probing_handshake_done_map_[dst]; });
    }
    probe_threads_[dst].bw_thread = std::thread(&NetworkProbeManager::ProbeRespListener, this, dst);
  }
  
  if (enable_master_sync_) {
    master_sync_thread_ = std::thread(&NetworkProbeManager::MasterSyncThread, this);
  }
  
  LOG(ERROR) << "Started " << probe_threads_.size() << " probe threads";
  return absl::OkStatus();
}

absl::Status NetworkProbeManager::Sync() {
  // Wait for current window to complete (placeholder)
  return absl::OkStatus();
}

absl::Status NetworkProbeManager::SetupSockets() {
  // Port assignments are centrally allocated by master and stored in config_.edge_ports
  // No local calculation needed - just query the map
  
  LOG(ERROR) << "SetupSockets: node=" << config_.node_id 
          << " out_neighbors=" << out_neighbors_.size()
          << " in_neighbors=" << in_neighbors_.size()
          << " edge_ports=" << config_.edge_ports.size();
  
  int ts_flags = SOF_TIMESTAMPING_SOFTWARE | 
                 SOF_TIMESTAMPING_TX_SOFTWARE | 
                 SOF_TIMESTAMPING_RX_SOFTWARE;
  struct timeval tv = {.tv_sec = 0, .tv_usec = kRecvTimeoutMs * 1000};
  
  sockaddr_in bind_addr{};
  bind_addr.sin_family = AF_INET;
  bind_addr.sin_addr.s_addr = INADDR_ANY;
  
  // Create 2 sockets for each OUT-neighbor (I probe them)
  for (int dst : out_neighbors_) {
    NeighborSockets& socks = neighbor_socks_[dst];
    
    // Look up port assignment from config
    std::string edge_key = absl::StrCat("probe_edge:", config_.node_id, "->", dst);
    auto it = config_.edge_ports.find(edge_key);
    if (it == config_.edge_ports.end()) {
      return absl::InternalError(
          absl::StrCat("No port assignment found for OUT-edge ", edge_key));
    }
    uint16_t dst_listen_port = it->second.first;
    uint16_t my_response_port = it->second.second;
    
    socks.dst_listen_port = dst_listen_port;
    socks.dst_addr = ParseAddress(config_.node_addresses[dst], dst_listen_port);
    
    // Probe socket (send Pt)
    socks.probe_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (socks.probe_sock < 0) {
      return absl::InternalError(absl::StrCat("Failed to create probe socket for neighbor ", dst));
    }
    
    if (setsockopt(socks.probe_sock, SOL_SOCKET, SO_TIMESTAMPING, &ts_flags, sizeof(ts_flags)) < 0) {
      LOG(WARNING) << "Kernel SW timestamping not supported, using clock_gettime fallback";
      hw_timestamp_enabled_ = false;
    } else {
      hw_timestamp_enabled_ = true;
    }
    setsockopt(socks.probe_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
    bind_addr.sin_port = 0;  // Ephemeral
    if (bind(socks.probe_sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
      close(socks.probe_sock);
      return absl::InternalError(absl::StrCat("Failed to bind probe socket for ", dst));
    }
    
    // Probe response socket (recv Pr)
    socks.probe_response_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (socks.probe_response_sock < 0) {
      close(socks.probe_sock);
      return absl::InternalError(absl::StrCat("Failed to create probe response socket for ", dst));
    }
    
    setsockopt(socks.probe_response_sock, SOL_SOCKET, SO_TIMESTAMPING, &ts_flags, sizeof(ts_flags));
    setsockopt(socks.probe_response_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
    bind_addr.sin_port = htons(my_response_port);
    if (bind(socks.probe_response_sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
      close(socks.probe_sock);
      close(socks.probe_response_sock);
      return absl::InternalError(absl::StrCat("Failed to bind response socket to port ", my_response_port));
    }
    
    char dst_ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(socks.dst_addr.sin_addr), dst_ip_str, INET_ADDRSTRLEN);
    LOG(ERROR) << "Created OUT sockets for neighbor " << dst 
              << " probe_sock=" << socks.probe_sock
              << " probe_response_sock=" << socks.probe_response_sock
              << " (send to " << dst_ip_str << ":" << socks.dst_listen_port 
              << ", recv on port " << my_response_port << ")";
  }
  
  // Create 2 sockets for each IN-neighbor (they probe me)
  for (int src : in_neighbors_) {
    NeighborSockets& socks = neighbor_socks_[src];  // May already exist if bidirectional
    
    // Look up port assignment from config
    std::string edge_key = absl::StrCat("probe_edge:", src, "->", config_.node_id);
    auto it = config_.edge_ports.find(edge_key);
    if (it == config_.edge_ports.end()) {
      return absl::InternalError(
          absl::StrCat("No port assignment found for IN-edge ", edge_key));
    }
    uint16_t my_listen_port = it->second.first;
    uint16_t src_response_port = it->second.second;
    
    // Listen socket (recv Pt)
    socks.listen_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (socks.listen_sock < 0) {
      return absl::InternalError(absl::StrCat("Failed to create listen socket for neighbor ", src));
    }
    
    setsockopt(socks.listen_sock, SOL_SOCKET, SO_TIMESTAMPING, &ts_flags, sizeof(ts_flags));
    setsockopt(socks.listen_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
    socks.my_listen_port = my_listen_port;
    bind_addr.sin_port = htons(my_listen_port);
    if (bind(socks.listen_sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
      close(socks.listen_sock);
      return absl::InternalError(absl::StrCat("Failed to bind listen socket to port ", socks.my_listen_port));
    }
    
    // Listen response socket (send Pr)
    socks.listen_response_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (socks.listen_response_sock < 0) {
      close(socks.listen_sock);
      return absl::InternalError(absl::StrCat("Failed to create listen response socket for ", src));
    }
    
    setsockopt(socks.listen_response_sock, SOL_SOCKET, SO_TIMESTAMPING, &ts_flags, sizeof(ts_flags));
    setsockopt(socks.listen_response_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
    bind_addr.sin_port = 0;  // Ephemeral
    if (bind(socks.listen_response_sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
      close(socks.listen_sock);
      close(socks.listen_response_sock);
      return absl::InternalError(absl::StrCat("Failed to bind listen response socket for ", src));
    }
    
    socks.src_response_addr = ParseAddress(config_.node_addresses[src], src_response_port);
    
    char src_ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(socks.src_response_addr.sin_addr), src_ip_str, INET_ADDRSTRLEN);
    LOG(ERROR) << "Created IN sockets for neighbor " << src 
              << " listen_sock=" << socks.listen_sock
              << " listen_response_sock=" << socks.listen_response_sock
              << " (listen on port " << socks.my_listen_port 
              << ", send to " << src_ip_str << ":" << src_response_port << ")";
}

  return absl::OkStatus();
}

absl::Status NetworkProbeManager::BuildGraph() {
  out_neighbors_ = config_.neighbors;
  
  if (out_neighbors_.empty()) {
    LOG(ERROR) << "No out-neighbors configured for node " << config_.node_id;
  } else {
    LOG(ERROR) << "Node " << config_.node_id << " has " << out_neighbors_.size() 
              << " out-neighbors: " << absl::StrJoin(out_neighbors_, ", ");
  }
  
  return absl::OkStatus();
}

absl::Status NetworkProbeManager::ComputeInNeighbors() {
  // Discover in-neighbors by checking who has us in their out-neighbors list
  // This requires the full graph to be available in config
  
  in_neighbors_ = config_.in_neighbors;
  LOG(ERROR) << "Node " << config_.node_id << " has " << in_neighbors_.size() 
            << " in-neighbors: " << absl::StrJoin(in_neighbors_, ", ");
  
  return absl::OkStatus();
}

absl::Status NetworkProbeManager::InitializeMasterSyncSockets() {
  enable_master_sync_ = config_.enable_master_sync;
  master_node_id_ = config_.master_node_id;
  if (!enable_master_sync_) {
    return absl::OkStatus();
  }
  
  worker_participants_.clear();
  if (!config_.probe_participants.empty()) {
    worker_participants_ = config_.probe_participants;
  }
  absl::c_sort(worker_participants_);
  worker_participants_.erase(
      std::unique(worker_participants_.begin(), worker_participants_.end()),
      worker_participants_.end());
  if (worker_participants_.empty()) {
    for (int i = 0; i < config_.num_nodes; ++i) {
      worker_participants_.push_back(i);
    }
  }
  worker_participants_.erase(
      std::remove(worker_participants_.begin(), worker_participants_.end(),
                  master_node_id_),
      worker_participants_.end());
  expected_worker_reports_ = worker_participants_.size();
  
  control_port_ = config_.master_control_port + config_.node_id;
  master_sync_port_ = config_.master_sync_port;
  
  control_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (control_sock_ < 0) {
    return absl::InternalError("Failed to create master control socket");
  }
  
  int reuse = 1;
  setsockopt(control_sock_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  
  sockaddr_in control_bind{};
  control_bind.sin_family = AF_INET;
  control_bind.sin_addr.s_addr = INADDR_ANY;
  control_bind.sin_port = htons(control_port_);
  
  if (bind(control_sock_, reinterpret_cast<sockaddr*>(&control_bind),
           sizeof(control_bind)) < 0) {
    close(control_sock_);
    control_sock_ = -1;
    return absl::InternalError(
        absl::StrCat("Failed to bind control socket on port ", control_port_,
                     " errno=", errno));
  }
  
  struct timeval tv = {
      .tv_sec = static_cast<long>(config_.probe_window_s),
      .tv_usec = 0,
  };
  setsockopt(control_sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  
  if (config_.node_id == master_node_id_) {
    master_listen_sock_ = socket(AF_INET, SOCK_STREAM, 0);
    if (master_listen_sock_ < 0) {
      close(control_sock_);
      control_sock_ = -1;
      return absl::InternalError("Failed to create master sync TCP socket");
    }
    
    setsockopt(master_listen_sock_, SOL_SOCKET, SO_REUSEADDR, &reuse,
               sizeof(reuse));
    
    sockaddr_in listen_addr{};
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port = htons(master_sync_port_);
    
    if (bind(master_listen_sock_,
             reinterpret_cast<sockaddr*>(&listen_addr),
             sizeof(listen_addr)) < 0) {
      close(master_listen_sock_);
      master_listen_sock_ = -1;
      close(control_sock_);
      control_sock_ = -1;
      return absl::InternalError(
          absl::StrCat("Failed to bind master sync socket to port ",
                       master_sync_port_, " errno=", errno));
    }
    
    if (listen(master_listen_sock_, config_.num_nodes) < 0) {
      close(master_listen_sock_);
      master_listen_sock_ = -1;
      close(control_sock_);
      control_sock_ = -1;
      return absl::InternalError("Failed to listen on master sync socket");
    }
    
    LOG(INFO) << "Master node " << master_node_id_
              << " listening for worker sync on TCP port " << master_sync_port_
              << " expecting " << expected_worker_reports_ << " workers";
  }
  
  LOG(INFO) << "Control socket bound on port " << control_port_
            << " (enable_master_sync=" << enable_master_sync_ << ")";
  return absl::OkStatus();
}

absl::Status NetworkProbeManager::PerformMasterSync(
    const NodeWindowData& local_data) {
  if (!enable_master_sync_) {
    return absl::OkStatus();
  }
  if (!has_probe_senders_) {
    return absl::OkStatus();
  }
  
  if (config_.node_id == master_node_id_) {
    PerformMasterCollection(local_data);
    {
      std::lock_guard<std::mutex> lock(sync_mutex_);
      round_sync_pending_ = false;
    }
    sync_cv_.notify_all();
  } else {
    PerformWorkerSync(local_data);
  }
  
  return absl::OkStatus();
}

void NetworkProbeManager::PerformMasterCollection(
    const NodeWindowData& local_data) {
  if (master_listen_sock_ < 0) {
    LOG(WARNING) << "Master sync socket not initialized";
    return;
  }
  
  std::vector<NodeWindowData> aggregated;
  aggregated.reserve(config_.num_nodes);
  aggregated.push_back(local_data);
  
  int expected_workers = expected_worker_reports_;
  int received_workers = 0;
  while (received_workers < expected_workers && running_.load()) {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int conn = accept(master_listen_sock_,
                      reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (conn < 0) {
      LOG(WARNING) << "Master failed to accept worker sync connection: errno="
                   << errno;
      continue;
    }
    
    uint32_t payload_size = 0;
    if (RecvAll(conn, &payload_size, sizeof(payload_size)) < 0) {
      LOG(WARNING) << "Master failed to read payload size from worker";
      close(conn);
      continue;
    }
    if (payload_size == 0 || payload_size > (4 * 1024 * 1024)) {
      LOG(WARNING) << "Master received invalid payload size " << payload_size;
      close(conn);
      continue;
    }
    
    std::vector<char> payload(payload_size);
    if (RecvAll(conn, payload.data(), payload.size()) < 0) {
      LOG(WARNING) << "Master failed to read payload data";
      close(conn);
      continue;
    }
    
    auto worker_data =
        DeserializeNodeWindowData(payload.data(), payload.size());
    if (!worker_data.ok()) {
      LOG(WARNING) << "Failed to deserialize worker window: "
                   << worker_data.status();
    } else {
      aggregated.push_back(*worker_data);
    }
    
    close(conn);
    ++received_workers;
  }
  
  LOG(INFO) << "Master aggregated " << aggregated.size()
            << " node windows for sequence " << current_sequence_.load();

  GlobalWindowData global;
  if (!aggregated.empty()) {
    const NodeWindowData& baseline = aggregated.front();
    global.window_id = baseline.window_id;
    global.round_id = baseline.round_id;
  }
  global.sequence_number = current_sequence_.load();
  global.all_nodes = aggregated;
  global.window_start_ns.clear();
  global.window_end_ns.clear();
  global.window_start_ns.reserve(global.all_nodes.size());
  global.window_end_ns.reserve(global.all_nodes.size());
  for (const auto& node_window : global.all_nodes) {
    global.window_start_ns.push_back(node_window.window_start_ns);
    global.window_end_ns.push_back(node_window.window_end_ns);
  }

  if (graph_calc_) {
    auto round_result = graph_calc_->ProcessRound(global);
    if (!round_result.ok()) {
      LOG(WARNING) << "GraphCalc processing failed: "
                   << round_result.status();
    } else {
      const auto& summary = *round_result;
      LOG(INFO) << "GraphCalc summary: round=" << summary.round_id
                << " edges=" << summary.edges_used
                << " loops=" << summary.loops_constructed
                << " converged=" << summary.converged;
      if (!summary.converged && !summary.failure_reason.empty()) {
        LOG(WARNING) << "GraphCalc convergence issue: "
                     << summary.failure_reason;
      }
    }
  }
}

void NetworkProbeManager::PerformWorkerSync(
    const NodeWindowData& local_data) {
  if (!has_probe_senders_) {
    return;
  }
  auto conn_or = ConnectToMaster();
  if (!conn_or.ok()) {
    LOG(WARNING) << "Worker failed to connect to master: "
                 << conn_or.status();
    return;
  }
  int conn = *conn_or;
  
  std::string serialized = SerializeNodeWindowData(local_data);
  uint32_t payload_size = static_cast<uint32_t>(serialized.size());
  
  if (SendAll(conn, &payload_size, sizeof(payload_size)) < 0) {
    LOG(WARNING) << "Worker failed to send payload size to master";
    close(conn);
    return;
  }
  
  if (SendAll(conn, serialized.data(), serialized.size()) < 0) {
    LOG(WARNING) << "Worker failed to send payload data to master";
    close(conn);
    return;
  }
  
  close(conn);
  LOG(INFO) << "Worker " << config_.node_id << " sent "
            << serialized.size() << " bytes to master";
}

absl::StatusOr<int> NetworkProbeManager::ConnectToMaster() const {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    return absl::InternalError("Failed to create worker sync socket");
  }
  
  struct timeval tv = {.tv_sec = 5, .tv_usec = 0};
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  
  sockaddr_in master_addr =
      ParseAddress(config_.node_addresses[master_node_id_], master_sync_port_);
  if (connect(sock, reinterpret_cast<sockaddr*>(&master_addr),
              sizeof(master_addr)) < 0) {
    close(sock);
    return absl::UnavailableError(
        absl::StrCat("Failed to connect to master ", master_node_id_,
                     " errno=", errno));
  }
  return sock;
}

ssize_t NetworkProbeManager::SendAll(int sockfd, const void* data,
                                     size_t len) const {
  size_t total = 0;
  const char* ptr = static_cast<const char*>(data);
  while (total < len) {
    ssize_t sent = send(sockfd, ptr + total, len - total, 0);
    if (sent < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (sent == 0) {
      break;
    }
    total += sent;
  }
  return static_cast<ssize_t>(total);
}

ssize_t NetworkProbeManager::RecvAll(int sockfd, void* data,
                                     size_t len) const {
  size_t total = 0;
  char* ptr = static_cast<char*>(data);
  while (total < len) {
    ssize_t received = recv(sockfd, ptr + total, len - total, 0);
    if (received < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (received == 0) {
      return -1;
    }
    total += received;
  }
  return static_cast<ssize_t>(total);
}

void NetworkProbeManager::MasterSyncThread() {
  if (!enable_master_sync_) {
    return;
  }
  if (control_sock_ < 0) {
    LOG(WARNING) << "Master sync thread exiting: control socket not initialized";
    return;
  }
  
  LOG(INFO) << "Master sync thread started on node " << config_.node_id
            << " (master=" << master_node_id_ << ")";
  
  auto control_addr_for = [&](int node_id) {
    return ParseAddress(config_.node_addresses[node_id],
                        static_cast<uint16_t>(config_.master_control_port +
                                              node_id));
  };
  
  while (running_.load()) {
    if (config_.node_id == master_node_id_) {
      std::this_thread::sleep_for(
          std::chrono::seconds(config_.probe_window_s));
      if (!running_.load()) {
        break;
      }
      
      SyncMessage end_msg;
      end_msg.command = SyncCommand::kRoundEnd;
      end_msg.sequence_number = current_sequence_.load(
          std::memory_order_relaxed);
      end_msg.timestamp_ns = GetSystemNs();
      
      for (int node = 0; node < config_.num_nodes; ++node) {
        if (node == master_node_id_) continue;
        sockaddr_in dest = control_addr_for(node);
        sendto(control_sock_, &end_msg, sizeof(end_msg), 0,
               reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
      }
      
      stop_probing_.store(true, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lock(sync_mutex_);
        round_sync_pending_ = true;
      }
      {
        std::unique_lock<std::mutex> lock(sync_mutex_);
        sync_cv_.wait(lock, [this] {
          return !round_sync_pending_ || !running_.load();
        });
      }
      
      uint64_t new_seq =
          current_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
      SyncMessage start_msg;
      start_msg.command = SyncCommand::kRoundStart;
      start_msg.sequence_number = new_seq;
      start_msg.timestamp_ns = GetSystemNs();
      
      if (window_manager_) {
        window_manager_->SetCurrentRoundId(new_seq);
      }
      
      for (int node = 0; node < config_.num_nodes; ++node) {
        if (node == master_node_id_) continue;
        sockaddr_in dest = control_addr_for(node);
        sendto(control_sock_, &start_msg, sizeof(start_msg), 0,
               reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
      }
      
      stop_probing_.store(false, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lock(round_mutex_);
        round_start_cv_.notify_all();
      }
    } else {
      SyncMessage msg{};
      sockaddr_in sender{};
      socklen_t sender_len = sizeof(sender);
      ssize_t bytes =
          recvfrom(control_sock_, &msg, sizeof(msg), 0,
                   reinterpret_cast<sockaddr*>(&sender), &sender_len);
      if (bytes < 0) {
        if (!running_.load()) {
          break;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          continue;
        }
        LOG(WARNING) << "Worker control recv failed: errno=" << errno;
        continue;
      }
      if (bytes != sizeof(msg)) {
        LOG(WARNING) << "Worker received partial control message (bytes="
                     << bytes << ")";
        continue;
      }
      
      if (msg.command == SyncCommand::kRoundEnd) {
        stop_probing_.store(true, std::memory_order_relaxed);
      } else if (msg.command == SyncCommand::kRoundStart) {
        current_sequence_.store(msg.sequence_number,
                                std::memory_order_relaxed);
        if (window_manager_) {
          window_manager_->SetCurrentRoundId(msg.sequence_number);
        }
        stop_probing_.store(false, std::memory_order_relaxed);
        {
          std::lock_guard<std::mutex> lock(round_mutex_);
          round_start_cv_.notify_all();
        }
      }
    }
  }
  
  LOG(INFO) << "Master sync thread exiting on node " << config_.node_id;
}

// Helper: Send packet with timestamping
bool NetworkProbeManager::SendPacket(int sockfd, const sockaddr_in& dest,
                                     ProbeMessageType type,
                                     uint64_t embed1, uint64_t embed2,
                                     uint32_t sequence_id,
                                     uint64_t* send_ts_ns) {
  ProbePacket pkt{};
  pkt.embed1 = embed1;
  pkt.embed2 = embed2;
  pkt.type = static_cast<uint8_t>(type);
  pkt.version = 1;
  pkt.src_node_id = static_cast<uint16_t>(config_.node_id);
  pkt.dst_node_id = 0;  // Set by caller if needed
  pkt.sequence_id = sequence_id;
  std::memcpy(pkt.marker, kMarker, sizeof(kMarker));
  
  // Try sendmsg for kernel timestamps
  struct iovec iov = {.iov_base = &pkt, .iov_len = sizeof(pkt)};
  struct msghdr msg{};
  msg.msg_name = const_cast<sockaddr_in*>(&dest);
  msg.msg_namelen = sizeof(dest);
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  
  ssize_t sent = sendmsg(sockfd, &msg, 0);
  if (sent < 0) {
    LOG(ERROR) << "sendmsg failed: errno=" << errno << " (" << strerror(errno) 
            << "), type=" << static_cast<int>(type) << ", seq=" << sequence_id
            << ", sockfd=" << sockfd;
    // Fallback to sendto + clock_gettime
    // ssize_t fallback = sendto(sockfd, &pkt, sizeof(pkt), 0, (struct sockaddr*)&dest, sizeof(dest));
    // if (fallback < 0) {
    //   LOG(ERROR) << "sendto also failed: errno=" << errno << " (" << strerror(errno) << ")";
    // }
    *send_ts_ns = GetSystemNs();
    return false;
  }
  
  // Try to read TX timestamp from error queue
  char ctrl[256];
  struct iovec iov_err = {.iov_base = ctrl, .iov_len = sizeof(ctrl)};
  struct msghdr msg_err{};
  msg_err.msg_iov = &iov_err;
  msg_err.msg_iovlen = 1;
  msg_err.msg_control = ctrl;
  msg_err.msg_controllen = sizeof(ctrl);
  
  ssize_t err_len = recvmsg(sockfd, &msg_err, MSG_ERRQUEUE | MSG_DONTWAIT);
  if (err_len >= 0) {
    for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg_err); cmsg != nullptr;
         cmsg = CMSG_NXTHDR(&msg_err, cmsg)) {
      if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_TIMESTAMPING) {
        struct scm_timestamping* tss = (struct scm_timestamping*)CMSG_DATA(cmsg);
        *send_ts_ns = tss->ts[0].tv_sec * 1e9 + tss->ts[0].tv_nsec;
      return true;
    }
  }
  }
  
  // Fallback
  *send_ts_ns = GetSystemNs();
  return false;
}

// Helper: Receive packet with timestamping
bool NetworkProbeManager::RecvPacket(int sockfd, ProbePacket* pkt,
                                     uint64_t* recv_ts_ns,
                                     sockaddr_in* sender) {
  if (sockfd < 0) {
    LOG(ERROR) << "RecvPacket called with invalid sockfd=" << sockfd;
    return false;
  }
  
  char ctrl[256];
  struct iovec iov = {.iov_base = pkt, .iov_len = sizeof(*pkt)};
  struct msghdr msg{};
  msg.msg_name = sender;
  msg.msg_namelen = sender ? sizeof(*sender) : 0;
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl;
  msg.msg_controllen = sizeof(ctrl);
  
  ssize_t len = recvmsg(sockfd, &msg, 0);
  if (len < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      VLOG(2) << "recvmsg timeout on sockfd=" << sockfd;
      return false;
    }
    LOG(ERROR) << "recvmsg failed: len=" << len << " errno=" << errno << " (" << strerror(errno) 
            << "), sockfd=" << sockfd;
    // Fallback to recvfrom
    socklen_t addrlen = sender ? sizeof(*sender) : 0;
    len = recvfrom(sockfd, pkt, sizeof(*pkt), 0, (struct sockaddr*)sender,
                   sender ? &addrlen : nullptr);
    if (len < 0) {
      LOG(ERROR) << "recvfrom also failed: len=" << len << " errno=" << errno << " (" << strerror(errno) << ")";
      return false;
    }
    *recv_ts_ns = GetSystemNs();
    return false;
  }
  
  if (len != sizeof(*pkt)) {
    LOG(ERROR) << "recvmsg received partial packet: len=" << len << " expected=" << sizeof(*pkt) 
            << " sockfd=" << sockfd;
    return false;
  }
  
  // Extract RX timestamp from ancillary data
  for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr;
       cmsg = CMSG_NXTHDR(&msg, cmsg)) {
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_TIMESTAMPING) {
      struct scm_timestamping* tss = (struct scm_timestamping*)CMSG_DATA(cmsg);
      *recv_ts_ns = tss->ts[0].tv_sec * 1e9 + tss->ts[0].tv_nsec;
      return true;
    }
  }
  
  // Fallback
  *recv_ts_ns = GetSystemNs();
  return false;
}

bool NetworkProbeManager::PerformHandshake(int neighbor_id, bool is_prober) {
  auto it = neighbor_socks_.find(neighbor_id);
  if (it == neighbor_socks_.end()) {
    LOG(ERROR) << "No socket info for neighbor " << neighbor_id;
    return false;
  }
  
  constexpr int kHandshakeRetries = 5;
  constexpr int kHandshakeTimeoutMs = 1000;
  
  if (is_prober) {
    // PROBER: Send SYN, wait for ACK
    int probe_sock = it->second.probe_sock;
    int response_sock = it->second.probe_response_sock;
    sockaddr_in dst_addr = it->second.dst_addr;
    
    char dst_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &dst_addr.sin_addr, dst_ip, INET_ADDRSTRLEN);
    
    LOG(ERROR) << "Handshake: Prober sending SYN to neighbor " << neighbor_id
              << " via probe_sock=" << probe_sock
              << " to " << dst_ip << ":" << ntohs(dst_addr.sin_port)
              << ", waiting for ACK on response_sock=" << response_sock;
    
    for (int retry = 0; retry < kHandshakeRetries; ++retry) {
      // Send SYN
      uint64_t syn_tx;
      if (!SendPacket(probe_sock, dst_addr, ProbeMessageType::kSyn, 
                      0, 0, 0, &syn_tx)) {
        LOG(WARNING) << "Failed to send SYN to neighbor " << neighbor_id 
                     << " retry=" << retry;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        continue;
      }
      
      LOG(ERROR) << "Handshake: SYN sent to neighbor " << neighbor_id << ", waiting for ACK...";
      
      // Wait for ACK
      ProbePacket ack;
      uint64_t ack_rx;
      if (RecvPacket(response_sock, &ack, &ack_rx, nullptr)) {
        LOG(ERROR) << "Handshake: Received packet type=" << static_cast<int>(ack.type)
                  << " from node=" << ack.src_node_id
                  << " (expected type=8, node=" << neighbor_id << ")";
        if (ack.type == static_cast<uint8_t>(ProbeMessageType::kAck) &&
            ack.src_node_id == neighbor_id) {
          LOG(ERROR) << "Handshake: Received ACK from neighbor " << neighbor_id;
          return true;
        } else {
          LOG(WARNING) << "Handshake: Wrong packet type or source";
        }
      } else {
        LOG(WARNING) << "Handshake: RecvPacket timeout/failed for neighbor " << neighbor_id;
      }
      
      LOG(WARNING) << "Handshake: No ACK from neighbor " << neighbor_id 
                   << " retry=" << retry;
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
    
    LOG(ERROR) << "Handshake FAILED: No ACK from neighbor " << neighbor_id 
               << " after " << kHandshakeRetries << " retries";
    return false;
    
  } else {
    // LISTENER: Wait for SYN, send ACK
    int listen_sock = it->second.listen_sock;
    int response_sock = it->second.listen_response_sock;
    sockaddr_in src_response_addr = it->second.src_response_addr;
    
    char resp_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &src_response_addr.sin_addr, resp_ip, INET_ADDRSTRLEN);
    
    LOG(ERROR) << "Handshake: Listener waiting for SYN from neighbor " << neighbor_id
              << " on listen_sock=" << listen_sock
              << ", will ACK via response_sock=" << response_sock
              << " to " << resp_ip << ":" << ntohs(src_response_addr.sin_port);
    
    for (int retry = 0; retry < kHandshakeRetries; ++retry) {
      ProbePacket syn;
      uint64_t syn_rx;
      sockaddr_in sender_addr;
      
      if (RecvPacket(listen_sock, &syn, &syn_rx, &sender_addr)) {
        char sender_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &sender_addr.sin_addr, sender_ip, INET_ADDRSTRLEN);
        
        LOG(ERROR) << "Handshake: Listener received packet type=" << static_cast<int>(syn.type)
                  << " from node=" << syn.src_node_id
                  << " sender=" << sender_ip << ":" << ntohs(sender_addr.sin_port)
                  << " (expected type=7, node=" << neighbor_id << ")";
        
        if (syn.type == static_cast<uint8_t>(ProbeMessageType::kSyn) &&
            syn.src_node_id == neighbor_id) {
          LOG(ERROR) << "Handshake: Received SYN from neighbor " << neighbor_id;
          
          // Send ACK
          uint64_t ack_tx;
          if (SendPacket(response_sock, src_response_addr, ProbeMessageType::kAck,
                        0, 0, 0, &ack_tx)) {
            LOG(ERROR) << "Handshake: Sent ACK to neighbor " << neighbor_id
                      << " to " << resp_ip << ":" << ntohs(src_response_addr.sin_port);
            return true;
          } else {
            LOG(ERROR) << "Failed to send ACK to neighbor " << neighbor_id;
            return false;
      }
    } else {
          LOG(WARNING) << "Handshake: Wrong packet type or source, ignoring";
    }
  } else {
        LOG(WARNING) << "Handshake: RecvPacket timeout on listen_sock, retry=" << retry;
      }
      
      // Timeout, wait longer
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
    
    LOG(ERROR) << "Handshake FAILED: No SYN from neighbor " << neighbor_id 
               << " after " << kHandshakeRetries << " retries";
    return false;
  }
}

void NetworkProbeManager::ProbeSender(int dst_node_id) {
  LOG(ERROR) << "ProbeSender thread started for OUT-neighbor " << dst_node_id;
  
  auto it = neighbor_socks_.find(dst_node_id);
  if (it == neighbor_socks_.end() || it->second.probe_sock < 0) {
    LOG(ERROR) << "No probe socket found for neighbor " << dst_node_id;
    return;
  }
  
  int probe_sock = it->second.probe_sock;
  int response_sock = it->second.probe_response_sock;
  sockaddr_in dst_addr = it->second.dst_addr;
  
  LOG(ERROR) << "Prober using probe_sock=" << probe_sock 
          << " response_sock=" << response_sock 
          << " dst_listen_port=" << it->second.dst_listen_port
          << " dst_addr=" << inet_ntoa(dst_addr.sin_addr) << ":" << ntohs(dst_addr.sin_port);
  
  // Perform handshake before starting probe loop
  if (!PerformHandshake(dst_node_id, true)) {
    LOG(ERROR) << "Handshake failed with neighbor " << dst_node_id << ", aborting prober";
    return;
  }
  {
    std::unique_lock<std::mutex> lock(*probing_handshake_mutex_map_[dst_node_id]);
    probing_handshake_done_map_[dst_node_id] = true;
  }
  probing_handshake_cond_map_[dst_node_id]->notify_one();
  
  LOG(ERROR) << "Handshake complete with neighbor " << dst_node_id << ", entering probe loop";
  
  while (running_.load()) {
    bool window_expired = window_manager_->IsWindowExpired();
    bool master_stop = enable_master_sync_ && stop_probing_.load();
    bool should_rotate = enable_master_sync_ ? master_stop : window_expired;
    if (should_rotate) {
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[dst_node_id]);
      
       // Train SVM on current window's pairs
       auto& pairs_map = probe_pairs_[dst_node_id];
       if (!pairs_map.empty()) {
         // Filter to only complete pairs
         absl::flat_hash_map<uint32_t, probe_info::ProbePair> complete_pairs;
         for (const auto& [seq_id, pair] : pairs_map) {
           // Check if pair is complete (all fields non-zero)
           if (pair.pt1_tx > 0 && pair.pt2_tx > 0 &&
               pair.pt1_rx > 0 && pair.pt2_rx > 0 &&
               pair.pr1_tx > 0 && pair.pr2_tx > 0 &&
               pair.pr1_rx > 0 && pair.pr2_rx > 0) {
             complete_pairs[seq_id] = pair;
           }
         }
         
         if (complete_pairs.size() >= 10) {
           auto prob_info = probe_utils::convert_probe_pairs_to_xy_pairs(
               complete_pairs, 0.2, true);
          if (!prob_info.points.empty()) {
            SVMModel svm_model;
            if (svm_model.train(prob_info)) {
              double alpha = svm_model.getAlpha();
              double beta = svm_model.getBeta();
              
              // Record stats in shared window manager
              int pairs_count = complete_pairs.size();
              auto stat_key = std::make_pair(config_.node_id, dst_node_id);
              int lost_count = edge_stats_[stat_key].lost_packets;
              window_manager_->RecordEdgeStats(dst_node_id, alpha, beta, 
                                               pairs_count, lost_count);
              
              LOG(ERROR) << "Edge " << config_.node_id << "->" << dst_node_id
                        << " window stats: α=" << alpha << " β=" << beta 
                        << " pairs=" << pairs_count << " lost=" << lost_count;
            }
          }
        }
      }
      
      probe_pairs_[dst_node_id].clear();
      
      // Notify window expiry and check if this is the last thread
      if (window_manager_->NotifyWindowExpired()) {
        // This is the last thread - rotate the window
        NodeWindowData local_data =
            window_manager_->CollectWindowData(config_.node_id);
        auto status = PerformMasterSync(local_data);
        if (!status.ok()) {
          LOG(ERROR) << "Master synchronization failed: " << status;
        }
        window_manager_->RotateWindow();
        LOG(ERROR) << "Thread for dst=" << dst_node_id 
                  << " was last at barrier, rotated window";
      } else {
        VLOG(1) << "Thread for dst=" << dst_node_id 
                << " reached barrier, waiting for others";
      }
      
      if (enable_master_sync_ && stop_probing_.load()) {
        std::unique_lock<std::mutex> round_lock(round_mutex_);
        round_start_cv_.wait(
            round_lock, [this] { return !stop_probing_.load(); });
      }
    }
    
    // Generate unique sequence ID
    uint32_t seq_id = seq_counter_map_[dst_node_id]->fetch_add(1);
    
    // Send Pt1/Pt2/Pt3
    uint64_t pt1_tx, pt2_tx;
    if (!SendPacket(probe_sock, dst_addr, ProbeMessageType::kPt1,
                    0, 0, seq_id, &pt1_tx)) {
      LOG(ERROR) << "Failed to send Pt1 to neighbor " << dst_node_id << " seq=" << seq_id;
      UpdateEdgeStat(dst_node_id, 1, 0);  // lost_packets++
    continue;
  }
    else{
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[dst_node_id]);
      probe_pairs_[dst_node_id][seq_id].pt1_tx = pt1_tx;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(kPacketSpacingNs));
    
    if (!SendPacket(probe_sock, dst_addr, ProbeMessageType::kPt2,
                    pt1_tx, 0, seq_id, &pt2_tx)) {
      LOG(ERROR) << "Failed to send Pt2 to neighbor " << dst_node_id << " seq=" << seq_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    else{
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[dst_node_id]);
      probe_pairs_[dst_node_id][seq_id].pt2_tx = pt2_tx;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
}
    

void NetworkProbeManager::ProbeRespListener(int dst_node_id) {
  
  LOG(ERROR) << "Probe Listener thread started for OUT-neighbor " << dst_node_id;
  
  auto it = neighbor_socks_.find(dst_node_id);
  if (it == neighbor_socks_.end() || it->second.probe_sock < 0) {
    LOG(ERROR) << "No probe socket found for neighbor " << dst_node_id;
    return;
  }
  
  int probe_sock = it->second.probe_sock;
  int response_sock = it->second.probe_response_sock;
  sockaddr_in dst_addr = it->second.dst_addr;
  
  LOG(ERROR) << "Prober using probe_sock=" << probe_sock 
          << " response_sock=" << response_sock 
          << " dst_listen_port=" << it->second.dst_listen_port
          << " dst_addr=" << inet_ntoa(dst_addr.sin_addr) << ":" << ntohs(dst_addr.sin_port);
  
  // uint64_t window_start_ns = GetSystemNs();
  // uint64_t window_end_ns = window_start_ns + config_.probe_window_s * 1'000'000'000ULL;
  while (running_.load()) {
    // Receive Pr (must match seq_id)
    ProbePacket pr;
    uint64_t pr_rx;
    if (!RecvPacket(response_sock, &pr, &pr_rx, nullptr)) {
      continue;  // Timeout - packet lost
    }
    if (pr.type == static_cast<uint8_t>(ProbeMessageType::kPr1)) {
      auto pt1_rx = pr.embed2;
      auto seq_id = pr.sequence_id;
      // Note: dst_node_id is the function parameter, not from packet
      auto pr1_rx = pr_rx;
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[dst_node_id]);
      probe_pairs_[dst_node_id][seq_id].pt1_rx = pt1_rx;
      probe_pairs_[dst_node_id][seq_id].pr1_rx = pr1_rx;
    }
    else if (pr.type == static_cast<uint8_t>(ProbeMessageType::kPr2)) {
      auto pt2_rx = pr.embed2;
      auto seq_id = pr.sequence_id;
      auto pr1_tx = pr.embed1;
      auto pr2_rx = pr_rx;
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[dst_node_id]);
      if(probe_pairs_[dst_node_id][seq_id].pr1_rx > 0) {
        probe_pairs_[dst_node_id][seq_id].pt2_rx = pt2_rx;
        probe_pairs_[dst_node_id][seq_id].pr2_rx = pr2_rx;
        probe_pairs_[dst_node_id][seq_id].pr1_tx = pr1_tx;
      }
    }
    else if (pr.type == static_cast<uint8_t>(ProbeMessageType::kPd1)) {
      auto seq_id = pr.sequence_id;
      auto pr2_tx = pr.embed1;
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[dst_node_id]);
      probe_pairs_[dst_node_id][seq_id].pr2_tx = pr2_tx;
    }
    // std::this_thread::sleep_for(std::chrono::nanoseconds(kPacketSpacingNs));
  }
}

void NetworkProbeManager::ProbedResponder(int src_neighbor_id) {
  
  int response_sock = neighbor_socks_[src_neighbor_id].listen_response_sock;
  sockaddr_in src_response_addr = neighbor_socks_[src_neighbor_id].src_response_addr;
  
  while (running_.load()) {
    // read from queue when queue is not empty
    // wait for condition variable when queue is empty
    // generate Pr1/Pr2/Pd1 responses
    uint64_t pt1_rx, pt2_rx;
    uint32_t seq_id;
    {
      std::unique_lock<std::mutex> lock(listener_queue_map_[src_neighbor_id]->mutex);
      listener_queue_map_[src_neighbor_id]->cond.wait(lock, [this, src_neighbor_id] { 
        return !running_.load() || !listener_queue_map_[src_neighbor_id]->queue.empty(); 
      });
      
      // Check if we're shutting down
      if (!running_.load()) {
        break;
      }
      
      auto entry = listener_queue_map_[src_neighbor_id]->queue.front();
      pt1_rx = entry.pt1_rx;
      pt2_rx = entry.pt2_rx;
      seq_id = entry.seq_id;
      listener_queue_map_[src_neighbor_id]->queue.pop_front();
    }
    // generate Pr1/Pr2/Pd1 responses
    uint64_t pr1_tx;
    if (!SendPacket(response_sock, src_response_addr, ProbeMessageType::kPr1,
               0, pt1_rx, seq_id, &pr1_tx)) {
      LOG(ERROR) << "Failed to send Pr1 to neighbor " << src_neighbor_id << " seq=" << seq_id;
      continue;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(kPacketSpacingNs));
    uint64_t pr2_tx;
    if (!SendPacket(response_sock, src_response_addr, ProbeMessageType::kPr2,
               pr1_tx, pt2_rx, seq_id, &pr2_tx)) {
      LOG(ERROR) << "Failed to send Pr2 to neighbor " << src_neighbor_id << " seq=" << seq_id;
      continue;
    }
    uint64_t pd1_tx;
    if (!SendPacket(response_sock, src_response_addr, ProbeMessageType::kPd1,
               pr2_tx, 0, seq_id, &pd1_tx)) {
      LOG(ERROR) << "Failed to send Pd1 to neighbor " << src_neighbor_id << " seq=" << seq_id;
      continue;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
  
  LOG(ERROR) << "Listener thread exiting for IN-neighbor " << src_neighbor_id;
  
}

void NetworkProbeManager::ProbedListener(int src_neighbor_id) {
  LOG(ERROR) << "Listener thread started for IN-neighbor " << src_neighbor_id;
  
  auto it = neighbor_socks_.find(src_neighbor_id);
  if (it == neighbor_socks_.end() || it->second.listen_sock < 0) {
    LOG(ERROR) << "No listen socket found for neighbor " << src_neighbor_id;
    return;
  }
  
  int listen_sock = it->second.listen_sock;
  int response_sock = it->second.listen_response_sock;
  sockaddr_in src_response_addr = it->second.src_response_addr;
  
  LOG(ERROR) << "Listener using listen_sock=" << listen_sock 
          << " response_sock=" << response_sock 
          << " my_listen_port=" << it->second.my_listen_port;
  
  // Perform handshake before starting probe loop
  if (!PerformHandshake(src_neighbor_id, false)) {
    LOG(ERROR) << "Handshake failed with neighbor " << src_neighbor_id << ", aborting listener";
    return;
  }
  {
    std::unique_lock<std::mutex> lock(*listener_handshake_mutex_map_[src_neighbor_id]);
    listener_handshake_done_map_[src_neighbor_id] = true;
  }
  listener_handshake_cond_map_[src_neighbor_id]->notify_one();
  LOG(ERROR) << "Handshake complete with neighbor " << src_neighbor_id << ", entering probe loop";
  
  while (running_.load()) {
    // Receive Pt1 from this specific neighbor
    ProbePacket pt;
    uint64_t pt_rx;
    sockaddr_in sender_addr;
    if (!RecvPacket(listen_sock, &pt, &pt_rx, &sender_addr)) {
      continue;  // Timeout - this is normal, just continue
    }
    uint32_t seq_id = pt.sequence_id;
    if(pt.type == static_cast<uint8_t>(ProbeMessageType::kPt1)) {
      auto pt1_rx = pt_rx;
      auto seq_id = pt.sequence_id;
      // Note: src_neighbor_id is the function parameter, not from packet
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[src_neighbor_id]);
      probe_pairs_[src_neighbor_id][seq_id].pt1_rx = pt1_rx;
    }
    else if(pt.type == static_cast<uint8_t>(ProbeMessageType::kPt2)) {
      auto pt2_rx = pt_rx;
      auto seq_id = pt.sequence_id;
      std::scoped_lock<std::mutex> lock(*probing_mutex_map_[src_neighbor_id]);
      probe_pairs_[src_neighbor_id][seq_id].pt2_rx = pt2_rx;
      if(probe_pairs_[src_neighbor_id][seq_id].pt1_rx > 0) {
        auto pt1_rx = probe_pairs_[src_neighbor_id][seq_id].pt1_rx;
        {
          std::unique_lock<std::mutex> lock(listener_queue_map_[src_neighbor_id]->mutex);
          listener_queue_map_[src_neighbor_id]->queue.push_back({pt1_rx, pt2_rx, seq_id});
        }
        listener_queue_map_[src_neighbor_id]->cond.notify_one();
      }
    }
    // wait for condition variable when queue is empty
    std::this_thread::sleep_for(std::chrono::nanoseconds(kPacketSpacingNs));
  }
}

void NetworkProbeManager::ListenerLoop(int src_neighbor_id) {
  LOG(ERROR) << "Listener thread started for IN-neighbor " << src_neighbor_id;
  
  auto it = neighbor_socks_.find(src_neighbor_id);
  if (it == neighbor_socks_.end() || it->second.listen_sock < 0) {
    LOG(ERROR) << "No listen socket found for neighbor " << src_neighbor_id;
    return;
  }
  
  int listen_sock = it->second.listen_sock;
  int response_sock = it->second.listen_response_sock;
  sockaddr_in src_response_addr = it->second.src_response_addr;
  
  LOG(ERROR) << "Listener using listen_sock=" << listen_sock 
          << " response_sock=" << response_sock 
          << " my_listen_port=" << it->second.my_listen_port;
  
  // Perform handshake before starting probe loop
  if (!PerformHandshake(src_neighbor_id, false)) {
    LOG(ERROR) << "Handshake failed with neighbor " << src_neighbor_id << ", aborting listener";
    return;
  }
  
  LOG(ERROR) << "Handshake complete with neighbor " << src_neighbor_id << ", entering probe loop";
  
  while (running_.load()) {
    // Receive Pt1 from this specific neighbor
    ProbePacket pt1;
    uint64_t t1_rx;
    sockaddr_in sender_addr;
    
    if (!RecvPacket(listen_sock, &pt1, &t1_rx, &sender_addr)) {
      // Timeout - this is normal, just continue
      continue;
    }
    
    if (pt1.type != static_cast<uint8_t>(ProbeMessageType::kPt1)) {
      continue;  // Out of order, skip
    }
    
    uint32_t seq_id = pt1.sequence_id;
    
    // Receive Pt2 (must match seq_id)
    ProbePacket pt2;
    uint64_t t2_rx;
    if (!RecvPacket(listen_sock, &pt2, &t2_rx, nullptr)) {
      continue;  // Timeout - packet lost
    }
    if (pt2.type != static_cast<uint8_t>(ProbeMessageType::kPt2) ||
        pt2.sequence_id != seq_id) {
      continue;  // Sequence mismatch
    }
    uint64_t t1_tx = pt2.embed1;
    
    // Receive Pt3
    
    
    // Send Pr1/Pr2/Pr3 responses to sender's response port
    uint64_t pr1_tx, pr2_tx, pd1_tx;
    SendPacket(response_sock, src_response_addr, ProbeMessageType::kPr1,
               0, t1_rx, seq_id, &pr1_tx);
    std::this_thread::sleep_for(std::chrono::nanoseconds(kPacketSpacingNs));
    
    SendPacket(response_sock, src_response_addr, ProbeMessageType::kPr2,
               pr1_tx, t2_rx, seq_id, &pr2_tx);    
    
    // uint64_t flag = pure_pt ? (1ULL << 63) : 0ULL;
    SendPacket(response_sock, src_response_addr, ProbeMessageType::kPd1,
               pr2_tx, 0, seq_id, &pd1_tx);
    ProbePacket pd2;
    uint64_t t3_rx;
    if (!RecvPacket(listen_sock, &pd2, &t3_rx, nullptr)) {
      continue;  // Timeout - packet lost
    }
    if (pd2.type != static_cast<uint8_t>(ProbeMessageType::kPd2) ||
        pd2.sequence_id != seq_id) {
      continue;  // Sequence mismatch
    }
    uint64_t t2_tx = pd2.embed1;
  }
  
  LOG(ERROR) << "Listener thread exiting for IN-neighbor " << src_neighbor_id;
}

void NetworkProbeManager::ProbeNeighbor(int dst_node_id) {
  LOG(ERROR) << "Probe thread started for OUT-neighbor " << dst_node_id;
  
  auto it = neighbor_socks_.find(dst_node_id);
  if (it == neighbor_socks_.end() || it->second.probe_sock < 0) {
    LOG(ERROR) << "No probe socket found for neighbor " << dst_node_id;
    return;
  }
  
  int probe_sock = it->second.probe_sock;
  int response_sock = it->second.probe_response_sock;
  sockaddr_in dst_addr = it->second.dst_addr;
  
  LOG(ERROR) << "Prober using probe_sock=" << probe_sock 
          << " response_sock=" << response_sock 
          << " dst_listen_port=" << it->second.dst_listen_port
          << " dst_addr=" << inet_ntoa(dst_addr.sin_addr) << ":" << ntohs(dst_addr.sin_port);
  
  // Perform handshake before starting probe loop
  if (!PerformHandshake(dst_node_id, true)) {
    LOG(ERROR) << "Handshake failed with neighbor " << dst_node_id << ", aborting prober";
    return;
  }
  
  LOG(ERROR) << "Handshake complete with neighbor " << dst_node_id << ", entering probe loop";
  
  uint64_t window_start_ns = GetSystemNs();
  uint64_t window_end_ns = window_start_ns + config_.probe_window_s * 1'000'000'000ULL;
  
  // std::vector<probe_info::ProbePair> pairs;
  
  while (running_.load()) {
    uint64_t now_ns = GetSystemNs();
    if (now_ns > window_end_ns) {
      // End of window: train SVM
      // TrainAndStoreSVM(dst_node_id, pairs);
      // pairs.clear();
      window_start_ns = now_ns;
      window_end_ns = now_ns + config_.probe_window_s * 1'000'000'000ULL;
    }
    
    // Generate unique sequence ID
    uint32_t seq_id = seq_counter_map_[dst_node_id]->fetch_add(1);
    
    // Send Pt1/Pt2/Pt3
    uint64_t pt1_tx, pt2_tx;
    if (!SendPacket(probe_sock, dst_addr, ProbeMessageType::kPt1,
                    0, 0, seq_id, &pt1_tx)) {
      LOG(ERROR) << "Failed to send Pt1 to neighbor " << dst_node_id << " seq=" << seq_id;
      UpdateEdgeStat(dst_node_id, 1, 0);  // lost_packets++
      continue;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(kPacketSpacingNs));
    
    if (!SendPacket(probe_sock, dst_addr, ProbeMessageType::kPt2,
                    pt1_tx, 0, seq_id, &pt2_tx)) {
      LOG(ERROR) << "Failed to send Pt2 to neighbor " << dst_node_id << " seq=" << seq_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }


    
    // Receive Pr1/Pr2/Pr3 responses (on dedicated response socket)
    ProbePacket pr1, pr2, pd1, pd2;
    uint64_t pr1_rx, pr2_rx, pd1_rx, pd2_tx;
    
    bool recv_ok = RecvPacket(response_sock, &pr1, &pr1_rx, nullptr);
    if (!recv_ok) {
      LOG(ERROR) << "Failed to recv Pr1 from neighbor " << dst_node_id << " seq=" << seq_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    if (pr1.sequence_id != seq_id) {
      LOG(ERROR) << "Pr1 seq mismatch: expected=" << seq_id << " got=" << pr1.sequence_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    if (pr1.src_node_id != dst_node_id) {
      LOG(ERROR) << "Pr1 src mismatch: expected=" << dst_node_id << " got=" << pr1.src_node_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    if(pr1.type != static_cast<uint8_t>(ProbeMessageType::kPr1)) {
      LOG(ERROR) << "Pr1 type mismatch: expected=3 got=" << static_cast<int>(pr1.type);
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    
    recv_ok = RecvPacket(response_sock, &pr2, &pr2_rx, nullptr);
    if (!recv_ok) {
      LOG(ERROR) << "Failed to recv Pr2 from neighbor " << dst_node_id << " seq=" << seq_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    if (pr2.sequence_id != seq_id) {
      LOG(ERROR) << "Pr2 seq mismatch: expected=" << seq_id << " got=" << pr2.sequence_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    if (pr2.src_node_id != dst_node_id) {
      LOG(ERROR) << "Pr2 src mismatch: expected=" << dst_node_id << " got=" << pr2.src_node_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    if(pr2.type != static_cast<uint8_t>(ProbeMessageType::kPr2)) {
      LOG(ERROR) << "Pr2 type mismatch: expected=4 got=" << static_cast<int>(pr2.type);
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }

    recv_ok = RecvPacket(response_sock, &pd1, &pd1_rx, nullptr);
    if (!recv_ok) {
      LOG(ERROR) << "Failed to recv Pd1 from neighbor " << dst_node_id << " seq=" << seq_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    if (pd1.sequence_id != seq_id) {
      LOG(ERROR) << "Pd1 seq mismatch: expected=" << seq_id << " got=" << pd1.sequence_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    if (pd1.src_node_id != dst_node_id) {
      LOG(ERROR) << "Pd1 src mismatch: expected=" << dst_node_id << " got=" << pd1.src_node_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }
    if(pd1.type != static_cast<uint8_t>(ProbeMessageType::kPd1)) {
      LOG(ERROR) << "Pd1 type mismatch: expected=5 got=" << static_cast<int>(pd1.type);
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }


    if (!SendPacket(probe_sock, dst_addr, ProbeMessageType::kPd2,
      0, 0, seq_id, &pd2_tx)) {
      LOG(ERROR) << "Failed to send Pd2 to neighbor " << dst_node_id << " seq=" << seq_id;
      UpdateEdgeStat(dst_node_id, 1, 0);
      continue;
    }

    // Extract embedded data
    // uint64_t pr1_tx = pr1.embed1;
    uint64_t pt1_rx = pr1.embed2;
    uint64_t pr1_tx = pr2.embed1;
    uint64_t pt2_rx = pr2.embed2;
    uint64_t pr2_tx = pd1.embed1;

    int64_t dt_sender = pt2_tx - pt1_tx;
    int64_t dt_receiver = pt2_rx - pt1_rx;
    bool pure_pt = std::abs(dt_receiver - dt_sender) < 50000;
    
    // Compute pure_pr
    int64_t dr_sender = pr2_tx - pr1_tx;
    int64_t dr_receiver = pr2_rx - pr1_rx;
    bool pure_pr = std::abs(dr_sender - dr_receiver) < 50000;
    
    // Store pair
    probe_info::ProbePair pair{};
    pair.pt1_tx = pt1_tx;
    pair.pt2_tx = pt2_tx;
    pair.pt1_rx = pt1_rx;
    pair.pt2_rx = pt2_rx;
    pair.pr1_tx = pr1_tx;
    pair.pr2_tx = pr2_tx;
    pair.pr1_rx = pr1_rx;
    pair.pr2_rx = pr2_rx;
    pair.pure_pt = pure_pt;
    pair.pure_pr = pure_pr;
    // pairs.push_back(pair);
    
    UpdateEdgeStat(dst_node_id, 0, 1);  // successful_pairs++
    // VLOG(2) << "Successfully collected pair for edge " << config_.node_id << "->" << dst_node_id 
    //         << " seq=" << seq_id << " (total=" << pairs.size() << ")";
    
    // Sleep until next probe
    std::this_thread::sleep_for(std::chrono::microseconds(config_.probe_cadence_us));
  }
  
  LOG(ERROR) << "Probe thread exiting for OUT-neighbor " << dst_node_id;
}

std::string NetworkProbeManager::SerializeNodeWindowData(
    const NodeWindowData& data) const {
  std::string buffer;
  buffer.reserve(sizeof(NodeWindowData) +
                 data.edges.size() * sizeof(EdgeAlphaBeta));
  auto append = [&buffer](const void* src, size_t len) {
    buffer.append(static_cast<const char*>(src), len);
  };
  
  append(&data.node_id, sizeof(data.node_id));
  append(&data.window_id, sizeof(data.window_id));
  append(&data.round_id, sizeof(data.round_id));
  append(&data.window_start_ns, sizeof(data.window_start_ns));
  append(&data.window_end_ns, sizeof(data.window_end_ns));
  
  uint64_t edge_count = data.edges.size();
  append(&edge_count, sizeof(edge_count));
  for (const auto& edge : data.edges) {
    append(&edge, sizeof(edge));
  }
  
  return buffer;
}

absl::StatusOr<NodeWindowData> NetworkProbeManager::DeserializeNodeWindowData(
    const char* data, size_t len) const {
  auto has_bytes = [len](size_t offset, size_t bytes) {
    return offset + bytes <= len;
  };
  
  size_t offset = 0;
  NodeWindowData window;
  auto read_into = [&](void* dst, size_t bytes) -> absl::Status {
    if (!has_bytes(offset, bytes)) {
      return absl::InternalError(
          "Serialized NodeWindowData unexpectedly truncated");
    }
    std::memcpy(dst, data + offset, bytes);
    offset += bytes;
    return absl::OkStatus();
  };
  
  TF_RETURN_IF_ERROR(read_into(&window.node_id, sizeof(window.node_id)));
  TF_RETURN_IF_ERROR(read_into(&window.window_id, sizeof(window.window_id)));
  TF_RETURN_IF_ERROR(read_into(&window.round_id, sizeof(window.round_id)));
  TF_RETURN_IF_ERROR(
      read_into(&window.window_start_ns, sizeof(window.window_start_ns)));
  TF_RETURN_IF_ERROR(
      read_into(&window.window_end_ns, sizeof(window.window_end_ns)));
  
  uint64_t edge_count = 0;
  TF_RETURN_IF_ERROR(read_into(&edge_count, sizeof(edge_count)));
  if (!has_bytes(offset, edge_count * sizeof(EdgeAlphaBeta))) {
    return absl::InternalError(
        "Serialized NodeWindowData edge payload truncated");
  }
  
  window.edges.resize(edge_count);
  for (uint64_t i = 0; i < edge_count; ++i) {
    TF_RETURN_IF_ERROR(read_into(&window.edges[i], sizeof(EdgeAlphaBeta)));
  }
  
  return window;
}

void NetworkProbeManager::TrainAndStoreSVM(
    int dst_node_id,
    const absl::flat_hash_map<uint32_t, probe_info::ProbePair>& pairs) {
  if (pairs.size() < 10) {
    LOG(WARNING) << "Insufficient pairs for SVM training: " << pairs.size();
    return;
  }

  auto probed_pairs_info = probe_utils::convert_probe_pairs_to_xy_pairs(pairs, 300000);
  auto& xy_pairs = probed_pairs_info.points;
  if (xy_pairs.empty()) {
    return;
  }
  
  SVMModel svm_model;
  if (!svm_model.train(probed_pairs_info)) {
    LOG(WARNING) << "SVM training failed for edge " << config_.node_id 
                 << "->" << dst_node_id;
    return;
  }
  
  double alpha = svm_model.getAlpha();
  double beta = svm_model.getBeta();
  
  {
    absl::MutexLock lock(&stats_mu_);
    auto key = std::make_pair(config_.node_id, dst_node_id);
    edge_stats_[key].last_alpha = alpha;
    edge_stats_[key].last_beta = beta;
    edge_stats_[key].successful_pairs += xy_pairs.size();
  }
  
  LOG(ERROR) << "Edge " << config_.node_id << "->" << dst_node_id
            << ": α=" << alpha << ", β=" << beta << " ns (" 
            << pairs.size() << " pairs)";
}

void NetworkProbeManager::UpdateEdgeStat(int dst_node_id, int lost, int success) {
  absl::MutexLock lock(&stats_mu_);
  auto key = std::make_pair(config_.node_id, dst_node_id);
  edge_stats_[key].lost_packets += lost;
  edge_stats_[key].successful_pairs += success;
}

absl::Status NetworkProbeManager::ExportData() {
  std::string output_dir = "/tmp";  // TODO: Read from config
  
  // Export α/β summary
  std::string summary_file = absl::StrCat(output_dir, "/alpha_beta_node",
                                          config_.node_id, ".csv");
  std::ofstream summary_out(summary_file);
  summary_out << "src,dst,alpha,beta_ns,num_pairs,lost_packets\n";
  
  {
    absl::MutexLock lock(&stats_mu_);
    for (const auto& [edge, stats] : edge_stats_) {
      summary_out << edge.first << "," << edge.second << ","
                  << stats.last_alpha << "," << stats.last_beta << ","
                  << stats.successful_pairs << "," << stats.lost_packets << "\n";
    }
  }
  
  summary_out.close();
  LOG(ERROR) << "Exported probe data to " << summary_file;

  return absl::OkStatus();
}

void NetworkProbeManager::Shutdown() {
  LOG(ERROR) << "Shutdown initiated...";
  running_.store(false);
  
  if (control_sock_ >= 0) {
    close(control_sock_);
    control_sock_ = -1;
  }
  if (master_listen_sock_ >= 0) {
    close(master_listen_sock_);
    master_listen_sock_ = -1;
  }
  
  // Wake up all condition variables waiting in ProbedResponder threads
  for (auto& [neighbor_id, queue] : listener_queue_map_) {
    queue->cond.notify_all();
  }
  for (auto& [neighbor_id, queue] : probe_queue_map_) {
    queue->cond.notify_all();
  }
  
  // Close all sockets FIRST to unblock recv() calls immediately
  LOG(ERROR) << "Closing sockets...";
  for (auto& [neighbor_id, socks] : neighbor_socks_) {
    if (socks.probe_sock >= 0) {
      close(socks.probe_sock);
      socks.probe_sock = -1;
    }
    if (socks.probe_response_sock >= 0) {
      close(socks.probe_response_sock);
      socks.probe_response_sock = -1;
    }
    if (socks.listen_sock >= 0) {
      close(socks.listen_sock);
      socks.listen_sock = -1;
    }
    if (socks.listen_response_sock >= 0) {
      close(socks.listen_response_sock);
      socks.listen_response_sock = -1;
    }
  }
  
  // Now join listener threads (for IN-neighbors)
  LOG(ERROR) << "Joining listener threads...";
  for (auto& [neighbor_id, threads] : listener_threads_) {
    if (threads.fw_thread.joinable()) {
      threads.fw_thread.join();
    }
    if (threads.bw_thread.joinable()) {
      threads.bw_thread.join();
    }
  }
  
  // Join probe threads (for OUT-neighbors)
  LOG(ERROR) << "Joining probe threads...";
  for (auto& [neighbor_id, threads] : probe_threads_) {
    if (threads.fw_thread.joinable()) {
      threads.fw_thread.join();
    }
    if (threads.bw_thread.joinable()) {
      threads.bw_thread.join();
    }
  }
  
  if (master_sync_thread_.joinable()) {
    master_sync_thread_.join();
  }
  
  // Export ALL accumulated windows when shutdown
  if (window_manager_) {
    std::string export_file = absl::StrCat("/tmp/probe_windows_node", 
                                           config_.node_id, ".jsonl");
    std::ofstream out(export_file, std::ios::out | std::ios::trunc);  // Overwrite mode
    if (out.is_open()) {
      LOG(ERROR) << "Exporting all accumulated windows to " << export_file;
      window_manager_->ExportAllWindows(out, config_.node_id,
                                        config_.collector_start_walltime_ns,
                                        config_.collector_start_gpu_ns);
      out.close();
      LOG(ERROR) << "Export complete";
    } else {
      LOG(ERROR) << "Failed to open export file: " << export_file;
    }
  }
  
  LOG(ERROR) << "NetworkProbeManager shutdown complete";
}

}  // namespace xla::profiler
