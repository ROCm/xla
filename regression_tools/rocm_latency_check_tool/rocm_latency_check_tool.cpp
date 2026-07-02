// ROCm Latency Check Tool: HIP-runtime latency probe.
// Per-request pipeline: H2D -> timedBusyKernel -> D2D -> timedBusyKernel -> D2H
// (two synthetic compute stages that mimic a DL/LLM workload).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <cstring>

// Runtime configuration. Every field is overridable via a command-line flag
// (see parse_args / --help); the initializers below are the defaults and match
// the values this tool historically hard-coded.
struct Config {
  int num_gpus = 4;                 // --num-gpus
  int streams_per_gpu = 4;          // --streams-per-gpu (per-GPU slots)
  bool use_reduced_d2h = true;      // --reduced-d2h / --no-reduced-d2h
  size_t reduced_d2h_bytes = 11800; // --reduced-d2h-bytes
  int duration_sec = 10;            // --duration-sec (total run length)
  int report_interval_sec = 3;      // --report-interval-sec (print/CSV cadence)
  int target_qps = 0;               // --target-qps (0 = max throughput, no pacing)
  size_t max_queue_size = 10000;    // --max-queue-size
  int consumer_threads = 16;        // --consumer-threads
  size_t d2d_bytes = 0;             // --d2d-bytes (0 = mirror input B size)
  bool d2d_peer = false;            // --d2d-peer (cross-GPU D2D; needs >=2 GPUs)
  std::string csv_path;             // --csv (empty = CSV disabled)
  std::string label;                // --label (ROCm release tag, e.g. 7.2.1)

  // Total number of slots across all GPUs (old NUM_SLOTS).
  int total_slots() const { return num_gpus * streams_per_gpu; }
};

// Parsed once at the very start of main() before any GPU resources exist.
Config g_cfg;

// The synthetic compute op is a fixed on-GPU busy kernel. These are compile-time
// constants by design (no flags, no env) so the compute baseline is fixed and
// fully reproducible across ROCm versions; only HIP runtime + copy overhead varies.
constexpr long long kBusyKernelDurationNs = 80000;  // ~80us synthetic compute
constexpr int kBusyKernelThreads = 256;
constexpr int kBusyKernelBlocks = 0;                // 0 = auto (use CU count)
constexpr bool kBusyKernelUseSleep = true;

// When true, weights A are copied to GPU once at startup (matches TF/JAX).
// When false, weights A are H2D'd every request (~24MB extra SDMA stress).
constexpr bool kPreloadWeights = false;  // false = H2D weights A (~24MB) per request for SDMA stress

// Extra H2D copies per request to simulate additional TF/JAX/XLA model tensor traffic.
// kExtraH2DBytes: size of each extra copy (0 = disabled; e.g., 24*1024*1024)
// kExtraH2DCount: number of separate copies (each is a distinct hipMemcpyAsync)
constexpr size_t kExtraH2DBytes = 0;
constexpr int    kExtraH2DCount = 1;


// When true, skips the actual hipMemcpyAsync call — isolates SDMA vs compute impact
constexpr bool kSkipH2D = false;
constexpr bool kSkipD2H = false;

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      std::cerr << "HIP error (" << __FILE__ << ":" << __LINE__ << "): "      \
                << hipGetErrorString(error) << "\n";                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// RAII wrapper for hipHostMalloc pinned memory (matches TF/JAX/XLA allocate memory on the host)
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t bytes, uint8_t fill = 0) : size_(bytes) {
    if (bytes > 0) {
      HIP_CHECK(hipHostMalloc(&ptr_, bytes, hipHostMallocPortable));
      std::memset(ptr_, fill, bytes);
    }
  }
  ~PinnedBuffer() { if (ptr_) (void)hipHostFree(ptr_); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) { o.ptr_ = nullptr; o.size_ = 0; }
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) { if (ptr_) (void)hipHostFree(ptr_); ptr_ = o.ptr_; size_ = o.size_; o.ptr_ = nullptr; o.size_ = 0; }
    return *this;
  }

  uint8_t* data() { return static_cast<uint8_t*>(ptr_); }
  const uint8_t* data() const { return static_cast<const uint8_t*>(ptr_); }
  size_t size() const { return size_; }
  uint8_t& operator[](size_t i) { return data()[i]; }
  const uint8_t& operator[](size_t i) const { return data()[i]; }
  uint8_t* begin() { return data(); }
  uint8_t* end() { return data() + size_; }
  const uint8_t* begin() const { return data(); }
  const uint8_t* end() const { return data() + size_; }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

// Workload sizing: the m/n/k shape only determines the H2D/D2D/D2H copy sizes
// (the compute is the synthetic busy kernel, not a GEMM).
struct WorkloadConfig {
  int64_t m, n, k;
  int64_t stride_a, stride_b, stride_c;
  bool trans_a, trans_b;               // true = transposed orientation for sizing
  int64_t batch_count;
  size_t elem_bytes_a, elem_bytes_b, elem_bytes_c;
};

struct GpuResources {
  // 1 HIP stream per slot — all ops (H2D, compute, D2D, D2H) serialized on same
  // stream. This aliases h2d/d2h/d2d to compute stream; confirmed by rocprofv3.
  static constexpr int kStreamsPerSlot = 1;
  int num_slots;    // per-GPU slots (= g_cfg.streams_per_gpu)
  int num_streams;  // num_slots * kStreamsPerSlot
  int device_id;
  int wall_clock_rate_khz = 0;
  int multiprocessor_count = 1;
  std::vector<hipStream_t> streams;

  hipStream_t h2d_stream(int slot) const { return streams[slot]; }
  hipStream_t compute_stream(int slot) const { return streams[slot]; }
  hipStream_t d2h_stream(int slot) const { return streams[slot]; }
  hipStream_t d2d_stream(int slot) const { return streams[slot]; }

  explicit GpuResources(int dev_id)
      : num_slots(g_cfg.streams_per_gpu),
        num_streams(g_cfg.streams_per_gpu * kStreamsPerSlot),
        device_id(dev_id) {
    HIP_CHECK(hipSetDevice(device_id));
    HIP_CHECK(hipDeviceGetAttribute(&wall_clock_rate_khz,
                                    hipDeviceAttributeClockRate,
                                    device_id));
    HIP_CHECK(hipDeviceGetAttribute(&multiprocessor_count,
                                    hipDeviceAttributeMultiprocessorCount,
                                    device_id));
    streams.resize(num_streams);
    for (int i = 0; i < num_streams; ++i) {
      HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamDefault));
    }
  }

  ~GpuResources() {
    HIP_CHECK(hipSetDevice(device_id));
    for (int i = 0; i < num_streams; ++i) HIP_CHECK(hipStreamDestroy(streams[i]));
  }
};

// Per-slot device buffers + streams for one workload instance.
struct PreparedSlot {
  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  void* dExtraH2D = nullptr;
  void* dD2D = nullptr;
  size_t bytesA = 0, bytesB = 0, bytesC = 0, bytesD2H = 0, bytesD2D = 0;
  int d2d_device = -1;  // device where dD2D lives (self, or peer under --d2d-peer)

  hipStream_t h2d_stream = nullptr;
  hipStream_t compute_stream = nullptr;
  hipStream_t d2h_stream = nullptr;
  hipStream_t d2d_stream = nullptr;
  int slot_id = -1;
};

struct Request {
  int id;
};

template <typename T>
class BoundedQueue {
 public:
  bool Enqueue(T* obj) {
    std::lock_guard<std::mutex> guard(mu_);
    if (q_.size() >= g_cfg.max_queue_size) return false;
    q_.push(obj);
    return true;
  }

  T* Dequeue() {
    std::lock_guard<std::mutex> guard(mu_);
    if (q_.empty()) return nullptr;
    T* head = q_.front();
    q_.pop();
    return head;
  }

  size_t size() {
    std::lock_guard<std::mutex> guard(mu_);
    return q_.size();
  }

 private:
  std::queue<T*> q_;
  std::mutex mu_;
};

using RequestQueue = BoundedQueue<Request>;

struct BenchControl {
  std::atomic<bool> stopped{false};
  void stop() { stopped.store(true, std::memory_order_release); }
  bool isStopped() const { return stopped.load(std::memory_order_acquire); }
};

struct BenchmarkStats {
  std::vector<int64_t> latencies_us;
  mutable std::mutex mu;

  void update(int64_t latency_us) {
    std::lock_guard<std::mutex> guard(mu);
    latencies_us.push_back(latency_us);
  }

  std::vector<int64_t> snapshot() const {
    std::lock_guard<std::mutex> guard(mu);
    return latencies_us;
  }

  // Single computed summary shared by the console report and the CSV writer so
  // both always emit identical numbers.
  struct StatRow {
    size_t count = 0;
    double wall_sec = 0.0;
    double qps = 0.0;
    int64_t min_us = 0, max_us = 0;
    double mean_us = 0.0, stddev_us = 0.0;
    double p50_us = 0.0, p75_us = 0.0, p95_us = 0.0;
    double p98_us = 0.0, p99_us = 0.0, p999_us = 0.0;
  };

  static double getValue(const std::vector<int64_t>& sorted, double quantile) {
    if (sorted.empty()) return 0.0;
    const double pos = quantile * (static_cast<double>(sorted.size()) + 1.0);
    if (pos < 1.0) return static_cast<double>(sorted.front());
    if (pos >= static_cast<double>(sorted.size())) return static_cast<double>(sorted.back());
    const size_t idx = static_cast<size_t>(pos);
    const double lower = static_cast<double>(sorted[idx - 1]);
    const double upper = static_cast<double>(sorted[idx]);
    return lower + (pos - std::floor(pos)) * (upper - lower);
  }

  static StatRow compute(const std::vector<int64_t>& data, double wall_elapsed_ms) {
    std::vector<int64_t> sorted(data);
    std::sort(sorted.begin(), sorted.end());
    StatRow r;
    r.count = sorted.size();
    if (!sorted.empty()) {
      r.min_us = sorted.front();
      r.max_us = sorted.back();
      double sum = 0.0;
      for (int64_t v : sorted) sum += static_cast<double>(v);
      r.mean_us = sum / static_cast<double>(r.count);
      if (r.count > 1) {
        double sum_sq = 0.0;
        for (int64_t v : sorted) {
          const double diff = static_cast<double>(v) - r.mean_us;
          sum_sq += diff * diff;
        }
        r.stddev_us = std::sqrt(sum_sq / static_cast<double>(r.count - 1));
      }
    }
    r.wall_sec = wall_elapsed_ms / 1000.0;
    r.qps = (r.wall_sec > 0.0) ? static_cast<double>(r.count) / r.wall_sec : 0.0;
    r.p50_us = getValue(sorted, 0.5);
    r.p75_us = getValue(sorted, 0.75);
    r.p95_us = getValue(sorted, 0.95);
    r.p98_us = getValue(sorted, 0.98);
    r.p99_us = getValue(sorted, 0.99);
    r.p999_us = getValue(sorted, 0.999);
    return r;
  }

  static void printReport(const char* title, const std::vector<int64_t>& data,
                           double wall_elapsed_ms) {
    const StatRow r = compute(data, wall_elapsed_ms);

    std::printf("\n");
    std::printf("%.64s ", title);
    for (int i = 0; i < 60 - static_cast<int>(std::strlen(title)); ++i) std::printf("=");
    std::printf("\n");
    std::printf("-- Latency Histogram (us)\n");
    std::printf("             count = %zu\n", r.count);
    std::printf("               min = %lld\n", static_cast<long long>(r.min_us));
    std::printf("               max = %lld\n", static_cast<long long>(r.max_us));
    std::printf("              mean = %.2f\n", r.mean_us);
    std::printf("            stddev = %.2f\n", r.stddev_us);
    std::printf("            median = %.2f\n", r.p50_us);
    std::printf("              75%% <= %.2f\n", r.p75_us);
    std::printf("              95%% <= %.2f\n", r.p95_us);
    std::printf("              98%% <= %.2f\n", r.p98_us);
    std::printf("              99%% <= %.2f\n", r.p99_us);
    std::printf("            99.9%% <= %.2f\n", r.p999_us);
    std::printf("-- Throughput\n");
    std::printf("       wall_time_s = %.4f\n", r.wall_sec);
    std::printf("       total_iters = %zu\n", r.count);
    std::printf("               QPS = %.2f\n", r.qps);
    std::printf("\n");
  }
};

__global__ void timedBusyKernel(long long durationNs, long long wallClkRateKHz,
                                bool useSleep,
                                long long *actualElapsedNs)
{
    long long start = wall_clock64();
    long long durationTicks = (durationNs * wallClkRateKHz) / 1000000LL;
    long long end = start + durationTicks;

    long long now;
    do {
        now = wall_clock64();
        if (useSleep && (now & 0xFFF) == 0) {
            __builtin_amdgcn_s_sleep(1);
        }
    } while (now < end);

    if (actualElapsedNs != NULL) {
        *actualElapsedNs = ((now - start) * 1000000LL) / wallClkRateKHz;
    }
}

static inline void launch_compute_op(GpuResources& gpu,
                                     PreparedSlot* /*prep*/,
                                     hipStream_t stream) {
  const int blocks = (kBusyKernelBlocks > 0) ? kBusyKernelBlocks
                                             : std::max(1, gpu.multiprocessor_count);
  timedBusyKernel<<<blocks, kBusyKernelThreads, 0, stream>>>(
      kBusyKernelDurationNs,
      static_cast<long long>(gpu.wall_clock_rate_khz),
      kBusyKernelUseSleep,
      nullptr);
  HIP_CHECK(hipGetLastError());
}


size_t contiguous_or_strided_bytes(int64_t rows, int64_t cols, int64_t stride, int64_t batch_count, size_t elem_size) {
  const int64_t one_matrix_elems = rows * cols;
  if (batch_count <= 1) return static_cast<size_t>(one_matrix_elems) * elem_size;
  const int64_t full_elems = (batch_count - 1) * stride + one_matrix_elems;
  return static_cast<size_t>(full_elems) * elem_size;
}

void validate_workload_or_die(const WorkloadConfig& cfg) {
  if (cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0) {
    std::cerr << "Invalid shape: m/n/k must be > 0\n";
    std::exit(EXIT_FAILURE);
  }
  if (cfg.batch_count <= 0) {
    std::cerr << "Invalid batch_count: must be > 0\n";
    std::exit(EXIT_FAILURE);
  }
  if (cfg.elem_bytes_a == 0 || cfg.elem_bytes_b == 0 || cfg.elem_bytes_c == 0) {
    std::cerr << "Invalid element sizes: elem_bytes must be > 0\n";
    std::exit(EXIT_FAILURE);
  }
}

PreparedSlot* prepare_slot(GpuResources& gpu, int slot_id, const WorkloadConfig& cfg) {
  validate_workload_or_die(cfg);
  HIP_CHECK(hipSetDevice(gpu.device_id));

  auto* prep = new PreparedSlot();
  prep->h2d_stream = gpu.h2d_stream(slot_id);
  prep->compute_stream = gpu.compute_stream(slot_id);
  prep->d2h_stream = gpu.d2h_stream(slot_id);
  prep->d2d_stream = gpu.d2d_stream(slot_id);
  prep->slot_id = slot_id;

  const int64_t a_rows = cfg.trans_a ? cfg.k : cfg.m;
  const int64_t a_cols = cfg.trans_a ? cfg.m : cfg.k;
  const int64_t b_rows = cfg.trans_b ? cfg.n : cfg.k;
  const int64_t b_cols = cfg.trans_b ? cfg.k : cfg.n;
  prep->bytesA = contiguous_or_strided_bytes(a_rows, a_cols, cfg.stride_a, cfg.batch_count, cfg.elem_bytes_a);
  prep->bytesB = contiguous_or_strided_bytes(b_rows, b_cols, cfg.stride_b, cfg.batch_count, cfg.elem_bytes_b);
  prep->bytesC = contiguous_or_strided_bytes(cfg.m, cfg.n, cfg.stride_c, cfg.batch_count, cfg.elem_bytes_c);
  prep->bytesD2H = g_cfg.use_reduced_d2h ? std::min(prep->bytesC, g_cfg.reduced_d2h_bytes) : prep->bytesC;
  prep->bytesD2D = (g_cfg.d2d_bytes > 0) ? g_cfg.d2d_bytes : prep->bytesB;

  HIP_CHECK(hipMalloc(&prep->dA, prep->bytesA));
  HIP_CHECK(hipMalloc(&prep->dB, prep->bytesB));
  HIP_CHECK(hipMalloc(&prep->dC, prep->bytesC));
  if (kExtraH2DBytes > 0) {
    HIP_CHECK(hipMalloc(&prep->dExtraH2D, kExtraH2DBytes));
  }

  // D2D destination: same device by default, or the neighbor device under
  // --d2d-peer (cross-GPU copy; peer access is enabled in main()).
  prep->d2d_device = g_cfg.d2d_peer ? ((gpu.device_id + 1) % g_cfg.num_gpus)
                                    : gpu.device_id;
  if (prep->d2d_device != gpu.device_id) {
    HIP_CHECK(hipSetDevice(prep->d2d_device));
    HIP_CHECK(hipMalloc(&prep->dD2D, prep->bytesD2D));
    HIP_CHECK(hipSetDevice(gpu.device_id));
  } else {
    HIP_CHECK(hipMalloc(&prep->dD2D, prep->bytesD2D));
  }
  return prep;
}

// TF/XLA: A is the weight matrix (tf.Variable, pre-loaded on GPU)
void preload_weights(PreparedSlot* prep, const PinnedBuffer& hA) {
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(prep->dA),
                          const_cast<uint8_t*>(hA.data()), prep->bytesA));
}

void cleanup_slot(PreparedSlot* prep) {
  if (!prep) return;
  if (prep->dExtraH2D) HIP_CHECK(hipFree(prep->dExtraH2D));
  if (prep->dD2D) HIP_CHECK(hipFree(prep->dD2D));
  if (prep->dA) HIP_CHECK(hipFree(prep->dA));
  if (prep->dB) HIP_CHECK(hipFree(prep->dB));
  if (prep->dC) HIP_CHECK(hipFree(prep->dC));
  delete prep;
}

// Warmup mirrors the measured pipeline: H2D -> compute -> D2D -> compute -> D2H.
void warmup_slot(GpuResources& gpu, PreparedSlot* prep,
                 const PinnedBuffer& hB,
                 PinnedBuffer& hC, int iters) {
  HIP_CHECK(hipSetDevice(gpu.device_id));
  hipStream_t s = prep->compute_stream;
  for (int i = 0; i < iters; ++i) {
    HIP_CHECK(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(prep->dB),
                                 const_cast<uint8_t*>(hB.data()), prep->bytesB, s));
    launch_compute_op(gpu, prep, s);
    HIP_CHECK(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(prep->dD2D),
                                 reinterpret_cast<hipDeviceptr_t>(prep->dB), prep->bytesD2D, s));
    launch_compute_op(gpu, prep, s);
    HIP_CHECK(hipMemcpyDtoHAsync(hC.data(),
                                 reinterpret_cast<hipDeviceptr_t>(prep->dC), prep->bytesD2H, s));
  }
  HIP_CHECK(hipStreamSynchronize(s));
}

// Prepare all slots, preload weights (when enabled), and warm up each slot so
// the measured phase excludes one-time allocation/first-launch costs.
void prepare_and_warmup_slots(std::vector<std::unique_ptr<GpuResources>>& resources,
                              const WorkloadConfig& cfg,
                              const PinnedBuffer& hA, const PinnedBuffer& hB,
                              std::vector<PinnedBuffer>& hC,
                              std::vector<PreparedSlot*>& prep_cache,
                              int warmup_iters) {
  for (int g = 0; g < g_cfg.num_gpus; ++g) {
    for (int s = 0; s < g_cfg.streams_per_gpu; ++s) {
      const int idx = g * g_cfg.streams_per_gpu + s;
      prep_cache[idx] = prepare_slot(*resources[g], s, cfg);
      if (kPreloadWeights) {
        preload_weights(prep_cache[idx], hA);
      }
      warmup_slot(*resources[g], prep_cache[idx], hB, hC[idx], warmup_iters);
    }
  }
}

class Producer {
 public:
  Producer(RequestQueue* queue, BenchControl* ctrl,
               int target_qps, int pool_size)
      : queue_(queue), ctrl_(ctrl), target_qps_(target_qps) {
    pool_.resize(pool_size);
    for (int i = 0; i < pool_size; ++i) pool_[i].id = i;
  }

  void Start() {
    size_t pos = 0;
    while (!ctrl_->isStopped()) {
      if (target_qps_ > 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        const uint64_t interval_us = (800ULL * 1000ULL) / static_cast<uint64_t>(target_qps_);
        for (int i = 0; i < target_qps_ && !ctrl_->isStopped(); ++i) {
          if (pos >= pool_.size()) pos = 0;
          queue_->Enqueue(&pool_[pos++]);
          std::this_thread::sleep_for(std::chrono::microseconds(interval_us));
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        if (elapsed_us < 1000000) {
          std::this_thread::sleep_for(std::chrono::microseconds(1000000 - elapsed_us));
        }
      } else {
        if (pos >= pool_.size()) pos = 0;
        queue_->Enqueue(&pool_[pos++]);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
  }

 private:
  RequestQueue* queue_;
  BenchControl* ctrl_;
  int target_qps_;
  std::vector<Request> pool_;
};

// Shared session/slot pool — mirrors TF's Model::Borrow()/Return() pattern.
// Multiple consumer threads compete for the (num_gpus * streams_per_gpu)
// sessions, round-robin.
struct BorrowedSlot {
  int index;
  GpuResources* gpu;
  PreparedSlot* prep;
  PinnedBuffer* hC;
};

class SlotPool {
 public:
  struct SlotEntry {
    GpuResources* gpu = nullptr;
    PreparedSlot* prep = nullptr;
    PinnedBuffer* hC = nullptr;
    bool borrowed = false;
    int completed = 0;
  };

  explicit SlotPool(int num_slots) : slots_(num_slots) {}

  void Register(int idx, GpuResources* gpu, PreparedSlot* prep, PinnedBuffer* hC) {
    slots_[idx].gpu = gpu;
    slots_[idx].prep = prep;
    slots_[idx].hC = hC;
  }

  // Round-robin scan for an available slot; blocks up to 500us if all busy.
  BorrowedSlot Borrow() {
    const int n = static_cast<int>(slots_.size());
    std::unique_lock<std::mutex> lock(mu_);
    while (true) {
      for (int i = 0; i < n; ++i) {
        int c = next_idx_++ % n;
        if (!slots_[c].borrowed) {
          slots_[c].borrowed = true;
          return {c, slots_[c].gpu, slots_[c].prep, slots_[c].hC};
        }
      }
      cv_.wait_for(lock, std::chrono::microseconds(500));
    }
  }

  void Return(int slot_index) {
    std::unique_lock<std::mutex> lock(mu_);
    slots_[slot_index].borrowed = false;
    ++slots_[slot_index].completed;
    cv_.notify_one();
  }

  int slot_completed(int idx) const { return slots_[idx].completed; }
  int num_slots() const { return static_cast<int>(slots_.size()); }

 private:
  std::vector<SlotEntry> slots_;
  std::mutex mu_;
  std::condition_variable cv_;
  int next_idx_ = 0;
};

class Consumer {
 public:
  Consumer(SlotPool* pool,
               RequestQueue* queue, BenchControl* ctrl,
               BenchmarkStats* global_stats,
               const PinnedBuffer& hA, const PinnedBuffer& hB,
               const PinnedBuffer& hExtraH2D)
      : pool_(pool), queue_(queue), ctrl_(ctrl),
        global_stats_(global_stats),
        hA_(hA), hB_(hB), hExtraH2D_(hExtraH2D) {}

  void Start() {
    int current_device = -1;

    while (!ctrl_->isStopped()) {
      Request* req = queue_->Dequeue();
      if (!req) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        continue;
      }

      BorrowedSlot slot = pool_->Borrow();
      PreparedSlot* prep = slot.prep;
      GpuResources* gpu = slot.gpu;
      PinnedBuffer* hC = slot.hC;

      if (gpu->device_id != current_device) {
        HIP_CHECK(hipSetDevice(gpu->device_id));
        current_device = gpu->device_id;
      }

      auto wall_start = std::chrono::high_resolution_clock::now();

      hipStream_t stream = prep->compute_stream;

      // Pipeline (mimics a DL/LLM step): H2D input -> compute/transpose ->
      // device-to-device copy -> compute -> D2H output. Uses XLA main's typed
      // async copy APIs (hipMemcpyHtoDAsync / DtoDAsync / DtoHAsync).
      if (!kSkipH2D) {
        if (!kPreloadWeights) {
          HIP_CHECK(hipMemcpyHtoDAsync(
              reinterpret_cast<hipDeviceptr_t>(prep->dA),
              const_cast<uint8_t*>(hA_.data()), prep->bytesA, stream));
        }
        if (kExtraH2DBytes > 0 && prep->dExtraH2D) {
          for (int i = 0; i < kExtraH2DCount; ++i) {
            HIP_CHECK(hipMemcpyHtoDAsync(
                reinterpret_cast<hipDeviceptr_t>(prep->dExtraH2D),
                const_cast<uint8_t*>(hExtraH2D_.data()), kExtraH2DBytes, stream));
          }
        }
        HIP_CHECK(hipMemcpyHtoDAsync(
            reinterpret_cast<hipDeviceptr_t>(prep->dB),
            const_cast<uint8_t*>(hB_.data()), prep->bytesB, stream));
      }

      // Compute stage 1 (input compute/transpose).
      launch_compute_op(*gpu, prep, stream);

      // Device-to-device copy (intra-device, or cross-GPU under --d2d-peer).
      if (!kSkipH2D) {
        HIP_CHECK(hipMemcpyDtoDAsync(
            reinterpret_cast<hipDeviceptr_t>(prep->dD2D),
            reinterpret_cast<hipDeviceptr_t>(prep->dB), prep->bytesD2D, stream));
      }

      // Compute stage 2 (follow-on compute).
      launch_compute_op(*gpu, prep, stream);

      if (!kSkipD2H) {
        HIP_CHECK(hipMemcpyDtoHAsync(
            hC->data(), reinterpret_cast<hipDeviceptr_t>(prep->dC),
            prep->bytesD2H, stream));
      }

      HIP_CHECK(hipStreamSynchronize(stream));

      auto wall_end = std::chrono::high_resolution_clock::now();
      const int64_t latency_us =
          std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start).count();

      global_stats_->update(latency_us);
      ++completed_;

      pool_->Return(slot.index);
    }

  }

  int completed() const { return completed_; }

 private:
  SlotPool* pool_;
  RequestQueue* queue_;
  BenchControl* ctrl_;
  BenchmarkStats* global_stats_;
  const PinnedBuffer& hA_;
  const PinnedBuffer& hB_;
  const PinnedBuffer& hExtraH2D_;
  int completed_ = 0;
};

// ---------------------------------------------------------------------------
// ROCm / HIP version detection (for stamping CSV rows).
// ---------------------------------------------------------------------------
struct VersionInfo {
  std::string hip_version;      // compiled HIP_VERSION_* (reflects build ROCm)
  int hip_runtime_version = 0;  // hipRuntimeGetVersion()
  int hip_driver_version = 0;   // hipDriverGetVersion()
};

static VersionInfo detect_versions() {
  VersionInfo v;
#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && defined(HIP_VERSION_PATCH)
  v.hip_version = std::to_string(HIP_VERSION_MAJOR) + "." +
                  std::to_string(HIP_VERSION_MINOR) + "." +
                  std::to_string(HIP_VERSION_PATCH);
#else
  v.hip_version = "unknown";
#endif
  (void)hipRuntimeGetVersion(&v.hip_runtime_version);
  (void)hipDriverGetVersion(&v.hip_driver_version);
  return v;
}

// ---------------------------------------------------------------------------
// CSV writer. One self-describing row per report (interval snapshots + final).
// If the target file already has content, rows are appended and the header is
// not rewritten, so multiple ROCm-version runs can accumulate into one file.
// ---------------------------------------------------------------------------
class CsvWriter {
 public:
  CsvWriter(const std::string& path, const VersionInfo& ver, const WorkloadConfig& cfg,
            const char* compute_op, const std::string& rocm_version)
      : ver_(ver), work_m_(cfg.m), work_n_(cfg.n), work_k_(cfg.k),
        compute_op_(compute_op), rocm_version_(rocm_version) {
    if (path.empty()) return;
    bool exists_nonempty = false;
    {
      std::ifstream in(path, std::ios::binary | std::ios::ate);
      exists_nonempty = in.good() && in.tellg() > 0;
    }
    out_.open(path, std::ios::app);
    if (!out_.is_open()) {
      std::cerr << "Warning: could not open CSV file '" << path << "'; CSV disabled\n";
      return;
    }
    enabled_ = true;
    if (!exists_nonempty) {
      out_ << "timestamp_iso,rocm_version,hip_version,hip_runtime_version,"
              "hip_driver_version,label,phase,elapsed_sec,duration_sec,"
              "report_interval_sec,num_gpus,streams_per_gpu,total_slots,"
              "consumer_threads,target_qps,max_queue_size,reduced_d2h,"
              "reduced_d2h_bytes,d2d_bytes,d2d_peer,compute_op,"
              "busy_kernel_ns,busy_kernel_threads,busy_kernel_blocks,busy_kernel_use_sleep,"
              "work_m,work_n,work_k,count,qps,"
              "min_us,max_us,mean_us,stddev_us,p50_us,p75_us,p95_us,p98_us,"
              "p99_us,p999_us\n";
    }
    out_.flush();
  }

  bool enabled() const { return enabled_; }

  void writeRow(const char* phase, int elapsed_sec,
                const BenchmarkStats::StatRow& r) {
    if (!enabled_) return;
    char ts[64];
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
    localtime_r(&now, &tm_buf);
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tm_buf);

    out_ << ts << ','
         << csv_escape(rocm_version_) << ','
         << csv_escape(ver_.hip_version) << ','
         << ver_.hip_runtime_version << ','
         << ver_.hip_driver_version << ','
         << csv_escape(g_cfg.label) << ','
         << phase << ','
         << elapsed_sec << ','
         << g_cfg.duration_sec << ','
         << g_cfg.report_interval_sec << ','
         << g_cfg.num_gpus << ','
         << g_cfg.streams_per_gpu << ','
         << g_cfg.total_slots() << ','
         << g_cfg.consumer_threads << ','
         << g_cfg.target_qps << ','
         << g_cfg.max_queue_size << ','
         << (g_cfg.use_reduced_d2h ? 1 : 0) << ','
         << g_cfg.reduced_d2h_bytes << ','
         << g_cfg.d2d_bytes << ','
         << (g_cfg.d2d_peer ? 1 : 0) << ','
         << compute_op_ << ','
         << kBusyKernelDurationNs << ','
         << kBusyKernelThreads << ','
         << kBusyKernelBlocks << ','
         << (kBusyKernelUseSleep ? 1 : 0) << ','
         << work_m_ << ',' << work_n_ << ',' << work_k_ << ','
         << r.count << ','
         << fmt(r.qps) << ','
         << r.min_us << ',' << r.max_us << ','
         << fmt(r.mean_us) << ',' << fmt(r.stddev_us) << ','
         << fmt(r.p50_us) << ',' << fmt(r.p75_us) << ',' << fmt(r.p95_us) << ','
         << fmt(r.p98_us) << ',' << fmt(r.p99_us) << ',' << fmt(r.p999_us) << '\n';
    out_.flush();
  }

 private:
  static std::string fmt(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
  }
  static std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    out += '"';
    return out;
  }

  std::ofstream out_;
  bool enabled_ = false;
  VersionInfo ver_;
  int64_t work_m_, work_n_, work_k_;
  const char* compute_op_;
  std::string rocm_version_;
};

// ---------------------------------------------------------------------------
// Command-line parsing. Every Config field is overridable; --help prints all.
// ---------------------------------------------------------------------------
static void print_usage(const char* prog) {
  std::printf(
      "ROCm Latency Check Tool\n"
      "Measures per-request latency of the HIP-runtime pipeline\n"
      "H2D -> timedBusyKernel -> D2D -> timedBusyKernel -> D2H,\n"
      "reported as a percentile histogram and QPS.\n\n"
      "Usage: %s [flags]\n\n"
      "Flags (default in brackets):\n"
      "  --num-gpus N              GPUs to use [%d]\n"
      "  --streams-per-gpu N       per-GPU slots/streams [%d]\n"
      "  --duration-sec N          total run length in seconds [%d]\n"
      "  --report-interval-sec N   print/CSV cadence in seconds [%d]\n"
      "  --target-qps N            paced QPS; 0 = max throughput [%d]\n"
      "  --consumer-threads N      consumer threads [%d]\n"
      "  --max-queue-size N        per-consumer bounded queue size [%zu]\n"
      "  --reduced-d2h             copy only reduced-d2h-bytes back (eval mode) [%s]\n"
      "  --no-reduced-d2h          copy the full output matrix back\n"
      "  --reduced-d2h-bytes N     bytes for reduced D2H [%zu]\n"
      "  --d2d-bytes N             bytes for the D2D copy; 0 = input B size [%zu]\n"
      "  --d2d-peer                D2D targets the neighbor GPU (needs >=2 GPUs) [%s]\n"
      "  --csv PATH                append per-report rows to this CSV file [disabled]\n"
      "  --label STR               ROCm release tag stamped into CSV (e.g. 7.2.1) [auto]\n"
      "  -h, --help                show this help and exit\n\n"
      "Flags accept both '--flag value' and '--flag=value'.\n",
      prog, g_cfg.num_gpus, g_cfg.streams_per_gpu, g_cfg.duration_sec,
      g_cfg.report_interval_sec, g_cfg.target_qps, g_cfg.consumer_threads,
      g_cfg.max_queue_size, g_cfg.use_reduced_d2h ? "on" : "off",
      g_cfg.reduced_d2h_bytes, g_cfg.d2d_bytes, g_cfg.d2d_peer ? "on" : "off");
}

// Returns: 0 = parsed OK (continue), 1 = exit success (e.g. --help), 2 = error.
static int parse_args(int argc, char** argv) {
  auto need_value = [&](int& i, const char* flag, std::string& out) -> bool {
    // Supports --flag=value and --flag value.
    std::string arg = argv[i];
    auto eq = arg.find('=');
    if (eq != std::string::npos) { out = arg.substr(eq + 1); return true; }
    if (i + 1 >= argc) {
      std::cerr << "Missing value for " << flag << "\n";
      return false;
    }
    out = argv[++i];
    return true;
  };
  auto matches = [](const std::string& arg, const char* name) {
    std::string a = arg;
    auto eq = a.find('=');
    if (eq != std::string::npos) a = a.substr(0, eq);
    return a == name;
  };
  auto to_int = [](const std::string& s, int& out) {
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
  };
  auto to_size = [](const std::string& s, size_t& out) {
    char* end = nullptr;
    long long v = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v < 0) return false;
    out = static_cast<size_t>(v);
    return true;
  };

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    std::string val;
    if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 1; }
    else if (matches(arg, "--num-gpus")) {
      if (!need_value(i, "--num-gpus", val) || !to_int(val, g_cfg.num_gpus)) return 2;
    } else if (matches(arg, "--streams-per-gpu")) {
      if (!need_value(i, "--streams-per-gpu", val) || !to_int(val, g_cfg.streams_per_gpu)) return 2;
    } else if (matches(arg, "--duration-sec")) {
      if (!need_value(i, "--duration-sec", val) || !to_int(val, g_cfg.duration_sec)) return 2;
    } else if (matches(arg, "--report-interval-sec")) {
      if (!need_value(i, "--report-interval-sec", val) || !to_int(val, g_cfg.report_interval_sec)) return 2;
    } else if (matches(arg, "--target-qps")) {
      if (!need_value(i, "--target-qps", val) || !to_int(val, g_cfg.target_qps)) return 2;
    } else if (matches(arg, "--consumer-threads")) {
      if (!need_value(i, "--consumer-threads", val) || !to_int(val, g_cfg.consumer_threads)) return 2;
    } else if (matches(arg, "--max-queue-size")) {
      if (!need_value(i, "--max-queue-size", val) || !to_size(val, g_cfg.max_queue_size)) return 2;
    } else if (matches(arg, "--reduced-d2h-bytes")) {
      if (!need_value(i, "--reduced-d2h-bytes", val) || !to_size(val, g_cfg.reduced_d2h_bytes)) return 2;
    } else if (arg == "--reduced-d2h") {
      g_cfg.use_reduced_d2h = true;
    } else if (arg == "--no-reduced-d2h") {
      g_cfg.use_reduced_d2h = false;
    } else if (matches(arg, "--d2d-bytes")) {
      if (!need_value(i, "--d2d-bytes", val) || !to_size(val, g_cfg.d2d_bytes)) return 2;
    } else if (arg == "--d2d-peer") {
      g_cfg.d2d_peer = true;
    } else if (matches(arg, "--csv")) {
      if (!need_value(i, "--csv", val)) return 2;
      g_cfg.csv_path = val;
    } else if (matches(arg, "--label")) {
      if (!need_value(i, "--label", val)) return 2;
      g_cfg.label = val;
    } else {
      std::cerr << "Unknown flag: " << arg << " (use --help)\n";
      return 2;
    }
  }

  if (g_cfg.num_gpus <= 0 || g_cfg.streams_per_gpu <= 0 || g_cfg.duration_sec <= 0 ||
      g_cfg.consumer_threads <= 0 || g_cfg.max_queue_size == 0) {
    std::cerr << "Invalid config: num-gpus, streams-per-gpu, duration-sec, "
                 "consumer-threads must be > 0 and max-queue-size > 0\n";
    return 2;
  }
  if (g_cfg.d2d_peer && g_cfg.num_gpus < 2) {
    std::cerr << "--d2d-peer requires --num-gpus >= 2 (cross-GPU copy needs a second device)\n";
    return 2;
  }
  return 0;
}

int main(int argc, char** argv) {
  const int parse_rc = parse_args(argc, argv);
  if (parse_rc == 1) return EXIT_SUCCESS;
  if (parse_rc == 2) return EXIT_FAILURE;

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < g_cfg.num_gpus) {
    std::cerr << "Need at least " << g_cfg.num_gpus << " GPUs, found " << device_count << "\n";
    return EXIT_FAILURE;
  }

  // Workload sizing (fp16 -> 2 bytes/elem). Only determines copy sizes; the
  // compute is the synthetic busy kernel, not a GEMM.
  WorkloadConfig cfg{
      .m = 100000,
      .n = 59,
      .k = 128,
      .stride_a = 0,
      .stride_b = 0,
      .stride_c = 0,
      .trans_a = true,
      .trans_b = false,
      .batch_count = 1,
      .elem_bytes_a = 2,
      .elem_bytes_b = 2,
      .elem_bytes_c = 2,
  };

  const int warmup_iters = 20;

  const int64_t a_rows = cfg.trans_a ? cfg.k : cfg.m;
  const int64_t a_cols = cfg.trans_a ? cfg.m : cfg.k;
  const int64_t b_rows = cfg.trans_b ? cfg.n : cfg.k;
  const int64_t b_cols = cfg.trans_b ? cfg.k : cfg.n;
  const size_t bytesA = contiguous_or_strided_bytes(
      a_rows, a_cols, cfg.stride_a, cfg.batch_count, cfg.elem_bytes_a);
  const size_t bytesB = contiguous_or_strided_bytes(
      b_rows, b_cols, cfg.stride_b, cfg.batch_count, cfg.elem_bytes_b);
  const size_t bytesC = contiguous_or_strided_bytes(
      cfg.m, cfg.n, cfg.stride_c, cfg.batch_count, cfg.elem_bytes_c);

  PinnedBuffer hA(bytesA, 0x3C);
  PinnedBuffer hB(bytesB, 0x3C);
  PinnedBuffer hExtraH2D(kExtraH2DBytes, 0x42);
  std::vector<PinnedBuffer> hC;
  hC.reserve(g_cfg.total_slots());
  for (int i = 0; i < g_cfg.total_slots(); ++i) hC.emplace_back(bytesC, 0);

  std::vector<std::unique_ptr<GpuResources>> resources;
  resources.reserve(g_cfg.num_gpus);
  for (int dev = 0; dev < g_cfg.num_gpus; ++dev) {
    resources.emplace_back(std::make_unique<GpuResources>(dev));
  }

  // Cross-GPU D2D: enable peer access from each device to its neighbor so a
  // slot's stream can write into the neighbor's dD2D buffer.
  if (g_cfg.d2d_peer) {
    for (int g = 0; g < g_cfg.num_gpus; ++g) {
      const int peer = (g + 1) % g_cfg.num_gpus;
      HIP_CHECK(hipSetDevice(g));
      hipError_t e = hipDeviceEnablePeerAccess(peer, 0);
      if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
        std::cerr << "hipDeviceEnablePeerAccess(" << peer << ") from device " << g
                  << " failed: " << hipGetErrorString(e) << "\n";
        return EXIT_FAILURE;
      }
      (void)hipGetLastError();  // clear sticky already-enabled error
    }
  }

  // Prepare device buffers for every slot and warm them up (excludes one-time
  // allocation / first-launch cost from the measured phase).
  std::vector<PreparedSlot*> prep_cache(g_cfg.total_slots(), nullptr);
  prepare_and_warmup_slots(resources, cfg, hA, hB, hC, prep_cache, warmup_iters);

  const int num_slots = g_cfg.total_slots();
  const VersionInfo version_info = detect_versions();
  const std::string rocm_version =
      g_cfg.label.empty() ? version_info.hip_version : g_cfg.label;
  const char* compute_op = "timedBusyKernel";

  std::printf("ROCm/HIP: rocm_version=%s hip_version=%s runtime=%d driver=%d\n",
              rocm_version.c_str(), version_info.hip_version.c_str(),
              version_info.hip_runtime_version, version_info.hip_driver_version);
  std::printf("Pipeline: H2D -> timedBusyKernel -> D2D -> timedBusyKernel -> D2H (mimics a DL/LLM step)\n");
  std::printf("Warmup=%d, Duration=%ds, ReportInterval=%ds, Slots=%d, Consumers=%d, TargetQPS=%d (%s)\n",
              warmup_iters, g_cfg.duration_sec, g_cfg.report_interval_sec, num_slots,
              g_cfg.consumer_threads,
              g_cfg.target_qps, g_cfg.target_qps > 0 ? "paced" : "max-throughput");
  std::printf("Dispatch: dynamic (%d consumers competing for %d slots, TF/JAX/XLA Borrow/Return)\n",
              g_cfg.consumer_threads, num_slots);
  std::printf("D2H mode: %s (bytes=%zu)\n",
              g_cfg.use_reduced_d2h ? "reduced-eval" : "full-output",
              g_cfg.use_reduced_d2h ? g_cfg.reduced_d2h_bytes : bytesC);
  std::printf("D2D mode: %s (bytes=%zu)\n",
              g_cfg.d2d_peer ? "peer (cross-GPU to neighbor)" : "intra-device",
              (g_cfg.d2d_bytes > 0) ? g_cfg.d2d_bytes : bytesB);
  std::printf("Host memory: pinned (hipHostMalloc, matches TF/JAX/XLA allocate memory on the host)\n");
  std::printf("Weights A: %s (%.2f MB), Per-request H2D input B: %.2f KB\n",
              kPreloadWeights ? "pre-loaded" : "H2D per request",
              bytesA / (1024.0 * 1024.0), bytesB / 1024.0);
  if (kExtraH2DBytes > 0) {
    std::printf("Extra H2D per request: %d x %.2f MB = %.2f MB (simulate TF/JAX/XLA multi-tensor traffic)\n",
                kExtraH2DCount, kExtraH2DBytes / (1024.0 * 1024.0),
                kExtraH2DCount * kExtraH2DBytes / (1024.0 * 1024.0));
  }
  if (kSkipH2D || kSkipD2H) {
    std::printf("TF_SKIP_H2D=%s, TF_SKIP_D2H=%s (diagnostic: isolate SDMA impact)\n",
                kSkipH2D ? "true" : "false", kSkipD2H ? "true" : "false");
  }
  std::printf("Streams per slot: %d (all ops on same stream)\n",
              GpuResources::kStreamsPerSlot);
  std::printf("Total HIP streams: %d GPUs x %d slots x %d = %d (+default)\n",
              g_cfg.num_gpus, g_cfg.streams_per_gpu, GpuResources::kStreamsPerSlot,
              g_cfg.num_gpus * g_cfg.streams_per_gpu * GpuResources::kStreamsPerSlot);
  std::printf("Compute op: timedBusyKernel x2 per request (HIP runtime latency probe)\n");
  std::printf("timedBusyKernel config: duration_ns=%lld blocks=%d threads=%d use_sleep=%s\n",
              static_cast<long long>(kBusyKernelDurationNs),
              (kBusyKernelBlocks > 0 ? kBusyKernelBlocks : std::max(1, resources[0]->multiprocessor_count)),
              kBusyKernelThreads,
              kBusyKernelUseSleep ? "true" : "false");

  CsvWriter csv_writer(g_cfg.csv_path, version_info, cfg, compute_op, rocm_version);
  if (!g_cfg.csv_path.empty() && csv_writer.enabled()) {
    std::printf("CSV output: %s (label=%s)\n", g_cfg.csv_path.c_str(),
                g_cfg.label.empty() ? "auto" : g_cfg.label.c_str());
  }

  BenchControl bench_ctrl;
  BenchmarkStats global_stats;

  // Build shared SlotPool (total_slots sessions, like TF's PredictContext pool)
  SlotPool slot_pool(num_slots);
  for (int slot = 0; slot < num_slots; ++slot) {
    const int gpu_idx = slot / g_cfg.streams_per_gpu;
    slot_pool.Register(slot, resources[gpu_idx].get(), prep_cache[slot], &hC[slot]);
  }

  const int num_consumers = g_cfg.consumer_threads;
  const int per_consumer_qps = (g_cfg.target_qps > 0) ? std::max(1, g_cfg.target_qps / num_consumers) : 0;
  const int request_pool_size = (per_consumer_qps > 0) ? per_consumer_qps * 2 : 1024;

  std::vector<RequestQueue> queues(num_consumers);
  std::vector<std::unique_ptr<Producer>> producers;
  std::vector<std::unique_ptr<Consumer>> consumers;
  producers.reserve(num_consumers);
  consumers.reserve(num_consumers);

  for (int i = 0; i < num_consumers; ++i) {
    producers.emplace_back(std::make_unique<Producer>(
        &queues[i], &bench_ctrl, per_consumer_qps, request_pool_size));
    consumers.emplace_back(std::make_unique<Consumer>(
        &slot_pool, &queues[i], &bench_ctrl,
        &global_stats,
        hA, hB, hExtraH2D));
  }

  std::vector<std::thread> consumer_threads;
  std::vector<std::thread> producer_threads;
  consumer_threads.reserve(num_consumers);
  producer_threads.reserve(num_consumers);

  auto wall_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_consumers; ++i) {
    consumer_threads.emplace_back(&Consumer::Start, consumers[i].get());
  }
  for (int i = 0; i < num_consumers; ++i) {
    producer_threads.emplace_back(&Producer::Start, producers[i].get());
  }

  // Periodic reporting (it's cumulative)
  int elapsed_sec = 0;
  const int interval = (g_cfg.report_interval_sec > 0) ? g_cfg.report_interval_sec : g_cfg.duration_sec;
  while (elapsed_sec < g_cfg.duration_sec) {
    const int sleep_sec = std::min(interval, g_cfg.duration_sec - elapsed_sec);
    std::this_thread::sleep_for(std::chrono::seconds(sleep_sec));
    elapsed_sec += sleep_sec;

    auto now = std::chrono::high_resolution_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(now - wall_start).count();
    auto snap = global_stats.snapshot();

    char title[128];
    std::snprintf(title, sizeof(title), "[%ds / %ds] Benchmark Statistics",
                  elapsed_sec, g_cfg.duration_sec);
    BenchmarkStats::printReport(title, snap, wall_ms);
    csv_writer.writeRow("interval", elapsed_sec, BenchmarkStats::compute(snap, wall_ms));
  }

  bench_ctrl.stop();
  auto wall_end = std::chrono::high_resolution_clock::now();
  const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  for (auto& t : consumer_threads) t.join();
  for (auto& t : producer_threads) t.join();

  std::printf("\n-- Per-slot completion (TF sessions) --\n");
  int total_completed = 0;
  for (int slot = 0; slot < num_slots; ++slot) {
    const int gpu_idx = slot / g_cfg.streams_per_gpu;
    const int slot_id = slot % g_cfg.streams_per_gpu;
    const int sc = slot_pool.slot_completed(slot);
    std::printf("[GPU %d Slot %d] completed=%d requests\n",
                gpu_idx, slot_id, sc);
    total_completed += sc;
  }
  std::printf("-- Per-consumer thread completion --\n");
  for (int i = 0; i < num_consumers; ++i) {
    std::printf("[Consumer %d] completed=%d requests\n",
                i, consumers[i]->completed());
  }

  auto final_snap = global_stats.snapshot();
  BenchmarkStats::printReport("Final Benchmark Statistics", final_snap, wall_ms);
  csv_writer.writeRow("final", g_cfg.duration_sec, BenchmarkStats::compute(final_snap, wall_ms));

  for (auto* prep : prep_cache) {
    cleanup_slot(prep);
  }

  uint64_t checksum = 0;
  for (uint8_t v : hC[0]) checksum += v;
  std::printf("Task0 byte-checksum=%llu\n", static_cast<unsigned long long>(checksum));
  return 0;
}

