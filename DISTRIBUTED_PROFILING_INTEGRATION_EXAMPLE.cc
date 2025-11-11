// ============================================================================
// DISTRIBUTED PROFILING INTEGRATION EXAMPLE (SINGLETON APPROACH)
// ============================================================================
// This file demonstrates the cleaner integration flow using a singleton
// DistributedProfilerContextManager to store distributed context,
// WITHOUT modifying ProfileOptions (which is in TensorFlow).
// ============================================================================

#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/distributed_timestamp_sync.h"
#include "xla/backends/profiler/gpu/device_tracer_rocm.h"

namespace xla {
namespace profiler {

// ============================================================================
// STEP 1: PJRT CLIENT LAYER - Exchange Addresses & Store in Singleton
// ============================================================================
// In: se_gpu_pjrt_client.cc::BuildDistributedDevices()

absl::StatusOr<std::vector<std::string>> ExchangeNodeAddresses(
    int node_id, int num_nodes, KeyValueStoreInterface* kv_store) {
  
  std::string my_address = absl::StrCat(
      GetHostname(), ":profiler_", std::to_string(5000 + node_id));
  
  LOG(INFO) << "Node " << node_id << " publishing address: " << my_address;
  
  // Publish to KV store
  TF_RETURN_IF_ERROR(
      kv_store->Set(absl::StrCat("profiler_node_addr_", node_id), my_address));
  
  // Wait for all nodes to publish
  TF_RETURN_IF_ERROR(kv_store->Barrier(
      "profiler_addresses_barrier", num_nodes, absl::Seconds(30)));
  
  // Retrieve all addresses
  std::vector<std::string> all_addresses(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    auto addr_result = kv_store->Get(
        absl::StrCat("profiler_node_addr_", i), absl::Seconds(30));
    
    if (!addr_result.ok()) {
      return absl::UnavailableError(
          absl::StrCat("Failed to get address for node ", i));
    }
    
    all_addresses[i] = addr_result.value();
    LOG(INFO) << "Node " << node_id << " learned Node " << i 
              << " address: " << all_addresses[i];
  }
  
  return all_addresses;
}

// NEW: Initialize distributed context in singleton
// Called right after address exchange in BuildDistributedDevices
absl::Status InitializeDistributedProfilerContext(
    int node_id, int num_nodes, 
    const std::vector<std::string>& node_addresses,
    bool enable_socket_timestamping = true) {
  
  DistributedProfilerContext dist_ctx;
  dist_ctx.node_id = node_id;
  dist_ctx.num_nodes = num_nodes;
  dist_ctx.node_addresses = node_addresses;
  dist_ctx.enable_socket_timestamping = enable_socket_timestamping;
  dist_ctx.timestamp_sync_timeout = absl::Seconds(5);
  
  // Store in singleton
  DistributedProfilerContextManager::Get().SetDistributedContext(dist_ctx);
  
  LOG(INFO) << "Distributed context stored in singleton";
  return absl::OkStatus();
}

// ============================================================================
// STEP 2: PJRT CLIENT INTEGRATION
// ============================================================================
// In: se_gpu_pjrt_client.cc::BuildDistributedDevices()
// 
// Simplified flow:

absl::StatusOr<DeviceTopologyPair> BuildDistributedDevices(
    absl::string_view platform_name,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id, int num_nodes,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store,
    bool enable_mock_nccl,
    std::optional<absl::string_view> mock_gpu_topology) {
  
  // ... existing code for topology building ...
  
  // NEW: Exchange addresses and initialize profiler context
  if (num_nodes > 1 && !enable_mock_nccl) {
    TF_ASSIGN_OR_RETURN(
        auto addresses,
        ExchangeNodeAddresses(node_id, num_nodes, kv_store.get()));
    
    // Store in singleton for profiler to access later
    TF_RETURN_IF_ERROR(InitializeDistributedProfilerContext(
        node_id, num_nodes, addresses, 
        /*enable_socket_timestamping=*/true));
  }
  
  // ... rest of existing code ...
  
  return std::make_pair(std::move(devices), gpu_topology);
}

// ============================================================================
// STEP 3: PROFILER FACTORY - Access Singleton Context
// ============================================================================
// In: device_tracer_rocm.cc::CreateGpuTracer()

std::unique_ptr<tsl::profiler::ProfilerInterface> CreateGpuTracer(
    const tensorflow::ProfileOptions& profile_options) {
  
  if (profile_options.device_type() != ProfileOptions::GPU &&
      profile_options.device_type() != ProfileOptions::UNSPECIFIED) {
    return nullptr;
  }

  auto& rocm_tracer = profiler::RocmTracer::GetRocmTracerSingleton();
  if (!rocm_tracer.IsAvailable()) return nullptr;
  
  // NO NEED to read from ProfileOptions anymore!
  // The context is already in the singleton.
  // Just create the tracer as usual.
  
  return std::make_unique<profiler::GpuTracer>(&rocm_tracer);
}

// ============================================================================
// STEP 4: COLLECTOR INITIALIZATION - Get Context from Singleton
// ============================================================================
// In: rocm_collector.cc::RocmTraceCollectorImpl::Flush() or 
//     rocm_tracer.cc::RocmTracer::Enable()

absl::Status RocmTraceCollectorImpl::InitializeDistributedSync() {
  // Get distributed context from singleton (if available)
  auto dist_ctx_opt = 
      DistributedProfilerContextManager::Get().GetDistributedContext();
  
  if (!dist_ctx_opt.has_value()) {
    VLOG(1) << "No distributed context available, single-node profiling";
    return absl::OkStatus();
  }
  
  const auto& dist_ctx = dist_ctx_opt.value();
  
  LOG(INFO) << "Initializing distributed timestamp synchronization...";
  
  // Create synchronizer with the context from singleton
  ts_sync_ = std::make_unique<DistributedTimestampSynchronizer>(dist_ctx);
  TF_RETURN_IF_ERROR(ts_sync_->Initialize());
  
  auto synced_ts = ts_sync_->GetLastSyncTimestamps();
  LOG(INFO) << "Distributed sync initialized. Clock offset: "
            << ts_sync_->GetClockOffset() << " ns";
  
  for (const auto& ts : synced_ts) {
    LOG(INFO) << "  Node " << ts.node_id << ": " 
              << "local=" << ts.local_ns << " ns, "
              << "socket=" << ts.socket_ts_ns << " ns";
  }
  
  return absl::OkStatus();
}

// ============================================================================
// STEP 5 & 6: Profiling and Export (unchanged)
// ============================================================================
// See DISTRIBUTED_PROFILING_INTEGRATION_EXAMPLE.cc for these steps

// ============================================================================
// USAGE EXAMPLE - Application Code (Simplified)
// ============================================================================

int main() {
  // Setup GPU client with distributed options
  GpuClientOptions gpu_opts;
  gpu_opts.node_id = 0;
  gpu_opts.num_nodes = 4;
  
  // Create distributed runtime and get KV store
  auto dist_runtime = xla::CreateDistributedRuntime(...);
  auto kv_store = dist_runtime->GetKeyValueStore();
  
  // This is where the magic happens:
  // BuildDistributedDevices() will:
  //   1. Exchange addresses via KV store
  //   2. Initialize distributed context in singleton
  //   3. Return devices for the PJRT client
  
  auto client_status = GetStreamExecutorGpuClient(gpu_opts);
  if (!client_status.ok()) {
    LOG(FATAL) << "Failed to create GPU client";
  }
  auto client = std::move(client_status.value());
  
  // Create profilers (can be standard ProfileOptions, no modifications needed!)
  tensorflow::ProfileOptions profile_opts;
  // profile_opts stays simple and unchanged - no distributed config needed!
  
  auto profilers = tsl::profiler::CreateProfilers(profile_opts);
  
  // Start profiling
  for (auto& profiler : profilers) {
    if (!profiler->Start().ok()) {
      LOG(ERROR) << "Failed to start profiler";
    }
  }
  
  // During profiler initialization, it will:
  //   1. Check singleton for distributed context
  //   2. If present, create DistributedTimestampSynchronizer
  //   3. Exchange timestamps via sockets
  
  // Run GPU workload
  // ... execute computation ...
  
  // Stop profiling
  for (auto& profiler : profilers) {
    if (!profiler->Stop().ok()) {
      LOG(ERROR) << "Failed to stop profiler";
    }
  }
  
  // Collect and export data
  tensorflow::profiler::XSpace space;
  for (auto& profiler : profilers) {
    if (!profiler->CollectData(&space).ok()) {
      LOG(ERROR) << "Failed to collect profiler data";
    }
  }
  
  // space.planes now contains synchronized timestamps!
  
  return 0;
}

}  // namespace profiler
}  // namespace xla

// ============================================================================
// ADVANTAGES OF THE SINGLETON APPROACH:
// ============================================================================
// ✓ No need to modify TensorFlow's ProfileOptions proto
// ✓ Clean separation: addresses exchanged in PJRT client, used in profiler
// ✓ Thread-safe: singleton with mutex guards
// ✓ Easy to integrate: just call SetDistributedContext() once
// ✓ Profiler layer doesn't need to know about PJRT details
// ✓ Can be disabled gracefully: GetDistributedContext() returns optional
// ✓ Testing friendly: ResetContext() for test isolation
// ============================================================================
// 
// COMPLETE FLOW:
// 
//   BuildDistributedDevices() [PJRT]
//       │
//       ├─ ExchangeNodeAddresses() via KV store
//       │
//       └─ InitializeDistributedProfilerContext()
//           │
//           └─ DistributedProfilerContextManager::Get().SetDistributedContext()
//               (stores in singleton)
//
//   CreateGpuTracer() [Profiler Factory]
//       │
//       └─ Creates GpuTracer as usual
//           (no need to read ProfileOptions)
//
//   GpuTracer::Start()
//       │
//       └─ CreateRocmCollector()
//           │
//           └─ RocmTraceCollectorImpl::InitializeDistributedSync()
//               │
//               ├─ GetDistributedContext() from singleton
//               │
//               └─ Create DistributedTimestampSynchronizer
//                   │
//                   └─ Initialize sockets and exchange timestamps
//
// ============================================================================
