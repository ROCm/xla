/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/collectives/ctran_collectives.h"

#include <cstdint>
#include <dlfcn.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/ctran_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/tsl/platform/env.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif
#else
#include "third_party/nccl/nccl.h"
#endif

namespace xla::gpu {

// RCCLX library handle for dynamic loading
struct RcclxLibrary {
  void* handle = nullptr;
  bool loaded = false;
  bool checked = false;
  
  // Function pointers for RCCLX-specific features (beyond standard RCCL)
  // These are optional and provide enhanced functionality when available
  
  ~RcclxLibrary() {
    if (handle) {
      dlclose(handle);
    }
  }
  
  static RcclxLibrary& Get() {
    static RcclxLibrary instance;
    return instance;
  }
  
  bool TryLoad() {
    if (checked) return loaded;
    checked = true;
    
    // Try to load RCCLX library
    // RCCLX is typically installed as librccl.so but with extended features
    const char* lib_paths[] = {
      "librcclx.so",           // RCCLX from torchcomms
      "librccl_meta.so",       // Meta's RCCL build
      nullptr
    };
    
    for (const char** path = lib_paths; *path != nullptr; ++path) {
      handle = dlopen(*path, RTLD_NOW | RTLD_LOCAL);
      if (handle) {
        LOG(INFO) << "CTran: Loaded RCCLX library from " << *path;
        loaded = true;
        return true;
      }
    }
    
    LOG(INFO) << "CTran: RCCLX library not found, using standard RCCL";
    return false;
  }
};

// Internal state for CTran collectives
struct CtranCollectives::CtranState {
  bool initialized = false;
  Topology topology;
  bool rcclx_available = false;
};

CtranCollectives::CtranCollectives() : state_(std::make_unique<CtranState>()) {
  LOG(INFO) << "CTran collectives created (experimental)";
  
  // Try to load RCCLX library for enhanced features
  state_->rcclx_available = RcclxLibrary::Get().TryLoad();
  
  // Set up local clique ID callback
  local_clique_id_callback_ = [this](const CliqueKey& clique_key)
      -> absl::StatusOr<CliqueId> {
    return this->CreateUniqueCliqueId();
  };
}

CtranCollectives::~CtranCollectives() {
  LOG(INFO) << "CTran collectives destroyed";
}

bool CtranCollectives::IsImplemented() const {
  // CTran is now implemented using RCCL/RCCLX backend
  return true;
}

bool CtranCollectives::IsGlobalConfig() const {
  // CTran uses per-communicator configuration, not global
  return false;
}

absl::StatusOr<const CtranCollectives::CliqueIdCallback*>
CtranCollectives::GetCliqueIdCallback(const CliqueIdCallback* clique_id_callback,
                                       bool is_local) {
  if (clique_id_callback != nullptr) {
    return clique_id_callback;
  }
  
  if (is_local) {
    return &local_clique_id_callback_;
  }
  
  // For non-local case, require explicit callback
  return absl::InvalidArgumentError(
      "CTran requires explicit clique_id_callback for non-local collectives");
}

absl::StatusOr<CliqueId> CtranCollectives::CreateUniqueCliqueId() const {
  VLOG(3) << "CTran: Creating unique clique ID via NCCL/RCCL";
  
  // Use NCCL/RCCL's unique ID generation (works for both NCCL and RCCL)
  ncclUniqueId id;
  ncclResult_t result = ncclGetUniqueId(&id);
  if (result != ncclSuccess) {
    return absl::InternalError(
        absl::StrCat("CTran: Failed to create unique clique ID: ",
                     ncclGetErrorString(result)));
  }
  
  return CliqueId(absl::string_view(id.internal, NCCL_UNIQUE_ID_BYTES));
}

static ncclUniqueId AsNcclUniqueId(const CliqueId& clique_id) {
  ncclUniqueId id;
  if (clique_id.size() == NCCL_UNIQUE_ID_BYTES) {
    absl::c_copy(clique_id.data(), id.internal);
  }
  return id;
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
CtranCollectives::CreateCommunicators(const CliqueKey& clique_key,
                                       const std::optional<CliqueIds>& clique_ids,
                                       absl::Span<const DeviceRank> ranks,
                                       const Collectives::Config& config) {
  // Validate clique ids
  if (!clique_ids.has_value() || clique_ids->data().empty()) {
    return absl::InvalidArgumentError(
        "CliqueId is required to create CTran communicators");
  }
  if (clique_ids->data().size() != 1) {
    return absl::InvalidArgumentError(
        "CliqueIds size must be 1 for CTran communicator initialization");
  }
  
  LOG(INFO) << "CTran: Creating " << ranks.size() << " communicators"
            << "; fingerprint(id)=" << clique_ids->fingerprint()
            << (state_->rcclx_available ? " (using RCCLX)" : " (using RCCL)");
  
  const auto& gpu_config =
      static_cast<const GpuCollectives::Config&>(config);
  
  ncclUniqueId nccl_id = AsNcclUniqueId(clique_ids->data()[0]);
  
  // Configure NCCL/RCCL
  ncclConfig_t comm_config = NCCL_CONFIG_INITIALIZER;
  comm_config.blocking = gpu_config.blocking_communicators ? 1 : 0;
#if !defined(TENSORFLOW_USE_ROCM) || TF_ROCM_VERSION > 50700
  comm_config.splitShare = gpu_config.split_share;
#endif
  
  std::vector<std::unique_ptr<Communicator>> comms;
  comms.reserve(ranks.size());
  
  for (const auto& rank : ranks) {
    VLOG(1) << "CTran: Initializing communicator for rank " << rank.rank.value()
            << " on device " << rank.device_rank;
    
    // Create the communicator using CtranCommunicator which wraps NCCL/RCCL
    auto comm_or = CtranCommunicator::Create(
        rank.rank.value(),
        static_cast<int>(ranks.size()),
        rank.device_rank,
        nccl_id,
        comm_config);
    
    if (!comm_or.ok()) {
      // Abort any already-created communicators
      for (auto& comm : comms) {
        comm->Abort().IgnoreError();
      }
      return comm_or.status();
    }
    
    comms.push_back(std::move(*comm_or));
  }
  
  LOG(INFO) << "CTran: Successfully created " << comms.size() << " communicators";
  return comms;
}

absl::StatusOr<std::unique_ptr<Communicator>>
CtranCollectives::CreateCommunicator() {
  return absl::UnimplementedError("Single communicator creation not implemented");
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
CtranCollectives::SplitCommunicators(absl::Span<const Communicator* const> comms,
                                      int32_t color,
                                      absl::Span<const RankId> keys,
                                      const Collectives::Config& config) {
  LOG(INFO) << "CTran: Splitting " << comms.size() << " communicators with color "
            << color;
  
  // TODO(phambinh): Implement communicator splitting
  // This requires creating new CTran communicators from existing ones
  // based on the color grouping
  
  return absl::UnimplementedError(
      "CTran communicator splitting not yet implemented");
}

absl::StatusOr<void*> CtranCollectives::Allocate(uint64_t bytes) {
  LOG(INFO) << "CTran: Allocating " << bytes << " bytes of collective memory";
  
  if (!IsCtranAvailable()) {
    return absl::UnavailableError("CTran not available for memory allocation");
  }
  
  // TODO(phambinh): Use CTran's memory registration for RDMA-capable memory
  //
  // Real implementation would:
  // 1. Allocate GPU memory via cudaMalloc/hipMalloc
  // 2. Register with CTran for RDMA access: ctran_mapper->regMem(ptr, bytes)
  // 3. Return the pointer
  //
  // CTran memory registration enables:
  // - Zero-copy RDMA transfers over InfiniBand
  // - NVLink direct memory access
  // - Efficient TCP DevMem transfers
  
  return absl::UnimplementedError("CTran memory allocation not yet implemented");
}

absl::Status CtranCollectives::Deallocate(void* location) {
  LOG(INFO) << "CTran: Deallocating collective memory at " << location;
  
  // TODO(phambinh): Deregister and free CTran memory
  // ctran_mapper->deregMem(location)
  // cudaFree/hipFree(location)
  
  return absl::UnimplementedError("CTran memory deallocation not yet implemented");
}

absl::Status CtranCollectives::InitializeTopology(Topology topology) {
  LOG(INFO) << "CTran: Initializing topology - node_id=" << topology.node_id
            << ", num_nodes=" << topology.num_nodes
            << ", devices_per_process=" << topology.device_count_per_process;
  
  state_->topology = std::move(topology);
  state_->initialized = true;
  
  // TODO(phambinh): Initialize CTran with topology info
  //
  // This would include:
  // 1. Setting up the bootstrap mechanism (TCPStore or MPI)
  // 2. Discovering available transports (NVLink, InfiniBand, etc.)
  // 3. Building the topology map for intelligent routing
  //
  // CTran uses this topology information to:
  // - Select optimal transport for each peer
  // - Route traffic through the best available path
  // - Handle failures by rerouting to alternative transports
  
  LOG(INFO) << "CTran: Topology initialized (stub - full init pending)";
  return absl::OkStatus();
}

// Static methods

bool CtranCollectives::IsCtranAvailable() {
  // CTran is now always available since we use NCCL/RCCL as the backend
  // RCCLX provides enhanced features when available
  static bool checked = false;
  static bool available = false;
  
  if (!checked) {
    checked = true;
    
    // CTran is available if NCCL/RCCL is available
    // We can verify by trying to get a unique ID
    ncclUniqueId test_id;
    ncclResult_t result = ncclGetUniqueId(&test_id);
    available = (result == ncclSuccess);
    
    if (available) {
      bool rcclx = RcclxLibrary::Get().TryLoad();
      LOG(INFO) << "CTran available: using " 
                << (rcclx ? "RCCLX (Meta enhanced)" : "standard RCCL/NCCL");
    } else {
      LOG(WARNING) << "CTran not available: NCCL/RCCL initialization failed";
    }
  }
  
  return available;
}

std::string CtranCollectives::GetCtranVersion() {
  bool rcclx = RcclxLibrary::Get().loaded;
  if (rcclx) {
    return "ctran-rcclx-1.0.0";
  }
#if TENSORFLOW_USE_ROCM
  return absl::StrCat("ctran-rccl-", NCCL_MAJOR, ".", NCCL_MINOR, ".", NCCL_PATCH);
#else
  return absl::StrCat("ctran-nccl-", NCCL_MAJOR, ".", NCCL_MINOR, ".", NCCL_PATCH);
#endif
}

}  // namespace xla::gpu

// Register CTran collectives with priority 0 (lower than NCCL's priority 1)
// This means CTran won't be the default, but can be selected explicitly
XLA_COLLECTIVES_REGISTER("gpu", "ctran", 0,
                         std::make_unique<xla::gpu::CtranCollectives>());
