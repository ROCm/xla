/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/backends/gpu/collectives/mori_collectives.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/mori_communicator.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/numbers.h"

#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/mori_kernels.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/debug_options_flags.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
//#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"

using namespace mori;

namespace se = ::stream_executor;

namespace xla::gpu {

#define XLA_MORI_RETURN_IF_ERROR(expr) \
  do { \
    auto status = (expr); \
    if (status != 0) { \
      return absl::InternalError(absl::StrFormat("MORI operation failed: %d", status)); \
    } \
  } while (0)

  //===----------------------------------------------------------------------===//
// RcclIdStore
//===----------------------------------------------------------------------===//

namespace {

// Well-known key under which the single world-wide MORI unique id is published
// to the key-value store during multi-process eager initialization.
constexpr absl::string_view kGlobalCliqueUidKey = "mori_shmem_global_clique_uid";

// Exchanges a single MORI unique id across processes via `kv_store`: the owning
// (root) process generates a fresh id and publishes it under `key`, while every
// other process blocks (up to `timeout`) until it can read it back. This is the
// common building block previously buried in the lazy per-clique id store; it is
// now shared by the multi-process eager InitializeTopology path.
absl::StatusOr<CliqueId> ExchangeUniqueId(KeyValueStoreInterface& kv_store,
                                          absl::string_view key, bool is_owner,
                                          MoriCollectives& mori,
                                          absl::Duration timeout) {
  if (is_owner) {
    TF_ASSIGN_OR_RETURN(CliqueId clique_id, mori.CreateUniqueCliqueId());
    TF_RETURN_IF_ERROR(kv_store.Set(key, clique_id.ToString()));
    return clique_id;
  }
  TF_ASSIGN_OR_RETURN(std::string id_str, kv_store.Get(key, timeout));
  return CliqueId(id_str);
}

}  // namespace

MoriCollectives::~MoriCollectives() {
  // NOTE this is most probably wrong since we need to call finalize 
  // for all threads !
  if (initialized_) Finalize();
}

absl::StatusOr<CliqueId> MoriCollectives::CreateUniqueCliqueId() const {
  VLOG(3) << "Create MORI unique clique id";
  shmem::mori_shmem_uniqueid_t id;
  XLA_MORI_RETURN_IF_ERROR(shmem::ShmemGetUniqueId(&id));
  return CliqueId(absl::string_view(
      reinterpret_cast<char*>(id.data()), MORI_SHMEM_UNIQUE_ID_BYTES));
}

static absl::StatusOr<shmem::mori_shmem_uniqueid_t> AsMoriUniqueId(
                                                    const CliqueId& clique_id) {
  if (clique_id.size() != MORI_SHMEM_UNIQUE_ID_BYTES) {
    return Internal(
        "CliqueId size is not equal to MORI_SHMEM_UNIQUE_ID_BYTES: %d vs %d",
        clique_id.size(), MORI_SHMEM_UNIQUE_ID_BYTES);
  }
  shmem::mori_shmem_uniqueid_t id;
  absl::c_copy(clique_id.data(), id.data());
  return id;
}

void MoriCollectives::Finalize() {
  VLOG(3) << "Finilizing MORI";
  shmem::ShmemFinalize();   
}

absl::Status MoriCollectives::InitPe(int32_t rank, int32_t nranks,
                                     const CliqueId& clique_id,
                                     se::StreamExecutor* executor) {
  // ShmemInitAttr keys the per-device MORI state off the calling thread's active
  // HIP device, so we must activate `executor`'s context here.
  auto activate_context = executor->Activate();
  TF_ASSIGN_OR_RETURN(auto uid, AsMoriUniqueId(clique_id));
  shmem::mori_shmem_init_attr_t init_attr;
  XLA_MORI_RETURN_IF_ERROR(
      shmem::ShmemSetAttrUniqueIdArgs(rank, nranks, &uid, &init_attr));
  XLA_MORI_RETURN_IF_ERROR(
      shmem::ShmemInitAttr(shmem::MORI_SHMEM_INIT_WITH_UNIQUEID, &init_attr));
  VLOG(1) << "Initialized MORI PE rank " << rank << " of " << nranks;
  return absl::OkStatus();
}

absl::StatusOr<void*> MoriCollectives::Allocate(uint64_t bytes) {
  void* buffer = shmem::ShmemMalloc(bytes); // ShmemMallocAlign
  if (buffer == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s (%llu bytes) from MORI memory",
        tsl::strings::HumanReadableNumBytes(bytes), bytes));
  }
  VLOG(3) << absl::StreamFormat(
    "Allocated %s (%llu bytes) for MORI: %p",
    tsl::strings::HumanReadableNumBytes(bytes), bytes, buffer);

  return buffer;
}

absl::Status MoriCollectives::Deallocate(void* buffer) {
  VLOG(3) << absl::StreamFormat("Start de-allocation for MORI buffer: %p",
                                buffer);
  shmem::ShmemFree(buffer);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
MoriCollectives::CreateCommunicatorsWithCancel(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, const Collectives::Config& config,
    std::shared_ptr<CancellationToken> cancel) {
  // Validate clique ids. With the MORI backend, we rely on the host to exchange
  // unique clique ids.
  if (!clique_ids.has_value() || clique_ids->data().empty()) {
    return InvalidArgument("CliqueId is required to create MORI communicators");
  }
  if (clique_ids->data().size() != 1) {
    return InvalidArgument(
        "CliqueIds size must be 1 for MORI communicator initialization");
  }
  VLOG(1) << "Initialize MORI communicator for " << ranks.size() << " devices"
          << "; fingerprint(id)=" << clique_ids->fingerprint();

  const auto& gpu_config =
      tsl::down_cast<const GpuCollectives::Config&>(config);
  if (!gpu_config.blocking_communicators && !gpu_config.async_execution) {
    return FailedPrecondition(
        "GpuCollectives::Config blocking_communicators is false, but "
        "async_execution is false. Non-blocking communicators require "
        "asynchronous execution.");
  }

  // make_comm returns a new ncclComm_t.
  auto make_comm = [&, this](int i) 
                        -> absl::StatusOr<std::unique_ptr<MoriCommunicator>> {
    VLOG(1) << "Initialize MORI communicator for rank #" << ranks[i].rank
            << " of " << clique_key.num_devices()
            << "; fingerprint(id)=" << clique_ids->fingerprint()
            << "; size(id)=" << clique_ids->data().size();
    auto* device = tsl::down_cast<GpuCollectives::Device*>(ranks[i].device);
    //TF_RET_CHECK(device != nullptr);

    // When MORI was already initialized eagerly (see InitializeTopology), we
    // only build the communicator wrapper. Otherwise (e.g. unit tests that
    // bypass InitializeTopology) we lazily initialize this PE here. ShmemInitAttr
    // is idempotent, but the initialized_ gate avoids redundant uid setup.
    auto activate_context = device->stream_executor()->Activate();
    if (!initialized_) {
      TF_RETURN_IF_ERROR(InitPe(ranks[i].rank.value(), clique_key.num_devices(),
                                clique_ids->at(0),
                                device->stream_executor()));
    }
    return MoriCommunicator::Create(this, cancel);
  };

  // Create all communicators. Each communicator is created on its own thread.
  std::vector<std::unique_ptr<Communicator>> comms(ranks.size());
  absl::Status status;
  absl::once_flag once;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "CreateCommunicators",
                                 ranks.size());
    for (size_t i = 0; i < ranks.size(); ++i) {
      pool.Schedule([&, i]() {
        auto status_or_comm = make_comm(i);
        if (!status_or_comm.ok()) {
          absl::call_once(once, [&] { status = status_or_comm.status(); });
          return;
        }
        comms[i] = std::move(status_or_comm.value());
      });
    }
  }  // pool's destructor blocks until all scheduled work is done.
  TF_RETURN_IF_ERROR(status);
  initialized_ = true;
  return comms;
}

absl::Status MoriCollectives::EagerInitPes(absl::Span<const PeInit> pes,
                                           int32_t nranks,
                                           const CliqueId& clique_id) {
  // All PEs share one unique id and must initialize concurrently so that
  // ShmemInitAttr's bootstrap collective (which spans every PE in the world) can
  // complete. Each PE is initialized on its own thread because ShmemInitAttr
  // keys off the calling thread's active device.
  absl::Status status;
  absl::once_flag once;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "MoriEagerInit",
                                 pes.size());
    for (const PeInit& pe : pes) {
      pool.Schedule([&, pe]() {
        if (absl::Status s =
                InitPe(pe.global_rank, nranks, clique_id, pe.executor);
            !s.ok()) {
          absl::call_once(once, [&] { status = s; });
        }
      });
    }
  }  // pool's destructor blocks until all scheduled work is done.
  return status;
}

absl::StatusOr<GpuCollectives::CliqueIdCallback>
MoriCollectives::InitializeTopology(const Topology& topology) {
  VLOG(1) << "InitializeTopology: num_processes=" << topology.num_processes
          << " device_count_per_process=" << topology.device_count_per_process
          << " kv_store=" << (topology.kv_store != nullptr);

  const int32_t local_device_count =
      static_cast<int32_t>(topology.device_count_per_process);
  if (local_device_count <= 0) {
    return nullptr;
  }

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithName("ROCM"));

  // Single process: eagerly initialize MORI for every local device as a single
  // global clique so that collective-memory allocations can use MORI's static
  // heap (ShmemMalloc) before any executable runs. The unique id is generated
  // locally and local device ordinals are exactly the global ranks.
  if (topology.num_processes <= 1) {
    TF_ASSIGN_OR_RETURN(CliqueId clique_id, CreateUniqueCliqueId());
    std::vector<PeInit> pes;
    pes.reserve(local_device_count);
    for (int32_t rank = 0; rank < local_device_count; ++rank) {
      TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                          platform->ExecutorForDevice(rank));
      pes.push_back(PeInit{/*global_rank=*/rank, executor});
    }
    TF_RETURN_IF_ERROR(
        EagerInitPes(pes, /*nranks=*/local_device_count, clique_id));
    initialized_ = true;
    VLOG(1) << "Eagerly initialized MORI for " << local_device_count
            << " local devices";
    return nullptr;
  }

  // Multi-process: eagerly initialize MORI as a single world-wide clique, using
  // the key-value store to distribute the one shared unique id. This mirrors the
  // single-process path (so ShmemMalloc works before any executable runs) but
  // maps local devices to their global ranks derived from `device_to_process`.
  if (topology.kv_store == nullptr) {
    return InvalidArgument(
        "Multi-process MORI initialization requires a key-value store");
  }

  // The global rank of a device is the index of its GlobalDeviceId in the
  // globally-sorted list of all devices; the local ordinal is that device's
  // position among this process's devices in the same ascending order (which is
  // the order StreamExecutor uses for local device indices).
  std::vector<GlobalDeviceId> global_devices;
  global_devices.reserve(topology.device_to_process.size());
  for (const auto& device_and_process : topology.device_to_process) {
    global_devices.push_back(device_and_process.first);
  }
  absl::c_sort(global_devices);

  const int32_t nranks = static_cast<int32_t>(global_devices.size());
  TF_RET_CHECK(nranks == static_cast<int32_t>(topology.num_processes *
                                              topology.device_count_per_process))
      << "device_to_process size (" << nranks
      << ") must equal num_processes * device_count_per_process ("
      << topology.num_processes * topology.device_count_per_process << ")";

  // The process owning global rank 0 (the smallest GlobalDeviceId) is the root
  // that generates and publishes the shared unique id.
  const ProcessId owner =
      topology.device_to_process.at(global_devices.front());
  TF_ASSIGN_OR_RETURN(
      CliqueId clique_id,
      ExchangeUniqueId(*topology.kv_store, kGlobalCliqueUidKey,
                       /*is_owner=*/topology.process_id == owner, *this,
                       absl::Minutes(10)));

  std::vector<PeInit> pes;
  pes.reserve(local_device_count);
  int32_t local_ordinal = 0;
  for (int32_t global_rank = 0; global_rank < nranks; ++global_rank) {
    if (topology.device_to_process.at(global_devices[global_rank]) !=
        topology.process_id) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        platform->ExecutorForDevice(local_ordinal));
    pes.push_back(PeInit{global_rank, executor});
    ++local_ordinal;
  }
  TF_RET_CHECK(local_ordinal == local_device_count)
      << "Number of local devices found in device_to_process (" << local_ordinal
      << ") does not match device_count_per_process (" << local_device_count
      << ")";

  TF_RETURN_IF_ERROR(EagerInitPes(pes, nranks, clique_id));
  initialized_ = true;
  VLOG(1) << "Eagerly initialized MORI for " << local_device_count
          << " local devices out of " << nranks << " global ranks";

  // MORI is now fully initialized as a single global clique; communicators only
  // wrap the already-initialized state. Non-local cliques still require a
  // (placeholder) clique id to satisfy CreateCommunicators validation, but the
  // id itself is unused once `initialized_` is set (as in the single-process
  // path). Returning a trivial callback avoids the framework's default local-
  // only clique id callback, which rejects multi-process cliques.
  return [](const CliqueKey&) { return CliqueIds(CliqueId("")); };
}

}  // namespace xla::gpu

// MoriCollectives currently does not implement GpuCollectives, so it cannot
// be used as a host-side collectives library. Therefore, set priority to -100.
XLA_COLLECTIVES_REGISTER("ROCM", "mori", -100,
                         std::make_unique<xla::gpu::MoriCollectives>());
