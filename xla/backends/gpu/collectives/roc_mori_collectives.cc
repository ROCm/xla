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
#include "xla/backends/gpu/collectives/roc_mori_collectives.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/roc_mori_communicator.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/numbers.h"

namespace xla::gpu {

MoriCollectives::~MoriCollectives() {
  if (initialized_) Finalize();
}

MoriCollectives* MoriCollectives::Default() {
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Get("ROCM", "nvshmem");
  CHECK_OK(collectives) << "Failed to get MORI collectives";  // Crash OK

  if (auto* mori_collectives =
          tsl::down_cast<MoriCollectives*>(*collectives)) {
    return mori_collectives;
  }
  LOG(FATAL) << "Unsupported collectives implementation for MORI";
}

absl::StatusOr<GpuCollectives::CliqueIdCallback>
MoriCollectives::InitializeTopology(const Topology& topology) {
  process_id_ = topology.process_id;
  num_processes_ = topology.num_processes;
  device_count_per_process_ = topology.device_count_per_process;
  kv_store_ = topology.kv_store;
  
  TF_RETURN_IF_ERROR(InitializeOnce());
  return [](const CliqueKey&) { return CliqueIds(CliqueId("")); };
}

absl::Status MoriCollectives::InitializeOnce() {
  auto init_fn = [this]() -> absl::Status {
    if (process_id_ == -1) {
      LOG(FATAL)
          << "MoriCollectives::SetEnvInfo was not called before using "
             "MORI API";
    }
    if (device_count_per_process_ != 1) {
      LOG(FATAL) << "MORI API is only supported with one device per process";
    }
    // rocm_mori_init_attr_t init_attr;
    // rocm_mori_uniqueid_t rocm_mori_id;

    // // Initialize MORI
    // if (std::shared_ptr<KeyValueStoreInterface> kv_store = kv_store_.lock()) {
    //   if (process_id_ == 0) {
    //     if (rocm_mori_get_uniqueid(&rocm_mori_id) != 0) {
    //       return absl::InternalError("rocm_mori_get_uniqueid failed.");
    //     }
    //     char buf[sizeof(rocm_mori_uniqueid_t)];
    //     std::memcpy(buf, &rocm_mori_id, sizeof(rocm_mori_uniqueid_t));
    //     absl::string_view rocm_mori_id_str{buf, sizeof(buf)};
    //     TF_RETURN_IF_ERROR(kv_store->Set(kKvStoreKey, rocm_mori_id_str));
    //   } else {
    //     TF_ASSIGN_OR_RETURN(std::string id_str,
    //                         kv_store->Get(kKvStoreKey, absl::Minutes(10)));
    //     CHECK(id_str.size() >= sizeof(rocm_mori_uniqueid_t));
    //     std::memcpy(&rocm_mori_id, id_str.data(), sizeof(rocm_mori_uniqueid_t));
    //   }
    // } else {
    //   return absl::InternalError(
    //       "KV store is not available for rocshmem initialization.");
    // }

    // if (rocm_mori_set_attr_uniqueid_args(process_id_, num_processes_,
    //                                     &rocm_mori_id, &init_attr) != 0) {
    //   return absl::InternalError("rocm_mori_set_attr_uniqueid_args failed.");
    // }
    // if (rocm_mori_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &init_attr) != 0) {
    //   return absl::InternalError("rocm_mori_hostlib_init_attr failed.");
    // }

    VLOG(3) << absl::StreamFormat(
      "Initialized MORI on process %d; num_processes=%llu", process_id_.value(),
      num_processes_);
    return absl::OkStatus();
  };

  static absl::once_flag once_flag;
  absl::Status status = absl::OkStatus();
  absl::call_once(once_flag, [&]() {
    status = init_fn();
    initialized_ = true;
  });
  return status;
}

void MoriCollectives::Finalize() {
  VLOG(0) << absl::StreamFormat(
      "Finilizing MORI on process %d; num_processes=%llu", process_id_.value(),
      num_processes_);
  // rocm_mori_finalize();
}

absl::StatusOr<void*> MoriCollectives::Allocate(uint64_t bytes) {
  TF_RETURN_IF_ERROR(InitializeOnce());
  VLOG(3) << absl::StreamFormat(
      "Start allocation of %s (%llu bytes) for MORI",
      tsl::strings::HumanReadableNumBytes(bytes), bytes);
  void* buffer = nullptr; //rocm_mori_malloc(bytes);
  if (buffer == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s (%llu bytes) from MORI memory",
        tsl::strings::HumanReadableNumBytes(bytes), bytes));
  }
  return buffer;
}

absl::Status MoriCollectives::Deallocate(void* buffer) {
  TF_RETURN_IF_ERROR(InitializeOnce());
  VLOG(3) << absl::StreamFormat("Start de-allocation for MORI buffer: %p",
                                buffer);
  // rocm_mori_free(buffer);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Communicator>>
MoriCollectives::CreateCommunicator() {

  // TF_ASSIGN_OR_RETURN(auto *ptr, 
  //       Allocate(sizeof(rocm_mori_team_t) * MoriCommunicator::kMaxTeams));
  // auto *teams = static_cast< rocm_mori_team_t *>(ptr);
  auto comm = absl::WrapUnique(new MoriCommunicator(this));
  
  // int npes = rocm_mori_team_n_pes(ROCSHMEM_TEAM_WORLD);
  // for (uint32_t i = 0; i < MoriCommunicator::kMaxTeams; i++) {
  //   auto res = rocm_mori_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, 
  //             npes, nullptr, 0, &teams[i]);
  //   if (res != 0) {
  //     return absl::InternalError("Unable to create a rocshmem team!");
  //   }
  // }
  VLOG(1) << "Created " << *comm;// << " npes " << npes;
  return comm;
}

}  // namespace xla::gpu

// MoriCollectives currently does not implement GpuCollectives, so it cannot
// be used as a host-side collectives library. Therefore, set priority to -100.
XLA_COLLECTIVES_REGISTER("ROCM", "nvshmem", -100,
                         std::make_unique<xla::gpu::MoriCollectives>());
