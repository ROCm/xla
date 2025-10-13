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
#include "xla/backends/gpu/collectives/rocshmem_collectives.h"

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
#include "xla/backends/gpu/collectives/rocshmem_communicator.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/numbers.h"

using namespace rocshmem;

namespace xla::gpu {

RocshmemCollectives::~RocshmemCollectives() {
  if (initialized_) Finalize();
}

RocshmemCollectives* RocshmemCollectives::Default() {
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Get("gpu", "nvshmem");
  CHECK_OK(collectives) << "Failed to get ROCSHMEM collectives";  // Crash OK

  if (auto* rocshmem_collectives =
          tsl::down_cast<RocshmemCollectives*>(*collectives)) {
    return rocshmem_collectives;
  }
  LOG(FATAL) << "Unsupported collectives implementation for ROCSHMEM";
}

absl::Status RocshmemCollectives::InitializeTopology(Topology topology) {
  process_id_ = topology.node_id;
  num_processes_ = topology.num_nodes;
  device_count_per_process_ = topology.device_count_per_process;
  kv_store_ = topology.kv_store;
  return InitializeOnce();
}

absl::Status RocshmemCollectives::InitializeOnce() {
  auto init_fn = [this]() -> absl::Status {
    if (process_id_ == -1) {
      LOG(FATAL)
          << "RocshmemCollectives::SetEnvInfo was not called before using "
             "ROCSHMEM API";
    }
    if (device_count_per_process_ != 1) {
      LOG(FATAL) << "ROCSHMEM API is only supported with one device per process";
    }
    rocshmem_init_attr_t init_attr;
    rocshmem_uniqueid_t rocshmem_id;

    // Initialize ROCSHMEM
    if (std::shared_ptr<KeyValueStoreInterface> kv_store = kv_store_.lock()) {
      if (process_id_ == 0) {
        if (rocshmem_get_uniqueid(&rocshmem_id) != 0) {
          return absl::InternalError("rocshmem_get_uniqueid failed.");
        }
        char buf[sizeof(rocshmem_uniqueid_t)];
        std::memcpy(buf, &rocshmem_id, sizeof(rocshmem_uniqueid_t));
        absl::string_view rocshmem_id_str{buf, sizeof(buf)};
        TF_RETURN_IF_ERROR(kv_store->Set(kKvStoreKey, rocshmem_id_str));
      } else {
        TF_ASSIGN_OR_RETURN(std::string id_str,
                            kv_store->Get(kKvStoreKey, absl::Minutes(10)));
        CHECK(id_str.size() >= sizeof(rocshmem_uniqueid_t));
        std::memcpy(&rocshmem_id, id_str.data(), sizeof(rocshmem_uniqueid_t));
      }
    } else {
      return absl::InternalError(
          "KV store is not available for rocshmem initialization.");
    }

    if (rocshmem_set_attr_uniqueid_args(process_id_, num_processes_,
                                        &rocshmem_id, &init_attr) != 0) {
      return absl::InternalError("rocshmem_set_attr_uniqueid_args failed.");
    }
    if (rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &init_attr) != 0) {
      return absl::InternalError("rocshmem_hostlib_init_attr failed.");
    }

    VLOG(3) << absl::StreamFormat(
        "Initialized ROCSHMEM on process %d; num_processes=%llu", process_id_,
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

void RocshmemCollectives::Finalize() {
  VLOG(0) << absl::StreamFormat(
      "Finilizing ROCSHMEM on process %d; num_processes=%llu", process_id_,
      num_processes_);
  rocshmem_finalize();
  VLOG(0) << absl::StreamFormat(
      "Finilize ROCSHMEM on process %d; num_processes=%llu DONE", process_id_,
      num_processes_);

}

absl::StatusOr<void*> RocshmemCollectives::Allocate(uint64_t bytes) {
  TF_RETURN_IF_ERROR(InitializeOnce());
  VLOG(3) << absl::StreamFormat(
      "Start allocation of %s (%llu bytes) for ROCSHMEM",
      tsl::strings::HumanReadableNumBytes(bytes), bytes);
  void* buffer = rocshmem_malloc(bytes);
  if (buffer == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %s (%llu bytes) from ROCSHMEM memory",
        tsl::strings::HumanReadableNumBytes(bytes), bytes));
  }
  return buffer;
}

absl::Status RocshmemCollectives::Deallocate(void* buffer) {
  TF_RETURN_IF_ERROR(InitializeOnce());
  VLOG(3) << absl::StreamFormat("Start de-allocation for ROCSHMEM buffer: %p",
                                buffer);
  rocshmem_free(buffer);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Communicator>>
RocshmemCollectives::CreateCommunicator() {

  TF_ASSIGN_OR_RETURN(auto *ptr, 
        Allocate(sizeof(rocshmem_team_t) * RocshmemCommunicator::kMaxTeams));
  auto *teams = static_cast< rocshmem_team_t *>(ptr);
  auto comm = absl::WrapUnique(new RocshmemCommunicator(this, teams));
  
  int npes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);
  for (uint32_t i = 0; i < RocshmemCommunicator::kMaxTeams; i++) {
    auto res = rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, 
              npes, nullptr, 0, &teams[i]);
    if (res != 0) {
      return absl::InternalError("Unable to create a rocshmem team!");
    }
  }
  VLOG(1) << "Created " << *comm << " npes " << npes;
  return comm;
}

}  // namespace xla::gpu

// RocshmemCollectives currently does not implement GpuCollectives, so it cannot
// be used as a host-side collectives library. Therefore, set priority to -100.
XLA_COLLECTIVES_REGISTER("gpu", "nvshmem", -100,
                         std::make_unique<xla::gpu::RocshmemCollectives>());
