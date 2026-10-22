/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/rccl_symmetric_memory.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/rccl_errors.h"
#include "xla/stream_executor/device_address.h"
#include "xla/util.h"

namespace xla::gpu {

RcclSymmetricMemory::RcclSymmetricMemory(
    ncclComm_t comm, ncclWindow_t win, stream_executor::DeviceAddressBase addr)
    : comm_(comm), win_(win), addr_(addr) {}

absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>>
RcclSymmetricMemory::Create(ncclComm_t comm,
                            stream_executor::DeviceAddressBase addr) {
  VLOG(3) << absl::StrFormat(
      "Create RCCL symmetric memory on comm=%p from: ptr=%p; size=%ld", comm,
      addr.opaque(), addr.size());

  // A collective call: every rank of `comm` registers its buffer. RCCL >= 2.30
  // implements it (with copy-engine collectives on registered windows); older
  // RCCL fails or returns a null window, as do single-rank communicators. Either
  // way the buffer still works for collectives, unregistered.
  ncclWindow_t win = nullptr;
  ncclResult_t status = ncclCommWindowRegister(
      comm, addr.opaque(), addr.size(), &win, NCCL_WIN_COLL_SYMMETRIC);
  if (status != ncclSuccess) {
    LOG(WARNING) << "ncclCommWindowRegister failed ("
                 << ncclGetErrorString(status)
                 << "); using the buffer unregistered: ptr=" << addr.opaque()
                 << " size=" << addr.size();
    win = nullptr;
  }
  return absl::WrapUnique(new RcclSymmetricMemory(comm, win, addr));
}

RcclSymmetricMemory::~RcclSymmetricMemory() {
  VLOG(3) << absl::StrFormat("Destroy %v", *this);
  if (win_ == nullptr) {
    return;
  }
  ncclResult_t status = ncclCommWindowDeregister(comm_, win_);
  if (status != ncclSuccess) {
    LOG(ERROR) << "Failed to deregister RCCL symmetric memory: "
               << ncclGetErrorString(status);
  }
}

stream_executor::DeviceAddressBase RcclSymmetricMemory::addr() const {
  return addr_;
}

std::string RcclSymmetricMemory::ToString() const {
  return absl::StrFormat(
      "RcclSymmetricMemory(comm=%p, win=%p, ptr=%p, size=%ld)", comm_, win_,
      addr_.opaque(), addr_.size());
}

RcclSymmetricMemory::PackedKernelArg RcclSymmetricMemory::PackKernelArg()
    const {
  return win_;
}

}  // namespace xla::gpu
