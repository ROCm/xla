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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/rccl_errors.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

// Returns the outcome of an RCCL call; on a non-blocking communicator, where
// the call returns ncclInProgress, first polls until it has completed.
absl::Status WaitUntilDone(ncclComm_t comm, ncclResult_t status,
                           const CancellationToken* cancel) {
  if (status != ncclInProgress) {
    return XLA_RCCL_STATUS(status);
  }
  CancellationToken never_cancelled;
  return PollUntilDone(comm, cancel != nullptr ? *cancel : never_cancelled);
}

}  // namespace

RcclSymmetricMemory::RcclSymmetricMemory(
    ncclComm_t comm, ncclWindow_t win, stream_executor::DeviceAddressBase addr,
    tsl::Executor* executor, const CancellationToken* cancel)
    : comm_(comm),
      win_(win),
      addr_(addr),
      executor_(executor),
      cancel_(cancel) {}

absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>>
RcclSymmetricMemory::Create(ncclComm_t comm,
                            stream_executor::DeviceAddressBase addr,
                            bool symmetric_window, tsl::Executor* executor,
                            const CancellationToken* cancel) {
  // NCCL_WIN_COLL_SYMMETRIC lets RCCL use its copy-engine (SDMA) collectives
  // on the window; the caller guarantees the memory is symmetric across ranks.
  const int win_flags =
      symmetric_window ? NCCL_WIN_COLL_SYMMETRIC : NCCL_WIN_DEFAULT;
  VLOG(3) << absl::StrFormat(
      "Create RCCL symmetric memory on comm=%p from: ptr=%p; size=%ld; "
      "win_flags=%d",
      comm, addr.opaque(), addr.size(), win_flags);

  // A collective call: every rank of `comm` registers its buffer. RCCL >= 2.30
  // implements it; older RCCL fails or returns a null window, as do single-rank
  // communicators. Either way the buffer still works for collectives,
  // unregistered.
  // `win` is only set once the registration has completed.
  ncclWindow_t win = nullptr;
  const absl::Status status = WaitUntilDone(
      comm,
      ncclCommWindowRegister(comm, addr.opaque(), addr.size(), &win, win_flags),
      cancel);
  if (!status.ok()) {
    LOG(WARNING) << "ncclCommWindowRegister failed (" << status
                 << "); using the buffer unregistered: ptr=" << addr.opaque()
                 << " size=" << addr.size();
    win = nullptr;
  }
  return absl::WrapUnique(
      new RcclSymmetricMemory(comm, win, addr, executor, cancel));
}

RcclSymmetricMemory::~RcclSymmetricMemory() {
  VLOG(3) << absl::StrFormat("Destroy %v", *this);
  if (win_ == nullptr) {
    return;
  }
  // Like every operation on the communicator, run on its executor (if any).
  auto deregister = [this]() -> absl::Status {
    return WaitUntilDone(comm_, ncclCommWindowDeregister(comm_, win_), cancel_);
  };
  const absl::Status status =
      executor_ != nullptr ? MakeFutureOn<void>(*executor_, deregister).Await()
                           : deregister();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to deregister RCCL symmetric memory: " << status;
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
