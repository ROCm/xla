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
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/rccl_errors.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {

#if (TF_ROCM_VERSION >= 100000)
// Only used by the destructor's ROCm/RCCL 10.0+ path (window deregistration).
// Guard the definition so it is not compiled as an unused static function on
// ROCm < 10.0 builds (which would trip -Wunused-function under -Werror).
Future<> Execute(absl::AnyInvocable<absl::Status() &&> f,
                 std::shared_ptr<tsl::Executor> executor) {
  return executor ? MakeFutureOn<void>(*executor, std::move(f))
                  : Future<>(std::move(f)());
}
#endif  // TF_ROCM_VERSION >= 100000

}  // namespace

RcclSymmetricMemory::RcclSymmetricMemory(
    ncclComm_t comm, ncclWindow_t win, stream_executor::DeviceAddressBase addr,
    std::shared_ptr<tsl::Executor> executor)
    : comm_(comm), win_(win), addr_(addr), executor_(std::move(executor)) {}

absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>>
RcclSymmetricMemory::Create(ncclComm_t comm,
                            stream_executor::DeviceAddressBase addr,
                            std::shared_ptr<tsl::Executor> executor) {
  VLOG(3) << absl::StrFormat(
      "Create RCCL symmetric memory on comm=%p from: ptr=%p; size=%ld", comm,
      addr.opaque(), addr.size());
#if (TF_ROCM_VERSION >= 100000)
  // ncclCommWindowRegister is available in ROCm/RCCL 10.0+.
  ncclWindow_t win = nullptr;
  ncclResult_t register_status = ncclCommWindowRegister(
      comm, addr.opaque(), addr.size(), &win, NCCL_WIN_COLL_SYMMETRIC);
  if (register_status != ncclSuccess) {
    LOG(ERROR) << absl::StrFormat(
        "ncclCommWindowRegister failed on comm=%p (ptr=%p, size=%ld): %s", comm,
        addr.opaque(), addr.size(), ncclGetErrorString(register_status));
    XLA_RCCL_RETURN_IF_ERROR(register_status);
  }
  return absl::WrapUnique(
      new RcclSymmetricMemory(comm, win, addr, std::move(executor)));
#else
  // Symmetric-memory windows require ncclCommWindowRegister (ROCm/RCCL
  // >= 10.0).
  return absl::UnimplementedError(
      "RCCL symmetric memory requires ncclCommWindowRegister (ROCm/RCCL "
      ">= 10.0)");
#endif  // TF_ROCM_VERSION >= 100000
}

RcclSymmetricMemory::~RcclSymmetricMemory() {
#if (TF_ROCM_VERSION >= 100000)
  // ncclCommWindowDeregister is available in ROCm/RCCL 10.0+.
  //
  // comm_ is a non-owning ncclComm_t. If this window outlived its
  // RcclCommunicator (whose destructor calls ncclCommDestroy), the
  // ncclCommWindowDeregister() below would be a use-after-free. Callers (the
  // collective clique) must destroy all windows before destroying the
  // communicator.
  if (win_ != nullptr) {
    CHECK(comm_ != nullptr)
        << "RcclSymmetricMemory destroyed with a live window but null comm_; "
           "the owning ncclComm_t was destroyed first (window="
        << win_ << ").";
  }

  // The deregistration below is posted to executor_ and awaited. If this
  // destructor ran on executor_'s thread (e.g. released inside another Execute
  // callback), that would self-deadlock.
  VLOG(3) << absl::StrFormat(
      "Destroy %v with addr=%p, size=%ld executor=%p (awaiting window "
      "deregistration on executor)",
      *this, addr_.opaque(), addr_.size(), executor_.get());

  absl::Status status =
      Execute(
          [&] {
            if (win_ != nullptr) {
              XLA_RCCL_LOG_IF_ERROR(ncclCommWindowDeregister(comm_, win_));
            }
            return absl::OkStatus();
          },
          executor_)
          .Await();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to destroy RCCL symmetric memory: " << status;
  }
#endif  // TF_ROCM_VERSION >= 100000
}

stream_executor::DeviceAddressBase RcclSymmetricMemory::addr() const {
  return addr_;
}

absl::StatusOr<stream_executor::DeviceAddressBase>
RcclSymmetricMemory::multimem_addr() const {
#if (TF_ROCM_VERSION >= 100000)
  // ncclGetLsaMultimemDevicePointer is available in ROCm/RCCL 10.0+.
  void* multimem = nullptr;
  XLA_RCCL_RETURN_IF_ERROR(ncclGetLsaMultimemDevicePointer(win_, 0, &multimem));
  if (multimem) {
    return stream_executor::DeviceAddressBase(multimem, addr_.size());
  }
#endif  // TF_ROCM_VERSION >= 100000
  return absl::UnimplementedError(
      "Multimem not supported on this RCCL version or device");
}

absl::StatusOr<stream_executor::DeviceAddressBase>
RcclSymmetricMemory::peer_addr(RankId peer) const {
#if (TF_ROCM_VERSION >= 100000)
  // ncclGetPeerDevicePointer is available in ROCm/RCCL 10.0+.
  void* peer_addr = nullptr;
  XLA_RCCL_RETURN_IF_ERROR(
      ncclGetPeerDevicePointer(win_, 0, peer.value(), &peer_addr));
  if (peer_addr) {
    return stream_executor::DeviceAddressBase(peer_addr, addr_.size());
  }
  return absl::FailedPreconditionError(absl::StrFormat(
      "Peer rank %d is not load/store accessible from this rank",
      peer.value()));
#else
  return absl::UnimplementedError(
      "Peer address requires ncclGetPeerDevicePointer (ROCm/RCCL >= 10.0)");
#endif  // TF_ROCM_VERSION >= 100000
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
