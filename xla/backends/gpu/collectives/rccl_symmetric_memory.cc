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

  ncclWindow_t win = nullptr;
#if (TF_ROCM_VERSION >= 71400)
  // ncclCommWindowRegister is available in ROCm/RCCL 7.14+.
  XLA_RCCL_RETURN_IF_ERROR(ncclCommWindowRegister(
      comm, addr.opaque(), addr.size(), &win, NCCL_WIN_COLL_SYMMETRIC));
#end
  return absl::WrapUnique(new RcclSymmetricMemory(comm, win, addr));
}

RcclSymmetricMemory::~RcclSymmetricMemory() {
  VLOG(3) << absl::StrFormat("Destroy %v", *this);
#if (TF_ROCM_VERSION >= 71400)
  // ncclCommWindowDeregister is available in ROCm/RCCL 7.14+.
  XLA_RCCL_LOG_IF_ERROR(ncclCommWindowDeregister(comm_, win_));
#endif
}

stream_executor::DeviceAddressBase RcclSymmetricMemory::addr() const {
  return addr_;
}

absl::StatusOr<stream_executor::DeviceAddressBase>
RcclSymmetricMemory::multimem_addr() const {
#if (TF_ROCM_VERSION >= 71400)
  // ncclGetLsaMultimemDevicePointer is available in ROCm/RCCL 7.14+.
  void* multimem = nullptr;
  XLA_RCCL_RETURN_IF_ERROR(ncclGetLsaMultimemDevicePointer(win_, 0, &multimem));
  if (multimem) {
    return stream_executor::DeviceAddressBase(multimem, addr_.size());
  }
#endif  // TF_ROCM_VERSION >= 71400
  return absl::UnimplementedError(
      "Multimem not supported on this RCCL version or device");
}

absl::StatusOr<stream_executor::DeviceAddressBase>
RcclSymmetricMemory::peer_addr(RankId peer) const {
#if (TF_ROCM_VERSION >= 71400)
  // ncclGetPeerDevicePointer is available in ROCm/RCCL 7.14+.
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
      "Peer address requires ncclGetPeerDevicePointer (ROCm/RCCL >= 7.14)");
#endif  // TF_ROCM_VERSION >= 71400
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
