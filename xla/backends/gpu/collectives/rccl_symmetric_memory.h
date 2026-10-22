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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_

#include <cstddef>  // for size_t in the RCCL 10.0+ forward declarations below
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "rocm/rocm_config.h"  // IWYU pragma: keep  (defines TF_ROCM_VERSION)
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/executor.h"

#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200

#if (TF_ROCM_VERSION >= 100000)
// ROCm/RCCL 10.0+ exposes the host-side symmetric-memory pointer helpers
// (ncclGetLsaMultimemDevicePointer, ncclGetPeerDevicePointer) as
// `NCCL_EXTERN_C __host__` symbols exported from librccl.so.
//
// These are declared in RCCL's nccl_device/core.h, but that header is NOT
// shipped by the ROCm install -- the installed RCCL dev package only provides
// rocm/include/rccl/rccl.h (nccl_device/ lives solely in the RCCL source
// tree).
//
// PROVENANCE: copied verbatim from RCCL hip-version_10.2.62640,
// nccl_device/core.h.
extern "C" {
ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset,
                                             void** outPtr);
ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset,
                                      int peer, void** outPtr);
}  // extern "C"
#endif  // TF_ROCM_VERSION >= 100000

namespace xla::gpu {

// An RCCL window registration handle that makes local buffers accessible from
// remote peers via symmetric memory registration. Analogous to
// NcclSymmetricMemory for the ROCm/RCCL platform.
class RcclSymmetricMemory final : public SymmetricMemory {
 public:
  ~RcclSymmetricMemory() final;

  // `executor` is the communicator's single-threaded executor (may be null for
  // blocking/synchronous communicators). It is used to serialize the
  // window-lifecycle operations that mutate communicator state - symmetric
  // window registration (in Create) and deregistration (in the destructor) -
  // onto the communicator's owning thread.
  //
  // The read-only device-pointer lookups (multimem_addr()/peer_addr(), i.e.
  // ncclGet{LsaMultimem,Peer}DevicePointer) are NOT dispatched onto `executor`:
  // they only read already-populated window/comm metadata and do not touch
  // RCCL's per-thread enqueue/group state.
  static absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>> Create(
      ncclComm_t comm, stream_executor::DeviceAddressBase addr,
      std::shared_ptr<tsl::Executor> executor);

  stream_executor::DeviceAddressBase addr() const final;

  // For RCCL versions that expose ncclGetLsaMultimemDevicePointer (ROCm 10.0+)
  // returns the multimem address for the LSA (load/store accessible) team
  // associated with this symmetric memory.
  absl::StatusOr<stream_executor::DeviceAddressBase> multimem_addr()
      const final;

  // For RCCL versions that expose ncclGetPeerDevicePointer (ROCm 10.0+)
  // returns the address of the symmetric memory on a peer device.
  absl::StatusOr<stream_executor::DeviceAddressBase> peer_addr(
      RankId peer) const final;

  ncclWindow_t win() const { return win_; }

  std::string ToString() const final;

  PackedKernelArg PackKernelArg() const final;

 private:
  RcclSymmetricMemory(ncclComm_t comm, ncclWindow_t win,
                      stream_executor::DeviceAddressBase addr,
                      std::shared_ptr<tsl::Executor> executor);

  ncclComm_t comm_;
  ncclWindow_t win_;
  stream_executor::DeviceAddressBase addr_;
  std::shared_ptr<tsl::Executor> executor_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_
