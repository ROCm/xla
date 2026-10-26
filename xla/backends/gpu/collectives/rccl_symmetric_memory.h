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

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "rocm/include/rccl/rccl.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla::gpu {

// An RCCL window registration handle that makes local buffers accessible from
// remote peers via symmetric memory registration. Analogous to
// NcclSymmetricMemory for the ROCm/RCCL platform.
class RcclSymmetricMemory final : public SymmetricMemory {
 public:
  ~RcclSymmetricMemory() final;

  // Registers `addr` with NCCL_WIN_COLL_SYMMETRIC if `symmetric_window`, else
  // with NCCL_WIN_DEFAULT (see NcclSymmetricMemory::Create). Must be called on
  // `executor` if it is not null, which then also runs the deregistration; on a
  // non-blocking communicator, `cancel` (if not null) aborts the polling. Both
  // are owned by the communicator and must outlive the returned object.
  static absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>> Create(
      ncclComm_t comm, stream_executor::DeviceAddressBase addr,
      bool symmetric_window = false, tsl::Executor* executor = nullptr,
      const CancellationToken* cancel = nullptr);

  stream_executor::DeviceAddressBase addr() const final;

  // multimem_addr() and peer_addr() are not supported by RCCL; the base-class
  // defaults (returning Unimplemented) are used.

  ncclWindow_t win() const { return win_; }

  std::string ToString() const final;

  PackedKernelArg PackKernelArg() const final;

 private:
  RcclSymmetricMemory(ncclComm_t comm, ncclWindow_t win,
                      stream_executor::DeviceAddressBase addr,
                      tsl::Executor* executor, const CancellationToken* cancel);

  ncclComm_t comm_;
  ncclWindow_t win_;
  stream_executor::DeviceAddressBase addr_;
  tsl::Executor* executor_;
  const CancellationToken* cancel_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_
