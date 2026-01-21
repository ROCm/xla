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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_CTRAN_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_CTRAN_COMMUNICATOR_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"

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

// CTran-based GPU communicator implementation.
//
// CTran (Communication Transport) provides enhanced collective communication
// using NCCL/RCCL as the underlying transport, with optional RCCLX features:
// - Multi-transport support (NVLink, InfiniBand, TCP)
// - Intelligent transport selection based on topology
// - Fault tolerance and automatic failover
// - Zero-copy RDMA transfers
//
// This communicator wraps NCCL/RCCL (or RCCLX when available) to provide
// the XLA Communicator interface for GPU collective operations.
class CtranCommunicator : public GpuCommunicator {
 public:
  // Factory method to create a CTran communicator
  static absl::StatusOr<std::unique_ptr<CtranCommunicator>> Create(
      int rank, int num_ranks, int device_ordinal,
      ncclUniqueId nccl_id, ncclConfig_t config);

  ~CtranCommunicator() override;

  // Prevent copying
  CtranCommunicator(const CtranCommunicator&) = delete;
  CtranCommunicator& operator=(const CtranCommunicator&) = delete;

  // Get the underlying NCCL communicator
  ncclComm_t comm() const { return comm_; }

  // Communicator interface
  absl::Status Abort() final;
  absl::Status HealthCheck() const final;
  absl::StatusOr<size_t> NumRanks() const final;

  // Buffer registration for RDMA
  absl::Status RegisterBufferOnce(se::DeviceMemoryBase buffer_range,
                                  int device_ordinal,
                                  bool use_symmetric_buffer) final;

  // Group execution for batching operations
  Future<> GroupExecute(
      absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) final;

  // Collective operations (async interface)
  Future<> AllReduce(se::DeviceMemoryBase send_buffer,
                     se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                     size_t count, ReductionKind reduction_kind,
                     const Executor& executor) final;

  Future<> Broadcast(se::DeviceMemoryBase send_buffer,
                     se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                     size_t count, RankId root, const Executor& executor) final;

  Future<> ReduceScatter(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) final;

  Future<> AllGather(se::DeviceMemoryBase send_buffer,
                     se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                     size_t count, const Executor& executor) final;

  Future<> AllToAll(absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
                    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
                    PrimitiveType dtype, size_t count,
                    const Executor& executor) final;

  Future<> CollectivePermute(se::DeviceMemoryBase send_buffer,
                             se::DeviceMemoryBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             std::optional<RankId> source_rank,
                             absl::Span<const RankId> target_ranks,
                             const Executor& executor) final;

  Future<> Send(se::DeviceMemoryBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  Future<> Recv(se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  // Collective operations (sync interface for GpuCommunicator)
  absl::Status LaunchAllReduce(se::DeviceMemoryBase send_buffer,
                               se::DeviceMemoryBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               const Executor& executor) final;

  absl::Status LaunchBroadcast(se::DeviceMemoryBase send_buffer,
                               se::DeviceMemoryBase recv_buffer,
                               PrimitiveType dtype, size_t count, RankId root,
                               const Executor& executor) final;

  absl::Status LaunchReduceScatter(se::DeviceMemoryBase send_buffer,
                                   se::DeviceMemoryBase recv_buffer,
                                   PrimitiveType dtype, size_t count,
                                   ReductionKind reduction_kind,
                                   const Executor& executor) final;

  absl::Status LaunchAllGather(se::DeviceMemoryBase send_buffer,
                               se::DeviceMemoryBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               const Executor& executor) final;

  absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) final;

  absl::Status LaunchCollectivePermute(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       std::optional<RankId> source_rank,
                                       absl::Span<const RankId> target_ranks,
                                       const Executor& executor) final;

  absl::Status LaunchSend(se::DeviceMemoryBase send_buffer, PrimitiveType dtype,
                          size_t count, RankId peer,
                          const Executor& executor) final;

  absl::Status LaunchRecv(se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                          size_t count, RankId peer,
                          const Executor& executor) final;

  std::string ToString() const final;

 private:
  CtranCommunicator(ncclComm_t comm, int rank, int num_ranks, int device_ordinal);

  // Internal state
  int rank_;
  int num_ranks_;
  int device_ordinal_;
  ncclComm_t comm_;
  bool aborted_ = false;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_CTRAN_COMMUNICATOR_H_
