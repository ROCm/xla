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

// TODO(phambinh): Include actual CTran headers when linked
// #include "comms/ctran/CtranComm.h"
// #include "comms/ctran/mapper/CtranMapper.h"
// #include "comms/ctran/algos/CtranAlgo.h"

namespace xla::gpu {

// XLA collectives communicator wrapping a CTran communicator.
//
// This communicator maps XLA's collective operations to CTran's mapper and
// algorithm APIs. CTran provides several advantages over NCCL/RCCL:
//
// 1. Multi-transport: Automatically selects NVLink, InfiniBand, or TCP
// 2. Fault tolerance: Graceful handling of network failures
// 3. Optimized algorithms: Hand-tuned collective implementations
//
// Mapping of XLA operations to CTran:
//   XLA AllReduce     -> CtranAlgo::allReduce()
//   XLA AllGather     -> CtranAlgo::allGather()
//   XLA ReduceScatter -> CtranAlgo::reduceScatter()
//   XLA AllToAll      -> CtranAlgo::allToAll()
//   XLA Broadcast     -> CtranAlgo::broadcast()
//   XLA Send/Recv     -> CtranMapper::isendCtrl()/irecvCtrl()
class CtranCommunicator : public GpuCommunicator {
 public:
  // Factory method to create a CTran communicator
  //
  // TODO(phambinh): Full implementation requires:
  // - CtranComm pointer from CTran library
  // - Rank and size information
  // - CUDA/HIP stream for async execution
  static absl::StatusOr<std::unique_ptr<CtranCommunicator>> Create(
      int rank, int num_ranks, int device_ordinal);

  ~CtranCommunicator() override;

  // Prevent copying
  CtranCommunicator(const CtranCommunicator&) = delete;
  CtranCommunicator& operator=(const CtranCommunicator&) = delete;

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
  CtranCommunicator(int rank, int num_ranks, int device_ordinal);

  // Internal state
  int rank_;
  int num_ranks_;
  int device_ordinal_;
  bool aborted_ = false;

  // TODO(phambinh): Add CTran-specific state when linked
  // std::unique_ptr<CtranComm> ctran_comm_;
  // std::shared_ptr<CtranMapper> mapper_;
  // std::unique_ptr<CtranAlgo> algo_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_CTRAN_COMMUNICATOR_H_
