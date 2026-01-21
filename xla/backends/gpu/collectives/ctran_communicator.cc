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

#include "xla/backends/gpu/collectives/ctran_communicator.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"

// TODO(phambinh): Include actual CTran headers when linked
// #include "comms/ctran/CtranComm.h"
// #include "comms/ctran/mapper/CtranMapper.h"
// #include "comms/ctran/algos/CtranAlgo.h"

namespace xla::gpu {

namespace {

// Helper to create a future that's immediately ready with an error
Future<> ErrorFuture(absl::Status status) {
  auto ref = tsl::MakeConstructedAsyncValueRef<absl::Status>(std::move(status));
  return Future<>(std::move(ref));
}

// Helper to create a future that's immediately ready with success
Future<> OkFuture() {
  auto ref = tsl::MakeAvailableAsyncValueRef<absl::Status>(absl::OkStatus());
  return Future<>(std::move(ref));
}

// Convert XLA ReductionKind to string for logging
const char* ReductionKindToString(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return "SUM";
    case ReductionKind::PRODUCT:
      return "PRODUCT";
    case ReductionKind::MIN:
      return "MIN";
    case ReductionKind::MAX:
      return "MAX";
    default:
      return "UNKNOWN";
  }
}

}  // namespace

// Static factory method
absl::StatusOr<std::unique_ptr<CtranCommunicator>> CtranCommunicator::Create(
    int rank, int num_ranks, int device_ordinal) {
  LOG(INFO) << "CTran: Creating communicator rank=" << rank
            << " num_ranks=" << num_ranks << " device=" << device_ordinal;

  // TODO(phambinh): Initialize CTran here
  //
  // Real implementation would:
  // 1. Create CtranComm with bootstrap
  // 2. Initialize mapper with available backends (NVL, IB, Socket)
  // 3. Create algorithm handler
  //
  // Pseudocode:
  //   auto config = ctranConfig{.blocking = true};
  //   auto ctran_comm = CtranComm::create(bootstrap, rank, num_ranks,
  //                                        device_ordinal, config);
  //   auto mapper = ctran_comm->mapper();
  //   auto algo = std::make_unique<CtranAlgo>(ctran_comm.get());

  return std::unique_ptr<CtranCommunicator>(
      new CtranCommunicator(rank, num_ranks, device_ordinal));
}

CtranCommunicator::CtranCommunicator(int rank, int num_ranks, int device_ordinal)
    : rank_(rank), num_ranks_(num_ranks), device_ordinal_(device_ordinal) {
  LOG(INFO) << "CTran communicator created: " << ToString();
}

CtranCommunicator::~CtranCommunicator() {
  LOG(INFO) << "CTran communicator destroyed: rank=" << rank_;
}

absl::Status CtranCommunicator::Abort() {
  LOG(WARNING) << "CTran: Aborting communicator rank=" << rank_;
  aborted_ = true;

  // TODO(phambinh): Call CTran abort
  // ctran_comm_->abort();

  return absl::OkStatus();
}

absl::Status CtranCommunicator::HealthCheck() const {
  if (aborted_) {
    return absl::FailedPreconditionError("CTran communicator has been aborted");
  }
  // TODO(phambinh): Check CTran health
  return absl::OkStatus();
}

absl::StatusOr<size_t> CtranCommunicator::NumRanks() const {
  return static_cast<size_t>(num_ranks_);
}

absl::Status CtranCommunicator::RegisterBufferOnce(
    se::DeviceMemoryBase buffer_range, int device_ordinal,
    bool use_symmetric_buffer) {
  LOG(INFO) << "CTran: RegisterBuffer ptr=" << buffer_range.opaque()
            << " size=" << buffer_range.size() << " device=" << device_ordinal;

  // TODO(phambinh): Register buffer with CTran for RDMA
  //
  // CTran buffer registration enables:
  // - Zero-copy RDMA over InfiniBand
  // - Direct NVLink access
  // - Efficient TCP DevMem transfers
  //
  // void* handle;
  // ctran_mapper_->regMem(buffer_range.opaque(), buffer_range.size(), &handle);

  return absl::OkStatus();
}

Future<> CtranCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) {
  // Execute the function directly (no grouping yet)
  absl::Status status = f(this);
  if (!status.ok()) {
    return ErrorFuture(status);
  }
  return OkFuture();
}

// Async collective operations

Future<> CtranCommunicator::AllReduce(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       ReductionKind reduction_kind,
                                       const Executor& executor) {
  absl::Status status =
      LaunchAllReduce(send_buffer, recv_buffer, dtype, count, reduction_kind,
                      executor);
  if (!status.ok()) {
    return ErrorFuture(status);
  }
  return OkFuture();
}

Future<> CtranCommunicator::Broadcast(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       RankId root, const Executor& executor) {
  absl::Status status =
      LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root, executor);
  if (!status.ok()) {
    return ErrorFuture(status);
  }
  return OkFuture();
}

Future<> CtranCommunicator::ReduceScatter(se::DeviceMemoryBase send_buffer,
                                           se::DeviceMemoryBase recv_buffer,
                                           PrimitiveType dtype, size_t count,
                                           ReductionKind reduction_kind,
                                           const Executor& executor) {
  absl::Status status = LaunchReduceScatter(send_buffer, recv_buffer, dtype,
                                            count, reduction_kind, executor);
  if (!status.ok()) {
    return ErrorFuture(status);
  }
  return OkFuture();
}

Future<> CtranCommunicator::AllGather(se::DeviceMemoryBase send_buffer,
                                       se::DeviceMemoryBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       const Executor& executor) {
  absl::Status status =
      LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  if (!status.ok()) {
    return ErrorFuture(status);
  }
  return OkFuture();
}

Future<> CtranCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  absl::Status status = LaunchAllToAll(std::move(send_buffers),
                                       std::move(recv_buffers), dtype, count,
                                       executor);
  if (!status.ok()) {
    return ErrorFuture(status);
  }
  return OkFuture();
}

Future<> CtranCommunicator::CollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  absl::Status status =
      LaunchCollectivePermute(send_buffer, recv_buffer, dtype, count,
                              source_rank, target_ranks, executor);
  if (!status.ok()) {
    return ErrorFuture(status);
  }
  return OkFuture();
}

Future<> CtranCommunicator::Send(se::DeviceMemoryBase send_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) {
  absl::Status status = LaunchSend(send_buffer, dtype, count, peer, executor);
  if (!status.ok()) {
    return ErrorFuture(status);
  }
  return OkFuture();
}

Future<> CtranCommunicator::Recv(se::DeviceMemoryBase recv_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) {
  absl::Status status = LaunchRecv(recv_buffer, dtype, count, peer, executor);
  if (!status.ok()) {
    return ErrorFuture(status);
  }
  return OkFuture();
}

// Sync collective operations

absl::Status CtranCommunicator::LaunchAllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  LOG(INFO) << "CTran: AllReduce count=" << count
            << " op=" << ReductionKindToString(reduction_kind)
            << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  // TODO(phambinh): Call CTran AllReduce
  //
  // Real implementation would:
  //   CtranAlgoHandle handle;
  //   auto status = algo_->allReduce(
  //       send_buffer.opaque(),
  //       recv_buffer.opaque(),
  //       count,
  //       ToCtranDataType(dtype),
  //       ToCtranReductionOp(reduction_kind),
  //       stream,
  //       &handle);
  //   // Wait for completion or return async handle

  return absl::UnimplementedError(
      "CTran AllReduce not yet implemented. "
      "Requires linking to torchcomms library.");
}

absl::Status CtranCommunicator::LaunchBroadcast(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  LOG(INFO) << "CTran: Broadcast count=" << count << " root=" << root.value()
            << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  // TODO(phambinh): Call CTran Broadcast
  return absl::UnimplementedError("CTran Broadcast not yet implemented");
}

absl::Status CtranCommunicator::LaunchReduceScatter(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  LOG(INFO) << "CTran: ReduceScatter count=" << count
            << " op=" << ReductionKindToString(reduction_kind)
            << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  // TODO(phambinh): Call CTran ReduceScatter
  return absl::UnimplementedError("CTran ReduceScatter not yet implemented");
}

absl::Status CtranCommunicator::LaunchAllGather(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  LOG(INFO) << "CTran: AllGather count=" << count << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  // TODO(phambinh): Call CTran AllGather
  //
  // CTran AllGather uses the mapper's efficient multi-transport algorithm:
  //   - NVLink for intra-node (fast path)
  //   - InfiniBand for inter-node (RDMA)
  //   - Socket/TCP fallback if others unavailable

  return absl::UnimplementedError("CTran AllGather not yet implemented");
}

absl::Status CtranCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  LOG(INFO) << "CTran: AllToAll count=" << count
            << " num_buffers=" << send_buffers.size() << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  // TODO(phambinh): Call CTran AllToAll
  return absl::UnimplementedError("CTran AllToAll not yet implemented");
}

absl::Status CtranCommunicator::LaunchCollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  LOG(INFO) << "CTran: CollectivePermute count=" << count << " rank=" << rank_
            << " targets=" << target_ranks.size();

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  // TODO(phambinh): Implement using CTran Send/Recv
  return absl::UnimplementedError(
      "CTran CollectivePermute not yet implemented");
}

absl::Status CtranCommunicator::LaunchSend(se::DeviceMemoryBase send_buffer,
                                            PrimitiveType dtype, size_t count,
                                            RankId peer,
                                            const Executor& executor) {
  LOG(INFO) << "CTran: Send count=" << count << " to=" << peer.value()
            << " from=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  // TODO(phambinh): Call CTran isendCtrl via mapper
  //
  // CTran's mapper handles the actual data transfer:
  //   mapper_->isendCtrl(send_buffer.opaque(), count * sizeof_dtype,
  //                      peer.value(), stream, &request);

  return absl::UnimplementedError("CTran Send not yet implemented");
}

absl::Status CtranCommunicator::LaunchRecv(se::DeviceMemoryBase recv_buffer,
                                            PrimitiveType dtype, size_t count,
                                            RankId peer,
                                            const Executor& executor) {
  LOG(INFO) << "CTran: Recv count=" << count << " from=" << peer.value()
            << " to=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  // TODO(phambinh): Call CTran irecvCtrl via mapper
  return absl::UnimplementedError("CTran Recv not yet implemented");
}

std::string CtranCommunicator::ToString() const {
  return absl::StrFormat("CtranCommunicator(rank=%d, num_ranks=%d, device=%d)",
                         rank_, num_ranks_, device_ordinal_);
}

}  // namespace xla::gpu
