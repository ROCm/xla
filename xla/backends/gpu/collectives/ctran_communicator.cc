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
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#define gpuStream_t hipStream_t
#else
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#define gpuStream_t cudaStream_t
#endif

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

// Convert XLA PrimitiveType to NCCL data type
ncclDataType_t ToNcclDataType(PrimitiveType dtype) {
  switch (dtype) {
    case S8:
      return ncclInt8;
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
      return ncclFloat32;
    case F64:
      return ncclFloat64;
    case BF16:
      return ncclBfloat16;
    default:
      LOG(FATAL) << "Unsupported data type for NCCL: " << dtype;
  }
}

// Convert XLA ReductionKind to NCCL reduction op
ncclRedOp_t ToNcclReductionOp(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ncclSum;
    case ReductionKind::PRODUCT:
      return ncclProd;
    case ReductionKind::MIN:
      return ncclMin;
    case ReductionKind::MAX:
      return ncclMax;
    default:
      LOG(FATAL) << "Unsupported reduction kind for NCCL";
  }
}

// Get GPU stream from executor
gpuStream_t GetGpuStream(const Communicator::Executor& executor) {
  auto* gpu_executor = static_cast<se::gpu::GpuStream*>(executor.get());
  return reinterpret_cast<gpuStream_t>(gpu_executor->gpu_stream());
}

}  // namespace

// Factory method
absl::StatusOr<std::unique_ptr<CtranCommunicator>> CtranCommunicator::Create(
    int rank, int num_ranks, int device_ordinal,
    ncclUniqueId nccl_id, ncclConfig_t config) {
  LOG(INFO) << "CTran: Creating communicator rank=" << rank
            << " num_ranks=" << num_ranks << " device=" << device_ordinal;

#if TENSORFLOW_USE_ROCM
  hipError_t hip_result = hipSetDevice(device_ordinal);
  if (hip_result != hipSuccess) {
    return absl::InternalError(
        absl::StrFormat("Failed to set HIP device %d: %s", device_ordinal,
                        hipGetErrorString(hip_result)));
  }
#else
  cudaError_t cuda_result = cudaSetDevice(device_ordinal);
  if (cuda_result != cudaSuccess) {
    return absl::InternalError(
        absl::StrFormat("Failed to set CUDA device %d: %s", device_ordinal,
                        cudaGetErrorString(cuda_result)));
  }
#endif

  ncclComm_t comm;
  ncclResult_t result = ncclCommInitRankConfig(&comm, num_ranks, nccl_id, rank, &config);
  if (result != ncclSuccess) {
    return absl::InternalError(
        absl::StrFormat("CTran: Failed to initialize NCCL communicator: %s",
                        ncclGetErrorString(result)));
  }

  LOG(INFO) << "CTran: Successfully initialized communicator for rank " << rank;
  return std::unique_ptr<CtranCommunicator>(
      new CtranCommunicator(comm, rank, num_ranks, device_ordinal));
}

CtranCommunicator::CtranCommunicator(ncclComm_t comm, int rank, int num_ranks,
                                     int device_ordinal)
    : rank_(rank),
      num_ranks_(num_ranks),
      device_ordinal_(device_ordinal),
      comm_(comm) {
  VLOG(1) << "CTran communicator created: " << ToString();
}

CtranCommunicator::~CtranCommunicator() {
  if (comm_ != nullptr && !aborted_) {
    ncclCommDestroy(comm_);
  }
  VLOG(1) << "CTran communicator destroyed: rank=" << rank_;
}

absl::Status CtranCommunicator::Abort() {
  LOG(WARNING) << "CTran: Aborting communicator rank=" << rank_;
  if (!aborted_ && comm_ != nullptr) {
    ncclCommAbort(comm_);
    aborted_ = true;
  }
  return absl::OkStatus();
}

absl::Status CtranCommunicator::HealthCheck() const {
  if (aborted_) {
    return absl::FailedPreconditionError("CTran communicator has been aborted");
  }
  ncclResult_t async_error;
  ncclResult_t result = ncclCommGetAsyncError(comm_, &async_error);
  if (result != ncclSuccess) {
    return absl::InternalError(
        absl::StrFormat("Failed to get async error: %s",
                        ncclGetErrorString(result)));
  }
  if (async_error != ncclSuccess) {
    return absl::InternalError(
        absl::StrFormat("NCCL async error: %s",
                        ncclGetErrorString(async_error)));
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> CtranCommunicator::NumRanks() const {
  return static_cast<size_t>(num_ranks_);
}

absl::Status CtranCommunicator::RegisterBufferOnce(
    se::DeviceMemoryBase buffer_range, int device_ordinal,
    bool use_symmetric_buffer) {
  VLOG(2) << "CTran: RegisterBuffer ptr=" << buffer_range.opaque()
          << " size=" << buffer_range.size() << " device=" << device_ordinal;
  
  // NCCL 2.19+ supports buffer registration for optimized transfers
#if NCCL_VERSION_CODE >= 21900
  void* handle;
  ncclResult_t result = ncclCommRegister(comm_, buffer_range.opaque(),
                                         buffer_range.size(), &handle);
  if (result != ncclSuccess) {
    LOG(WARNING) << "CTran: Buffer registration failed: " 
                 << ncclGetErrorString(result);
  }
#endif
  return absl::OkStatus();
}

Future<> CtranCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) {
  ncclGroupStart();
  absl::Status status = f(this);
  ncclGroupEnd();
  
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
  auto status = LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                                reduction_kind, executor);
  return status.ok() ? OkFuture() : ErrorFuture(status);
}

Future<> CtranCommunicator::Broadcast(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      RankId root, const Executor& executor) {
  auto status = LaunchBroadcast(send_buffer, recv_buffer, dtype, count,
                                root, executor);
  return status.ok() ? OkFuture() : ErrorFuture(status);
}

Future<> CtranCommunicator::ReduceScatter(se::DeviceMemoryBase send_buffer,
                                          se::DeviceMemoryBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          ReductionKind reduction_kind,
                                          const Executor& executor) {
  auto status = LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                                    reduction_kind, executor);
  return status.ok() ? OkFuture() : ErrorFuture(status);
}

Future<> CtranCommunicator::AllGather(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      const Executor& executor) {
  auto status = LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  return status.ok() ? OkFuture() : ErrorFuture(status);
}

Future<> CtranCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  auto status = LaunchAllToAll(std::move(send_buffers), std::move(recv_buffers),
                               dtype, count, executor);
  return status.ok() ? OkFuture() : ErrorFuture(status);
}

Future<> CtranCommunicator::CollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  auto status = LaunchCollectivePermute(send_buffer, recv_buffer, dtype, count,
                                        source_rank, target_ranks, executor);
  return status.ok() ? OkFuture() : ErrorFuture(status);
}

Future<> CtranCommunicator::Send(se::DeviceMemoryBase send_buffer,
                                 PrimitiveType dtype, size_t count,
                                 RankId peer, const Executor& executor) {
  auto status = LaunchSend(send_buffer, dtype, count, peer, executor);
  return status.ok() ? OkFuture() : ErrorFuture(status);
}

Future<> CtranCommunicator::Recv(se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 RankId peer, const Executor& executor) {
  auto status = LaunchRecv(recv_buffer, dtype, count, peer, executor);
  return status.ok() ? OkFuture() : ErrorFuture(status);
}

// Sync collective operations

absl::Status CtranCommunicator::LaunchAllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  VLOG(2) << "CTran: AllReduce count=" << count << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  gpuStream_t stream = GetGpuStream(executor);
  ncclResult_t result = ncclAllReduce(
      send_buffer.opaque(), recv_buffer.opaque(), count,
      ToNcclDataType(dtype), ToNcclReductionOp(reduction_kind),
      comm_, stream);

  XLA_NCCL_RETURN_IF_ERROR(result);
  return absl::OkStatus();
}

absl::Status CtranCommunicator::LaunchBroadcast(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  VLOG(2) << "CTran: Broadcast count=" << count << " root=" << root.value()
          << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  gpuStream_t stream = GetGpuStream(executor);
  ncclResult_t result = ncclBroadcast(
      send_buffer.opaque(), recv_buffer.opaque(), count,
      ToNcclDataType(dtype), root.value(), comm_, stream);

  XLA_NCCL_RETURN_IF_ERROR(result);
  return absl::OkStatus();
}

absl::Status CtranCommunicator::LaunchReduceScatter(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  VLOG(2) << "CTran: ReduceScatter count=" << count << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  gpuStream_t stream = GetGpuStream(executor);
  ncclResult_t result = ncclReduceScatter(
      send_buffer.opaque(), recv_buffer.opaque(), count,
      ToNcclDataType(dtype), ToNcclReductionOp(reduction_kind),
      comm_, stream);

  XLA_NCCL_RETURN_IF_ERROR(result);
  return absl::OkStatus();
}

absl::Status CtranCommunicator::LaunchAllGather(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  VLOG(2) << "CTran: AllGather count=" << count << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  gpuStream_t stream = GetGpuStream(executor);
  ncclResult_t result = ncclAllGather(
      send_buffer.opaque(), recv_buffer.opaque(), count,
      ToNcclDataType(dtype), comm_, stream);

  XLA_NCCL_RETURN_IF_ERROR(result);
  return absl::OkStatus();
}

absl::Status CtranCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  VLOG(2) << "CTran: AllToAll count=" << count 
          << " num_buffers=" << send_buffers.size() << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  // AllToAll is implemented as a series of send/recv pairs
  gpuStream_t stream = GetGpuStream(executor);
  
  ncclGroupStart();
  for (size_t i = 0; i < send_buffers.size(); ++i) {
    ncclSend(send_buffers[i].opaque(), count, ToNcclDataType(dtype),
             static_cast<int>(i), comm_, stream);
    ncclRecv(recv_buffers[i].opaque(), count, ToNcclDataType(dtype),
             static_cast<int>(i), comm_, stream);
  }
  ncclResult_t result = ncclGroupEnd();

  XLA_NCCL_RETURN_IF_ERROR(result);
  return absl::OkStatus();
}

absl::Status CtranCommunicator::LaunchCollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  VLOG(2) << "CTran: CollectivePermute count=" << count << " rank=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  gpuStream_t stream = GetGpuStream(executor);
  
  ncclGroupStart();
  // Send to all targets
  for (const RankId& target : target_ranks) {
    ncclSend(send_buffer.opaque(), count, ToNcclDataType(dtype),
             target.value(), comm_, stream);
  }
  // Receive from source if specified
  if (source_rank.has_value()) {
    ncclRecv(recv_buffer.opaque(), count, ToNcclDataType(dtype),
             source_rank->value(), comm_, stream);
  }
  ncclResult_t result = ncclGroupEnd();

  XLA_NCCL_RETURN_IF_ERROR(result);
  return absl::OkStatus();
}

absl::Status CtranCommunicator::LaunchSend(
    se::DeviceMemoryBase send_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  VLOG(2) << "CTran: Send count=" << count << " to=" << peer.value()
          << " from=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  gpuStream_t stream = GetGpuStream(executor);
  ncclResult_t result = ncclSend(
      send_buffer.opaque(), count, ToNcclDataType(dtype),
      peer.value(), comm_, stream);

  XLA_NCCL_RETURN_IF_ERROR(result);
  return absl::OkStatus();
}

absl::Status CtranCommunicator::LaunchRecv(
    se::DeviceMemoryBase recv_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  VLOG(2) << "CTran: Recv count=" << count << " from=" << peer.value()
          << " to=" << rank_;

  if (aborted_) {
    return absl::FailedPreconditionError("Communicator has been aborted");
  }

  gpuStream_t stream = GetGpuStream(executor);
  ncclResult_t result = ncclRecv(
      recv_buffer.opaque(), count, ToNcclDataType(dtype),
      peer.value(), comm_, stream);

  XLA_NCCL_RETURN_IF_ERROR(result);
  return absl::OkStatus();
}

std::string CtranCommunicator::ToString() const {
  return absl::StrFormat("CtranCommunicator(rank=%d, num_ranks=%d, device=%d)",
                         rank_, num_ranks_, device_ordinal_);
}

}  // namespace xla::gpu
