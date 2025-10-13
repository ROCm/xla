/* Copyright 2025 The OpenXLA Authors.
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

#include "xla/backends/gpu/collectives/rocshmem_communicator.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/rocshmem_collectives.h"
#include "xla/backends/gpu/collectives/rocshmem_kernels.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

using namespace rocshmem;

namespace xla::gpu {

RocshmemCommunicator::~RocshmemCommunicator() {
  
  if (teams_ != nullptr) {
    for (uint32_t i = 0; i < kMaxTeams; i++) {
      rocshmem_team_destroy(teams_[i]);
    }
  }
  collectives_->Deallocate(teams_).IgnoreError();
}

#define CHECK_ABORTED() \
  if (aborted_) return FailedPrecondition("RocshmemCommunicator aborted");

absl::Status RocshmemCommunicator::Abort() {
  VLOG(1) << "Abort ROCSHMEM communicator: " << ToString();
  CHECK_ABORTED()
  aborted_ = true;
  // Call rocshmem_global_exit with a non-zero return code to abort the program.
  rocshmem_global_exit(1);
  return absl::OkStatus();
}

absl::Status RocshmemCommunicator::Barrier(
    const Communicator::Executor& executor) {
  VLOG(1) << "Barrier ROCSHMEM communicator: " << ToString();
  CHECK_ABORTED()

  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  auto gpu_stream = se::gpu::AsGpuStreamValue(stream);
  rocshmem_barrier_all_on_stream(/*host_team_,*/ gpu_stream);
  return absl::OkStatus();
}
absl::StatusOr<size_t> RocshmemCommunicator::NumRanks() const {
  VLOG(5) << "Get the number of ranks in ROCSHMEM communicator: " << ToString();
  CHECK_ABORTED()

  int32_t count = 0;
  count = rocshmem_team_n_pes(host_team_);
  if (count < 0) {
    return absl::InvalidArgumentError(
        "RocshmemCommunicator::NumRanks invalid team.");
  }
  return count;
}

absl::StatusOr<size_t> RocshmemCommunicator::CurrentRank() {
  VLOG(5) << "Get current rank in ROCSHMEM communicator: " << ToString();
  CHECK_ABORTED()

  int32_t rank = 0;
  rank = rocshmem_team_my_pe(host_team_);
  if (rank < 0) {
    return absl::InvalidArgumentError(
        "RocshmemCommunicator::NumRanks invalid team.");
  }
  return rank;
}

std::string RocshmemCommunicator::ToString() const {
  return absl::StrFormat("RocshmemCommunicator(rocshmem_team_t=%p)",
                         host_team_);
}

absl::StatusOr<se::Stream*> RocshmemCommunicator::ToStream(
    const Executor& executor) {
  if (auto* gpu_executor =
          tsl::down_cast<const GpuCollectives::Executor*>(&executor)) {
    return gpu_executor->stream();
  }
  return InvalidArgument("Communicator executor is not a GPU executor");
}

Future<> RocshmemCommunicator::AllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {

  CHECK_ABORTED()

  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  auto gpu_stream = se::gpu::AsGpuStreamValue(stream);

  void* source_ptr = send_buffer.opaque();
  void* dest_ptr = recv_buffer.opaque();
  if (primitive_util::IsComplexType(dtype)) count *= 2;

  VLOG(3) << absl::StreamFormat(
      "Launch ROCSHMEM AllReduce operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%s; comm=node; "
      "team=%p;"
      "stream=%p",
      rocshmem_team_my_pe(host_team_), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, ReductionKindToString(reduction_kind), host_team_, stream);

  auto call = [&](auto T) -> absl::Status {
    using Type = decltype(T);
    auto *dest = static_cast< Type *>(dest_ptr);
    const auto *source = static_cast< const Type *>(source_ptr);
    return allreduce_on_stream< Type >(
          teams_, kMaxTeams, dest, source, count, reduction_kind, gpu_stream);
  };

  switch(dtype) {
  case PrimitiveType::F64: return call(double{});
  case PrimitiveType::F32: return call(float{});
  case PrimitiveType::S64: return call(longlong{});
  case PrimitiveType::S32: return call(int{});
  case PrimitiveType::S16: return call(short{});
  }
  return absl::InternalError("Invalid ROCSHMEM reduction type.");
}

// Performs point-to-point communication between two ranks using ROCSHMEM.
// This is a helper function used by both Send and Recv operations to handle
// the actual data transfer between peers.
absl::Status RocshmemCommunicator::P2P(P2PType p2p_type,
                                      PrimitiveType dtype,
                                      se::DeviceMemoryBase recv_buffer,
                                      se::DeviceMemoryBase send_buffer,
                                      size_t count, RankId peer,
                                      const Executor& executor) {
  
  VLOG(1) << (p2p_type == P2PType::Send ? "Send" : "Recv") 
          << " ROCSHMEM communicator: " << ToString();
  CHECK_ABORTED()

  if (primitive_util::IsComplexType(dtype)) count *= 2;
  void* source_ptr = send_buffer.opaque();
  void* dest_ptr = recv_buffer.opaque();

  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  auto gpu_stream = se::gpu::AsGpuStreamValue(stream);

  auto call = [&](auto T) -> absl::Status {
    using Type = decltype(T);
    auto *dest = static_cast< Type *>(dest_ptr);
    const auto *source = static_cast< const Type *>(source_ptr);
    if (p2p_type == P2PType::Send) {
      return put_nbi_on_stream(peer.value(), dest, source, count, gpu_stream);
    } else {
      return get_nbi_on_stream(peer.value(), dest, source, count, gpu_stream);
    }
  };

  switch(dtype) {
  case PrimitiveType::F64: return call(double{});
  case PrimitiveType::F32: return call(float{});
  case PrimitiveType::S64: 
  case PrimitiveType::U64: 
    return call(longlong{});
  case PrimitiveType::S32: 
  case PrimitiveType::U32:
    return call(int{});
  case PrimitiveType::S16: 
  case PrimitiveType::BF16: 
  case PrimitiveType::F16: 
    return call(short{});
  case PrimitiveType::S8:
  case PrimitiveType::U8:
    return call(char{});
  }
  return absl::InternalError(
          absl::StrFormat("Invalid ROCSHMEM %s type.", 
            p2p_type == P2PType::Recv ? "recv" : "send"));
}

Future<> RocshmemCommunicator::Send(se::DeviceMemoryBase recv_buffer,
                                   se::DeviceMemoryBase send_buffer,
                                   PrimitiveType dtype, size_t count,
                                   RankId peer, const Executor& executor) {
  return P2P(P2PType::Send, dtype, recv_buffer, send_buffer, count, 
                                                              peer, executor);
}

Future<> RocshmemCommunicator::Recv(se::DeviceMemoryBase recv_buffer,
                                   se::DeviceMemoryBase send_buffer,
                                   PrimitiveType dtype, size_t count,
                                   RankId peer, const Executor& executor) {
  return P2P(P2PType::Recv, dtype, recv_buffer, send_buffer, count, 
                                                              peer, executor);
}

absl::Status RocshmemCommunicator::Quiet(const Executor& executor) {
  VLOG(1) << "Quiet ROCSHMEM communicator: " << ToString();
  CHECK_ABORTED()
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  rocshmem_quiet_on_stream(se::gpu::AsGpuStreamValue(stream));
  return absl::OkStatus();
}

absl::Status RocshmemCommunicator::Fence() {
  VLOG(1) << "Fence ROCSHMEM communicator: " << ToString();
  CHECK_ABORTED()
  rocshmem_fence();
  return absl::OkStatus();
}

}  // namespace xla::gpu
