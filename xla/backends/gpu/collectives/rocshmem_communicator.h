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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_ROCSHMEM_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_ROCSHMEM_COMMUNICATOR_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

#include "third_party/rocshmem/rocshmem.hpp"
#include "third_party/rocshmem/rocshmem_COLL.hpp"

namespace xla::gpu {

class RocshmemCollectives;

// XLA collectives communicator wrapping an ROCSHMEM communicator.
class RocshmemCommunicator : public Communicator {
 public:
  constexpr static uint32_t kMaxTeams = 24;

  friend class RocshmemCollectives;
  ~RocshmemCommunicator() override;

  // RocshmemCommunicator is not copyable or movable.
  RocshmemCommunicator(const RocshmemCommunicator&) = delete;
  RocshmemCommunicator(RocshmemCommunicator&&) = delete;
  RocshmemCommunicator& operator=(const RocshmemCommunicator&) = delete;
  RocshmemCommunicator& operator=(RocshmemCommunicator&&) = delete;

  absl::Status Abort() final;
  absl::StatusOr<size_t> NumRanks() const final;
  absl::StatusOr<size_t> CurrentRank() final;

  absl::Status Barrier(const Executor& executor) final;

  Future<> AllReduce(se::DeviceMemoryBase send_buffer,
                     se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                     size_t count, ReductionKind reduction_kind,
                     const Executor& executor) final;

  Future<> Broadcast(se::DeviceMemoryBase send_buffer,
                     se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                     size_t count, RankId root,
                     const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  Future<> ReduceScatter(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  Future<> AllGather(se::DeviceMemoryBase send_buffer,
                     se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                     size_t count, const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  Future<> AllToAll(absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
                    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
                    PrimitiveType dtype, size_t count,
                    const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  Future<> CollectivePermute(se::DeviceMemoryBase send_buffer,
                             se::DeviceMemoryBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             std::optional<RankId> source_rank,
                             absl::Span<const RankId> target_ranks,
                             const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  Future<> Send(se::DeviceMemoryBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  Future<> Recv(se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final {
    return absl::UnimplementedError("Not implemented.");
  };

  Future<> Send(se::DeviceMemoryBase recv_buffer,
                se::DeviceMemoryBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  Future<> Recv(se::DeviceMemoryBase recv_buffer,
                se::DeviceMemoryBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  absl::Status Quiet(const Executor& executor) final;

  absl::Status Fence() final;

  std::string ToString() const final;

 private:
  RocshmemCommunicator(RocshmemCollectives* coll, 
      rocshmem::rocshmem_team_t *teams) : collectives_(coll), teams_(teams) { }

  enum class P2PType : int32_t {
    Send,
    Recv
  };

  absl::Status P2P(P2PType p2p_type, PrimitiveType type,
                   se::DeviceMemoryBase recv_buffer,
                   se::DeviceMemoryBase send_buffer, size_t count, RankId peer,
                   const Executor& executor);

  static absl::StatusOr<se::Stream*> ToStream(const Executor& executor);

  RocshmemCollectives* collectives_;  // Parent RocshmemCollectives instance
  bool aborted_ = false;             // Has Abort() been called?
  rocshmem::rocshmem_team_t *teams_ = nullptr;
  rocshmem::rocshmem_team_t host_team_ = rocshmem::ROCSHMEM_TEAM_WORLD;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_ROCSHMEM_COMMUNICATOR_H_
