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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_CTRAN_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_CTRAN_COLLECTIVES_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"

namespace xla::gpu {

// XLA host-initiated collectives implemented on top of Meta's CTran transport.
//
// CTran (Communication Transport) is Meta's multi-transport communication layer
// that provides:
//   - Multi-transport support: NVLink, InfiniBand/RoCE, TCP/Socket, TCP DevMem
//   - Intelligent routing: automatic transport selection based on topology
//   - Fault tolerance: graceful handling of network failures
//   - Optimized algorithms: AllReduce, AllGather, ReduceScatter, AllToAll
//
// This implementation serves as the XLA integration layer for CTran, mapping
// XLA's collective operations to CTran's mapper and algorithm APIs.
//
// Integration Architecture:
//   XLA HLO Collective Ops
//         |
//   CtranCollectives (this class)
//         |
//   CtranCommunicator (wraps CTran mapper)
//         |
//   CTran Mapper/Algorithms
//         |
//   CTran Backends (NVL, IB, Socket, TcpDm)
//
// Requirements:
//   - torchcomms library must be installed
//   - CTran must be compiled with ROCm/CUDA support
//   - Required environment variables for CTran configuration
//
// TODO(phambinh): Full CTran integration requires linking to torchcomms library
// and implementing the actual CTran API calls.
class CtranCollectives : public GpuCollectives {
 public:
  CtranCollectives();
  ~CtranCollectives() override;

  // Returns true - CTran is implemented (when linked properly)
  bool IsImplemented() const final;

  // Returns false - CTran uses per-communicator config, not global
  bool IsGlobalConfig() const final;

  // Get clique ID callback for CTran
  absl::StatusOr<const CliqueIdCallback*> GetCliqueIdCallback(
      const CliqueIdCallback* clique_id_callback, bool is_local) final;

  // Create a unique clique ID for communicator initialization
  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const final;

  // Create communicators for the given clique
  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Collectives::Config& config) final;

  // Create a single communicator (not yet implemented)
  absl::StatusOr<std::unique_ptr<Communicator>> CreateCommunicator() final;

  // Split existing communicators
  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> SplitCommunicators(
      absl::Span<const Communicator* const> comms, int32_t color,
      absl::Span<const RankId> keys, const Collectives::Config& config) final;

  // Allocate collective memory (uses CTran's memory registration)
  absl::StatusOr<void*> Allocate(uint64_t bytes) final;

  // Deallocate collective memory
  absl::Status Deallocate(void* location) final;

  // Initialize topology information for CTran
  absl::Status InitializeTopology(Topology topology) final;

  // Check if CTran library is available at runtime
  static bool IsCtranAvailable();

  // Get CTran version string
  static std::string GetCtranVersion();

 private:
  // Internal state for CTran initialization
  struct CtranState;
  std::unique_ptr<CtranState> state_;

  // Clique ID callback for local mode
  CliqueIdCallback local_clique_id_callback_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_CTRAN_COLLECTIVES_H_
