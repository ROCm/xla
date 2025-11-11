/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/profiler/gpu/distributed_timestamp_sync.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/env.h"
#include "xla/backends/profiler/gpu/network_probe.h"

namespace xla::profiler {

// ============================================================================
// DistributedProfilerContextManager - Singleton Implementation
// ============================================================================

DistributedProfilerContextManager& DistributedProfilerContextManager::Get() {
  static DistributedProfilerContextManager instance;
  return instance;
}

void DistributedProfilerContextManager::SetDistributedContext(
    const DistributedProfilerContext& ctx) {
  absl::MutexLock lock(&mu_);
  
  if (context_set_) {
    LOG(WARNING) << "Distributed profiler context already set. "
                 << "Ignoring new context. "
                 << "(node_id=" << ctx.node_id << ", num_nodes=" << ctx.num_nodes << ")";
    return;
  }
  
  context_ = ctx;
  context_set_ = true;
  
  LOG(INFO) << "Distributed profiler context initialized: "
            << "node_id=" << ctx.node_id << ", num_nodes=" << ctx.num_nodes
            << ", addresses=" << ctx.node_addresses.size()
            << ", socket_ts=" << (ctx.enable_socket_timestamping ? "enabled" : "disabled");
}

std::optional<DistributedProfilerContext> 
DistributedProfilerContextManager::GetDistributedContext() const {
  absl::MutexLock lock(&mu_);
  return context_;
}

bool DistributedProfilerContextManager::HasDistributedContext() const {
  absl::MutexLock lock(&mu_);
  return context_set_;
}

void DistributedProfilerContextManager::ResetContext() {
  absl::MutexLock lock(&mu_);
  context_.reset();
  context_set_ = false;
  LOG(INFO) << "Distributed profiler context reset";
}

// ============================================================================
// DistributedTimestampSynchronizer Implementation
// ============================================================================

DistributedTimestampSynchronizer::DistributedTimestampSynchronizer(
    const DistributedProfilerContext& config)
    : config_(config) {}

DistributedTimestampSynchronizer::~DistributedTimestampSynchronizer() = default;

absl::Status DistributedTimestampSynchronizer::Initialize() {
  LOG(INFO) << "Initializing DistributedTimestampSynchronizer for node "
            << config_.node_id << " of " << config_.num_nodes;
  
  // Single-node case: no sync needed
  if (config_.num_nodes <= 1) {
    LOG(INFO) << "Single-node setup, skipping timestamp synchronization";
    return absl::OkStatus();
  }
  
  // Multi-node: perform initial sync
  TF_RETURN_IF_ERROR(SyncTimestamps());

  // NEW: Initialize probe manager if enabled
  if (config_.probe_cadence_us > 0) {
    LOG(INFO) << "Enabling network probing with cadence " << config_.probe_cadence_us << " us";
    probe_manager_ = std::make_unique<NetworkProbeManager>(config_);
    TF_RETURN_IF_ERROR(probe_manager_->Initialize());
  } else {
    LOG(INFO) << "Network probing disabled";
  }
  
  return absl::OkStatus();
}

absl::Status DistributedTimestampSynchronizer::StartProbing() {
  if (!probe_manager_) {
    VLOG(1) << "Probing not enabled, skipping StartProbing";
    return absl::OkStatus();
  }
  return probe_manager_->Start();
}

int64_t DistributedTimestampSynchronizer::GetClockOffset() const {
  return clock_offset_ns_;
}

uint64_t DistributedTimestampSynchronizer::LocalToGlobal(
    uint64_t local_ns) const {
  return local_ns + clock_offset_ns_;
}

std::vector<SyncedTimestamp> 
DistributedTimestampSynchronizer::GetLastSyncTimestamps() const {
  return last_sync_timestamps_;
}

absl::Status DistributedTimestampSynchronizer::CreateTimestampSocket() {
  // TODO: Implement socket creation with SOF_TIMESTAMPING_RX_SOFTWARE
  // This is a placeholder for the actual implementation
  LOG(INFO) << "Creating timestamp socket for node " << config_.node_id;
  return absl::OkStatus();
}

absl::Status DistributedTimestampSynchronizer::ExchangeTimestampWithNode(
    int remote_node_id, int socket_fd) {
  // TODO: Implement actual timestamp exchange protocol
  LOG(INFO) << "Exchanging timestamps with node " << remote_node_id;
  return absl::OkStatus();
}

absl::Status DistributedTimestampSynchronizer::SyncTimestamps() {
  if (config_.num_nodes <= 1) {
    return absl::OkStatus();
  }
  
  LOG(INFO) << "Starting timestamp synchronization for node " 
            << config_.node_id << "...";
  
  // TODO: Implement full timestamp exchange protocol
  // For now, this is a placeholder that shows the structure:
  
  // 1. Create listening socket for this node
  // TF_ASSIGN_OR_RETURN(int listen_sock, CreateTimestampSocket());
  
  // 2. Connect to all other nodes and exchange timestamps
  // for (int i = 0; i < config_.num_nodes; ++i) {
  //   if (i != config_.node_id) {
  //     TF_RETURN_IF_ERROR(ExchangeTimestampWithNode(i, listen_sock));
  //   }
  // }
  
  // 3. Compute clock offset based on exchanged timestamps
  // clock_offset_ns_ = /* computed from exchanges */;
  
  // NEW: If probe manager active, perform probe-based sync
  if (probe_manager_) {
    TF_RETURN_IF_ERROR(probe_manager_->Sync());
  }
  
  LOG(INFO) << "Timestamp synchronization complete. "
            << "Clock offset: " << clock_offset_ns_ << " ns";
  
  return absl::OkStatus();
}

absl::Status DistributedTimestampSynchronizer::ExportProbeData() {
  if (!probe_manager_) {
    VLOG(1) << "Probe manager not initialized, skipping probe data export";
    return absl::OkStatus();
  }
  
  return probe_manager_->ExportData();
}

}  // namespace xla::profiler
