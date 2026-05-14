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

#ifndef XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_CLIENT_OPTIONS_H_
#define XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_CLIENT_OPTIONS_H_

#include <memory>
#include <optional>
#include <set>
#include <string>

#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"

namespace xla {

// Options for creating a XLA:GPU PjRtClient.
struct GpuClientOptions {
  GpuAllocatorConfig allocator_config;

  int node_id = 0;

  int num_nodes = 1;

  std::optional<std::set<int>> allowed_devices = std::nullopt;

  std::optional<std::string> platform_name = std::nullopt;

  bool should_stage_host_to_device_transfers = true;

  // Optional factory for a host memory allocator to use for transfer. Used only
  // if `should_stage_host_to_device_transfers` is true.
  HostMemoryAllocator::Factory host_memory_allocator_factory;

  // kv_store must be non-null if num_nodes > 1.
  std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;

  bool abort_collectives_on_failure = false;

  bool enable_mock_nccl = false;

  std::optional<std::string> mock_gpu_topology;

  std::optional<int> partition_index;

  bool use_tfrt_gpu_client = false;

  std::optional<bool> use_async_dispatch;

  std::optional<int> max_inflight_computations = 32;

  // When true, GPU streams (hipStream_t / cudaStream_t) are not created during
  // client construction but are instead created on demand the first time they
  // are accessed. This eliminates the dominant client-construction cost on
  // platforms like ROCm where hipStreamCreate is serialized at the KFD driver
  // level (~3-4 ms per stream × 14–18 streams × N GPUs).
  //
  // Enabled by default. Set to false (or export PJRT_GPU_LAZY_STREAM_CREATION=0)
  // to restore the original eager-creation behaviour if needed for debugging.
  //
  // Trade-off: the one-time stream-creation latency is moved from client
  // construction to the first use of each stream type (compute, h2d, d2h,
  // d2d, etc.). In typical workloads all streams are created on the first
  // compilation/execution, so the total wall-clock time is unchanged;
  // only the timing of when it is paid differs.
  bool lazy_stream_creation = true;
};

}  //  namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_CLIENT_OPTIONS_H_
