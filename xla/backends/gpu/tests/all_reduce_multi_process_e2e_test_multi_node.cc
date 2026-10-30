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

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tsl/platform/path.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

inline constexpr size_t kMB = 1024LL * 1024LL;

// Default single-host configuration used when the test is driven by `bazel
// test` without extra flags: 2 ranks, 1 rank per node => a 2-GPU single-host
// run that reproduces the original behavior.
constexpr int kDefaultNumNodes = 2;
constexpr int kDefaultRanksPerNode = 1;

static const char* test_binary_name;

// How the ranks are laid out across interconnect (NVLink / xGMI) slices.
//
// Both layouts give every rank its own process (num_devices_per_process == 1),
// so the collective is never single-host: it always goes through the cross-host
// symmetric-memory path. They differ in whether the replica group also crosses
// a slice boundary, which the symmetric-memory backend must handle too.
enum class SliceMode {
  // All hosts belong to a single slice (e.g. a GB200 NVL rack).
  kSingleSlice,
  // Every host is its own slice, so the replica group spans several slices.
  kMultiSlice,
};

absl::string_view SliceModeName(SliceMode slice_mode) {
  return slice_mode == SliceMode::kMultiSlice ? "multi_slice" : "single_slice";
}

absl::StatusOr<SliceMode> ParseSliceMode(absl::string_view name) {
  if (name.empty() || name == "single_slice") {
    return SliceMode::kSingleSlice;
  }
  if (name == "multi_slice") {
    return SliceMode::kMultiSlice;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unrecognized --slice_mode: ", name,
                   ". Expected 'single_slice' or 'multi_slice'."));
}

// Validates the worker flags of a child process. Unset integer flags default to
// -1, so a typo -- or an unexpanded shell variable in a hand-written launch
// command -- would otherwise reach ExecutorForDevice(-1) and surface as an
// opaque HIP_ERROR_InvalidDevice / CUDA invalid device error.
absl::Status ValidateWorkerFlags(int node_id, int world_size,
                                 int local_device_id) {
  if (world_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("--world_size must be positive, got %d.", world_size));
  }
  if (node_id < 0 || node_id >= world_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "--node_id must be in [0, %d), got %d. Pass the GLOBAL rank of this "
        "worker, i.e. node_index * ranks_per_node + local GPU index.",
        world_size, node_id));
  }
  if (local_device_id < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "--local_device_id must be non-negative, got %d. It is the GPU index "
        "within this host.",
        local_device_id));
  }
  return absl::OkStatus();
}

// Builds a GpuTopology that reflects the ACTUAL run layout, so the collective
// kernel is gated on the real multi-node shape rather than on a faked
// single-host one.
//
// Each process owns exactly one addressable device (allowed_devices =
// {local_device_id}), hence num_devices_per_process == 1 and the replica group
// always spans processes. The `ranks_per_node` GPUs of a host form the per-host
// device count, and the ranks are spread over `world_size / ranks_per_node`
// hosts.
//
// A populated gpu_target_config is REQUIRED: IsLsaPossible()
// (xla/backends/gpu/transforms/collectives/collective_ops_utils.cc) returns
// false for a topology without one, which forces the strategy back to
// KERNEL_STRATEGY_DEFAULT. Building it from this process's StreamExecutor also
// lets the compiler check the ROCm gfx9 MI300-series (gfx942/gfx950) LSA
// capability.
absl::StatusOr<GpuTopology> BuildMultiNodeGpuTopology(
    se::StreamExecutor* executor, int world_size, int ranks_per_node,
    SliceMode slice_mode) {
  const int devices_per_host = ranks_per_node > 0 ? ranks_per_node : 1;
  TF_RET_CHECK(world_size % devices_per_host == 0)
      << "world_size (" << world_size
      << ") must be a multiple of ranks_per_node (" << devices_per_host << ").";
  const int num_hosts = world_size / devices_per_host;
  const bool multi_slice = slice_mode == SliceMode::kMultiSlice;
  return GpuTopology(
      /*platform_version=*/"",
      /*num_partitions=*/multi_slice ? num_hosts : 1,
      /*num_hosts_per_partition=*/multi_slice ? 1 : num_hosts,
      /*num_devices_per_host=*/devices_per_host,
      /*gpu_target_config=*/gpu::GpuTargetConfig(executor),
      /*host_target_machine_options=*/std::nullopt,
      /*num_devices_per_process=*/1);
}

struct MultiProcessGpuClientSetup {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  std::unique_ptr<PjRtClient> client;
};

// Sets up the distributed runtime and a single-device PjRt GPU client for one
// rank.
//
//  - `rank_id`            : global rank (0 .. world_size-1).
//  - `world_size`         : total number of ranks across all nodes.
//  - `local_device_id`    : local GPU index on this host for this rank.
//  - `coordinator_address`: host:port of rank 0's coordination service.
absl::StatusOr<MultiProcessGpuClientSetup> SetUpMultiProcessGpuClient(
    int rank_id, int world_size, int local_device_id, int partition_index,
    absl::string_view coordinator_address, absl::string_view log_prefix) {
  MultiProcessGpuClientSetup prepared_test;

  // Extract the port from the coordinator address so that rank 0 can bind the
  // coordination service to all interfaces on that port.
  absl::string_view port_str = coordinator_address;
  auto colon_pos = port_str.rfind(':');
  if (colon_pos != absl::string_view::npos) {
    port_str = port_str.substr(colon_pos + 1);
  }

  // Rank 0 creates the coordination service.
  if (rank_id == 0) {
    LOG(INFO) << log_prefix << ": creating coordination service on "
              << coordinator_address;
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = world_size;
    ABSL_ASSIGN_OR_RETURN(
        prepared_test.service,
        xla::GetDistributedRuntimeService(
            absl::StrFormat("[::]:%s", port_str), service_options));
    LOG(INFO) << log_prefix << ": created coordination service";
  }

  // Connect to the coordination service.
  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = rank_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client = GetDistributedRuntimeClient(
      std::string(coordinator_address), distributed_options);

  LOG(INFO) << log_prefix << ": connecting distributed client";
  ABSL_RETURN_IF_ERROR(distributed_client->Connect());
  LOG(INFO) << log_prefix << ": distributed client connected";

  // Create the GPU client with a single addressable device per process. The
  // local_device_id selects which physical GPU on this host the process uses.
  GpuClientOptions options;
  options.node_id = rank_id;
  options.num_nodes = world_size;
  options.allowed_devices = {local_device_id};
  // When partition_index >= 0, every rank is placed into the same partition
  // (slice). This makes the derived GpuTopology report
  // slice_size == world_size while num_devices_per_process == 1, which is the
  // condition required for the cross-host Triton one-shot collective kernel
  // (see GetCollectiveKernelDomainSize / IsCrossHostCollectiveKernelPossible).
  // Leaving it unset keeps the default (one partition per process), i.e. the
  // single-host behavior where each process is its own slice.
  if (partition_index >= 0) {
    options.partition_index = partition_index;
  }
  options.kv_store =
      GetDistributedKeyValueStore(distributed_client, /*key_prefix=*/"gpu:");
  options.distributed_client = distributed_client;
  options.allocator_config.kind = xla::GpuAllocatorConfig::Kind::kBFC;
  options.allocator_config.gpu_system_memory_size = 32 * kMB;
  options.allocator_config.collective_memory_size = 0;
  options.use_tfrt_gpu_client = true;

  LOG(INFO) << log_prefix << ": creating PjRtClient";
  ABSL_ASSIGN_OR_RETURN(prepared_test.client, GetXlaPjrtGpuClient(options));
  LOG(INFO) << log_prefix << ": PjRtClient created";

  return prepared_test;
}

absl::Status AllReduceMultiProcessTestBody(
    int node_id, int world_size, int local_device_id, int partition_index,
    absl::string_view coordinator_address, int ranks_per_node,
    SliceMode slice_mode) {
  const int kWorldSize = world_size;
  std::string log_prefix = absl::StrFormat(
      "rank_%d[%s]", node_id, SliceModeName(slice_mode));
  ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                        PlatformUtil::GetPlatform("gpu"));
  ABSL_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        platform->ExecutorForDevice(local_device_id));
  const auto& desc = executor->GetDeviceDescription();
  if (desc.gpu_compute_capability().IsCuda() && !desc.gpu_compute_capability()
                                                     .cuda_compute_capability()
                                                     ->IsAtLeastAmpere()) {
    LOG(INFO) << log_prefix
              << ": skipping one-shot all-reduce test, requires Ampere+ for "
                 "CUDA.";
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(
      const GpuTopology gpu_topology,
      BuildMultiNodeGpuTopology(executor, kWorldSize, ranks_per_node,
                                slice_mode));
  // Every rank here is in its own process, so the collective is cross-host and
  // can only be served by load-store accessible symmetric memory. Targets that
  // cannot back it (pre-MI300 ROCm, ROCm < 10.0, NCCL_CUMEM_ENABLE unset) fall
  // back to the library collective, which this test is not about.
  if (!gpu::IsLsaPossible(gpu_topology)) {
    LOG(INFO) << log_prefix
              << ": skipping one-shot all-reduce test, the target does not "
                 "support load-store accessible symmetric memory.";
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(
      MultiProcessGpuClientSetup setup,
      SetUpMultiProcessGpuClient(node_id, kWorldSize, local_device_id,
                                 partition_index, coordinator_address,
                                 log_prefix));
  std::unique_ptr<PjRtClient> client = std::move(setup.client);

  TF_RET_CHECK(client->addressable_device_count() == 1)
      << "Expected exactly 1 local addressable device per process.";
  TF_RET_CHECK(client->device_count() == kWorldSize)
      << "Expected " << kWorldSize << " global devices.";

  // Build a single replica group "0,1,...,world_size-1".
  std::string replica_group;
  for (int i = 0; i < kWorldSize; ++i) {
    if (i != 0) {
      absl::StrAppend(&replica_group, ",");
    }
    absl::StrAppend(&replica_group, i);
  }
  const std::string module_str = absl::StrFormat(R"(
    HloModule test

    apply_op {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT apply_op = f32[] add(x, y)
    }

    ENTRY test_computation {
      param_0 = f32[128] parameter(0)
      ROOT all-reduce = f32[128] all-reduce(param_0), to_apply=apply_op,
      replica_groups={{%s}}
    }
  )",
                                                 replica_group);

  ABSL_ASSIGN_OR_RETURN(auto hlo_module, ParseAndReturnUnverifiedModule(
                                             module_str, /*config=*/{}));
  xla::XlaComputation computation(hlo_module->ToProto());

  xla::CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(kWorldSize);
  compile_options.executable_build_options.set_num_partitions(1);
  DeviceAssignment device_assignment(kWorldSize, 1);
  for (int i = 0; i < kWorldSize; ++i) {
    device_assignment(i, 0) = i;
  }
  compile_options.executable_build_options.set_device_assignment(
      device_assignment);

  compile_options.executable_build_options.set_gpu_topology(gpu_topology);

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_autotune_level(0);
  // Enable the collective-kernel pass for all-reduce so the compiler annotates
  // the instruction with a Triton collective strategy. Without this the
  // annotator never runs for all-reduce and the strategy stays DEFAULT (RCCL).
  debug_options.add_xla_gpu_experimental_use_collective_kernels(
      DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE);
  debug_options.set_xla_gpu_experimental_enable_tiling_propagation(true);
  // Opt in to the cross-host one-shot kernel. This is the only switch left for
  // a multi-node run: the kernel no longer requires the ranks to share a host
  // or a slice, so it is selected in both SliceMode layouts.
  debug_options.add_xla_gpu_unsupported_use_cross_host_one_shot_kernel(
      DebugOptions::ALLREDUCE);
  *compile_options.executable_build_options.mutable_debug_options() =
      debug_options;

  LOG(INFO) << log_prefix << ": compiling HLO module with one-shot all-reduce";
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                        client->CompileAndLoad(computation, compile_options));
  LOG(INFO) << log_prefix << ": compilation succeeded";

  // Inspect the collective kernel strategy selected in the optimized HLO.
  ABSL_ASSIGN_OR_RETURN(auto hlo_modules,
                        executable->GetExecutable()->GetHloModules());
  TF_RET_CHECK(!hlo_modules.empty());
  const HloModule* optimized_module = hlo_modules.front().get();
  bool found_all_reduce = false;
  for (const HloComputation* comp : optimized_module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAllReduce ||
          instr->opcode() == HloOpcode::kAllReduceStart) {
        found_all_reduce = true;
        auto gpu_config = instr->backend_config<gpu::GpuBackendConfig>();
        TF_RET_CHECK(gpu_config.ok())
            << "Failed to get GpuBackendConfig for " << instr->name() << ": "
            << gpu_config.status();
        LOG(INFO)
            << log_prefix << ": AllReduce instruction " << instr->name()
            << " collective kernel strategy: "
            << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                   gpu_config->collective_backend_config().kernel_strategy());
        TF_RET_CHECK(
            gpu_config->collective_backend_config().kernel_strategy() ==
            gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT)
            << "Expected AllReduce to use KERNEL_STRATEGY_TRITON_ONE_SHOT, but "
               "got: "
            << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                   gpu_config->collective_backend_config().kernel_strategy());
      }
    }
  }
  TF_RET_CHECK(found_all_reduce)
      << "Expected to find an AllReduce instruction in optimized HLO.";

  // Prepare input literal: each rank r provides the value 10*(r+1).
  const float input_val = 10.0f * static_cast<float>(node_id + 1);
  Literal input_literal =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, input_val));

  ABSL_ASSIGN_OR_RETURN(
      auto* memory_space,
      client->addressable_devices()[0]->default_memory_space());
  ABSL_ASSIGN_OR_RETURN(auto input_buffer, client->BufferFromHostLiteral(
                                               input_literal, memory_space));

  std::vector<std::vector<PjRtBuffer*>> input_ptrs = {{input_buffer.get()}};
  LOG(INFO) << log_prefix << ": executing one-shot all-reduce";
  ABSL_ASSIGN_OR_RETURN(auto results,
                        executable->Execute(input_ptrs, ExecuteOptions()));
  LOG(INFO) << log_prefix << ": execution finished";

  TF_RET_CHECK(results.size() == 1 && results[0].size() == 1);
  Literal result_literal(ShapeUtil::MakeShape(F32, {128}));
  ABSL_RETURN_IF_ERROR(results[0][0]->ToLiteralSync(&result_literal));

  // Expected result is the sum over all ranks of 10*(r+1).
  float expected_sum = 0.0f;
  for (int r = 0; r < kWorldSize; ++r) {
    expected_sum += 10.0f * static_cast<float>(r + 1);
  }
  Literal expected_literal =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, expected_sum));
  if (!LiteralTestUtil::Equal(expected_literal, result_literal)) {
    return absl::InternalError(absl::StrFormat(
        "Result literal %s does not match expected %s on rank %d",
        result_literal.ToString(), expected_literal.ToString(), node_id));
  }

  LOG(INFO) << log_prefix << ": verified result successfully";
  return absl::OkStatus();
}

int GetEnvInt(absl::string_view name, int default_value) {
  const char* v = std::getenv(std::string(name).c_str());
  return v != nullptr ? std::atoi(v) : default_value;
}

// Spawns `num_nodes * ranks_per_node` worker processes, one per GPU, all
// coordinating over 127.0.0.1, and checks that every one of them succeeds.
//
// The counts are configurable via environment variables so the same binary can
// also be launched manually across several physical nodes (see main()).
void RunMultiProcessAllReduce(SliceMode slice_mode) {
  const int num_nodes = GetEnvInt("XLA_TEST_NUM_NODES", kDefaultNumNodes);
  const int ranks_per_node =
      GetEnvInt("XLA_TEST_RANKS_PER_NODE", kDefaultRanksPerNode);
  const int world_size = num_nodes * ranks_per_node;

  absl::StatusOr<se::Platform*> platform = PlatformUtil::GetPlatform("gpu");
  // The single-host driver needs one local GPU per rank.
  if (!platform.ok() || (*platform)->VisibleDeviceCount() < world_size) {
    GTEST_SKIP() << "Test requires at least " << world_size
                 << " GPU devices, but found "
                 << (platform.ok() ? (*platform)->VisibleDeviceCount() : 0);
  }

  // Re-pass XLA_FLAGS environment variable to child processes.
  const char* xla_flags = std::getenv("XLA_FLAGS");
  if (xla_flags != nullptr) {
    tsl::setenv("XLA_FLAGS", xla_flags, /*overwrite=*/true);
  }

  int port = tsl::testing::PickUnusedPortOrDie();
  std::vector<tsl::SubProcess> child(world_size);

  for (int rank = 0; rank < world_size; ++rank) {
    // In kSingleSlice every rank joins partition 0, so the derived topology
    // reports one slice spanning all hosts. In kMultiSlice each host gets its
    // own partition, so the replica group also crosses slice boundaries.
    const int partition_index =
        slice_mode == SliceMode::kMultiSlice ? rank / ranks_per_node : 0;
    // On a single host, each rank uses a distinct local GPU index equal to its
    // global rank.
    std::vector<std::string> argv = {
        test_binary_name,
        "--test_to_run=AllReduceMultiProcessHelper",
        absl::StrFormat("--node_id=%d", rank),
        absl::StrFormat("--world_size=%d", world_size),
        absl::StrFormat("--ranks_per_node=%d", ranks_per_node),
        absl::StrFormat("--local_device_id=%d", rank),
        absl::StrFormat("--partition_index=%d", partition_index),
        absl::StrFormat("--slice_mode=%s", SliceModeName(slice_mode)),
        absl::StrFormat("--coordinator_address=127.0.0.1:%d", port),
        "--alsologtostderr",
        "--vmodule=gpu_executable=1,thunk_executor=1,all_reduce_thunk=5,"
        "collective_kernel_thunk=5,collective_memory=5",
    };
    child[rank].SetProgram(test_binary_name, argv);
    child[rank].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child[rank].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    ASSERT_TRUE(child[rank].Start()) << "Failed to start rank " << rank;
  }

  for (int rank = 0; rank < world_size; ++rank) {
    std::string stdout_str, stderr_str;
    int status = child[rank].Communicate(nullptr, &stdout_str, &stderr_str);

    const char* undeclared_outputs_dir =
        std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (undeclared_outputs_dir != nullptr &&
        undeclared_outputs_dir[0] != '\0') {
      std::string stderr_file = tsl::io::JoinPath(
          undeclared_outputs_dir,
          absl::StrFormat("subprocess_%s_rank_%d_stderr.log",
                          SliceModeName(slice_mode), rank));
      absl::Status write_status =
          tsl::WriteStringToFile(tsl::Env::Default(), stderr_file, stderr_str);
      if (!write_status.ok()) {
        LOG(WARNING) << "Failed to write stderr to " << stderr_file << ": "
                     << write_status;
      }
    }

    EXPECT_EQ(status, 0) << "rank " << rank << " failed with status " << status
                         << "\nstdout:\n"
                         << stdout_str << "\nstderr:\n"
                         << stderr_str;
  }
}

// All hosts in one interconnect slice (e.g. a GB200 NVL rack).
TEST(AllReduceMultiProcessE2ETest, OneShotAllReduceSingleSlice) {
  RunMultiProcessAllReduce(SliceMode::kSingleSlice);
}

// One slice per host: the replica group spans both host and slice boundaries.
// Before symmetric-memory windows were enabled for arbitrary multi-node
// targets, this layout fell back to the library collective.
TEST(AllReduceMultiProcessE2ETest, OneShotAllReduceMultiSlice) {
  RunMultiProcessAllReduce(SliceMode::kMultiSlice);
}

}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  std::string test_to_run;
  int node_id = -1;
  int world_size = -1;
  int num_nodes = -1;
  int ranks_per_node = -1;
  int local_device_id = -1;
  int partition_index = -1;
  int port = -1;
  std::string coordinator_address;
  std::string slice_mode_name;
  xla::test_binary_name = argv[0];

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("test_to_run", &test_to_run,
                "The test to run in the child process."),
      tsl::Flag("node_id", &node_id,
                "The global rank id for this worker process (0..world_size-1)."),
      tsl::Flag("world_size", &world_size,
                "Total number of ranks across all nodes. If unset, it is "
                "computed from --num_nodes * --ranks_per_node."),
      tsl::Flag("num_nodes", &num_nodes,
                "Number of physical nodes (hosts). Used with --ranks_per_node "
                "to compute world_size when --world_size is unset."),
      tsl::Flag("ranks_per_node", &ranks_per_node,
                "Number of ranks (GPUs) per node. Used with --num_nodes to "
                "compute world_size when --world_size is unset."),
      tsl::Flag("local_device_id", &local_device_id,
                "The local GPU device index this process should use. Defaults "
                "to node_id %% ranks_per_node (or node_id if unknown)."),
      tsl::Flag("partition_index", &partition_index,
                "Partition (slice) index for this process. Pass the SAME value "
                "(e.g. 0) on every rank to group all hosts into one NVLink/xGMI "
                "slice; pass node_id / ranks_per_node to give every host its "
                "own slice. Left unset (<0), each process is its own partition."),
      tsl::Flag("slice_mode", &slice_mode_name,
                "Slice layout of the GpuTopology handed to the compiler: "
                "'single_slice' (all hosts in one NVLink/xGMI slice, the "
                "default) or 'multi_slice' (one slice per host, so the replica "
                "group crosses slice boundaries). Should match "
                "--partition_index."),
      tsl::Flag("port", &port,
                "The coordinator port for distributed runtime. Ignored if "
                "--coordinator_address is set."),
      tsl::Flag("coordinator_address", &coordinator_address,
                "The coordinator address (host:port) for the distributed "
                "runtime. For multi-node runs, point every process at rank 0's "
                "reachable address. Defaults to 127.0.0.1:<port>."),
  };

  xla::AppendDebugOptionsFlags(&flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);

  if (test_to_run.empty()) {
    return RUN_ALL_TESTS();
  }

  // Resolve world_size from num_nodes * ranks_per_node when not given directly.
  if (world_size < 0) {
    const int nn = num_nodes > 0 ? num_nodes : xla::kDefaultNumNodes;
    const int rpn =
        ranks_per_node > 0 ? ranks_per_node : xla::kDefaultRanksPerNode;
    world_size = nn * rpn;
  }
  // Default the coordinator address to localhost when only --port is given.
  if (coordinator_address.empty()) {
    coordinator_address = absl::StrFormat("127.0.0.1:%d", port);
  }
  // Default the local device index. When ranks_per_node is known, the local
  // GPU index is node_id % ranks_per_node; otherwise fall back to node_id.
  if (local_device_id < 0) {
    if (ranks_per_node > 0) {
      local_device_id = node_id % ranks_per_node;
    } else {
      local_device_id = node_id;
    }
  }

  if (absl::Status status =
          xla::ValidateWorkerFlags(node_id, world_size, local_device_id);
      !status.ok()) {
    LOG(ERROR) << status;
    return status.raw_code();
  }

  absl::Status result = absl::OkStatus();
  if (test_to_run == "AllReduceMultiProcessHelper") {
    // Resolve ranks_per_node for topology construction. When not provided
    // explicitly, infer it: if --num_nodes is known, ranks_per_node =
    // world_size / num_nodes; otherwise fall back to the default (1 GPU per
    // host), which matches the common 1-rank-per-node manual launch.
    int effective_ranks_per_node = ranks_per_node;
    if (effective_ranks_per_node <= 0) {
      if (num_nodes > 0) {
        effective_ranks_per_node = world_size / num_nodes;
      } else {
        effective_ranks_per_node = xla::kDefaultRanksPerNode;
      }
    }
    absl::StatusOr<xla::SliceMode> slice_mode =
        xla::ParseSliceMode(slice_mode_name);
    if (!slice_mode.ok()) {
      LOG(ERROR) << slice_mode.status();
      return slice_mode.status().raw_code();
    }
    result = xla::AllReduceMultiProcessTestBody(
        node_id, world_size, local_device_id, partition_index,
        coordinator_address, effective_ranks_per_node, *slice_mode);
  } else {
    result = absl::InvalidArgumentError(absl::StrFormat(
        "Unrecognized multiprocess test name: %s", test_to_run));
  }

  if (!result.ok()) {
    LOG(ERROR) << "Child process (node_id " << node_id
               << ") failed: " << result;
  }
  return result.raw_code();
}
