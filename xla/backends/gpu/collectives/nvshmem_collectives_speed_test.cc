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

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/rocshmem_kernels.h"
#include "xla/core/collectives/collectives_registry.h"

#include "xla/debug_options_flags.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/primitive_util.h"
#include "xla/service/platform_util.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/util/command_line_flags.h"

namespace xla::gpu {
namespace {

// Tests that NVSHMEM library can be loaded and initialized.
TEST(NvshmemTest, Initialization) {
  const int num_nodes = 2;
  tsl::SubProcess child[num_nodes];
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    std::vector<std::string> argv;
    argv.push_back("nvshmem_test");
    argv.push_back(absl::StrFormat("--node_id=%d", node_id));
    argv.push_back(absl::StrFormat("--num_nodes=%d", num_nodes));
    child[node_id].SetProgram("/proc/self/exe", argv);
    child[node_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child[node_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    ASSERT_TRUE(child[node_id].Start()) << "node " << node_id;
  }
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    std::string stdout_str;
    std::string stderr_str;
    int child_status =
        child[node_id].Communicate(nullptr, &stdout_str, &stderr_str);
    EXPECT_EQ(child_status, 0) << " node " << node_id << "\nstdout:\n"
                               << stdout_str << "\nstderr:\n"
                               << stderr_str;
  }
}

using BenchmarkFunc = absl::AnyInvocable<Future<>(se::DeviceMemoryBase send_buf, 
      se::DeviceMemoryBase recv_buf, size_t num_elems, 
      const Communicator::Executor& executor)>;

absl::Status BenchmarkCollectivesOp(PrimitiveType dtype, 
        se::StreamExecutor *stream_exec, GpuCollectives *gpu_coll, 
     size_t num_ranks, size_t min_elems, size_t max_elems, BenchmarkFunc&& F) {

  TF_ASSIGN_OR_RETURN(auto stream, stream_exec->CreateStream());
  size_t dbytes = primitive_util::BitWidth(dtype) / 8,
         max_bytes = max_elems * dbytes;

#define USE_COMM_ALLOC 1
  se::DeviceMemoryBase send_buf, recv_buf;
  void *send_ptr = nullptr;

  absl::Cleanup cleanup = [&](){
#if USE_COMM_ALLOC
    gpu_coll->Deallocate(send_ptr).IgnoreError();
#else
    stream_exec->Deallocate(&send_buf);
#endif
  };

#if USE_COMM_ALLOC
  TF_ASSIGN_OR_RETURN(send_ptr, gpu_coll->Allocate(max_bytes*2));
  send_buf = se::DeviceMemoryBase{send_ptr, max_bytes*2 };
#else
  send_buf = stream_exec->AllocateArray< uint8_t >(max_bytes*2, 0);
  EXPECT_TRUE(!send_buf.is_null());
#endif
  recv_buf = send_buf.GetByteSlice(max_bytes, max_bytes);

  uint32_t n_warmups = 2, n_runs = 10;
  auto executor = GpuCollectives::On(*stream);

  EXPECT_TRUE(stream_exec->SynchronizeAllActivity());
  for (auto num_elems = min_elems; num_elems <= max_elems; 
            num_elems = num_elems*3/2) {

    std::unique_ptr<se::EventBasedTimer> timer;
    Future<> future; 
    for (uint32_t i = 0; i < n_warmups + n_runs; i++) {
      if (i == n_warmups) {
        TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(false));
      }
      future = F(send_buf, recv_buf, num_elems, executor);
    }
    EXPECT_TRUE(stream_exec->SynchronizeAllActivity());
    TF_RETURN_IF_ERROR(future.Await()); // do we need this ??
  
    TF_ASSIGN_OR_RETURN(auto elapsed, timer->GetElapsedDuration());
    auto msec = absl::ToDoubleMilliseconds(elapsed) / n_runs;
    // total bytes transferred / seconds elapsed
    double alg_bw = num_elems*dbytes / (msec * 1e6);
    double bus_bw = 2.0*alg_bw*(num_ranks - 1)/num_ranks;

    VLOG(0) << "num_bytes: " << (num_elems*dbytes) 
        << " time: " << msec << "ms, alg_bw: " << alg_bw << " Gbps "
        << " bus_bw: " << bus_bw << " Gbps";

    EXPECT_TRUE(stream_exec->SynchronizeAllActivity());
  } // for num_elems
 
  return absl::OkStatus();
}

absl::Status InitializationTestBody(const int node_id, 
      const int num_nodes, size_t min_elems, size_t max_elems) {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (node_id == 0) {
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = num_nodes;
    TF_ASSIGN_OR_RETURN(service, xla::GetDistributedRuntimeService(
                                     "[::]:12345", service_options));
  }

  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = node_id;
  distributed_options.init_timeout = absl::Seconds(120);
  distributed_options.heartbeat_timeout = absl::Seconds(1000);

  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12345", distributed_options);
  TF_QCHECK_OK(distributed_client->Connect());
  auto kv_store =
      GetDistributedKeyValueStore(distributed_client, /*key_prefix=*/"gpu:");

  TF_ASSIGN_OR_RETURN(auto platform, xla::PlatformUtil::GetPlatform("gpu"));
  TF_ASSIGN_OR_RETURN(auto executors, xla::PlatformUtil::GetStreamExecutors(
      platform, std::set< int >{ node_id }));

  EXPECT_TRUE(!executors.empty());
  auto ctx = executors[0]->Activate();

  gpu::GpuExecutableRunOptions run_opts;
  absl::flat_hash_map<GlobalDeviceId, int> device_to_node;
  std::map<int, GlobalDeviceId> gpu_device_ids;
  gpu_device_ids[node_id] = GlobalDeviceId{node_id};
  for(int i = 0; i < num_nodes; i++) {
    device_to_node[GlobalDeviceId{i}] = i;
  }
  run_opts.set_gpu_global_device_ids(
      std::move(gpu_device_ids));

  GpuCollectives::Topology topo{
    .node_id = node_id,
    .num_nodes = num_nodes,
    .device_count_per_process = 1,
    .kv_store = kv_store,
    .device_id_to_node_id = device_to_node,
    .gpu_executable_run_options = &run_opts,
  };

  using Type = float;
  auto dtype = primitive_util::NativeToPrimitiveType< Type >();

  bool use_nccl = true;
  std::unique_ptr< Communicator > comm;
  TF_ASSIGN_OR_RETURN(auto coll,  
          xla::CollectivesRegistry::Get("gpu", use_nccl ? "nccl" : "nvshmem"));
  
  auto *gpu_coll = tsl::down_cast<GpuCollectives*>(coll);
  if (gpu_coll == nullptr) {
    return absl::InternalError("Unsupported collectives implementation!");
  }
  TF_RETURN_IF_ERROR(gpu_coll->InitializeTopology(topo));

  if(use_nccl) 
  {
    std::vector<GlobalDeviceId> devices(num_nodes);
    for(int i = 0; i < num_nodes; i++) {
      devices[i] = GlobalDeviceId{i};
    }
    GpuCollectives::Device local_dev(executors[0]);
    std::vector<Collectives::DeviceRank> ranks(1, 
          Collectives::DeviceRank(&local_dev, RankId{node_id}));

    GpuCliqueKey clique_key(devices, 1);
    CliqueIds clique_ids;
    const auto& subkeys = clique_key.GetSubKeys(1);
    for (const auto& subkey : subkeys) {
      TF_ASSIGN_OR_RETURN(auto clique_id, run_opts.clique_id_callback()(subkey));
      clique_ids.Add(clique_id);
    }
    GpuCollectives::Config config;
    config.blocking_communicators = false;
    config.async_execution = true;
    TF_ASSIGN_OR_RETURN(auto comms, 
        gpu_coll->CreateCommunicatorsWithCancel(clique_key, clique_ids,
                                                   ranks, config, nullptr));
    EXPECT_TRUE(!comms.empty());
    comm = std::move(comms[0]);
  } else {
    TF_ASSIGN_OR_RETURN(comm, gpu_coll->CreateCommunicator());
  }

  return BenchmarkCollectivesOp(dtype, executors[0], gpu_coll, num_nodes,
      min_elems, max_elems, 
      [&](auto send_buf, auto recv_buf, size_t num_elems, 
                                          const auto& executor) -> Future<> {
        auto future = comm->AllReduce(send_buf, recv_buf, dtype, num_elems, 
              ReductionKind::SUM, executor);
        if (!use_nccl) {
          TF_RETURN_IF_ERROR(comm->Barrier(executor));
        }
        //TF_RETURN_IF_ERROR(comm->Quiet(executor));
        return future;
     }
  );
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  int node_id = -1;
  int num_nodes = -1;
  int min_elems = 1024, max_elems = 16*1024*1024;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("node_id", &node_id, "Node ID for Initialization test."),
      tsl::Flag("num_nodes", &num_nodes,
                "Number of nodes for Initialization test."),
      tsl::Flag("min_elems", &min_elems,
                "Min number of elements to reduce."),
      tsl::Flag("max_elems", &max_elems,
                "Max number of elements to reduce."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (node_id >= 0) {
    absl::Status result = xla::gpu::InitializationTestBody(
          node_id, num_nodes, min_elems, max_elems);
    if (!result.ok()) {
      LOG(ERROR) << result;
    }
    return result.raw_code();
  }
  return RUN_ALL_TESTS();
}
