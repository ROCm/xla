/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/command_buffer_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::gpu {

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

//===----------------------------------------------------------------------===//
// CommandBufferThunk
//===----------------------------------------------------------------------===//

void CommandBufferThunk::ExecutorCommandBuffer::AddNew(int64_t graph_id,
          BufferAllocation::Index max_index,
          std::unique_ptr<se::CommandBuffer> command_buffer) {
  
  // absl::MutexLock _(&mutex); ???
  cached_graphs_[graph_id] = std::move(command_buffer);
  recorded_allocs_[graph_id].resize(max_index + 1);
}


CommandBufferThunk::CommandBufferThunk(
    CommandBufferCmdExecutor commands, ThunkInfo thunk_info,
    std::unique_ptr<SequentialThunk> thunks,
    bool enable_command_buffers_during_profiling)
    : Thunk(Thunk::kCommandBuffer, std::move(thunk_info)),
      commands_(std::move(commands)),
      //thunks_(std::move(thunks)), // do not initialize thunks which 
      enable_command_buffers_during_profiling_(
          enable_command_buffers_during_profiling),
      state_(std::make_shared<State>()) {
  // When we create a new command buffer thunk (which happens when we
  // instantiate a new Gpu executable) we evict command buffers for all
  // previously instantiated executables. If previously instantiated executable
  // will be executed again, it will simply reconstruct command buffer from
  // a command buffer cmd sequence which is not terribly expensive (few
  // milliseconds for large command buffers). With this approach we keep command
  // buffers (CUDA graphs) resident in device memory only for executable that
  // are actually used.
  //
  // In a perfect world higher level framework (JAX, Tensorflow, PyTorch) would
  // be more aggressive with destroying unused executables, however today they
  // all have a pretty large LRU cache for keeping O(1000) XLA executables.
  EvictCommandBuffers();
  TrackCommandBuffers(state_);
}

bool CommandBufferThunk::ExecutorCommandBuffer::UpdateBufferAllocations(
    const CommandBufferCmdExecutor& commands,
    const Thunk::ExecuteParams& params) {

  const BufferAllocations* allocs = params.buffer_allocations;

  // first search if any of recorded graphs is fine
  auto start_id = (active_graph_ + NumCachedGraphs-1) % NumCachedGraphs;
  for (auto id = start_id; id < start_id + NumCachedGraphs; id++) { 
    bool should_update = false;
    auto& recorded = recorded_allocs_[id % NumCachedGraphs];
    for (const auto idx : commands.allocs_indices()) {
      auto alloc = allocs->GetDeviceAddress(idx);
      if (!recorded[idx].IsSameAs(alloc)) {
        should_update = true;
        break;
      }
    }
    if (!should_update) {
      active_graph_ = id % NumCachedGraphs;
      // if(params.stream->parent()->device_ordinal()==0)
      // VLOG(0) << "Setting active graph to: " << active_graph_;
      return false;
    }
  }
  // otherwise, we change the active graph to the LRU one ??
  active_graph_ = (active_graph_ + NumCachedGraphs-1) % NumCachedGraphs;
  if(params.stream->parent()->device_ordinal()==0) 
   VLOG(0) << "Recording to new active graph: " << active_graph_;

  // then there is no need to allocate graph earlier in AddNew function..
  // NOTE: well using params.stream->parent() is not very clean..
  auto command_buffer =
        params.stream->parent()->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary).value();
  cached_graphs_[active_graph_] = std::move(command_buffer);

  auto& recorded = recorded_allocs_[active_graph_];
  // We check only allocations referenced by commands in a cmd sequence, and
  // leave every other entry default initialized (nullptr device memory).
  for (const auto idx : commands.allocs_indices()) {
    recorded[idx] = allocs->GetDeviceAddress(idx);
  }
  return true;
}

absl::Status CommandBufferThunk::Prepare(
    const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) return absl::OkStatus();

  TF_RETURN_IF_ERROR(commands_.Prepare(params, resource_requests));

  // Always prepare thunks if they are present so we are ready to fall back
  // on them if we detect profiling activity.
  if (thunks_) {
    TF_RETURN_IF_ERROR(thunks_->Prepare(params, resource_requests));
  }

  return absl::OkStatus();
}

absl::Status CommandBufferThunk::Initialize(const InitializeParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) return absl::OkStatus();

  se::StreamExecutor* executor = params.executor;
  TF_ASSIGN_OR_RETURN(std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
                      GetOrCreateCommandBuffer(executor,
                            commands_.maximal_index()));

  absl::MutexLock lock(&cmd_buffer->mutex);

  // NOTE NOTE: this initialize must be called only once!!!
  // Initialize commands.
  TF_RETURN_IF_ERROR(commands_.Initialize(params, cmd_buffer->state));

  // Always initialize thunks if they are present so we are ready to fall back
  // on them if we detect profiling activity.
  if (thunks_) {
    TF_RETURN_IF_ERROR(thunks_->Initialize(params));
  }
#if 0
  // Construct ExecuteParams with empty fields for everything that is not needed
  // for recording commands.
  Thunk::ExecuteParams execute_params(
      params.buffer_allocations, params.stream,
      params.command_buffer_trace_stream, params.collective_params,
      params.collective_cliques, /*device_to_host_stream=*/nullptr,
      /*host_to_device_stream=*/nullptr,
      /*send_device_memory_function=*/nullptr,
      /*recv_device_memory_function=*/nullptr, params.ffi_execution_context,
      /*additional_compute_streams=*/{}, /*mock_collectives=*/false);

  // If command buffer is in `kCreate` state it means that command buffer
  // sequence was never recorded into it. We initialize all command buffers
  // before execution, because command buffers when instantiated will allocate
  // memory on device and this might lead to deadlocks when we have concurrent
  // NCCL operations in flight.
  //
  // If commands require initialization, we also record them into the command
  // buffer before execution. This is required to guarantee that collective
  // commands recorded on all participating ranks to avoid deadlocks.
  if (cmd_buffer->command_buffer->state() ==
          se::CommandBuffer::State::kCreate ||
      commands_.requires_initialization()) {
    VLOG(3) << "Initialize command buffer on device #"
            << params.executor->device_ordinal()
            << " by recoding command buffer cmd sequence"
            << "; num_commands=" << commands_.size();

    TraceMe trace([&] {
      return TraceMeEncode("command_buffer::initialize",
                           {{"device", params.executor->device_ordinal()},
                            {"num_commands", commands_.size()}});
    });

    uint64_t start_micros = tsl::Env::Default()->NowMicros();

    // Update recorded buffer allocations.
    auto updated_allocs =
        cmd_buffer->UpdateBufferAllocations(commands_, execute_params);

    CommandBufferCmd::RecordParams record_params = {cmd_buffer->state,
                                                    std::move(updated_allocs),
                                                    /*is_initialization=*/true};
    TF_RETURN_IF_ERROR(commands_.Record(execute_params, record_params,
                                        cmd_buffer->command_buffer.get()));

    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    VLOG(3) << "Initialized command buffer on device #"
            << params.executor->device_ordinal() << " in "
            << (end_micros - start_micros)
            << " μs; num_commands=" << commands_.size();
    cmd_buffer->num_executions = 0;
  }
#endif
  return absl::OkStatus();
}

absl::Status CommandBufferThunk::ExecuteOnStream(const ExecuteParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) return absl::OkStatus();

#if CMD_BUF_THUNK_ENABLE_TIMING
  uint64_t xstart = tsl::Env::Default()->NowMicros();
#endif

  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
                      GetOrCreateCommandBuffer(executor,
                            commands_.maximal_index()));

  absl::MutexLock lock(&cmd_buffer->mutex);

  // Update buffer allocations and collect all allocations that changed since
  // the last command buffer execution.
  if (cmd_buffer->UpdateBufferAllocations(commands_, params)) {

    // HACK HACK
    std::vector<BufferAllocation::Index> updated_allocs;
    VLOG(3) << "Update command buffer on device #" << executor->device_ordinal()
            << " by recoding command buffer cmd sequence after "
            << cmd_buffer->num_executions << " executions since last update"
            << "; num_commands=" << commands_.size()
            << "; updated_allocs=" << updated_allocs.size();

    TraceMe trace([&] {
      cmd_buffer->mutex.AssertHeld();
      return TraceMeEncode("command_buffer::update",
                           {{"device", executor->device_ordinal()},
                            {"num_commands", commands_.size()},
                            {"num_executions", cmd_buffer->num_executions}});
    });

    uint64_t start_micros = tsl::Env::Default()->NowMicros();

    CommandBufferCmd::RecordParams record_params = {cmd_buffer->state,
                                                    std::move(updated_allocs)};
    TF_RETURN_IF_ERROR(commands_.Record(params, record_params,
                                        cmd_buffer->ActiveGraph()));

    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    auto ss = (double)(end_micros - start_micros)/1e6;
#if CMD_BUF_THUNK_ENABLE_TIMING
    VLOG(0)
#else
    VLOG(3)
#endif
       << executor->device_ordinal() << " updated command buffer in " << ss
            << " sec; num_commands=" << commands_.size();
    cmd_buffer->num_executions = 0;
  }

  ++cmd_buffer->num_executions;

  VLOG(3) << "Execute command buffer on device #" << executor->device_ordinal()
          << "; num_executions=" << cmd_buffer->num_executions;

  TraceMe trace([&] {
    cmd_buffer->mutex.AssertHeld();
    return TraceMeEncode("command_buffer::execute",
                         {{"device", executor->device_ordinal()},
                          {"num_commands", commands_.size()},
                          {"num_executions", cmd_buffer->num_executions}});
  });
  auto s = cmd_buffer->ActiveGraph()->Submit(params.stream);
#if CMD_BUF_THUNK_ENABLE_TIMING
  params.stream->BlockHostUntilDone();
  uint64_t xend = tsl::Env::Default()->NowMicros();

  auto ss = (double)(xend - xstart)/1e6;
  VLOG(0) << executor->device_ordinal() << " total exec time " << ss
            << " sec; num_commands=" << commands_.size();
#endif
  return s;
}

absl::StatusOr<std::shared_ptr<CommandBufferThunk::ExecutorCommandBuffer>>
CommandBufferThunk::GetOrCreateCommandBuffer(se::StreamExecutor* executor,
        BufferAllocation::Index max_index) {
  absl::MutexLock lock(&state_->mutex);

  // Check if command buffer already exists
  if (auto it = state_->command_buffers.find(executor);
      it != state_->command_buffers.end()) {
    return it->second;
  }

  auto [it, _] = state_->command_buffers.emplace(
      executor, std::make_shared<ExecutorCommandBuffer>());

  for (int64_t i = 0; i < NumCachedGraphs; i++) {
    // Create a new empty command buffer.
    TF_ASSIGN_OR_RETURN(
        auto command_buffer,
        executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
    it->second->AddNew(i, max_index, std::move(command_buffer));
  }
  return it->second; // active graph is 0 by default
}

//===----------------------------------------------------------------------===//
// Command buffer eviction
//===----------------------------------------------------------------------===//

struct CommandBufferThunk::GlobalState {
  absl::Mutex mutex;
  std::vector<std::weak_ptr<CommandBufferThunk::State>> state
      ABSL_GUARDED_BY(mutex);
};

CommandBufferThunk::GlobalState* CommandBufferThunk::GetGlobalState() {
  static auto* const global_state = new GlobalState();
  return global_state;
}

void CommandBufferThunk::TrackCommandBuffers(
    std::weak_ptr<CommandBufferThunk::State> state) {
  auto* global_state = GetGlobalState();
  absl::MutexLock global_state_lock(&global_state->mutex);
  global_state->state.push_back(state);
}

void CommandBufferThunk::EvictCommandBuffers() {
  TraceMe trace([&] { return "EvictCommandBuffers"; });

  auto* global_state = GetGlobalState();
  absl::MutexLock global_state_lock(&global_state->mutex);
  VLOG(3) << "Evict command buffer thunk command buffers; tracked thunks = "
          << global_state->state.size();

  // Erase state for already destroyed thunks.
  global_state->state.erase(
      std::remove_if(global_state->state.begin(), global_state->state.end(),
                     [](auto& weak_ptr) { return weak_ptr.expired(); }),
      global_state->state.end());

  // Evict command buffers for all tracked thunks.
  int64_t num_evicted = 0;
  for (auto& weak_ptr : global_state->state) {
    auto ptr = weak_ptr.lock();
    if (!ptr) continue;

    // Evict all command buffers.
    absl::MutexLock state_lock(&ptr->mutex);
    num_evicted += ptr->command_buffers.size();
    ptr->command_buffers.clear();
  }

  if (num_evicted > 0) {
    VLOG(3) << "Evicted " << num_evicted
            << " command buffer thunk command buffers";
  }
}

void CommandBufferThunk::ForAllThunks(
    absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
  if (thunks_ != nullptr) {
    thunks_->ForAllThunks(fn);
  }
}

std::string CommandBufferThunk::ToString(int indent) const {
  std::string result = "\n";
  if(thunks_) {
    absl::StrAppend(&result, thunks_->ToString(indent + 1));
  }
  return result;
}

}  // namespace xla::gpu
