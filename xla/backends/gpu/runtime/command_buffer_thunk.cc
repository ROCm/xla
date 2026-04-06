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
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::gpu {

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

//===----------------------------------------------------------------------===//
// CommandBufferThunk
//===----------------------------------------------------------------------===//

CommandBufferThunk::ExecutorCommandBuffer::ExecutorCommandBuffer(
    std::unique_ptr<se::CommandBuffer> command_buffer)
    : command_buffer(std::move(command_buffer)) {}

CommandBufferThunk::CommandBufferThunk(
    CommandExecutor commands, ThunkInfo thunk_info,
    std::unique_ptr<SequentialThunk> thunks,
    bool enable_command_buffers_during_profiling,
    bool enable_command_buffer_va_remapping)
    : Thunk(Thunk::kCommandBuffer, std::move(thunk_info)),
      commands_(std::move(commands)),
      thunks_(std::move(thunks)),
      enable_command_buffers_during_profiling_(
          enable_command_buffers_during_profiling),
      enable_command_buffer_va_remapping_(enable_command_buffer_va_remapping),
      enable_cached_allocs_([] {
        const char* env = std::getenv("XLA_COMMAND_BUFFERS_USE_CACHED_ALLOCS");
        return env && (std::string(env) == "1" || std::string(env) == "true");
      }()),
      state_(std::make_shared<State>()) {
  if (VLOG_IS_ON(5)) {
    absl::StatusOr<std::string> graph = commands_.RenderExecutionGraph();
    if (graph.ok()) {
      VLOG(5) << "Rendered command buffer execution graph: " << *graph;
    } else {
      VLOG(5) << "Failed to render command buffer execution graph: "
              << graph.status();
    }
  }

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

std::vector<BufferAllocation::Index>
CommandBufferThunk::ExecutorCommandBuffer::UpdateBufferAllocations(
    const CommandExecutor& commands, const Thunk::ExecuteParams& params) {
  std::vector<BufferAllocation::Index> updated_allocs;
  const BufferAllocations* allocs = params.buffer_allocations;

  // We check only allocations referenced by commands in a cmd sequence, and
  // leave every other entry default initialized (nullptr device memory).
  for (BufferAllocation::Index index : commands.allocs_indices()) {
    se::DeviceAddressBase alloc = allocs->GetDeviceAddress(index);

    if (recorded_allocs.size() <= index) {
      recorded_allocs.resize(index + 1);
      updated_allocs.push_back(index);
    }

    if (!recorded_allocs[index].IsSameAs(alloc)) {
      recorded_allocs[index] = alloc;
      updated_allocs.push_back(index);
    }
  }

  return updated_allocs;
}

absl::Status CommandBufferThunk::Prepare(const PrepareParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // Always prepare thunks if they are present so we are ready to fall back
  // on them if we detect profiling activity.
  if (thunks_) {
    TF_RETURN_IF_ERROR(thunks_->Prepare(params));
  }

  // TODO(b/290773547): Disabled CUDA graphs when profiling is active because of
  // memory corruption.
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Prepare command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TraceMe trace("WARNING: CommandBuffer disabled when profiling");
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(commands_.Prepare(params));

  return absl::OkStatus();
}

absl::Status CommandBufferThunk::Initialize(const InitializeParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
                      GetOrCreateCommandBuffer(params.executor));
  absl::MutexLock lock(cmd_buffer->mutex);

  // Initialize commands.
  TF_RETURN_IF_ERROR(commands_.Initialize(params));

  // Always initialize thunks if they are present so we are ready to fall back
  // on them if we detect profiling activity.
  if (thunks_) {
    TF_RETURN_IF_ERROR(thunks_->Initialize(params));
  }

  // TODO(b/290773547): Disabled CUDA graphs when profiling is active because of
  // memory corruption.
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Initialize command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TraceMe trace("WARNING: CommandBuffer disabled when profiling");
    return absl::OkStatus();
  }

  // If there are no thunks, or command buffer does not require initialization,
  // we can mark warm up as done immediately.
  if (!thunks_ || !commands_.requires_initialization()) {
    cmd_buffer->warmup_done = true;
  }

  // Construct ExecuteParams with empty fields for everything that is not needed
  // for recording commands.
  Thunk::ExecuteParams execute_params(
      params.buffer_allocations, params.stream,
      params.command_buffer_trace_stream, params.collective_params,
      params.collective_cliques, params.collective_memory,
      /*device_to_host_stream=*/nullptr,
      /*host_to_device_stream=*/nullptr,
      /*send_device_memory_function=*/nullptr,
      /*recv_device_memory_function=*/nullptr, params.ffi_execution_context,
      /*additional_compute_streams=*/{}, params.execution_scoped_state,
      /*mock_collectives=*/false);

  // If command buffer is in `kCreate` state it means that command buffer
  // sequence was never recorded into it. We initialize all command buffers
  // before execution, because command buffers when instantiated will allocate
  // memory on device and this might lead to deadlocks when we have concurrent
  // NCCL operations in flight.
  //
  // If commands require initialization (and VA remapping is not enabled), we
  // also record them into the command buffer before execution. This is required
  // to guarantee that collective commands are recorded on all participating
  // ranks to avoid deadlocks.
  // Skip re-initialization when VA remapping is enabled (addresses are
  // remapped at the hardware level) or when the command buffer has already
  // been initialized (addresses are stable via CachedAllocator or similar).
  bool first_init = cmd_buffer->command_buffer->state() ==
                    se::CommandBuffer::State::kCreate;
  bool needs_collective_init =
      !enable_command_buffer_va_remapping_ &&
      !cmd_buffer->collective_init_done &&
      commands_.requires_initialization();
  if (cmd_buffer->warmup_done && (first_init || needs_collective_init)) {
    cmd_buffer->collective_init_done = true;
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

    auto updated_allocs =
        cmd_buffer->UpdateBufferAllocations(commands_, execute_params);

    Command::RecordParams record_params = {cmd_buffer->state,
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
  return absl::OkStatus();
}

absl::Status CommandBufferThunk::ExecuteOnStream(const ExecuteParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // TODO(b/290773547): Profiler (CUPTI) + CUDA graphs lead to memory
  // corruption. As a work around disable command buffers (CUDA graphs) and run
  // everything in op-by-op mode.
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Execute command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TraceMe trace("WARNING: CommandBuffer disabled when profiling");
    return thunks_->ExecuteOnStream(params);
  }

  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<ExecutorCommandBuffer> cmd_buffer,
                      GetOrCreateCommandBuffer(executor));

  absl::MutexLock lock(cmd_buffer->mutex);

  // warm up iteration, run through thunks if they are present.
  if (!cmd_buffer->warmup_done && thunks_) {
    VLOG(2) << "Executing warm up iteration of command buffer thunk";
    TF_RETURN_IF_ERROR(thunks_->ExecuteOnStream(params));
    cmd_buffer->warmup_done = true;
    return absl::OkStatus();
  }

  auto updated_allocs = cmd_buffer->UpdateBufferAllocations(commands_, params);

  // Stabilize allocation addresses: for allocations that changed address
  // (e.g. donated parameters), memcpy data to a cached stable address and
  // revert recorded_allocs to the stable address. This prevents both the
  // coarse-grained UpdateBufferAllocations check and the fine-grained
  // TracedCommandBuffer cache from seeing address churn.
  const Thunk::ExecuteParams* effective_params = &params;
  std::optional<BufferAllocations> stable_allocs;
  std::optional<Thunk::ExecuteParams> stable_params;
  // Indices that were stabilized and need post-execution copy-back.
  std::vector<BufferAllocation::Index> copyback_indices;

  if (enable_cached_allocs_ && cmd_buffer->alloc_cache_ready &&
      !updated_allocs.empty()) {
    int64_t stabilized = 0;
    const BufferAllocations* orig = params.buffer_allocations;
    auto& cache = cmd_buffer->alloc_address_cache;

    for (auto it = updated_allocs.begin(); it != updated_allocs.end();) {
      BufferAllocation::Index idx = *it;
      se::DeviceAddressBase current = orig->GetDeviceAddress(idx);
      if (current.is_null() || current.size() == 0) {
        ++it;
        continue;
      }

      auto cached_it = cache.find(idx);
      if (cached_it != cache.end() &&
          cached_it->second.size() >= current.size()) {
        se::DeviceAddressBase& cached = cached_it->second;
        if (!cached.IsSameAs(current)) {
          TF_RETURN_IF_ERROR(
              params.stream->MemcpyD2D(&cached, current, current.size()));
          cmd_buffer->recorded_allocs[idx] = cached;
          copyback_indices.push_back(idx);
          ++stabilized;
        }
        it = updated_allocs.erase(it);
      } else {
        se::DeviceAddressBase persistent =
            executor->Allocate(current.size(), /*memory_space=*/0);
        if (!persistent.is_null()) {
          TF_RETURN_IF_ERROR(params.stream->MemcpyD2D(
              &persistent, current, current.size()));
          cache[idx] = persistent;
          cmd_buffer->recorded_allocs[idx] = persistent;
          copyback_indices.push_back(idx);
          ++stabilized;
        } else {
          LOG(WARNING) << "CachedAllocs: failed to allocate persistent "
                       << "buffer for alloc " << idx
                       << " size=" << current.size();
        }
        ++it;
      }
    }

    if (stabilized > 0) {
      // Build a BufferAllocations with cached addresses so that
      // TracedCommandBuffer sees stable pointers.
      size_t n = orig->size();
      std::vector<se::DeviceAddressBase> stable_bufs;
      stable_bufs.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        auto ci = cache.find(i);
        if (ci != cache.end()) {
          stable_bufs.push_back(ci->second);
        } else {
          stable_bufs.push_back(orig->GetDeviceAddress(i));
        }
      }
      stable_allocs.emplace(stable_bufs, orig->device_ordinal(),
                            orig->memory_allocator());
      stable_params.emplace(
          Thunk::ExecuteParams::CloneWithNewAllocations(params, *stable_allocs));
      effective_params = &*stable_params;
    }
  }

  bool is_first_record =
      enable_command_buffer_va_remapping_ &&
      cmd_buffer->command_buffer->state() == se::CommandBuffer::State::kCreate;
  bool needs_update = !enable_command_buffer_va_remapping_ &&
                      (!updated_allocs.empty() || commands_.force_update());

  VLOG(3) << "CmdBufThunk dev=" << executor->device_ordinal()
           << " updated=" << updated_allocs.size()
           << " stabilized=" << copyback_indices.size()
           << " force=" << commands_.force_update()
           << " needs_update=" << needs_update;

  if (is_first_record || needs_update) {
    XLA_VLOG_DEVICE(3, executor->device_ordinal())
        << "Create/Update command buffer"
        << " by recoding command buffer cmd sequence after "
        << cmd_buffer->num_executions << " executions since last update"
        << "; num_commands=" << commands_.size()
        << "; updated_allocs=" << updated_allocs.size()
        << "; is_first_record=" << is_first_record
        << "; needs_update=" << needs_update;

    TraceMe trace([&] {
      cmd_buffer->mutex.AssertHeld();
      return TraceMeEncode(needs_update
                               ? "command_buffer::update"
                               : "command_buffer::record_for_va_remapping",
                           {{"device", executor->device_ordinal()},
                            {"num_commands", commands_.size()},
                            {"num_executions", cmd_buffer->num_executions}});
    });

    uint64_t start_micros = tsl::Env::Default()->NowMicros();

    Command::RecordParams record_params = {
        cmd_buffer->state, std::move(updated_allocs),
        /*is_initialization=*/is_first_record};
    TF_RETURN_IF_ERROR(commands_.Record(*effective_params, record_params,
                                        cmd_buffer->command_buffer.get()));

    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    XLA_VLOG_DEVICE(3, executor->device_ordinal())
        << (needs_update ? "Updated" : "Recorded") << " command buffer in "
        << (end_micros - start_micros)
        << " μs; num_commands=" << commands_.size();
    cmd_buffer->num_executions = 0;

    if (enable_cached_allocs_ && !cmd_buffer->alloc_cache_ready) {
      cmd_buffer->alloc_cache_ready = true;
    }
  }

  ++cmd_buffer->num_executions;

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "Execute command buffer"
      << "; num_executions=" << cmd_buffer->num_executions;

  TraceMe trace([&] {
    cmd_buffer->mutex.AssertHeld();
    return TraceMeEncode("command_buffer::execute",
                         {{"device", executor->device_ordinal()},
                          {"num_commands", commands_.size()},
                          {"num_executions", cmd_buffer->num_executions}});
  });

  TF_RETURN_IF_ERROR(cmd_buffer->command_buffer->Submit(params.stream));

  // Copy results from persistent buffers back to the original (donated)
  // addresses so the framework receives correct output data.
  if (!copyback_indices.empty()) {
    auto& cache = cmd_buffer->alloc_address_cache;
    const BufferAllocations* orig = params.buffer_allocations;
    for (auto idx : copyback_indices) {
      se::DeviceAddressBase donated = orig->GetDeviceAddress(idx);
      auto ci = cache.find(idx);
      if (ci == cache.end() || donated.is_null()) continue;
      uint64_t copy_size = std::min(ci->second.size(), donated.size());
      TF_RETURN_IF_ERROR(params.stream->MemcpyD2D(
          &donated, ci->second, copy_size));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<CommandBufferThunk::ExecutorCommandBuffer>>
CommandBufferThunk::GetOrCreateCommandBuffer(se::StreamExecutor* executor) {
  absl::MutexLock lock(state_->mutex);

  // Check if command buffer already exists
  if (auto it = state_->command_buffers.find(executor);
      it != state_->command_buffers.end()) {
    return it->second;
  }

  // Create a new empty command buffer.
  TF_ASSIGN_OR_RETURN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  auto emplaced = state_->command_buffers.emplace(
      executor,
      std::make_shared<ExecutorCommandBuffer>(std::move(command_buffer)));

  return emplaced.first->second;
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
  absl::MutexLock global_state_lock(global_state->mutex);
  global_state->state.push_back(state);
}

void CommandBufferThunk::EvictCommandBuffers() {
  TraceMe trace("EvictCommandBuffers");

  auto* global_state = GetGlobalState();
  absl::MutexLock global_state_lock(global_state->mutex);
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
    if (!ptr) {
      continue;
    }

    // Evict all command buffers.
    absl::MutexLock state_lock(ptr->mutex);
    num_evicted += ptr->command_buffers.size();
    ptr->command_buffers.clear();
  }

  if (num_evicted > 0) {
    VLOG(3) << "Evicted " << num_evicted
            << " command buffer thunk command buffers";
  }
}

absl::Status CommandBufferThunk::WalkNested(Walker callback) {
  if (thunks_ != nullptr) {
    TF_RETURN_IF_ERROR(thunks_->Walk(callback));
  }
  return absl::OkStatus();
}

std::string CommandBufferThunk::ToString(int indent) const {
  std::string result = "\n";
  absl::StrAppend(&result, thunks_->ToString(indent + 1));
  return result;
}

}  // namespace xla::gpu
