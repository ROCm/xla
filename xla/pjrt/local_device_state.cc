/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/pjrt/local_device_state.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/worker_thread.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

LocalDeviceState::LocalDeviceState(se::StreamExecutor* executor,
                                   LocalClient* client,
                                   AllocationModel allocation_model,
                                   std::optional<int> max_inflight_computations,
                                   bool allow_event_reuse,
                                   bool use_callback_stream, int device_ordinal,
                                   std::optional<StreamOptions> stream_options,
                                   bool schedule_async)
    : allocation_model_(allocation_model),
      event_pool_(allow_event_reuse),
      executor_(executor),
      client_(client),
      prng_seed_generator_(prng_seed_device_()),
      prng_seed_distribution_(std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()) {
  if (max_inflight_computations.has_value()) {
    compute_semaphore_.emplace(*max_inflight_computations);
  }

  // Setting XLA_PJRT_GPU_ALLOW_DELETE_BEFORE_FULFILL to false will:
  // 1. disallow the host to schedule `create buffer -> use -> delete ->
  // fulfill`, which is a use case unit tested in
  // StreamExecutorGpuClientTest.DeleteBufferThenFulfillBufferNoDeadLock.
  // 2. potentially reduce spikes in HBM usage because the host will wait for
  // buffer fulfillment to be scheduled before destructing it.
  absl::Status status =
      tsl::ReadBoolFromEnvVar("XLA_PJRT_GPU_ALLOW_DELETE_BEFORE_FULFILL", true,
                              &allow_delete_before_fulfill_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to read XLA_PJRT_GPU_ALLOW_DELETE_BEFORE_FULFILL: "
               << status;
  }

  local_hardware_id_ = executor_->device_ordinal();
  local_device_id_ =
      device_ordinal != -1 ? device_ordinal : executor_->device_ordinal();

  // Store priority options for lazy stream creation path.
  stream_options_priority_ =
      stream_options.has_value()
          ? std::optional<int>(stream_options->priority)
          : std::nullopt;

  // Compute and store the device-to-device stream priority once. Highest
  // priority is used on all platforms except SYCL (which does not yet support
  // non-default priorities for GPU streams).
  d2d_stream_priority_ =
      executor->GetPlatform()->id() == stream_executor::sycl::kSyclPlatformId
          ? se::StreamPriority::Default
          : se::StreamPriority::Highest;

  const bool do_lazy =
      stream_options.has_value() && stream_options->lazy_stream_creation;
  lazy_stream_creation_ = do_lazy;

  int num_device_to_host_streams =
      stream_options.has_value() ? stream_options->num_device_to_host_streams
                                 : kNumDeviceToHostStreams;
  int num_device_to_device_streams =
      stream_options.has_value() ? stream_options->num_device_to_device_streams
                                 : kNumDeviceToDeviceStreams;

  if (!do_lazy) {
    // -----------------------------------------------------------------------
    // Eager path (original behaviour): create all streams in the constructor.
    // -----------------------------------------------------------------------
    compute_stream_ = MakeStream("Compute");
    compute_stream_ptr_.store(compute_stream_.get(),
                              std::memory_order_release);

    host_to_device_stream_ = MakeStream("Host-to-device");
    host_to_device_stream_ptr_.store(host_to_device_stream_.get(),
                                     std::memory_order_release);

    device_to_host_streams_.reserve(num_device_to_host_streams);
    for (int i = 0; i < num_device_to_host_streams; ++i) {
      device_to_host_streams_.emplace_back(
          MakeStream(absl::StrFormat("Device-to-host #%d", i)));
    }

    device_to_device_streams_.reserve(num_device_to_device_streams);
    for (int i = 0; i < num_device_to_device_streams; ++i) {
      device_to_device_streams_.emplace_back(MakeStream(
          absl::StrFormat("Device-to-device #%d", i), d2d_stream_priority_));
    }

    fixed_size_pool_usage_streams_.reserve(kNumFixedSizePoolUsageStreams);
    for (int i = 0; i < kNumFixedSizePoolUsageStreams; ++i) {
      fixed_size_pool_usage_streams_.emplace_back(
          MakeStream(absl::StrFormat("Fixed size pool #%d", i)));
    }

    external_ready_event_streams_.reserve(kNumExternalReadyEventStreams);
    for (int i = 0; i < kNumExternalReadyEventStreams; ++i) {
      external_ready_event_streams_.emplace_back(
          MakeStream(absl::StrFormat("External ready event #%d", i)));
    }
  } else {
    // -----------------------------------------------------------------------
    // Lazy path: pre-size the vectors with nullptr entries. Streams are
    // created on first access under the appropriate lock.
    // compute_stream_ and host_to_device_stream_ remain nullptr; their atomic
    // fast-path pointers (compute_stream_ptr_ / host_to_device_stream_ptr_)
    // are already initialised to nullptr in the member initialisers.
    // -----------------------------------------------------------------------
    device_to_host_streams_.resize(num_device_to_host_streams);
    device_to_device_streams_.resize(num_device_to_device_streams);
    fixed_size_pool_usage_streams_.resize(kNumFixedSizePoolUsageStreams);
    external_ready_event_streams_.resize(kNumExternalReadyEventStreams);
  }

  if (use_callback_stream) {
    callback_stream_map_ =
        absl::flat_hash_map<se::Stream*, std::unique_ptr<se::Stream>>();
  }

  tsl::ThreadOptions thread_options;
  thread_options.numa_node = executor->numa_node();
  execute_thread_ = std::make_unique<WorkerThread>(
      tsl::Env::Default(), thread_options, "py_xla_execute");
  if (schedule_async) {
    async_dispatch_thread_ = std::make_unique<WorkerThread>(
        tsl::Env::Default(), thread_options, "py_xla_dispatch");
  }
  callback_thread_ = std::make_unique<WorkerThread>(
      tsl::Env::Default(), thread_options, "py_xla_callback");
  cleanup_thread_ = std::make_unique<WorkerThread>(
      tsl::Env::Default(), thread_options, "py_xla_cleanup");
}

LocalDeviceState::~LocalDeviceState() {
  absl::Status status = SynchronizeAllActivity();
  if (!status.ok()) {
    LOG(ERROR) << "Error when closing device: " << status;
  }

  // Explicitly delete all the streams and events to ensure that their callbacks
  // are executed before the destruction of the LocalDeviceState and its
  // callback threads.
  external_ready_event_streams_.clear();
  fixed_size_pool_usage_streams_.clear();
  device_to_device_streams_.clear();
  device_to_host_streams_.clear();
  host_to_device_stream_.reset();
  compute_stream_.reset();
  compute_events_.clear();
}

absl::Status LocalDeviceState::Reset() {
  // Step 1: Drain all pending GPU work on all streams.
  // After a normal execution this completes quickly (streams already idle).
  TF_RETURN_IF_ERROR(SynchronizeAllActivity());

  // Step 2: Reset XLA-level sequencing bookkeeping.
  // All events in compute_events_ have already fired (GPU work completed above),
  // so clearing the deque is safe. Resetting the counters to 0 is required so
  // the new client's operations start from a consistent baseline.
  {
    absl::MutexLock lock(&mu_);
    compute_events_.clear();
    next_compute_stream_sync_point_.store(0);
    base_compute_event_sequence_id_ = 0;
  }

  // Step 3: Clear the callback stream map.
  // Callback streams are re-created on demand when ThenExecuteCallback is
  // called. Each entry destroys a se::Stream (hipStreamDestroy).
  if (callback_stream_map_.has_value()) {
    absl::MutexLock lock(&callback_stream_map_mu_);
    callback_stream_map_->clear();
  }

  // Step 4: Drain the usage stream pool.
  // Pooled streams have been synchronized above; clear the pool so the new
  // client starts from a known empty state.
  {
    absl::MutexLock lock(&stream_pool_mu_);
    while (!usage_stream_pool_.empty()) usage_stream_pool_.pop();
  }

  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// MakeStream — central helper for GPU stream creation
// ---------------------------------------------------------------------------
// This helper centralises the stream creation logic that was previously
// embedded in the `create_stream` lambda inside the constructor. It must be
// callable from const methods (for the lazy path) because the stream vectors
// and unique_ptr members are `mutable` in that case.
//
// Priority resolution order:
//   1. priority_override (e.g. Highest for device-to-device streams)
//   2. stream_options_priority_ (caller-supplied StreamOptions::priority)
//   3. executor default (CreateStream with no args)
std::unique_ptr<se::Stream> LocalDeviceState::MakeStream(
    absl::string_view name,
    std::optional<se::StreamPriority> priority_override) const {
  std::unique_ptr<se::Stream> stream;
  if (priority_override.has_value()) {
    stream = executor_->CreateStream(*priority_override).value();
  } else if (stream_options_priority_.has_value()) {
    stream = executor_->CreateStream(*stream_options_priority_).value();
  } else {
    stream = executor_->CreateStream().value();
  }
  if (stream) {
    stream->SetName(std::string(name));
  }
  return stream;
}

// ---------------------------------------------------------------------------
// compute_stream() — lazy-initialised hot-path accessor
// ---------------------------------------------------------------------------
// Non-lazy case (lazy_stream_creation_ == false):
//   compute_stream_ptr_ is set in the constructor; the load is essentially
//   free (no branch misprediction on the hot path).
//
// Lazy case (lazy_stream_creation_ == true):
//   Uses double-checked locking with an atomic pointer as the fast path:
//     - acquire load: visible to all threads after the release store below.
//     - mutex slow path: only entered on the very first call.
//   After first call compute_stream_ptr_ is non-null and no lock is taken.
se::Stream* LocalDeviceState::compute_stream() const {
  se::Stream* ptr = compute_stream_ptr_.load(std::memory_order_acquire);
  if (ABSL_PREDICT_TRUE(ptr != nullptr)) {
    return ptr;
  }
  // Slow path — runs at most once per device.
  absl::MutexLock lock(&lazy_init_mu_);
  ptr = compute_stream_ptr_.load(std::memory_order_relaxed);
  if (ptr == nullptr) {
    tsl::profiler::TraceMe traceme("LazyCreateComputeStream");
    compute_stream_ = MakeStream("Compute");
    ptr = compute_stream_.get();
    compute_stream_ptr_.store(ptr, std::memory_order_release);
  }
  return ptr;
}

// ---------------------------------------------------------------------------
// host_to_device_stream() — same pattern as compute_stream()
// ---------------------------------------------------------------------------
se::Stream* LocalDeviceState::host_to_device_stream() const {
  se::Stream* ptr = host_to_device_stream_ptr_.load(std::memory_order_acquire);
  if (ABSL_PREDICT_TRUE(ptr != nullptr)) {
    return ptr;
  }
  absl::MutexLock lock(&lazy_init_mu_);
  ptr = host_to_device_stream_ptr_.load(std::memory_order_relaxed);
  if (ptr == nullptr) {
    tsl::profiler::TraceMe traceme("LazyCreateHostToDeviceStream");
    host_to_device_stream_ = MakeStream("Host-to-device");
    ptr = host_to_device_stream_.get();
    host_to_device_stream_ptr_.store(ptr, std::memory_order_release);
  }
  return ptr;
}

absl::Status LocalDeviceState::SynchronizeAllActivity() {
  absl::Status status;
  // TODO(phawkins): in theory the call to SynchronizeAllActivity below should
  // suffice. However on the Host platform SynchronizeAllActivity is a dummy
  // implementation that doesn't actually block. To make sure activity has
  // stopped, also block on the compute stream. If SynchronizeAllActivity is
  // fixed, we could remove the BlockHostUntilDone call.
  //
  // When lazy stream creation is enabled, a stream may never have been
  // created (if the device was constructed but never used). Guard every
  // access with a null check so we don't crash in that case.
  if (compute_stream_) {
    status.Update(compute_stream_->BlockHostUntilDone());
  }
  if (callback_stream_map_.has_value()) {
    absl::MutexLock lock(&callback_stream_map_mu_);
    for (auto& callback_stream : callback_stream_map_.value()) {
      status.Update(callback_stream.second->BlockHostUntilDone());
    }
  }
  for (auto& stream : device_to_host_streams_) {
    if (stream) {
      status.Update(stream->BlockHostUntilDone());
    }
  }
  // Use executor_ directly instead of compute_stream_->parent() so that
  // SynchronizeAllActivity works even when compute_stream_ is null (lazy).
  bool ok = executor_->SynchronizeAllActivity();
  if (!ok) {
    status.Update(Unknown("SynchronizeAllActivity failed."));
  }
  return status;
}

absl::Status LocalDeviceState::ThenMemcpyDeviceToDevice(
    se::Stream* transfer_stream, se::Stream* dst_stream,
    se::DeviceAddressBase src_buffer, se::DeviceAddressBase dst_buffer) {
  // The default implementation simply calls MemcpyD2D, and assumes that
  // the buffer addresses identify the devices. This does not work
  // on all platforms; this method is virtual so it can be overridden.
  return transfer_stream->MemcpyD2D(&dst_buffer, src_buffer, dst_buffer.size());
}

absl::Status LocalDeviceState::ThenExecuteCallback(
    se::Stream* stream, absl::AnyInvocable<void() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_cb, absl::string_view tag) {
  tsl::profiler::TraceMe traceme([&] {
    return tag.empty() ? "ThenExecuteCallback"
                       : absl::StrCat("ThenExecuteCallback:", tag);
  });
  if (callback_stream_map_.has_value()) {
    se::Stream* callback_exec_stream = nullptr;
    {
      // Prevent concurrent updates to the callback stream map.
      absl::MutexLock lock(&callback_stream_map_mu_);
      auto it = callback_stream_map_->find(stream);
      if (it == callback_stream_map_->end()) {
        tsl::profiler::TraceMe traceme_create("CreateCallbackStream");
        TF_ASSIGN_OR_RETURN(auto new_stream, executor_->CreateStream());
        new_stream->SetName(
            absl::StrFormat("Callback for %s", stream->GetName()));
        it =
            callback_stream_map_->insert({stream, std::move(new_stream)}).first;
      }
      callback_exec_stream = it->second.get();
    }
    tsl::profiler::TraceMe traceme_create("LocalDeviceState::WaitFor");
    TF_RETURN_IF_ERROR(callback_exec_stream->WaitFor(stream));
    stream = callback_exec_stream;
  }
  if (error_cb) {
    error_cb = [cb = std::move(error_cb),
                worker = callback_thread_.get()](absl::Status status) mutable {
      worker->Schedule(
          [cb = std::move(cb), status]() mutable { std::move(cb)(status); });
    };
  }
  return stream->DoHostCallback(
      [worker = callback_thread_.get(),
       callback{std::move(callback)}]() mutable {
        worker->Schedule(std::move(callback));
      },
      std::move(error_cb));
}

// ---------------------------------------------------------------------------
// Round-robin stream accessors with lazy slot creation
// ---------------------------------------------------------------------------
// When lazy_stream_creation_ is false the vectors hold fully-constructed
// streams, so the null check is a no-op compile-time branch that the
// compiler can eliminate (the pointers are always non-null in that case).
// When lazy_stream_creation_ is true the null check triggers the one-time
// stream creation for that slot, safely serialised by mu_.

se::Stream* LocalDeviceState::GetDeviceToHostStream() {
  absl::MutexLock lock(&mu_);
  int i = next_device_to_host_stream_;
  next_device_to_host_stream_ =
      (next_device_to_host_stream_ + 1) % device_to_host_streams_.size();
  if (!device_to_host_streams_[i]) {
    tsl::profiler::TraceMe traceme("LazyCreateDeviceToHostStream");
    device_to_host_streams_[i] =
        MakeStream(absl::StrFormat("Device-to-host #%d", i));
  }
  return device_to_host_streams_.at(i).get();
}

se::Stream* LocalDeviceState::GetDeviceToDeviceStream() {
  absl::MutexLock lock(&mu_);
  int i = next_device_to_device_stream_;
  next_device_to_device_stream_ =
      (next_device_to_device_stream_ + 1) % device_to_device_streams_.size();
  if (!device_to_device_streams_[i]) {
    tsl::profiler::TraceMe traceme("LazyCreateDeviceToDeviceStream");
    device_to_device_streams_[i] = MakeStream(
        absl::StrFormat("Device-to-device #%d", i), d2d_stream_priority_);
  }
  return device_to_device_streams_.at(i).get();
}

se::Stream* LocalDeviceState::GetFixedSizePoolUsageStream() {
  absl::MutexLock lock(&mu_);
  int i = next_fixed_size_pool_usage_stream_;
  next_fixed_size_pool_usage_stream_ =
      (next_fixed_size_pool_usage_stream_ + 1) %
      fixed_size_pool_usage_streams_.size();
  if (!fixed_size_pool_usage_streams_[i]) {
    tsl::profiler::TraceMe traceme("LazyCreateFixedSizePoolUsageStream");
    fixed_size_pool_usage_streams_[i] =
        MakeStream(absl::StrFormat("Fixed size pool #%d", i));
  }
  return fixed_size_pool_usage_streams_.at(i).get();
}

se::Stream* LocalDeviceState::GetExternalReadyEventStream() {
  absl::MutexLock lock(&mu_);
  int i = next_external_ready_event_stream_;
  next_external_ready_event_stream_ = (next_external_ready_event_stream_ + 1) %
                                      external_ready_event_streams_.size();
  if (!external_ready_event_streams_[i]) {
    tsl::profiler::TraceMe traceme("LazyCreateExternalReadyEventStream");
    external_ready_event_streams_[i] =
        MakeStream(absl::StrFormat("External ready event #%d", i));
  }
  return external_ready_event_streams_.at(i).get();
}

absl::StatusOr<se::Stream*> LocalDeviceState::GetStreamFromExternalStream(
    std::intptr_t stream) {
  // TODO(skyewm): replace with map lookup if performance is an issue (currently
  // it just iterates over 4 streams).
  for (const std::unique_ptr<se::Stream>& se_stream :
       external_ready_event_streams_) {
    // With lazy stream creation some slots may still be null (not yet
    // accessed). Skip them — they cannot match a caller-provided handle.
    if (!se_stream) continue;
    if (absl::bit_cast<std::intptr_t>(
            se_stream->platform_specific_handle().stream) == stream) {
      return se_stream.get();
    }
  }
  return NotFound(
      "GetStreamFromExternalStream failed to find stream. Only GPU streams "
      "used for dlpack imports are supported.");
}

std::vector<se::Stream*> LocalDeviceState::GetDeviceToDeviceStreams() {
  absl::MutexLock lock(&mu_);
  std::vector<se::Stream*> result;
  result.reserve(device_to_device_streams_.size());
  for (const auto& stream : device_to_device_streams_) {
    // Skip slots that have not yet been lazily created.
    if (stream) {
      result.push_back(stream.get());
    }
  }
  return result;
}

std::unique_ptr<se::Stream> LocalDeviceState::BorrowStreamFromPool() {
  {
    absl::MutexLock lock(&stream_pool_mu_);
    if (!usage_stream_pool_.empty()) {
      std::unique_ptr<se::Stream> stream = std::move(usage_stream_pool_.top());
      usage_stream_pool_.pop();
      auto status = stream->RefreshStatus();  // Can return error::Unimplemented
      // Stream may fail with "ABORTED: Bad connection".
      if (status.code() != tsl::error::ABORTED) {
        CHECK(stream->ok()) << status;
      }
      return stream;
    }
  }

  // The stream pool is empty, create a new stream.
  // Use executor_ directly rather than compute_stream_->parent() so that
  // BorrowStreamFromPool works even when lazy stream creation is enabled and
  // the compute stream has not yet been created.
  absl::StatusOr<std::unique_ptr<se::Stream>> stream =
      executor_->CreateStream();
  CHECK_OK(stream);
  (*stream)->SetName("Pool stream");
  return std::move(*stream);
}

void LocalDeviceState::ReturnStreamToPool(std::unique_ptr<se::Stream> stream) {
  auto status = stream->RefreshStatus();  // Can return error::Unimplemented
  // Stream may fail with "ABORTED: Bad connection".
  if (status.code() != tsl::error::ABORTED) {
    CHECK(stream->ok()) << status;
  }
  absl::MutexLock lock(&stream_pool_mu_);
  usage_stream_pool_.push(std::move(stream));
}

int LocalDeviceState::GetNewPrngSeed() {
  absl::MutexLock lock(&mu_);
  int x = 0;
  do {
    x = prng_seed_distribution_(prng_seed_generator_);
  } while (x == 0);
  return x;
}

absl::Status LocalDeviceState::AllocateAndRecordEvent(
    AsyncWorkRunner* async_work_runner, BufferSequencingEventRef event,
    se::Stream* stream, absl::string_view tag) {
  auto status = [&]() {
    TF_ASSIGN_OR_RETURN(
        EventPool::Handle device_event,
        event_pool().AllocateEvent(async_work_runner, stream->parent()));
    event_pool().ThenRecordEvent(stream, device_event);
    event->SetSequencingEvent(std::move(device_event), stream);
    return ThenExecuteCallback(
        stream, [event]() { event.SetStateConcrete(); },
        [event](absl::Status status) {
          event.SetError(event->AppendErrorContext(status));
        },
        tag);
  }();
  if (!status.ok()) {
    event.SetError(event->AppendErrorContext(status));
  }
  return status;
}

absl::StatusOr<BufferSequencingEventRef>
LocalDeviceState::GetEventForComputeStreamSyncPoint(
    size_t sync_point, AsyncWorkRunner* async_work_runner,
    bool nullptr_if_past) {
  mu_.lock();
  size_t cur_sync_point = next_compute_stream_sync_point_.load();
  if (sync_point < base_compute_event_sequence_id_ + compute_events_.size()) {
    BufferSequencingEventRef event;
    if (sync_point < base_compute_event_sequence_id_) {
      if (!nullptr_if_past) {
        DCHECK_GT(compute_events_.size(), 0);
        event = compute_events_.front();
      }
    } else {
      event = compute_events_[sync_point - base_compute_event_sequence_id_];
    }
    mu_.unlock();
    return event;
  }
  next_compute_stream_sync_point_.store(cur_sync_point + 1);
  auto event = BufferSequencingEvent::Create(async_work_runner);
  auto status =
      AllocateAndRecordEvent(async_work_runner, event, compute_stream(),
                             "GetEventForComputeStreamSyncPoint");
  if (!status.ok()) {
    mu_.unlock();
    return status;
  }
  while (!(cur_sync_point - base_compute_event_sequence_id_ <
           compute_events_.size())) {
    // Pad for any failed event allocations.
    compute_events_.push_back(event);
  }
  mu_.unlock();
  event.AndThen([this, cur_sync_point]() {
    absl::MutexLock l(&mu_);
    while (base_compute_event_sequence_id_ < cur_sync_point) {
      compute_events_.pop_front();
      ++base_compute_event_sequence_id_;
    }
  });
  return event;
}

}  // namespace xla
