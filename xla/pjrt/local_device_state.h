/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_PJRT_LOCAL_DEVICE_STATE_H_
#define XLA_PJRT_LOCAL_DEVICE_STATE_H_

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <stack>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/worker_thread.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

// Class that encapsulates state relating to a device (e.g., a GPU) on which we
// can perform computation and transfers. LocalDeviceState objects only exist
// for devices local to this host.
class LocalDeviceState {
 public:
  // There are three different semantics used by memory allocators on different
  // devices.
  enum AllocationModel {
    // kSynchronous is used by CPU devices.
    //
    // A buffer returned from the allocator can be used immediately.
    //
    // A buffer cannot be freed until after the last stream operation
    // referencing the buffer has completed, so the client is responsible for
    // keeping buffers alive until all device-side activity that consumes those
    // buffers has completed.
    //
    // The client's use of the device allocator corresponds to a view of the
    // tail of the last stream using a buffer.
    kSynchronous,

    // kComputeSynchronous is used by GPU devices.
    //
    // A buffer returned from the allocator at time t can be used after the
    // compute stream has finished executing the last computation enqueued
    // before time t.
    //
    // A buffer b can be freed after:
    //   1) The last use of b on the compute stream has been enqueued, and
    //   2) For any non-compute stream s on which an operation o using b is
    //      enqueued, either:
    //     a) The host has been notified that o has completed, or
    //     b) The next operation to be enqueued on the compute stream is
    //        guaranteed to be started after o has completed.
    //
    // The client's use of the device allocator corresponds to a view of the
    // tail of the compute stream.
    kComputeSynchronized,

    // kAsynchronous is used by TPU devices.
    //
    // A buffer returned from the allocator can be used immediately.
    //
    // A buffer b can be freed as soon as the last stream operation using b has
    // been enqueued.
    //
    // The allocator and lower-level runtime are responsible for keeping buffers
    // alive (if that is needed) from the perspective of the device until any
    // device-side work actually completes.
    //
    // The only exception is when a buffer is transferred between devices since
    // only one of the device executors knows about the transfer, so the buffer
    // must be manually kept alive from the perspective of the other executor.
    kAsynchronous
  };

  // Options for stream creations.
  struct StreamOptions {
    int priority = 0;
    int num_device_to_host_streams = 1;
    int num_device_to_device_streams = 1;

    // When true, GPU streams are not created in the constructor but are instead
    // created on demand the first time they are accessed. This can dramatically
    // reduce client construction time (e.g. ~385ms for 112 hipStreamCreate
    // calls on 8 GPUs) at the cost of a one-time latency on first stream use.
    //
    // Thread-safe: concurrent first-access from multiple threads is safe;
    // only one thread will create each stream.
    bool lazy_stream_creation = false;
  };

  // `device_ordinal` is the logical local device ordinal (returned by
  // `local_device_id()`), and it's used to look up an addressable device local
  // to a given client. If it is not set (-1 by default), the device's logical
  // device ordinal will be the same as its physical device ordinal (returned by
  // `local_hardware_id()`). In general, different PJRT devices have different
  // logical device ordinals, and several PJRT devices can have the same
  // physical device ordinal if they share the same physical device.
  LocalDeviceState(se::StreamExecutor* executor, LocalClient* client,
                   AllocationModel allocation_model,
                   std::optional<int> max_inflight_computations,
                   bool allow_event_reuse, bool use_callback_stream,
                   int device_ordinal = -1,
                   std::optional<StreamOptions> stream_options = std::nullopt,
                   bool schedule_async = false);
  virtual ~LocalDeviceState();

  se::StreamExecutor* executor() const { return executor_; }

  LocalDeviceId local_device_id() { return local_device_id_; }
  LocalChipId local_hardware_id() { return local_hardware_id_; }

  LocalClient* client() const { return client_; }

  AllocationModel allocation_model() const { return allocation_model_; }

  EventPool& event_pool() { return event_pool_; }

  // Returns the compute stream, creating it on first call when lazy stream
  // creation is enabled.
  se::Stream* compute_stream() const;

  // Returns the host-to-device stream, creating it on first call when lazy
  // stream creation is enabled.
  se::Stream* host_to_device_stream() const;

  // Returns a device to host stream. Allocates streams in a round-robin fashion
  // amongst the available streams.
  se::Stream* GetDeviceToHostStream();

  // Returns a device to device stream. Allocates streams in a round-robin
  // fashion amongst the available streams.
  se::Stream* GetDeviceToDeviceStream();

  // Returns a usage stream. Allocates streams in a round-robin fashion amongst
  // the available streams. When the overhead from BorrowStreamFromPool is too
  // large for a use case, consider using this API instead.
  se::Stream* GetFixedSizePoolUsageStream();

  // Return a stream that should be used to track when an externally-managed
  // buffer is ready. This is intended to support dlpack on GPU. Allocates
  // streams in a round-robin fashion amongst the available streams.
  se::Stream* GetExternalReadyEventStream();

  // Maps a raw platform-specific stream to an se::Stream* owned by this
  // LocalDeviceState. `stream` should have been derived from a se::Stream*
  // returned by GetExternalReadyEventStream.
  // TODO(skyewm): this function could map other raw streams if needed. It's
  // currently only used with external ready event streams.
  absl::StatusOr<se::Stream*> GetStreamFromExternalStream(std::intptr_t stream);

  // Returns a vector of device to device streams.
  std::vector<se::Stream*> GetDeviceToDeviceStreams();

  // Borrows a stream from a pool. The stream is guaranteed not to have any
  // currently outstanding work at its tail.
  std::unique_ptr<se::Stream> BorrowStreamFromPool();
  // Returns a stream to the pool. The caller must ensure the stream does not
  // have any outstanding work at its tail.
  void ReturnStreamToPool(std::unique_ptr<se::Stream> stream);

  // Enqueues a copy of `src_buffer` to `dst_buffer` onto `transfer_stream`.
  virtual absl::Status ThenMemcpyDeviceToDevice(
      se::Stream* transfer_stream, se::Stream* dst_stream,
      se::DeviceAddressBase src_buffer, se::DeviceAddressBase dst_buffer);

  WorkerThread* execute_thread() const { return execute_thread_.get(); }

  WorkerThread* async_dispatch_thread() const {
    return async_dispatch_thread_.get();
  }

  WorkerThread* cleanup_thread() const { return cleanup_thread_.get(); }

  // Enqueues a host callback on 'stream'. `stream` may, but need not, wait for
  // `callback` to complete. It is safe to call runtime methods from the
  // callback.
  // This API differs from ThenDoHostCallback in two ways:
  // a) ThenDoHostCallback is often constrained in what it can do, in
  //    particular, on GPU the callback runs on a thread belonging to the GPU
  //    runtime and cannot perform GPU operations itself. On GPU, callbacks
  //    execute in a separate thread.
  // b) ThenDoHostCallback waits for the callback to complete.
  absl::Status ThenExecuteCallback(
      se::Stream* stream, absl::AnyInvocable<void() &&> callback,
      absl::AnyInvocable<void(absl::Status) &&> error_cb = nullptr,
      absl::string_view tag = "");

  // Helpers for releasing values on a worker thread at the tail of a stream on
  // a worker thread. Copies `object`, and destroys the copy when the tail of
  // the stream is reached. The destruction happens either in the caller's
  // thread or on the worker thread (depending on thread schedules), not a
  // device callback, so it is safe if the destructor frees device resource
  // (e.g., GPU objects).
  template <typename T>
  absl::Status ThenRelease(se::Stream* stream, T&& object) {
    return ThenExecuteCallback(
        stream, [object = std::forward<T>(object)]() { /* releases object */ },
        nullptr, "ThenRelease");
  }

  std::optional<Semaphore>& compute_semaphore() { return compute_semaphore_; }

  // Returns a fresh, PRNG-generated random seed for an XLA computation.
  int GetNewPrngSeed();

  // Whether to allow deleting a buffer before the operation fulfilling the
  // buffer is scheduled by the host.
  bool allow_delete_before_fulfill() const {
    return allow_delete_before_fulfill_;
  }

  absl::Status AllocateAndRecordEvent(
      AsyncWorkRunner* async_work_runner, BufferSequencingEventRef event,
      se::Stream* stream, absl::string_view tag = "AllocateAndRecordEvent");

  size_t GetNextComputeStreamSyncPoint() {
    return next_compute_stream_sync_point_.load();
  }

  // Allows handing out very cheap event ids (GetNextComputeStreamSyncPoint())
  // which only incur the expense of constructing a cuda event if they're really
  // needed. This allows constructing a definition event per buffer.
  absl::StatusOr<BufferSequencingEventRef> GetEventForComputeStreamSyncPoint(
      size_t sync_point, AsyncWorkRunner* async_work_runner,
      bool nullptr_if_past = false);

 private:
  absl::Status SynchronizeAllActivity();

  // Creates a single GPU stream with the given name and optional priority
  // override. Uses stored stream_options_priority_ if no override is given.
  // This is the central stream-creation helper used by both the eager
  // (constructor) and lazy (first-access) paths. Safe to call from const
  // methods.
  std::unique_ptr<se::Stream> MakeStream(
      absl::string_view name,
      std::optional<se::StreamPriority> priority_override =
          std::nullopt) const;

  AllocationModel allocation_model_;

  EventPool event_pool_;

  // Semaphore used to limit how many programs can be enqueued on the compute
  // stream by the host ahead of the device.
  std::optional<Semaphore> compute_semaphore_;

  LocalDeviceId local_device_id_;
  LocalChipId local_hardware_id_;
  se::StreamExecutor* const executor_;
  LocalClient* const client_;

  // When lazy_stream_creation_ is true, these two streams start as nullptr and
  // are created on first access under lazy_init_mu_. The double-checked locking
  // pattern (atomic load + mutex) ensures a single creation with no lock
  // overhead on the hot path after initialisation.
  //
  // Declared mutable so that compute_stream() and host_to_device_stream(),
  // which are const methods, can lazily initialise them.
  mutable std::unique_ptr<se::Stream> compute_stream_;
  mutable std::unique_ptr<se::Stream> host_to_device_stream_;

  // Atomic "fast path" pointers: nullptr until the corresponding stream is
  // created. Read without locks; written once under lazy_init_mu_ using
  // memory_order_release / memory_order_acquire ordering.
  mutable std::atomic<se::Stream*> compute_stream_ptr_{nullptr};
  mutable std::atomic<se::Stream*> host_to_device_stream_ptr_{nullptr};

  // Protects the one-time lazy creation of compute_stream_ and
  // host_to_device_stream_. Not used when lazy_stream_creation_ is false.
  mutable absl::Mutex lazy_init_mu_;

  // When lazy_stream_creation_ is true these vectors are pre-sized with
  // nullptr entries. Entries are created on first access under mu_.
  std::vector<std::unique_ptr<se::Stream>> device_to_host_streams_;
  std::vector<std::unique_ptr<se::Stream>> device_to_device_streams_;
  std::vector<std::unique_ptr<se::Stream>> fixed_size_pool_usage_streams_;
  std::vector<std::unique_ptr<se::Stream>> external_ready_event_streams_;

  static constexpr int kNumDeviceToHostStreams = 4;
  static constexpr int kNumDeviceToDeviceStreams = 4;
  static constexpr int kNumFixedSizePoolUsageStreams = 4;
  static constexpr int kNumExternalReadyEventStreams = 4;

  // Whether GPU streams are created lazily on first access.
  bool lazy_stream_creation_ = false;

  // Stored for lazy stream creation: optional priority from StreamOptions
  // (nullopt means use executor default, i.e. CreateStream with no args).
  std::optional<int> stream_options_priority_;

  // Stored for lazy creation of device-to-device streams (highest priority on
  // GPU, default on SYCL).
  se::StreamPriority d2d_stream_priority_;

  // mu_ guards the round-robin indices and PRNG state. Declared mutable so
  // that const accessor methods can create lazy stream slots under the lock.
  mutable absl::Mutex mu_;
  int next_device_to_host_stream_ ABSL_GUARDED_BY(mu_) = 0;
  int next_device_to_device_stream_ ABSL_GUARDED_BY(mu_) = 0;
  int next_fixed_size_pool_usage_stream_ ABSL_GUARDED_BY(mu_) = 0;
  int next_external_ready_event_stream_ ABSL_GUARDED_BY(mu_) = 0;

  std::random_device prng_seed_device_ ABSL_GUARDED_BY(mu_);
  std::mt19937 prng_seed_generator_ ABSL_GUARDED_BY(mu_);
  std::uniform_int_distribution<> prng_seed_distribution_ ABSL_GUARDED_BY(mu_);

  absl::Mutex stream_pool_mu_;
  std::stack<std::unique_ptr<se::Stream>> usage_stream_pool_
      ABSL_GUARDED_BY(stream_pool_mu_);

  // Callback map pairs callback stream with a device stream and is used for
  // running short host-side callbacks after device side events, without
  // preventing the device-side stream from doing useful work.
  absl::Mutex callback_stream_map_mu_;
  std::optional<absl::flat_hash_map<se::Stream*, std::unique_ptr<se::Stream>>>
      callback_stream_map_;

  // A worker thread, used for replicated computation launches.
  std::unique_ptr<WorkerThread> execute_thread_;

  // A worker thread, used for launching executables async
  // Only if schedule_async=true is passed in the constructor.
  std::unique_ptr<WorkerThread> async_dispatch_thread_;

  // A worker thread, used for callbacks. It is necessary that this be a
  // different thread to the execute thread because we acquire the compute
  // semaphore during calls to Execute but release it from a callback and if
  // they are the same thread we might deadlock.
  std::unique_ptr<WorkerThread> callback_thread_;

  // One thread dedicated to cleaning up buffers. Scheduled work on this thread
  // may wait for other threads to schedule writes to buffers.
  std::unique_ptr<WorkerThread> cleanup_thread_;

  bool allow_delete_before_fulfill_ = true;

  std::atomic<size_t> next_compute_stream_sync_point_{0};
  size_t base_compute_event_sequence_id_ ABSL_GUARDED_BY(mu_) = 0;
  std::deque<BufferSequencingEventRef> compute_events_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // XLA_PJRT_LOCAL_DEVICE_STATE_H_
