/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/profiler/gpu/rocm_device_map.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/stream_executor/rocm/roctracer_wrapper.h"

namespace xla {
namespace profiler {
// forward declare (interface)
class RocmTraceCollector;

// The class use to enable rocprofiler-sdk buffered callback/activity tracing
// and forward the collected trace events to RocmTraceCollector. There should be
// only one RocmTracer per process.
class RocmTracer {
 public:
  // Returns a reference to the singleton instance of RocmTracer.
  // This ensures that only one global instance exists throughout the process
  // lifetime. The first call to this function lazily constructs the instance in
  // a thread-safe manner. Subsequent calls return the same instance, enabling
  // centralized tracer state management.
  static RocmTracer& GetRocmTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;

  absl::Status Enable(RocmTraceCollector* collector);
  void Disable();

  static uint64_t GetTimestamp();

  // Rows are numbered by HIP device. The first call asks HIP which GPUs it
  // numbers and matches them to the SDK's agents; every call returns that
  // map. Until then NumGpus() and GpuAgents() report no GPUs, and rows carry
  // kInvalidDeviceId. Calls HIP, so never call it from an SDK callback or
  // while holding collector_mutex_ or kernel_lock_.
  const HipDeviceMap& EnsureHipDeviceMap();
  uint32_t NumGpus() const { return static_cast<uint32_t>(GpuAgents().size()); }
  // Indexed by HIP device number.
  absl::Span<const rocprofiler_agent_v0_t> GpuAgents() const {
    const HipDeviceMap* map = hip_device_map_.load(std::memory_order_acquire);
    if (map == nullptr) return {};
    return map->gpu_agents;
  }
  RocmTraceCollector* collector() { return collector_; }

  // The SDK GPU agent that owns the allocation containing `address`, or
  // nullopt if HSA doesn't know the address, a CPU agent owns it, or no single
  // SDK GPU agent has the owner's PCI address. Calls HSA, so never call it on
  // the SDK buffer thread.
  std::optional<uint64_t> AgentOwningAddress(uint64_t address);

  int toolInit(rocprofiler_client_finalize_t finalize_func, void* tool_data);
  static void toolFinalize(void* tool_data);

  void TracingCallback(rocprofiler_context_id_t context,
                       rocprofiler_buffer_id_t buffer_id,
                       rocprofiler_record_header_t** headers,
                       size_t num_headers, uint64_t drop_count);

  void CodeObjectCallback(rocprofiler_callback_tracing_record_t record,
                          void* callback_data);

  // Called from the MARKER_CORE_API callback. Handles roctxRangePushA,
  // roctxRangePop, and roctxMarkA, emitting RocmTracerEvent(Generic) for each
  // completed range or instantaneous mark.
  void MarkerCallback(const rocprofiler_callback_tracing_record_t& record);

  // Returns the label of the innermost ROCTX range active on the CALLING
  // thread, or empty if none. Takes no thread id: the range stack is
  // thread_local, and the HIP API callback that consumes this runs on the same
  // thread that pushed the range.
  //
  // The view aliases the thread_local frame, so it is only valid until this
  // thread makes its next roctx call. That is enough for the caller: the HIP
  // API callback runs synchronously on the thread that owns the stack, so no
  // pop can interleave, and AnnotationMap::Add interns a copy before
  // returning. Do not store the view.
  absl::string_view GetCurrentRoctxLabel();

  // Builds and hands a Generic (ROCTX marker) event to the collector. Shared
  // by the range-pop and mark paths so the two cannot drift in how they set
  // source/domain/ids -- an earlier duplicated version diverged on empty-label
  // handling and emitted anonymous "Generic" bands.
  void EmitMarkerEvent(std::string label, uint64_t start_ns, uint64_t end_ns,
                       uint64_t tid);

  AnnotationMap* annotation_map() { return &annotation_map_; }

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  RocmTracer() = default;

  void HipApiEvent(const rocprofiler_record_header_t* hdr, RocmTracerEvent* ev);
  void KernelEvent(const rocprofiler_record_header_t* hdr, RocmTracerEvent* ev);
  void MemcpyEvent(const rocprofiler_record_header_t* hdr, RocmTracerEvent* ev);

 private:
  absl::Status InitProfiling(void* tool_data);
  // The HIP device number of an agent, or kInvalidDeviceId.
  uint32_t HipDeviceOf(uint64_t agent_handle) const;
  rocprofiler_agent_type_t AgentType(uint64_t agent_handle) const;
  // MEMORY_COPY callback. On ENTER of a peer copy, which runs on the issuing
  // thread while the copy's buffers are alive, stores the owners of its
  // addresses for MemcpyEvent().
  void MemoryCopyCallback(const rocprofiler_callback_tracing_record_t& record);

  // Set once by EnsureHipDeviceMap() and never freed, because the SDK's buffer
  // thread reads it, possibly during exit.
  absl::once_flag hip_device_map_once_;
  std::atomic<const HipDeviceMap*> hip_device_map_{nullptr};
  RocmTraceCollector* collector_{nullptr};
  absl::Mutex collector_mutex_;

  // Never held during an HSA call, or together with another lock.
  absl::Mutex peer_owners_mutex_;
  // Key: HSA agent handle. Value: the only SDK GPU agent at its PCI address.
  absl::flat_hash_map<uint64_t, std::optional<uint64_t>>
      sdk_gpu_agent_of_hsa_agent_ ABSL_GUARDED_BY(peer_owners_mutex_);
  struct PeerCopyOwners {
    std::optional<uint64_t> src_owner;
    std::optional<uint64_t> dst_owner;
    // Identical copies in one launch share a key.
    int count = 0;
  };
  // Key: {correlation id, source address, destination address}.
  using PeerCopyKey = std::tuple<uint64_t, uint64_t, uint64_t>;
  absl::flat_hash_map<PeerCopyKey, PeerCopyOwners> peer_owners_
      ABSL_GUARDED_BY(peer_owners_mutex_);
  // Peer copies placed without trusted owners in this session.
  std::atomic<uint64_t> peer_copy_fallbacks_{0};
  // Set once by InitProfiling(): the MEMORY_COPY callback looks up owners.
  bool peer_owner_lookup_{false};

  bool api_tracing_enabled_{false};
  bool activity_tracing_enabled_{false};

  AnnotationMap annotation_map_;

  // ROCTX range state lives in a thread_local stack in rocm_tracer.cc, not
  // here. roctx pushes and pops are thread-local by definition and the
  // rocprofiler callback runs synchronously on the calling thread, so no
  // shared structure is needed -- and the HIP API callback reads the current
  // label on every single HIP call, which must not touch a process-wide lock.
  // Keeping it thread_local also means no per-thread map entry outlives the
  // thread, and there is no lock to order against collector_mutex_.
  //
  // Session isolation is by generation instead of by clearing: Enable() bumps
  // this counter, and a pop whose frame carries an older generation is
  // discarded rather than emitted into the new session. Relaxed ordering is
  // sufficient -- a racing push either sees the old or the new value, and
  // either way the frame is consistently tagged and consistently judged.
  std::atomic<uint64_t> roctx_generation_{0};

 public:
  using kernel_symbol_data_t =
      rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

  struct ProfilerKernelInfo {
    std::string name;
    kernel_symbol_data_t data;
  };

  using kernel_info_map_t =
      absl::flat_hash_map<rocprofiler_kernel_id_t, ProfilerKernelInfo>;

  using agent_info_map_t =
      absl::flat_hash_map<uint64_t, rocprofiler_agent_v0_t>;

  using callback_name_info = rocprofiler::sdk::callback_name_info;

  rocprofiler_client_id_t* client_id_{nullptr};
  // Contexts ----------------------------------------------------------
  // for registering kernel names
  rocprofiler_context_id_t utility_context_{};
  // for buffered callback services
  rocprofiler_context_id_t context_{};
  rocprofiler_buffer_id_t buffer_{};
  // for HIP stream tracking (ROCPROFILER_CALLBACK_TRACING_HIP_STREAM)
  rocprofiler_context_id_t hip_stream_ctx_{};

  // Maps & misc -------------------------------------------------------
  kernel_info_map_t kernel_info_{};
  absl::Mutex kernel_lock_;

  callback_name_info name_info_;
  agent_info_map_t agents_;

 public:
  // Disable copy and move.
  RocmTracer(const RocmTracer&) = delete;
  RocmTracer& operator=(const RocmTracer&) = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
