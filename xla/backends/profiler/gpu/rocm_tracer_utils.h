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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_UTILS_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace xla {
namespace profiler {

// Mirrors the identical typedef in cupti_buffer_events.h for the CUPTI path.
// Both resolve to the same type; ROCm and CUPTI are never compiled together.
using ScopeRangeIdTree = absl::flat_hash_map<int64_t, int64_t>;

// Every field below carries a default member initializer so that a
// default-constructed detail record reads as "nothing measured" rather than as
// indeterminate stack residue. Reporting a truthful zero is recoverable; a
// plausible-looking garbage byte count is not.

struct MemcpyDetails {
  // The amount of data copied for memcpy events.
  size_t num_bytes = 0;
  // The destination device for peer-2-peer communication (memcpy). The source
  // device is implicit: it's the current device.
  uint32_t destination = 0;
  // Whether or not the memcpy is asynchronous.
  bool async = false;
};

struct MemAllocDetails {
  // The amount of data requested for cudaMalloc events.
  uint64_t num_bytes = 0;
};

struct MemsetDetails {
  // The number of memory elements getting set
  size_t num_bytes = 0;
  // Whether or not the memset is asynchronous.
  bool async = false;
};

struct KernelDetails {
  // The amount of private memory used by kernel,
  // number of register per thread (register spillage if > 0)
  uint32_t private_segment_size = 0;
  // The amount of shared memory (SMEM)
  uint32_t group_segment_size = 0;
  // X-dimension of a workgroup (grid.x*block.x)
  uint32_t workgroup_x = 0;
  // Y-dimension of a workgroup (grid.x*block.x)
  uint32_t workgroup_y = 0;
  // Z-dimension of a workgroup (grid.x*block.x)
  uint32_t workgroup_z = 0;
  // X-dimension of a grid.
  uint32_t grid_x = 0;
  // Y-dimension of a grid.
  uint32_t grid_y = 0;
  // Z-dimension of a grid.
  uint32_t grid_z = 0;

  // kernel address. Used for calculating core occupancy
  void* func_ptr = nullptr;
};

enum class RocmTracerEventType {
  Unsupported = 0,
  Kernel,
  MemcpyH2D,
  MemcpyD2H,
  MemcpyD2D,
  MemcpyP2P,
  MemcpyOther,
  MemoryAlloc,
  MemoryFree,
  Memset,
  Synchronization,
  Generic,
};

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type);

// What a HIP copy API asked for, recovered from its callback-tracing
// arguments.
//
// Neither buffered source can supply this. The buffered HIP API record carries
// no arguments at all, and ROCclr implements most hipMemcpy* calls as blit
// *kernel* dispatches rather than SDMA transfers, so no MEMORY_COPY activity
// record is emitted for them either -- which is why these copies used to
// report a size of zero (or, before the union was replaced, whatever a
// workgroup dimension happened to look like). Callback tracing is the only
// path that sees the typed arguments.
struct CopyApiDetails {
  // Direction as derived from the API id, or from the hipMemcpyKind argument
  // for the kind-taking variants. MemcpyOther when it cannot be determined.
  RocmTracerEventType type = RocmTracerEventType::MemcpyOther;
  MemcpyDetails details;
};

enum class RocmTracerEventSource {
  Invalid = 0,
  ApiCallback,
  Activity,
};

const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source);

enum class RocmTracerEventDomain {
  InvalidDomain = 0,
  HIP_API,
  HIP_OPS,
};

const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain);

// RocmTracerSyncTypes forward declaration
enum class RocmTracerSyncTypes;

struct SynchronizationDetails {
  RocmTracerSyncTypes sync_type = {};
};

// Per-event detail payload: exactly one alternative is active at a time, and
// which one is a property of the object rather than of the reader's
// expectations. std::monostate means no detail record was ever attached.
//
// This replaces an anonymous union that tracked no active member, so every
// read was a leap of faith. ApiActivityInfoExchange() would assign
// `memcpy_info` from an event whose live member was in fact KernelDetails,
// reinterpreting a workgroup dimension as a byte count and a shared-memory
// size as a device id -- undefined behaviour that surfaced as a plausible but
// fabricated "dest:512 async:1" in Trace Viewer. With a variant that same code
// is a checked access: the typed accessor returns nullptr and the caller is
// forced to decide what to do about it.
using RocmTracerEventDetails =
    std::variant<std::monostate, MemcpyDetails, MemsetDetails, MemAllocDetails,
                 KernelDetails, SynchronizationDetails>;

struct RocmTracerEvent {
  static constexpr uint32_t kInvalidDeviceId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidThreadId =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint32_t kInvalidCorrelationId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidStreamId =
      std::numeric_limits<uint64_t>::max();
  RocmTracerEventType type = RocmTracerEventType::Unsupported;
  RocmTracerEventSource source = RocmTracerEventSource::Invalid;
  RocmTracerEventDomain domain = RocmTracerEventDomain::InvalidDomain;
  std::string name;
  // This points to strings in AnnotationMap, which should outlive the point
  // where serialization happens.
  absl::string_view annotation;
  absl::string_view roctx_range;
  uint64_t start_time_ns = 0;
  uint64_t end_time_ns = 0;
  uint32_t device_id = kInvalidDeviceId;
  uint32_t correlation_id = kInvalidCorrelationId;
  uint64_t thread_id = kInvalidThreadId;
  uint64_t stream_id = kInvalidStreamId;
  uint64_t queue_id = 0;
  int64_t scope_range_id = 0;

  // The active alternative corresponds to `type`: MemcpyDetails for Memcpy*,
  // MemsetDetails for Memset, and so on. Prefer the accessors below to
  // touching this directly.
  RocmTracerEventDetails details;

  // Set on a Kernel event that is really a ROCclr blit copy (ROCm implements
  // most hipMemcpy* calls as a kernel dispatch rather than an SDMA transfer).
  // Deliberately outside the variant: such a dispatch is not a kernel *or* a
  // copy, it is both, and both sets of facts are worth exporting -- the real
  // grid geometry the GPU ran, and the direction and byte count the user asked
  // to move. `type` stays Kernel for these, so the direction has to travel
  // here rather than being read back off the event.
  std::optional<CopyApiDetails> blit_copy_info;

  // Defines the checked accessor triplet for one detail alternative:
  //   const T* name() const  -- nullptr unless T is the active alternative, so
  //                             a type mismatch is a null check rather than a
  //                             reinterpretation of unrelated bytes
  //   T& mutable_name()      -- makes T active (default-constructed if it was
  //                             not already) and returns it for field-by-field
  //                             population
  //   void set_name(T)       -- makes T active with the given value
#define ROCM_TRACER_EVENT_DETAIL_ACCESSORS(Type, name)                 \
  const Type* name() const { return std::get_if<Type>(&details); }     \
  Type& mutable_##name() {                                             \
    if (!std::holds_alternative<Type>(details)) {                      \
      details.emplace<Type>();                                         \
    }                                                                  \
    return std::get<Type>(details);                                    \
  }                                                                    \
  void set_##name(Type value) { details = std::move(value); }

  ROCM_TRACER_EVENT_DETAIL_ACCESSORS(MemcpyDetails, memcpy_info)
  ROCM_TRACER_EVENT_DETAIL_ACCESSORS(MemsetDetails, memset_info)
  ROCM_TRACER_EVENT_DETAIL_ACCESSORS(MemAllocDetails, memalloc_info)
  ROCM_TRACER_EVENT_DETAIL_ACCESSORS(KernelDetails, kernel_info)
  ROCM_TRACER_EVENT_DETAIL_ACCESSORS(SynchronizationDetails,
                                     synchronization_info)

#undef ROCM_TRACER_EVENT_DETAIL_ACCESSORS
};

struct RocmTraceCollectorOptions {
  // Maximum number of events to collect from callback API; if -1, no limit.
  // if 0, the callback API is enabled to build a correlation map, but no
  // events are collected.
  uint64_t max_callback_api_events;
  // Maximum number of events to collect from activity API; if -1, no limit.
  uint64_t max_activity_api_events;
  // Maximum number of annotation strings that we can accommodate.
  uint64_t max_annotation_strings;
  // Number of GPUs involved.
  uint32_t num_gpus;
};

class AnnotationMap {
 public:
  explicit AnnotationMap(uint64_t max_size) : max_size_(max_size) {}
  void Add(uint32_t correlation_id, const std::string& annotation,
           absl::Span<const int64_t> scope_range_ids = {});
  absl::string_view LookUp(uint32_t correlation_id);
  int64_t LookUpScopeRangeId(uint32_t correlation_id);
  ScopeRangeIdTree TakeScopeRangeIdTree();
  void Clear();

 private:
  struct AnnotationMapImpl {
    // The population/consumption of annotations might happen from multiple
    // callback/activity api related threads.
    absl::Mutex mutex;
    // Annotation tends to be repetitive, use a hash_set to store the strings,
    // an use the reference to the string in the map.
    absl::node_hash_set<std::string> annotations ABSL_GUARDED_BY(mutex);
    absl::flat_hash_map<uint32_t, absl::string_view> correlation_map
        ABSL_GUARDED_BY(mutex);
    absl::flat_hash_map<uint32_t, int64_t> scope_range_id_map
        ABSL_GUARDED_BY(mutex);
    ScopeRangeIdTree scope_range_id_tree ABSL_GUARDED_BY(mutex);
  };
  const uint64_t max_size_;
  AnnotationMapImpl map_;

 public:
  // Disable copy and move.
  AnnotationMap(const AnnotationMap&) = delete;
  AnnotationMap& operator=(const AnnotationMap&) = delete;
};

// Copy API arguments stashed on callback entry and rejoined to the buffered
// records by correlation id -- the buffered path never sees them itself.
class CopyInfoMap {
 public:
  explicit CopyInfoMap(uint64_t max_size) : max_size_(max_size) {}
  void Add(uint32_t correlation_id, const CopyApiDetails& copy_details);
  std::optional<CopyApiDetails> LookUp(uint32_t correlation_id);
  void Clear();

 private:
  const uint64_t max_size_;
  absl::Mutex mutex_;
  absl::flat_hash_map<uint32_t, CopyApiDetails> correlation_map_
      ABSL_GUARDED_BY(mutex_);

 public:
  // Disable copy and move.
  CopyInfoMap(const CopyInfoMap&) = delete;
  CopyInfoMap& operator=(const CopyInfoMap&) = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_UTILS_H_
