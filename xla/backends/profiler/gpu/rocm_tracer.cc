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

// ROCm profiler integration using rocprofiler-sdk.
// Provides RocmTracer singleton that manages rocprofiler contexts,
// buffer tracing, and callback services for GPU event collection.

#include "xla/backends/profiler/gpu/rocm_tracer.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "rocm/include/rocprofiler-sdk/agent.h"
#include "rocm/include/rocprofiler-sdk/buffer.h"
#include "rocm/include/rocprofiler-sdk/buffer_tracing.h"
#include "rocm/include/rocprofiler-sdk/callback_tracing.h"
#include "rocm/include/rocprofiler-sdk/context.h"
#include "rocm/include/rocprofiler-sdk/cxx/details/name_info.hpp"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "rocm/include/rocprofiler-sdk/hip/runtime_api_id.h"
#include "rocm/include/rocprofiler-sdk/internal_threading.h"
#include "rocm/include/rocprofiler-sdk/registration.h"
#include "rocm/include/rocprofiler-sdk/rocprofiler.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/platform/abi.h"

namespace xla {
namespace profiler {
namespace {

absl::Status RocprofilerStatusToAbslStatus(rocprofiler_status_t status) {
  if (ABSL_PREDICT_TRUE(status == ROCPROFILER_STATUS_SUCCESS)) {
    return absl::OkStatus();
  }
  const char* errstr = rocprofiler_get_status_string(status);
  return absl::InternalError(
      absl::StrCat("rocprofiler error: ", errstr ? errstr : "unknown"));
}

// Thread-local HIP stream stack. The rocprofiler-SDK fires
// ROCPROFILER_HIP_STREAM_SET callbacks around every HIP API call that uses a
// stream: PHASE_ENTER pushes the stream, PHASE_EXIT pops it. Between enter
// and exit, the external correlation callback snapshots the current stream.
// Initialized with 0 (the default HIP stream).
thread_local std::vector<uint64_t> tls_stream_stack = {0};

}  // namespace

using tsl::profiler::AnnotationStack;

// represents an invalid or uninitialized device ID used in RocmTracer events.
constexpr uint32_t RocmTracerEvent::kInvalidDeviceId;

inline auto GetCallbackTracingNames() {
  return rocprofiler::sdk::get_callback_tracing_names();
}

std::vector<rocprofiler_agent_v0_t> GetGpuDeviceAgents();

//-----------------------------------------------------------------------------
// copy api calls
bool isCopyApi(uint32_t id) {
  switch (id) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArrayAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArrayAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoA:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeer:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream:
      return true;
    default: {
    };
  }
  return false;
}

namespace {

// Maps the hipMemcpyKind argument of the direction-taking copy APIs onto our
// event types. hipMemcpyDefault asks the runtime to infer the direction from
// the pointer attributes, which we cannot replay here, and HostToHost never
// touches the GPU -- both stay MemcpyOther rather than guessing.
RocmTracerEventType MemcpyKindToEventType(hipMemcpyKind kind) {
  switch (kind) {
    case hipMemcpyHostToDevice:
      return RocmTracerEventType::MemcpyH2D;
    case hipMemcpyDeviceToHost:
      return RocmTracerEventType::MemcpyD2H;
    case hipMemcpyDeviceToDevice:
      return RocmTracerEventType::MemcpyD2D;
    default:
      return RocmTracerEventType::MemcpyOther;
  }
}

// Recovers direction and byte count from a HIP copy API's callback arguments.
//
// Each entry point has a differently shaped args struct, so this handles the
// variants that carry a plain `sizeBytes` and returns nullopt for the rest --
// the 2D/3D/array/symbol forms describe their extent through pitches,
// descriptors or symbol lookups, and inventing a number for them would be
// worse than reporting none. The covered set is what JAX/XLA actually issues.
std::optional<CopyApiDetails> ExtractCopyApiDetails(
    uint32_t operation, const rocprofiler_hip_api_args_t& args) {
  CopyApiDetails copy;
  switch (operation) {
    // Direction carried in a hipMemcpyKind argument.
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy:
      copy.type = MemcpyKindToEventType(args.hipMemcpy.kind);
      copy.details.num_bytes = args.hipMemcpy.sizeBytes;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
      copy.type = MemcpyKindToEventType(args.hipMemcpyAsync.kind);
      copy.details.num_bytes = args.hipMemcpyAsync.sizeBytes;
      copy.details.async = true;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream:
      copy.type = MemcpyKindToEventType(args.hipMemcpyWithStream.kind);
      copy.details.num_bytes = args.hipMemcpyWithStream.sizeBytes;
      copy.details.async = true;
      break;

    // Direction implied by the entry point itself.
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD:
      copy.type = RocmTracerEventType::MemcpyD2D;
      copy.details.num_bytes = args.hipMemcpyDtoD.sizeBytes;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
      copy.type = RocmTracerEventType::MemcpyD2D;
      copy.details.num_bytes = args.hipMemcpyDtoDAsync.sizeBytes;
      copy.details.async = true;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH:
      copy.type = RocmTracerEventType::MemcpyD2H;
      copy.details.num_bytes = args.hipMemcpyDtoH.sizeBytes;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
      copy.type = RocmTracerEventType::MemcpyD2H;
      copy.details.num_bytes = args.hipMemcpyDtoHAsync.sizeBytes;
      copy.details.async = true;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD:
      copy.type = RocmTracerEventType::MemcpyH2D;
      copy.details.num_bytes = args.hipMemcpyHtoD.sizeBytes;
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
      copy.type = RocmTracerEventType::MemcpyH2D;
      copy.details.num_bytes = args.hipMemcpyHtoDAsync.sizeBytes;
      copy.details.async = true;
      break;

    // Peer-to-peer: the only forms that name a destination device.
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeer:
      copy.type = RocmTracerEventType::MemcpyP2P;
      copy.details.num_bytes = args.hipMemcpyPeer.sizeBytes;
      copy.details.destination =
          static_cast<uint32_t>(args.hipMemcpyPeer.dstDeviceId);
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
      copy.type = RocmTracerEventType::MemcpyP2P;
      copy.details.num_bytes = args.hipMemcpyPeerAsync.sizeBytes;
      copy.details.destination =
          static_cast<uint32_t>(args.hipMemcpyPeerAsync.dstDeviceId);
      copy.details.async = true;
      break;

    default:
      // 2D/3D/array/symbol copies: extent is not a single sizeBytes field.
      return std::nullopt;
  }
  return copy;
}

}  // namespace

// ----------------------------------------------------------------------------
// Stub implementations for RocmTracer static functions expected by
// rocprofiler-sdk.
// ----------------------------------------------------------------------------
RocmTracer& RocmTracer::GetRocmTracerSingleton() {
  static RocmTracer obj;
  return obj;
}

bool RocmTracer::IsAvailable() const {
  return !activity_tracing_enabled_ && !api_tracing_enabled_;  // &&NumGpus()
}

/*static*/ uint64_t RocmTracer::GetTimestamp() {
  uint64_t ts;
  if (rocprofiler_get_timestamp(&ts) != ROCPROFILER_STATUS_SUCCESS) {
    LOG(ERROR) << "function rocprofiler_get_timestamp failed with error ";
    return 0;
  }
  return ts;
}

absl::Status RocmTracer::Enable(const RocmTracerOptions& options,
                                RocmTraceCollector* collector) {
  absl::MutexLock lock(collector_mutex_);
  if (collector_ != nullptr) {
    return absl::AlreadyExistsError("ROCM tracer is already running");
  }
  options_ = options;
  collector_ = collector;

  rocprofiler_status_t rc = rocprofiler_start_context(context_);
  if (rc != ROCPROFILER_STATUS_SUCCESS) {
    const char* errstr = rocprofiler_get_status_string(rc);
    options_ = {};
    collector_ = nullptr;
    return absl::InternalError(
        absl::StrCat("rocprofiler_start_context failed: ", errstr));
  }
  annotation_map_.Clear();
  copy_info_map_.Clear();
  api_tracing_enabled_ = true;
  activity_tracing_enabled_ = true;
  VLOG(1) << "GpuTracer started with number of GPUs = " << NumGpus();
  return absl::OkStatus();
}

void RocmTracer::HipApiEvent(const rocprofiler_record_header_t* hdr,
                             RocmTracerEvent* trace_event) {
  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_hip_api_record_t*>(
          hdr->payload);

  trace_event->type = RocmTracerEventType::Kernel;
  trace_event->source = RocmTracerEventSource::ApiCallback;
  trace_event->domain = RocmTracerEventDomain::HIP_API;
  trace_event->name = "??";
  trace_event->start_time_ns = rec.start_timestamp;
  trace_event->end_time_ns = rec.end_timestamp;
  trace_event->device_id = RocmTracerEvent::kInvalidDeviceId;
  trace_event->correlation_id = rec.correlation_id.internal;
  trace_event->annotation =
      annotation_map()->LookUp(trace_event->correlation_id);
  trace_event->scope_range_id =
      annotation_map()->LookUpScopeRangeId(trace_event->correlation_id);
  trace_event->thread_id = rec.thread_id;
  trace_event->stream_id = RocmTracerEvent::kInvalidStreamId;
  // A buffered HIP API record carries no dispatch geometry of its own; the
  // zeroed placeholder is replaced from the matching activity record in
  // ApiActivityInfoExchange().
  trace_event->set_kernel_info(KernelDetails{});

  {
    // bounds-check name table: kind and operation
    absl::MutexLock lock(kernel_lock_);
    const size_t kind = static_cast<size_t>(rec.kind);
    if (kind < name_info_.size()) {
      const auto& vec = name_info_[kind];
      const size_t op = static_cast<size_t>(rec.operation);
      if (op < vec.operations.size()) {
        trace_event->name = vec[op];
      } else {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
          LOG(ERROR) << "HIP op OOB: kind " << kind << " op = " << op
                     << " vec.size() = " << vec.operations.size();
        }
        trace_event->name = "HIP_UNKNOWN_OP";
      }
    } else {
      static std::atomic<int> once{0};
      if (once.fetch_add(1) == 0) {
        LOG(ERROR) << "HIP kind OOB: kind = " << kind
                   << " name_info_.size() = " << name_info_.size();
      }
      trace_event->name = "HIP_UNKNOWN_KIND";
    }
  }

  if (isCopyApi(rec.operation)) {
    // The buffered API record has no direction or size field of its own, so
    // recover both from the arguments the callback service saw on entry. Swap
    // the payload to match the new type so the two can never disagree --
    // leaving KernelDetails live under a Memcpy* type is what used to make
    // downstream reads reinterpret a workgroup dimension as a byte count.
    // MemcpyOther with a zero size remains the honest answer for the copy
    // shapes ExtractCopyApiDetails() declines to guess at.
    trace_event->type = RocmTracerEventType::MemcpyOther;
    trace_event->set_memcpy_info(MemcpyDetails{});
    if (auto copy = copy_info_map()->LookUp(trace_event->correlation_id)) {
      trace_event->type = copy->type;
      trace_event->set_memcpy_info(copy->details);
    }
  }
}

void RocmTracer::MemcpyEvent(const rocprofiler_record_header_t* hdr,
                             RocmTracerEvent* trace_event) {
  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_memory_copy_record_t*>(
          hdr->payload);

#define OO(src, target)                              \
  case ROCPROFILER_MEMORY_COPY_##src:                \
    trace_event->type = RocmTracerEventType::target; \
    trace_event->name = #target;                     \
    break;

  switch (rec.operation) {
    OO(NONE, MemcpyOther)
    OO(HOST_TO_HOST, MemcpyOther)
    OO(HOST_TO_DEVICE, MemcpyH2D)
    OO(DEVICE_TO_HOST, MemcpyD2H)
    OO(DEVICE_TO_DEVICE, MemcpyD2D)
    default:
      LOG(WARNING) << "Unexpected memcopy operation " << rec.operation;
      trace_event->type = RocmTracerEventType::MemcpyOther;
  }
#undef OO
  const auto &src_gpu = agents_[static_cast<uint32_t>(rec.src_agent_id.handle)],
             &dst_gpu = agents_[static_cast<uint32_t>(rec.dst_agent_id.handle)];

  // Assign device_id based on copy direction
  if (trace_event->type == RocmTracerEventType::MemcpyH2D &&
      dst_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
    trace_event->device_id = dst_gpu.id.handle;  // Destination is GPU
  } else if (trace_event->type == RocmTracerEventType::MemcpyD2H &&
             src_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
    trace_event->device_id = src_gpu.id.handle;  // Source is GPU
  } else if (trace_event->type == RocmTracerEventType::MemcpyD2D) {
    // Prefer destination GPU for D2D
    trace_event->device_id = dst_gpu.id.handle;
  } else {
    // Fallback for MemcpyOther or HOST_TO_HOST
    if (dst_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
      trace_event->device_id = dst_gpu.id.handle;
    } else if (src_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
      trace_event->device_id = src_gpu.id.handle;
    } else {
      LOG(WARNING) << "No GPU ID available for memory copy operation: "
                   << trace_event->name << ", src_agent_type=" << src_gpu.type
                   << ", dst_agent_type=" << dst_gpu.type;
      trace_event->device_id = 0;  // Invalid ID or default
    }
  }

  trace_event->source = RocmTracerEventSource::Activity;
  trace_event->domain = RocmTracerEventDomain::HIP_OPS;
  trace_event->start_time_ns = rec.start_timestamp;
  trace_event->end_time_ns = rec.end_timestamp;
  trace_event->correlation_id = rec.correlation_id.internal;
  trace_event->annotation =
      annotation_map()->LookUp(trace_event->correlation_id);
  trace_event->scope_range_id =
      annotation_map()->LookUpScopeRangeId(trace_event->correlation_id);
  trace_event->thread_id = rec.thread_id;
  trace_event->stream_id = rec.correlation_id.external.value;
  trace_event->set_memcpy_info(MemcpyDetails{
      .num_bytes = rec.bytes,
      .destination = static_cast<uint32_t>(dst_gpu.id.handle),
      .async = false,
  });

  VLOG(2) << "copy bytes: " << trace_event->memcpy_info()->num_bytes
          << " stream: " << trace_event->stream_id << " src_id "
          << trace_event->device_id << " dst_id "
          << trace_event->memcpy_info()->destination;
}

void RocmTracer::KernelEvent(const rocprofiler_record_header_t* hdr,
                             RocmTracerEvent* trace_event) {
  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
          hdr->payload);

  const auto& kinfo = rec.dispatch_info;
  trace_event->type = RocmTracerEventType::Kernel;
  trace_event->source = RocmTracerEventSource::Activity;
  trace_event->domain = RocmTracerEventDomain::HIP_OPS;
  trace_event->name = "??";
  trace_event->start_time_ns = rec.start_timestamp;
  trace_event->end_time_ns = rec.end_timestamp;
  trace_event->device_id = agents_[kinfo.agent_id.handle].id.handle;
  trace_event->correlation_id = rec.correlation_id.internal;
  trace_event->annotation =
      annotation_map()->LookUp(trace_event->correlation_id);
  trace_event->scope_range_id =
      annotation_map()->LookUpScopeRangeId(trace_event->correlation_id);
  trace_event->thread_id = rec.thread_id;
  trace_event->stream_id = rec.correlation_id.external.value;
  trace_event->queue_id = kinfo.queue_id.handle;
  trace_event->set_kernel_info(KernelDetails{
      .private_segment_size = kinfo.private_segment_size,
      .group_segment_size = kinfo.group_segment_size,
      .workgroup_x = kinfo.workgroup_size.x,
      .workgroup_y = kinfo.workgroup_size.y,
      .workgroup_z = kinfo.workgroup_size.z,
      .grid_x = kinfo.grid_size.x,
      .grid_y = kinfo.grid_size.y,
      .grid_z = kinfo.grid_size.z,
      .func_ptr = nullptr,
  });

  auto it = kernel_info_.find(kinfo.kernel_id);
  if (it != kernel_info_.end()) trace_event->name = it->second.name;

  // A dispatch whose correlation id belongs to a copy API is a ROCclr blit
  // kernel (__amd_rocclr_copyBuffer and friends) standing in for an SDMA
  // transfer. It stays a Kernel event -- the grid geometry above is real and
  // worth keeping -- but it also gets the byte count the user asked to move,
  // which no activity record for it would otherwise carry.
  //
  // Gating on the correlation id rather than on the kernel name is deliberate:
  // a recorded copy API for this id is definitive, whereas the blit kernel
  // names are ROCclr internals that have changed between ROCm releases.
  trace_event->blit_copy_info =
      copy_info_map()->LookUp(trace_event->correlation_id);
}

void RocmTracer::TracingCallback(rocprofiler_context_id_t context,
                                 rocprofiler_buffer_id_t buffer_id,
                                 rocprofiler_record_header_t** headers,
                                 size_t num_headers, uint64_t drop_count) {
  if (collector() == nullptr) {
    return;
  }
  if (num_headers == 0) {
    return;
  }
  assert(drop_count == 0 && "drop count should be zero for lossless policy");

  if (headers == nullptr) {
    LOG(ERROR)
        << "rocprofiler invoked a buffer callback with a null pointer to the "
           "array of headers. this should never happen";
    return;
  }

  for (size_t i = 0; i < num_headers; i++) {
    RocmTracerEvent event;
    auto header = headers[i];

    if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;

    switch (header->kind) {
      case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API:
        HipApiEvent(header, &event);
        break;

      case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
        KernelEvent(header, &event);
        break;

      case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
        MemcpyEvent(header, &event);
        break;

      default:
        continue;
    }  // switch

    absl::MutexLock lock(collector_mutex_);
    if (collector()) {
      collector()->AddEvent(std::move(event), false);
    }
  }  // for
}

void RocmTracer::CodeObjectCallback(
    rocprofiler_callback_tracing_record_t record, void* callback_data) {
  if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
      record.operation == ROCPROFILER_CODE_OBJECT_LOAD) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // mainly for debugging
      LOG(WARNING)
          << "Callback phase unload without registering kernel names ...";
    }
  } else if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
             record.operation ==
                 ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
      absl::MutexLock lock(kernel_lock_);
      kernel_info_.emplace(
          data->kernel_id,
          ProfilerKernelInfo{tsl::port::MaybeAbiDemangle(data->kernel_name),
                             *data});
    } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // FIXME: clear these?  At minimum need kernel names at shutdown, async
      // completion We don't erase it just in case a buffer callback still needs
      // this kernel_info_.erase(data->kernel_id);
    }
  }
}

static void code_object_callback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t* user_data,
                                 void* callback_data) {
  RocmTracer::GetRocmTracerSingleton().CodeObjectCallback(record,
                                                          callback_data);
}

static void tool_tracing_callback(rocprofiler_context_id_t context,
                                  rocprofiler_buffer_id_t buffer_id,
                                  rocprofiler_record_header_t** headers,
                                  size_t num_headers, void* user_data,
                                  uint64_t drop_count) {
  RocmTracer::GetRocmTracerSingleton().TracingCallback(
      context, buffer_id, headers, num_headers, drop_count);
}

// Callback for ROCPROFILER_CALLBACK_TRACING_HIP_STREAM events.
// Maintains the thread-local stream stack so the external correlation callback
// can snapshot the current HIP stream for each GPU operation.
static void hip_stream_callback(rocprofiler_callback_tracing_record_t record,
                                rocprofiler_user_data_t* /*user_data*/,
                                void* /*callback_data*/) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_STREAM) return;

  switch (record.operation) {
    case ROCPROFILER_HIP_STREAM_SET:
      if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        const auto* data =
            static_cast<const rocprofiler_callback_tracing_hip_stream_data_t*>(
                record.payload);
        tls_stream_stack.push_back(data->stream_id.handle);
      } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
        if (tls_stream_stack.size() > 1) {
          tls_stream_stack.pop_back();
        }
      }
      break;
    case ROCPROFILER_HIP_STREAM_CREATE:
    case ROCPROFILER_HIP_STREAM_DESTROY:
      break;
    default:
      break;
  }
}

// External correlation ID request callback. Invoked by rocprofiler-SDK for
// every kernel dispatch and memory copy to attach the current HIP stream_id.
static int stream_external_correlation_callback(
    rocprofiler_thread_id_t /*thread_id*/,
    rocprofiler_context_id_t /*context_id*/,
    rocprofiler_external_correlation_id_request_kind_t /*kind*/,
    rocprofiler_tracing_operation_t /*operation*/,
    uint64_t /*internal_corr_id_value*/,
    rocprofiler_user_data_t* external_corr_id_value, void* /*data*/) {
  external_corr_id_value->value = tls_stream_stack.back();
  return 0;
}

absl::Status RocmTracer::InitProfiling(void* tool_data) {
  name_info_ = GetCallbackTracingNames();

  // Build an ordered list of GPU agents for use by the profiler collector
  // (e.g. GetDeviceCapabilities).
  num_gpus_ = 0;
  gpu_agents_.clear();
  for (const auto& agent : GetGpuDeviceAgents()) {
    VLOG(1) << "agent id = " << agent.id.handle << ", dev = " << agent.device_id
            << ", name = " << (agent.name ? agent.name : "null");
    agents_[agent.id.handle] = agent;
    if (agent.type == ROCPROFILER_AGENT_TYPE_GPU) {
      gpu_agents_.push_back(agent);
      num_gpus_++;
    }
  }

  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_create_context(&utility_context_)));

  auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_configure_callback_tracing_service(
          utility_context_, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
          code_object_ops.data(), code_object_ops.size(), code_object_callback,
          nullptr)));

  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_start_context(utility_context_)));
  VLOG(1) << "rocprofiler start utilityContext";

  constexpr auto buffer_size_bytes = 100 * 4096;
  constexpr auto buffer_watermark_bytes = 40 * 4096;

  RETURN_IF_ERROR(
      RocprofilerStatusToAbslStatus(rocprofiler_create_context(&context_)));

  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(rocprofiler_create_buffer(
      context_, buffer_size_bytes, buffer_watermark_bytes,
      ROCPROFILER_BUFFER_POLICY_LOSSLESS, tool_tracing_callback, tool_data,
      &buffer_)));

  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_configure_buffer_tracing_service(
          context_, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0,
          buffer_)));

  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_configure_buffer_tracing_service(
          context_, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
          buffer_)));

  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_configure_buffer_tracing_service(
          context_, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0,
          buffer_)));

  // Configure external correlation ID request service on the main context.
  // This attaches the current HIP stream_id (from tls_stream_stack) to every
  // kernel dispatch and memory copy record via correlation_id.external.value.
  {
    rocprofiler_external_correlation_id_request_kind_t kinds[] = {
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
    };
    RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
        rocprofiler_configure_external_correlation_id_request_service(
            context_, kinds, 2, stream_external_correlation_callback,
            nullptr)));
  }

  // Create a dedicated context for HIP stream tracking callbacks.
  // This fires ROCPROFILER_HIP_STREAM_SET around every HIP API call that
  // uses a stream, maintaining the thread-local stream stack.
  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_create_context(&hip_stream_ctx_)));

  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_configure_callback_tracing_service(
          hip_stream_ctx_, ROCPROFILER_CALLBACK_TRACING_HIP_STREAM, nullptr, 0,
          hip_stream_callback, nullptr)));

  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_start_context(hip_stream_ctx_)));
  VLOG(1) << "rocprofiler start hip_stream_ctx";

  {
    const rocprofiler_tracing_operation_t* hip_ops = nullptr;
    size_t hip_ops_count = 0;

    RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
        rocprofiler_configure_callback_tracing_service(
            context_, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, hip_ops,
            hip_ops_count,
            [](rocprofiler_callback_tracing_record_t record,
               rocprofiler_user_data_t*, void*) {
              if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
                const std::string& annotation =
                    tsl::profiler::AnnotationStack::Get();
                if (!annotation.empty()) {
                  absl::Span<const int64_t> range_ids =
                      tsl::profiler::AnnotationStack::GetScopeRangeIds();
                  RocmTracer::GetRocmTracerSingleton().annotation_map()->Add(
                      record.correlation_id.internal, annotation, range_ids);
                }
                // Callback tracing is the only path that sees a copy API's
                // arguments; the buffered record it will later produce has
                // none. Stash the size and direction now, keyed by correlation
                // id, for HipApiEvent() and KernelEvent() to rejoin.
                if (isCopyApi(record.operation) && record.payload != nullptr) {
                  const auto& data = *static_cast<
                      const rocprofiler_callback_tracing_hip_api_data_t*>(
                      record.payload);
                  if (auto copy =
                          ExtractCopyApiDetails(record.operation, data.args)) {
                    RocmTracer::GetRocmTracerSingleton().copy_info_map()->Add(
                        record.correlation_id.internal, *copy);
                  }
                }
              }
            },
            nullptr)));
  }

  auto client_thread = rocprofiler_callback_thread_t{};
  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_create_callback_thread(&client_thread)));
  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_assign_callback_thread(buffer_, client_thread)));

  int isValid = 0;
  RETURN_IF_ERROR(RocprofilerStatusToAbslStatus(
      rocprofiler_context_is_valid(context_, &isValid)));
  if (isValid == 0) {
    context_.handle = 0;
    return absl::InternalError(
        "rocprofiler context is not valid after initialization");
  }

  return absl::OkStatus();
}

int RocmTracer::toolInit(rocprofiler_client_finalize_t fini_func,
                         void* tool_data) {
  absl::Status status = InitProfiling(tool_data);
  if (!status.ok()) {
    LOG(ERROR) << "RocmTracer initialization failed: " << status.message();
    return -1;
  }
  return 0;
}

void RocmTracer::toolFinalize(void* tool_data) {
  auto& obj = RocmTracer::GetRocmTracerSingleton();
  VLOG(1) << "Calling toolFinalize!";
  rocprofiler_stop_context(obj.utility_context_);
  obj.utility_context_.handle = 0;
  rocprofiler_stop_context(obj.hip_stream_ctx_);
  obj.hip_stream_ctx_.handle = 0;
  rocprofiler_stop_context(obj.context_);
  obj.context_.handle = 0;
}

void RocmTracer::Disable() {
  // Stop first so no new records enter the rocprofiler buffer; this pairs
  // with the rocprofiler_start_context() in Enable().
  rocprofiler_status_t status = rocprofiler_stop_context(context_);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    LOG(WARNING) << "rocprofiler_stop_context failed with error " << status;
  }

  status = rocprofiler_flush_buffer(buffer_);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    LOG(WARNING) << "rocprofiler_flush_buffer failed with error " << status;
  }
  absl::MutexLock lock(collector_mutex_);
  collector_->Flush();
  collector_ = nullptr;
  api_tracing_enabled_ = false;
  activity_tracing_enabled_ = false;
  VLOG(1) << "GpuTracer stopped";
}

// ----------------------------------------------------------------------------
// Helper that returns all device agents (GPU + CPU for now).
// ----------------------------------------------------------------------------
std::vector<rocprofiler_agent_v0_t> GetGpuDeviceAgents() {
  std::vector<rocprofiler_agent_v0_t> agents;

  rocprofiler_query_available_agents_cb_t iterate_cb =
      [](rocprofiler_agent_version_t agents_ver, const void** agents_arr,
         size_t num_agents, void* udata) {
        if (agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0) {
          LOG(ERROR) << "unexpected rocprofiler agent version: " << agents_ver;
          return ROCPROFILER_STATUS_ERROR;
        }
        auto* agents_vec =
            static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for (size_t i = 0; i < num_agents; ++i) {
          const auto* agent =
              static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
          agents_vec->push_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
      };

  rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                     iterate_cb, sizeof(rocprofiler_agent_t),
                                     static_cast<void*>(&agents));
  return agents;
}

static int toolInitStatic(rocprofiler_client_finalize_t finalize_func,
                          void* tool_data) {
  return RocmTracer::GetRocmTracerSingleton().toolInit(finalize_func,
                                                       tool_data);
}

// ----------------------------------------------------------------------------
// C‑linkage entry‑point expected by rocprofiler-sdk.
// ----------------------------------------------------------------------------
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t version, const char* runtime_version, uint32_t priority,
    rocprofiler_client_id_t* id) {
  auto& obj = RocmTracer::GetRocmTracerSingleton();  // Ensure constructed,
                                                     // critical for tracing.

  id->name = "XLA-with-rocprofiler-sdk";
  obj.client_id_ = id;

  VLOG(1) << "Configure rocprofiler-sdk...";

  const uint32_t major = version / 10000;
  const uint32_t minor = (version % 10000) / 100;
  const uint32_t patch = version % 100;

  VLOG(1) << absl::StrFormat(
      "%s Configure XLA with rocprofv3... (priority=%u) is using "
      "rocprofiler-sdk v%u.%u.%u (%s)",
      id->name, static_cast<unsigned>(priority), static_cast<unsigned>(major),
      static_cast<unsigned>(minor), static_cast<unsigned>(patch),
      runtime_version ? runtime_version : "unknown");

  static rocprofiler_tool_configure_result_t cfg{
      sizeof(rocprofiler_tool_configure_result_t), &toolInitStatic,
      &RocmTracer::toolFinalize, nullptr};

  return &cfg;
}

}  // namespace profiler
}  // namespace xla

void __attribute__((constructor)) init_rocm_lib() {
  rocprofiler_force_configure(xla::profiler::rocprofiler_configure);
}
