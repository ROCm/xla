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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
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
#include "rocm/include/rocprofiler-sdk/intercept_table.h"
#include "rocm/include/rocprofiler-sdk/internal_threading.h"
#include "rocm/include/rocprofiler-sdk/registration.h"
#include "rocm/include/rocprofiler-sdk/rocprofiler.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_pm_sampler.h"
#include "xla/backends/profiler/gpu/rocm_pm_sampler_factory.h"
#include "xla/backends/profiler/gpu/rocm_pm_samples.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
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

absl::Status RocmTracer::Enable(
    const RocmTracerOptions& options, RocmTraceCollector* collector,
    const std::vector<std::unique_ptr<tensorflow::profiler::XPlane>>& xplanes,
    uint64_t start_gputime_ns) {
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
  api_tracing_enabled_ = true;
  activity_tracing_enabled_ = true;

  // PM sampling: the sampler was built (contexts started, config+service
  // registered) at InitProfiling time -- it must happen before HIP creates its
  // device queues. Here we only inject the xplane sink and start the sampling
  // threads.
  if (rocm_pm_sampler_) {
    auto process_samples = [&xplanes,
                            start_gputime_ns](RocmPmSamples* samples) {
      if (samples == nullptr) return;
      int device_id = samples->GetDeviceId();
      if (device_id < 0 || device_id >= static_cast<int>(xplanes.size()) ||
          !xplanes[device_id]) {
        LOG(ERROR) << "PM sample device id " << device_id << " out of range";
        return;
      }
      tensorflow::profiler::XPlane* xplane = xplanes[device_id].get();
      xplane->set_name(tsl::profiler::GpuPlaneName(device_id));
      tsl::profiler::XPlaneBuilder builder(xplane);
      samples->PopulateCounterLine(&builder, start_gputime_ns);
    };
    if (absl::Status s = rocm_pm_sampler_->StartSampler(process_samples);
        !s.ok()) {
      LOG(WARNING) << "Failed to start PM sampler: " << s;
    } else {
      pm_sampling_enabled_ = true;
    }
  }

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
  trace_event->kernel_info = KernelDetails{};

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
    // actually one needs to set the real type
    trace_event->type = RocmTracerEventType::MemcpyOther;
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
  // we do not know valid stream ID for memcpy
  // rec.stream_id.handle;
  trace_event->stream_id = RocmTracerEvent::kInvalidStreamId;
  trace_event->memcpy_info = MemcpyDetails{
      .num_bytes = rec.bytes,
      .destination = static_cast<uint32_t>(dst_gpu.id.handle),
      .async = false,
  };

  VLOG(2) << "copy bytes: " << trace_event->memcpy_info.num_bytes
          << " stream: " << trace_event->stream_id << " src_id "
          << trace_event->device_id << " dst_id "
          << trace_event->memcpy_info.destination;
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
  trace_event->stream_id = kinfo.queue_id.handle;
  trace_event->kernel_info = KernelDetails{
      .private_segment_size = kinfo.private_segment_size,
      .group_segment_size = kinfo.group_segment_size,
      .workgroup_x = kinfo.workgroup_size.x,
      .workgroup_y = kinfo.workgroup_size.y,
      .workgroup_z = kinfo.workgroup_size.z,
      .grid_x = kinfo.grid_size.x,
      .grid_y = kinfo.grid_size.y,
      .grid_z = kinfo.grid_size.z,
      .func_ptr = nullptr,
  };

  auto it = kernel_info_.find(kinfo.kernel_id);
  if (it != kernel_info_.end()) trace_event->name = it->second.name;
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

// Returns true if PM hardware-counter sampling was explicitly requested via the
// XLA_ROCM_PM_SAMPLE_COUNTERS env var (non-empty). This is the single gate that
// decides whether the plugin does any PM-sampling work at load time -- including
// interposing on the HSA runtime table. It must be checkable at
// rocprofiler_configure time (before HIP init), which an env var is.
static bool PmSamplingRequestedFromEnv() {
  const char* counters_env = std::getenv("XLA_ROCM_PM_SAMPLE_COUNTERS");
  return counters_env != nullptr && counters_env[0] != '\0';
}

void RocmTracer::MaybeInitPmSampler() {
  const char* counters_env = std::getenv("XLA_ROCM_PM_SAMPLE_COUNTERS");
  if (!PmSamplingRequestedFromEnv()) {
    return;  // PM sampling not requested.
  }
  if (gpu_agents_.empty()) {
    LOG(WARNING) << "XLA_ROCM_PM_SAMPLE_COUNTERS set but no GPU agents found";
    return;
  }

  RocmPmSamplerOptions pm_options;
  pm_options.enable = true;
  for (absl::string_view metric :
       absl::StrSplit(counters_env, ',', absl::SkipEmpty())) {
    pm_options.metrics.push_back(
        std::string(absl::StripAsciiWhitespace(metric)));
  }
  if (pm_options.metrics.empty()) {
    return;
  }
  if (const char* iv = std::getenv("XLA_ROCM_PM_SAMPLE_INTERVAL_US");
      iv != nullptr && iv[0] != '\0') {
    int us = std::atoi(iv);
    if (us > 0) pm_options.sample_interval_ns = static_cast<size_t>(us) * 1000;
  }

  // One counting context per GPU agent (a context profiles a single agent).
  pm_contexts_.assign(gpu_agents_.size(), rocprofiler_context_id_t{0});
  std::vector<rocprofiler_agent_id_t> agent_ids;
  agent_ids.reserve(gpu_agents_.size());
  for (size_t i = 0; i < gpu_agents_.size(); ++i) {
    if (rocprofiler_create_context(&pm_contexts_[i]) !=
        ROCPROFILER_STATUS_SUCCESS) {
      LOG(WARNING) << "Failed to create PM context for GPU " << i
                   << "; disabling PM sampling";
      pm_contexts_.clear();
      return;
    }
    agent_ids.push_back(gpu_agents_[i].id);
  }

  absl::StatusOr<std::unique_ptr<RocmPmSampler>> sampler_or =
      CreateRocmPmSampler(pm_contexts_, agent_ids, pm_options);
  if (!sampler_or.ok()) {
    LOG(WARNING) << "Failed to create PM sampler, continuing without hardware "
                 << "counters: " << sampler_or.status();
    return;
  }
  rocm_pm_sampler_ = std::move(sampler_or).value();
  LOG(INFO) << "ROCm PM sampling configured: " << pm_options.metrics.size()
            << " counters, interval " << pm_options.sample_interval_ns
            << "ns, on " << gpu_agents_.size() << " GPUs";
}

void RocmTracer::HsaTableRegistrationCallback(
    rocprofiler_intercept_table_t type, uint64_t /*lib_version*/,
    uint64_t /*lib_instance*/, void** /*tables*/, uint64_t /*num_tables*/,
    void* /*user_data*/) {
  if (type != ROCPROFILER_HSA_TABLE) {
    return;
  }
  // HSA is now loaded but HIP has not yet created its device queues. This is the
  // required moment to start the PM counting contexts (rocprofiler_start_context
  // needs HSA loaded, and rocprofiler installs the per-agent profile queue by
  // intercepting the upcoming HSA queue creation).
  auto& obj = RocmTracer::GetRocmTracerSingleton();
  if (obj.rocm_pm_sampler_ == nullptr) {
    return;  // PM sampling not configured.
  }
  if (absl::Status s = obj.rocm_pm_sampler_->StartContexts(); !s.ok()) {
    LOG(WARNING) << "(Profiling::PM Sampling) StartContexts failed: " << s;
  }
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

  // PM (hardware-counter) sampling must be set up HERE, before HIP creates its
  // device queues -- rocprofiler creates the per-agent profile queue by
  // intercepting HSA queue creation, and only does so if the counter config +
  // device-counting service are already registered and the context started.
  // Doing this lazily at trace-start (in Enable) yields "No profile queue is
  // available for this agent" and zero counter records.
  //
  // The counter list therefore cannot come from ProfileOptions
  // advanced_configuration (not available until start_trace, long after HIP is
  // up). It comes from the XLA_ROCM_PM_SAMPLE_COUNTERS env var instead, e.g.
  //   XLA_ROCM_PM_SAMPLE_COUNTERS=SQ_WAVES,GRBM_GUI_ACTIVE,GRBM_COUNT
  // Optional: XLA_ROCM_PM_SAMPLE_INTERVAL_US (default 1000).
  MaybeInitPmSampler();

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
  rocprofiler_stop_context(obj.context_);
  obj.context_.handle = 0;
  for (auto& pm_context : obj.pm_contexts_) {
    if (pm_context.handle != 0) {
      rocprofiler_stop_context(pm_context);
      pm_context.handle = 0;
    }
  }
}

void RocmTracer::Disable() {
  // Stop PM sampling before activity tracing (symmetry with the CUDA path).
  if (pm_sampling_enabled_ && rocm_pm_sampler_) {
    if (absl::Status s = rocm_pm_sampler_->StopSampler(); !s.ok()) {
      LOG(WARNING) << "Failed to stop PM sampler: " << s;
    }
    if (absl::Status s = rocm_pm_sampler_->Deinitialize(); !s.ok()) {
      LOG(WARNING) << "Failed to deinitialize PM sampler: " << s;
    }
    rocm_pm_sampler_.reset();
    pm_sampling_enabled_ = false;
  }

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

  // Register for the HSA API-table registration callback so we can start the PM
  // counting contexts at exactly the right moment (HSA loaded, HIP queues not
  // yet created). The SDK requires this to be registered from within
  // rocprofiler_configure; doing it later returns CONFIGURATION_LOCKED.
  //
  // Only register when PM sampling was explicitly requested. Registering the
  // HSA-table intercept is NOT free even if the callback no-ops: it inserts
  // rocprofiler-sdk into the HSA queue/dispatch path for the whole process,
  // which destabilizes HIP-graph (command-buffer) launch on ROCm 7.2.4 (flaky
  // SIGSEGV in libamdhip64 during RocmCommandBuffer::LaunchGraph). Gating on the
  // same env var as MaybeInitPmSampler keeps non-profiling runs off that path.
  if (PmSamplingRequestedFromEnv()) {
    if (rocprofiler_status_t rc = rocprofiler_at_intercept_table_registration(
            &RocmTracer::HsaTableRegistrationCallback, ROCPROFILER_HSA_TABLE,
            nullptr);
        rc != ROCPROFILER_STATUS_SUCCESS) {
      LOG(WARNING) << "(Profiling::PM Sampling) failed to register HSA table "
                   << "callback: " << rocprofiler_get_status_string(rc);
    }
  }

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
