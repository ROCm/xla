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

// This translation unit is **self‑contained**: it provides minimal stub
// implementations for the rocprofiler callbacks that XLA needs to register
// (toolInit / toolFinialize / code_object_callback).  They do nothing except
// keep the compiler and linker happy.  Once real logging is implemented, you
// can replace the stubs with the actual logic.

#include "xla/backends/profiler/gpu/rocm_tracer.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <chrono>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "rocm/rocm_config.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mem.h"


using tsl::profiler::XEventBuilder;
using tsl::profiler::XEventMetadata;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XSpace;

namespace xla {
namespace profiler {

// ----------------------------------------------------------------------------
// Convenience aliases for rocprofiler types.
// ----------------------------------------------------------------------------
using kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

struct ProfilerKernelInfo {
  std::string name;
  kernel_symbol_data_t data;
};

using kernel_info_map_t =
    std::unordered_map<rocprofiler_kernel_id_t, ProfilerKernelInfo>;

using rocprofiler::sdk::callback_name_info;
using agent_info_map_t =
    std::unordered_map<uint64_t, rocprofiler_agent_v0_t>;

using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
    
rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;

static constexpr int kMaxSymbolSize = 1024;

std::string demangle(const char* name) {
#ifndef _MSC_VER
  if (!name) {
    return "";
  }

  if (strlen(name) > kMaxSymbolSize) {
    return name;
  }

  int status;
  size_t len = 0;
  char* demangled = abi::__cxa_demangle(name, nullptr, &len, &status);
  if (status != 0) {
    return name;
  }
  std::string res(demangled);
  // The returned buffer must be freed!
  free(demangled);
  return res;
#else
  // TODO: demangling on Windows
  if (!name) {
    return "";
  } else {
    return name;
  }
#endif
}

std::string demangle(const std::string& name) {
  return demangle(name.c_str());
}

namespace {

const char *RocProfBufferCategory(/*rocprofiler_buffer_category_t*/int32_t c) {
#define OO(x) case ROCPROFILER_BUFFER_CATEGORY_##x: return #x;
  switch(c) {
    OO(NONE)
    OO(TRACING)
    OO(PC_SAMPLING)
    OO(COUNTERS)
  }
  return "unknown category";
#undef OO
}

const char *RocProfBufferKind(/*rocprofiler_buffer_tracing_kind_t*/int32_t c) {
#define OO(x) case ROCPROFILER_BUFFER_TRACING_##x: return #x;
  switch(c) {
    OO(NONE)
    OO(HSA_CORE_API)
    OO(HSA_AMD_EXT_API)
    OO(HSA_IMAGE_EXT_API)
    OO(HSA_FINALIZE_EXT_API)
    OO(HIP_RUNTIME_API)
    OO(HIP_COMPILER_API)
    OO(MARKER_CORE_API)
    OO(MARKER_CONTROL_API)
    OO(MARKER_NAME_API)
    OO(MEMORY_COPY)
    OO(KERNEL_DISPATCH)
    OO(PAGE_MIGRATION)
    OO(SCRATCH_MEMORY)
    OO(CORRELATION_ID_RETIREMENT)
    OO(RCCL_API)
    OO(OMPT)
    OO(MEMORY_ALLOCATION)
    OO(RUNTIME_INITIALIZATION)
    OO(ROCDECODE_API)
    // OO(ROCJPEG_API)
    // OO(HIP_STREAM)
    // OO(HIP_RUNTIME_API_EXT)
    // OO(HIP_COMPILER_API_EXT)
    // OO(ROCDECODE_API_EXT)
  }
  return "unknown buffer tracing kind";
#undef OO
}

const char *RocProfCallbackKind(/*rocprofiler_callback_tracing_kind_t*/int32_t c) {
#define OO(x) case ROCPROFILER_CALLBACK_TRACING_##x: return #x;
  switch(c) {
    OO(NONE)
    OO(HSA_CORE_API)
    OO(HSA_AMD_EXT_API)
    OO(HSA_IMAGE_EXT_API)
    OO(HSA_FINALIZE_EXT_API)
    OO(HIP_RUNTIME_API)
    OO(HIP_COMPILER_API)
    OO(MARKER_CORE_API)
    OO(MARKER_CONTROL_API)
    OO(MARKER_NAME_API)
    OO(CODE_OBJECT)
    OO(SCRATCH_MEMORY)
    OO(KERNEL_DISPATCH)
    OO(MEMORY_COPY)
    OO(RCCL_API)
    OO(OMPT)
    OO(MEMORY_ALLOCATION)
    OO(RUNTIME_INITIALIZATION)
    OO(ROCDECODE_API)
    // OO(ROCJPEG_API)
    // OO(HIP_STREAM)
  }
  return "unknown callback tracing kind";
#undef OO
}

const char *RocProfCodeObjOperation(/*rocprofiler_code_object_operation_t*/int32_t c) {
#define OO(x) case ROCPROFILER_CODE_OBJECT_##x: return #x;
  switch(c) {
    OO(NONE)
    OO(LOAD) //  Code object containing kernel symbols
    OO(DEVICE_KERNEL_SYMBOL_REGISTER) // Kernel symbols - Device
    OO(HOST_KERNEL_SYMBOL_REGISTER)   // Kernel symbols - Host
  }
  return "unknown code object operation";
#undef OO
}

} // namespace

//----------------------------------------------------------------------------
ApiIdList::ApiIdList() : invert_(true) {}

void ApiIdList::add(const std::string& apiName) {
  uint32_t cid = mapName(apiName);
  if (cid > 0)
    filter_[cid] = 1;
}

void ApiIdList::remove(const std::string& apiName) {
  uint32_t cid = mapName(apiName);
  if (cid > 0)
    filter_.erase(cid);
}

bool ApiIdList::loadUserPrefs() {
  // FIXME: check an ENV variable that points to an exclude file
  return false;
}

bool ApiIdList::contains(uint32_t apiId) {
  return (filter_.find(apiId) != filter_.end()) ? !invert_ : invert_; // XOR
}

// ----------------------------------------------------------------------------
// RocprofApiIdList – thin wrapper that maps API name → operation id.
// ----------------------------------------------------------------------------
class RocprofApiIdList : public ApiIdList {
  public:
   explicit RocprofApiIdList(callback_name_info& names);  // Defined elsewhere.
   uint32_t mapName(const std::string& apiName) override;  // Implemented elsewhere.
   std::vector<rocprofiler_tracing_operation_t> allEnabled();
 
  private:
   std::unordered_map<std::string, size_t> nameMap_;
 };

//-----------------------------------------------------------------------------
//
// ApiIdList
//   Jump through some extra hoops
//
//
RocprofApiIdList::RocprofApiIdList(callback_name_info& names) : nameMap_() {
  auto& hipapis =
      names[ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API].operations;

  for (size_t i = 0; i < hipapis.size(); ++i) {
    nameMap_.emplace(hipapis[i], i);
  }
}

uint32_t RocprofApiIdList::mapName(const std::string& apiName) {
  auto it = nameMap_.find(apiName);
  if (it != nameMap_.end()) {
    return it->second;
  }
  return 0;
}

std::vector<rocprofiler_tracing_operation_t> RocprofApiIdList::allEnabled() {
  std::vector<rocprofiler_tracing_operation_t> oplist;
  for (auto& it : nameMap_) {
    if (contains(it.second))
      oplist.push_back(it.second);
  }
  return oplist;
}

// ----------------------------------------------------------------------------
// Forward declarations / helpers
// ----------------------------------------------------------------------------
class RocprofLoggerShared;  // Defined later inside this namespace.

inline auto GetCallbackTracingNames() {
  return rocprofiler::sdk::get_callback_tracing_names();
}
std::vector<rocprofiler_agent_v0_t> GetGpuDeviceAgents();

// ----------------------------------------------------------------------------
// Global pointer that rocprofiler callbacks can dereference.
// ----------------------------------------------------------------------------
RocprofLoggerShared* g_shared = nullptr;  // NOLINT

// ----------------------------------------------------------------------------
// Utility structs & extraction lambdas for HIP memcpy / kernel launch callbacks
// ----------------------------------------------------------------------------
struct copy_args {
  const char* dst{ "" };
  const char* src{ "" };
  size_t size{ 0 };
  const char* copyKindStr{ "" };
  hipMemcpyKind copyKind{ hipMemcpyDefault };
  hipStream_t stream{ nullptr };
  rocprofiler_callback_tracing_kind_t kind{};
  rocprofiler_tracing_operation_t operation{};
};
auto ExtractCopyArgs = [](rocprofiler_callback_tracing_kind_t,
                          rocprofiler_tracing_operation_t,
                          uint32_t /*arg_num*/,
                          const void* const arg_value_addr,
                          int32_t /*indirection_count*/,
                          const char* /*arg_type*/,
                          const char* arg_name,
                          const char* arg_value_str,
                          int32_t /*dereference_count*/,
                          void* cb_data) -> int {
  auto& args = *static_cast<copy_args*>(cb_data);
  if (std::strcmp("dst", arg_name) == 0) {
    args.dst = arg_value_str;
  } else if (std::strcmp("src", arg_name) == 0) {
    args.src = arg_value_str;
  } else if (std::strcmp("sizeBytes", arg_name) == 0) {
    args.size = *reinterpret_cast<const size_t*>(arg_value_addr);
  } else if (std::strcmp("kind", arg_name) == 0) {
    args.copyKindStr = arg_value_str;
    args.copyKind = *reinterpret_cast<const hipMemcpyKind*>(arg_value_addr);
  } else if (std::strcmp("stream", arg_name) == 0) {
    args.stream = *reinterpret_cast<const hipStream_t*>(arg_value_addr);
  }
  return 0;
};

struct kernel_args {
  hipStream_t stream{ nullptr };
  rocprofiler_callback_tracing_kind_t kind{};
  rocprofiler_tracing_operation_t operation{};
};
auto extract_kernel_args = [](rocprofiler_callback_tracing_kind_t,
  rocprofiler_tracing_operation_t,
  uint32_t arg_num,
  const void* const arg_value_addr,
  int32_t indirection_count,
  const char* arg_type,
  const char* arg_name,
  const char* arg_value_str,
  int32_t dereference_count,
  void* cb_data) -> int {
if (strcmp("stream", arg_name) == 0) {
auto& args = *(static_cast<kernel_args*>(cb_data));
// args.stream = arg_value_str;
args.stream = *(reinterpret_cast<const hipStream_t*>(arg_value_addr));
}
return 0;
};

//-----------------------------------------------------------------------------
auto extract_copy_args = [](rocprofiler_callback_tracing_kind_t,
                            rocprofiler_tracing_operation_t,
                            uint32_t arg_num,
                            const void* const arg_value_addr,
                            int32_t indirection_count,
                            const char* arg_type,
                            const char* arg_name,
                            const char* arg_value_str,
                            int32_t dereference_count,
                            void* cb_data) -> int {
  auto& args = *(static_cast<copy_args*>(cb_data));
  if (strcmp("dst", arg_name) == 0) {
    args.dst = arg_value_str;
  } else if (strcmp("src", arg_name) == 0) {
    args.src = arg_value_str;
  } else if (strcmp("sizeBytes", arg_name) == 0) {
    args.size = *(reinterpret_cast<const size_t*>(arg_value_addr));
  } else if (strcmp("kind", arg_name) == 0) {
    args.copyKindStr = arg_value_str;
    args.copyKind = *(reinterpret_cast<const hipMemcpyKind*>(arg_value_addr));
  } else if (strcmp("stream", arg_name) == 0) {
    args.stream = *(reinterpret_cast<const hipStream_t*>(arg_value_addr));
  }
  return 0;
};

// extract malloc args
struct malloc_args {
  const char* ptr;
  size_t size;
};
auto extract_malloc_args = [](rocprofiler_callback_tracing_kind_t,
                              rocprofiler_tracing_operation_t,
                              uint32_t arg_num,
                              const void* const arg_value_addr,
                              int32_t indirection_count,
                              const char* arg_type,
                              const char* arg_name,
                              const char* arg_value_str,
                              int32_t dereference_count,
                              void* cb_data) -> int {
  auto& args = *(static_cast<malloc_args*>(cb_data));
  if (strcmp("ptr", arg_name) == 0) {
    args.ptr = arg_value_str;
  }
  if (strcmp("size", arg_name) == 0) {
    args.size = *(reinterpret_cast<const size_t*>(arg_value_addr));
  }
  return 0;
};

//-----------------------------------------------------------------------------
const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source) {
  switch (source) {
    case RocmTracerEventSource::ApiCallback:
      return "ApiCallback";
      break;
    case RocmTracerEventSource::Activity:
      return "Activity";
      break;
    case RocmTracerEventSource::Invalid:
      return "Invalid";
      break;
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

// FIXME(rocm-profiler): These domain names are not consistent with the
// GetActivityDomainName function
const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain) {
  switch (domain) {
    case RocmTracerEventDomain::HIP_API:
      return "HIP_API";
      break;
    case RocmTracerEventDomain::HIP_OPS:
      return "HIP_OPS";
      break;
    default:
      VLOG(3) << "RocmTracerEventDomain::InvalidDomain";
      DCHECK(false);
      return "";
  }
  return "";
}

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type) {
#define OO(x)  case RocmTracerEventType::x: return #x;
  switch (type) {
    OO(Kernel)
    OO(MemcpyH2D)
    OO(MemcpyD2H)
    OO(MemcpyD2D)
    OO(MemcpyP2P)
    OO(MemcpyOther)
    OO(MemoryAlloc)
    OO(MemoryFree)
    OO(Memset)
    OO(Synchronization)
    OO(Generic)
    default:;
  }
#undef OO
  DCHECK(false);
  return "";
}

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
      break;
    default:;
  }
  return false;
}

// kernel api calls
bool isKernelApi(uint32_t id) {
  switch (id) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel_spt:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel_spt:
      return true;
      break;
    default:;
  }
  return false;
}

// malloc api calls
bool isMallocApi(uint32_t id) {
  switch (id) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipFree:
      return true;
      break;
    default:;
  }
  return false;
}

// ----------------------------------------------------------------------------
// Shared singleton that holds all state needed by the profiler callbacks.
// ----------------------------------------------------------------------------
class RocprofLoggerShared {
 public:
  static RocprofLoggerShared& Singleton();

  // using NameInfo = absl::flat_hash_map<
  //   std::tuple< rocprofiler_buffer_tracing_kind_t,
                

  rocprofiler_client_id_t* clientId{ nullptr };
  rocprofiler_tool_configure_result_t cfg = {
      sizeof(rocprofiler_tool_configure_result_t),
      &RocmTracer::toolInit,
      &RocmTracer::toolFinalize, 
      nullptr };

  // XPlaneBuilder host_plane;

  // Contexts ----------------------------------------------------------
  rocprofiler_context_id_t utilityContext{0};
  rocprofiler_context_id_t context{0};
  rocprofiler_buffer_id_t buffer{0};

  // Maps & misc -------------------------------------------------------
  kernel_info_map_t kernel_info;
  tsl::mutex kernel_lock;

  callback_name_info name_info;
  agent_info_map_t agents;

  std::map<uint64_t, kernel_args> kernelargs;
  std::map<uint64_t, copy_args> copyargs;
  tsl::mutex copyargs_lock;

 private:
  RocprofLoggerShared() { g_shared = this; }
  ~RocprofLoggerShared() { g_shared = nullptr; }
};

RocprofLoggerShared& RocprofLoggerShared::Singleton() {
  static auto* instance = new RocprofLoggerShared;  // Intentional leak.
  return *instance;
}

// ----------------------------------------------------------------------------
// Aliases / convenience to avoid long namespace names.
// ----------------------------------------------------------------------------
namespace se = ::stream_executor;
using tsl::profiler::AnnotationStack;

constexpr uint32_t RocmTracerEvent::kInvalidDeviceId;

#define RETURN_IF_ROCTRACER_ERROR(expr)                                     \
  do {                                                                      \
    roctracer_status_t status = expr;                                       \
    if (status != ROCTRACER_STATUS_SUCCESS) {                               \
      const char* errstr = roctracer_error_string();              \
      LOG(ERROR) << "function " << #expr << " failed with error " << errstr; \
      return tsl::errors::Internal(                                         \
          absl::StrCat("roctracer call error", errstr));                  \
    }                                                                       \
  } while (false)

// ----------------------------------------------------------------------------
// Stub implementations for RocmTracer static functions expected by rocprofiler.
// ----------------------------------------------------------------------------
RocmTracer& RocmTracer::i() {
  static RocmTracer obj;
  return obj;
}

bool RocmTracer::IsAvailable() const {
  return !activity_tracing_enabled_ && !api_tracing_enabled_;  // &&NumGpus()
}

int RocmTracer::NumGpus() {
  static int num_gpus = []() -> int {
    if (hipInit(0) != hipSuccess) {
      return 0;
    }
    int gpu_count;
    if (hipGetDeviceCount(&gpu_count) != hipSuccess) {
      return 0;
    }
    LOG(INFO) << "Profiler found " << gpu_count << " GPUs";
    return gpu_count;
  }();
  return num_gpus;
}

/*static*/ uint64_t RocmTracer::GetTimestamp() {
  uint64_t ts;
  if (rocprofiler_get_timestamp(&ts) != ROCPROFILER_STATUS_SUCCESS) {
    // const char* errstr = rocprofiler_error_string();
    VLOG(-1) << "function rocprofiler_get_timestamp failed with error ";
              // << errstr;
    // Return 0 on error.
    return 0;
  }
  return ts;
}

void RocmTracer::Enable(const RocmTracerOptions& options,
  RocmTraceCollector* collector) {
  // options_ = options;
  collector_ = collector;
  VLOG(-1) << "cj401 collector_ = " << collector_;
  if (g_shared != nullptr) {
    VLOG(-1) << "GpuTracer started";
    rocprofiler_start_context(g_shared->context);
  } else {
    VLOG(-1) << "GpuTracer failed to start due to rocprofiler_configure failure";
  }
  VLOG(-1) << "GpuTracer started";
}

void RocmTracer::HipApiEvent(const rocprofiler_record_header_t *hdr,
        RocmTracerEvent *ev) { 

  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_hip_api_record_t*>(hdr->payload);
  
  ev->type = RocmTracerEventType::Kernel;
  ev->source = RocmTracerEventSource::ApiCallback;
  ev->domain = RocmTracerEventDomain::HIP_API;
  ev->name = "??";
  ev->annotation = "??";
  ev->roctx_range = "??";
  ev->start_time_ns = rec.start_timestamp;
  ev->end_time_ns = rec.end_timestamp;
  ev->device_id = RocmTracerEvent::kInvalidDeviceId;
  ev->correlation_id = rec.correlation_id.internal;
  ev->thread_id = rec.thread_id;
  ev->stream_id = RocmTracerEvent::kInvalidStreamId;
  ev->kernel_info = KernelDetails{
  };

  std::lock_guard<tsl::mutex> lock(g_shared->kernel_lock);
  if (static_cast< size_t >(rec.kind) < g_shared->name_info.size()) {
    auto& vec = g_shared->name_info[rec.kind];
    //if (static_cast< size_t >(rec.operation) < vec.size()) {
    ev->name = vec[rec.operation];
   // }
  }
  if (isCopyApi(rec.operation)) {
    // actually one needs to set the real type
    ev->type = RocmTracerEventType::MemcpyOther;
  }

  if (isKernelApi(rec.operation)) {

  }
}

void RocmTracer::MemcpyEvent(const rocprofiler_record_header_t *hdr,
        RocmTracerEvent *ev) { 

  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_memory_copy_record_t*>(hdr->payload);

#define OO(src, target) \
  case ROCPROFILER_MEMORY_COPY_##src: \
    ev->type = RocmTracerEventType::target; \
    ev->name = #target; \
  break; 

  switch(rec.operation) {
  OO(NONE, MemcpyOther)
  OO(HOST_TO_HOST, MemcpyOther)
  OO(HOST_TO_DEVICE, MemcpyH2D)
  OO(DEVICE_TO_HOST, MemcpyD2H)
  OO(DEVICE_TO_DEVICE, MemcpyD2D)
  default: 
    LOG(WARNING) << "Unexpected memcopy operation " << rec.operation;
    ev->type = RocmTracerEventType::MemcpyOther; 
  }
#undef OO
  auto src_id = rec.src_agent_id.handle,
       dst_id = rec.dst_agent_id.handle;

  ev->source = RocmTracerEventSource::Activity;
  ev->domain = RocmTracerEventDomain::HIP_OPS;
  ev->annotation = "??";
  ev->roctx_range = "??";
  ev->start_time_ns = rec.start_timestamp;
  ev->end_time_ns = rec.end_timestamp;
  ev->device_id = src_id;
  ev->correlation_id = rec.correlation_id.internal;
  ev->thread_id = rec.thread_id;
  ev->stream_id = 0; // we do not know valid stream ID for memcpy
  ev->memcpy_info = MemcpyDetails{
    .num_bytes = rec.bytes,
    .destination = dst_id,
    .async = false,
  };

  if (src_id != dst_id) {
    std::lock_guard<tsl::mutex> lock(g_shared->kernel_lock);
    if (src_id < g_shared->agents.size() && dst_id < g_shared->agents.size() &&
        g_shared->agents[src_id].type == ROCPROFILER_AGENT_TYPE_GPU &&
        g_shared->agents[dst_id].type == ROCPROFILER_AGENT_TYPE_GPU) {
      ev->type = RocmTracerEventType::MemcpyP2P; 
    }
  }
}

void RocmTracer::KernelEvent(const rocprofiler_record_header_t *hdr,
    RocmTracerEvent *ev) {

  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(hdr->payload);

  const auto& kinfo = rec.dispatch_info;
  ev->type = RocmTracerEventType::Kernel;
  ev->source = RocmTracerEventSource::Activity;
  ev->domain = RocmTracerEventDomain::HIP_OPS;
  ev->name = "??";
  ev->annotation = "??";
  ev->roctx_range = "??";
  ev->start_time_ns = rec.start_timestamp;
  ev->end_time_ns = rec.end_timestamp;
  ev->device_id = (uint32_t)kinfo.agent_id.handle;
        // kinfo.dispatch_id ??
  ev->correlation_id = rec.correlation_id.internal;
  ev->thread_id = rec.thread_id;
  ev->stream_id = kinfo.queue_id.handle;
  ev->kernel_info = KernelDetails{
    .registers_per_thread = 0,
    .static_shared_memory_usage = 0,
    .dynamic_shared_memory_usage = 0,
    .block_x = kinfo.workgroup_size.x,
    .block_y = kinfo.workgroup_size.y,
    .block_z = kinfo.workgroup_size.z,
    .grid_x = kinfo.grid_size.x,
    .grid_y = kinfo.grid_size.y,
    .grid_z = kinfo.grid_size.z,
    .func_ptr = nullptr,
  };

  std::lock_guard<tsl::mutex> lock(g_shared->kernel_lock);
  auto it = g_shared->kernel_info.find(kinfo.kernel_id);
  if (it != g_shared->kernel_info.end()) ev->name = it->second.name;

  // VLOG(0) << "Kernel: device: " << ev->device_id 
  //             << " stream: " << ev->stream_id
  //             << " corr: " << ev->correlation_id
  //             << " dispatch: " << rec.dispatch_info.dispatch_id
  //             << " name: " << ev->name;
}

void RocmTracer::TracingCallback(rocprofiler_context_id_t context,
                      rocprofiler_buffer_id_t buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t num_headers, uint64_t drop_count) {

  if (collector() == nullptr) return;  
  assert(drop_count == 0 && "drop count should be zero for lossless policy");

  if (headers == nullptr) {
    VLOG(-1) << "rocprofiler invoked a buffer callback with a null pointer to the "
                  "array of headers. this should never happen";
    return;
  }

  for (size_t i = 0; i < num_headers; i++)
  {
    RocmTracerEvent event;
    auto header = headers[i];
    // VLOG(0) << "category: " << RocProfBufferCategory(h->category)
    //         << " kind: " << RocProfBufferKind(h->kind);

    if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
    
    switch(header->kind) {
    case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API: 
      HipApiEvent(header, &event);
      break;

    case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH: 
      KernelEvent(header, &event);
      break;

    case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY: 
      MemcpyEvent(header, &event);
      break;

    default: continue;
    } // switch

    std::lock_guard<tsl::mutex> lk(collector_mutex_);
    if (collector()) {
      collector()->AddEvent(std::move(event), false);
    }
  } // for 
}

void RocmTracer::CodeObjectCallback(rocprofiler_callback_tracing_record_t record,
                          void* callback_data) {
  
  // VLOG(0) << "code_object_callback kind: " << RocProfCallbackKind(record.kind)
  //         << " op: " << RocProfCodeObjOperation(record.operation);

  if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
      record.operation == ROCPROFILER_CODE_OBJECT_LOAD) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // flush the buffer to ensure that any lookups for the client kernel names
      // for the code object are completed NOTE: not using buffer ATM
    }
  } else if (
      record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
      record.operation ==
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
      std::lock_guard<tsl::mutex> lock(g_shared->kernel_lock);
      g_shared->kernel_info.emplace(data->kernel_id, 
          ProfilerKernelInfo{demangle(data->kernel_name), *data});
    } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // FIXME: clear these?  At minimum need kernel names at shutdown, async
      // completion
      // g_shared->kernel_info.erase(data->kernel_id);
      // g_shared->kernel_names.erase(data->kernel_id);
    }
  }
}

//------------------------------------------------------------------------
static void code_object_callback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t* user_data,
    void* callback_data) {

  RocmTracer::i().CodeObjectCallback(record, callback_data);
}

static void
tool_tracing_callback(rocprofiler_context_id_t      context,
                      rocprofiler_buffer_id_t       buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         user_data,
                      uint64_t                      drop_count) {
  RocmTracer::i().TracingCallback(context, buffer_id, headers,
            num_headers, drop_count);
}


int RocmTracer::toolInit(rocprofiler_client_finalize_t fini_func, void* tool_data) {

  // Gather API names
  g_shared->name_info = GetCallbackTracingNames();

  // Gather agent info
  for (const auto& agent : GetGpuDeviceAgents()) {
    VLOG(-1) <<"agent id = " << agent.id.handle 
             << ", type = " << agent.type << ", name = " << (agent.name ? agent.name : "null");
    g_shared->agents[agent.id.handle] = agent;
  }

  // Utility context to gather code‑object info
  rocprofiler_create_context(&g_shared->utilityContext);

  // buffered tracing
  auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
    ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

  rocprofiler_configure_callback_tracing_service(
    g_shared->utilityContext,
    ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
    code_object_ops.data(),
    code_object_ops.size(),
    code_object_callback,
    nullptr);

  rocprofiler_start_context(g_shared->utilityContext);
  VLOG(-1) << "rocprofiler start utilityContext";

  constexpr auto buffer_size_bytes = 4096;
  constexpr auto buffer_watermark_bytes = buffer_size_bytes - (buffer_size_bytes / 8);

  // Utility context to gather code‑object info
  rocprofiler_create_context(&g_shared->context);

  rocprofiler_create_buffer(g_shared->context,
    buffer_size_bytes,
    buffer_watermark_bytes,
    ROCPROFILER_BUFFER_POLICY_LOSSLESS,
    tool_tracing_callback,
    tool_data,
    &g_shared->buffer);

  rocprofiler_configure_buffer_tracing_service(
    g_shared->context, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0, g_shared->buffer);

  rocprofiler_configure_buffer_tracing_service(
      g_shared->context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, g_shared->buffer);

  rocprofiler_configure_buffer_tracing_service(
        g_shared->context, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, g_shared->buffer);

  auto client_thread = rocprofiler_callback_thread_t{};
  rocprofiler_create_callback_thread(&client_thread);
    
  rocprofiler_assign_callback_thread(g_shared->buffer, client_thread);

  int isValid = 0;
  rocprofiler_context_is_valid(g_shared->context, &isValid);
  if (isValid == 0) {
    g_shared->context.handle = 0;  // Leak on failure.
    return -1;
  }

  rocprofiler_start_context(g_shared->context);
  VLOG(-1) << "cj401 rocprofiler start context";
 
  rocprofiler_stop_context(g_shared->context);
  VLOG(-1) << "cj401 rocprofiler stop context...";
  return 0;
}

void RocmTracer::toolFinalize(void* tool_data) {

  VLOG(0) << "Calling toolFinalize!";
  rocprofiler_stop_context(g_shared->utilityContext);
  g_shared->utilityContext.handle = 0;
  rocprofiler_stop_context(g_shared->context);
  // flush buffer here or in disable?
  g_shared->context.handle = 0;
}

void RocmTracer::Disable() {

  std::lock_guard<tsl::mutex> lk(collector_mutex_);
  collector_->Flush();
  collector_ = nullptr;
  VLOG(-1) << "GpuTracer stopped"; 
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
        auto* agents_vec = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for (size_t i = 0; i < num_agents; ++i) {
          const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
          agents_vec->push_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
      };

  rocprofiler_query_available_agents(
      ROCPROFILER_AGENT_INFO_VERSION_0, iterate_cb, sizeof(rocprofiler_agent_t),
      static_cast<void*>(&agents));
  return agents;
}

// ----------------------------------------------------------------------------
// C‑linkage entry‑point expected by rocprofiler-sdk.
// ----------------------------------------------------------------------------
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t version, const char* runtime_version, uint32_t priority,
    rocprofiler_client_id_t* id) {
  RocprofLoggerShared::Singleton();  // Ensure constructed, critical for tracing.

  id->name = "XLA-with-rocprofiler-sdk";
  g_shared->clientId = id;

  std::cerr << "Configure rocprofiler-sdk..." << std::endl << std::flush;

  const uint32_t major = version / 10000;
  const uint32_t minor = (version % 10000) / 100;
  const uint32_t patch = version % 100;

  std::stringstream info;
  info << id->name << " Configure XLA with rocprofv3... (priority=" << priority
       << ") is using rocprofiler-sdk v" << major << '.' << minor << '.'
       << patch << " (" << runtime_version << ')';
  std::cerr << info.str() << std::endl << std::flush;

  return &g_shared->cfg;
}

}  // namespace profiler
}  // namespace xla
