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

#define ROCPROFILER_VAR_NAME_COMBINE(X, Y) X##Y
#define ROCPROFILER_VARIABLE(X, Y)         ROCPROFILER_VAR_NAME_COMBINE(X, Y)

#define ROCPROFILER_WARN(result)                                                                   \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << #result                     \
                      << " returned error code " << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__)    \
                      << ": " << status_msg << ". This is just a warning!" << std::endl;           \
        }                                                                                          \
    }

#define ROCPROFILER_CHECK(result)                                                                  \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" << __FILE__ << ":" << __LINE__ << "] " << #result                        \
                   << " failed with error code " << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__)    \
                   << " :: " << status_msg;                                                        \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) \
                      << ": " << status_msg << std::endl;                                          \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            VLOG(-1) << errmsg.str();                                                \
        }                                                                                          \
    }


namespace xla {
namespace profiler {

// ----------------------------------------------------------------------------
// Convenience aliases for rocprofiler types.
// ----------------------------------------------------------------------------
using kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t =
    std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
using kernel_name_map_t =
    std::unordered_map<rocprofiler_kernel_id_t, std::string>;
using rocprofiler::sdk::callback_name_info;
using agent_info_map_t =
    std::unordered_map<uint64_t, rocprofiler_agent_v0_t>;

using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
    // using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
    
rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
using buffer_name_info   = rocprofiler::sdk::buffer_name_info;
// rocprofiler_context_id_t      client_ctx       = {0};
// rocprofiler_buffer_id_t       client_buffer    = {};
// buffer_name_info              client_name_info = {};
// kernel_symbol_map_t           client_kernels   = {};

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
  switch (type) {
    case RocmTracerEventType::Kernel:
      return "Kernel";
    case RocmTracerEventType::MemcpyH2D:
      return "MemcpyH2D";
    case RocmTracerEventType::MemcpyD2H:
      return "MemcpyD2H";
    case RocmTracerEventType::MemcpyD2D:
      return "MemcpyD2D";
    case RocmTracerEventType::MemcpyP2P:
      return "MemcpyP2P";
    case RocmTracerEventType::MemcpyOther:
      return "MemcpyOther";
    case RocmTracerEventType::MemoryAlloc:
      return "MemoryAlloc";
    case RocmTracerEventType::MemoryFree:
      return "MemoryFree";
    case RocmTracerEventType::Memset:
      return "Memset";
    case RocmTracerEventType::Synchronization:
      return "Synchronization";
    case RocmTracerEventType::Generic:
      return "Generic";
    default:
      DCHECK(false);
      return "";
  }
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
  kernel_symbol_map_t kernel_info;
  kernel_name_map_t kernel_names;
  rocprofiler_buffer_id_t  client_buffer    = {};
  buffer_name_info         client_name_info = {};
  kernel_symbol_map_t      client_kernels   = {};

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
RocmTracer* RocmTracer::GetRocmTracerSingleton() {
  static auto* singleton = new RocmTracer();
  return singleton;
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

void
tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 callback_data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
            auto flush_status = rocprofiler_flush_buffer(g_shared->client_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                ROCPROFILER_CALL(flush_status, "buffer flush");
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            g_shared->client_kernels.emplace(data->kernel_id, *data);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // do not erase just in case a buffer callback needs this
            // client_kernels.erase(data->kernel_id);
        }
    }

    (void) user_data;
    (void) callback_data;
}

void
tool_tracing_callback(rocprofiler_context_id_t      context,
                      rocprofiler_buffer_id_t       buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         user_data,
                      uint64_t                      drop_count)
{
    // assert(user_data != nullptr);
    assert(drop_count == 0 && "drop count should be zero for lossless policy");

    if(num_headers == 0){
      VLOG(-1) << "rocprofiler invoked a buffer callback with no headers. this should never happen";
      return;
    } else if(headers == nullptr) {
      VLOG(-1) << "rocprofiler invoked a buffer callback with a null pointer to the "
                  "array of headers. this should never happen";
      return;
    }
    
    for(size_t i = 0; i < num_headers; ++i)
    {
      auto* tracer = RocmTracer::GetRocmTracerSingleton();
      if (!tracer || !tracer->collector()) {
        VLOG(-1) << "tool_tracing_callback called after collector teardown.";
        return;
      }

      auto* header = headers[i];

      auto get_name = [](const auto* _record) -> std::string_view {
        // try
        // {
            return g_shared->client_name_info.at(_record->kind, _record->operation);
        //} 
        /*
        catch(std::exception& e)
        {
            std::cerr << __FUNCTION__
                      << " threw an exception for buffer tracing kind=" << _record->kind
                      << ", operation=" << _record->operation << "\nException: " << e.what()
                      << std::flush;
            abort();
        }*/
        // return std::string_view{"??"};
    };

    if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
      header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API){
      auto* record =
          static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);
      auto info = std::stringstream{};
      info << "tid=" << record->thread_id << ", context=" << context.handle
          << ", buffer_id=" << buffer_id.handle
          << ", cid=" << record->correlation_id.internal
          << ", extern_cid=" << record->correlation_id.external.value
          << ", kind=" << record->kind << ", operation=" << record->operation
          << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
          << ", name=" << g_shared->client_name_info[record->kind][record->operation];

      if(record->start_timestamp > record->end_timestamp){
          auto msg = std::stringstream{};
          msg << "hip api: start > end (" << record->start_timestamp << " > "
              << record->end_timestamp
              << "). diff = " << (record->start_timestamp - record->end_timestamp);
          VLOG(-1) << "threw an exception " << msg.str() << "\n" << std::flush;
          // throw std::runtime_error{msg.str()};
      }

      RocmTracerEvent event;
      event.type = RocmTracerEventType::Kernel;
      event.source = RocmTracerEventSource::ApiCallback;
      event.domain = RocmTracerEventDomain::HIP_API;
      event.name = g_shared->client_name_info[record->kind][record->operation];
      event.annotation = g_shared->client_name_info[record->kind][record->operation];
      event.start_time_ns = record->start_timestamp;
      event.end_time_ns = record->end_timestamp;
      event.device_id = 0;
      event.correlation_id = record->correlation_id.internal;
      event.thread_id = record->thread_id;
      // event.stream_id = 2;
      tracer->collector()->AddEvent(std::move(event), false);

      VLOG(-1) << info.str();
    } else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
        header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
          auto* record =
              static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);

          auto info = std::stringstream{};

          auto kernel_id   = record->dispatch_info.kernel_id;
          auto kernel_name = (g_shared->client_kernels.count(kernel_id) > 0)
                                ? std::string_view{g_shared->client_kernels.at(kernel_id).kernel_name}
                                : std::string_view{"??"};

          info << "tid=" << record->thread_id << ", context=" << context.handle
              << ", buffer_id=" << buffer_id.handle
              << ", cid=" << record->correlation_id.internal
              << ", extern_cid=" << record->correlation_id.external.value
              << ", kind=" << record->kind << ", operation=" << record->operation
              << ", agent_id=" << record->dispatch_info.agent_id.handle
              << ", queue_id=" << record->dispatch_info.queue_id.handle
              << ", kernel_id=" << record->dispatch_info.kernel_id << ", kernel=" << kernel_name
              << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
              << ", private_segment_size=" << record->dispatch_info.private_segment_size
              << ", group_segment_size=" << record->dispatch_info.group_segment_size
              << ", workgroup_size=(" << record->dispatch_info.workgroup_size.x << ","
              << record->dispatch_info.workgroup_size.y << ","
              << record->dispatch_info.workgroup_size.z << "), grid_size=("
              << record->dispatch_info.grid_size.x << "," << record->dispatch_info.grid_size.y
              << "," << record->dispatch_info.grid_size.z << ")";

          if(record->start_timestamp > record->end_timestamp)
              VLOG(-1) << "kernel dispatch: start > end";

          RocmTracerEvent activity_event;
          activity_event.type = RocmTracerEventType::Kernel;
          activity_event.source = RocmTracerEventSource::Activity;
          activity_event.domain = RocmTracerEventDomain::HIP_OPS;
          activity_event.name = kernel_name; // Replace with actual kernel name
          activity_event.annotation = kernel_name;
          activity_event.start_time_ns = record->start_timestamp;
          activity_event.end_time_ns = record->end_timestamp;   // Adjusted duration
          activity_event.device_id = 0;
          activity_event.correlation_id = record->correlation_id.internal; // Matches API event
          activity_event.thread_id = record->thread_id; // Typically 0 for activity records
          activity_event.stream_id = 2; // Matches API event
          activity_event.kernel_info = {
            0,      // registers_per_thread (unknown, set to 0)
            0,      // static_shared_memory_usage (unknown, set to 0)
            0,      // dynamic_shared_memory_usage (from original)
            record->dispatch_info.workgroup_size.x,    // block_x (threads per block in X)
            record->dispatch_info.workgroup_size.y,      // block_y (1D block)
            record->dispatch_info.workgroup_size.z,      // block_z (1D block)
            record->dispatch_info.grid_size.x,    // grid_x (blocks in X)
            record->dispatch_info.grid_size.y,      // grid_y (1D grid)
            record->dispatch_info.grid_size.z,      // grid_z (1D grid)
            nullptr // func_ptr (from original)
        };
  
        tracer->collector()->AddEvent(std::move(activity_event), false);
          
          VLOG(-1) << info.str();
      } else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
          header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

            auto info = std::stringstream{};

            info << "tid=" << record->thread_id << ", context=" << context.handle
                << ", buffer_id=" << buffer_id.handle
                << ", cid=" << record->correlation_id.internal
                << ", extern_cid=" << record->correlation_id.external.value
                << ", kind=" << record->kind << ", operation=" << record->operation
                << ", src_agent_id=" << record->src_agent_id.handle
                << ", dst_agent_id=" << record->dst_agent_id.handle
                << ", direction=" << record->operation << ", start=" << record->start_timestamp
                << ", stop=" << record->end_timestamp << ", name=" << get_name(record);

            if(record->start_timestamp > record->end_timestamp)
                VLOG(-1) << "memory copy: start > end";

            RocmTracerEvent activity_event;
            activity_event.type = RocmTracerEventType::Kernel;
            activity_event.source = RocmTracerEventSource::Activity;
            activity_event.domain = RocmTracerEventDomain::HIP_OPS;
            activity_event.name = get_name(record); 
            activity_event.annotation = get_name(record);
            activity_event.start_time_ns = record->start_timestamp;
            activity_event.end_time_ns = record->end_timestamp;   // Adjusted duration
            activity_event.device_id = 0;
            activity_event.correlation_id = record->correlation_id.internal; // Matches API event
            activity_event.thread_id = record->thread_id; // Typically 0 for activity records
            activity_event.stream_id = 2; // Matches API event
            
            tracer->collector()->AddEvent(std::move(activity_event), false);

            VLOG(-1) << info.str();
      }
    
  }    
}

int RocmTracer::toolInit(rocprofiler_client_finalize_t fini_func, void* tool_data) {
  VLOG(-1) << "cj401 gather api names";
  // Gather API names
  g_shared->name_info = GetCallbackTracingNames();
  g_shared->client_name_info = rocprofiler::sdk::get_buffer_tracing_names();

  // Gather agent info
  for (const auto& agent : GetGpuDeviceAgents()) {
    VLOG(-1) <<"cj401 agent id = " << agent.id.handle << ", type = " << agent.type << ", name = " << (agent.name ? agent.name : "null");
    g_shared->agents[agent.id.handle] = agent;
  }

  // Utility context to gather code‑object info
  rocprofiler_create_context(&g_shared->utilityContext);

  // buffered tracing
  auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
    ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

  rocprofiler_configure_callback_tracing_service(g_shared->utilityContext,
    ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
    code_object_ops.data(),
    code_object_ops.size(),
    tool_code_object_callback,
    nullptr);

  rocprofiler_start_context(g_shared->utilityContext);
  VLOG(-1) << "cj401 rocprofiler start utilityContext";

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
  rocprofiler_stop_context(g_shared->utilityContext);
  g_shared->utilityContext.handle = 0;


  rocprofiler_stop_context(g_shared->context);
  // flush buffer here or in disable?

  g_shared->context.handle = 0;
  
}

void RocmTracer::Disable() {
  collector_->Flush();
  collector_ = nullptr;
  
  VLOG(-1) << "GpuTracer stopped"; 
}

//------------------------------------------------------------------------
void RocmTracer::code_object_callback(
  rocprofiler_callback_tracing_record_t record,
  rocprofiler_user_data_t* user_data,
  void* callback_data) {
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
    g_shared->kernel_info.emplace(data->kernel_id, *data);
    g_shared->kernel_names.emplace(data->kernel_id, demangle(data->kernel_name));
    VLOG(-1) << "cj401 registered kernel ID " << data->kernel_id << ": " << demangle(data->kernel_name);
  } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
    // FIXME: clear these?  At minimum need kernel names at shutdown, async
    // completion
    // g_shared->kernel_info.erase(data->kernel_id);
    // g_shared->kernel_names.erase(data->kernel_id);
  }
}
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
