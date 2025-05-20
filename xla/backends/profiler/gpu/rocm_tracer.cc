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

  // Contexts ----------------------------------------------------------
  rocprofiler_context_id_t utilityContext{ 0 };
  rocprofiler_context_id_t context{ 0 };

  // Maps & misc -------------------------------------------------------
  kernel_symbol_map_t kernel_info;
  kernel_name_map_t kernel_names;
  std::mutex kernel_lock;

  callback_name_info name_info;
  agent_info_map_t agents;

  std::map<uint64_t, kernel_args> kernelargs;
  std::map<uint64_t, copy_args> copyargs;

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

void RocmTracer::Enable() {
  externalCorrelationEnabled_ = true;
  logging_ = true;
  if (g_shared != nullptr) {
    VLOG(-1) << "GpuTracer started";
    rocprofiler_start_context(g_shared->context);
  } else {
    VLOG(-1) << "GpuTracer failed to start due to rocprofiler_configure failure";
  }
}

int RocmTracer::toolInit(rocprofiler_client_finalize_t /*fini_func*/, void* /*tool_data*/) {
  VLOG(-1) << "cj401 gather api names";
  // Gather API names
  g_shared->name_info = GetCallbackTracingNames();

  // Gather agent info
  for (const auto& agent : GetGpuDeviceAgents()) {
    VLOG(-1) <<"cj401 agent id = " << agent.id.handle << ", type = " << agent.type << ", name = " << (agent.name ? agent.name : "null");
    g_shared->agents[agent.id.handle] = agent;
  }

  // Utility context to gather code‑object info
  rocprofiler_create_context(&g_shared->utilityContext);
  const std::vector<rocprofiler_tracing_operation_t> code_object_ops = {
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

  rocprofiler_configure_callback_tracing_service(
      g_shared->utilityContext, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
      code_object_ops.data(), code_object_ops.size(),
      &RocmTracer::code_object_callback, nullptr);

  int isValid = 0;
  rocprofiler_context_is_valid(g_shared->utilityContext, &isValid);
  if (isValid == 0) {
    g_shared->utilityContext.handle = 0;  // Leak on failure.
    return -1;
  }

  rocprofiler_start_context(g_shared->utilityContext);
  VLOG(-1) << "cj401 rocprofiler start context";

  // select some api calls to omit, in the most inconvenient way possible
  // #betterThanRoctracer
  RocprofApiIdList apiList(g_shared->name_info);
  apiList.setInvertMode(true); // Omit the specified api
  apiList.add("hipGetDevice");
  apiList.add("hipSetDevice");
  apiList.add("hipGetLastError");
  apiList.add("__hipPushCallConfiguration");
  apiList.add("__hipPopCallConfiguration");
  apiList.add("hipCtxSetCurrent");
  apiList.add("hipEventRecord");
  apiList.add("hipEventQuery");
  apiList.add("hipGetDeviceProperties");
  apiList.add("hipPeekAtLastError");
  apiList.add("hipModuleGetFunction");
  apiList.add("hipEventCreateWithFlags");

  // Get a vector of the enabled api calls
  auto apis = apiList.allEnabled();

  //
  // Setup main context to collect runtime and kernel info
  //
  rocprofiler_create_context(&g_shared->context);

  rocprofiler_configure_callback_tracing_service(
    g_shared->context,
    ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
    apis.data(),
    apis.size(),
    api_callback,
    nullptr);

  rocprofiler_configure_callback_tracing_service(
    g_shared->context,
    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
    nullptr,
    0,
    api_callback,
    nullptr);

  rocprofiler_configure_callback_tracing_service(
    g_shared->context,
    ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
    nullptr,
    0,
    api_callback,
    nullptr);
  
  {
    int isValid = 0;
    rocprofiler_context_is_valid(g_shared->context, &isValid);
    if (isValid == 0) {
      g_shared->context.handle = 0; // Can't destroy it, so leak it
      return -1;
    }
  }  
  rocprofiler_stop_context(g_shared->context);
  VLOG(-1) << "cj401 rocprofiler stop context...";
  return 0;
}

void RocmTracer::toolFinalize(void* tool_data) {
  rocprofiler_stop_context(g_shared->utilityContext);
  g_shared->utilityContext.handle = 0;
  rocprofiler_stop_context(g_shared->context);
  g_shared->context.handle = 0;
}

void RocmTracer::Disable() { LOG(INFO) << "GpuTracer stopped"; }

//------------------------------------------------------------------------
void RocmTracer::api_callback(
  rocprofiler_callback_tracing_record_t record,
  rocprofiler_user_data_t* user_data,
  void* callback_data) {
thread_local std::unordered_map<uint64_t, timespec> timestamps;

if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp); // record proper clock
    timestamps[record.correlation_id.internal] = timestamp;

    //---- Capture api args for copy and kernel apis
    // These will be used during dispatch and copy callbacks to complete
    // records
    if (isCopyApi(record.operation)) {
      auto& args = g_shared->copyargs[record.correlation_id.internal];
      rocprofiler_iterate_callback_tracing_kind_operation_args(
          record,
          extract_copy_args,
          1 /*max_deref*/
          ,
          &args);
      args.kind = record.kind;
      args.operation = record.operation;
    }
    if (isKernelApi(record.operation)) {
      auto& args = g_shared->kernelargs[record.correlation_id.internal];
      rocprofiler_iterate_callback_tracing_kind_operation_args(
          record,
          extract_kernel_args,
          1 /*max_deref*/
          ,
          &args);
      args.kind = record.kind;
      args.operation = record.operation;
    }
    //-----------------------------------------------

  } // ROCPROFILER_CALLBACK_PHASE_ENTER
  else { // ROCPROFILER_CALLBACK_PHASE_EXIT
    timespec startTime;
    startTime = timestamps[record.correlation_id.internal];
    timestamps.erase(record.correlation_id.internal);
    timespec endTime;
    clock_gettime(CLOCK_MONOTONIC, &endTime); // record proper clock

    // Kernel Launch Records
    if (isKernelApi(record.operation)) {
      // handled in dispatch callback
      g_shared->kernelargs.erase(record.correlation_id.internal);
    }
    // Copy Records
    else if (isCopyApi(record.operation)) {
      // handled in copy callback
      // FIXME: do not remove here.  Used after the async operation
      // DO it anyway, wait for crash,  async SDMA should assert below
      g_shared->copyargs.erase(record.correlation_id.internal);
    }
    // Malloc Records
    else if (isMallocApi(record.operation)) {
      malloc_args args;
      args.size = 0;
      rocprofiler_iterate_callback_tracing_kind_operation_args(
          record,
          extract_malloc_args,
          1 /*max_deref*/
          ,
          &args);
      // rVLOG(-1) << "cj401 record = " << record.kind << " ";
      /*
      rocprofMallocRow* row = new rocprofMallocRow(
          record.correlation_id.internal,
          record.kind,
          record.operation,
          processId(),
          systemThreadId(),
          timespec_to_ns(startTime),
          timespec_to_ns(endTime),
          args.ptr,
          args.size);
      insert_row_to_buffer(row);
      */
    }
    // Default Records
    else {
      /*
      rocprofRow* row = new rocprofRow(
          record.correlation_id.internal,
          record.kind,
          record.operation,
          processId(),
          systemThreadId(),
          timespec_to_ns(startTime),
          timespec_to_ns(endTime));
      insert_row_to_buffer(row);
      */
    }
  } // ROCPROFILER_CALLBACK_PHASE_EXIT
} // ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API

else if (record.kind == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) {
  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    ;
  } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
    auto& dispatch =
        *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(
            record.payload));
    auto& info = dispatch.dispatch_info;

    // Lookup the stream and apiname from the enclosing hip call.
    //  These are not provided in the dispatch record
    hipStream_t stream = 0;
    auto kind = record.kind;
    auto operation = record.operation;
    if (g_shared->kernelargs.count(record.correlation_id.internal) > 0) {
      // This row can be missing.  Some copy api dispatch kernels under the
      // hood
      auto& kargs = g_shared->kernelargs.at(record.correlation_id.internal);
      stream = kargs.stream;
      kind = kargs.kind;
      operation = kargs.operation;
    } else if (g_shared->copyargs.count(record.correlation_id.internal) > 0) {
      // Grab the stream from the copy row instead
      auto& cargs = g_shared->copyargs.at(record.correlation_id.internal);
      stream = cargs.stream;
      kind = cargs.kind;
      operation = cargs.operation;
    }

    // fetch up the timestamps
    timespec startTime;
    startTime = timestamps[record.correlation_id.internal];
    timespec endTime;
    clock_gettime(CLOCK_MONOTONIC, &endTime); // record proper clock
    /*
    rocprofKernelRow* row = new rocprofKernelRow(
        record.correlation_id.internal,
        kind,
        operation,
        processId(),
        systemThreadId(),
        timespec_to_ns(startTime),
        timespec_to_ns(endTime),
        nullptr,
        nullptr,
        info.workgroup_size.x,
        info.workgroup_size.y,
        info.workgroup_size.z,
        info.grid_size.x,
        info.grid_size.y,
        info.grid_size.z,
        info.group_segment_size,
        stream);
    insert_row_to_buffer(row);
    */
  } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_NONE) {
    // completion callback - runtime thread
    auto& dispatch =
        *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(
            record.payload));
    auto& info = dispatch.dispatch_info;

    std::lock_guard<std::mutex> lock(g_shared->kernel_lock);

    /*
    rocprofAsyncRow* row = new rocprofAsyncRow(
        record.correlation_id.internal,
        record.kind,
        record.operation,
        record.operation, // shared op - No longer a thing.  Placeholder
        g_shared->agents.at(info.agent_id.handle).logical_node_type_id,
        info.queue_id.handle,
        dispatch.start_timestamp,
        dispatch.end_timestamp,
        g_shared->kernel_names.at(info.kernel_id));
    insert_row_to_buffer(row);
    */
  }
} // ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH

else if (record.kind == ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY) {
  if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
    auto& copy =
        *(static_cast<rocprofiler_callback_tracing_memory_copy_data_t*>(
            record.payload));

    // Fetch args from the enclosing hip call
    // FIXME async?  May need to remove it here rather than above
    auto& args = g_shared->copyargs.at(record.correlation_id.internal);

    /*
    rocprofCopyRow* row = new rocprofCopyRow(
        record.correlation_id.internal,
        args.kind,
        args.operation,
        processId(),
        systemThreadId(),
        copy.start_timestamp,
        copy.end_timestamp,
        args.src,
        args.dst,
        args.size,
        args.copyKind,
        args.stream);
    insert_row_to_buffer(row);
    */
  }
} // ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY
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
    std::lock_guard<std::mutex> lock(g_shared->kernel_lock);
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

/*
extern "C" __attribute__((constructor)) __attribute__((used))
void InitRocprofilerOnLoad() {
  fprintf(stderr, "[XLA ROCm] InitRocprofilerOnLoad() triggered\n");

  static rocprofiler_client_id_t dummy_id = {};
  dummy_id.name = "XLA-dummy";

  // We pass 0 as version/priority and "" for runtime_version to satisfy API
  xla::profiler::rocprofiler_configure(0, "", 0, &dummy_id);
}
*/