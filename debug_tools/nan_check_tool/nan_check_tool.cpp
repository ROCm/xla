// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/**
 * @file nan_check_tool.cpp
 *
 * @brief Sets DX10_CLAMP = 0 and EXCP_EN.INVALID = 1 bits of MODE register for
 * each kernel.
 * hipcc -fpic -g3 -fvisibility=hidden -fno-exceptions -shared \
 * -lrocprofiler-sdk nan_check_tool.cpp -o nan_check_tool.so
 * ROCP_TOOL_LIBRARIES=$PWD/nan_check_tool.so NAN_CHECK_VERBOSE=1 rocgdb --args
 * ./my_program
 */

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

#define ROCPROFILER_VAR_NAME_COMBINE(X, Y) X##Y
#define ROCPROFILER_VARIABLE(X, Y) ROCPROFILER_VAR_NAME_COMBINE(X, Y)

#define ROCPROFILER_CALL(result, msg)                                          \
  {                                                                            \
    rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result; \
    if (ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) !=                         \
        ROCPROFILER_STATUS_SUCCESS) {                                          \
      std::string status_msg = rocprofiler_get_status_string(                  \
          ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));                        \
      std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] "     \
                << msg << " failed with error code "                           \
                << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) << ": "         \
                << status_msg << std::endl;                                    \
      std::abort();                                                            \
    }                                                                          \
  }

#define HSA_CHECK(cmd)                                                       \
  {                                                                          \
    hsa_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = (cmd);        \
    if (ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != HSA_STATUS_SUCCESS) { \
      const char* _err_str = nullptr;                                        \
      hsa_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__),         \
                        &_err_str);                                          \
      std::cerr << "HSA error: " << (_err_str ? _err_str : "Unknown error")  \
                << " in " << #cmd << " at " << __FILE__ << ":" << __LINE__   \
                << std::endl;                                                \
      std::abort();                                                          \
    }                                                                        \
  }

#define HSA_CHECK_ITER(cmd)                                                \
  {                                                                        \
    hsa_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = (cmd);      \
    if (ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) !=                     \
        HSA_STATUS_INFO_BREAK) {                                           \
      const char* _err_str = nullptr;                                      \
      hsa_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__),       \
                        &_err_str);                                        \
      std::cerr << "HSA error: "                                           \
                << (ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) ==         \
                            HSA_STATUS_SUCCESS                             \
                        ? "Iterator condition failed"                      \
                        : (_err_str ? _err_str : "Unknown error"))         \
                << " in " << #cmd << " at " << __FILE__ << ":" << __LINE__ \
                << std::endl;                                              \
      std::abort();                                                        \
    }                                                                      \
  }

#define HIP_CHECK(cmd)                                                  \
  {                                                                     \
    hipError_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = cmd;       \
    if (ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != hipSuccess) {    \
      std::cerr << "HIP error: "                                        \
                << hipGetErrorString(                                   \
                       ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__))     \
                << " (" << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__)  \
                << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
      std::abort();                                                     \
    }                                                                   \
  }

namespace client {
namespace {
using code_obj_load_data_t =
    rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

struct alignas(32) kernel_descriptor_t {
  uint32_t group_segment_fixed_size;
  uint32_t private_segment_fixed_size;
  uint32_t kernarg_size;
  uint8_t reserved0[4];
  int64_t kernel_code_entry_byte_offset;
  uint8_t reserved1[20];
  uint32_t compute_pgm_rsrc3;  // GFX10+ and GFX90A+
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint16_t kernarg_preload;
  uint8_t reserved3[4];
};

constexpr int32_t kTrampolineSize = 256;
constexpr int32_t kTrampolineIsaSize = 8;
bool is_verbose_on = false;
bool is_inf_check_on = false;

struct alignas(kTrampolineSize) Trampoline {
  uint32_t isa[kTrampolineIsaSize];
  Trampoline* next;
  Trampoline* page_next;
};

// This is slow beacuse nodes are accessed over pcie
// But hopefuly it is lightweight enough to outweight that
struct TrampolineList {
  Trampoline* head = nullptr;
};

struct Allocator {
  std::mutex mutex;
  TrampolineList free_list;
  hsa_amd_memory_pool_t pool;
  TrampolineList page_list;
  int32_t gpu_id;
  ~Allocator() {
    size_t count = 0;
    for (auto next = page_list.head; next != nullptr;) {
      auto status =
          hsa_amd_memory_pool_free(std::exchange(next, next->page_next));
      if (status == HSA_STATUS_ERROR_NOT_INITIALIZED) {
        break;
      }
      HSA_CHECK(status);
      ++count;
    }
    if (is_verbose_on) {
      std::cerr << "Freed " << count << " pages on GPU " << gpu_id << "\n";
    }
  }
  inline Trampoline* AllocateToList(TrampolineList& alloc_list);
  inline void DeallocateFromList(TrampolineList alloc_list);
};

struct CodeObjectData {
  Allocator* allocator;
  TrampolineList alloc_list;
};

struct ToolContext {
  std::shared_mutex mutex;
  std::map<uint64_t, CodeObjectData> code_object_map;
  std::vector<std::pair<hsa_agent_t, std::unique_ptr<Allocator>>> allocators;
  inline void OnCodeObjectLoad(const code_obj_load_data_t& data);
  inline void OnCodeObjectUnload(const code_obj_load_data_t& data);
  inline void OnKernelSymbolRegister(const kernel_symbol_data_t& data);
};

inline Trampoline* Allocator::AllocateToList(TrampolineList& alloc_list) {
  std::unique_lock lock(mutex);

  if (__builtin_expect(free_list.head != nullptr, 1)) {
    auto* res = std::exchange(free_list.head, free_list.head->next);
    res->next = alloc_list.head;
    alloc_list.head = res;
    return res;
  }

  constexpr int32_t kPageSize = 4 * 1024;
  Trampoline* page;

  HSA_CHECK(hsa_amd_memory_pool_allocate(pool, kPageSize,
                                         HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG,
                                         reinterpret_cast<void**>(&page)));

  page->page_next = page_list.head;

  for (int i = 1; i < kPageSize / kTrampolineSize - 1; i++) {
    page[i].next = &page[i + 1];
  }
  page[kPageSize / kTrampolineSize - 1].next = nullptr;

  page_list.head = page;

  free_list.head = &page[1];

  page->next = alloc_list.head;

  alloc_list.head = page;

  return page;
}

inline void Allocator::DeallocateFromList(TrampolineList alloc_list) {
  if (__builtin_expect(alloc_list.head == nullptr, 0)) {
    return;
  }

  auto head = alloc_list.head;
  auto tail = alloc_list.head;

  while (tail->next) {
    tail = tail->next;
  }

  {
    std::unique_lock lock(mutex);
    tail->next = free_list.head;
    free_list.head = head;
  }
}

inline void ToolContext::OnCodeObjectLoad(const code_obj_load_data_t& data) {
  std::unique_lock lock(mutex);
  auto* allocator = [&]() -> Allocator* {
    auto it =
        std::find_if(allocators.begin(), allocators.end(), [&](const auto& kv) {
          return kv.first.handle == data.hsa_agent.handle;
        });
    if (__builtin_expect(it != allocators.end(), 1)) {
      return it->second.get();
    }

    HSA_CHECK_ITER(hsa_agent_iterate_isas(
        data.hsa_agent,
        +[](hsa_isa_t isa, void* data) -> hsa_status_t {
          char name[128];
          hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name);
          return (std::strstr(name, "gfx9") != nullptr) ? HSA_STATUS_INFO_BREAK
                                                        : HSA_STATUS_SUCCESS;
        },
        nullptr));

    auto allocator = std::make_unique<Allocator>();
    HSA_CHECK_ITER(hsa_amd_agent_iterate_memory_pools(
        data.hsa_agent,
        +[](hsa_amd_memory_pool_t pool, void* data) -> hsa_status_t {
          if (!data) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

          auto* pool_ptr = static_cast<hsa_amd_memory_pool_t*>(data);
          hsa_amd_segment_t segment;
          HSA_CHECK(hsa_amd_memory_pool_get_info(
              pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));

          if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

          uint32_t flag;
          HSA_CHECK(hsa_amd_memory_pool_get_info(
              pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag));
          if (flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
            return HSA_STATUS_SUCCESS;
          }
          *(pool_ptr) = pool;
          return HSA_STATUS_INFO_BREAK;
        },
        &allocator->pool));

    HIP_CHECK(hipGetDevice(&allocator->gpu_id));

    allocators.emplace_back(data.hsa_agent, std::move(allocator));
    return allocators.back().second.get();
  }();

  code_object_map.emplace(data.code_object_id, CodeObjectData{allocator});
}
inline void ToolContext::OnCodeObjectUnload(const code_obj_load_data_t& data) {
  CodeObjectData co_data;
  {
    std::unique_lock lock(mutex);
    auto it = code_object_map.find(data.code_object_id);
    assert(it != code_object_map.end());
    co_data = it->second;
    code_object_map.erase(it);
  }
  co_data.allocator->DeallocateFromList(co_data.alloc_list);
}

inline void ToolContext::OnKernelSymbolRegister(
    const kernel_symbol_data_t& data) {
  constexpr char kAmdPrefix[] = "__amd_";
  if (std::strncmp(data.kernel_name, kAmdPrefix, sizeof(kAmdPrefix) - 1) == 0) {
    return;
  }

  auto& co_data = [&]() -> CodeObjectData& {
    std::shared_lock lock(mutex);
    auto it = code_object_map.find(data.code_object_id);
    assert(it != code_object_map.end());
    return it->second;
  }();

  auto* trampoline = co_data.allocator->AllocateToList(co_data.alloc_list);
  auto* kd = reinterpret_cast<kernel_descriptor_t*>(data.kernel_object);

  uintptr_t kernel_entry =
      reinterpret_cast<uintptr_t>(kd) + kd->kernel_code_entry_byte_offset;

  uint32_t compute_pgm_rsrc1 = kd->compute_pgm_rsrc1;
  bool kernarg_preload = kd->kernarg_preload != 0;

  constexpr int32_t kKernargPreloadSkip = 256;

  if (kernarg_preload) {
    kernel_entry += kKernargPreloadSkip;
  }

  int i = 0;

  uint32_t isa[kTrampolineIsaSize];

  if (is_inf_check_on) {
    isa[i++] = 0xBA003A01;  // s_setreg_imm32_b32 hwreg(HW_REG_MODE, 8, 8), 0b110010010
    isa[i++] = 0x00000192;
  } else {
    isa[i++] = 0xBA003201;  // s_setreg_imm32_b32 hwreg(HW_REG_MODE, 8, 7), 0b000010010
    isa[i++] = 0x00000012;
  }
  isa[i++] = 0xBE9600FF;  // s_mov_b32 s22, kernel_entry_lo32
  isa[i++] = static_cast<uint32_t>(kernel_entry);
  isa[i++] = 0xBE9700FF;  // s_mov_b32 s23, kernel_entry_hi32
  isa[i++] = static_cast<uint32_t>(kernel_entry >> 32);
  // isa[i++] = 0xBF920003;  // s_trap 0x3
  isa[i++] = 0xBE801D16;  // s_setpc_b64 s[22:23]

  assert(i <= kTrampolineIsaSize);

  for (; i < kTrampolineIsaSize; i++) {
    isa[i] = 0xBF920002;  // s_trap 0x2
  }

  typedef uint32_t isa_vec __attribute__((vector_size(kTrampolineIsaSize * sizeof(uint32_t))));
  // Hopefuly this results in only two pcie transactions
  *reinterpret_cast<volatile isa_vec*>(&trampoline->isa[0]) =
      *reinterpret_cast<const isa_vec*>(&isa[0]);

  int64_t new_entry_offset =
      reinterpret_cast<uintptr_t>(trampoline) - reinterpret_cast<uintptr_t>(kd);

  if (kernarg_preload) {
    new_entry_offset -= kKernargPreloadSkip;
  }

  kd->kernel_code_entry_byte_offset = new_entry_offset;

  // Allocate at least 32 sgprs
  // https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-compute-pgm-rsrc1-gfx6-gfx12-table
  // TODO(rocm) Too conservative? Detect unused sgpr pair if possible.
  constexpr int32_t GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT = 6;
  constexpr int32_t GRANULATED_WAVEFRONT_SGPR_COUNT_MASK = 0xf;
  if (((compute_pgm_rsrc1 >> GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT) &
       GRANULATED_WAVEFRONT_SGPR_COUNT_MASK) < 0x2 /* (32 / 16 - 1) * 2 */) {
    kd->compute_pgm_rsrc1 =
        compute_pgm_rsrc1 | (0x2 << GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT);
  }

  if (__builtin_expect(is_verbose_on, 0)) {
    std::cerr << "Patching kernel  " << data.kernel_name << std::hex << " (0x"
              << data.kernel_object << ") with " << trampoline << std::dec
              << "\n";
  }
}

rocprofiler_client_id_t* client_id = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t client_ctx = {0};

void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t* user_data,
                           void* callback_data) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) {
    return;
  }

  auto context = static_cast<ToolContext*>(callback_data);

  if (record.operation == ROCPROFILER_CODE_OBJECT_LOAD) {
    auto* data = static_cast<code_obj_load_data_t*>(record.payload);
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
      context->OnCodeObjectLoad(*data);
    }

    if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      context->OnCodeObjectUnload(*data);
    }
    return;
  }

  if (record.operation ==
          ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER &&
      record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
    auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
    context->OnKernelSymbolRegister(*data);
  }
  return;
}

int tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data) {
  client_fini_func = fini_func;

  ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

  ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                       client_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                       nullptr, 0, tool_tracing_callback, tool_data),
                   "code object tracing service configure");

  int valid_ctx = 0;
  ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                   "context validity check");
  if (valid_ctx == 0) {
    // notify rocprofiler that initialization failed
    // and all the contexts, buffers, etc. created
    // should be ignored
    return -1;
  }

  ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "context start");

  // no errors
  return 0;
}

void tool_fini(void* tool_data) { delete static_cast<ToolContext*>(tool_data); }

}  // namespace

}  // namespace client

extern "C" __attribute__((visibility("default")))
rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t version, const char* runtime_version,
                      uint32_t priority, rocprofiler_client_id_t* id) {
  // set the client name
  id->name = "NanCheckTool";

  // store client info
  client::client_id = id;

  // compute major/minor/patch version info
  uint32_t major = version / 10000;
  uint32_t minor = (version % 10000) / 100;
  uint32_t patch = version % 100;

  // generate info string
  auto info = std::stringstream{};
  info << id->name << " (priority=" << priority
       << ") is using rocprofiler-sdk v" << major << "." << minor << "."
       << patch << " (" << runtime_version << ")";

  std::clog << info.str() << std::endl;

  auto tool_data = new client::ToolContext{};

  auto* env = std::getenv("NAN_CHECK_VERBOSE");

  if (env != nullptr && std::strcmp(env, "0") != 0) {
    client::is_verbose_on = true;
  }

  env = std::getenv("NAN_CHECK_OVERFLOW");

  if (env != nullptr && std::strcmp(env, "0") != 0) {
    client::is_inf_check_on = true;
  }

  // create configure data
  static auto cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), &client::tool_init,
      &client::tool_fini, tool_data};

  // return pointer to configure data
  return &cfg;
}
