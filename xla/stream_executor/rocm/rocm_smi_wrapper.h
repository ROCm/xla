/* Copyright 2025 The OpenXLA Authors.
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

// This file wraps rocm_smi API calls with dso loader so that we don't need to
// have explicit linking to librocm_smi64. All rocm_smi API usage should route
// through this wrapper.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_SMI_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_SMI_WRAPPER_H_

#include "rocm/include/rocm_smi/rocm_smi.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/dso_loader.h"

namespace stream_executor {
namespace wrap {

#ifdef PLATFORM_GOOGLE

#define ROCM_SMI_API_WRAPPER(api_name)                            \
  template <typename... Args>                                     \
  auto api_name(Args... args) -> decltype(::api_name(args...)) {  \
    return ::api_name(args...);                                   \
  }

#else

#define ROCM_SMI_API_WRAPPER(api_name)                                      \
  template <typename... Args>                                               \
  auto api_name(Args... args) -> decltype(::api_name(args...)) {            \
    using FuncPtrT = std::add_pointer<decltype(::api_name)>::type;          \
    static FuncPtrT loaded = []() -> FuncPtrT {                             \
      static const char* kName = #api_name;                                 \
      void* f;                                                              \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(                   \
          tsl::internal::CachedDsoLoader::GetRocmSmiDsoHandle().value(),    \
          kName, &f);                                                       \
      CHECK(s.ok()) << "could not find " << kName                           \
                    << " in rocm_smi lib; dlerror: " << s.message();        \
      return reinterpret_cast<FuncPtrT>(f);                                 \
    }();                                                                    \
    return loaded(args...);                                                 \
  }

#endif

// clang-format off
#define FOREACH_ROCM_SMI_API(__macro)              \
  __macro(rsmi_init)                               \
  __macro(rsmi_shut_down)                          \
  __macro(rsmi_num_monitor_devices)                \
  __macro(rsmi_dev_pci_id_get)                     \
  __macro(rsmi_dev_gpu_metrics_info_get)            \
  __macro(rsmi_status_string)                      \
  __macro(rsmi_topo_get_link_type)                 \
  __macro(rsmi_dev_xgmi_hive_id_get)
// clang-format on

FOREACH_ROCM_SMI_API(ROCM_SMI_API_WRAPPER)

#undef FOREACH_ROCM_SMI_API
#undef ROCM_SMI_API_WRAPPER

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_SMI_WRAPPER_H_
