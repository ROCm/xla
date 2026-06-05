/* Copyright 2026 The OpenXLA Authors.
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

// This file wraps amdsmi API calls with dso loader so that we don't need to
// have explicit linking to libamd_smi. All amdsmi API usage should route
// through this wrapper.
//
// amd_smi must NOT be linked: it embeds a copy of rocm_smi, so it exports the
// same `amd::smi::` C++ symbols as librocm_smi64. Linking both into one process
// makes the dynamic loader interpose those duplicate (but ABI-incompatible)
// weak symbols, crashing at static-init. Loading amd_smi via dlopen with
// RTLD_LOCAL (as the dso loader does) keeps its symbols out of the global
// namespace, so they cannot collide with the linked rocm_smi.

#ifndef XLA_STREAM_EXECUTOR_ROCM_AMDSMI_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_AMDSMI_WRAPPER_H_

#include <type_traits>

#include "rocm/include/amd_smi/amdsmi.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/dso_loader.h"

namespace stream_executor {
namespace wrap {

#ifdef PLATFORM_GOOGLE

#define AMDSMI_API_WRAPPER(api_name)                              \
  template <typename... Args>                                     \
  auto api_name(Args... args) -> decltype(::api_name(args...)) {  \
    return ::api_name(args...);                                   \
  }

#else

#define AMDSMI_API_WRAPPER(api_name)                                        \
  template <typename... Args>                                               \
  auto api_name(Args... args) -> decltype(::api_name(args...)) {            \
    using FuncPtrT = std::add_pointer<decltype(::api_name)>::type;          \
    static FuncPtrT loaded = []() -> FuncPtrT {                             \
      static const char* kName = #api_name;                                 \
      void* f;                                                              \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(                   \
          tsl::internal::CachedDsoLoader::GetAmdSmiDsoHandle().value(),     \
          kName, &f);                                                       \
      CHECK(s.ok()) << "could not find " << kName                           \
                    << " in amdsmi lib; dlerror: " << s.message();          \
      return reinterpret_cast<FuncPtrT>(f);                                 \
    }();                                                                    \
    return loaded(args...);                                                 \
  }

#endif

// clang-format off
#define FOREACH_AMDSMI_API(__macro)              \
  __macro(amdsmi_init)                           \
  __macro(amdsmi_shut_down)                      \
  __macro(amdsmi_get_processor_handle_from_bdf)  \
  __macro(amdsmi_get_gpu_metrics_info)           \
  __macro(amdsmi_status_code_to_string)
// clang-format on

FOREACH_AMDSMI_API(AMDSMI_API_WRAPPER)

#undef FOREACH_AMDSMI_API
#undef AMDSMI_API_WRAPPER

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_AMDSMI_WRAPPER_H_
