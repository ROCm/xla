/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCPROFILER_SDK_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCPROFILER_SDK_WRAPPER_H_

#include "rocm/rocm_config.h"

#include "rocm/include/rocprofiler-sdk/registration.h"
#include "rocm/include/rocprofiler-sdk/buffer.h"
#include "rocm/include/rocprofiler-sdk/buffer_tracing.h"
#include "rocm/include/rocprofiler-sdk/callback_tracing.h"
#include "rocm/include/rocprofiler-sdk/external_correlation.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "rocm/include/rocprofiler-sdk/internal_threading.h"
#include "rocm/include/rocprofiler-sdk/cxx/name_info.hpp"
#include "rocm/include/rocprofiler-sdk/rocprofiler.h"

#include "tsl/platform/dso_loader.h"
#include "tsl/platform/env.h"
#include "tsl/platform/platform.h"

namespace stream_executor {
namespace wrap {

#ifdef PLATFORM_GOOGLE

#define ROCPROFILER_API_WRAPPER(API_NAME)                            \
  template <typename... Args>                                      \
  auto API_NAME(Args... args) -> decltype((::API_NAME)(args...)) { \
    return (::API_NAME)(args...);                                  \
  }

#else

#define ROCPROFILER_API_WRAPPER(API_NAME)                                    \
  template <typename... Args>                                              \
  auto API_NAME(Args... args) -> decltype(::API_NAME(args...)) {           \
    using FuncPtrT = std::add_pointer<decltype(::API_NAME)>::type;         \
    static FuncPtrT loaded = []() -> FuncPtrT {                            \
      static const char* kName = #API_NAME;                                \
      void* f;                                                             \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(                  \
          tsl::internal::CachedDsoLoader::GetRocprofilerSdkDsoHandle().value(), \
          kName, &f);                                                      \
      CHECK(s.ok()) << "could not find " << kName                          \
                    << " in rocprofiler-sdk DSO; dlerror: " << s.message();      \
      return reinterpret_cast<FuncPtrT>(f);                                \
    }();                                                                   \
    return loaded(args...);                                                \
  }

#endif  // PLATFORM_GOOGLE

#define FOREACH_ROCPROFILER_API(DO_FUNC)                          \
  DO_FUNC(rocprofiler_configure)                                \
  DO_FUNC(rocprofiler_at_internal_thread_create)                \
  DO_FUNC(rocprofiler_create_buffer)                            \
  DO_FUNC(rocprofiler_create_context)                           \
  DO_FUNC(rocprofiler_flush_buffer)                             \
  DO_FUNC(rocprofiler_get_status_string)                        \
  DO_FUNC(rocprofiler_context_is_valid)                         \
  DO_FUNC(rocprofiler_start_context)                            \
  DO_FUNC(rocprofiler_stop_context)                             \
  DO_FUNC(rocprofiler_configure_callback_tracing_service)       \
  DO_FUNC(rocprofiler_configure_buffer_tracing_service)         \
  DO_FUNC(rocprofiler_get_timestamp)                            \
  DO_FUNC(rocprofiler_query_available_agents)                   \
  DO_FUNC(rocprofiler_iterate_callback_tracing_kinds)           \
  DO_FUNC(rocprofiler_assign_callback_thread)                   \
  DO_FUNC(rocprofiler_create_callback_thread)                   \
  DO_FUNC(rocprofiler_query_callback_tracing_kind_name)         \
  DO_FUNC(rocprofiler_iterate_callback_tracing_kind_operations) \
  DO_FUNC(rocprofiler_query_callback_tracing_kind_operation_name)

FOREACH_ROCPROFILER_API(ROCPROFILER_API_WRAPPER)

#undef FOREACH_ROCPROFILER_API
#undef ROCPROFILER_API_WRAPPER

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCPROFILER_SDK_WRAPPER_H_
