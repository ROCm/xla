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

// This file wraps roctracer API calls with dso loader so that we don't need to
// have explicit linking to libroctracer. All TF hipsarse API usage should route
// through this wrapper.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_

#ifndef TF_ROCM_VERSION
#define TF_ROCM_VERSION 600000  // Default value if not defined
#endif

#include <rocm/include/rocprofiler-sdk/registration.h>
#include <rocm/include/rocprofiler-sdk/buffer.h>
#include <rocm/include/rocprofiler-sdk/buffer_tracing.h>
#include <rocm/include/rocprofiler-sdk/callback_tracing.h>
#include <rocm/include/rocprofiler-sdk/external_correlation.h>
#include <rocm/include/rocprofiler-sdk/fwd.h>
#include <rocm/include/rocprofiler-sdk/internal_threading.h>
#include <rocm/include/rocprofiler-sdk/cxx/name_info.hpp>
#include <rocm/include/rocprofiler-sdk/rocprofiler.h>

#include "tsl/platform/dso_loader.h"
#include "tsl/platform/env.h"
#include "tsl/platform/platform.h"

namespace stream_executor {
namespace wrap {

#ifdef PLATFORM_GOOGLE

#define ROCTRACER_API_WRAPPER(API_NAME)                            \
  template <typename... Args>                                       \
  auto API_NAME(Args... args) -> decltype(::API_NAME(args...)) {    \
    return (::API_NAME)(args...);                                  \
  }

#else

#define ROCTRACER_API_WRAPPER(API_NAME)                                    \
  template <typename... Args>                                               \
  auto API_NAME(Args... args) -> decltype(::API_NAME(args...)) {            \
    using FuncPtrT = std::add_pointer<decltype(::API_NAME)>::type;          \
    static FuncPtrT loaded = []() -> FuncPtrT {                            \
      static const char* kName = #API_NAME;                                \
      void* f;                                                             \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(                  \
          tsl::internal::CachedDsoLoader::GetRoctracerDsoHandle().value(), \
          kName, &f);                                                      \
      CHECK(s.ok()) << "could not find " << kName                          \
                    << " in roctracer DSO; dlerror: " << s.message();      \
      return reinterpret_cast<FuncPtrT>(f);                                \
    }();                                                                   \
    return loaded(args...);                                                \
  }

#endif  // PLATFORM_GOOGLE

#if TF_ROCM_VERSION >= 600000

// Define the FOREACH_ROCTRACER_API macro to apply ROCTRACER_API_WRAPPER to each API
#define FOREACH_ROCTRACER_API(DO_FUNC)                          \
  DO_FUNC(rocprofiler_start_context)                            \
  DO_FUNC(rocprofiler_stop_context)                             \
  DO_FUNC(rocprofiler_create_context)                           \
  DO_FUNC(rocprofiler_is_initialized)                           \
  DO_FUNC(rocprofiler_context_is_valid)                         \
  DO_FUNC(rocprofiler_assign_callback_thread)                   \
  DO_FUNC(rocprofiler_create_callback_thread)                   \
  DO_FUNC(rocprofiler_configure_callback_tracing_service)       \
  DO_FUNC(rocprofiler_create_buffer)                            \
  DO_FUNC(rocprofiler_flush_buffer)                             \
  DO_FUNC(rocprofiler_configure_buffer_tracing_service)         \
  DO_FUNC(rocprofiler_get_thread_id)                            \
  DO_FUNC(rocprofiler_at_internal_thread_create)                \
  DO_FUNC(rocprofiler_push_external_correlation_id)             \
  DO_FUNC(rocprofiler_query_buffer_tracing_kind_name)           \
  DO_FUNC(rocprofiler_query_available_agents)                   \
  DO_FUNC(rocprofiler_query_buffer_tracing_kind_operation_name) \
  DO_FUNC(rocprofiler_iterate_buffer_tracing_kind_operations)   \
  DO_FUNC(rocprofiler_iterate_buffer_tracing_kinds)             \
  DO_FUNC(rocprofiler_get_status_string)                        \
  DO_FUNC(rocprofiler_configure)                                \
  DO_FUNC(rocprofiler_get_timestamp)

// Apply the wrapper to each API
FOREACH_ROCTRACER_API(ROCTRACER_API_WRAPPER)

#endif  // TF_ROCM_VERSION >= 600000

#undef FOREACH_ROCTRACER_API
#undef ROCTRACER_API_WRAPPER

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCTRACER_WRAPPER_H_