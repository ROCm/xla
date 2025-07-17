
#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCPRFILERSDK_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCPRFILERSDK_WRAPPER_H_

#include "rocm/rocm_config.h"
#if TF_ROCM_VERSION >= 60300
#include "rocm/include/rocprofiler-sdk/registration.h"
#include "rocm/include/rocprofiler-sdk/context.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "rocm/include/rocprofiler-sdk/buffer.h"
#include "rocm/include/rocprofiler-sdk/buffer_tracing.h"
#endif

#include "xla/tsl/platform/env.h"
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/platform.h"


namespace xla {
namespace profiler {
namespace wrap {

#define ROCPROFILERSDK_API_WRAPPER(API_NAME)                                    \
  template <typename... Args>                                              \
  auto API_NAME(Args... args) -> decltype(::API_NAME(args...)) {           \
    using FuncPtrT = std::add_pointer<decltype(::API_NAME)>::type;         \
    static FuncPtrT loaded = []() -> FuncPtrT {                            \
      static const char* kName = #API_NAME;                                \
      void* f;                                                             \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(                  \
          tsl::internal::DsoLoader::GetRocprofilerDsoHandle().value(), \
          kName, &f);                                                      \
      CHECK(s.ok()) << "could not find " << kName                          \
                    << " in rocprofiler-sdk DSO; dlerror: " << s.message();      \
      return reinterpret_cast<FuncPtrT>(f);                                \
    }();                                                                   \
    return loaded(args...);                                                \
  }


#define FOREACH_ROCPROFILERSDK_API(DO_FUNC) \
  DO_FUNC(rocprofiler_force_configure)      \
  DO_FUNC(rocprofiler_is_initialized)       \
  DO_FUNC(rocprofiler_create_context) \
  DO_FUNC(rocprofiler_start_context) \
  DO_FUNC(rocprofiler_configure_buffer_tracing_service) \
  DO_FUNC(rocprofiler_create_buffer) \
  DO_FUNC(rocprofiler_context_is_valid)

FOREACH_ROCPROFILERSDK_API(ROCPROFILERSDK_API_WRAPPER)

#undef FOREACH_ROCPROFILERSDK_API
#undef ROCPROFILERSDK_API_WRAPPER

}  // namespace wrap
}  // namespace profiler
}  // namespace xla

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCPROFILERSDK_WRAPPER_H_
