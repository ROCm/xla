// xla/backends/profiler/gpu/rocm_tracer.h  (FAÇADE)
#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_FACADE_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_FACADE_H_

// Backend: 3=v3 (rocprofiler-sdk), 1=v1 (roctracer). Default to v3.
#ifndef XLA_GPU_ROCM_TRACER_BACKEND
#define XLA_GPU_ROCM_TRACER_BACKEND 3
#endif

#if XLA_GPU_ROCM_TRACER_BACKEND == 3
  #include "xla/backends/profiler/gpu/rocm_profiler_sdk.h"
#else
  #include "xla/backends/profiler/gpu/rocm_tracer_v1.h"
#endif

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_FACADE_H_
