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

// ROCm implementation of the nvtx_utils.h interface using roctx.
//
// Dual-push design: RangePush/RangePop populate BOTH the AnnotationStack
// (Pipeline A — annotations reach kernel events via correlation ID) AND emit
// roctx markers via librocprofiler-sdk-roctx.so (Pipeline B — named range
// events visible to ROCm-native profilers and captured by MarkerCallback in
// rocm_tracer.cc).

#include "tsl/profiler/lib/nvtx_utils.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "rocm/include/rocprofiler-sdk-roctx/roctx.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"

namespace tsl::profiler {

ProfilerDomainHandle DefaultProfilerDomain() {
  // Non-null sentinel triggers the domain path in PushAnnotation()
  // (scoped_annotation.h:41). ProfilerDomain is a forward-declared opaque
  // struct — we cast the address of a static variable to satisfy the
  // non-null check. This value is never dereferenced or passed to any
  // roctx API — roctx has no domain concept.
  static char sentinel;
  return reinterpret_cast<ProfilerDomainHandle>(&sentinel);
}

void RangePush(ProfilerDomainHandle /*domain*/, const char* ascii) {
  // Pipeline A: populate AnnotationStack so the HIP API callback in
  // rocm_tracer.cc:603-605 can read it and attach annotations to kernel
  // dispatch events via correlation ID.
  //
  // The IsEnabled() guard mirrors scoped_annotation.h:47. If Enable() is
  // toggled between a push and its matching pop, the generation-based
  // cleanup in AnnotationStack resets thread-local state, so the stack
  // cannot become permanently unbalanced.
  if (AnnotationStack::IsEnabled()) {
    AnnotationStack::PushAnnotation(ascii);
  }

  // Pipeline B: emit roctx marker so MarkerCallback in rocm_tracer.cc:347
  // can capture it as a named range event with kNVTXRange stat.
  roctxRangePushA(ascii);
}

void RangePop(ProfilerDomainHandle /*domain*/) {
  if (AnnotationStack::IsEnabled()) {
    AnnotationStack::PopAnnotation();
  }

  roctxRangePop();
}

// Return values from roctx naming APIs intentionally discarded to match
// the void signatures declared in nvtx_utils.h.
void NameCurrentThread(const std::string& name) {
  (void)roctxNameOsThread(name.c_str());
}

void NameDevice(int device_id, const std::string& name) {
  (void)roctxNameHipDevice(name.c_str(), device_id);
}

void NameStream(StreamHandle stream, const std::string& name) {
  // StreamHandle is an opaque tsl::profiler::Stream*. Callers pass
  // hipStream_t (== ihipStream_t*) through this opaque handle, mirroring
  // how nvtx_utils.cc casts StreamHandle to CUstream.
  (void)roctxNameHipStream(name.c_str(),
                           reinterpret_cast<const ihipStream_t*>(stream));
}

namespace detail {
void RangePush(ProfilerDomainHandle, StringHandle, uint64_t, const void*,
               size_t) {}
}  // namespace detail

uint64_t RegisterSchema(ProfilerDomainHandle, const void*) { return 0; }
StringHandle RegisterString(ProfilerDomainHandle, const std::string&) {
  return {};
}
void MarkMemoryInitialized(void const*, size_t, StreamHandle) {}

}  // namespace tsl::profiler
