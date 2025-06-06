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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/types/optional.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/stream_executor/rocm/roctracer_wrapper.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/status.h"
#include "tsl/platform/types.h"


namespace xla {
namespace profiler {

std::string demangle(const char* name);
std::string demangle(const std::string& name);

// This class is used to store the correlation information for a single
enum CorrelationDomain {
  begin,
  Default = begin,
  Domain0 = begin,
  Domain1,
  end,
  size = end
};

class ApiIdList {
  public:
   ApiIdList();
   virtual ~ApiIdList() {}
   bool invertMode() {
     return invert_;
   }
   void setInvertMode(bool invert) {
     invert_ = invert;
   }
   void add(const std::string& apiName);
   void remove(const std::string& apiName);
   bool loadUserPrefs();
 
   // Map api string to cnid enum
   virtual uint32_t mapName(const std::string& apiName) = 0;
 
   bool contains(uint32_t apiId);
   const std::unordered_map<uint32_t, uint32_t>& filterList() {
     return filter_;
   }
 
  private:
   std::unordered_map<uint32_t, uint32_t> filter_;
   bool invert_;
};

struct RocmTracerOptions {
  std::set<uint32_t> api_tracking_set;  // actual api set we want to profile

  // map of domain --> ops for which we need to enable the API callbacks
  // If the ops vector is empty, then enable API callbacks for entire domain
  // absl::flat_hash_map<activity_domain_t, std::vector<uint32_t> > api_callbacks;

  // map of domain --> ops for which we need to enable the Activity records
  // If the ops vector is empty, then enable Activity records for entire domain
  // absl::flat_hash_map<activity_domain_t, std::vector<uint32_t> > activity_tracing;
};

class RocmTracer {
 public:
  // Returns a pointer to singleton RocmTracer.
  static RocmTracer& i();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;

  // std::optional<RocmTracerOptions> options_;

  void Enable(const RocmTracerOptions& options, RocmTraceCollector* collector_);
  void Disable();

  static uint64_t GetTimestamp();
  static int NumGpus();
  RocmTraceCollector* collector() { return collector_; }
  
  static int toolInit(
    rocprofiler_client_finalize_t finalize_func,
    void* tool_data);
  static void toolFinalize(void* tool_data);

  static std::string opString(
    rocprofiler_callback_tracing_kind_t kind,
    rocprofiler_tracing_operation_t op);

  static void pushCorrelationID(uint64_t id, CorrelationDomain type);
  static void popCorrelationID(CorrelationDomain type);

  void TracingCallback(rocprofiler_context_id_t context,
                      rocprofiler_buffer_id_t buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t num_headers, uint64_t drop_count);

  void CodeObjectCallback(rocprofiler_callback_tracing_record_t record,
                          void* callback_data);

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit RocmTracer() {}

  void HipApiEvent(const rocprofiler_record_header_t *hdr,
        RocmTracerEvent *ev); 
  void KernelEvent(const rocprofiler_record_header_t *hdr,
    RocmTracerEvent *ev);

  void endTracing();

private:
  bool registered_{false};
  int num_gpus_;
  std::optional<RocmTracerOptions> options_;
  RocmTraceCollector* collector_ = nullptr;

  uint32_t maxBufferSize_{1000000}; // 1M GPU runtime/kernel events

  tsl::mutex collector_mutex_;

  bool externalCorrelationEnabled_{true};
  bool logging_{false};

  bool api_tracing_enabled_ = false;
  bool activity_tracing_enabled_ = false;

 public:
  // Disable copy and move.
  RocmTracer(const RocmTracer&) = delete;
  RocmTracer& operator=(const RocmTracer&) = delete;
};

}  // namespace profiler
}  // namespace xla
#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
