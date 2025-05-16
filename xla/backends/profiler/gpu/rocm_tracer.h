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

class RocmTracer {
 public:
  // Returns a pointer to singleton RocmTracer.
  static RocmTracer* GetRocmTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;

  void Enable();
  void Disable();

  static uint64_t GetTimestamp();
  static int NumGpus();

  static int toolInit(
    rocprofiler_client_finalize_t finalize_func,
    void* tool_data);
  static void toolFinalize(void* tool_data);

  static std::string opString(
    rocprofiler_callback_tracing_kind_t kind,
    rocprofiler_tracing_operation_t op);

  static void pushCorrelationID(uint64_t id, CorrelationDomain type);
  static void popCorrelationID(CorrelationDomain type);

  void AddToPendingActivityRecords(uint32_t correlation_id) {
    pending_activity_records_.Add(correlation_id);
  }

  void RemoveFromPendingActivityRecords(uint32_t correlation_id) {
    pending_activity_records_.Remove(correlation_id);
  }

  void ClearPendingActivityRecordsCount() { pending_activity_records_.Clear(); }

  size_t GetPendingActivityRecordsCount() {
    return pending_activity_records_.Count();
  }

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit RocmTracer() {}

 private:
  bool registered_{false};
  void endTracing();

  static void api_callback(rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t* user_data,
    void* callback_data);

  static void code_object_callback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t* user_data,
    void* callback_data);

  uint32_t maxBufferSize_{1000000}; // 1M GPU runtime/kernel events

  int num_gpus_;


  bool api_tracing_enabled_ = false;
  bool activity_tracing_enabled_ = false;


  class PendingActivityRecords {
   public:
    // add a correlation id to the pending set
    void Add(uint32_t correlation_id) {
      absl::MutexLock lock(&mutex);
      pending_set.insert(correlation_id);
    }
    // remove a correlation id from the pending set
    void Remove(uint32_t correlation_id) {
      absl::MutexLock lock(&mutex);
      pending_set.erase(correlation_id);
    }
    // clear the pending set
    void Clear() {
      absl::MutexLock lock(&mutex);
      pending_set.clear();
    }
    // count the number of correlation ids in the pending set
    size_t Count() {
      absl::MutexLock lock(&mutex);
      return pending_set.size();
    }
  
   private:
    // set of co-relation ids for which the hcc activity record is pending
    absl::flat_hash_set<uint32_t> pending_set;
    // the callback which processes the activity records (and consequently
    // removes items from the pending set) is called in a separate thread
    // from the one that adds item to the list.
    absl::Mutex mutex;
  };
  PendingActivityRecords pending_activity_records_;

 public:
  // Disable copy and move.
  RocmTracer(const RocmTracer&) = delete;
  RocmTracer& operator=(const RocmTracer&) = delete;
};

}  // namespace profiler
}  // namespace xla
#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
