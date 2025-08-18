#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_V1_ADAPTER_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_V1_ADAPTER_H_

#include <optional>
#include "xla/backends/profiler/gpu/rocm_tracer_v1.h"     // v1 definitions  :contentReference[oaicite:3]{index=3}
#include "xla/backends/profiler/gpu/rocm_collector.h"

namespace xla {
namespace profiler {

// Keep the v3-shaped options so GpuTracer doesn't change.
struct RocmTracerOptions {
  uint64_t max_annotation_strings = 1024 * 1024;
};

class RocmTracer {
 public:
  static RocmTracer& GetRocmTracerSingleton() {
    static RocmTracer instance;
    return instance;
  }

  bool IsAvailable() const {
    return !inner_api_enabled_ && !inner_activity_enabled_;
  }

  void Enable(const RocmTracerOptions& opts, RocmTraceCollector* collector) {
    opts_ = opts;

    // Build v1 options set (HIP API + HIP OPS).
    ::xla::profiler::RocmTracerOptions v1opts;
    v1opts.api_callbacks[ACTIVITY_DOMAIN_HIP_API] = {};    // all HIP API
    v1opts.activity_tracing[ACTIVITY_DOMAIN_HIP_OPS] = {}; // all HIP OPS

    inner_.Enable(v1opts, collector);
    inner_api_enabled_ = true;
    inner_activity_enabled_ = true;
  }

  void Disable() {
    inner_.Disable();
    inner_api_enabled_ = inner_activity_enabled_ = false;
  }

  static uint64_t GetTimestamp() { return ::xla::profiler::RocmTracer::GetTimestamp(); }

  uint32_t NumGpus() const { return ::xla::profiler::RocmTracer::NumGpus(); }

 private:
  RocmTracer() = default;

  ::xla::profiler::RocmTracer& inner_ = ::xla::profiler::RocmTracer::GetRocmTracerSingleton();
  std::optional<RocmTracerOptions> opts_;
  bool inner_api_enabled_ = false;
  bool inner_activity_enabled_ = false;
};

}  // namespace profiler
}  // namespace xla

#endif
