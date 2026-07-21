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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_PM_SAMPLER_IMPL_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_PM_SAMPLER_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "rocm/include/rocprofiler-sdk/device_counting_service.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "xla/backends/profiler/gpu/rocm_pm_sampler.h"
#include "xla/backends/profiler/gpu/rocm_pm_samples.h"

namespace xla {
namespace profiler {

// Per-GPU PM sampler. Owns the rocprofiler counter config + device-counting
// service for one agent on its own context, AND its own sampling thread.
//
// This collapses NVIDIA's two-tier split (a per-device config owner plus a
// separate decode thread that drains a hardware ring buffer). On ROCm 7.2.4
// there is no hardware time-interval trigger (SPM is unavailable), so the host
// thread IS the sampling clock: it calls rocprofiler_sample_device_counting_
// service every sample_interval_ns and aggregates the returned records. Decode
// is trivial (records already carry a double), so a separate decode thread
// would only add cross-device sampling skew.
//
// Each device uses its OWN context: rocprofiler documents that a context
// profiles a single agent, and rocprofiler_sample_device_counting_service takes
// only a context (no agent), so one context per agent is required.
class RocmPmSamplerDevice {
 public:
  RocmPmSamplerDevice(int device_id, rocprofiler_context_id_t context,
                      rocprofiler_agent_id_t agent_id,
                      const RocmPmSamplerOptions& options);
  ~RocmPmSamplerDevice();

  RocmPmSamplerDevice(const RocmPmSamplerDevice&) = delete;
  RocmPmSamplerDevice& operator=(const RocmPmSamplerDevice&) = delete;

  // Resolve requested metric names to counter ids for this agent and create the
  // counter config (with a reduction-retry loop if the set is too large).
  absl::Status CreateConfig();

  // Register the device-counting service on this device's context. The service
  // callback binds the config when the context starts.
  absl::Status ConfigureService();

  // Start/stop this device's counting context. StartContext triggers the
  // service callback that binds the config.
  absl::Status StartContext();
  absl::Status StopContext();

  // State transitions (request/await handshake with the sampling thread).
  void Enable() { ChangeState(ThreadState::kEnabled); }
  void AwaitEnablement() { AwaitState(ThreadState::kEnabled); }
  void Disable() { ChangeState(ThreadState::kDisabled); }
  void AwaitDisablement() { AwaitState(ThreadState::kDisabled); }

  int device_id() const { return device_id_; }
  const std::vector<std::string>& GetEnabledMetrics() const {
    return enabled_metrics_;
  }

 private:
  enum class ThreadState {
    kUninitialized,
    kInitialized,
    kDisabled = kInitialized,
    kEnabled,
    kExiting
  };

  // Bound as the device-counting service callback; calls set_config with our
  // config when the context starts.
  static void ServiceCallback(
      rocprofiler_context_id_t context_id, rocprofiler_agent_id_t agent_id,
      rocprofiler_device_counting_agent_cb_t set_config, void* user_data);

  // One synchronous sample sweep -> one RocmSamplerRange.
  absl::Status SampleOnce(RocmSamplerRange* range);
  // Sum counter records by counter id into per-metric values (NaN if missing).
  void Aggregate(size_t rec_count, RocmSamplerRange* range);

  void MainFunc();
  void SampleUntilDisabled();

  void ChangeState(ThreadState state) ABSL_LOCKS_EXCLUDED(state_mutex_) {
    absl::MutexLock lock(&state_mutex_);
    next_state_ = state;
  }
  bool NextStateIs(ThreadState state) ABSL_LOCKS_EXCLUDED(state_mutex_) {
    absl::ReaderMutexLock lock(&state_mutex_);
    return next_state_ == state;
  }
  void AwaitState(ThreadState state) ABSL_LOCKS_EXCLUDED(state_mutex_) {
    absl::ReaderMutexLock lock(&state_mutex_);
    auto equals = [this, state] {
      state_mutex_.AssertReaderHeld();
      return current_state_ == state;
    };
    state_mutex_.Await(absl::Condition(&equals));
  }
  void Exit() { ChangeState(ThreadState::kExiting); }

  const int device_id_;
  const rocprofiler_context_id_t context_;
  const rocprofiler_agent_id_t agent_id_;
  const size_t sample_interval_ns_;
  const absl::Duration flush_period_;
  const size_t max_samples_;
  std::function<void(RocmPmSamples*)> process_samples_;

  // Requested metric names (input order) and the subset that resolved+fit.
  std::vector<std::string> config_metrics_;
  std::vector<std::string> enabled_metrics_;
  std::vector<rocprofiler_counter_id_t> counter_ids_;
  // counter_id.handle -> index into enabled_metrics_ / metric_values.
  absl::flat_hash_map<uint64_t, size_t> id_to_index_;
  rocprofiler_counter_config_id_t config_id_{0};
  bool config_created_ = false;

  // Reused across sweeps (ROCm analogue of NVIDIA's buffer recycling).
  std::vector<rocprofiler_counter_record_t> records_;

  absl::Mutex state_mutex_;
  ThreadState current_state_ ABSL_GUARDED_BY(state_mutex_) =
      ThreadState::kUninitialized;
  ThreadState next_state_ ABSL_GUARDED_BY(state_mutex_) =
      ThreadState::kInitialized;
  std::unique_ptr<std::thread> thread_;
};

// Orchestrator: owns one RocmPmSamplerDevice per GPU and drives them together.
class RocmPmSamplerImpl : public RocmPmSampler {
 public:
  // gpu_contexts and gpu_agents are parallel arrays (context[i] profiles
  // agent[i]). Contexts are created by the caller (RocmTracer) at init time.
  static absl::StatusOr<std::unique_ptr<RocmPmSamplerImpl>> Create(
      const std::vector<rocprofiler_context_id_t>& gpu_contexts,
      const std::vector<rocprofiler_agent_id_t>& gpu_agents,
      const RocmPmSamplerOptions& options);

  absl::Status StartSampler() override;
  absl::Status StopSampler() override;
  absl::Status Deinitialize() override;

 private:
  RocmPmSamplerImpl() = default;

  absl::Status Initialize(
      const std::vector<rocprofiler_context_id_t>& gpu_contexts,
      const std::vector<rocprofiler_agent_id_t>& gpu_agents,
      const RocmPmSamplerOptions& options);

  bool initialized_ = false;
  bool enabled_ = false;
  std::vector<std::unique_ptr<RocmPmSamplerDevice>> devices_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_PM_SAMPLER_IMPL_H_
