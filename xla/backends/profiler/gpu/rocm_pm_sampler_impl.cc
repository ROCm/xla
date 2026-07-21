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

#include "xla/backends/profiler/gpu/rocm_pm_sampler_impl.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "rocm/include/rocprofiler-sdk/agent.h"
#include "rocm/include/rocprofiler-sdk/context.h"
#include "rocm/include/rocprofiler-sdk/counter_config.h"
#include "rocm/include/rocprofiler-sdk/counters.h"
#include "rocm/include/rocprofiler-sdk/device_counting_service.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "rocm/include/rocprofiler-sdk/rocprofiler.h"
#include "xla/backends/profiler/gpu/rocm_pm_sampler.h"
#include "xla/backends/profiler/gpu/rocm_pm_samples.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace profiler {
namespace {

absl::Status ToStatus(rocprofiler_status_t status, absl::string_view what) {
  if (status == ROCPROFILER_STATUS_SUCCESS) {
    return absl::OkStatus();
  }
  const char* errstr = rocprofiler_get_status_string(status);
  return absl::InternalError(
      absl::StrCat(what, " failed: ", errstr ? errstr : "unknown"));
}

uint64_t Timestamp() {
  rocprofiler_timestamp_t ts = 0;
  if (rocprofiler_get_timestamp(&ts) != ROCPROFILER_STATUS_SUCCESS) {
    return 0;
  }
  return ts;
}

}  // namespace

RocmPmSamplerDevice::RocmPmSamplerDevice(int device_id,
                                         rocprofiler_context_id_t context,
                                         rocprofiler_agent_id_t agent_id,
                                         const RocmPmSamplerOptions& options)
    : device_id_(device_id),
      context_(context),
      agent_id_(agent_id),
      sample_interval_ns_(options.sample_interval_ns),
      flush_period_(options.flush_period),
      max_samples_(options.max_samples),
      config_metrics_(options.metrics) {}

void RocmPmSamplerDevice::SetSink(
    std::function<void(RocmPmSamples*)> process_samples) {
  process_samples_ = std::move(process_samples);
  if (!process_samples_) {
    process_samples_ = [](RocmPmSamples* s) {
      LOG(WARNING) << "(Profiling::PM Sampling) No decode handler specified, "
                   << "discarding " << s->GetSamplerRanges().size()
                   << " samples";
    };
  }
  // Spawn the sampling thread only once the sink is known.
  thread_ = std::make_unique<std::thread>(&RocmPmSamplerDevice::MainFunc, this);
}

RocmPmSamplerDevice::~RocmPmSamplerDevice() {
  Exit();
  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
  if (config_created_) {
    rocprofiler_destroy_counter_config(config_id_);
  }
}

absl::Status RocmPmSamplerDevice::CreateConfig() {
  // Enumerate this agent's supported counters into name -> id.
  absl::flat_hash_map<std::string, rocprofiler_counter_id_t> name_to_id;
  auto enum_cb = [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters,
                    size_t num_counters, void* user_data) {
    auto* map =
        static_cast<absl::flat_hash_map<std::string, rocprofiler_counter_id_t>*>(
            user_data);
    for (size_t i = 0; i < num_counters; ++i) {
      rocprofiler_counter_info_v0_t info;
      if (rocprofiler_query_counter_info(counters[i],
                                         ROCPROFILER_COUNTER_INFO_VERSION_0,
                                         &info) == ROCPROFILER_STATUS_SUCCESS &&
          info.name != nullptr) {
        (*map)[std::string(info.name)] = counters[i];
      }
    }
    return ROCPROFILER_STATUS_SUCCESS;
  };
  RETURN_IF_ERROR(ToStatus(
      rocprofiler_iterate_agent_supported_counters(agent_id_, enum_cb,
                                                   &name_to_id),
      "rocprofiler_iterate_agent_supported_counters"));

  // Resolve requested metrics in order; drop+warn on miss.
  for (const std::string& metric : config_metrics_) {
    auto it = name_to_id.find(metric);
    if (it == name_to_id.end()) {
      LOG(WARNING) << "(Profiling::PM Sampling) counter '" << metric
                   << "' is not supported on device " << device_id_
                   << ", dropping it. Run `rocprofv3 --list-avail` to see "
                   << "available counters.";
      continue;
    }
    enabled_metrics_.push_back(metric);
    counter_ids_.push_back(it->second);
  }
  if (enabled_metrics_.empty()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "No valid PM counters for device ", device_id_));
  }

  // Create the config, reducing the set until it fits. rocprofiler exposes no
  // "passes" API, so we key the reduction on create_counter_config success
  // rather than a pass count (NVIDIA's shape, ROCm's mechanism).
  while (!counter_ids_.empty()) {
    rocprofiler_counter_config_id_t cfg{0};
    rocprofiler_status_t rc = rocprofiler_create_counter_config(
        agent_id_, counter_ids_.data(), counter_ids_.size(), &cfg);
    if (rc == ROCPROFILER_STATUS_SUCCESS) {
      config_id_ = cfg;
      config_created_ = true;
      break;
    }
    if (counter_ids_.size() == 1) {
      return ToStatus(rc, absl::StrCat("rocprofiler_create_counter_config for '",
                                       enabled_metrics_.back(), "'"));
    }
    LOG(WARNING) << "(Profiling::PM Sampling) counter set too large for a "
                 << "single config on device " << device_id_ << ", dropping '"
                 << enabled_metrics_.back() << "' and retrying.";
    counter_ids_.pop_back();
    enabled_metrics_.pop_back();
  }

  // Build reverse map for per-record bucketing.
  for (size_t i = 0; i < counter_ids_.size(); ++i) {
    id_to_index_[counter_ids_[i].handle] = i;
  }
  // Reserve a generous record buffer; grows on OUT_OF_RESOURCES. Counters may
  // report multiple dimension instances (XCC/SE/CU) per id.
  records_.resize(counter_ids_.size() * 64);
  return absl::OkStatus();
}

void RocmPmSamplerDevice::ServiceCallback(
    rocprofiler_context_id_t context_id, rocprofiler_agent_id_t /*agent_id*/,
    rocprofiler_device_counting_agent_cb_t set_config, void* user_data) {
  auto* self = static_cast<RocmPmSamplerDevice*>(user_data);
  rocprofiler_status_t rc = set_config(context_id, self->config_id_);
  if (rc != ROCPROFILER_STATUS_SUCCESS) {
    LOG(ERROR) << "(Profiling::PM Sampling) set_config failed on device "
               << self->device_id_ << ": "
               << rocprofiler_get_status_string(rc);
  }
}

absl::Status RocmPmSamplerDevice::ConfigureService() {
  // buffer_id{0} => samples are returned synchronously via the sample call's
  // output_records array (no hardware buffer on this path).
  return ToStatus(
      rocprofiler_configure_device_counting_service(
          context_, rocprofiler_buffer_id_t{0}, agent_id_, &ServiceCallback,
          this),
      "rocprofiler_configure_device_counting_service");
}

absl::Status RocmPmSamplerDevice::StartContext() {
  return ToStatus(rocprofiler_start_context(context_),
                  "rocprofiler_start_context(pm)");
}

absl::Status RocmPmSamplerDevice::StopContext() {
  return ToStatus(rocprofiler_stop_context(context_),
                  "rocprofiler_stop_context(pm)");
}

void RocmPmSamplerDevice::Aggregate(size_t rec_count, RocmSamplerRange* range) {
  const size_t n = enabled_metrics_.size();
  std::vector<double> values(n, 0.0);
  std::vector<uint32_t> hit(n, 0);
  for (size_t i = 0; i < rec_count; ++i) {
    rocprofiler_counter_id_t cid{0};
    if (rocprofiler_query_record_counter_id(records_[i].id, &cid) !=
        ROCPROFILER_STATUS_SUCCESS) {
      continue;
    }
    auto it = id_to_index_.find(cid.handle);
    if (it == id_to_index_.end()) {
      continue;
    }
    // Sum across dimension instances to get one value per named metric.
    values[it->second] += records_[i].counter_value;
    ++hit[it->second];
  }
  for (size_t k = 0; k < n; ++k) {
    if (hit[k] == 0) {
      values[k] = std::numeric_limits<double>::quiet_NaN();
    }
  }
  range->metric_values = std::move(values);
}

absl::Status RocmPmSamplerDevice::SampleOnce(RocmSamplerRange* range) {
  uint64_t t0 = Timestamp();
  size_t rec_count = records_.size();
  rocprofiler_status_t rc = rocprofiler_sample_device_counting_service(
      context_, rocprofiler_user_data_t{}, ROCPROFILER_COUNTER_FLAG_NONE,
      records_.data(), &rec_count);
  if (rc == ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES) {
    // Grow and retry once.
    records_.resize(records_.size() * 2);
    rec_count = records_.size();
    rc = rocprofiler_sample_device_counting_service(
        context_, rocprofiler_user_data_t{}, ROCPROFILER_COUNTER_FLAG_NONE,
        records_.data(), &rec_count);
  }
  RETURN_IF_ERROR(
      ToStatus(rc, "rocprofiler_sample_device_counting_service"));
  uint64_t t1 = Timestamp();
  range->start_timestamp_ns = t0;
  range->end_timestamp_ns = t1;
  Aggregate(rec_count, range);
  return absl::OkStatus();
}

void RocmPmSamplerDevice::SampleUntilDisabled() {
  bool disabling = false;
  bool final_pass = false;
  size_t range_index = 0;
  std::vector<RocmSamplerRange> batch;
  batch.reserve(max_samples_);
  absl::Time last_flush = absl::Now();

  auto flush = [&]() {
    if (batch.empty()) return;
    RocmPmSamples samples(enabled_metrics_, std::move(batch), device_id_);
    process_samples_(&samples);
    batch.clear();
    batch.reserve(max_samples_);
  };

  while (!disabling) {
    if (!NextStateIs(ThreadState::kEnabled)) {
      // One more sweep to drain, then stop.
      if (!final_pass) {
        final_pass = true;
      } else {
        disabling = true;
      }
    }

    absl::Time begin = absl::Now();
    RocmSamplerRange range;
    range.range_index = range_index++;
    if (SampleOnce(&range).ok()) {
      batch.push_back(std::move(range));
    } else {
      LOG(WARNING) << "(Profiling::PM Sampling) sample failed on device "
                   << device_id_;
    }

    if (absl::Now() - last_flush >= flush_period_ ||
        batch.size() >= max_samples_) {
      flush();
      last_flush = absl::Now();
    }

    absl::Duration elapsed = absl::Now() - begin;
    absl::Duration interval = absl::Nanoseconds(sample_interval_ns_);
    if (elapsed < interval) {
      absl::SleepFor(interval - elapsed);
    } else {
      LOG_EVERY_N(WARNING, 100)
          << "(Profiling::PM Sampling) sampling on device " << device_id_
          << " took longer than the configured interval (" << elapsed
          << " > " << interval << "); reduce metric count or sample rate.";
    }
  }
  flush();
}

void RocmPmSamplerDevice::MainFunc() {
  absl::MutexLock lock(&state_mutex_);
  do {
    auto state_changed = [this] {
      state_mutex_.AssertReaderHeld();
      return current_state_ != next_state_;
    };
    state_mutex_.Await(absl::Condition(&state_changed));

    switch (next_state_) {
      case ThreadState::kInitialized:
        current_state_ = ThreadState::kDisabled;
        break;
      case ThreadState::kEnabled:
        current_state_ = ThreadState::kEnabled;
        state_mutex_.unlock();
        SampleUntilDisabled();
        state_mutex_.lock();
        current_state_ = ThreadState::kDisabled;
        break;
      case ThreadState::kUninitialized:
        current_state_ = ThreadState::kUninitialized;
        break;
      case ThreadState::kExiting:
        current_state_ = ThreadState::kExiting;
        return;
    }
  } while (true);
}

absl::StatusOr<std::unique_ptr<RocmPmSamplerImpl>> RocmPmSamplerImpl::Create(
    const std::vector<rocprofiler_context_id_t>& gpu_contexts,
    const std::vector<rocprofiler_agent_id_t>& gpu_agents,
    const RocmPmSamplerOptions& options) {
  std::unique_ptr<RocmPmSamplerImpl> sampler(new RocmPmSamplerImpl());
  if (gpu_agents.empty()) {
    return sampler;
  }
  RETURN_IF_ERROR(sampler->Initialize(gpu_contexts, gpu_agents, options));
  return sampler;
}

absl::Status RocmPmSamplerImpl::Initialize(
    const std::vector<rocprofiler_context_id_t>& gpu_contexts,
    const std::vector<rocprofiler_agent_id_t>& gpu_agents,
    const RocmPmSamplerOptions& options) {
  if (initialized_) {
    return absl::AlreadyExistsError("PM sampler already initialized");
  }
  if (gpu_contexts.size() != gpu_agents.size()) {
    return absl::InvalidArgumentError(
        "gpu_contexts and gpu_agents size mismatch");
  }

  absl::Cleanup cleanup([this]() { devices_.clear(); });

  // All of this must run BEFORE HIP creates its device queues (i.e. at
  // library-load / rocprofiler tool-init time, before `import jax` finishes
  // bringing up the runtime). rocprofiler creates the per-agent profile queue
  // by intercepting HSA queue creation, but only if the counter config and
  // device-counting service are already registered AND the context is started.
  // Configuring or starting later yields "No profile queue is available for
  // this agent" and zero counter records.
  for (size_t i = 0; i < gpu_agents.size(); ++i) {
    auto dev = std::make_unique<RocmPmSamplerDevice>(
        static_cast<int>(i), gpu_contexts[i], gpu_agents[i], options);
    RETURN_IF_ERROR(dev->CreateConfig());
    RETURN_IF_ERROR(dev->ConfigureService());
    if (absl::Status s = dev->StartContext(); !s.ok()) {
      LOG(WARNING) << "(Profiling::PM Sampling) failed to start context on "
                   << "device " << dev->device_id() << ": " << s;
    }
    devices_.push_back(std::move(dev));
  }

  std::move(cleanup).Cancel();
  initialized_ = true;
  return absl::OkStatus();
}

absl::Status RocmPmSamplerImpl::StartSampler(
    std::function<void(RocmPmSamples*)> process_samples) {
  if (enabled_) {
    return absl::AlreadyExistsError("Already started");
  }
  // Contexts were already started in Initialize (pre-HIP). Now that the xplane
  // sink is known, wire it in and enable the sampling threads.
  for (auto& dev : devices_) {
    dev->SetSink(process_samples);
  }
  for (auto& dev : devices_) {
    dev->Enable();
  }
  for (auto& dev : devices_) {
    dev->AwaitEnablement();
  }
  enabled_ = true;
  return absl::OkStatus();
}

absl::Status RocmPmSamplerImpl::StopSampler() {
  if (!enabled_) {
    return absl::FailedPreconditionError("StopSampler called before start");
  }
  for (auto& dev : devices_) {
    dev->Disable();
  }
  for (auto& dev : devices_) {
    dev->AwaitDisablement();
  }
  for (auto& dev : devices_) {
    if (absl::Status s = dev->StopContext(); !s.ok()) {
      LOG(WARNING) << "(Profiling::PM Sampling) failed to stop context on "
                   << "device " << dev->device_id() << ": " << s;
    }
  }
  enabled_ = false;
  return absl::OkStatus();
}

absl::Status RocmPmSamplerImpl::Deinitialize() {
  if (enabled_) {
    StopSampler().IgnoreError();
  }
  if (!initialized_) {
    return absl::FailedPreconditionError("Deinitialize called before init");
  }
  // Device dtors signal exit + join their threads and destroy configs.
  devices_.clear();
  initialized_ = false;
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace xla
