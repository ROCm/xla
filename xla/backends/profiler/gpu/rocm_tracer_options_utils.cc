/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/profiler/gpu/rocm_tracer_options_utils.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/profiler_options_util.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using tensorflow::ProfileOptions;

// Keys the CUPTI backend recognises that have no ROCm implementation yet.
// Each carries the type it is documented with, so that a wrong-typed value is
// still reported now rather than in the release that implements the key.
struct UnimplementedKey {
  enum Type { kInt64, kBool, kString };

  absl::string_view name;
  Type type;
  absl::string_view reason;
};

constexpr UnimplementedKey kUnimplementedOnRocm[] = {
    {"gpu_enable_nvtx_tracking", UnimplementedKey::kBool,
     "ROCTX marker tracing is currently always on and cannot be toggled; a "
     "dedicated rocprofiler context is required to make it switchable"},
    {"gpu_enable_cupti_activity_graph_trace", UnimplementedKey::kBool,
     "HIP graph tracing is not yet wired into the ROCm tracer"},
    {"gpu_dump_graph_node_mapping", UnimplementedKey::kBool,
     "not implemented on any backend, including CUDA"},
    {"gpu_pm_sample_counters", UnimplementedKey::kString,
     "performance-monitor counter sampling is not yet available on ROCm"},
    {"gpu_pm_sample_interval_us", UnimplementedKey::kInt64,
     "performance-monitor counter sampling is not yet available on ROCm"},
    {"gpu_pm_sample_buffer_size_per_gpu_mb", UnimplementedKey::kInt64,
     "performance-monitor counter sampling is not yet available on ROCm"},
    {"gpu_aggregated_tracing", UnimplementedKey::kBool,
     "the ROCm collector has no aggregated-tracing mode"},
};

// Wrapper around tsl::profiler::SetValue that accumulates instead of returning
// early, so that one bad key does not hide the others.
//
// SetValue returns its type error *before* erasing the key from the working
// set, so the erase below is load-bearing: without it a wrong-typed key would
// be reported twice, once as a type error and once as unrecognised. The CUPTI
// parser never hits this because RETURN_IF_ERROR stops at the first failure.
template <typename T>
void Apply(const ProfileOptions& options, absl::string_view key,
           absl::flat_hash_set<absl::string_view>& keys,
           RocmTracerOptionDiagnostics& diagnostics,
           std::function<void(T)> setter) {
  const absl::Status status = tsl::profiler::SetValue<T>(
      options, std::string(key), keys, std::move(setter));
  if (!status.ok()) {
    keys.erase(key);
    diagnostics.errors.push_back(
        absl::StrCat("advanced_configuration key '", key,
                     "': ", status.message(), " The key was ignored."));
  }
}

void WarnUnimplemented(const ProfileOptions& options,
                       absl::flat_hash_set<absl::string_view>& keys,
                       RocmTracerOptionDiagnostics& diagnostics) {
  for (const UnimplementedKey& key : kUnimplementedOnRocm) {
    if (!keys.contains(key.name)) continue;
    // Type-check even though the value is discarded.
    switch (key.type) {
      case UnimplementedKey::kInt64:
        Apply<int64_t>(options, key.name, keys, diagnostics, [](int64_t) {});
        break;
      case UnimplementedKey::kBool:
        Apply<bool>(options, key.name, keys, diagnostics, [](bool) {});
        break;
      case UnimplementedKey::kString:
        Apply<std::string>(options, key.name, keys, diagnostics,
                           [](const std::string&) {});
        break;
    }
    keys.erase(key.name);
    diagnostics.warnings.push_back(absl::StrCat(
        "advanced_configuration key '", key.name,
        "' is accepted but has no effect on the ROCm backend: ", key.reason,
        "."));
  }
}

}  // namespace

void UpdateRocmTracerOptionsFromProfilerOptions(
    const ProfileOptions& profile_options, RocmTracerOptions& tracer_options,
    RocmTraceCollectorOptions& collector_options,
    RocmTracerOptionDiagnostics& diagnostics) {
  absl::flat_hash_set<absl::string_view> input_keys;
  for (const auto& [key, unused_value] :
       profile_options.advanced_configuration()) {
    input_keys.insert(key);
  }

  Apply<int64_t>(profile_options, "gpu_max_callback_api_events", input_keys,
                 diagnostics, [&](int64_t value) {
                   collector_options.max_callback_api_events = value;
                 });

  // The cap that reads this field is currently gated on a predicate that no
  // activity event satisfies, so the value lands but is not yet enforced. The
  // fix is a separate change; no further plumbing is needed here.
  Apply<int64_t>(profile_options, "gpu_max_activity_api_events", input_keys,
                 diagnostics, [&](int64_t value) {
                   collector_options.max_activity_api_events = value;
                 });

  // One key, two fields. The tracer-side field sizes the AnnotationMap; the
  // collector-side field is currently unread and is set for consistency.
  Apply<int64_t>(profile_options, "gpu_max_annotation_strings", input_keys,
                 diagnostics, [&](int64_t value) {
                   tracer_options.max_annotation_strings = value;
                   collector_options.max_annotation_strings = value;
                 });

  // Matches the CUPTI backend, including its reset-to-all disposition for
  // values above the device count. That reset is performed by the caller,
  // which is the only place that knows how many GPUs are present -- but only
  // for values that fit in the uint32_t field. A value that does not fit never
  // reaches the caller's fixup, so it has to be reported here or not at all.
  Apply<int64_t>(
      profile_options, "gpu_num_chips_to_profile_per_task", input_keys,
      diagnostics, [&](int64_t value) {
        if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
          diagnostics.errors.push_back(absl::StrCat(
              "advanced_configuration key "
              "'gpu_num_chips_to_profile_per_task': ",
              value, " is outside the representable range [0, ",
              std::numeric_limits<uint32_t>::max(),
              "]. The key was ignored."));
          return;
        }
        collector_options.num_gpus = static_cast<uint32_t>(value);
        if (value == 0) {
          diagnostics.warnings.push_back(
              "advanced_configuration key 'gpu_num_chips_to_profile_per_task' "
              "is 0, which is treated as 'all GPUs' rather than 'none'. This "
              "matches the CUDA backend.");
          return;
        }
        diagnostics.warnings.push_back(
            "advanced_configuration key 'gpu_num_chips_to_profile_per_task' "
            "is applied post-hoc on ROCm: events from excluded devices are "
            "discarded after collection, so tracing overhead is unchanged. "
            "Devices are selected by ascending device id, not by topology.");
      });

  WarnUnimplemented(profile_options, input_keys, diagnostics);

  // Report every remaining key, not just the first, and sort so that the
  // output is stable across runs.
  std::vector<absl::string_view> unknown(input_keys.begin(), input_keys.end());
  absl::c_sort(unknown);
  for (absl::string_view key : unknown) {
    diagnostics.errors.push_back(absl::StrCat(
        "advanced_configuration key '", key,
        "' is not recognised by the ROCm GPU tracer and was ignored."));
  }
}

void AppendOptionDiagnostics(const RocmTracerOptionDiagnostics& diagnostics,
                             tensorflow::profiler::XSpace* space) {
  if (space == nullptr) return;
  for (const std::string& message : diagnostics.errors) {
    LOG(ERROR) << message;
    space->add_errors(message);
  }
  for (const std::string& message : diagnostics.warnings) {
    LOG(WARNING) << message;
    space->add_warnings(message);
  }
}

}  // namespace profiler
}  // namespace xla
