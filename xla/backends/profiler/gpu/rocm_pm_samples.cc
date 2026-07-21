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

#include "xla/backends/profiler/gpu/rocm_pm_samples.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"

namespace xla {
namespace profiler {

using tsl::profiler::StatType;
using tsl::profiler::XEventBuilder;
using tsl::profiler::XEventMetadata;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XStatMetadata;

namespace {

// ROCm counters are already human-readable (e.g. SQ_WAVES, GRBM_GUI_ACTIVE), so
// this map is small and mostly a place to add friendlier labels over time. A
// miss falls back to the raw counter name, matching the CUDA behavior.
absl::flat_hash_map<absl::string_view, absl::string_view> ProfileMetricNameMap() {
  return {
      {"GRBM_GUI_ACTIVE", "GPU Active Cycles"},
      {"GRBM_COUNT", "GPU Total Cycles"},
      {"SQ_WAVES", "Wavefronts Launched (Count)"},
      {"SQ_BUSY_CYCLES", "SQ Busy Cycles"},
      {"SQ_INSTS_VALU", "VALU Instructions (Count)"},
      {"SQ_INSTS_VMEM", "VMEM Instructions (Count)"},
      {"SQ_INSTS_LDS", "LDS Instructions (Count)"},
      {"SQ_INSTS_VALU_MFMA_F16", "MFMA FP16 Instructions (Count)"},
      {"SQ_INSTS_VALU_MFMA_BF16", "MFMA BF16 Instructions (Count)"},
      {"SQ_INSTS_VALU_MFMA_F8", "MFMA FP8 Instructions (Count)"},
      {"TCC_HIT_sum", "L2 Cache Hits (Count)"},
      {"TCC_MISS_sum", "L2 Cache Misses (Count)"},
      {"TCC_EA_RDREQ_sum", "HBM Read Requests (Count)"},
      {"TCC_EA_WRREQ_sum", "HBM Write Requests (Count)"},
  };
}

const absl::flat_hash_map<absl::string_view, absl::string_view>&
MetricNameMap() {
  static const absl::NoDestructor<
      absl::flat_hash_map<absl::string_view, absl::string_view>>
      kMetricNameMap(ProfileMetricNameMap());
  return *kMetricNameMap;
}

}  // namespace

std::string GetRocmProfileMetricName(absl::string_view metric_name) {
  const auto& metric_name_map = MetricNameMap();
  if (auto it = metric_name_map.find(metric_name);
      it != metric_name_map.end()) {
    return std::string(it->second);
  }
  return std::string(metric_name);
}

void RocmPmSamples::PopulateCounterLine(XPlaneBuilder* plane,
                                        uint64_t start_gpu_time_ns) {
  absl::flat_hash_map<std::string, int> skipped_nan_count_per_metric;
  XLineBuilder line = plane->GetOrCreateCounterLine();
  std::vector<std::pair<XEventMetadata*, XStatMetadata*>> counter_metadata;
  counter_metadata.reserve(metrics_.size());
  for (const auto& metric : metrics_) {
    std::string display_name = GetRocmProfileMetricName(metric);
    counter_metadata.emplace_back(plane->GetOrCreateEventMetadata(display_name),
                                  plane->GetOrCreateStatMetadata(metric));
  }
  for (auto& sampler_range : sampler_ranges_) {
    DCHECK_EQ(metrics_.size(), sampler_range.metric_values.size());
    for (int i = 0; i < sampler_range.metric_values.size(); ++i) {
      if (std::isnan(sampler_range.metric_values[i])) {
        ++skipped_nan_count_per_metric[counter_metadata[i].first->name()];
        continue;
      }
      XEventBuilder event = line.AddEvent(
          tsl::profiler::Timespan(
              tsl::profiler::NanoToPico(sampler_range.start_timestamp_ns -
                                        start_gpu_time_ns),
              0),
          *counter_metadata[i].first);
      event.AddStatValue(*counter_metadata[i].second,
                         sampler_range.metric_values[i]);
    }
  }
  for (const auto& [metric, count] : skipped_nan_count_per_metric) {
    plane->AddStatValue(
        *plane->GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
            tsl::profiler::StatType::kNanCounterEvents)),
        absl::StrFormat("Skipped %d NaN counter events for %s: ", count,
                        metric));
  }
}

size_t RocmPmSamples::GetNumSamples() const { return sampler_ranges_.size(); }

const std::vector<std::string>& RocmPmSamples::GetMetrics() const {
  return metrics_;
}

const std::vector<RocmSamplerRange>& RocmPmSamples::GetSamplerRanges() const {
  return sampler_ranges_;
}

int64_t RocmPmSamples::GetDeviceId() const { return device_id_; }

}  // namespace profiler
}  // namespace xla
