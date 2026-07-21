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

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/backends/profiler/gpu/rocm_pm_samples.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using tsl::profiler::XEventVisitor;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneVisitor;

TEST(RocmPmSamplesTest, PopulatesCounterLineWithDisplayNames) {
  std::vector<std::string> metrics = {"SQ_WAVES", "SomeRawCounter"};
  std::vector<RocmSamplerRange> ranges;
  // start_gpu_time_ns rebases event offsets; use it as the base timestamp.
  constexpr uint64_t kStartGpuTimeNs = 1000;
  ranges.push_back(RocmSamplerRange{/*range_index=*/0,
                                    /*start_timestamp_ns=*/1000,
                                    /*end_timestamp_ns=*/1500,
                                    /*metric_values=*/{3.0, 7.0}});
  ranges.push_back(RocmSamplerRange{/*range_index=*/1,
                                    /*start_timestamp_ns=*/1500,
                                    /*end_timestamp_ns=*/2000,
                                    /*metric_values=*/{5.0, 9.0}});
  RocmPmSamples samples(metrics, std::move(ranges), /*device_id=*/0);

  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder builder(&plane);
  samples.PopulateCounterLine(&builder, kStartGpuTimeNs);

  XPlaneVisitor visitor(&plane);
  int counter_line_events = 0;
  bool saw_display_name = false;
  bool saw_raw_name = false;
  visitor.ForEachLine([&](const XLineVisitor& line) {
    EXPECT_EQ(line.Name(), tsl::profiler::kCounterEventsLineName);
    line.ForEachEvent([&](const XEventVisitor& event) {
      ++counter_line_events;
      // Zero-duration counter events.
      EXPECT_EQ(event.DurationPs(), 0);
      if (event.Name() == "Wavefronts Launched (Count)") saw_display_name = true;
      if (event.Name() == "SomeRawCounter") saw_raw_name = true;
    });
  });
  // 2 ranges x 2 metrics = 4 events.
  EXPECT_EQ(counter_line_events, 4);
  EXPECT_TRUE(saw_display_name);  // mapped name
  EXPECT_TRUE(saw_raw_name);      // passthrough on miss
}

TEST(RocmPmSamplesTest, SkipsNanValues) {
  std::vector<std::string> metrics = {"SQ_WAVES"};
  std::vector<RocmSamplerRange> ranges;
  ranges.push_back(RocmSamplerRange{
      0, 1000, 1500,
      {std::numeric_limits<double>::quiet_NaN()}});
  ranges.push_back(RocmSamplerRange{1, 1500, 2000, {4.0}});
  RocmPmSamples samples(metrics, std::move(ranges), 0);

  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder builder(&plane);
  samples.PopulateCounterLine(&builder, 1000);

  XPlaneVisitor visitor(&plane);
  int counter_line_events = 0;
  visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent(
        [&](const XEventVisitor& event) { ++counter_line_events; });
  });
  // The NaN value is skipped; only 1 real event remains.
  EXPECT_EQ(counter_line_events, 1);
}

TEST(RocmPmSamplesTest, MetricNameMappingFallsBackToRaw) {
  EXPECT_EQ(GetRocmProfileMetricName("SQ_WAVES"),
            "Wavefronts Launched (Count)");
  EXPECT_EQ(GetRocmProfileMetricName("TotallyUnknownCounter"),
            "TotallyUnknownCounter");
}

}  // namespace
}  // namespace profiler
}  // namespace xla
