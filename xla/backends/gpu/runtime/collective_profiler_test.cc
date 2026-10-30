/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/collective_profiler.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;

TEST(CollectiveProfilerTest, SizeBucket) {
  EXPECT_EQ(CollectiveProfiler::SizeBucket(0), -1);
  EXPECT_EQ(CollectiveProfiler::SizeBucket(1), 0);
  EXPECT_EQ(CollectiveProfiler::SizeBucket(2), 1);
  EXPECT_EQ(CollectiveProfiler::SizeBucket(3), 1);
  EXPECT_EQ(CollectiveProfiler::SizeBucket(1024), 10);
  EXPECT_EQ(CollectiveProfiler::SizeBucket(2047), 10);
  EXPECT_EQ(CollectiveProfiler::SizeBucket(int64_t{1} << 30), 30);
}

TEST(CollectiveProfilerTest, StatsAdd) {
  CollectiveProfiler::Stats stats;
  stats.Add(absl::Microseconds(10));
  stats.Add(absl::Microseconds(30));
  EXPECT_EQ(stats.count, 2);
  EXPECT_EQ(stats.total, absl::Microseconds(40));
  EXPECT_EQ(stats.min, absl::Microseconds(10));
  EXPECT_EQ(stats.max, absl::Microseconds(30));
}

TEST(CollectiveProfilerTest, Format) {
  CollectiveProfiler::Aggregates aggregates;
  aggregates["kAllReduce"][20].Add(absl::Milliseconds(3));
  aggregates["kAllReduce"][20].Add(absl::Milliseconds(1));
  aggregates["kAllGather"][10].Add(absl::Milliseconds(4));

  std::string out = CollectiveProfiler::Format(
      aggregates, /*skip_rounds=*/5, /*num_devices=*/2, /*pending=*/0,
      /*command_buffer_records=*/0);
  EXPECT_THAT(out, HasSubstr("skipped first 5"));
  EXPECT_THAT(out, HasSubstr("kAllReduce: calls=2 total=4.000 ms (50.0%)"));
  EXPECT_THAT(out, HasSubstr("[1 MiB, 2 MiB)"));
  EXPECT_THAT(out, HasSubstr("[1 KiB, 2 KiB)"));
  EXPECT_THAT(out, HasSubstr("TOTAL: 8.000 ms"));
}

TEST(CollectiveProfilerTest, NestedScopeDisablesTiming) {
  CollectiveProfiler profiler(/*skip_rounds=*/0);
  profiler.BeginRound(0);
  EXPECT_TRUE(profiler.IsActive(0));
  {
    CollectiveProfiler::NestedScope scope;
    EXPECT_FALSE(profiler.IsActive(0));
  }
  EXPECT_TRUE(profiler.IsActive(0));
}

TEST(CollectiveProfilerTest, SkipsRounds) {
  CollectiveProfiler profiler(/*skip_rounds=*/2);
  EXPECT_FALSE(profiler.IsActive(0));
  profiler.BeginRound(0);
  profiler.BeginRound(0);
  EXPECT_FALSE(profiler.IsActive(0));
  profiler.BeginRound(0);
  EXPECT_TRUE(profiler.IsActive(0));
  EXPECT_FALSE(profiler.IsActive(1));
}

}  // namespace
}  // namespace xla::gpu
