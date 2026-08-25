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

#include "xla/stream_executor/rocm/rocm_cache_info.h"

#include <gtest/gtest.h>

namespace stream_executor::gpu {
namespace {

TEST(RocmCacheInfoTest, DefaultsAreZero) {
  // Zero means "not available", which callers must treat as "leave the
  // DeviceDescription field unset" rather than substituting a guess.
  RocmCacheInfo info;
  EXPECT_EQ(info.last_level_cache_size_bytes, 0);
  EXPECT_EQ(info.fabric_clock_mhz, 0);
}

TEST(GetRocmCacheInfoTest, InvalidBdfFails) {
  // An unparsable PCI bus ID cannot name a device, so the query fails rather
  // than reporting an empty hierarchy.
  EXPECT_FALSE(GetRocmCacheInfo("invalid").ok());
}

TEST(GetRocmCacheInfoTest, EmptyBdfFails) {
  EXPECT_FALSE(GetRocmCacheInfo("").ok());
}

}  // namespace
}  // namespace stream_executor::gpu
