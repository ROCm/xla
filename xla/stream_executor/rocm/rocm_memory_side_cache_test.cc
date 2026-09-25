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

#include "xla/stream_executor/rocm/rocm_memory_side_cache.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/rocm/smi_util.h"

namespace stream_executor::gpu {
namespace {

constexpr int64_t kKiB = 1024;
constexpr int64_t kMiB = 1024 * kKiB;

TEST(MemorySideCacheSizeFromDataCachesTest, PicksLevelThree) {
  // What amd-smi reports on gfx950 in SPX/NPS1 mode, one entry per cache
  // type.
  std::vector<SmiDataCache> caches = {
      {1, 32 * kKiB}, {1, 16 * kKiB}, {1, 16 * kKiB},
      {2, 4 * kMiB},  {3, 256 * kMiB},
  };
  EXPECT_EQ(MemorySideCacheSizeFromDataCaches(caches), 256 * kMiB);
}

TEST(MemorySideCacheSizeFromDataCachesTest, NoneWithoutLevelThree) {
  // gfx90a has no memory-side cache, so its L2 must not be reported as one.
  std::vector<SmiDataCache> caches = {{1, 16 * kKiB}, {2, 8 * kMiB}};
  EXPECT_EQ(MemorySideCacheSizeFromDataCaches(caches), 0);
  EXPECT_EQ(MemorySideCacheSizeFromDataCaches({}), 0);
}

TEST(MemorySideCacheSizeFromDataCachesTest, DoesNotSumSameLevelEntries) {
  std::vector<SmiDataCache> caches = {{3, 64 * kMiB}, {3, 64 * kMiB}};
  EXPECT_EQ(MemorySideCacheSizeFromDataCaches(caches), 64 * kMiB);
}

TEST(MemorySideCacheSizeForArchTest, DataCenterParts) {
  EXPECT_EQ(MemorySideCacheSizeForArch(RocmComputeCapability("gfx942")),
            256 * kMiB);
  EXPECT_EQ(MemorySideCacheSizeForArch(RocmComputeCapability("gfx950")),
            256 * kMiB);
}

TEST(MemorySideCacheSizeForArchTest, NoneWhereAbsentOrSkuDependent) {
  EXPECT_EQ(MemorySideCacheSizeForArch(RocmComputeCapability("gfx90a")), 0);
  EXPECT_EQ(MemorySideCacheSizeForArch(RocmComputeCapability("gfx1100")), 0);
}

TEST(GetRocmMemorySideCacheSizeTest, InvalidBdfFallsBackToArch) {
  // An unparsable PCI bus ID cannot name a device, so SMI has no answer and
  // the per-gfx size is used.
  EXPECT_EQ(GetRocmMemorySideCacheSize("invalid",
                                       RocmComputeCapability("gfx950")),
            256 * kMiB);
  EXPECT_EQ(GetRocmMemorySideCacheSize("invalid",
                                       RocmComputeCapability("gfx90a")),
            0);
}

}  // namespace
}  // namespace stream_executor::gpu
