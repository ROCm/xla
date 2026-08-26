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

#include "xla/backends/gpu/rccl_benchmark/common/case_config.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/rccl_benchmark/common/guarded_buffer.h"

namespace xla::gpu::rccl_benchmark {
namespace {

int64_t Int64FromEnv(absl::string_view name, int64_t fallback) {
  const char* value = std::getenv(std::string(name).c_str());
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  int64_t parsed = 0;
  if (!absl::SimpleAtoi(value, &parsed)) {
    return fallback;
  }
  return parsed;
}

bool BoolFromEnv(absl::string_view name, bool fallback) {
  const char* value = std::getenv(std::string(name).c_str());
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  return absl::string_view(value) != "0";
}

// Default transfer size. Eight ranks contributing this much each produce a
// 64 MiB AllGather, which is the point at which the RCCL build used during the
// original investigation switched on the feature these cases target. The
// runner overrides it to walk both sides of that boundary.
constexpr int64_t kDefaultPerRankBytes = 8 << 20;

}  // namespace

CaseConfig CaseConfigFromEnv() {
  CaseConfig config;
  config.per_rank_bytes =
      Int64FromEnv("RCCL_BENCHMARK_PER_RANK_BYTES", kDefaultPerRankBytes);
  config.guard_bytes =
      Int64FromEnv("RCCL_BENCHMARK_GUARD_BYTES", kDefaultGuardBytes);
  config.expect_warp_speed =
      BoolFromEnv("RCCL_BENCHMARK_EXPECT_WARP_SPEED", true);
  config.repeats = static_cast<int>(Int64FromEnv("RCCL_BENCHMARK_REPEATS", 1));
  return config;
}

std::string CaseConfig::Describe() const {
  return absl::StrFormat(
      "per_rank_bytes=%d guard_bytes=%d expect_warp_speed=%s repeats=%d",
      per_rank_bytes, guard_bytes, expect_warp_speed ? "yes" : "no", repeats);
}

}  // namespace xla::gpu::rccl_benchmark
