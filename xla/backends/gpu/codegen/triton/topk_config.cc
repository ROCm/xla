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

#include "xla/backends/gpu/codegen/triton/topk_config.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "re2/re2.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr int64_t kMaxK = 128;
constexpr double kMaxKOverN = 0.85;

// A program holds about this many keys, so short rows share a program.
constexpr int64_t kTargetKeysPerProgram = 4096;
constexpr int64_t kTargetKeysPerThread = 8;
// Each 64-bit key takes two registers per thread.
constexpr int64_t kMaxKeysPerThread = 32;
constexpr int64_t kMaxNumWarps = 8;

// A region of TopK shapes where the Triton kernel was measured faster than the
// default path. Bounds are inclusive.
struct TopKSpeedRegion {
  PrimitiveType input_type;
  int64_t min_batch, max_batch;
  int64_t min_n, max_n;
  int64_t min_k, max_k;
};

absl::Span<const TopKSpeedRegion> GetTopKSpeedTable(
    const se::DeviceDescription& device) {
  // Regions are added per architecture once they are measured faster than the
  // default path; none are measured yet.
  return {};
}

int64_t NextPowerOfTwo(int64_t x) {
  return static_cast<int64_t>(absl::bit_ceil(static_cast<uint64_t>(x)));
}

}  // namespace

std::string TopKKernelSpec::ToString() const {
  return absl::StrFormat("%s[%d,%d] k=%d order=%s",
                         primitive_util::LowercasePrimitiveTypeName(input_type),
                         batch, n, k, ComparisonOrderToString(order));
}

std::string TopKKernelConfig::ToString() const {
  return absl::StrFormat(
      "block_n=%d rows_per_program=%d num_warps=%d num_programs=%d", block_n,
      rows_per_program, num_warps, num_programs);
}

absl::StatusOr<TopKKernelSpec> GetTopKKernelSpec(
    const HloCustomCallInstruction& topk) {
  if (topk.operand_count() != 1 || !topk.shape().IsTuple() ||
      topk.shape().tuple_shapes().size() < 2) {
    return InvalidArgument("Not a TopK custom call: %s", topk.ToString());
  }
  const Shape& data_shape = topk.operand(0)->shape();
  const int64_t rank = data_shape.dimensions().size();
  if (rank != 1 && rank != 2) {
    return InvalidArgument("TopK input must have rank 1 or 2: %s",
                           data_shape.ToString());
  }
  const bool has_batch = rank == 2;
  return TopKKernelSpec{
      data_shape.element_type(),
      hlo_instruction_utils::GetTopKOrder(&topk),
      has_batch ? data_shape.dimensions(0) : 1,
      data_shape.dimensions(has_batch ? 1 : 0),
      topk.shape().tuple_shapes(0).dimensions(has_batch ? 1 : 0),
  };
}

absl::Status CheckTopKKernelLimits(const TopKKernelSpec& spec) {
  if (spec.input_type != F32 && spec.input_type != BF16) {
    return InvalidArgument(
        "unsupported type %s",
        primitive_util::LowercasePrimitiveTypeName(spec.input_type));
  }
  if (spec.order != ComparisonOrder::kTotal &&
      spec.order != ComparisonOrder::kPartial) {
    return InvalidArgument("unsupported order %s",
                           ComparisonOrderToString(spec.order));
  }
  if (spec.k < 1 || spec.k > kMaxK) {
    return InvalidArgument("k=%d outside [1, %d]", spec.k, kMaxK);
  }
  // The PARTIAL key keeps a 31-bit reversed index that must stay non-zero.
  if (spec.n >= (int64_t{1} << 31)) {
    return InvalidArgument("n=%d is not below 2^31", spec.n);
  }
  if (static_cast<double>(spec.k) / spec.n >= kMaxKOverN) {
    return InvalidArgument("k/n=%f is not below %f",
                           static_cast<double>(spec.k) / spec.n, kMaxKOverN);
  }
  return absl::OkStatus();
}

std::optional<TopKKernelConfig> GetTopKKernelConfig(
    const TopKKernelSpec& spec, const se::DeviceDescription& device) {
  const int64_t wave_size = device.threads_per_warp();
  if (wave_size <= 0) {
    return std::nullopt;
  }
  const int64_t max_num_warps =
      std::min(kMaxNumWarps, device.threads_per_block_limit() / wave_size);
  if (max_num_warps < 1) {
    return std::nullopt;
  }

  TopKKernelConfig config;
  config.block_n = NextPowerOfTwo(spec.n);
  config.rows_per_program = std::clamp(kTargetKeysPerProgram / config.block_n,
                                       int64_t{1}, NextPowerOfTwo(spec.batch));
  const int64_t keys_per_program = config.rows_per_program * config.block_n;
  config.num_warps =
      std::clamp(NextPowerOfTwo(CeilOfRatio(keys_per_program,
                                            wave_size * kTargetKeysPerThread)),
                 int64_t{1}, max_num_warps);
  config.num_programs = CeilOfRatio(spec.batch, config.rows_per_program);

  // Rows too long for one program need the chunked kernels.
  if (CeilOfRatio(keys_per_program, config.num_warps * wave_size) >
      kMaxKeysPerThread) {
    return std::nullopt;
  }
  // Cross-warp reductions exchange one 64-bit key per row and warp in LDS.
  const int64_t lds_bytes =
      config.rows_per_program * config.num_warps * int64_t{8};
  if (lds_bytes > device.shared_memory_per_block()) {
    return std::nullopt;
  }
  if (config.num_programs > device.block_dim_limit().x) {
    return std::nullopt;
  }
  return config;
}

bool IsTritonTopKFaster(const TopKKernelSpec& spec,
                        const se::DeviceDescription& device) {
  return absl::c_any_of(
      GetTopKSpeedTable(device), [&](const TopKSpeedRegion& region) {
        return region.input_type == spec.input_type &&
               spec.batch >= region.min_batch &&
               spec.batch <= region.max_batch && spec.n >= region.min_n &&
               spec.n <= region.max_n && spec.k >= region.min_k &&
               spec.k <= region.max_k;
      });
}

bool IsTritonTopK(const HloCustomCallInstruction& topk) {
  static const LazyRE2 kTritonImplRegex = {R"(impl\s*=\s*"triton")"};
  return RE2::PartialMatch(topk.raw_backend_config_string(), *kTritonImplRegex);
}

absl::Status MarkAsTritonTopK(HloCustomCallInstruction& topk) {
  if (IsTritonTopK(topk)) {
    return absl::OkStatus();
  }
  static const LazyRE2 kEmptyDictRegex = {R"(^\s*(\{\s*\})?\s*$)"};
  static const LazyRE2 kNonEmptyDictEndRegex = {R"(([^\s\{])\s*\}\s*$)"};
  std::string config = topk.raw_backend_config_string();
  if (RE2::FullMatch(config, *kEmptyDictRegex)) {
    config = R"({impl = "triton"})";
  } else if (!RE2::Replace(&config, *kNonEmptyDictEndRegex,
                           R"(\1, impl = "triton"})")) {
    return InvalidArgument("Unexpected TopK backend config: %s", config);
  }
  topk.set_raw_backend_config_string(config);
  return absl::OkStatus();
}

}  // namespace xla::gpu
