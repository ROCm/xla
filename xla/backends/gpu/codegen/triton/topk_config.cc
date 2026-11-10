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
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
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

// MI350 (gfx950): regions where the Triton kernel took at most 1/1.1 of the
// kernel time of the default path at every measured point: each power of two
// `n` alone, the rows between two powers of two at their shortest length, the
// largest `k` of each `k` range, and the batches at both ends of the range.
// Stable TOTAL-order TopKs on random data, rocprofv3 kernel times.
constexpr TopKSpeedRegion kMi350SpeedTable[] = {
    {F32, 64, 262144, 2, 2, 1, 1},
    {F32, 64, 262144, 3, 3, 1, 2},
    {F32, 1, 262144, 4, 4, 1, 2},
    {F32, 64, 262144, 4, 4, 3, 3},
    {F32, 1, 262144, 5, 7, 1, 4},
    {F32, 1, 262144, 8, 8, 1, 6},
    {F32, 1, 262144, 9, 15, 1, 4},
    {F32, 1024, 262144, 9, 15, 5, 7},
    {F32, 1, 262144, 16, 16, 1, 8},
    {F32, 64, 262144, 16, 16, 9, 13},
    {F32, 1, 262144, 17, 31, 1, 4},
    {F32, 1024, 262144, 17, 31, 5, 8},
    {F32, 1024, 16384, 17, 31, 9, 14},
    {F32, 1, 262144, 32, 32, 1, 8},
    {F32, 64, 262144, 32, 32, 9, 16},
    {F32, 1024, 262144, 32, 32, 17, 27},
    {F32, 1, 262144, 33, 63, 1, 4},
    {F32, 1024, 16384, 33, 63, 5, 16},
    {F32, 1, 262144, 64, 64, 1, 16},
    {F32, 1024, 262144, 64, 64, 17, 32},
    {F32, 1024, 16384, 64, 64, 33, 54},
    {F32, 1, 262144, 65, 127, 1, 2},
    {F32, 1024, 262144, 65, 127, 3, 8},
    {F32, 1024, 16384, 65, 127, 9, 16},
    {F32, 1, 262144, 128, 128, 1, 16},
    {F32, 1024, 262144, 128, 128, 17, 32},
    {F32, 1024, 16384, 128, 128, 33, 64},
    {F32, 1, 262144, 129, 255, 1, 4},
    {F32, 1024, 16384, 129, 255, 5, 8},
    {F32, 1, 262144, 256, 256, 1, 16},
    {F32, 1024, 262144, 256, 256, 17, 32},
    {F32, 1, 262144, 257, 511, 1, 8},
    {F32, 1, 262144, 512, 512, 1, 16},
    {F32, 1024, 262144, 512, 512, 17, 32},
    {F32, 1, 16384, 513, 1023, 1, 16},
    {F32, 1024, 131072, 1024, 1024, 1, 1},
    {F32, 1, 131072, 1024, 1024, 2, 16},
    {F32, 1024, 131072, 1024, 1024, 17, 32},
    {F32, 1, 130944, 1025, 2047, 1, 1},
    {F32, 1024, 130944, 1025, 2047, 2, 2},
    {F32, 1, 130944, 1025, 2047, 3, 16},
    {F32, 1, 1024, 1025, 2047, 17, 32},
    {F32, 1, 65536, 2048, 2048, 1, 16},
    {F32, 1, 65504, 2049, 4095, 1, 32},
    {F32, 64, 1024, 2049, 4095, 33, 64},
    {F32, 1, 32768, 4096, 4096, 1, 32},
    {F32, 1, 32760, 4097, 8191, 1, 64},
    {F32, 1, 16384, 8192, 8192, 1, 32},
    {F32, 64, 16384, 8192, 8192, 33, 64},
    {F32, 1, 16382, 8193, 16383, 1, 64},
    {F32, 64, 16382, 8193, 16383, 65, 128},
    {F32, 1, 8192, 16384, 16384, 1, 16},
    {F32, 64, 8192, 16384, 16384, 17, 64},
    {BF16, 1, 262144, 2, 2, 1, 1},
    {BF16, 1, 262144, 3, 3, 1, 2},
    {BF16, 1, 262144, 4, 4, 1, 3},
    {BF16, 1, 262144, 5, 7, 1, 4},
    {BF16, 1, 262144, 8, 8, 1, 6},
    {BF16, 1, 262144, 9, 15, 1, 4},
    {BF16, 16384, 262144, 9, 15, 5, 7},
    {BF16, 1, 262144, 16, 16, 1, 8},
    {BF16, 1, 64, 16, 16, 9, 13},
    {BF16, 16384, 262144, 16, 16, 9, 13},
    {BF16, 1, 262144, 17, 31, 1, 4},
    {BF16, 1024, 16384, 17, 31, 5, 8},
    {BF16, 1, 262144, 32, 32, 1, 16},
    {BF16, 1024, 16384, 32, 32, 17, 27},
    {BF16, 1, 262144, 33, 63, 1, 4},
    {BF16, 1024, 262144, 33, 63, 5, 8},
    {BF16, 1024, 16384, 33, 63, 9, 16},
    {BF16, 1, 262144, 64, 64, 1, 8},
    {BF16, 1024, 262144, 64, 64, 9, 32},
    {BF16, 1, 262144, 65, 127, 1, 2},
    {BF16, 1024, 16384, 65, 127, 3, 8},
    {BF16, 1, 262144, 128, 128, 1, 8},
    {BF16, 1024, 262144, 128, 128, 9, 16},
    {BF16, 1024, 16384, 128, 128, 17, 32},
    {BF16, 1, 262144, 129, 255, 1, 4},
    {BF16, 1024, 262144, 129, 255, 5, 8},
    {BF16, 1, 262144, 256, 256, 1, 16},
    {BF16, 1024, 262144, 256, 256, 17, 32},
    {BF16, 1, 262144, 257, 511, 1, 4},
    {BF16, 1, 16384, 257, 511, 5, 8},
    {BF16, 1, 262144, 512, 512, 1, 16},
    {BF16, 64, 262144, 512, 512, 17, 32},
    {BF16, 1, 16384, 513, 1023, 1, 8},
    {BF16, 1, 131072, 1024, 1024, 1, 8},
    {BF16, 1024, 131072, 1024, 1024, 9, 16},
    {BF16, 1, 130944, 1025, 2047, 1, 16},
    {BF16, 1, 65536, 2048, 2048, 1, 8},
    {BF16, 1024, 65536, 2048, 2048, 9, 16},
    {BF16, 1, 65504, 2049, 4095, 1, 32},
    {BF16, 1, 32768, 4096, 4096, 1, 32},
    {BF16, 1, 32760, 4097, 8191, 1, 32},
    {BF16, 1, 64, 4097, 8191, 33, 64},
    {BF16, 1, 16384, 8192, 8192, 1, 16},
    {BF16, 64, 16384, 8192, 8192, 17, 32},
    {BF16, 1, 16382, 8193, 16383, 1, 64},
    {BF16, 1, 8192, 16384, 16384, 1, 16},
    {BF16, 64, 8192, 16384, 16384, 17, 32},
};

absl::Span<const TopKSpeedRegion> GetTopKSpeedTable(
    const se::DeviceDescription& device) {
  const se::RocmComputeCapability* rocm =
      device.gpu_compute_capability().rocm_compute_capability();
  if (rocm != nullptr && rocm->gfx9_mi350()) {
    return kMi350SpeedTable;
  }
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
