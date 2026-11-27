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

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "re2/re2.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
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

// MI350 (gfx950): the Triton kernels are faster than the default path except
// in the regions below, measured on stable TOTAL-order TopKs of random data
// for rows of 2 to 2^24 elements, batches of 1 to 2^18 and k up to 128. Rows
// that are not a power of two were measured at the length just above one, where
// the kernels pad the most.
bool IsTritonTopKFasterOnMi350(const TopKKernelSpec& spec) {
  const bool bf16 = spec.input_type == BF16;
  if (!bf16 && spec.input_type != F32) {
    return false;
  }
  const int64_t batch = spec.batch, n = spec.n, k = spec.k;
  const bool padded = !absl::has_single_bit(static_cast<uint64_t>(n));
  // The k rounds scan the row padded to a power of two, up to twice its
  // length.
  if (padded && batch > 1 && batch <= 64 && n > 16 && n <= 256 && k >= 5 &&
      k <= 8) {
    return false;
  }
  if (padded && batch > 1 && n > 64 && n <= (bf16 ? 1024 : 512) && k >= 9 &&
      k <= 16) {
    return false;
  }
  // Rows of up to 16K keys fit one program, which runs the k rounds (up to
  // 128) over the whole padded row; longer rows are split into chunks that run
  // in parallel.
  if (bf16 && n > 4096 && n <= 16384 && k > 32 && k <= 64) {
    return false;
  }
  if (n > 8192 && n <= 16384 && k > 64 && batch < 16384) {
    return false;
  }
  // One long f32 row: merging the candidates of its chunks takes most of the
  // time.
  if (!bf16 && batch == 1 && n > (1 << 18) && k > 32) {
    return false;
  }
  return true;
}

int64_t NextPowerOfTwo(int64_t x) {
  return static_cast<int64_t>(absl::bit_ceil(static_cast<uint64_t>(x)));
}

// The routing, the chunk lengths and the selection in this file were measured
// on MI350 (gfx950, also MI355X). Other devices take the Triton TopK path only
// when it is forced, with chunks from 4096 up and the k rounds.
bool IsMi350(const se::DeviceDescription& device) {
  const se::RocmComputeCapability* rocm =
      device.gpu_compute_capability().rocm_compute_capability();
  return rocm != nullptr && rocm->gfx9_mi350();
}

// The longest row of `block_k` >= 32 keys that takes the bitonic selection.
int64_t MaxBitonicRowLength(int64_t block_k) { return 64 * block_k; }

// MI350 (gfx950): the chunk length of rows too long for one program, measured
// with one chunk per program on random data for rows of 2^15 to 2^22
// elements, batches of 1 to 1024 and k of 1 to 128. Each chunk leaves k
// candidates, which a merge program per row selects from.
int64_t GetChunkLengthOnMi350(const TopKKernelSpec& spec,
                              const se::DeviceDescription& device) {
  constexpr int64_t kMaxChunk = 16384;
  constexpr int64_t kTargetMergeKeys = 4096;
  // Up to this many elements in all rows, the merge takes most of the time.
  constexpr int64_t kFewRowsElements = int64_t{1} << 23;
  const int64_t block_k = NextPowerOfTwo(spec.k);
  const int64_t max_bitonic_n = MaxBitonicRowLength(block_k);
  if (block_k >= 32) {
    // The merge is fastest with rows of about 4096 candidates, as long as the
    // chunks keep the bitonic selection.
    const int64_t chunk = std::clamp(
        NextPowerOfTwo(CeilOfRatio(spec.n * spec.k, kTargetMergeKeys)),
        int64_t{2048}, max_bitonic_n);
    // With few rows, a merge past the bitonic selection takes most of the
    // time: chunks of 16384 when its rows would fill a program, otherwise the
    // chunks from 4096 up.
    const int64_t merge_n = NextPowerOfTwo(CeilOfRatio(spec.n, chunk) * spec.k);
    if (spec.batch * spec.n <= kFewRowsElements && merge_n > max_bitonic_n) {
      return merge_n >= kMaxChunk ? kMaxChunk : kTargetKeysPerProgram;
    }
    return chunk;
  }
  // Below k = 8 the merge is too short to be worth shrinking.
  if (spec.k < 8) {
    return kTargetKeysPerProgram;
  }
  // Longer chunks shrink the merge as long as the chunk programs fill the
  // device. Rows that are not a multiple of the chunk are slower with long
  // chunks.
  int64_t chunk = kMaxChunk;
  while (chunk > kTargetKeysPerProgram &&
         (spec.batch * CeilOfRatio(spec.n, chunk) < device.core_count() ||
          spec.n % chunk != 0)) {
    chunk /= 2;
  }
  return chunk;
}

}  // namespace

std::string TopKKernelSpec::ToString() const {
  return absl::StrFormat("%s[%d,%d] k=%d order=%s",
                         primitive_util::LowercasePrimitiveTypeName(input_type),
                         batch, n, k, ComparisonOrderToString(order));
}

std::string TopKKernelConfig::ToString() const {
  std::string s = absl::StrFormat(
      "block_n=%d rows_per_program=%d num_warps=%d num_programs=%d", block_n,
      rows_per_program, num_warps, num_programs);
  if (num_chunks > 1) {
    absl::StrAppendFormat(&s, " num_chunks=%d", num_chunks);
  }
  if (selection == TopKSelection::kBitonic) {
    absl::StrAppend(&s, " selection=bitonic");
  }
  return s;
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

absl::StatusOr<TopKKernelSpec> GetTopKKernelSpec(
    const HloTopKInstruction& topk) {
  const Shape& data_shape = topk.operand(0)->shape();
  if (data_shape.dimensions().empty()) {
    return InvalidArgument("TopK input must have rank 1 or more: %s",
                           data_shape.ToString());
  }
  const int64_t n = data_shape.dimensions().back();
  return TopKKernelSpec{
      data_shape.element_type(),
      ComparisonOrder::kTotal,
      n > 0 ? ShapeUtil::ElementsIn(data_shape) / n : 0,
      n,
      topk.k(),
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
  if (spec.batch < 1) {
    return InvalidArgument("batch=%d is not at least 1", spec.batch);
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

namespace {

// The configuration of a kernel over `num_rows` rows of `block_n` positions,
// with programs of about `target_keys_per_program` keys, or nullopt if a row
// does not fit one program on `device`.
std::optional<TopKKernelConfig> GetBlockConfig(
    int64_t num_rows, int64_t block_n, const se::DeviceDescription& device,
    int64_t target_keys_per_program = kTargetKeysPerProgram) {
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
  config.block_n = block_n;
  config.rows_per_program = std::clamp(target_keys_per_program / config.block_n,
                                       int64_t{1}, NextPowerOfTwo(num_rows));
  const int64_t keys_per_program = config.rows_per_program * config.block_n;
  config.num_warps =
      std::clamp(NextPowerOfTwo(CeilOfRatio(keys_per_program,
                                            wave_size * kTargetKeysPerThread)),
                 int64_t{1}, max_num_warps);
  config.num_programs = CeilOfRatio(num_rows, config.rows_per_program);

  if (CeilOfRatio(keys_per_program, config.num_warps * wave_size) >
      kMaxKeysPerThread) {
    return std::nullopt;
  }
  // Triton may stage a program's whole tile of 64-bit keys through LDS to
  // change its layout.
  const int64_t lds_bytes = keys_per_program * int64_t{8};
  if (lds_bytes > device.shared_memory_per_block()) {
    return std::nullopt;
  }
  if (config.num_programs > device.block_dim_limit().x) {
    return std::nullopt;
  }
  return config;
}

// Sets how `config`, which keeps the top `k` keys of each row, selects them.
// The k rounds grow with `k`; the bitonic selection pads `k` to a power of two
// and grows faster with `block_n`. The thresholds come from MI350 measurements:
// in each range of `k` below, the bitonic selection is at least as fast as the
// k rounds at both ends. Other devices use the k rounds.
TopKKernelConfig WithSelection(int64_t k, TopKKernelConfig config,
                               const se::DeviceDescription& device) {
  const int64_t block_k = NextPowerOfTwo(k);
  const bool bitonic =
      IsMi350(device) &&
      ((block_k >= 32 && config.block_n <= MaxBitonicRowLength(block_k)) ||
       (block_k == 16 && config.block_n <= 64));
  config.selection = bitonic ? TopKSelection::kBitonic : TopKSelection::kRounds;
  return config;
}

// The kChunk configuration of `spec` with one chunk of `chunk` elements per
// program, or nullopt if it or its merge step does not fit `device`.
std::optional<TopKKernelConfig> GetChunkConfig(
    const TopKKernelSpec& spec, int64_t chunk,
    const se::DeviceDescription& device) {
  const int64_t num_chunks = CeilOfRatio(spec.n, chunk);
  std::optional<TopKKernelConfig> config =
      GetBlockConfig(spec.batch * num_chunks, chunk, device, chunk);
  if (!config.has_value()) {
    return std::nullopt;
  }
  config->num_chunks = num_chunks;
  if (!GetTopKMergeConfig(spec, *config, device).has_value()) {
    return std::nullopt;
  }
  return WithSelection(spec.k, *config, device);
}

}  // namespace

std::optional<TopKKernelConfig> GetTopKKernelConfig(
    const TopKKernelSpec& spec, const se::DeviceDescription& device) {
  const int64_t block_n = NextPowerOfTwo(spec.n);
  if (std::optional<TopKKernelConfig> single =
          GetBlockConfig(spec.batch, block_n, device)) {
    return WithSelection(spec.k, *single, device);
  }
  // Rows too long for one program are split into chunks.
  if (IsMi350(device)) {
    const int64_t chunk = GetChunkLengthOnMi350(spec, device);
    if (chunk < block_n) {
      if (std::optional<TopKKernelConfig> config =
              GetChunkConfig(spec, chunk, device)) {
        return config;
      }
    }
  }
  // Longer chunks leave fewer candidates for the merge step, whose row must
  // fit one program.
  for (int64_t chunk = kTargetKeysPerProgram; chunk < block_n; chunk *= 2) {
    if (std::optional<TopKKernelConfig> config =
            GetChunkConfig(spec, chunk, device)) {
      return config;
    }
  }
  return std::nullopt;
}

TopKKernelSpec GetTopKMergeSpec(const TopKKernelSpec& spec,
                                const TopKKernelConfig& config) {
  TopKKernelSpec merge = spec;
  merge.n = config.num_chunks * spec.k;
  return merge;
}

std::optional<TopKKernelConfig> GetTopKMergeConfig(
    const TopKKernelSpec& spec, const TopKKernelConfig& config,
    const se::DeviceDescription& device) {
  if (config.num_chunks <= 1) {
    return std::nullopt;
  }
  std::optional<TopKKernelConfig> merge = GetBlockConfig(
      spec.batch, NextPowerOfTwo(GetTopKMergeSpec(spec, config).n), device);
  if (!merge.has_value()) {
    return std::nullopt;
  }
  return WithSelection(spec.k, *merge, device);
}

int64_t GetTopKScratchElements(const TopKKernelSpec& spec,
                               const TopKKernelConfig& config) {
  return config.num_chunks > 1 ? spec.batch * config.num_chunks * spec.k : 0;
}

std::optional<TopKKernelConfig> GetTopKKernelConfigForScratch(
    const TopKKernelSpec& spec, int64_t scratch_elements,
    const se::DeviceDescription& device) {
  if (scratch_elements == 0) {
    std::optional<TopKKernelConfig> single =
        GetBlockConfig(spec.batch, NextPowerOfTwo(spec.n), device);
    if (!single.has_value()) {
      return std::nullopt;
    }
    return WithSelection(spec.k, *single, device);
  }
  if (scratch_elements % (spec.batch * spec.k) != 0) {
    return std::nullopt;
  }
  const int64_t num_chunks = scratch_elements / (spec.batch * spec.k);
  if (num_chunks <= 1) {
    return std::nullopt;
  }
  // Chunk lengths are powers of two, and only one of them splits a row into
  // `num_chunks` chunks.
  const int64_t chunk = NextPowerOfTwo(CeilOfRatio(spec.n, num_chunks));
  // Each chunk keeps its k largest keys.
  if (CeilOfRatio(spec.n, chunk) != num_chunks ||
      chunk < NextPowerOfTwo(spec.k)) {
    return std::nullopt;
  }
  return GetChunkConfig(spec, chunk, device);
}

bool IsTritonTopKFaster(const TopKKernelSpec& spec,
                        const se::DeviceDescription& device) {
  return IsMi350(device) && IsTritonTopKFasterOnMi350(spec);
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
