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

#ifndef XLA_SERVICE_GPU_CONV_UTILS_H_
#define XLA_SERVICE_GPU_CONV_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

std::optional<Window> RestoreWindowFromBackwardFilter(
    const HloConvolutionInstruction* conv);
std::optional<Window> RestoreWindowFromBackwardInput(
    const HloConvolutionInstruction* conv);
std::optional<Window> RestoreWindow(const HloConvolutionInstruction* conv);
ConvolutionDimensionNumbers RestoreDimNumberFromBackwardInput(
    const HloConvolutionInstruction* conv);
ConvolutionDimensionNumbers RestoreDimNumberFromBackwardFilter(
    const HloConvolutionInstruction* conv);
ConvolutionDimensionNumbers RestoreDimNumber(
    const HloConvolutionInstruction* conv);

// We should use this in code instead of AutotuneResult::TritonConvKey.
// This has some advantages, for example it can be used in hashmaps.
struct TritonConvConfig {
  TritonConvConfig() = default;
  TritonConvConfig(int64_t batch_tile,
                   std::vector<int64_t> output_spatial_tiles,
                   int64_t output_feature_tile,
                   std::vector<int64_t> kernel_spatial_tiles,
                   int64_t input_feature_tile, int64_t num_stages,
                   int64_t num_warps, int64_t num_ctas = 1,
                   int64_t waves_per_eu = 0)
      : batch_tile(batch_tile),
        output_spatial_tiles(std::move(output_spatial_tiles)),
        output_feature_tile(output_feature_tile),
        kernel_spatial_tiles(std::move(kernel_spatial_tiles)),
        input_feature_tile(input_feature_tile),
        num_stages(num_stages),
        num_warps(num_warps),
        num_ctas(num_ctas),
        waves_per_eu(waves_per_eu) {}
  // LINT.IfChange
  int64_t batch_tile = 0;
  std::vector<int64_t> output_spatial_tiles;
  int64_t output_feature_tile = 0;
  std::vector<int64_t> kernel_spatial_tiles;
  int64_t input_feature_tile = 0;
  int64_t num_stages = 1;
  int64_t num_warps = 1;
  // Number of blocks in a block cluster.
  int64_t num_ctas = 1;
  // Number of waves per execution unit (0 = no restriction).
  int64_t waves_per_eu = 0;
  // LINT.ThenChange(//tensorflow/compiler/xla/autotuning.proto)

  // When adding new members, please update all methods, such as ToTuple,
  // FromProto, ToProto, ToString, etc. Updating ToTuple is not enough.

 private:
  auto ToTuple() const {
    return std::make_tuple(batch_tile, output_spatial_tiles,
                           output_feature_tile, kernel_spatial_tiles,
                           input_feature_tile, num_stages, num_warps, num_ctas,
                           waves_per_eu);
  }

 public:
  // Creates a TritonConvConfig from the supplied proto, doing a simple sanity
  // check.
  static absl::StatusOr<TritonConvConfig> FromProto(
      const AutotuneResult::TritonConvKey& proto);
  AutotuneResult::TritonConvKey ToProto() const;

  std::string ToString() const;

  bool operator==(const TritonConvConfig& other) const {
    return ToTuple() == other.ToTuple();
  }
  bool operator<(const TritonConvConfig& other) const {
    return ToTuple() < other.ToTuple();
  }

  template <typename H>
  friend H AbslHashValue(H h, const TritonConvConfig& config) {
    return H::combine(std::move(h), config.ToTuple());
  }
};

}  // end namespace gpu
}  // end namespace xla

#endif  // XLA_SERVICE_GPU_CONV_UTILS_H_
