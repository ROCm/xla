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

#include "xla/backends/gpu/autotuner/triton/conv_search_space.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/MathExtras.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/conv_utils.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

int64_t Pad(int64_t value) {
  return static_cast<int64_t>(
      llvm::PowerOf2Ceil(static_cast<uint64_t>(std::max<int64_t>(value, 1))));
}

// Candidate values swept per axis, smallest first so the cheapest survivors
// appear early in the list.
constexpr int64_t kSpatialTileCandidates[] = {1, 2, 4, 8, 16};
constexpr int64_t kFeatureTileCandidates[] = {16, 32, 64, 128};
constexpr int64_t kInputFeatureTileCandidates[] = {8, 16, 32, 64};
constexpr int64_t kNumWarpsCandidates[] = {1, 4, 8};
constexpr int64_t kWavesPerEu = 0;

// Largest shared-memory figure observed for a channels-strided input across the
// search space
constexpr int64_t kChannelsStridedFloorBytes = 16384;

}  // namespace

int64_t EstimateSharedMemoryBytes(const TritonConvConfig& config,
                                  int64_t element_bytes,
                                  bool input_is_channels_minor) {
  int64_t m = Pad(config.batch_tile);
  for (int64_t tile : config.output_spatial_tiles) {
    m *= Pad(tile);
  }
  const int64_t n = Pad(config.output_feature_tile);
  int64_t k = Pad(config.input_feature_tile);
  for (int64_t tile : config.kernel_spatial_tiles) {
    k *= Pad(tile);
  }
  const int64_t stages = std::max<int64_t>(config.num_stages, 1);
  if (input_is_channels_minor) {
    return k * std::max(m, n) * element_bytes * stages;
  }
  return std::max(k * n * element_bytes * stages, kChannelsStridedFloorBytes);
}

TritonConvConfig GetFloorConvConfig(const HloConvolutionInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  TritonConvConfig config;
  config.batch_tile = 1;
  config.output_spatial_tiles.assign(dnums.output_spatial_dimensions_size(), 1);
  config.kernel_spatial_tiles.assign(dnums.kernel_spatial_dimensions_size(), 1);
  config.output_feature_tile = std::min<int64_t>(
      16, conv->shape().dimensions(dnums.output_feature_dimension()));
  config.input_feature_tile =
      std::min<int64_t>(8, conv->operand(1)->shape().dimensions(
                               dnums.kernel_input_feature_dimension()));
  config.output_feature_tile = std::max<int64_t>(config.output_feature_tile, 1);
  config.input_feature_tile = std::max<int64_t>(config.input_feature_tile, 1);
  config.num_stages = 1;
  config.num_warps = 1;
  config.num_ctas = 1;
  config.waves_per_eu = kWavesPerEu;
  return config;
}

TritonConvFusionSearchSpace::TritonConvFusionSearchSpace(
    const se::DeviceDescription& device_description,
    const HloConvolutionInstruction* conv)
    : conv_(conv),
      shared_memory_limit_(device_description.shared_memory_per_block_optin()) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  const Shape& output_shape = conv->shape();
  const Shape& kernel_shape = conv->operand(1)->shape();

  element_bytes_ = primitive_util::ByteWidth(output_shape.element_type());
  batch_size_ = output_shape.dimensions(dnums.output_batch_dimension());
  output_feature_size_ =
      output_shape.dimensions(dnums.output_feature_dimension());
  input_feature_size_ =
      kernel_shape.dimensions(dnums.kernel_input_feature_dimension());
  for (int64_t dim : dnums.output_spatial_dimensions()) {
    output_spatial_sizes_.push_back(output_shape.dimensions(dim));
  }

  // Autotuning runs after layout assignment, so the input operand's layout is
  // available here. If it is somehow absent, assume channels-minor.
  const Shape& input_shape = conv->operand(0)->shape();
  input_is_channels_minor_ =
      !input_shape.has_layout() ||
      input_shape.layout().minor_to_major(0) == dnums.input_feature_dimension();
}

std::vector<TritonConvConfig> TritonConvFusionSearchSpace::GenerateConfigs()
    const {
  const int64_t spatial_rank = output_spatial_sizes_.size();
  std::vector<TritonConvConfig> configs;

  // The floor config always comes first, so that even if every swept candidate
  // is filtered out the list is non-empty.
  configs.push_back(GetFloorConvConfig(conv_));

  for (int64_t spatial_tile : kSpatialTileCandidates) {
    for (int64_t feature_tile : kFeatureTileCandidates) {
      for (int64_t input_feature_tile : kInputFeatureTileCandidates) {
        for (int64_t num_warps : kNumWarpsCandidates) {
          TritonConvConfig config;
          config.batch_tile = 1;
          config.output_spatial_tiles.reserve(spatial_rank);
          for (int64_t size : output_spatial_sizes_) {
            config.output_spatial_tiles.push_back(
                std::max<int64_t>(std::min(spatial_tile, size), 1));
          }
          config.kernel_spatial_tiles.assign(spatial_rank, 1);
          config.output_feature_tile = std::max<int64_t>(
              std::min(feature_tile, output_feature_size_), 1);
          config.input_feature_tile = std::max<int64_t>(
              std::min(input_feature_tile, input_feature_size_), 1);
          config.num_stages = 1;
          config.num_warps = num_warps;
          config.num_ctas = 1;
          config.waves_per_eu = kWavesPerEu;

          if (EstimateSharedMemoryBytes(config, element_bytes_,
                                        input_is_channels_minor_) >
              shared_memory_limit_) {
            continue;
          }
          if (std::find(configs.begin(), configs.end(), config) !=
              configs.end()) {
            continue;
          }
          configs.push_back(config);
        }
      }
    }
  }
  return configs;
}

std::string TritonConvFusionSearchSpace::ToString() const {
  return absl::StrCat("{batch:", batch_size_, ",output_spatial:[",
                      absl::StrJoin(output_spatial_sizes_, ","),
                      "],output_feature:", output_feature_size_,
                      ",input_feature:", input_feature_size_,
                      ",element_bytes:", element_bytes_,
                      ",shared_memory_limit:", shared_memory_limit_,
                      ",channels_minor:", input_is_channels_minor_, "}");
}

}  // namespace xla::gpu
