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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_TRITON_CONV_SEARCH_SPACE_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_TRITON_CONV_SEARCH_SPACE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout.h"
#include "xla/service/gpu/conv_utils.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Estimates the shared memory in bytes a Triton convolution kernel needs.
// Due to OOM issues when input feature axis is minormost there are two regimes,
// selected by `input_is_channels_minor`:
//   channels-minor:    K * max(M, N) * element_bytes * num_stages
//   otherwise:  max(K * N * element_bytes * num_stages, 16384)
//   where:
//     M = n*h_out*w_out
//     N = c_out
//     K =  c_in*kw*kh
int64_t EstimateSharedMemoryBytes(const TritonConvConfig& config,
                                  int64_t element_bytes,
                                  bool input_is_channels_minor);

// Returns a deliberately small config for `conv`, sized so that
// EstimateSharedMemoryBytes stays well inside any device's budget.
TritonConvConfig GetFloorConvConfig(const HloConvolutionInstruction* conv);

// Generates tiling candidates for a Triton convolution fusion.
//
// (TODO) Every kernel spatial tile is fixed at 1. Lifting
// that restriction is a change to this class, not to TritonConvKey.
class TritonConvFusionSearchSpace {
 public:
  TritonConvFusionSearchSpace(const se::DeviceDescription& device_description,
                              const HloConvolutionInstruction* conv);

  // Returns the candidate list. Never empty: the floor config is always
  // present.
  std::vector<TritonConvConfig> GenerateConfigs() const;

  std::string ToString() const;

 private:
  const HloConvolutionInstruction* conv_;
  int64_t shared_memory_limit_;
  int64_t element_bytes_;
  int64_t batch_size_;
  std::vector<int64_t> output_spatial_sizes_;
  int64_t output_feature_size_;
  int64_t input_feature_size_;
  bool input_is_channels_minor_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_TRITON_CONV_SEARCH_SPACE_H_
