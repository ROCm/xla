/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/triton_configs.h"

#include <initializer_list>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/service/gpu/matmul_utils.h"

namespace xla {
namespace gpu {
namespace {

// TODO(b/467265599): Replace string constants with cc_embed_data when
// https://github.com/bazelbuild/rules_cc/issues/41 is fixed.

constexpr absl::string_view kBlackwellTritonConfigs = R"(
config { block_m: 128 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 1 num_stages: 1 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 8 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 16 split_k: 512 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 32 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 1 num_stages: 5 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 64 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 2 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 4 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 16 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 8 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 64 split_k: 8 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 32 block_k: 64 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 256 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 16 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 256 block_n: 32 block_k: 32 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 512 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 1 num_warps: 16 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 1 num_stages: 2 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 64 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 128 split_k: 8 num_stages: 1 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 1 num_stages: 1 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 16 block_k: 32 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 16 num_stages: 4 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 16 block_n: 16 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 16 block_n: 32 block_k: 64 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 256 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 256 block_n: 16 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 256 block_n: 32 block_k: 32 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 32 block_n: 16 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 64 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 64 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
)";

constexpr absl::string_view kDefaultCudaTritonConfigs = R"(
config { block_m: 32 block_n: 32 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 64 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 128 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 64 block_k: 128 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 128 block_k: 32 split_k: 8 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 512 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 512 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 1 num_stages: 2 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 32 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 128 block_k: 32 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 256 block_n: 128 block_k: 128 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 64 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 256 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 32 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 256 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 2 num_stages: 1 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 1 num_stages: 2 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 64 block_k: 256 split_k: 8 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 256 block_n: 256 block_k: 128 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
)";

constexpr absl::string_view kDefaultRocmTritonConfigs = R"(
config { block_m: 32 block_n: 32 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 64 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 128 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
)";

// MI300 (gfx942) configs from multiple sources:
//   1. Original ROCm defaults (kDefaultRocmTritonConfigs).
//   2. Per-GEMM-shape winners from exhaustive search on isolated GEMM HLOs.
//   3. Full-model winners from exhaustive search on the complete Llama 3.1 8B
//      HLO (BMM attention configs + additional large-GEMM variants).
//   4. Gemma2 2B (f32) exhaustive search winners on 1-GPU MI300X. Projection
//      GEMMs with N=31 (short sequence). Closes 23% wall-clock gap vs default.
//   5. Gemma3 1B (bf16) exhaustive search winners on 1-GPU MI300X. Small-model
//      GEMMs with N=1,11 (decode + short sequence). Adds small-tile coverage.
constexpr absl::string_view kMI300TritonConfigs = R"(
# --- Original ROCm defaults ---
config { block_m: 32 block_n: 32 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 64 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 128 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
# --- Per-GEMM-shape exhaustive search winners ---
config { block_m: 256 block_n: 256 block_k: 32 split_k: 1 num_stages: 2 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 2 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 32 split_k: 1 num_stages: 2 num_warps: 4 num_ctas: 1 }
config { block_m: 256 block_n: 128 block_k: 64 split_k: 1 num_stages: 2 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 1 num_stages: 2 num_warps: 4 num_ctas: 1 }
# --- Full-model exhaustive search winners ---
config { block_m: 32 block_n: 8 block_k: 16 split_k: 1 num_stages: 2 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 8 block_k: 16 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 32 block_k: 16 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 128 split_k: 1 num_stages: 2 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 2 num_stages: 2 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 128 block_k: 32 split_k: 1 num_stages: 2 num_warps: 4 num_ctas: 1 }
config { block_m: 256 block_n: 256 block_k: 32 split_k: 1 num_stages: 1 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 256 block_k: 32 split_k: 4 num_stages: 2 num_warps: 8 num_ctas: 1 }
# --- Gemma2 2B (f32) exhaustive search winners (1-GPU, MI300X) ---
# Projection GEMMs (repeated per layer, ~26 layers):
#   4096x31x2304 W_qkv: default 3.67ms -> exhaustive 1.57ms (-57%)
config { block_m: 128 block_n: 32 block_k: 32 split_k: 8 num_stages: 1 num_warps: 4 num_ctas: 1 }
#   4096x31x2304 W_qkv variant: -57%
config { block_m: 64 block_n: 32 block_k: 32 split_k: 8 num_stages: 3 num_warps: 2 num_ctas: 1 }
#   2048x31x2304 W_qkv small: default 3.91ms -> exhaustive 1.97ms (-50%)
config { block_m: 64 block_n: 32 block_k: 32 split_k: 8 num_stages: 5 num_warps: 2 num_ctas: 1 }
#   2304x31x9216 MLP down: default 2.35ms -> exhaustive 1.50ms (-36%)
config { block_m: 128 block_n: 32 block_k: 32 split_k: 32 num_stages: 2 num_warps: 4 num_ctas: 1 }
#   2304x31x2048 O proj: default 2.49ms -> exhaustive 1.49ms (-40%)
config { block_m: 32 block_n: 32 block_k: 32 split_k: 8 num_stages: 2 num_warps: 2 num_ctas: 1 }
#   9216x31x2304 MLP gate: default 3.66ms -> exhaustive 3.14ms (-14%)
config { block_m: 64 block_n: 32 block_k: 128 split_k: 2 num_stages: 2 num_warps: 2 num_ctas: 1 }
#   256000x1x2304 LM head: default 68.01ms -> exhaustive 52.42ms (-23%)
config { block_m: 256 block_n: 8 block_k: 32 split_k: 4 num_stages: 1 num_warps: 2 num_ctas: 1 }
# --- Gemma3 1B (bf16) exhaustive search winners (1-GPU, MI300X) ---
#   13824x11x1152 MLP gate/up
config { block_m: 128 block_n: 16 block_k: 128 split_k: 1 num_stages: 2 num_warps: 8 num_ctas: 1 }
#   1024x11x1152 W_qkv (A)
config { block_m: 32 block_n: 16 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
#   1024x11x1152 W_qkv (B)
config { block_m: 32 block_n: 16 block_k: 128 split_k: 2 num_stages: 5 num_warps: 2 num_ctas: 1 }
#   512x11x1152 MLP proj
config { block_m: 32 block_n: 16 block_k: 128 split_k: 1 num_stages: 2 num_warps: 4 num_ctas: 1 }
#   262144x1x1152 LM head (bf16)
config { block_m: 64 block_n: 8 block_k: 128 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
#   1152x11x1024 O/down proj
config { block_m: 32 block_n: 16 block_k: 256 split_k: 1 num_stages: 2 num_warps: 2 num_ctas: 1 }
#   9216x1x2304 MLP gate decode (bs=1)
config { block_m: 256 block_n: 8 block_k: 16 split_k: 8 num_stages: 2 num_warps: 2 num_ctas: 1 }
#   2304x1x9216 MLP down decode (bs=1)
config { block_m: 128 block_n: 8 block_k: 16 split_k: 32 num_stages: 1 num_warps: 2 num_ctas: 1 }
)";

constexpr absl::string_view kAmpereTritonConfigs = R"(
config { block_m: 16 block_n: 16 block_k: 64 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 128 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 16 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 256 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 32 block_k: 128 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 256 block_k: 32 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 256 block_k: 32 split_k: 16 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 16 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 4 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 16 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 128 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 128 split_k: 16 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 128 num_stages: 2 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 4 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 128 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 256 split_k: 16 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 128 split_k: 8 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 32 split_k: 8 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 32 block_k: 32 split_k: 8 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 8 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 8 block_k: 128 split_k: 2 num_stages: 3 num_warps: 4 num_ctas: 1 }
)";

constexpr absl::string_view kHopperTritonConfigs = R"(
config { block_m: 16 block_n: 16 block_k: 64 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 128 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 16 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 256 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 32 block_k: 128 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 256 block_k: 32 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 256 block_k: 32 split_k: 16 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 16 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 4 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 16 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 128 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 128 split_k: 16 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 128 num_stages: 2 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 4 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 128 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 256 split_k: 16 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 128 split_k: 8 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 32 split_k: 8 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 32 block_k: 32 split_k: 8 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 8 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 8 block_k: 128 split_k: 2 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 64 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 16 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 16 block_n: 256 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 32 block_n: 32 block_k: 128 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 32 block_n: 256 block_k: 32 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 32 block_n: 256 block_k: 32 split_k: 16 num_stages: 3 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 16 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 16 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 16 block_k: 128 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 16 block_k: 128 split_k: 16 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 64 block_k: 128 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 64 block_k: 256 split_k: 16 num_stages: 4 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
)";

absl::flat_hash_map<TritonConfigsPlatform, std::vector<TritonGemmConfig>>
LoadTritonConfigs() {
  absl::flat_hash_map<TritonConfigsPlatform, std::vector<TritonGemmConfig>>
      result;

  auto parse_config =
      [](absl::string_view config_str) -> std::vector<TritonGemmConfig> {
    TritonGemmConfigsProto proto;
    CHECK(tsl::protobuf::TextFormat::ParseFromString(config_str, &proto))
        << config_str;
    std::vector<TritonGemmConfig> configs;
    absl::c_transform(proto.config(), std::back_inserter(configs),
                      [](const AutotuneResult::TritonGemmKey& config_proto) {
                        absl::StatusOr<TritonGemmConfig> config =
                            TritonGemmConfig::FromProto(config_proto);
                        CHECK_OK(config);
                        return *config;
                      });
    return configs;
  };

  const std::initializer_list<
      std::pair<TritonConfigsPlatform, absl::string_view>>
      kConfigsMap = {
          {TritonConfigsPlatform::kAmpere, kAmpereTritonConfigs},
          {TritonConfigsPlatform::kBlackwell, kBlackwellTritonConfigs},
          {TritonConfigsPlatform::kDefaultCuda, kDefaultCudaTritonConfigs},
          {TritonConfigsPlatform::kDefaultRocm, kDefaultRocmTritonConfigs},
          {TritonConfigsPlatform::kHopper, kHopperTritonConfigs},
          {TritonConfigsPlatform::kMI300, kMI300TritonConfigs},
      };
  for (const auto& [platform, config_str] : kConfigsMap) {
    result[platform] = parse_config(config_str);
  }

  return result;
}

}  // namespace

const std::vector<TritonGemmConfig>& GetTritonConfigsForPlatform(
    TritonConfigsPlatform platform) {
  static const absl::NoDestructor<
      absl::flat_hash_map<TritonConfigsPlatform, std::vector<TritonGemmConfig>>>
      kConfigs(LoadTritonConfigs());
  return kConfigs->at(platform);
}

}  // namespace gpu
}  // namespace xla
