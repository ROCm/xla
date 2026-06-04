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

#include "xla/backends/gpu/autotuner/triton.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/triton/cost_model_config_optimization.h"
#include "xla/backends/gpu/autotuner/triton/dot_search_space.h"
#include "xla/backends/gpu/autotuner/triton/triton_configs.h"
#include "xla/backends/gpu/transforms/convert_triton_gemm_config.h"
#include "xla/backends/gpu/transforms/fusion_wrapper.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {
std::vector<TritonGemmConfig> GetDefaultTritonConfigs(
    se::GpuComputeCapability compute_capability) {
  if (compute_capability.IsRocm()) {
    const auto* rocm_cc = compute_capability.rocm_compute_capability();
    if (rocm_cc->gfx9_mi300()) {
      return GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI300);
    }
    if (rocm_cc->gfx9_mi350()) {
      return GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI350);
    }
    return GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultRocm);
  }

  CHECK(compute_capability.IsCuda());
  auto* cuda_compute_capability = compute_capability.cuda_compute_capability();
  std::vector<TritonGemmConfig> configs;

  if (cuda_compute_capability->IsBlackwell()) {
    // SM 10.0 (datacenter: B200, B100)
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwell);
  } else if (cuda_compute_capability->IsAtLeastBlackwell()) {
    // SM 11.0+ / 12.0+ (consumer: RTX 5090, etc.)
    configs =
        GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwellConsumer);
  } else if (cuda_compute_capability->IsHopper()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kHopper);
  } else if (cuda_compute_capability->IsAmpere()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kAmpere);
  } else {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultCuda);
  }

  return configs;
}

bool IsWarpSpecializationAvailable(
    se::GpuComputeCapability compute_capability) {
  return compute_capability.IsCuda() &&
         compute_capability.cuda_compute_capability()->IsAtLeastBlackwell();
}

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<BackendConfig>> overridden_configs,
      GetOverriddenConfigs(&instr));
  if (!overridden_configs.empty()) {
    return overridden_configs;
  }

  const HloComputation* fused_comp = instr.fused_instructions_computation();

  const HloInstruction* dot_instr =
      hlo_query::GetFirstInstructionWithOpcode(*fused_comp, HloOpcode::kDot);
  if (dot_instr != nullptr) {
    return GetSupportedConfigsForDot(dot_instr);
  }
  const HloInstruction* scaled_dot_instr =
      hlo_query::GetFirstInstructionWithOpcode(*fused_comp,
                                               HloOpcode::kScaledDot);
  if (scaled_dot_instr != nullptr) {
    return GetSupportedConfigsForScaledDot(scaled_dot_instr);
  }
  // kRaggedDot fusions routed through the Triton XTile backend
  // (kTritonFusionKind / "__triton").
  const HloInstruction* ragged_dot_instr =
      hlo_query::GetFirstInstructionWithOpcode(*fused_comp,
                                               HloOpcode::kRaggedDot);
  if (ragged_dot_instr != nullptr) {
    return GetSupportedConfigsForRaggedDot(ragged_dot_instr);
  }
  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigsForDot(const HloInstruction* instr) {
  const HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
  TritonDotFusionSearchSpace search_space(target_config().device_description,
                                          dot);
  bool autotune_warp_specialization =
      debug_options()
          .xla_gpu_experimental_enable_triton_warp_specialization() &&
      IsWarpSpecializationAvailable(
          target_config().device_description.gpu_compute_capability());

  std::vector<std::unique_ptr<BackendConfig>> configs;
  VLOG(1) << "Generating configs from search space: "
          << search_space.ToString();
  // We don't need to consider small_dot here. The new search space will
  // already generate a unique config for small problems.
  std::vector<TritonGemmConfig> gemm_configs = search_space.GenerateConfigs(
      /*autotune_warp_specialization=*/autotune_warp_specialization);

  if (!debug_options().xla_gpu_exhaustive_tiling_search()) {
    VLOG(1) << "Restricting configs to the default set.";
    std::vector<TritonGemmConfig> all_configs = gemm_configs;
    gemm_configs = search_space.OptimizeConfigSet(
        gemm_configs, /*hints=*/GetDefaultTritonConfigs(
            target_config().device_description.gpu_compute_capability()));

    if (!debug_options()
             .xla_gpu_experimental_cost_model_gemm_tiling_options()
             .empty()) {
      ASSIGN_OR_RETURN(gemm_configs, OptimizeConfigsWithCostModel(
                                         dot, all_configs, gemm_configs,
                                         target_config().device_description,
                                         debug_options(), mlir_context_));
    }
  }
  configs.reserve(gemm_configs.size());
  for (const auto& gemm_config : gemm_configs) {
    auto config = std::make_unique<BackendConfig>();
    *config->mutable_triton() = gemm_config.ToProto();
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigsForScaledDot(const HloInstruction* instr) {
  // The ROCm Triton backend does not support mixed FP4/FP8 scaled-dot inputs.
  const auto& gpu_cc =
      target_config().device_description.gpu_compute_capability();
  if (gpu_cc.IsRocm()) {
    PrimitiveType lhs_type = instr->operand(0)->shape().element_type();
    PrimitiveType rhs_type = instr->operand(1)->shape().element_type();
    auto is_fp4 = [](PrimitiveType t) { return t == F4E2M1FN; };
    auto is_fp8 = [](PrimitiveType t) { return t == F8E4M3FN || t == F8E5M2; };
    if ((is_fp4(lhs_type) && is_fp8(rhs_type)) ||
        (is_fp8(lhs_type) && is_fp4(rhs_type))) {
      return std::vector<std::unique_ptr<BackendConfig>>();
    }
  }

  std::vector<std::unique_ptr<BackendConfig>> configs;

  const bool exhaustive_search =
      debug_options().xla_gpu_exhaustive_tiling_search();
  for (int block_m = 128; block_m <= 256; block_m *= 2) {
    for (int block_n = 16; block_n <= 256; block_n *= 2) {
      for (int block_k = 128; block_k <= 256; block_k *= 2) {
        // TODO(b/436988479): fine tune the search space.
        const int elements_per_thread = (block_m * block_n) / (4 * 32);
        if (!exhaustive_search &&
            (elements_per_thread > 64 ||
             (block_k >= 256 && elements_per_thread >= 32))) {
          VLOG(3) << "Ignoring spill over config: block_m=" << block_m
                  << " block_n=" << block_n << " block_k=" << block_k;
          continue;
        }

        auto config = std::make_unique<BackendConfig>();
        *config->mutable_triton() = TritonGemmConfig(block_m, block_n,
                                                     /*block_k=*/block_k,
                                                     /*num_stages=*/1,
                                                     /*num_warps=*/4,
                                                     /*num_ctas=*/1,
                                                     /*is_tma_allowed=*/false)
                                        .ToProto();
        configs.push_back(std::move(config));
      }
    }
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetOverriddenConfigs(const HloInstruction* instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::string& override_file =
      debug_options().xla_gpu_gemm_autotuner_override_file();
  if (!override_file.empty()) {
    std::string file_content;
    RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), override_file,
                                          &file_content));
    TritonGemmConfigsProto gemm_configs;
    if (!tsl::protobuf::TextFormat::ParseFromString(file_content,
                                                    &gemm_configs)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Could not parse override file: ", override_file));
    }
    configs.reserve(gemm_configs.config_size());
    for (const auto& gemm_config : gemm_configs.config()) {
      auto config = std::make_unique<BackendConfig>();
      *config->mutable_triton() = gemm_config;
      configs.push_back(std::move(config));
    }
  }
  if (!debug_options().xla_gpu_override_gemm_autotuner().empty()) {
    AutotuneResult::TritonGemmKey gemm_config;
    CHECK(tsl::protobuf::TextFormat::ParseFromString(
        debug_options().xla_gpu_override_gemm_autotuner(), &gemm_config));
    auto config = std::make_unique<BackendConfig>();
    *config->mutable_triton() = gemm_config;
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> TritonBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                   GetSupportedConfigs(instr));

  if (configs.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("TritonBackend has no supported configs for '",
                     instr.name(), "' instruction"));
  }
  return std::move(configs[0]);
}

absl::Status TritonBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "TritonBackend does not support this instruction.");
  }
  if (!config.has_triton()) {
    return absl::InvalidArgumentError(
        "Expected TritonGemmKey config for TritonBackend.");
  }
  const AutotuneResult::TritonGemmKey& triton_config_proto = config.triton();

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();

  // Detect ragged-dot XTile fusions by the presence of kRaggedDot inside the
  // fused computation.
  HloComputation* fused_comp = instr.fused_instructions_computation();
  HloInstruction* inner_ragged_dot = nullptr;
  for (HloInstruction* inner : fused_comp->instructions()) {
    if (inner->opcode() == HloOpcode::kRaggedDot) {
      inner_ragged_dot = inner;
      break;
    }
  }

  if (inner_ragged_dot != nullptr) {
    // kRaggedDot XTile path — update:
    //   (1) the fusion-level BlockLevelFusionConfig output tiles, and
    //   (2) the inner ragged-dot's Tile backend config (sequential G=1, K dims
    //       read by GetTilingSpaceConcreteSizes).
    const auto* ragged_dot = Cast<HloRaggedDotInstruction>(inner_ragged_dot);
    const DotDimensionNumbers& dot_dims =
        ragged_dot->ragged_dot_dimension_numbers().dot_dimension_numbers();

    BlockLevelFusionConfig* blk_cfg =
        backend_config.mutable_block_level_fusion_config();
    blk_cfg->clear_output_tiles();
    auto* output_tile = blk_cfg->add_output_tiles();
    // Batch dims each use tile size 1 (they become grid dimensions, matching
    // the HandleRaggedDot convention).
    for (int64_t b_dim : dot_dims.lhs_batch_dimensions()) {
      (void)b_dim;
      output_tile->add_sizes(1);
    }
    output_tile->add_sizes(triton_config_proto.block_m());
    output_tile->add_sizes(triton_config_proto.block_n());
    blk_cfg->set_num_warps(triton_config_proto.num_warps());
    blk_cfg->set_num_ctas(std::max(1, static_cast<int>(triton_config_proto.num_ctas())));
    blk_cfg->set_num_stages(triton_config_proto.num_stages());
    RETURN_IF_ERROR(instr.set_backend_config(gpu_config));

    // Update the inner ragged-dot's Tile: sizes = [G_tile=1, K_tile].
    Tile tile_config;
    tile_config.add_sizes(1);                              // G tile always 1
    tile_config.add_sizes(triton_config_proto.block_k());  // K tile
    RETURN_IF_ERROR(inner_ragged_dot->set_backend_config(tile_config));
    return absl::OkStatus();
  }

  // Regular dot / scaled-dot path: write a TritonGemmConfig into the fusion.
  backend_config.set_kind(kTritonGemmFusionKind);
  *backend_config.mutable_triton_gemm_config() = triton_config_proto;
  RETURN_IF_ERROR(instr.set_backend_config(gpu_config));

  // FromProto has validation checks, that's why we call it here.
  RETURN_IF_ERROR(TritonGemmConfig::FromProto(triton_config_proto).status());
  if (triton_config_proto.split_k() > 1) {
    return absl::InvalidArgumentError(
        "TritonBackend no longer supports split-k (split_k > 1).");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigsForRaggedDot(const HloInstruction* instr) {
  const auto* ragged_dot = Cast<HloRaggedDotInstruction>(instr);
  const RaggedDotDimensionNumbers& ragged_dims =
      ragged_dot->ragged_dot_dimension_numbers();
  const DotDimensionNumbers& dot_dims = ragged_dims.dot_dimension_numbers();

  // Extract problem dimensions for pruning the search space.
  const Shape& lhs_shape = ragged_dot->operand(0)->shape();
  const Shape& rhs_shape = ragged_dot->operand(1)->shape();

  const int64_t lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
  const int64_t M_total = lhs_shape.dimensions(lhs_ragged_dim);
  const int64_t K_dim =
      lhs_shape.dimensions(dot_dims.lhs_contracting_dimensions(0));

  // N is the non-contracting, non-group, non-batch RHS dimension.
  int64_t N_dim = 1;
  {
    auto is_non_n_dim = [&](int64_t d) -> bool {
      for (int64_t x : dot_dims.rhs_contracting_dimensions())
        if (x == d) return true;
      for (int64_t x : ragged_dims.rhs_group_dimensions())
        if (x == d) return true;
      for (int64_t x : dot_dims.rhs_batch_dimensions())
        if (x == d) return true;
      return false;
    };
    for (int64_t i = 0; i < static_cast<int64_t>(rhs_shape.dimensions_size());
         ++i) {
      if (!is_non_n_dim(i)) N_dim = rhs_shape.dimensions(i);
    }
  }

  const bool exhaustive_search =
      debug_options().xla_gpu_exhaustive_tiling_search();

  // Search space: {16,32,64,128}³ block sizes × {4,8} num_warps × {1,2}
  // num_stages.  Pruned by dimension sizes to avoid obviously invalid configs.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  for (int block_m : {16, 32, 64, 128}) {
    if (!exhaustive_search && block_m > M_total) continue;
    for (int block_n : {16, 32, 64, 128}) {
      if (!exhaustive_search && block_n > N_dim) continue;
      for (int block_k : {16, 32, 64, 128}) {
        if (!exhaustive_search && block_k > K_dim) continue;
        for (int num_warps : {4, 8}) {
          for (int num_stages : {1, 2}) {
            if (!exhaustive_search) {
              // Skip configs where M×N tile is too large per warp thread to
              // avoid register spilling (~64 elements/thread cutoff).
              int64_t elems_per_thread =
                  static_cast<int64_t>(block_m) * block_n / (num_warps * 32);
              if (elems_per_thread > 64) continue;
            }
            auto config = std::make_unique<BackendConfig>();
            *config->mutable_triton() =
                TritonGemmConfig(block_m, block_n, block_k,
                                 /*num_stages=*/num_stages,
                                 /*num_warps=*/num_warps,
                                 /*num_ctas=*/1,
                                 /*is_tma_allowed=*/false)
                    .ToProto();
            configs.push_back(std::move(config));
          }
        }
      }
    }
  }

  if (configs.empty()) {
    // Fallback: always emit at least the default config used by HandleRaggedDot
    // (BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4, num_stages=1).
    auto config = std::make_unique<BackendConfig>();
    *config->mutable_triton() =
        TritonGemmConfig(32, 32, 32, /*num_stages=*/1, /*num_warps=*/4,
                         /*num_ctas=*/1, /*is_tma_allowed=*/false)
            .ToProto();
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<HloModule>> TritonBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  auto gpu_device_info = target_config().device_description;
  for (PrimitiveType type :
       {BF16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ}) {
    GpuFloatSupport float_support(gpu_device_info.gpu_compute_capability(),
                                  type);
    FloatNormalization float_normalization(&float_support);
    RETURN_IF_ERROR(float_normalization.Run(hlo_module.get()).status());
  }

  HloCostAnalysis::Options priority_fusion_options;
  priority_fusion_options.count_multiple_input_accesses = true;
  PriorityFusion priority_fusion(
      /*thread_pool=*/nullptr, gpu_device_info, alias_info_,
      priority_fusion_options, mlir_context_);
  RETURN_IF_ERROR(priority_fusion.Run(hlo_module.get()).status());

  // If the priority fusion pass above skipped some instructions, turn them
  // into fusions.
  FusionWrapper fusion_wrapper(gpu_device_info);
  RETURN_IF_ERROR(fusion_wrapper.Run(hlo_module.get()).status());
  ConvertTritonGemmConfig convert_triton_gemm_config(gpu_device_info,
                                                     mlir_context_);
  RETURN_IF_ERROR(convert_triton_gemm_config.Run(hlo_module.get()).status());
  return hlo_module;
}

bool TritonBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  const FusionBackendConfig& backend_config =
      gpu_config->fusion_backend_config();

  // kTritonFusionKind ("__triton") may be used by multiple XTile fusion types.
  // We support autotuning for kRaggedDot fusions specifically.
  if (backend_config.kind() == kTritonFusionKind) {
    const HloInstruction* ragged_dot_instr =
        hlo_query::GetFirstInstructionWithOpcode(
            *instr.fused_instructions_computation(), HloOpcode::kRaggedDot);
    if (ragged_dot_instr != nullptr &&
        instr.GetModule()
            ->config()
            .debug_options()
            .xla_gpu_experimental_enable_tiling_propagation()) {
      return true;
    }
  }

  // TODO: b/487920266 - sometimes we create fusions that can't be tiled.
  // Bail out here if that's the case.
  if (backend_config.kind() == kTritonGemmFusionKind) {
    auto fusion = Cast<HloFusionInstruction>(&instr);
    std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
        HloFusionAdaptor::ForInstruction(fusion);
    if (instr.GetModule()
            ->config()
            .debug_options()
            .xla_gpu_experimental_enable_tiling_propagation()) {
      std::unique_ptr<experimental::TilingSpace> ts =
          experimental::TilingSpace::Create(*fusion_adaptor, mlir_context_);
      auto tiled_computation_or = experimental::TiledHloComputation::Tile(
          *fusion_adaptor, std::move(ts));
      if (!tiled_computation_or.ok()) {
        VLOG(1) << "Fusion is not tileable with experimental tiling: "
                << tiled_computation_or.status().message();
        return false;
      }
      // We don't have concrete tile sizes here and don't validate Triton
      // constraints here.
      return true;
    }

    auto device_info = target_config().device_description;
    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeFusion(
            *fusion_adaptor, mlir_context_,
            TritonEmitterConstraints::GetBuilder(device_info));
    if (const auto* fusion_decision =
            std::get_if<FusionDecision>(&analysis_or_error)) {
      VLOG(1) << "Fusion not tileable: " << fusion_decision->Explain();
      return false;
    }
    return true;
  }
  return backend_config.kind() == kCuDnnFusionKind ||
         backend_config.kind() == kCustomFusionKind;
}

}  // namespace gpu
}  // namespace xla
