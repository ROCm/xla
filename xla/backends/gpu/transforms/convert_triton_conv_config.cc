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

#include "xla/backends/gpu/transforms/convert_triton_conv_config.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/conv_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::xla::xtile::BlockLevelParameters;

// Extracts the TritonConvConfig from the given fusion's backend config.
absl::StatusOr<TritonConvConfig> GetTritonConvConfig(
    const HloFusionInstruction& fusion) {
  ABSL_ASSIGN_OR_RETURN(auto gpu_config,
                        fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  if (!backend_config.has_triton_conv_config()) {
    return absl::NotFoundError(
        "The fusion's backend config doesn't have a triton_conv_config.");
  }
  return TritonConvConfig::FromProto(backend_config.triton_conv_config());
}

class ConvertTritonConvConfigVisitor : public DfsHloRewriteVisitor {
 private:
  absl::Status HandleFusion(HloInstruction* instruction) override {
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(instruction);
    // Check if we target this fusion.
    absl::StatusOr<TritonConvConfig> config = GetTritonConvConfig(*fusion);
    if (absl::IsNotFound(config.status())) {
      VLOG(2) << "Skipping fusion as it does not have a TritonConvConfig";
      return absl::OkStatus();
    }
    ABSL_RETURN_IF_ERROR(config.status());
    return RewriteFusion(fusion, *config);
  }

  absl::Status RewriteFusion(HloFusionInstruction* fusion,
                             const TritonConvConfig& config) {
    HloComputation* computation = fusion->called_computation();
    HloInstruction* root = computation->root_instruction();

    if (root->opcode() != HloOpcode::kConvolution) {
      return absl::InternalError(
          absl::StrCat("Expected the root of a ", kTritonConvFusionKind,
                       " fusion to be a convolution, got ", root->ToString()));
    }
    HloConvolutionInstruction* conv = Cast<HloConvolutionInstruction>(root);
    const ConvolutionDimensionNumbers& dnums =
        conv->convolution_dimension_numbers();

    const int64_t spatial_rank = dnums.output_spatial_dimensions_size();
    TF_RET_CHECK(config.output_spatial_tiles.size() ==
                 static_cast<size_t>(spatial_rank))
        << "Config has " << config.output_spatial_tiles.size()
        << " output spatial tiles but the convolution has " << spatial_rank
        << " spatial dimensions.";
    TF_RET_CHECK(config.kernel_spatial_tiles.size() ==
                 static_cast<size_t>(spatial_rank))
        << "Config has " << config.kernel_spatial_tiles.size()
        << " kernel spatial tiles but the convolution has " << spatial_rank
        << " spatial dimensions.";

    // Annotate the conv with the contraction tile size.
    xla::xtile::Tile sequential_tile;
    for (int64_t tile : config.kernel_spatial_tiles) {
      sequential_tile.add_sizes(tile);
    }
    sequential_tile.add_sizes(config.input_feature_tile);
    ABSL_RETURN_IF_ERROR(conv->set_backend_config(sequential_tile));

    // Annotate the fusion itself with the block-level parameters.
    const int64_t output_rank = conv->shape().dimensions().size();
    TF_RET_CHECK(output_rank == spatial_rank + 2)
        << "Convolution output rank " << output_rank
        << " does not match spatial rank " << spatial_rank << " plus two.";
    std::vector<int64_t> output_tile(output_rank, 1);
    output_tile[dnums.output_batch_dimension()] = config.batch_tile;
    output_tile[dnums.output_feature_dimension()] = config.output_feature_tile;
    for (int64_t i = 0; i < spatial_rank; ++i) {
      output_tile[dnums.output_spatial_dimensions(i)] =
          config.output_spatial_tiles[i];
    }

    BlockLevelParameters params;
    params.output_tile_sizes = {output_tile};
    params.num_warps = config.num_warps;
    params.num_ctas = config.num_ctas;
    params.num_stages = config.num_stages;
    params.waves_per_eu = config.waves_per_eu;

    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
    FusionBackendConfig& mutable_backend_config =
        *gpu_config.mutable_fusion_backend_config();
    mutable_backend_config.clear_triton_conv_config();
    mutable_backend_config.set_kind(kTritonFusionKind);
    *mutable_backend_config.mutable_block_level_fusion_config() =
        params.ToBlockLevelFusionConfig();
    ABSL_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

    MarkAsChanged();
    return absl::OkStatus();
  }
};

}  // namespace

absl::StatusOr<bool> ConvertTritonConvConfig::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ConvertTritonConvConfigVisitor visitor;
    ABSL_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace xla::gpu
