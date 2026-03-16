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

// Worker binary for Triton config compilation validation.
//
// Each invocation compiles a single (HLO module, TritonGemmConfig) pair and
// exits with a structured exit code that the parent runner interprets:
//
//   Exit 0  -- compilation succeeded (executable produced).
//   Exit 1  -- compilation returned a non-OK absl::Status (unexpected error).
//   Exit 2  -- compilation produced a null executable (config silently
//              rejected, e.g. OOM or register-spill filter).
//   Signal  -- crash (SIGSEGV, SIGABRT, …); detected by runner via waitpid.
//
// Flags:
//   --hlo_file=<path>      HLO text file with a single gemm fusion module.
//   --config=<text_proto>  Serialised AutotuneResult::TritonGemmKey (single-
//                          line text format, matching
//                          --xla_gpu_override_gemm_autotuner).
//
// Compilation is performed in deviceless mode (no GPU interaction) when
// --xla_gpu_target_config_filename is set: the runner serialises the
// GpuTargetConfig once and workers read it, allowing hundreds of workers to
// run in parallel without touching the GPU.

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "tsl/platform/protobuf.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/split_k_gemm_rewriter.h"
#include "xla/service/gpu/transforms/fusion_wrapper.h"
#include "xla/service/gpu/transforms/hoist_fused_bitcasts.h"
#include "xla/service/gpu/transforms/nest_gemm_fusion.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/init_main.h"

namespace xla {
namespace gpu {
namespace {

// Exit codes used by the runner.
constexpr int kExitOk = 0;
constexpr int kExitStatusError = 1;
constexpr int kExitNullExecutable = 2;

absl::StatusOr<std::unique_ptr<HloModule>> LoadHloModule(
    const std::string& path) {
  std::string hlo_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), path, &hlo_text));
  return ParseAndReturnUnverifiedModule(hlo_text);
}

absl::StatusOr<TritonGemmConfig> ParseTritonConfig(
    const std::string& config_text) {
  AutotuneResult::TritonGemmKey key;
  if (!tsl::protobuf::TextFormat::ParseFromString(config_text, &key)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse TritonGemmKey text proto: ", config_text));
  }
  return TritonGemmConfig::FromProto(key);
}

// Replicates TritonGemmAutotuneExtractor: clones the root fusion into a new
// module, stamps the config into the backend config, and applies the same
// transformation passes the production autotuner uses, including the full
// split_k > 1 path (MakeDotSplitKBatch + FloatNormalization + PriorityFusion
// + FusionWrapper).
absl::StatusOr<std::unique_ptr<HloModule>> ExtractTritonModule(
    const TritonGemmConfig& config,
    const se::DeviceDescription& device_desc,
    const HloFusionInstruction* fusion,
    const DebugOptions& debug_opts,
    mlir::MLIRContext* mlir_context) {
  std::unique_ptr<HloModule> new_module =
      ExtractInstructionIntoNewModule(*fusion);
  new_module->mutable_config().set_debug_options(debug_opts);

  HloInstruction* cloned_dot_fusion =
      new_module->entry_computation()->root_instruction();

  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      cloned_dot_fusion->backend_config<GpuBackendConfig>());
  *gpu_config.mutable_fusion_backend_config()
       ->mutable_triton_gemm_config() = config.ToProto();
  TF_RETURN_IF_ERROR(cloned_dot_fusion->set_backend_config(gpu_config));

  if (config.split_k > 1) {
    TF_RETURN_IF_ERROR(MakeDotSplitKBatch(cloned_dot_fusion, config));
    for (PrimitiveType type :
         {BF16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ}) {
      GpuFloatSupport float_support(device_desc.gpu_compute_capability(),
                                    type);
      FloatNormalization float_normalization(&float_support);
      TF_RETURN_IF_ERROR(float_normalization.Run(new_module.get()).status());
    }
    PriorityFusion priority_fusion(
        /*thread_pool=*/nullptr, device_desc,
        HloCostAnalysis::Options{.count_multiple_input_accesses = true},
        mlir_context);
    TF_RETURN_IF_ERROR(priority_fusion.Run(new_module.get()).status());
    FusionWrapper fusion_wrapper(device_desc);
    TF_RETURN_IF_ERROR(fusion_wrapper.Run(new_module.get()).status());
  }

  HoistFusedBitcasts hoist_fused_bitcasts;
  TF_RETURN_IF_ERROR(hoist_fused_bitcasts.Run(new_module.get()).status());

  NestGemmFusion nest_gemm_fusion(device_desc, mlir_context);
  TF_RETURN_IF_ERROR(nest_gemm_fusion.Run(new_module.get()).status());

  return new_module;
}

absl::Status CompileOneConfig(const std::string& hlo_file,
                              const std::string& config_text) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      LoadHloModule(hlo_file));
  TF_ASSIGN_OR_RETURN(TritonGemmConfig config, ParseTritonConfig(config_text));

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_autotune_level(0);
  debug_options.set_xla_gpu_exhaustive_tiling_search(false);
  debug_options.clear_xla_dump_to();

  // Load the GPU target config from the file written by the runner.
  // This allows fully deviceless compilation — no GPU interaction at all.
  const std::string& target_config_file =
      debug_options.xla_gpu_target_config_filename();
  TF_RET_CHECK(!target_config_file.empty())
      << "Worker requires --xla_gpu_target_config_filename for deviceless "
         "compilation.";

  std::string config_string;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                           target_config_file,
                                           &config_string));
  se::GpuTargetConfigProto target_config_proto;
  if (!tsl::protobuf::TextFormat::ParseFromString(config_string,
                                                  &target_config_proto)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse GpuTargetConfigProto from ",
                     target_config_file));
  }
  TF_ASSIGN_OR_RETURN(GpuTargetConfig gpu_target_config,
                      GpuTargetConfig::FromProto(target_config_proto));
  const se::DeviceDescription& device_desc =
      gpu_target_config.device_description;

  // Find the root gemm fusion in the entry computation.
  HloInstruction* root = module->entry_computation()->root_instruction();
  if (root->opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Root of entry computation is not a fusion: ", root->name()));
  }
  const auto* fusion = Cast<HloFusionInstruction>(root);

  mlir::MLIRContext mlir_context;
  absl::StatusOr<std::unique_ptr<HloModule>> triton_module_or =
      ExtractTritonModule(config, device_desc, fusion, debug_options,
                          &mlir_context);
  if (triton_module_or.status().GetPayload(kUncompilableFusion).has_value()) {
    std::cerr << "SKIP: uncompilable fusion (e.g. incompatible split-k).\n";
    std::exit(kExitNullExecutable);
  }
  TF_RETURN_IF_ERROR(triton_module_or.status());

  // Get compiler for the GPU platform (CPU-only, no device init).
  TF_ASSIGN_OR_RETURN(std::string platform_name,
                      PlatformUtil::CanonicalPlatformName("gpu"));
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithName(
          absl::AsciiStrToUpper(platform_name)));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler,
                      Compiler::GetForPlatform(platform->id()));

  // Compile with stream_exec=nullptr (deviceless).
  Compiler::CompileOptions compile_options;
  compile_options.device_allocator = nullptr;
  compile_options.embed_hlo_module = false;
  absl::StatusOr<std::unique_ptr<Executable>> exec_or =
      compiler->RunBackend(std::move(*triton_module_or),
                           /*stream_exec=*/nullptr, compile_options);

  if (exec_or.status().code() == absl::StatusCode::kResourceExhausted ||
      exec_or.status().code() == absl::StatusCode::kCancelled) {
    std::cerr << "SKIP: " << exec_or.status() << "\n";
    std::exit(kExitNullExecutable);
  }

  if (!exec_or.ok()) {
    std::cerr << "COMPILE_ERROR: " << exec_or.status() << "\n";
    return exec_or.status();
  }

  if (*exec_or == nullptr) {
    std::cerr << "SKIP: config produced null executable.\n";
    std::exit(kExitNullExecutable);
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace gpu
}  // namespace xla

int main(int argc, char** argv) {
  std::string hlo_file;
  std::string config_text;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo_file", &hlo_file,
                "Path to the HLO text file containing a single gemm fusion."),
      tsl::Flag("config", &config_text,
                "Serialised TritonGemmKey proto (text format) for the Triton "
                "config to compile."),
  };

  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  if (!parse_ok || hlo_file.empty() || config_text.empty()) {
    std::cerr << "Required flags: --hlo_file and --config\n" << usage;
    return 1;
  }

  absl::Status status = xla::gpu::CompileOneConfig(hlo_file, config_text);
  if (!status.ok()) {
    std::cerr << "COMPILE_ERROR: " << status << "\n";
    return xla::gpu::kExitStatusError;
  }
  return xla::gpu::kExitOk;
}
