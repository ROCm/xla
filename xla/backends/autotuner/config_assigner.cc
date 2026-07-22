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

#include "xla/backends/autotuner/config_assigner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/config_runner.h"
#include "xla/backends/autotuner/config_selector.h"
#include "xla/backends/autotuner/dichotomic_search.h"
#include "xla/backends/autotuner/hlo_extractor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/autotuning/autotuner_status_key.h"
#include "xla/status_macros.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace {

// Returns true if every config was produced by the Triton backend (the
// `triton` TritonGemmKey or `block_level` BlockLevelFusionConfig oneof case).
bool AllConfigsAreTriton(
    absl::Span<const CodegenOrchestrator::Config> configs) {
  if (configs.empty()) {
    return false;
  }
  for (const CodegenOrchestrator::Config& c : configs) {
    if (c.backend_config == nullptr) {
      return false;
    }
    const BackendConfig::ConfigCase config_case =
        c.backend_config->config_case();
    if (config_case != BackendConfig::kTriton &&
        config_case != BackendConfig::kBlockLevel) {
      return false;
    }
  }
  return true;
}

// It is important to fingerprint the entire module not just the autotuning
// candidates, to avoid collisions in the key-value store when several
// distinct modules have the same fusions, and are compiled at different
// times by the same PjRt client.
//
// TODO(b/394763704): Eliminate the sharding feature when we have offline
// autotuning. See below for an explanation of some issues.
//
// Theoretically, we also want to include the hash of the module config
// to ensure that a module compiled twice with different configs is
// autotuned twice.
//
// This is important since the config could e.g. affect codegen, or the
// space of possible parameters for autotuning. As a result, the autotuning
// results could look very different for the same module.
//
// Why is it not done here? Well, proto serialization is non-deterministic
// and may change across different builds. Which means that users who run
// on several hosts with different CPUs may end up generating different
// fingerprints for the same module config. They would then fail to
// exchange results through the key value store, which would lead to
// deadlocks. Therefore, we don't hash the module config here.
//
// The flip side is this: if we compile the same module twice in the same
// client, but with a different module config each time, we may hit the
// cache the second time and recover potentially inferior, or incomplete
// autotuning results.
std::string GetKvStoreKey(
    const HloModule* module, int shard_index,
    const std::vector<std::unique_ptr<CodegenBackend>>& codegen_backends) {
  std::vector<std::string> names;
  names.reserve(codegen_backends.size());
  for (const auto& backend : codegen_backends) {
    names.push_back(std::string(backend->name()));
  }
  absl::c_sort(names);
  std::string backend_names = absl::StrJoin(names, ",");
  uint32_t backend_fingerprint = tsl::Fingerprint32(backend_names);
  return absl::StrCat("autotune_results_", module->GetFingerprint128(), "_",
                      backend_fingerprint, "_", shard_index);
}

}  // namespace

absl::StatusOr<std::unique_ptr<ConfigAssigner>> ConfigAssigner::Create(
    Options options,
    std::unique_ptr<AutotunerCacheInterface> absl_nonnull cache,
    std::unique_ptr<CodegenOrchestrator> absl_nonnull orchestrator,
    std::unique_ptr<Profiler> absl_nullable profiler) {
  std::unique_ptr<ConfigRunner> config_runner = nullptr;
  if (profiler != nullptr) {
    ConfigRunner::CorrectnessCheckOptions correctness_check_options;
    correctness_check_options.enable_correctness_check = options.check_buffers;
    correctness_check_options.relative_tolerance = options.relative_tolerance;
    correctness_check_options.crash_on_failure = options.crash_on_check_failure;
    ASSIGN_OR_RETURN(
        config_runner,
        ConfigRunner::Create(std::move(profiler), correctness_check_options));
  }
  return absl::WrapUnique(
      new ConfigAssigner(std::move(options), std::move(cache),
                         std::move(orchestrator), std::move(config_runner)));
}

absl::Status ConfigAssigner::AssignConfigs(
    HloModule* module, const InstructionFilterFn& should_assign_config) {
  std::vector<EquivalentInstructions> instruction_groups =
      ExtractEquivalentInstructions(*module, should_assign_config);
  if (instruction_groups.empty()) {
    VLOG(1) << "No instructions to autotune.";
    return absl::OkStatus();
  }
  VLOG(1) << "Finding configs for " << instruction_groups.size()
          << " unique instructions.";

  ASSIGN_OR_RETURN(std::vector<Config> configs,
                   GetConfigsForAll(instruction_groups));

  for (int i = 0; i < instruction_groups.size(); i++) {
    auto& instructions = instruction_groups[i];
    Config config = std::move(configs[i]);
    if (options_.dump_hlos) {
      RETURN_IF_ERROR(DumpHlo(*instructions[0], config));
    }
    for (auto* instr : instructions) {
      RETURN_IF_ERROR(orchestrator_->ApplyConfig(*instr, config));
    }
  }
  if (config_runner_ != nullptr) {
    RETURN_IF_ERROR(DumpTuningLogs());
  }
  return absl::OkStatus();
}

absl::Status ConfigAssigner::AssignConfigs(
    HloModule* module, const InstructionFilterFn& should_assign_config,
    MultiProcessKeyValueStore& sharding_kv_store) {
  // Sharding the instructions only makes sense if we can have different
  // configs for different shards, which only happens due to online tuning.
  if (options_.select_first_config || options_.use_default_config) {
    VLOG(1) << "Falling back to non-sharded config assignment as online "
               "tuning is disabled.";
    return AssignConfigs(module, should_assign_config);
  }

  int total_shards = sharding_kv_store.process_count;
  int my_shard_index = sharding_kv_store.process_index;

  // 1. Get all the instructions that needs a config assignment.
  std::vector<EquivalentInstructions> all_instruction_groups =
      ExtractEquivalentInstructions(*module, should_assign_config);
  if (all_instruction_groups.empty()) {
    VLOG(1) << "No instructions to autotune.";
    return absl::OkStatus();
  }

  // 2. Shard and get configs for instructions in the current shard.
  const size_t bucket_size = tsl::MathUtil::CeilOfRatio<size_t>(
      all_instruction_groups.size(), static_cast<size_t>(total_shards));
  const size_t start =
      std::min(bucket_size * my_shard_index, all_instruction_groups.size());
  const size_t end =
      std::min(start + bucket_size, all_instruction_groups.size());
  std::vector<EquivalentInstructions> instruction_groups;
  instruction_groups.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    instruction_groups.push_back(all_instruction_groups[i]);
  }

  // 3. Autotune instructions for this shard. Use cached configs if available,
  // otherwise get and cache the best config.
  VLOG(1) << "Shard " << my_shard_index << "/" << total_shards
          << ": finding configs for " << instruction_groups.size() << "/"
          << all_instruction_groups.size() << " unique instructions ";

  ASSIGN_OR_RETURN(std::vector<Config> configs,
                   GetConfigsForAll(instruction_groups));

  std::vector<const HloInstruction*> autotuned_instructions;
  autotuned_instructions.reserve(instruction_groups.size());
  for (int i = 0; i < instruction_groups.size(); ++i) {
    autotuned_instructions.push_back(instruction_groups[i][0]);
  }
  if (config_runner_ != nullptr) {
    RETURN_IF_ERROR(DumpTuningLogs());
  }

  // 4. Store the results for this shard as a serialized string to the KV store.
  KeyValueStoreInterface& kv_store = *sharding_kv_store.key_value_store;
  const std::string local_key =
      GetKvStoreKey(module, my_shard_index, orchestrator_->codegen_backends());
  std::string local_results;
  if (!autotuned_instructions.empty()) {
    ASSIGN_OR_RETURN(local_results,
                     optimal_config_cache_->Serialize(autotuned_instructions));
  }
  absl::StatusOr<std::string> stored_result = kv_store.TryGet(local_key);
  if (stored_result.status().code() == absl::StatusCode::kNotFound) {
    VLOG(2) << "Storing results for " << local_key;
    absl::Status set_result = kv_store.Set(local_key, local_results);
    if (absl::IsAlreadyExists(set_result)) {
      VLOG(2) << "Shard " << my_shard_index << " tried to store results at "
              << local_key << " but lost a race to do so";
    } else if (!set_result.ok()) {
      return set_result;
    } else {
      VLOG(2) << "Shard " << my_shard_index << " stored results at "
              << local_key;
    }
  } else if (!stored_result.ok()) {
    return stored_result.status();
  } else {
    VLOG(2) << "Results already exist for " << local_key << ", skipping store.";
  }

  // 5. Load the autotune results of other shards from the KV store and update
  // the current shard's cache by deserializing the results.
  for (int i = 0; i < total_shards; ++i) {
    if (i == my_shard_index) {
      continue;
    }
    const std::string remote_key =
        GetKvStoreKey(module, i, orchestrator_->codegen_backends());
    VLOG(2) << "Shard " << my_shard_index << ": waiting for results from shard "
            << i << " / " << total_shards << " at " << remote_key;
    // TODO(b/361009609): reset to infinite duration once issue with MPI is
    // fixed. https://github.com/google/jax/issues/22995.
    ASSIGN_OR_RETURN(std::string remote_results,
                     kv_store.Get(remote_key, absl::Hours(24)));
    if (!remote_results.empty()) {
      RETURN_IF_ERROR(optimal_config_cache_->Deserialize(remote_results));
    }
  }

  // 6. Apply the results to all candidate instructions, must be already in
  // cache_ due to step 3 and 5 above.
  for (auto& instruction_group : all_instruction_groups) {
    CHECK(!instruction_group.empty());
    std::optional<Config> cached_config = LookUp(instruction_group[0]);
    if (!cached_config.has_value()) {
      return absl::InternalError(absl::StrCat(
          "Autotuning failed for HLO: ", instruction_group[0]->ToString(),
          ". No configuration found in cache after synchronizing results "
          "across all shards."));
    }
    if (options_.dump_hlos) {
      RETURN_IF_ERROR(DumpHlo(*instruction_group[0], *cached_config));
    }
    for (auto* instr : instruction_group) {
      RETURN_IF_ERROR(orchestrator_->ApplyConfig(*instr, *cached_config));
    }
  }

  return absl::OkStatus();
}

absl::Status ConfigAssigner::AssignConfig(HloInstruction* instr) {
  ASSIGN_OR_RETURN(Config config, GetConfig(instr).Await());
  if (options_.dump_hlos) {
    RETURN_IF_ERROR(DumpHlo(*instr, config));
  }
  RETURN_IF_ERROR(orchestrator_->ApplyConfig(*instr, config));
  if (config_runner_ != nullptr) {
    RETURN_IF_ERROR(DumpTuningLogs());
  }
  return absl::OkStatus();
}

tsl::Future<ConfigAssigner::Config> ConfigAssigner::GetConfig(
    const HloInstruction* instr) {
  if (VLOG_IS_ON(1)) {
    HloPrintOptions print_options;
    if (VLOG_IS_ON(4)) {
      print_options.set_print_subcomputation_mode(
          HloPrintOptions::PrintSubcomputationMode::kFullBodies);
    }
    VLOG(1) << "Getting config for HLO: " << instr->ToString(print_options);
  }
  std::optional<Config> cached_config = LookUp(instr);
  if (cached_config.has_value()) {
    VLOG(1) << "Using cached config: " << cached_config->ToString();
    return std::move(cached_config.value());
  }

  if (options_.expect_all_instructions_in_cache) {
    absl::Status s = absl::NotFoundError(absl::StrCat(
        "No cached config found for HLO instr: ", instr->ToString()));
    tsl::errors::InsertPayloads(
        s, {{std::string(gpu::kAutotuneCacheRequiredErrorPayloadKey), ""}});
    return s;
  }

  // TODO (b/446870267): Improve the cache fallback logic as we move to offline
  // autotuning.
  if (options_.use_default_config) {
    ASSIGN_OR_RETURN(Config default_config,
                     orchestrator_->GetDefaultConfig(*instr));
    VLOG(1) << "Using default config: " << default_config.ToString();
    return default_config;
  }

  if (options_.select_first_config) {
    ASSIGN_OR_RETURN(std::vector<Config> supported_configs,
                     orchestrator_->GetSupportedConfigs(*instr));
    for (Config& config : supported_configs) {
      auto executable = orchestrator_->Compile(*instr, config);
      if (executable.ok()) {
        VLOG(1) << "Using first compilable config: " << config.ToString();
        return std::move(config);
      }
    }
    return absl::InternalError(
        absl::StrCat("No supported config found for HLO: ", instr->ToString()));
  }

  TF_RET_CHECK(config_runner_ != nullptr)
      << "Cannot autotune HLO: " << instr->ToString()
      << ". ConfigRunner is not initialized.";
  VLOG(1) << "Getting tuned config for HLO: " << instr->ToString();
  return GetTunedConfig(instr).Map(
      [this, instr](Config config) -> absl::StatusOr<Config> {
        RETURN_IF_ERROR(Insert(instr, config));
        return std::move(config);
      });
}

// TODO(b/444398084): Use Autouner::GetTunedConfig when the cache is migrated
// and we don't need backward compatibility.
tsl::Future<ConfigAssigner::Config> ConfigAssigner::GetTunedConfig(
    const HloInstruction* instr) {
  CHECK(config_runner_ != nullptr);
  ASSIGN_OR_RETURN(std::vector<CodegenOrchestrator::Config> supported_configs,
                   orchestrator_->GetSupportedConfigs(*instr));
  TF_RET_CHECK(!supported_configs.empty())
      << "Autotuning failed for HLO: " << instr->ToString()
      << ". No supported configs found for this instruction.";

  if (supported_configs.size() == 1) {
    VLOG(1) << "Found only one supported config: "
            << supported_configs[0].ToString();
    return std::move(supported_configs[0]);
  }

  VLOG(1) << "Found total of " << supported_configs.size()
          << " supported configs.";

  // Experimental Triton-only dichotomic search. Exhaustive search takes
  // precedence when both flags are set.
  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();
  VLOG(3) << "Dichotomic dispatch check for " << instr->name()
          << ": dichotomic_flag="
          << debug_options.xla_gpu_dichotomic_tiling_search()
          << " exhaustive_flag="
          << debug_options.xla_gpu_exhaustive_tiling_search()
          << " all_triton=" << AllConfigsAreTriton(supported_configs)
          << " num_configs=" << supported_configs.size();
  if (debug_options.xla_gpu_dichotomic_tiling_search() &&
      !debug_options.xla_gpu_exhaustive_tiling_search() &&
      AllConfigsAreTriton(supported_configs)) {
    return GetTunedConfigDichotomic(instr, std::move(supported_configs));
  }

  tsl::Future<std::vector<CodegenOrchestrator::MaybeExecutableCandidate>>
      maybe_candidates =
          orchestrator_->CompileAll(*instr, std::move(supported_configs));
  return std::move(maybe_candidates)
      .Map([instr,
            this](std::vector<CodegenOrchestrator::MaybeExecutableCandidate>
                      maybe_candidates) mutable -> absl::StatusOr<Config> {
        CHECK(config_runner_ != nullptr);  // To make clang-tidy happy.
        std::vector<ConfigRunner::ExecutableCandidate> candidates;
        std::vector<ConfigRunner::ConfigProfile> compilation_failures;
        for (auto& maybe_candidate : maybe_candidates) {
          if (maybe_candidate.executable.ok()) {
            candidates.push_back(
                {std::move(maybe_candidate.config),
                 std::move(maybe_candidate.executable.value())});
          } else {
            VLOG(3) << "Failed to compile config: "
                    << maybe_candidate.config.ToString()
                    << " with status: " << maybe_candidate.executable.status();
            compilation_failures.push_back(
                {std::move(maybe_candidate.config),
                 ConfigRunner::Failure{
                     ConfigRunner::FailureKind::kCompilationFailed,
                     maybe_candidate.executable.status().ToString()}});
          }
        }

        TF_RET_CHECK(!candidates.empty())
            << "Autotuning failed for HLO: " << instr->ToString()
            << ". No configs could be compiled.";

        VLOG(1) << "Successfully compiled " << candidates.size() << " configs.";

        if (candidates.size() == 1) {
          VLOG(1) << "Using the only compilable config: "
                  << candidates[0].config.ToString();
          return std::move(candidates[0].config);
        }

        ASSIGN_OR_RETURN(
            std::vector<ConfigRunner::ConfigProfile> profiles,
            config_runner_->ProfileAll(std::move(candidates), instr));

        TF_RET_CHECK(!profiles.empty())
            << "Autotuning failed for HLO: " << instr->ToString()
            << ". No configs could be profiled.";

        LogConfigProfiles(*instr, profiles, compilation_failures);
        ASSIGN_OR_RETURN(
            ConfigRunner::ConfigProfile best_profile,
            PickBestConfig(profiles, options_.scratch_bytes_window_size_us));
        return std::move(best_profile.config);
      });
}

tsl::Future<ConfigAssigner::Config> ConfigAssigner::GetTunedConfigDichotomic(
    const HloInstruction* instr, std::vector<Config> supported_configs) {
  CHECK(config_runner_ != nullptr);

  // Build the discretized search space from the exhaustive Triton config set.
  std::vector<const BackendConfig*> backend_configs;
  backend_configs.reserve(supported_configs.size());
  for (const Config& c : supported_configs) {
    backend_configs.push_back(c.backend_config.get());
  }
  absl::StatusOr<DichotomicSearchSpace> space_or =
      DichotomicSearchSpace::Build(backend_configs);
  if (!space_or.ok()) {
    return space_or.status();
  }
  DichotomicSearchSpace space = std::move(space_or).value();
  SearchProfile profile = MakeProfile(space, instr->opcode());

  // Renders a config's axis values as "name=value" pairs for logging.
  auto config_to_string = [&](const Coord& coord) -> std::string {
    std::string s;
    for (int a = 0; a < space.axes().size() && a < coord.size(); ++a) {
      if (!s.empty()) s += ", ";
      const ParameterAxis& axis = space.axes()[a];
      int idx = coord[a];
      int64_t value =
          (idx >= 0 && idx < axis.values.size()) ? axis.values[idx] : -1;
      absl::StrAppend(&s, axis.name, "=", value);
    }
    return s;
  };

  // Compiles + profiles the given subset (referenced by index), reusing the
  // existing CompileAll + ProfileAll pipeline. Sequential per-phase; blocking
  // is on GPU work, not CPU.
  auto eval_batch = [&](absl::Span<const int> indices)
      -> absl::StatusOr<std::vector<ConfigRunner::ConfigProfile>> {
    if (indices.empty()) {
      return std::vector<ConfigRunner::ConfigProfile>{};
    }
    std::vector<Config> batch;
    batch.reserve(indices.size());
    for (int idx : indices) {
      Config copy;
      copy.codegen_backend = supported_configs[idx].codegen_backend;
      copy.backend_config = std::make_unique<BackendConfig>(
          *supported_configs[idx].backend_config);
      batch.push_back(std::move(copy));
    }
    ASSIGN_OR_RETURN(
        std::vector<CodegenOrchestrator::MaybeExecutableCandidate>
            maybe_candidates,
        orchestrator_->CompileAll(*instr, std::move(batch)).Await());
    std::vector<ConfigRunner::ExecutableCandidate> candidates;
    for (auto& mc : maybe_candidates) {
      if (mc.executable.ok()) {
        candidates.push_back(
            {std::move(mc.config), std::move(mc.executable.value())});
      }
    }
    if (candidates.empty()) {
      return std::vector<ConfigRunner::ConfigProfile>{};
    }
    return config_runner_->ProfileAll(std::move(candidates), instr);
  };

  // Maps a profiled config back to its coordinate in `space`, logs it, and
  // records the measured time. Failed profiles are skipped.
  auto collect_samples =
      [&](absl::Span<const ConfigRunner::ConfigProfile> profiles,
          std::vector<Sample>* out) {
        std::vector<const BackendConfig*> singleton(1);
        for (const ConfigRunner::ConfigProfile& p : profiles) {
          if (p.failure.has_value() || p.config.backend_config == nullptr) {
            continue;
          }
          singleton[0] = p.config.backend_config.get();
          absl::StatusOr<DichotomicSearchSpace> single =
              DichotomicSearchSpace::Build(singleton);
          if (!single.ok()) {
            continue;
          }
          Coord coord(space.axes().size(), 0);
          const std::vector<ParameterAxis>& single_axes = single->axes();
          bool ok = true;
          for (int a = 0; a < space.axes().size(); ++a) {
            int64_t value = -1;
            for (const ParameterAxis& sa : single_axes) {
              if (sa.name == space.axes()[a].name && !sa.values.empty()) {
                value = sa.values.front();
                break;
              }
            }
            const std::vector<int64_t>& vals = space.axes()[a].values;
            int found = -1;
            for (int i = 0; i < vals.size(); ++i) {
              if (vals[i] == value) {
                found = i;
                break;
              }
            }
            if (found < 0) {
              ok = false;
              break;
            }
            coord[a] = found;
          }
          if (ok) {
            const double time_ms = absl::ToDoubleMilliseconds(p.duration);
            VLOG(2) << "Dichotomic search: tested config {"
                    << config_to_string(coord) << "} -> " << time_ms << " ms";
            out->push_back(
                Sample{std::move(coord), absl::ToDoubleSeconds(p.duration)});
          }
        }
      };

  std::vector<ConfigRunner::ConfigProfile> all_profiles;
  std::vector<Sample> all_samples;
  std::vector<int> evaluated;

  auto run_phase = [&](SearchPhase phase) -> absl::Status {
    std::vector<int> indices =
        SelectConfigs(space, profile, phase, all_samples, evaluated);
    if (indices.empty()) {
      return absl::OkStatus();
    }
    ASSIGN_OR_RETURN(std::vector<ConfigRunner::ConfigProfile> profiles,
                     eval_batch(indices));
    collect_samples(profiles, &all_samples);
    for (int idx : indices) {
      evaluated.push_back(idx);
    }
    for (auto& p : profiles) {
      all_profiles.push_back(std::move(p));
    }
    return absl::OkStatus();
  };

  // Phase 1: coarse grid + role verification.
  RETURN_IF_ERROR(run_phase(SearchPhase::kCoarseGrid));
  if (all_profiles.empty()) {
    return absl::InternalError(absl::StrCat(
        "Dichotomic search Phase 1: no configs compiled/profiled for HLO: ",
        instr->ToString()));
  }
  profile = RefineRoles(profile, space, all_samples);

  // Phase 2: coordinate-wise ternary refinement.
  RETURN_IF_ERROR(run_phase(SearchPhase::kTernaryRefine));

  // Phase 3: neighborhood + small-axis sweep.
  RETURN_IF_ERROR(run_phase(SearchPhase::kNeighborhoodSweep));

  VLOG(1) << "Dichotomic search evaluated " << evaluated.size() << " / "
          << space.num_configs() << " configs for: " << instr->ToString();

  // Log the finally selected config and its measured runtime BEFORE
  // PickBestConfig so the line is always emitted when the dichotomic search
  // completes its phases.
  {
    const int best_idx = BestSampleIndex(space, all_samples);
    std::string best_desc;
    double best_ms = -1.0;
    for (const Sample& s : all_samples) {
      if (space.LookupIndex(s.coord) == best_idx) {
        best_desc = config_to_string(s.coord);
        best_ms = s.time_seconds * 1e3;
        break;
      }
    }
    VLOG(1) << "Dichotomic search selected config {" << best_desc << "} -> "
            << best_ms << " ms for: " << instr->ToString();
  }

  ASSIGN_OR_RETURN(
      ConfigRunner::ConfigProfile best_profile,
      PickBestConfig(all_profiles, options_.scratch_bytes_window_size_us));

  return tsl::Future<Config>(std::move(best_profile.config));
}

std::optional<ConfigAssigner::Config> ConfigAssigner::LookUp(
    const HloInstruction* instr) const {
  auto cached_config = optimal_config_cache_->Lookup(instr);
  if (!cached_config.has_value()) {
    return std::nullopt;
  }
  VLOG(1) << "Found cached config for HLO: " << instr->ToString();
  for (const auto& codegen_backend : orchestrator_->codegen_backends()) {
    if (codegen_backend->backend() == cached_config->codegen_backend) {
      auto backend_config =
          std::make_unique<BackendConfig>(cached_config->backend_config);
      return Config{codegen_backend.get(), std::move(backend_config)};
    }
  }
  LOG(WARNING) << "Ignoring cached config from backend "
               << Backend_Name(cached_config->codegen_backend) << " for HLO '"
               << instr->ToString() << "'"
               << ", because this backend is not registered with the "
                  "autotuner.";
  return std::nullopt;
}

absl::Status ConfigAssigner::Insert(const HloInstruction* instr,
                                    const ConfigAssigner::Config& config) {
  AutotunerCacheInterface::Config cached_config;
  cached_config.codegen_backend = config.codegen_backend->backend();
  cached_config.backend_config = *config.backend_config;
  return optimal_config_cache_->Insert(instr, cached_config);
}

absl::StatusOr<std::vector<ConfigAssigner::Config>>
ConfigAssigner::GetConfigsForAll(
    const std::vector<InstructionGroup>& instruction_groups) {
  std::vector<tsl::Future<Config>> future_configs;
  future_configs.reserve(instruction_groups.size());
  for (int i = 0; i < instruction_groups.size(); i++) {
    future_configs.push_back(GetConfig(instruction_groups[i][0]));
  }

  std::vector<absl::StatusOr<Config>> status_or_configs;
  status_or_configs.reserve(instruction_groups.size());
  absl::Status combined_status = absl::OkStatus();
  int num_failures = 0;
  for (int i = 0; i < instruction_groups.size(); i++) {
    absl::StatusOr<Config> config_or = std::move(future_configs[i]).Await();
    combined_status.Update(config_or.status());
    if (!config_or.ok()) {
      LOG(ERROR)
          << "Could not get config for HLO: "
          << instruction_groups[i][0]->ToString(
                 HloPrintOptions().set_print_subcomputation_mode(
                     HloPrintOptions::PrintSubcomputationMode::kFullBodies))
          << ". Status: " << config_or.status();
      num_failures++;
    }
    status_or_configs.push_back(std::move(config_or));
  }

  if (!combined_status.ok() && num_failures > 1) {
    return tsl::errors::CreateWithUpdatedMessage(
        combined_status,
        absl::StrCat(
            "Failed to get configs for: ", num_failures, " out of ",
            instruction_groups.size(),
            " instructions. See logs for all failures. Example failure: \n",
            combined_status.message()));
  }
  RETURN_IF_ERROR(combined_status);

  std::vector<Config> configs;
  for (auto& config_or : status_or_configs) {
    if (config_or.ok()) {
      configs.push_back(std::move(*config_or));
    }
  }
  return configs;
}

absl::Status ConfigAssigner::DumpHlo(const HloInstruction& instr,
                                     const Config& config) {
  const HloModule* parent_module = instr.GetModule();
  std::unique_ptr<HloModule> module = ExtractInstructionIntoNewModule(instr);
  module->set_name(std::string(instr.name()));
  std::string id =
      absl::StrCat("autotuner_", dump_counter_++, ".", instr.name());
  DumpToFileInDirOrStdout(*parent_module, "", absl::StrCat(id, ".before.txt"),
                          module->ToString());
  HloInstruction* root = module->entry_computation()->root_instruction();
  RETURN_IF_ERROR(orchestrator_->ApplyConfig(*root, config));
  DumpToFileInDirOrStdout(*parent_module, "", absl::StrCat(id, ".after.txt"),
                          module->ToString());
  return absl::OkStatus();
}

void ConfigAssigner::LogConfigProfiles(
    const HloInstruction& instr,
    absl::Span<const ConfigRunner::ConfigProfile> profiles,
    absl::Span<const ConfigRunner::ConfigProfile> failed_configs) {
  for (const ConfigRunner::ConfigProfile& profile : profiles) {
    VLOG(2) << profile.ToString(/*verbose=*/VLOG_IS_ON(3));
  }
  for (const ConfigRunner::ConfigProfile& result : failed_configs) {
    VLOG(2) << result.ToString(/*verbose=*/VLOG_IS_ON(3));
  }
  if (options_.dump_logs_to.empty()) {
    return;
  }
  AutotuningLog log;
  log.mutable_instr()->PackFrom(instr.ToProto());
  for (const auto& profile : profiles) {
    *log.add_results() = profile.ToProto();
  }
  for (const auto& failed_config : failed_configs) {
    *log.add_results() = failed_config.ToProto();
  }
  *logs_.add_logs() = std::move(log);
}

absl::Status ConfigAssigner::DumpTuningLogs() {
  if (options_.dump_logs_to.empty()) {
    return absl::OkStatus();
  }

  std::string textproto;
  tsl::protobuf::TextFormat::PrintToString(logs_, &textproto);

  RETURN_IF_ERROR(tsl::AppendStringToFile(tsl::Env::Default(),
                                          options_.dump_logs_to, textproto));
  VLOG(1) << "Autotune logs appended to file: " << options_.dump_logs_to;
  logs_.Clear();
  return absl::OkStatus();
}

std::string ConfigAssigner::Options::ToString() const {
  return absl::StrFormat(
      R"json({
  "check_buffers": %v,
  "relative_tolerance": %g,
  "crash_on_check_failure": %v,
  "scratch_bytes_window_size_us": %d,
  "expect_all_instructions_in_cache": %v,
  "dump_logs_to": "%s",
  "select_first_config": %v,
  "use_default_config": %v,
  "dump_hlos": %v
})json",
      check_buffers, relative_tolerance, crash_on_check_failure,
      scratch_bytes_window_size_us, expect_all_instructions_in_cache,
      absl::CEscape(dump_logs_to), select_first_config, use_default_config,
      dump_hlos);
}

AutotunerCacheInterface::CacheStats ConfigAssigner::GetCacheStats() const {
  return optimal_config_cache_->GetCacheStats();
}

}  // namespace xla
