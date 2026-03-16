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

// Orchestrator for exhaustive Triton config compilation validation on ROCm.
//
// For each (dot shape, TritonGemmConfig) pair in the full search space it
// spawns a fresh worker process (triton_config_validator_worker) that compiles
// exactly that config, then collects the outcome.  Running compilations in
// isolated subprocesses is essential because the ROCm Triton compiler can
// SIGSEGV on certain configs; a signal in a child does not kill this process.
//
// Output: a TSV report written to --output_file (or stdout if not set).
// Each line has the fields:
//
//   shape_label  config_text  status  detail
//
// where status is one of: OK  SKIP  COMPILE_ERROR  CRASH
//
// Usage:
//   bazel run //xla/tools:triton_config_validator_runner -- \
//     --worker=<path/to/triton_config_validator_worker> \
//     --hlo_dir=<dir_with_hlo_files>   [one file per shape to test]
//     --output_file=results.tsv        [default: stdout]
//     --jobs=8                          [parallel subprocess count, default: 4]
//     --xla_gpu_target_config_filename=<device_proto>  [for deviceless shapes]

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "tsl/platform/protobuf.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/autotuning/dot_search_space.h"
#include "xla/service/platform_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/init_main.h"

namespace xla {
namespace gpu {
namespace {

// --------------------------------------------------------------------------
// Worker exit codes (mirrored from triton_config_validator_worker.cc).
// --------------------------------------------------------------------------
constexpr int kExitOk = 0;
constexpr int kExitStatusError = 1;
constexpr int kExitNullExecutable = 2;

// --------------------------------------------------------------------------
// Result classification.
// --------------------------------------------------------------------------
enum class Outcome { kOk, kSkip, kCompileError, kCrash };

absl::string_view OutcomeName(Outcome o) {
  switch (o) {
    case Outcome::kOk:          return "OK";
    case Outcome::kSkip:        return "SKIP";
    case Outcome::kCompileError: return "COMPILE_ERROR";
    case Outcome::kCrash:       return "CRASH";
  }
  return "UNKNOWN";
}

struct ConfigResult {
  std::string shape_label;
  std::string config_text;
  Outcome outcome;
  std::string detail;  // signal name or status text
};

// --------------------------------------------------------------------------
// Helpers.
// --------------------------------------------------------------------------

// Serialises a TritonGemmConfig into single-line text proto (same format that
// --xla_gpu_override_gemm_autotuner uses and that the worker --config flag
// accepts).
std::string SerialiseConfig(const TritonGemmConfig& cfg) {
  tsl::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(true);
  std::string text;
  printer.PrintToString(cfg.ToProto(), &text);
  // Remove trailing space that single-line mode sometimes adds.
  while (!text.empty() && text.back() == ' ') text.pop_back();
  return text;
}

// Returns a short human-readable label for the signal that terminated a child,
// or empty string if it exited normally.
std::string SignalName(int wstatus) {
  if (!WIFSIGNALED(wstatus)) return "";
  int sig = WTERMSIG(wstatus);
  const char* name = strsignal(sig);
  return absl::StrCat("SIG", name ? name : std::to_string(sig));
}

// Spawns the worker as a subprocess and waits for it to finish.
// Returns the raw waitpid status (use WIFEXITED / WIFSIGNALED to interpret).
int RunWorker(const std::string& worker_path,
              const std::string& hlo_file,
              const std::string& config_text,
              const std::string& target_config_path) {
  pid_t pid = fork();
  QCHECK_GE(pid, 0) << "fork() failed: " << strerror(errno);

  if (pid == 0) {
    // Suppress all worker output — the runner owns progress reporting.
    (void)freopen("/dev/null", "w", stdout);
    (void)freopen("/dev/null", "w", stderr);

    const std::string hlo_flag = absl::StrCat("--hlo_file=", hlo_file);
    const std::string cfg_flag = absl::StrCat("--config=", config_text);
    const std::string tc_flag = absl::StrCat(
        "--xla_gpu_target_config_filename=", target_config_path);
    char* const args[] = {
        const_cast<char*>(worker_path.c_str()),
        const_cast<char*>(hlo_flag.c_str()),
        const_cast<char*>(cfg_flag.c_str()),
        const_cast<char*>(tc_flag.c_str()),
        nullptr,
    };
    execv(worker_path.c_str(), args);
    std::perror("execv");
    _exit(127);
  }

  // Parent: wait for the child.
  int wstatus = 0;
  QCHECK_EQ(waitpid(pid, &wstatus, 0), pid)
      << "waitpid failed: " << strerror(errno);
  return wstatus;
}

// Classifies a waitpid status into a ConfigResult outcome.
Outcome ClassifyWstatus(int wstatus, std::string* detail_out) {
  if (WIFSIGNALED(wstatus)) {
    *detail_out = SignalName(wstatus);
    return Outcome::kCrash;
  }
  int code = WEXITSTATUS(wstatus);
  switch (code) {
    case kExitOk:
      *detail_out = "";
      return Outcome::kOk;
    case kExitNullExecutable:
      *detail_out = "null_executable";
      return Outcome::kSkip;
    default:  // kExitStatusError or unexpected code
      *detail_out = absl::StrCat("exit_code=", code);
      return Outcome::kCompileError;
  }
}

// --------------------------------------------------------------------------
// Config generation for a given HLO file.
// --------------------------------------------------------------------------

// Loads the first HloDotInstruction from the module so we can use
// TritonDotFusionSearchSpace to enumerate the exhaustive config set.
absl::StatusOr<std::vector<TritonGemmConfig>> GenerateExhaustiveConfigs(
    const std::string& hlo_file,
    const se::DeviceDescription& device_desc) {
  std::string hlo_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), hlo_file, &hlo_text));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnUnverifiedModule(hlo_text));

  const HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation()->root_instruction()
           ->called_computations()[0],
      HloOpcode::kDot);
  if (dot == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("No dot instruction found in ", hlo_file));
  }

  // Build exhaustive search space (same as autotuner does with
  // xla_gpu_exhaustive_tiling_search=true).
  TritonDotFusionSearchSpace search_space(device_desc,
                                          Cast<HloDotInstruction>(dot));
  std::vector<TritonGemmConfig> configs = search_space.GenerateConfigs(
      /*force_contracting_split=*/std::nullopt,
      /*autotune_warp_specialization=*/false);
  return configs;
}

// --------------------------------------------------------------------------
// Main validation loop.
// --------------------------------------------------------------------------

void ValidateAllConfigs(
    const std::vector<std::string>& hlo_files,
    const std::string& worker_path,
    const se::DeviceDescription& device_desc,
    const std::string& target_config_path,
    int num_jobs,
    std::ostream& out) {
  // Header.
  out << "shape_label\tconfig\tstatus\tdetail\n";

  // Work queue protected by a mutex so threads can pull items.
  struct WorkItem {
    std::string hlo_file;
    std::string shape_label;
    std::string config_text;
  };
  std::vector<WorkItem> work_queue;
  work_queue.reserve(hlo_files.size() * 50);  // rough upper bound

  // Build full work queue (single-threaded, fast — just string ops).
  for (const std::string& hlo_file : hlo_files) {
    // Derive a short label from the filename (basename without extension).
    absl::string_view label = hlo_file;
    auto slash = label.rfind('/');
    if (slash != absl::string_view::npos) label = label.substr(slash + 1);
    auto dot = label.rfind('.');
    if (dot != absl::string_view::npos) label = label.substr(0, dot);
    const std::string shape_label(label);

    absl::StatusOr<std::vector<TritonGemmConfig>> configs =
        GenerateExhaustiveConfigs(hlo_file, device_desc);
    if (!configs.ok()) {
      LOG(ERROR) << "Failed to generate configs for " << hlo_file << ": "
                 << configs.status();
      continue;
    }
    LOG(INFO) << shape_label << ": " << configs->size() << " configs to test.";
    for (const TritonGemmConfig& cfg : *configs) {
      work_queue.push_back({hlo_file, shape_label, SerialiseConfig(cfg)});
    }
  }

  std::atomic<size_t> next_item{0};
  std::mutex out_mu;

  // Worker threads.
  auto thread_fn = [&]() {
    while (true) {
      size_t idx = next_item.fetch_add(1, std::memory_order_relaxed);
      if (idx >= work_queue.size()) break;

      const WorkItem& item = work_queue[idx];
      int wstatus = RunWorker(worker_path, item.hlo_file, item.config_text,
                              target_config_path);
      std::string detail;
      Outcome outcome = ClassifyWstatus(wstatus, &detail);

      {
        std::lock_guard<std::mutex> lock(out_mu);
        out << item.shape_label << "\t" << item.config_text << "\t"
            << OutcomeName(outcome) << "\t" << detail << "\n";
        out.flush();
        std::cerr << "[" << idx + 1 << "/" << work_queue.size() << "] "
                  << item.shape_label << " | " << OutcomeName(outcome);
        if (!detail.empty()) std::cerr << " (" << detail << ")";
        std::cerr << " | " << item.config_text << "\n";
      }
    }
  };

  int threads_to_use = std::max(1, num_jobs);
  std::vector<std::thread> threads;
  threads.reserve(threads_to_use);
  for (int i = 0; i < threads_to_use; ++i) {
    threads.emplace_back(thread_fn);
  }
  for (auto& t : threads) t.join();
}

// --------------------------------------------------------------------------
// Entry point.
// --------------------------------------------------------------------------

absl::Status Run(const std::string& worker_path,
                 const std::string& hlo_dir,
                 const std::string& output_file,
                 int num_jobs) {
  // Locate GPU device to get DeviceDescription for search space generation.
  TF_ASSIGN_OR_RETURN(std::string platform_name,
                      PlatformUtil::CanonicalPlatformName("gpu"));
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithName(
          absl::AsciiStrToUpper(platform_name)));
  if (platform->VisibleDeviceCount() == 0) {
    return absl::InternalError("No GPU devices found.");
  }
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * se,
                      platform->ExecutorForDevice(0));
  const se::DeviceDescription& device_desc = se->GetDeviceDescription();

  // Serialize the GpuTargetConfig so workers can compile without a GPU.
  GpuTargetConfig gpu_target_config(se);
  std::string target_config_text;
  tsl::protobuf::TextFormat::PrintToString(gpu_target_config.ToProto(),
                                           &target_config_text);
  std::string target_config_path;
  auto* env = tsl::Env::Default();
  if (!env->LocalTempFilename(&target_config_path)) {
    return absl::InternalError("Failed to create temp file name.");
  }
  target_config_path += ".gpu_target_config.pbtxt";
  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, target_config_path,
                                            target_config_text));
  LOG(INFO) << "Wrote GpuTargetConfig to " << target_config_path;

  // Collect HLO files.
  std::vector<std::string> hlo_files;
  std::vector<std::string> children;
  TF_RETURN_IF_ERROR(env->GetChildren(hlo_dir, &children));
  for (const std::string& child : children) {
    if (absl::EndsWith(child, ".hlo") || absl::EndsWith(child, ".txt")) {
      hlo_files.push_back(absl::StrCat(hlo_dir, "/", child));
    }
  }
  if (hlo_files.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("No .hlo/.txt files found in ", hlo_dir));
  }
  LOG(INFO) << "Found " << hlo_files.size() << " HLO files in " << hlo_dir;

  // Open output.
  if (output_file.empty()) {
    ValidateAllConfigs(hlo_files, worker_path, device_desc,
                       target_config_path, num_jobs, std::cout);
  } else {
    std::ofstream ofs(output_file);
    if (!ofs) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot open output file: ", output_file));
    }
    ValidateAllConfigs(hlo_files, worker_path, device_desc,
                       target_config_path, num_jobs, ofs);
  }

  env->DeleteFile(target_config_path).IgnoreError();
  return absl::OkStatus();
}

}  // namespace
}  // namespace gpu
}  // namespace xla

int main(int argc, char** argv) {
  std::string worker_path;
  std::string hlo_dir;
  std::string output_file;
  int num_jobs = 4;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("worker", &worker_path,
                "Absolute path to the triton_config_validator_worker binary."),
      tsl::Flag("hlo_dir", &hlo_dir,
                "Directory containing .hlo files, one per dot shape to test."),
      tsl::Flag("output_file", &output_file,
                "Path to write the TSV result report. Defaults to stdout."),
      tsl::Flag("jobs", &num_jobs,
                "Number of parallel worker subprocesses (default: 4)."),
  };

  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  if (!parse_ok || worker_path.empty() || hlo_dir.empty()) {
    std::cerr << usage << "\n";
    std::cerr << "Required: --worker and --hlo_dir\n";
    return 1;
  }

  absl::Status s =
      xla::gpu::Run(worker_path, hlo_dir, output_file, num_jobs);
  if (!s.ok()) {
    std::cerr << "Runner failed: " << s << "\n";
    return 1;
  }
  return 0;
}
