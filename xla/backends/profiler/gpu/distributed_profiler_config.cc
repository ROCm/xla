/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/profiler/gpu/distributed_profiler_config.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "tsl/platform/logging.h"
#include "xla/tsl/util/env_var.h"

namespace xla {
namespace profiler {

DistributedProfilerConfig DistributedProfilerConfig::Load() {
  DistributedProfilerConfig config;  // Start with defaults
  
  // Step 1: Load from config file if specified
  const char* config_path = std::getenv("XLA_DIST_PROF_CONFIG");
  if (config_path != nullptr) {
    auto file_config = LoadFromFile(config_path);
    if (file_config.ok()) {
      config = *file_config;
      VLOG(1) << "Loaded distributed profiling config from: " << config_path;
    } else {
      LOG(WARNING) << "Failed to load config from " << config_path 
                   << ": " << file_config.status();
    }
  }
  
  // Step 2: Override with individual env vars (highest precedence)
  bool enabled;
  if (tsl::ReadBoolFromEnvVar("XLA_ENABLE_DISTRIBUTED_PROFILING", 
                               config.enabled, &enabled).ok()) {
    config.enabled = enabled;
  }
  
  int64_t cadence;
  if (tsl::ReadInt64FromEnvVar("XLA_PROBE_CADENCE_US", 
                                config.probe_cadence_us, &cadence).ok()) {
    config.probe_cadence_us = static_cast<int>(cadence);
  }
  
  int64_t window;
  if (tsl::ReadInt64FromEnvVar("XLA_PROBE_WINDOW_S", 
                                config.probe_window_s, &window).ok()) {
    config.probe_window_s = static_cast<int>(window);
  }
  
  int64_t spacing;
  if (tsl::ReadInt64FromEnvVar("XLA_PACKET_SPACING_US", 
                                config.packet_spacing_us, &spacing).ok()) {
    config.packet_spacing_us = static_cast<int>(spacing);
  }
  
  int64_t snapshot;
  if (tsl::ReadInt64FromEnvVar("XLA_SNAPSHOT_PERIOD_MS", 
                                config.snapshot_period_ms, &snapshot).ok()) {
    config.snapshot_period_ms = static_cast<int>(snapshot);
  }
  
  std::string output_dir;
  if (tsl::ReadStringFromEnvVar("XLA_DIST_PROF_OUTPUT_DIR", 
                                 config.output_dir, &output_dir).ok()) {
    config.output_dir = output_dir;
  }
  
  return config;
}

absl::StatusOr<DistributedProfilerConfig> 
DistributedProfilerConfig::LoadFromFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return absl::NotFoundError(absl::StrCat("Config file not found: ", path));
  }
  
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string contents = buffer.str();
  
  // Simple JSON parsing (not using nlohmann/json to avoid adding dependencies)
  // This is a minimal parser for our specific config format
  DistributedProfilerConfig config;
  
  auto find_bool = [&contents](const std::string& key) -> std::optional<bool> {
    size_t pos = contents.find("\"" + key + "\"");
    if (pos == std::string::npos) return std::nullopt;
    pos = contents.find(":", pos);
    if (pos == std::string::npos) return std::nullopt;
    // Skip whitespace
    pos = contents.find_first_not_of(" \t\n", pos + 1);
    if (pos == std::string::npos) return std::nullopt;
    if (contents.substr(pos, 4) == "true") return true;
    if (contents.substr(pos, 5) == "false") return false;
    return std::nullopt;
  };
  
  auto find_int = [&contents](const std::string& key) -> std::optional<int> {
    size_t pos = contents.find("\"" + key + "\"");
    if (pos == std::string::npos) return std::nullopt;
    pos = contents.find(":", pos);
    if (pos == std::string::npos) return std::nullopt;
    pos = contents.find_first_not_of(" \t\n", pos + 1);
    if (pos == std::string::npos) return std::nullopt;
    try {
      return std::stoi(contents.substr(pos));
    } catch (...) {
      return std::nullopt;
    }
  };
  
  auto find_string = [&contents](const std::string& key) -> std::optional<std::string> {
    size_t pos = contents.find("\"" + key + "\"");
    if (pos == std::string::npos) return std::nullopt;
    pos = contents.find(":", pos);
    if (pos == std::string::npos) return std::nullopt;
    pos = contents.find("\"", pos);
    if (pos == std::string::npos) return std::nullopt;
    size_t end = contents.find("\"", pos + 1);
    if (end == std::string::npos) return std::nullopt;
    return contents.substr(pos + 1, end - pos - 1);
  };
  
  if (auto val = find_bool("enabled")) config.enabled = *val;
  if (auto val = find_int("probe_cadence_us")) config.probe_cadence_us = *val;
  if (auto val = find_int("probe_window_s")) config.probe_window_s = *val;
  if (auto val = find_int("packet_spacing_us")) config.packet_spacing_us = *val;
  if (auto val = find_int("snapshot_period_ms")) config.snapshot_period_ms = *val;
  if (auto val = find_string("output_dir")) config.output_dir = *val;
  
  return config;
}

}  // namespace profiler
}  // namespace xla

