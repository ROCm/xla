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

#include "xla/stream_executor/rocm/rocm_smi_util.h"

#include <cstdint>
#include <optional>

#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/rocm/rocm_smi_wrapper.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/dso_loader.h"

namespace stream_executor::gpu {

bool IsRocmSmiAvailable() {
  static bool available = []() {
    auto status = tsl::internal::CachedDsoLoader::GetRocmSmiDsoHandle();
    if (!status.ok()) {
      VLOG(1) << "rocm_smi DSO not available: " << status.status().message();
      return false;
    }
    return true;
  }();
  return available;
}

bool InitRocmSmi() {
  static bool initialized = []() {
    if (!IsRocmSmiAvailable()) return false;
    rsmi_status_t status = wrap::rsmi_init(0);
    if (status != RSMI_STATUS_SUCCESS) {
      const char* err_str = nullptr;
      wrap::rsmi_status_string(status, &err_str);
      LOG(WARNING) << "rsmi_init failed: "
                   << (err_str ? err_str : "unknown error");
      return false;
    }
    return true;
  }();
  return initialized;
}

std::optional<BdfComponents> ParseBdf(absl::string_view pci_bus_id) {
  BdfComponents bdf = {};

  size_t first_colon = pci_bus_id.find(':');
  if (first_colon == absl::string_view::npos) return std::nullopt;

  size_t second_colon = pci_bus_id.find(':', first_colon + 1);
  size_t dot;

  if (second_colon != absl::string_view::npos) {
    // DDDD:BB:DD.F format
    dot = pci_bus_id.find('.', second_colon + 1);
    if (dot == absl::string_view::npos) return std::nullopt;

    if (!absl::SimpleHexAtoi(pci_bus_id.substr(0, first_colon), &bdf.domain))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(first_colon + 1, second_colon - first_colon - 1),
            &bdf.bus))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(second_colon + 1, dot - second_colon - 1),
            &bdf.device))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(dot + 1), &bdf.function))
      return std::nullopt;
  } else {
    // BB:DD.F format (domain = 0)
    dot = pci_bus_id.find('.', first_colon + 1);
    if (dot == absl::string_view::npos) return std::nullopt;

    bdf.domain = 0;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(0, first_colon), &bdf.bus))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(first_colon + 1, dot - first_colon - 1),
            &bdf.device))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(dot + 1), &bdf.function))
      return std::nullopt;
  }

  return bdf;
}

std::optional<uint32_t> FindDeviceIndex(const BdfComponents& target_bdf) {
  uint32_t num_devices = 0;
  rsmi_status_t status = wrap::rsmi_num_monitor_devices(&num_devices);
  if (status != RSMI_STATUS_SUCCESS || num_devices == 0) {
    return std::nullopt;
  }

  uint64_t target_bdfid =
      ((target_bdf.domain & 0xFFFFFFFF) << 32) |
      ((target_bdf.bus & 0xFF) << 8) |
      ((target_bdf.device & 0x1F) << 3) |
      (target_bdf.function & 0x7);

  for (uint32_t i = 0; i < num_devices; ++i) {
    uint64_t bdfid = 0;
    status = wrap::rsmi_dev_pci_id_get(i, &bdfid);
    if (status != RSMI_STATUS_SUCCESS) continue;

    // Mask out partition bits [31:28] for comparison.
    uint64_t bdfid_masked = bdfid & ~(static_cast<uint64_t>(0xF) << 28);
    uint64_t target_masked = target_bdfid & ~(static_cast<uint64_t>(0xF) << 28);

    if (bdfid_masked == target_masked) {
      return i;
    }
  }

  return std::nullopt;
}

}  // namespace stream_executor::gpu
