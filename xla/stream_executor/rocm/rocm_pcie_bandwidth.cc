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

#include "xla/stream_executor/rocm/rocm_pcie_bandwidth.h"

#include <cstdint>
#include <optional>

#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/rocm/amdsmi_wrapper.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/dso_loader.h"

namespace stream_executor::gpu {
namespace {

// Returns true if the amdsmi DSO handle is available, false otherwise.
bool IsAmdSmiAvailable() {
  static bool available = []() {
    auto status = tsl::internal::CachedDsoLoader::GetAmdSmiDsoHandle();
    if (!status.ok()) {
      VLOG(1) << "amdsmi DSO not available: " << status.status().message();
      return false;
    }
    return true;
  }();
  return available;
}

// Thread-safe singleton initialization of amdsmi.
// Returns true if amdsmi was successfully initialized.
bool InitAmdsmi() {
  static bool initialized = []() {
    if (!IsAmdSmiAvailable()) return false;
    amdsmi_status_t status = wrap::amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (status != AMDSMI_STATUS_SUCCESS) {
      const char* err_str = nullptr;
      wrap::amdsmi_status_code_to_string(status, &err_str);
      LOG(WARNING) << "amdsmi_init failed: "
                   << (err_str ? err_str : "unknown error");
      return false;
    }
    return true;
  }();
  return initialized;
}

// Parses a PCI bus ID string (e.g., "0000:41:00.0") into amdsmi_bdf_t.
// Returns std::nullopt on parse failure.
std::optional<amdsmi_bdf_t> ParseBdf(absl::string_view pci_bus_id) {
  // Expected format: DDDD:BB:DD.F (domain:bus:device.function)
  // or BB:DD.F (no domain, implies domain 0)
  amdsmi_bdf_t bdf = {};

  uint64_t domain = 0, bus = 0, device = 0, function = 0;

  // Try to parse DDDD:BB:DD.F format
  size_t first_colon = pci_bus_id.find(':');
  if (first_colon == absl::string_view::npos) return std::nullopt;

  size_t second_colon = pci_bus_id.find(':', first_colon + 1);
  size_t dot;

  if (second_colon != absl::string_view::npos) {
    // DDDD:BB:DD.F format
    dot = pci_bus_id.find('.', second_colon + 1);
    if (dot == absl::string_view::npos) return std::nullopt;

    if (!absl::SimpleHexAtoi(pci_bus_id.substr(0, first_colon), &domain))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(first_colon + 1, second_colon - first_colon - 1),
            &bus))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(second_colon + 1, dot - second_colon - 1),
            &device))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(dot + 1), &function))
      return std::nullopt;
  } else {
    // BB:DD.F format (domain = 0)
    dot = pci_bus_id.find('.', first_colon + 1);
    if (dot == absl::string_view::npos) return std::nullopt;

    domain = 0;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(0, first_colon), &bus))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(
            pci_bus_id.substr(first_colon + 1, dot - first_colon - 1), &device))
      return std::nullopt;
    if (!absl::SimpleHexAtoi(pci_bus_id.substr(dot + 1), &function))
      return std::nullopt;
  }

  bdf.domain_number = domain;
  bdf.bus_number = bus;
  bdf.device_number = device;
  bdf.function_number = function;

  return bdf;
}

}  // namespace

int64_t ComputePcieBandwidthFromSpeedAndWidth(uint32_t speed_mt_per_sec,
                                              uint16_t width) {
  // PCIe transfer rates and their encoding overhead:
  // Gen1: 2500 MT/s, 8b/10b encoding (80% efficiency)
  // Gen2: 5000 MT/s, 8b/10b encoding (80% efficiency)
  // Gen3: 8000 MT/s, 128b/130b encoding (~98.46% efficiency)
  // Gen4: 16000 MT/s, 128b/130b encoding (~98.46% efficiency)
  // Gen5: 32000 MT/s, 128b/130b encoding (~98.46% efficiency)
  // Gen6: 64000 MT/s, 242b/256b encoding (~94.53% efficiency)
  //
  // Each transfer moves 1 bit per lane.
  // Bandwidth = speed_MT_s * 1e6 bits/sec * width lanes / 8 bits_per_byte
  //             * encoding_efficiency

  if (width == 0 || speed_mt_per_sec == 0) return 0;

  double encoding_efficiency;
  if (speed_mt_per_sec <= 5000) {
    // Gen1/Gen2: 8b/10b encoding
    encoding_efficiency = 0.8;
  } else if (speed_mt_per_sec <= 32000) {
    // Gen3/Gen4/Gen5: 128b/130b encoding
    encoding_efficiency = 128.0 / 130.0;
  } else {
    // Gen6+: 242b/256b encoding (FLIT mode)
    encoding_efficiency = 242.0 / 256.0;
  }

  // bits_per_second = speed_MT_s * 1e6 * width
  // bytes_per_second = bits_per_second / 8 * encoding_efficiency
  double bandwidth =
      static_cast<double>(speed_mt_per_sec) * 1e6 * width / 8.0 *
      encoding_efficiency;

  return static_cast<int64_t>(bandwidth);
}

std::optional<int64_t> GetRocmPcieBandwidth(absl::string_view pci_bus_id) {
  if (!InitAmdsmi()) return std::nullopt;

  std::optional<amdsmi_bdf_t> bdf = ParseBdf(pci_bus_id);
  if (!bdf.has_value()) {
    LOG(WARNING) << "Failed to parse PCI bus ID: " << pci_bus_id;
    return std::nullopt;
  }

  amdsmi_processor_handle processor_handle;
  amdsmi_status_t status =
      wrap::amdsmi_get_processor_handle_from_bdf(*bdf, &processor_handle);
  if (status != AMDSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    wrap::amdsmi_status_code_to_string(status, &err_str);
    LOG(WARNING) << "amdsmi_get_processor_handle_from_bdf failed for "
                 << pci_bus_id << ": " << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  amdsmi_pcie_info_t pcie_info = {};
  status = wrap::amdsmi_get_pcie_info(processor_handle, &pcie_info);
  if (status != AMDSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    wrap::amdsmi_status_code_to_string(status, &err_str);
    LOG(WARNING) << "amdsmi_get_pcie_info failed for " << pci_bus_id << ": "
                 << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  uint32_t speed = pcie_info.pcie_metric.pcie_speed;
  uint16_t width = pcie_info.pcie_metric.pcie_width;

  if (speed == 0 || width == 0) {
    VLOG(1) << "amdsmi reported zero PCIe speed (" << speed << " MT/s) or "
            << "width (" << width << " lanes) for " << pci_bus_id
            << ". Falling back to static info.";
    speed = pcie_info.pcie_static.max_pcie_speed;
    width = pcie_info.pcie_static.max_pcie_width;
    if (speed == 0 || width == 0) {
      LOG(WARNING) << "amdsmi reported zero PCIe static info for "
                   << pci_bus_id;
      return std::nullopt;
    }
  }

  int64_t bandwidth = ComputePcieBandwidthFromSpeedAndWidth(speed, width);

  VLOG(1) << "PCIe bandwidth for " << pci_bus_id << ": " << speed
          << " MT/s x" << width << " = " << bandwidth / (1024 * 1024 * 1024)
          << " GB/s (" << bandwidth << " bytes/s)";

  return bandwidth;
}

}  // namespace stream_executor::gpu
