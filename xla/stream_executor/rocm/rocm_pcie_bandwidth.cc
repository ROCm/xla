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
#include "xla/stream_executor/rocm/rocm_smi_wrapper.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/dso_loader.h"

namespace stream_executor::gpu {
namespace {

// Returns true if the rocm_smi DSO handle is available, false otherwise.
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

// Thread-safe singleton initialization of rocm_smi.
// Returns true if rocm_smi was successfully initialized.
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

// Parses a PCI bus ID string (e.g., "0000:41:00.0") into its BDF components.
// Returns std::nullopt on parse failure.
struct BdfComponents {
  uint64_t domain;
  uint64_t bus;
  uint64_t device;
  uint64_t function;
};

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

// Finds the rocm_smi device index that matches the given PCI bus ID.
// rocm_smi uses rsmi_dev_pci_id_get which returns a packed uint64_t BDFID:
//   ((DOMAIN & 0xFFFFFFFF) << 32) | ((BUS & 0xFF) << 8)
//                                 | ((DEVICE & 0x1F) << 3) | (FUNCTION & 0x7)
// Note: bits [31:28] may contain partition ID, but we mask them out.
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
  if (!InitRocmSmi()) return std::nullopt;

  std::optional<BdfComponents> bdf = ParseBdf(pci_bus_id);
  if (!bdf.has_value()) {
    LOG(WARNING) << "Failed to parse PCI bus ID: " << pci_bus_id;
    return std::nullopt;
  }

  std::optional<uint32_t> dev_idx = FindDeviceIndex(*bdf);
  if (!dev_idx.has_value()) {
    LOG(WARNING) << "rocm_smi: could not find device for PCI bus ID "
                 << pci_bus_id;
    return std::nullopt;
  }

  rsmi_pcie_bandwidth_t pcie_bw = {};
  rsmi_status_t status = wrap::rsmi_dev_pci_bandwidth_get(*dev_idx, &pcie_bw);
  if (status != RSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    wrap::rsmi_status_string(status, &err_str);
    LOG(WARNING) << "rsmi_dev_pci_bandwidth_get failed for " << pci_bus_id
                 << ": " << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  uint32_t current_idx = pcie_bw.transfer_rate.current;
  if (current_idx >= pcie_bw.transfer_rate.num_supported) {
    // Fall back to the maximum supported entry.
    if (pcie_bw.transfer_rate.num_supported == 0) {
      LOG(WARNING) << "rocm_smi reported no supported PCIe rates for "
                   << pci_bus_id;
      return std::nullopt;
    }
    current_idx = pcie_bw.transfer_rate.num_supported - 1;
  }

  // transfer_rate.frequency[] is in Hz (transfers/second).
  // Convert to MT/s by dividing by 1e6.
  uint64_t speed_hz = pcie_bw.transfer_rate.frequency[current_idx];
  uint32_t width = pcie_bw.lanes[current_idx];

  uint32_t speed_mt_per_sec = static_cast<uint32_t>(speed_hz / 1000000);

  if (speed_mt_per_sec == 0 || width == 0) {
    LOG(WARNING) << "rocm_smi reported zero PCIe speed (" << speed_mt_per_sec
                 << " MT/s) or width (" << width << " lanes) for "
                 << pci_bus_id;
    return std::nullopt;
  }

  int64_t bandwidth = ComputePcieBandwidthFromSpeedAndWidth(speed_mt_per_sec,
                                                            width);

  VLOG(1) << "PCIe bandwidth for " << pci_bus_id << ": " << speed_mt_per_sec
          << " MT/s x" << width << " = " << bandwidth / (1024 * 1024 * 1024)
          << " GB/s (" << bandwidth << " bytes/s)";

  return bandwidth;
}

}  // namespace stream_executor::gpu
