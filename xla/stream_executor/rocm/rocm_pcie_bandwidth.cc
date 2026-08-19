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

#include "xla/stream_executor/rocm/rocm_pcie_bandwidth.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/rocm/rocm_smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

// PCIe encoding efficiencies by generation
constexpr double kPcieGen1Gen2Efficiency = 0.8;
constexpr double kPcieGen3To5Efficiency = 128.0 / 130.0;
constexpr double kPcieGen6Efficiency = 242.0 / 256.0;

// PCIe transfer rate thresholds in MT/s
constexpr uint32_t kPcieGen2MaxSpeedMTps = 5000;
constexpr uint32_t kPcieGen5MaxSpeedMTps = 32000;

// Rate of the newest PCIe generation this file knows about. Bump it when a
// faster one ships; exceeding it is diagnostic only, never a rejection.
constexpr uint32_t kNewestKnownPcieSpeedMTps = 128000;  // Gen7

// gpu_metrics marks an unpopulated field all ones, and amd_smi widens that
// uint16 sentinel to uint32 instead of clearing it, so it arrives verbatim.
constexpr uint32_t kUnpopulatedPcieSpeedMTps =
    std::numeric_limits<uint32_t>::max();

constexpr double PcieEncodingEfficiency(uint32_t speed_mt_per_sec) {
  if (speed_mt_per_sec <= kPcieGen2MaxSpeedMTps) return kPcieGen1Gen2Efficiency;
  if (speed_mt_per_sec <= kPcieGen5MaxSpeedMTps) return kPcieGen3To5Efficiency;
  return kPcieGen6Efficiency;
}

constexpr int64_t ComputePcieBandwidthFromSpeedAndWidth(
    uint32_t speed_mt_per_sec, uint16_t width) {
  if (width == 0 || speed_mt_per_sec == 0) return 0;
  double efficiency = PcieEncodingEfficiency(speed_mt_per_sec);
  return static_cast<int64_t>(static_cast<double>(speed_mt_per_sec) * 1e6 *
                              width / 8.0 * efficiency);
}

struct PcieLinkStatus {
  uint32_t speed_mt_per_sec;
  uint16_t width;
};

#if (TF_ROCM_VERSION >= 71300)

std::optional<PcieLinkStatus> QueryPcieLinkStatus(
    SmiDeviceHandle device, absl::string_view pci_bus_id) {
  amdsmi_pcie_info_t pcie_info = {};
  amdsmi_status_t status = amdsmi_get_pcie_info(device, &pcie_info);
  if (status != AMDSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    amdsmi_status_code_to_string(status, &err_str);
    LOG(WARNING) << "amdsmi_get_pcie_info failed for " << pci_bus_id << ": "
                 << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  // pcie_speed is MT/s per amdsmi.h ("current PCIe speed in MT/s"). amd_smi
  // reads the same raw field the rocm_smi path below does and applies the
  // NormalizePcieLinkSpeed conversion internally.
  return PcieLinkStatus{pcie_info.pcie_metric.pcie_speed,
                        pcie_info.pcie_metric.pcie_width};
}

#else

// Converts raw gpu_metrics pcie_link_speed to MT/s, nullopt if unpopulated.
// The field is documented as 0.1 GT/s, but some firmware reports the PCIe
// generation instead. Mirrors amd_smi's disambiguation of the same field
// (amdsmi_get_pcie_info, smi_amdgpu_get_pcie_speed_from_pcie_type); the two
// branches would otherwise differ by up to 40x on identical hardware.
std::optional<uint32_t> NormalizePcieLinkSpeed(uint16_t raw) {
  if (raw == std::numeric_limits<uint16_t>::max()) return std::nullopt;

  switch (raw) {
    case 1:
      return 2500;
    case 2:
      return 5000;
    case 3:
      return 8000;
    case 4:
      return 16000;
    case 5:
      return 32000;
    case 6:
      return 64000;
    default:
      // 0.1 GT/s form. Zero lands here too and the caller rejects it.
      return static_cast<uint32_t>(raw) * 100;
  }
}

std::optional<PcieLinkStatus> QueryPcieLinkStatus(
    SmiDeviceHandle device, absl::string_view pci_bus_id) {
  rsmi_gpu_metrics_t gpu_metrics = {};
  rsmi_status_t status = rsmi_dev_gpu_metrics_info_get(device, &gpu_metrics);
  if (status != RSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    rsmi_status_string(status, &err_str);
    LOG(WARNING) << "rsmi_dev_gpu_metrics_info_get failed for " << pci_bus_id
                 << ": " << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  std::optional<uint32_t> speed_mt_per_sec =
      NormalizePcieLinkSpeed(gpu_metrics.pcie_link_speed);
  if (!speed_mt_per_sec.has_value()) {
    LOG(WARNING) << "rocm_smi gpu_metrics carries no PCIe link speed for "
                 << pci_bus_id;
    return std::nullopt;
  }

  return PcieLinkStatus{*speed_mt_per_sec, gpu_metrics.pcie_link_width};
}

#endif  // TF_ROCM_VERSION >= 71300

}  // namespace

std::optional<int64_t> GetRocmPcieBandwidth(absl::string_view pci_bus_id) {
  absl::MutexLock lock(rocm_smi_mutex);

  if (!InitRocmSmi()) return std::nullopt;

  std::optional<BdfComponents> bdf = ParseBdf(pci_bus_id);
  if (!bdf.has_value()) {
    LOG(WARNING) << "Failed to parse PCI bus ID: " << pci_bus_id;
    return std::nullopt;
  }

  std::optional<SmiDeviceHandle> device = FindDeviceIndex(*bdf);
  if (!device.has_value()) {
    LOG(WARNING) << kSmiLibraryName << " could not find device for PCI bus ID "
                 << pci_bus_id;
    return std::nullopt;
  }

  std::optional<PcieLinkStatus> link = QueryPcieLinkStatus(*device, pci_bus_id);
  if (!link.has_value()) return std::nullopt;

  uint32_t speed_mt_per_sec = link->speed_mt_per_sec;
  uint16_t width = link->width;

  if (speed_mt_per_sec == 0 || width == 0) {
    LOG(WARNING) << kSmiLibraryName << " reported zero PCIe speed ("
                 << speed_mt_per_sec << " MT/s) or width (" << width
                 << " lanes) for " << pci_bus_id;
    return std::nullopt;
  }

  if (speed_mt_per_sec == kUnpopulatedPcieSpeedMTps) {
    LOG(WARNING) << kSmiLibraryName << " reported no PCIe speed for "
                 << pci_bus_id;
    return std::nullopt;
  }

  // Warn but keep the reading: magnitude cannot tell a new PCIe generation
  // from corruption, and discarding a valid rate is worse than trusting an
  // unfamiliar one. Corruption we can name is rejected above instead.
  LOG_IF(WARNING, speed_mt_per_sec > kNewestKnownPcieSpeedMTps)
      << kSmiLibraryName << " reported " << speed_mt_per_sec << " MT/s for "
      << pci_bus_id << ", above the newest PCIe generation known here ("
      << kNewestKnownPcieSpeedMTps
      << " MT/s). Using it anyway; raise kNewestKnownPcieSpeedMTps if this is "
         "a real link rate.";

  int64_t bandwidth =
      ComputePcieBandwidthFromSpeedAndWidth(speed_mt_per_sec, width);

  VLOG(1) << "PCIe bandwidth for " << pci_bus_id << " via " << kSmiLibraryName
          << ": " << speed_mt_per_sec << " MT/s x" << width << " = "
          << bandwidth / (1024 * 1024 * 1024) << " GB/s (" << bandwidth
          << " bytes/s)";

  return bandwidth;
}

}  // namespace stream_executor::gpu
