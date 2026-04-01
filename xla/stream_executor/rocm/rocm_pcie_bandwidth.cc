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

#include "absl/strings/string_view.h"
#include "xla/stream_executor/rocm/rocm_smi_util.h"
#include "xla/stream_executor/rocm/rocm_smi_wrapper.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {

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

  rsmi_gpu_metrics_t gpu_metrics = {};
  rsmi_status_t status =
      wrap::rsmi_dev_gpu_metrics_info_get(*dev_idx, &gpu_metrics);
  if (status != RSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    wrap::rsmi_status_string(status, &err_str);
    LOG(WARNING) << "rsmi_dev_gpu_metrics_info_get failed for " << pci_bus_id
                 << ": " << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  // pcie_link_speed is in 0.1 GT/s units. Convert to MT/s (1 GT/s = 1000 MT/s).
  uint32_t speed_mt_per_sec =
      static_cast<uint32_t>(gpu_metrics.pcie_link_speed) * 100;
  uint16_t width = gpu_metrics.pcie_link_width;

  if (speed_mt_per_sec == 0 || width == 0) {
    LOG(WARNING) << "rocm_smi gpu_metrics reported zero PCIe speed ("
                 << speed_mt_per_sec << " MT/s) or width (" << width
                 << " lanes) for " << pci_bus_id;
    return std::nullopt;
  }

  int64_t bandwidth = ComputePcieBandwidthFromSpeedAndWidth(speed_mt_per_sec,
                                                            width);

  VLOG(1) << "PCIe bandwidth for " << pci_bus_id << ": "
          << speed_mt_per_sec << " MT/s x" << width << " = "
          << bandwidth / (1024 * 1024 * 1024) << " GB/s (" << bandwidth
          << " bytes/s)";

  return bandwidth;
}

}  // namespace stream_executor::gpu
