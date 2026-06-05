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

#include "xla/stream_executor/rocm/rocm_memory_bandwidth.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "absl/strings/string_view.h"
#include "rocm/include/amd_smi/amdsmi.h"
#include "xla/stream_executor/rocm/amdsmi_wrapper.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/rocm/rocm_smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

constexpr int64_t kGbps = int64_t{1000} * 1000 * 1000;

bool InitAmdSmi() {
  static bool initialized = []() {
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

// Tier 1: firmware-reported peak VRAM bandwidth (bytes/s) from amd_smi
// gpu_metrics. amd_smi is used here rather than rocm_smi because the legacy
// rocm_smi parser strictly version-gates the gpu_metrics table and rejects the
// v1.9 format current kernels emit for the MI300/MI350 family, whereas amd_smi
// parses it. amd_smi is loaded via the dso loader (see amdsmi_wrapper.h) and
// must not be linked: it embeds rocm_smi's symbols, which would otherwise clash
// with the linked rocm_smi. SMU firmware leaves vram_max_bandwidth at a
// UINT64_MAX sentinel on parts that don't populate it (e.g. CDNA2 and RDNA), so
// guard for that.
std::optional<int64_t> FirmwareBandwidth(absl::string_view pci_bus_id) {
  if (!InitAmdSmi()) return std::nullopt;

  std::optional<BdfComponents> bdf = ParseBdf(pci_bus_id);
  if (!bdf.has_value()) {
    LOG(WARNING) << "Failed to parse PCI bus ID: " << pci_bus_id;
    return std::nullopt;
  }

  amdsmi_bdf_t amd_bdf = {};
  amd_bdf.bdf.domain_number = bdf->domain;
  amd_bdf.bdf.bus_number = bdf->bus;
  amd_bdf.bdf.device_number = bdf->device;
  amd_bdf.bdf.function_number = bdf->function;

  amdsmi_processor_handle handle = nullptr;
  amdsmi_status_t status =
      wrap::amdsmi_get_processor_handle_from_bdf(amd_bdf, &handle);
  if (status != AMDSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    wrap::amdsmi_status_code_to_string(status, &err_str);
    LOG(WARNING) << "amd_smi: could not find device for PCI bus ID "
                 << pci_bus_id << ": " << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  amdsmi_gpu_metrics_t gpu_metrics = {};
  status = wrap::amdsmi_get_gpu_metrics_info(handle, &gpu_metrics);
  if (status != AMDSMI_STATUS_SUCCESS) {
    const char* err_str = nullptr;
    wrap::amdsmi_status_code_to_string(status, &err_str);
    LOG(WARNING) << "amdsmi_get_gpu_metrics_info failed for " << pci_bus_id
                 << ": " << (err_str ? err_str : "unknown error");
    return std::nullopt;
  }

  // vram_max_bandwidth is the peak (at max memory clock), reported in GB/s.
  uint64_t gbps = gpu_metrics.vram_max_bandwidth;
  if (gbps == 0 || gbps == std::numeric_limits<uint64_t>::max()) {
    VLOG(1) << "amd_smi gpu_metrics has no usable vram_max_bandwidth for "
            << pci_bus_id << " (got " << gbps << ")";
    return std::nullopt;
  }
  VLOG(1) << "amd_smi gpu_metrics VRAM bandwidth for " << pci_bus_id << ": "
          << gbps << " GB/s (tier 1, firmware)";
  return static_cast<int64_t>(gbps) * kGbps;
}

// Tiers 2 and 3: a per-gfx peak where known, else the legacy formula. The
// per-gfx values are spec (or, for gfx950, firmware-reported) peaks; the
// HBM2/HBM2e entries equal what the formula already produces and are listed for
// a single source of truth. Cf. upstream Triton's amd_bps_by_arch table.
int64_t ArchOrFormulaBandwidth(const RocmComputeCapability& cc,
                               int64_t mem_bus_width_bits,
                               int64_t mem_clock_khz) {
  if (cc.gfx9_mi100()) return 1230 * kGbps;  // MI100, HBM2
  if (cc.gfx9_mi200()) return 1600 * kGbps;  // MI200, HBM2e
  if (cc.gfx9_mi300()) return 5300 * kGbps;  // MI300X, HBM3
  if (cc.gfx9_mi350()) return 6810 * kGbps;  // MI350X, HBM3e (MI355X via tier 1)
  if (cc.gfx12_discrete()) return 640 * kGbps;  // RX 9070 XT, GDDR6

  // Unmodeled arch: assume the reported clock is the data-rate clock (true for
  // HBM2/HBM2e). Undercounts on newer memory, but the best estimate available.
  // mem_bandwidth = 2 * mem_bus_width_in_bytes * mem_clock_rate_in_hz
  return 2 * (mem_bus_width_bits / 8) * (mem_clock_khz * 1000);
}

}  // namespace

int64_t GetRocmMemoryBandwidth(absl::string_view pci_bus_id,
                               const RocmComputeCapability& cc,
                               int64_t mem_bus_width_bits,
                               int64_t mem_clock_khz) {
  return FirmwareBandwidth(pci_bus_id)
      .value_or(
          ArchOrFormulaBandwidth(cc, mem_bus_width_bits, mem_clock_khz));
}

}  // namespace stream_executor::gpu
