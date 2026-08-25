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

#include "xla/stream_executor/rocm/rocm_cache_info.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/rocm/smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {

absl::StatusOr<RocmCacheInfo> GetRocmCacheInfo(absl::string_view pci_bus_id) {
  absl::MutexLock lock(smi_mutex);

  if (!InitSmi()) return absl::UnavailableError("SMI is not available");

  absl::StatusOr<BdfComponents> bdf = ParseBdf(pci_bus_id);
  if (!bdf.ok()) return bdf.status();

  absl::StatusOr<SmiDeviceHandle> device = FindDevice(*bdf);
  if (!device.ok()) return device.status();

  absl::StatusOr<std::vector<CacheLevelInfo>> levels =
      QueryDataCacheHierarchy(*device);
  if (!levels.ok()) return levels.status();

  RocmCacheInfo info;
  // QueryDataCacheHierarchy returns levels sorted shallowest to deepest.
  info.last_level_cache_size_bytes = levels->back().size_bytes;

  // The fabric clock is optional: some ASICs do not expose the domain, and
  // that costs us only the last level cache bandwidth, not the sizes.
  absl::StatusOr<int64_t> fabric_clock =
      QueryMaxClockMhz(*device, SmiClockDomain::kFabric);
  if (fabric_clock.ok()) {
    info.fabric_clock_mhz = *fabric_clock;
  } else {
    VLOG(1) << "Fabric clock unavailable for " << pci_bus_id << ": "
            << fabric_clock.status().message();
  }

  VLOG(1) << "Cache info for " << pci_bus_id << ": last level "
          << info.last_level_cache_size_bytes / (1024 * 1024) << " MiB (L"
          << levels->back().level << "), fabric clock "
          << info.fabric_clock_mhz << " MHz";

  return info;
}

}  // namespace stream_executor::gpu
