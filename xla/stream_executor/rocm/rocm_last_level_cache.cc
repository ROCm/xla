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

#include "xla/stream_executor/rocm/rocm_last_level_cache.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/rocm/smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {

absl::StatusOr<int64_t> GetRocmLastLevelCacheSize(
    absl::string_view pci_bus_id) {
  absl::MutexLock lock(smi_mutex);

  if (!InitSmi()) return absl::UnavailableError("SMI is not available");

  absl::StatusOr<BdfComponents> bdf = ParseBdf(pci_bus_id);
  if (!bdf.ok()) return bdf.status();

  absl::StatusOr<SmiDeviceHandle> device = FindDevice(*bdf);
  if (!device.ok()) return device.status();

  absl::StatusOr<int64_t> cache_size = QueryLastLevelCacheSize(*device);
  if (!cache_size.ok()) return cache_size.status();

  VLOG(1) << "Last level cache size for " << pci_bus_id << ": "
          << *cache_size / (1024 * 1024) << " MiB (" << *cache_size
          << " bytes)";

  return *cache_size;
}

}  // namespace stream_executor::gpu
