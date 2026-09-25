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

#include "xla/stream_executor/rocm/rocm_memory_side_cache.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/rocm/smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

absl::StatusOr<int64_t> SmiMemorySideCacheSize(absl::string_view pci_bus_id) {
  absl::MutexLock lock(smi_mutex);

  ABSL_RETURN_IF_ERROR(InitSmi());

  ABSL_ASSIGN_OR_RETURN(BdfComponents bdf, ParseBdf(pci_bus_id));
  ABSL_ASSIGN_OR_RETURN(SmiDeviceHandle device, FindDevice(bdf));
  ABSL_ASSIGN_OR_RETURN(std::vector<SmiDataCache> caches,
                        QueryDataCaches(device));

  int64_t size = 0;
  for (const SmiDataCache& cache : caches) {
    if (cache.level >= 3) size = std::max(size, cache.size_bytes);
  }
  return size;
}

}  // namespace

int64_t GetRocmMemorySideCacheSize(absl::string_view pci_bus_id) {
  absl::StatusOr<int64_t> size = SmiMemorySideCacheSize(pci_bus_id);
  if (size.ok()) {
    VLOG(1) << "Memory-side cache size for " << pci_bus_id << ": " << *size
            << " bytes, from the SMI cache topology.";
    return *size;
  }

  const absl::Status& status = size.status();
  if (absl::IsUnimplemented(status)) {
    VLOG(1) << "Memory-side cache size for " << pci_bus_id
            << " is not queried with the rocm-smi backend, reporting none.";
  } else {
    LOG(WARNING) << "Could not read memory-side cache size for " << pci_bus_id
                 << " over SMI (" << status.message() << "), reporting none.";
  }
  return 0;
}

}  // namespace stream_executor::gpu
