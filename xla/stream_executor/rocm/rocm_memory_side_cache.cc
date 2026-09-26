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
#include "absl/types/span.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/rocm/smi_util.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {
namespace {

constexpr int64_t kMiB = int64_t{1024} * 1024;

// Tier 1, the driver's cache topology.
absl::StatusOr<int64_t> TopologyMemorySideCacheSize(
    absl::string_view pci_bus_id) {
  absl::MutexLock lock(smi_mutex);

  ABSL_RETURN_IF_ERROR(InitSmi());

  ABSL_ASSIGN_OR_RETURN(BdfComponents bdf, ParseBdf(pci_bus_id));
  ABSL_ASSIGN_OR_RETURN(SmiDeviceHandle device, FindDevice(bdf));
  ABSL_ASSIGN_OR_RETURN(std::vector<SmiDataCache> caches,
                        QueryDataCaches(device));

  return MemorySideCacheSizeFromDataCaches(caches);
}

}  // namespace

int64_t MemorySideCacheSizeFromDataCaches(
    absl::Span<const SmiDataCache> caches) {
  // Entries at the same level are alternatives rather than slices. The driver
  // emits one per group of CUs and each already reports the whole pool, so
  // summing them would overcount.
  int64_t size = 0;
  for (const SmiDataCache& cache : caches) {
    if (cache.level >= 3) size = std::max(size, cache.size_bytes);
  }
  return size;
}

int64_t MemorySideCacheSizeForArch(const RocmComputeCapability& cc) {
  if (cc.gfx9_mi300()) return 256 * kMiB;  // MI300X, MI300A, MI325X
  if (cc.gfx9_mi350()) return 256 * kMiB;  // MI350X, MI355X
  return 0;
}

int64_t GetRocmMemorySideCacheSize(absl::string_view pci_bus_id,
                                   const RocmComputeCapability& cc) {
  // A successful query is authoritative even when it lists no memory-side
  // cache, as on gfx90a, so the per-gfx size is only consulted when SMI
  // cannot answer.
  absl::StatusOr<int64_t> topology = TopologyMemorySideCacheSize(pci_bus_id);
  if (topology.ok()) {
    VLOG(1) << "Memory-side cache size for " << pci_bus_id << ": " << *topology
            << " bytes, from the SMI cache topology.";
    return *topology;
  }

  // rocm_smi cannot report the cache topology below ROCm 7.13, which is
  // expected and not worth a warning.
  const absl::Status& status = topology.status();
  if (absl::IsUnimplemented(status)) {
    VLOG(1) << "No SMI cache topology for " << pci_bus_id << " ("
            << status.message() << "), using the per-gfx size.";
  } else {
    LOG(WARNING) << "No SMI cache topology for " << pci_bus_id << " ("
                 << status.message() << "), using the per-gfx size.";
  }
  return MemorySideCacheSizeForArch(cc);
}

}  // namespace stream_executor::gpu
