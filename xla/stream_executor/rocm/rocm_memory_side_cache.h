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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_SIDE_CACHE_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_SIDE_CACHE_H_

#include <cstdint>

#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

// Returns the memory-side cache size in bytes read over SMI, or 0 if the device
// has none or SMI cannot report it.
int64_t GetRocmMemorySideCacheSize(absl::string_view pci_bus_id);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_SIDE_CACHE_H_
