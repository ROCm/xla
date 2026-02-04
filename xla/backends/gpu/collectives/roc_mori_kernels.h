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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_ROC_MORI_KERNELS_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_ROC_MORI_KERNELS_H_

#include "absl/status/status.h"
#include "xla/service/collective_ops_utils.h"
// #include "xla/stream_executor/gpu/gpu_types.h"

// #include "third_party/rocshmem/rocshmem.hpp"
// #include "third_party/rocshmem/roc_mori_COLL.hpp"

namespace roc_mori {

//using stream_executor::gpu::GpuStreamHandle;

using GpuStreamHandle = std::intptr_t;

void synchronize_all();

} // namespace roc_mori

#endif // XLA_BACKENDS_GPU_COLLECTIVES_ROC_MORI_KERNELS_H_
