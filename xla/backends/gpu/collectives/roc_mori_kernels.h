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

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "xla/service/collective_ops_utils.h"
// #include "xla/stream_executor/gpu/gpu_types.h"

// #include "third_party/rocshmem/rocshmem.hpp"
// #include "third_party/rocshmem/roc_mori_COLL.hpp"

namespace roc_mori {

//using stream_executor::gpu::GpuStreamHandle;

using GpuStreamHandle = std::intptr_t;

void synchronize_all();

// Zero-initialise signal flag memory (device memory / symmetric heap).
void InitSignalMemory(void* ptr, size_t bytes);

// Intra-node P2P Send via MORI (put model, single kernel).
int Send(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
         uint32_t* signal_flags, std::intptr_t stream_handle);

// Intra-node P2P Recv via MORI (put model – wait only).
int Recv(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
         uint32_t* signal_flags, std::intptr_t stream_handle);

// P2P barrier across all PEs, enqueued on the given stream.
// barrier_count must be a single uint32_t in MORI's symmetric heap.
// round is 1-based and must be incremented by the caller each invocation.
void BarrierOnStream(uint32_t* barrier_count, uint32_t round,
                     std::intptr_t stream_handle);

} // namespace roc_mori

#endif // XLA_BACKENDS_GPU_COLLECTIVES_ROC_MORI_KERNELS_H_
