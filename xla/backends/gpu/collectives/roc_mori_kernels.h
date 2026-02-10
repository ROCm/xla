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

// Intra-node P2P Send via MORI (put model).
// Pushes 'count' elements from the local send_buffer to peer's recv_buffer,
// then writes a completion signal to peer's *signal location.
// Both data buffers and the signal must reside in MORI's symmetric heap
// (allocated via shmem::ShmemMalloc).
// The signal must be initialised to 0 before the first call.
template <class T>
int Send(T* recv_buffer, T* send_buffer, size_t count, int peer,
         uint64_t* signal);

// Intra-node P2P Recv via MORI (put model – wait only).
// Does NOT transfer any data.  Blocks the GPU stream until the local
// *signal becomes non-zero (written by the remote peer's Send), then
// resets it to 0 for the next round.
template <class T>
int Recv(T* recv_buffer, T* send_buffer, size_t count, int peer,
         uint64_t* signal);

} // namespace roc_mori

#endif // XLA_BACKENDS_GPU_COLLECTIVES_ROC_MORI_KERNELS_H_
