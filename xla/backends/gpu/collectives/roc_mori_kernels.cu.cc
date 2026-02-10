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

#include "xla/backends/gpu/collectives/roc_mori_kernels.h"

#include <algorithm>

#include "third_party/mori/shmem/shmem.hpp"

namespace roc_mori {
namespace {

// Launch configuration constants.
constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 128;

// Number of T-elements each warp processes per loop iteration.
// 1024 elements × 4 B/float = 4 KB per warp-chunk — large enough
// for good vectorisation yet small enough for balanced work distribution.
constexpr size_t kChunkElems = 1024;

// --------------------------------------------------------------------------
// MoriPutCopyKernel – pushes data from the local send_buffer to the peer's
// recv_buffer via P2P.  Both buffers live in MORI's symmetric heap.
// --------------------------------------------------------------------------
template <typename T>
__global__ void MoriPutCopyKernel(T* recv_buffer, const T* send_buffer,
                                  size_t count, int peer) {
  int myPe = mori::shmem::ShmemMyPe();

  // Translate the local symmetric address of recv_buffer to the
  // P2P-mapped address on the remote peer.
  uint64_t remote_addr = mori::shmem::ShmemPtrP2p(
      reinterpret_cast<uint64_t>(recv_buffer), myPe, peer);
  T* dst = reinterpret_cast<T*>(remote_addr);

  int warpSz     = warpSize;
  int warpId     = (blockIdx.x * blockDim.x + threadIdx.x) / warpSz;
  int totalWarps = (gridDim.x * blockDim.x) / warpSz;

  for (size_t off = static_cast<size_t>(warpId) * kChunkElems;
       off < count;
       off += static_cast<size_t>(totalWarps) * kChunkElems) {
    size_t n = (off + kChunkElems <= count) ? kChunkElems : (count - off);
    mori::core::WarpCopy<T>(dst + off, send_buffer + off, n);
  }

  // Make sure every write to the remote GPU memory is globally visible
  // before the kernel completes.
  __threadfence_system();
}

// --------------------------------------------------------------------------
// MoriPutSignalKernel – writes a signal value to the peer's signal location
// via P2P.  Launched after MoriPutCopyKernel on the same stream so that
// the signal is only written once all data has landed.
// --------------------------------------------------------------------------
__global__ void MoriPutSignalKernel(uint64_t* signal, int peer) {
  int myPe = mori::shmem::ShmemMyPe();
  uint64_t remote_addr = mori::shmem::ShmemPtrP2p(
      reinterpret_cast<uint64_t>(signal), myPe, peer);
  mori::core::AtomicStoreRelaxedSystem(
      reinterpret_cast<uint64_t*>(remote_addr), uint64_t{1});
}

// --------------------------------------------------------------------------
// MoriWaitSignalKernel – spins on a local signal location until the remote
// peer writes a non-zero value, then resets it to 0 for the next round.
// --------------------------------------------------------------------------
__global__ void MoriWaitSignalKernel(uint64_t* signal) {
  // Spin until the sender has set the signal.
  while (mori::core::AtomicLoadRelaxedSystem(signal) == 0) {
  }
  // Reset for the next Send/Recv round.
  mori::core::AtomicStoreRelaxedSystem(signal, uint64_t{0});
  __threadfence_system();
}

}  // anonymous namespace

// --------------------------------------------------------------------------
// Host-side Send (put model).
//   1. Launches the bulk copy kernel  (multi-block, multi-warp).
//   2. Launches a tiny signal kernel  (1 thread) that writes to the peer's
//      signal location.  Stream ordering guarantees the signal is only
//      written after the copy is complete.
// --------------------------------------------------------------------------
template <class T>
int Send(T* recv_buffer, T* send_buffer, size_t count, int peer,
         uint64_t* signal) {
  constexpr int kWarpSize = 64;  // AMD GFX9 warp (wavefront) size
  constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
  int numBlocks = static_cast<int>(
      (count + kChunkElems * kWarpsPerBlock - 1) /
      (kChunkElems * kWarpsPerBlock));
  numBlocks = std::max(1, std::min(numBlocks, kMaxBlocks));

  // Kernel 1: bulk WarpCopy data to peer's recv_buffer.
  MoriPutCopyKernel<<<numBlocks, kBlockSize>>>(
      recv_buffer, send_buffer, count, peer);

  // Kernel 2: write the completion signal on the peer (1 thread).
  if (signal != nullptr) {
    MoriPutSignalKernel<<<1, 1>>>(signal, peer);
  }
  return 0;
}

// --------------------------------------------------------------------------
// Host-side Recv (put model – wait only).
//   Does NOT copy any data.  The remote peer is expected to have pushed the
//   data via Send().  This call simply blocks the GPU stream until the
//   local signal becomes non-zero, then resets it.
// --------------------------------------------------------------------------
template <class T>
int Recv(T* /*recv_buffer*/, T* /*send_buffer*/, size_t /*count*/,
         int /*peer*/, uint64_t* signal) {
  if (signal != nullptr) {
    MoriWaitSignalKernel<<<1, 1>>>(signal);
  }
  return 0;
}

// Explicit template instantiations for the types used by XLA.
template int Send<float>(float*, float*, size_t, int, uint64_t*);
template int Send<double>(double*, double*, size_t, int, uint64_t*);
template int Send<long long>(long long*, long long*, size_t, int, uint64_t*);
template int Send<int>(int*, int*, size_t, int, uint64_t*);
template int Send<short>(short*, short*, size_t, int, uint64_t*);
template int Send<char>(char*, char*, size_t, int, uint64_t*);

template int Recv<float>(float*, float*, size_t, int, uint64_t*);
template int Recv<double>(double*, double*, size_t, int, uint64_t*);
template int Recv<long long>(long long*, long long*, size_t, int, uint64_t*);
template int Recv<int>(int*, int*, size_t, int, uint64_t*);
template int Recv<short>(short*, short*, size_t, int, uint64_t*);
template int Recv<char>(char*, char*, size_t, int, uint64_t*);

void synchronize_all() {
  hipDeviceSynchronize();
}

}  // namespace roc_mori
