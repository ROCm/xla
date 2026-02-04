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

constexpr int kWarpSize = 64, kWarpsPerBlock = 2;
constexpr int kBlockSize = kWarpSize * kWarpsPerBlock;
constexpr int kMaxBlocks = 16;
constexpr size_t kBytesPerWarp = 2*1024;

// --------------------------------------------------------------------------
// MoriPutKernel – single kernel that:
//   1. Copies data from local send_buffer to peer's recv_buffer via P2P.
//   2. Uses a retirement counter so that the LAST block to finish sets
//      a completion flag on the remote peer's signal_flags[myPe].
//
// signal_flags  – base of the per-PE signal array (in symmetric heap).
// block_counter – a single uint32_t in device memory, initialised to 0.
//                 It is used only within this kernel and reset before exit.
// --------------------------------------------------------------------------
__global__ void MoriPutKernel(void* recv_buffer, void* send_buffer,
                              size_t bytes, int peer,
                              uint32_t* signal_flags) {
  using T = uint8_t;
  T *src = static_cast<T *>(send_buffer), *dst;
  uint32_t *remote_sig;
  {
    int myPe = mori::shmem::ShmemMyPe();
    // Translate the local symmetric address of recv_buffer to the
    // P2P-mapped address on the remote peer.
    uint64_t remote_addr = mori::shmem::ShmemPtrP2p(
      reinterpret_cast<uint64_t>(recv_buffer), myPe, peer);
    dst = reinterpret_cast<T*>(remote_addr);
    remote_addr = mori::shmem::ShmemPtrP2p(
      reinterpret_cast<uint64_t>(signal_flags + myPe + 1), myPe, peer);
    remote_sig = reinterpret_cast<uint32_t*>(remote_addr);
    if (threadIdx.x == 0) {
      while (mori::core::AtomicLoadRelaxedSystem(remote_sig) != 0) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
  }
  __syncthreads();

  uint32_t warpId = blockIdx.x * blockDim.x + threadIdx.x,
           totalWarps = gridDim.x * blockDim.x;
  if (warpSize == 64) {
    warpId /= 64, totalWarps /= 64;
  } else {
    warpId /= 32, totalWarps /= 32;
  }
  for (size_t off = static_cast<size_t>(warpId) * kBytesPerWarp; off < bytes;
              off += static_cast<size_t>(totalWarps) * kBytesPerWarp) {
    size_t n = std::min(kBytesPerWarp, bytes - off);
    mori::core::WarpCopy<T>(dst + off, src + off, n);
  }

  // Ensure all P2P writes from this thread are globally visible.
  __threadfence_system();
  // Intra-block barrier: every thread in this block has completed its
  // WarpCopy + fence before we touch the retirement counter.
  __syncthreads();

  if (threadIdx.x == 0) {
    // no need to transfer this flag => it lives on this node
    auto prev = mori::core::AtomicAddRelaxed(signal_flags, uint32_t{1});
    if (prev + 1 == gridDim.x) { // all blocks are done => set global flag
      mori::core::AtomicStoreRelaxed(signal_flags, uint32_t{0}); // reset counter
      mori::core::AtomicStoreRelaxedSystem(remote_sig, uint32_t{1});
    }
  }
}

// --------------------------------------------------------------------------
// MoriWaitKernel – spins on a local signal location until the remote
// peer has written a non-zero value (via MoriPutKernel), then resets it.
// --------------------------------------------------------------------------
__global__ void MoriWaitKernel(uint32_t* signal) {
  while (mori::core::AtomicLoadRelaxedSystem(signal) == 0) {
    __builtin_amdgcn_s_sleep(1);
  }
  // Reset for the next round.
  mori::core::AtomicStoreRelaxedSystem(signal, uint32_t{0});
  // __threadfence_system(); ???
}

// --------------------------------------------------------------------------
// MoriBarrierKernel – P2P barrier across all PEs (intra-node).
//
//   Uses a monotonically increasing counter per PE in symmetric memory.
//   Each PE atomically adds 1 to every other PE's counter via P2P
//   (signalling arrival), then spins until the local counter reaches
//   round * (nPes - 1).
//
//   The host tracks 'round' (starting at 1, incremented each call).
//   Since kernel launches on the same stream are serialised, the
//   counter never races across rounds.
//
//   barrier_count – a single uint32_t in symmetric heap, one per PE.
//   round         – 1-based round number managed by the host.
// --------------------------------------------------------------------------
__global__ void MoriBarrierKernel(uint32_t* barrier_count, uint32_t round) {
  int myPe = mori::shmem::ShmemMyPe();
  int nPes = mori::shmem::ShmemNPes();

  // Phase 1: signal arrival to every other PE.
  for (int pe = 0; pe < nPes; ++pe) {
    if (pe == myPe) continue;
    uint64_t remote = mori::shmem::ShmemPtrP2p(
        reinterpret_cast<uint64_t>(barrier_count), myPe, pe);
    mori::core::AtomicAddRelaxedSystem(
        reinterpret_cast<uint32_t*>(remote), uint32_t{1});
  }
  __threadfence_system();

  // Phase 2: wait until all other PEs have signalled this round.
  uint32_t expected = round * static_cast<uint32_t>(nPes - 1);
  while (mori::core::AtomicLoadRelaxedSystem(barrier_count) < expected) {
    __builtin_amdgcn_s_sleep(1);
  }
}

}  // anonymous namespace

// --------------------------------------------------------------------------
// Host-side helpers
// --------------------------------------------------------------------------

void InitSignalMemory(void* ptr, size_t bytes) {
  hipMemset(ptr, 0, bytes);
}

int Send(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
         uint32_t* signal_flags, std::intptr_t stream_handle) {

  size_t total = kBytesPerWarp * kWarpsPerBlock;

  // 4K - handled by 1 block
  // 8K - by 2 blocks
  // anything until 6K - also handled by 1 block
  
  int numBlocks = static_cast<int>((bytes + total / 2) / total);
  numBlocks = std::max(1, std::min(numBlocks, kMaxBlocks));

  fprintf(stderr, "MORI send Using blocks %d\n", numBlocks);

  auto stream = reinterpret_cast< hipStream_t >(stream_handle);
  MoriPutKernel<<<numBlocks, kBlockSize, 0, stream>>>(
      recv_buffer, send_buffer, bytes, peer, signal_flags);
  return 0;
}

// --------------------------------------------------------------------------
// Host-side Recv (put model – wait only).
//   Does NOT copy any data.  Launches a single-thread kernel that spins on
//   the local signal_flags[peer] until the remote peer's Send has set it.
// --------------------------------------------------------------------------
int Recv(void* /*recv_buffer*/, void* /*send_buffer*/, size_t /*bytes*/,
         int peer, uint32_t* signal_flags, std::intptr_t stream_handle) {

  auto stream = reinterpret_cast< hipStream_t >(stream_handle);
  MoriWaitKernel<<<1, 1, 0, stream>>>(&signal_flags[peer + 1]);
  return 0;
}


// --------------------------------------------------------------------------
// Host-side BarrierOnStream – enqueues a P2P barrier on the given stream.
//   barrier_count : single uint32_t in MORI symmetric heap.
//   round         : 1-based, incremented by the caller each invocation.
// --------------------------------------------------------------------------
void BarrierOnStream(uint32_t* barrier_count, uint32_t round,
                     std::intptr_t stream_handle) {
  auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  MoriBarrierKernel<<<1, 1, 0, stream>>>(barrier_count, round);
}

void synchronize_all() {
  hipDeviceSynchronize();
}

}  // namespace roc_mori
