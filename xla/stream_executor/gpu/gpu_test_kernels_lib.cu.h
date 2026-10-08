/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_TEST_KERNELS_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_TEST_KERNELS_LIB_CU_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "xla/stream_executor/gpu/gpu_test_kernel_traits.h"

namespace stream_executor::gpu {
// When loading these kernels from a fatbin we find these kernels by symbol
// name, therefore we switch off name mangling
extern "C" {

__global__ void AddI32(int32_t* a, int32_t* b, int32_t* c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

__global__ void IncI32(int32_t a, int32_t* b, int32_t* c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a + b[index];
}

__global__ void MulI32(int32_t* a, int32_t* b, int32_t* c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] * b[index];
}

__global__ void IncAndCmp(int32_t* counter, bool* pred, int32_t* value) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  pred[index] = counter[index] < *value;
  counter[index] += 1;
}

__global__ void AddI32Ptrs3(Ptrs3<int32_t> ptrs) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  ptrs.c[index] = ptrs.a[index] + ptrs.b[index];
}

__global__ void CopyKernel(std::byte* dst, std::array<std::byte, 16> byval) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < byval.size(); i++) {
      dst[i] = byval[i];
    }
  }
}

// Copies each column of `buf` into dynamic shared memory, then writes it back
// with both axes reversed. The caller must launch with
// dynamic shared memory of at least n_cols * n_rows bytes.
__global__ void DynShmemKernel(uint8_t* buf, uint32_t* n_cols,
                               uint32_t* n_rows) {
  extern __shared__ uint8_t shmem[];
  uint32_t src_col = threadIdx.x;
  uint32_t dst_col = *n_cols - src_col - 1;
  for (uint32_t i = 0; i < *n_rows; i++) {
    shmem[src_col + *n_cols * i] = buf[src_col + *n_cols * i];
  }
  __syncthreads();
  for (uint32_t i = 0; i < *n_rows; i++) {
    buf[dst_col + *n_cols * (*n_rows - i - 1)] = shmem[src_col + *n_cols * i];
  }
}

// Uses 16 KiB of static shared memory (indexed by data, so it cannot be
// optimized away) and blockDim.x bytes of dynamic shared memory, one byte per
// thread: buf[i] becomes buf[i] + buf[blockDim.x - 1 - i]. The caller must
// launch with at least blockDim.x bytes of dynamic shared memory.
__global__ void StaticAndDynShmemKernel(uint8_t* buf) {
  __shared__ uint8_t static_shmem[16 * 1024];
  extern __shared__ uint8_t dyn_shmem[];
  uint32_t i = threadIdx.x;
  uint8_t value = buf[i];
  // In bounds only because `value` is a uint8_t: slot <= 255 * 64 + 63.
  uint32_t slot = value * 64 + i % 64;
  static_shmem[slot] = value;
  dyn_shmem[i] = value;
  __syncthreads();
  buf[i] = static_shmem[slot] + dyn_shmem[blockDim.x - 1 - i];
}
}
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_TEST_KERNELS_LIB_CU_H_
