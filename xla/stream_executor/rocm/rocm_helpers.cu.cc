/* Copyright 2022 The OpenXLA Authors.

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

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#include <cmath>
#include <cstdint>

#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"

namespace stream_executor {
namespace gpu {

__global__ void rocm_Broadcast_fp32Kernel(float* dst, int dst_stride,
                                          int batches, float* src, int size) {
  dst += blockIdx.y * 4 * dst_stride + blockIdx.z * dst_stride * batches;
  src += blockIdx.z * size;
  float* dst2 = dst + dst_stride;
  float* dst3 = dst + dst_stride * 2;
  float* dst4 = dst + dst_stride * 3;
  bool b2 = (blockIdx.y * 4 + 1 < batches);
  bool b3 = (blockIdx.y * 4 + 2 < batches);
  bool b4 = (blockIdx.y * 4 + 3 < batches);
  for (int i = threadIdx.x + blockIdx.x * 256; i < size;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
    if (b2) {
      dst2[i] = src[i];
    }
    if (b3) {
      dst3[i] = src[i];
    }
    if (b4) {
      dst4[i] = src[i];
    }
  }
}

void rocm_Broadcast_fp32(void* stream, float* dst, int dst_stride, int batches,
                         int src_batches, float* src, int size) {
  int x_blocks = (size + 255) / 256;
  hipLaunchKernelGGL(rocm_Broadcast_fp32Kernel,
                     dim3(x_blocks, (batches + 3) / 4, src_batches),
                     min(256, (int)size), 0, (hipStream_t)stream, dst,
                     dst_stride, batches, src, size);
}

__device__ float sigmoid(float x) {
  if (x > 0)
    return 1. / (1. + __expf(-x));
  else
    return __expf(x) / (__expf(x) + 1.);
}

template <typename T, typename Tbias, int act_mode>
__global__ void launchInplaceBiasActivation_kernel(
    T* c_data, const Tbias* bias_data, const T* side_input_data,
    float side_input_scale, uint64_t m, uint64_t n, int64_t ldc, float param,
    int transpose) {
  uint64_t x = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t y = threadIdx.y + blockIdx.y * blockDim.y;
  uint64_t z = blockIdx.z;
  if (x >= n || y >= m) return;
  float v;
  uint64_t addr = x + y * ldc + z * m * n;
  if (!transpose)
    v = static_cast<float>(c_data[addr]) + static_cast<float>(bias_data[x]);
  else
    v = static_cast<float>(c_data[addr]) + static_cast<float>(bias_data[y]);
  if (side_input_data != 0)
    v += static_cast<float>(side_input_data[addr]) * side_input_scale;
  if (act_mode == 1)
    v = sigmoid(v);
  else if (act_mode == 2)
    v = v > 0.0f ? v : 0.0f;
  else if (act_mode == 3)
    v = v > 0.0f ? (v > 6.0f ? 6.0f : v) : 0.0f;
  else if (act_mode == 4)
    v = v > 0.0f ? (v > param ? param : v) : 0.0f;
  else if (act_mode == 5)
    v = tanh(v);
  else if (act_mode == 6)
    v = v > -param ? (v > param ? param : v) : -param;
  else if (act_mode == 7)
    v = v > 0.0f ? v : __expf(v) - 1;
  else if (act_mode == 8)
    v = v > 0.0f ? v : param * v;
  else if (act_mode == 9)
    v = 0.5 * v * (1 + erf(v / sqrt(2.0f)));
  c_data[addr] = (T)v;
}

template <typename T, typename Tbias>
void launchInplaceBiasActivation(hipStream_t stream, void* c_data,
                                 const void* bias_data,
                                 const void* side_input_data,
                                 float side_input_scale, int activation_mode,
                                 uint64_t batch, uint64_t m, uint64_t n,
                                 int64_t ldc, float param) {
  uint64_t bx = min(n, static_cast<uint64_t>(256));
  uint64_t by = min(m, static_cast<uint64_t>(256) / bx);
  uint64_t gx = (n + bx - 1) / bx;
  uint64_t gy = (m + by - 1) / by;
  int transpose = (activation_mode >= 10);
  activation_mode %= 10;
  auto kernel = launchInplaceBiasActivation_kernel<T, Tbias, 0>;
  if (activation_mode == 1)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 1>;
  else if (activation_mode == 2)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 2>;
  else if (activation_mode == 3)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 3>;
  else if (activation_mode == 4)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 4>;
  else if (activation_mode == 5)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 5>;
  else if (activation_mode == 6)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 6>;
  else if (activation_mode == 7)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 7>;
  else if (activation_mode == 8)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 8>;
  else if (activation_mode == 9)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 9>;

  hipLaunchKernelGGL(kernel, dim3(gx, gy, batch), dim3(bx, by, 1), 0, stream,
                     static_cast<T*>(c_data),
                     static_cast<const Tbias*>(bias_data),
                     static_cast<const T*>(side_input_data), side_input_scale,
                     m, n, ldc, param, transpose);
}

#define INSTANTIATE_BIAS_ACTIVATION(X, Y)                          \
  template void launchInplaceBiasActivation<X, Y>(                 \
      hipStream_t stream, void* c_data, const void* bias_data,     \
      const void* side_input_data, float side_input_scale,         \
      int activation_mode, uint64_t batch, uint64_t m, uint64_t n, \
      int64_t ldc, float param);

INSTANTIATE_BIAS_ACTIVATION(__half, __half)
INSTANTIATE_BIAS_ACTIVATION(__half, float)
INSTANTIATE_BIAS_ACTIVATION(hip_bfloat16, hip_bfloat16)
INSTANTIATE_BIAS_ACTIVATION(hip_bfloat16, float)
INSTANTIATE_BIAS_ACTIVATION(float, float)
INSTANTIATE_BIAS_ACTIVATION(double, double)

};  // namespace gpu

namespace rocm {

constexpr int BLOCK_SIZE = 256;

template <typename T>
__global__ void SetUserArgsKernelRaggedInNonContractingDim(
    hipblaslt_ext::UserArguments* dest_args, void* a, void* b, void* c, void* d,
    void* e, const void* group_sizes, size_t byte_width_elem_a,
    size_t byte_width_elem_b, size_t byte_width_elem_c,
    size_t byte_width_elem_d, uint64_t stride_a, uint64_t stride_b,
    uint64_t output_stride_ragged_dim, bool must_swap_operands, uint32_t m,
    uint32_t n, uint32_t k, uint32_t batch, uint32_t strideA1,
    uint32_t strideA2, uint32_t strideB1, uint32_t strideB2, uint32_t strideC1,
    uint32_t strideC2, uint32_t strideD1, uint32_t strideD2,
    uint64_t num_gemms) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gemms) {
    return;
  }

  uint32_t offset_group = 0;
  const T* typed_group_sizes = static_cast<const T*>(group_sizes);
  if (blockIdx.x == 0) {
    // Shared memory for BlockScan
    __shared__ typename hipcub::BlockScan<uint32_t, BLOCK_SIZE>::TempStorage
        temp_storage;
    offset_group = typed_group_sizes[idx];
    hipcub::BlockScan<uint32_t, BLOCK_SIZE>(temp_storage)
        .ExclusiveSum(offset_group, offset_group);
  } else {
    for (uint32_t i = 0; i < idx; i++) {
      offset_group += typed_group_sizes[i];
    }
  }

  // Declare shared memory for UserArguments
  __shared__ hipblaslt_ext::UserArguments arg;

  if (must_swap_operands) {
    // The ragged matrix has been set as operand B.
    arg.n = typed_group_sizes[idx];
    arg.m = m;

    arg.a = static_cast<void*>(static_cast<uint8_t*>(a) +
                               (idx * stride_a * byte_width_elem_a));
    arg.b = static_cast<void*>(static_cast<uint8_t*>(b) +
                               (offset_group * stride_b * byte_width_elem_b));
  } else {
    arg.m = typed_group_sizes[idx];
    arg.n = n;

    arg.a = static_cast<void*>(static_cast<uint8_t*>(a) +
                               (offset_group * stride_a * byte_width_elem_a));
    arg.b = static_cast<void*>(static_cast<uint8_t*>(b) +
                               (idx * stride_b * byte_width_elem_b));
  }
  arg.c = static_cast<void*>(
      static_cast<uint8_t*>(c) +
      (offset_group * output_stride_ragged_dim * byte_width_elem_c));
  arg.d = static_cast<void*>(
      static_cast<uint8_t*>(d) +
      (offset_group * output_stride_ragged_dim * byte_width_elem_d));
  arg.k = k;
  arg.batch = batch;
  arg.strideA1 = strideA1;
  arg.strideA2 = strideA2;
  arg.strideB1 = strideB1;
  arg.strideB2 = strideB2;
  arg.strideC1 = strideC1;
  arg.strideC2 = strideC2;
  arg.strideD1 = strideD1;
  arg.strideD2 = strideD2;
  arg.strideE1 = 0;
  arg.strideE2 = 0;
  // Set alpha to float(1) and beta to float(0).
  // As these values are imposed in the gemm_rewritter pass anyway.
  for (int8_t i = 0; i < 16; i++) {
    arg.alpha[i] = 0;
    arg.beta[i] = 0;
  }
  arg.alpha[2] = -128;
  arg.alpha[3] = 63;
  arg.scaleA = nullptr;
  arg.scaleB = nullptr;
  arg.scaleC = nullptr;
  arg.scaleD = nullptr;
  arg.scaleAlphaVec = nullptr;
  arg.bias = nullptr;
  arg.biasType = 0;
  arg.e = nullptr;
  arg.act0 = 0.0;
  arg.act1 = 0.0;
  arg.activationType = 0;

  // Copy from shared memory to global memory
  // dest_args[idx] = sharedUserArgs[threadIdx.x];
  dest_args[idx] = arg;
  __threadfence();
}

template <typename T>
__global__ void SetUserArgsKernelRaggedInContractingDim(
    hipblaslt_ext::UserArguments* dest_args, void* a, void* b, void* c, void* d,
    void* e, const void* group_sizes, size_t byte_width_elem_a,
    size_t byte_width_elem_b, size_t byte_width_elem_c,
    size_t byte_width_elem_d, uint64_t stride_a, uint64_t stride_b,
    uint64_t output_stride_ragged_dim, bool must_swap_operands, uint32_t m,
    uint32_t n, uint32_t k, uint32_t batch, uint32_t strideA1,
    uint32_t strideA2, uint32_t strideB1, uint32_t strideB2, uint32_t strideC1,
    uint32_t strideC2, uint32_t strideD1, uint32_t strideD2,
    uint64_t num_gemms) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gemms) {
    return;
  }

  uint32_t offset_group = 0;
  const T* typed_group_sizes = static_cast<const T*>(group_sizes);
  if (blockIdx.x == 0) {
    // Shared memory for BlockScan
    __shared__ typename hipcub::BlockScan<uint32_t, BLOCK_SIZE>::TempStorage
        temp_storage;
    offset_group = typed_group_sizes[idx];
    hipcub::BlockScan<uint32_t, BLOCK_SIZE>(temp_storage)
        .ExclusiveSum(offset_group, offset_group);
  } else {
    for (uint32_t i = 0; i < idx; i++) {
      offset_group += typed_group_sizes[i];
    }
  }

  // Declare shared memory for UserArguments
  __shared__ hipblaslt_ext::UserArguments arg;

  arg.m = m;
  arg.n = n;
  arg.a = static_cast<void*>(static_cast<uint8_t*>(a) +
                             (offset_group * stride_a * byte_width_elem_a));
  arg.b = static_cast<void*>(static_cast<uint8_t*>(b) +
                             (offset_group * stride_b * byte_width_elem_b));
  arg.c = static_cast<void*>(static_cast<uint8_t*>(c) +
                             (idx * batch * strideC2 * byte_width_elem_c));
  arg.d = static_cast<void*>(static_cast<uint8_t*>(d) +
                             (idx * batch * strideD2 * byte_width_elem_d));
  arg.k = typed_group_sizes[idx];
  arg.batch = batch;
  arg.strideA1 = strideA1;
  arg.strideA2 = strideA2;
  arg.strideB1 = strideB1;
  arg.strideB2 = strideB2;
  arg.strideC1 = strideC1;
  arg.strideC2 = strideC2;
  arg.strideD1 = strideD1;
  arg.strideD2 = strideD2;
  arg.strideE1 = 0;
  arg.strideE2 = 0;
  // Set alpha to float(1) and beta to float(0).
  // As these values are imposed in the gemm_rewritter pass anyway.
  for (int8_t i = 0; i < 16; i++) {
    arg.alpha[i] = 0;
    arg.beta[i] = 0;
  }
  arg.alpha[2] = -128;
  arg.alpha[3] = 63;
  arg.scaleA = nullptr;
  arg.scaleB = nullptr;
  arg.scaleC = nullptr;
  arg.scaleD = nullptr;
  arg.scaleAlphaVec = nullptr;
  arg.bias = nullptr;
  arg.biasType = 0;
  arg.e = nullptr;
  arg.act0 = 0.0;
  arg.act1 = 0.0;
  arg.activationType = 0;

  // Copy from shared memory to global memory
  dest_args[idx] = arg;
  __threadfence();
}

template <typename T>
__global__ void SetUserArgsKernelRaggedInBatchDim(
    hipblaslt_ext::UserArguments* dest_args, void* a, void* b, void* c, void* d,
    void* e, const void* group_sizes, size_t byte_width_elem_a,
    size_t byte_width_elem_b, size_t byte_width_elem_c,
    size_t byte_width_elem_d, uint64_t stride_a, uint64_t stride_b,
    uint64_t output_stride_ragged_dim, bool must_swap_operands, uint32_t m,
    uint32_t n, uint32_t k, uint32_t batch, uint32_t strideA1,
    uint32_t strideA2, uint32_t strideB1, uint32_t strideB2, uint32_t strideC1,
    uint32_t strideC2, uint32_t strideD1, uint32_t strideD2,
    uint64_t num_gemms) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gemms) {
    return;
  }

  uint32_t offset_group = 0;
  const T* typed_group_sizes = static_cast<const T*>(group_sizes);
  if (blockIdx.x == 0) {
    // Shared memory for BlockScan
    __shared__ typename hipcub::BlockScan<uint32_t, BLOCK_SIZE>::TempStorage
        temp_storage;
    offset_group = typed_group_sizes[idx];
    hipcub::BlockScan<uint32_t, BLOCK_SIZE>(temp_storage)
        .ExclusiveSum(offset_group, offset_group);
  } else {
    for (uint32_t i = 0; i < idx; i++) {
      offset_group += typed_group_sizes[i];
    }
  }

  // Declare shared memory for UserArguments
  __shared__ hipblaslt_ext::UserArguments arg;

  arg.m = m;
  arg.n = n;
  arg.a = static_cast<void*>(static_cast<uint8_t*>(a) +
                             (offset_group * stride_a * byte_width_elem_a));
  arg.b = static_cast<void*>(static_cast<uint8_t*>(b) +
                             (offset_group * stride_b * byte_width_elem_b));
  arg.c = static_cast<void*>(
      static_cast<uint8_t*>(c) +
      (offset_group * output_stride_ragged_dim * byte_width_elem_c));
  arg.d = static_cast<void*>(
      static_cast<uint8_t*>(d) +
      (offset_group * output_stride_ragged_dim * byte_width_elem_d));
  arg.k = k;
  arg.batch = typed_group_sizes[idx];
  arg.strideA1 = strideA1;
  arg.strideA2 = strideA2;
  arg.strideB1 = strideB1;
  arg.strideB2 = strideB2;
  arg.strideC1 = strideC1;
  arg.strideC2 = strideC2;
  arg.strideD1 = strideD1;
  arg.strideD2 = strideD2;
  arg.strideE1 = 0;
  arg.strideE2 = 0;
  // Set alpha to float(1) and beta to float(0).
  // As these values are imposed in the gemm_rewritter pass anyway.
  for (int8_t i = 0; i < 16; i++) {
    arg.alpha[i] = 0;
    arg.beta[i] = 0;
  }
  arg.alpha[2] = -128;
  arg.alpha[3] = 63;
  arg.scaleA = nullptr;
  arg.scaleB = nullptr;
  arg.scaleC = nullptr;
  arg.scaleD = nullptr;
  arg.scaleAlphaVec = nullptr;
  arg.bias = nullptr;
  arg.biasType = 0;
  arg.e = nullptr;
  arg.act0 = 0.0;
  arg.act1 = 0.0;
  arg.activationType = 0;

  // Copy from shared memory to global memory
  dest_args[idx] = arg;
  __threadfence();
}

void GroupGemmUpdateArgs(
    hipStream_t stream, hipblaslt_ext::UserArguments* args, void* a, void* b,
    void* c, void* d, void* e, const void* group_sizes,
    size_t group_size_bytewidth, size_t byte_width_elem_a,
    size_t byte_width_elem_b, size_t byte_width_elem_c,
    size_t byte_width_elem_d, uint64_t stride_ragged_dim,
    uint64_t stride_group_dim, uint64_t output_stride_ragged_dim,
    bool must_swap_operands, uint32_t m, uint32_t n, uint32_t k, uint32_t batch,
    uint32_t strideA1, uint32_t strideA2, uint32_t strideB1, uint32_t strideB2,
    uint32_t strideC1, uint32_t strideC2, uint32_t strideD1, uint32_t strideD2,
    const uint8_t ragged_mode, uint64_t num_gemms) {
  const uint64_t block_sz = BLOCK_SIZE;
  const uint64_t n_blocks = (num_gemms + block_sz - 1) / block_sz;
  auto kernel = SetUserArgsKernelRaggedInNonContractingDim<uint64_t>;
  switch (ragged_mode) {
    case 0: {  // RaggedInNonContractingDim
      if (group_size_bytewidth == 4) {
        kernel = SetUserArgsKernelRaggedInNonContractingDim<uint32_t>;
      }
      break;
    }
    case 1: {  // RaggedInContractingDim
      kernel = SetUserArgsKernelRaggedInContractingDim<uint64_t>;
      if (group_size_bytewidth == 4) {
        kernel = SetUserArgsKernelRaggedInContractingDim<uint32_t>;
      }
      break;
    }
    case 2: {  // RaggedInBatchDim
      kernel = SetUserArgsKernelRaggedInBatchDim<uint64_t>;
      if (group_size_bytewidth == 4) {
        kernel = SetUserArgsKernelRaggedInBatchDim<uint32_t>;
      }
      break;
    }
  }
  auto stride_a = stride_ragged_dim;
  auto stride_b = stride_group_dim;
  if (must_swap_operands) {
    std::swap(stride_a, stride_b);
  }
  hipLaunchKernelGGL(kernel, n_blocks, std::min(block_sz, num_gemms), 0, stream,
                     args, a, b, c, d, e, group_sizes, byte_width_elem_a,
                     byte_width_elem_b, byte_width_elem_c, byte_width_elem_d,
                     stride_a, stride_b, output_stride_ragged_dim,
                     must_swap_operands, m, n, k, batch, strideA1, strideA2,
                     strideB1, strideB2, strideC1, strideC2, strideD1, strideD2,
                     num_gemms);
}
};  // namespace rocm

};  // namespace stream_executor
