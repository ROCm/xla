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

#ifndef XLA_STREAM_EXECUTOR_GPU_NAN_CHECK_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_NAN_CHECK_KERNEL_LIB_CU_H_

#include <sys/types.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "xla/primitive_util.h"
#include "xla/stream_executor/gpu/nan_check_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/types.h"

namespace stream_executor::gpu {

template <typename T>
__global__ void xla_nan_check(T* buffer, 
      uint64_t buffer_length, float threshold, bool verbose,
      uint32_t* nan_signal) {
  const uint64_t block_dim_x = static_cast<uint64_t>(blockDim.x),
                 stride = block_dim_x * gridDim.x;

  __shared__ uint32_t prev_signal_val;
  
  // Constants from xla/backends/gpu/runtime/nan_check.h
  uint32_t found_flag = 0;
  T last_val;
  uint64_t idx = 0;
  for (idx = threadIdx.x + blockIdx.x * block_dim_x;
       idx < buffer_length && found_flag == 0; idx += stride) {
    last_val = buffer[idx];
    if (Eigen::numext::isnan(last_val)) {
      found_flag = 1;
    } else if (!Eigen::numext::isfinite(last_val)) {
       // found_flag = 2; // do not detect Infs by now
    } else if (Eigen::numext::isfinite(threshold) && 
          Eigen::numext::abs(static_cast< float >(last_val)) > threshold) {
      found_flag = 3;
    }
  }
  if (TF_PREDICT_TRUE(__all(found_flag == 0))) {
    return;
  }

  auto print = [idx, last_val](auto SZ) {
    union {
      T fp;
      decltype(SZ) dec;
    } S = { last_val };
    printf("%lld: b:%d th:%d NaN/Inf/Large value: %f (0x%X)\n",
          idx, blockIdx.x, threadIdx.x, (float)last_val, S.dec);
  };

  uint32_t prev = found_flag != 0 ? atomicExch(nan_signal, found_flag) : 1;
  // we are the first wavefront that set the nan_signal flag
  if (verbose && __any(prev == 0) && found_flag != 0) { 
    if constexpr(sizeof(T) == 1) print(uint8_t{});
    else if constexpr(sizeof(T) == 2) print(uint16_t{});
    else if constexpr(sizeof(T) == 4) print(uint32_t{});
    else print(uint64_t{});
  }
}

template <typename NativeT>
void RegisterNanCheckKernelParametrized(Platform::Id platform_id) {
  constexpr xla::PrimitiveType p_type =
      xla::primitive_util::NativeToPrimitiveType<NativeT>();
  CHECK(xla::primitive_util::IsFloatingPointType(p_type));

  auto kernel_symbol = absl::bit_cast<void*>(&xla_nan_check<NativeT>);

  std::string kernel_name = absl::StrCat(
      xla::primitive_util::LowercasePrimitiveTypeName(p_type), "_nan_check");

  using Kernel = NanCheckKernel<NativeT>;
  MultiKernelLoaderSpec spec(Kernel::KernelType::kNumberOfParameters);
  spec.AddInProcessSymbol(kernel_symbol, kernel_name);

  absl::Status result =
      GpuKernelRegistry::GetGlobalRegistry()
          .RegisterKernel<Kernel>(platform_id, spec);

  if (!result.ok()) {
    LOG(FATAL) << "Failed to register nan check kernel for type "
               << xla::primitive_util::LowercasePrimitiveTypeName(p_type)
               << ": " << result;
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_NAN_CHECK_KERNEL_LIB_CU_H_
