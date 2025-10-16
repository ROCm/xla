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
__global__ void xla_nan_check(T* buffer, uint64_t buffer_length, uint32_t* nan_signal) {
  const uint64_t block_dim_x = static_cast<uint64_t>(blockDim.x),
                 stride = block_dim_x * gridDim.x;

  constexpr bool verbose = false;
  if (verbose && threadIdx.x + blockIdx.x * block_dim_x == 0) {
    printf("Running nan check on %p\n", buffer);
  }

  bool found_nan = false;
  // TODO(rocm): vectorize
  for (uint64_t idx = threadIdx.x + blockIdx.x * block_dim_x;
       idx < buffer_length; idx += stride) {
    found_nan |= Eigen::numext::isnan(buffer[idx]);
  }

  if (TF_PREDICT_TRUE(__all(!found_nan))) {
    return;
  }

  atomicExch(nan_signal, 1);
}

template <typename NativeT>
void RegisterNanCheckKernelParametrized(Platform::Id platform_id) {
  constexpr xla::PrimitiveType p_type =
      xla::primitive_util::NativeToPrimitiveType<NativeT>();
  CHECK(xla::primitive_util::IsFloatingPointType(p_type));

  auto kernel_symbol = absl::bit_cast<void*>(&xla_nan_check<NativeT>);

  std::string kernel_name = absl::StrCat(
      xla::primitive_util::LowercasePrimitiveTypeName(p_type), "_nan_check");

  stream_executor::MultiKernelLoaderSpec spec(3);
  spec.AddInProcessSymbol(kernel_symbol, kernel_name);

  absl::Status result =
      stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .RegisterKernel<NanCheckKernel<NativeT>>(platform_id, spec);

  if (!result.ok()) {
    LOG(FATAL) << "Failed to register nan check kernel for type "
               << xla::primitive_util::LowercasePrimitiveTypeName(p_type)
               << ": " << result;
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_NAN_CHECK_KERNEL_LIB_CU_H_
