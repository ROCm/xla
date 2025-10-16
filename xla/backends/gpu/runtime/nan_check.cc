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

#include "xla/backends/gpu/runtime/nan_check.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/status/statusor.h"
#include "Eigen/Core"
#include "xla/primitive_util.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/gpu/nan_check_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

struct NanCheckParams {
  se::Stream* stream = nullptr;
  se::DeviceMemoryBase buffer{};
  PrimitiveType element_type = S1;
  se::DeviceMemory<uint8_t> msg{};
  se::DeviceMemory<uint32_t> abort_lock{};
};

template <typename ElementT>
static absl::Status LaunchNanCheckKernelTyped(const NanCheckParams& params) {
  se::StreamExecutor* executor = params.stream->parent();
  se::DeviceMemory<ElementT> buffer_typed(params.buffer);
  int64_t buffer_size = buffer_typed.ElementCount();

  TF_ASSIGN_OR_RETURN(
      auto nan_check_kernel,
      stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .LoadKernel<stream_executor::gpu::NanCheckKernel<ElementT>>(
              executor));

  const se::DeviceDescription& gpu_device_info =
      executor->GetDeviceDescription();

  LaunchDimensions dim = CalculateLaunchDimensions(
      Shape(params.element_type, {buffer_size}, {}), gpu_device_info);
  // Limit # of blocks to some meaningful number which is large enough to
  // occupy all GPU cores if necessary but not too large to reduce # of idle
  // blocks
  constexpr uint64_t kMaxNumThreadBlocksForKernel = 32768;
  dim = LaunchDimensions(
      se::BlockDim(std::min(dim.num_blocks(), kMaxNumThreadBlocksForKernel), 1, 1),
      dim.thread_counts_per_block());

  return nan_check_kernel.Launch(
      dim.thread_counts_per_block(), dim.block_counts(), params.stream,
      buffer_typed, static_cast<uint64_t>(buffer_size), params.msg,
      params.abort_lock);
}

absl::Status LaunchNanCheckKernel(se::Stream* stream,
                                  const se::DeviceMemoryBase& buffer,
                                  const PrimitiveType element_type,
                                  const se::DeviceMemory<uint8_t>& msg,
                                  se::DeviceMemory<uint32_t>& abort_lock) {
  NanCheckParams params{stream, buffer, element_type, msg, abort_lock};

  auto do_launch = [&](auto cst_type) {
    using ElementT = primitive_util::NativeTypeOf<cst_type>;
    return LaunchNanCheckKernelTyped<ElementT>(params);
  };

  CHECK(primitive_util::IsFloatingPointType(element_type));
  return xla::primitive_util::FloatingPointTypeSwitch<absl::Status>(
      do_launch, element_type);
}

}  // namespace gpu
}  // namespace xla
