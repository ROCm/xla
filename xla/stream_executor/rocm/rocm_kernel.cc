/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_kernel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

namespace {

absl::Status FuncGetAttribute(hipFunction_attribute attribute,
                              hipFunction_t func, int* attribute_value) {
  return ToStatus(
      hipFuncGetAttribute(attribute_value, attribute, func),
      absl::StrCat("Failed to query kernel attribute: ", attribute));
}

}  // namespace

absl::StatusOr<uint64_t> GetDynamicSharedMemoryLimit(hipFunction_t function) {
  int limit = 0;
  ABSL_RETURN_IF_ERROR(FuncGetAttribute(
      HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function, &limit));
  if (limit < 0) {
    return absl::InternalError(absl::StrCat(
        "Negative dynamic shared memory limit reported by HIP: ", limit));
  }
  return static_cast<uint64_t>(limit);
}

absl::Status CheckDynamicSharedMemoryBytes(
    absl::string_view kernel_name, uint64_t bytes,
    const absl::StatusOr<uint64_t>& limit) {
  if (bytes == 0) {
    return absl::OkStatus();
  }
  if (!limit.ok()) {
    return absl::Status(
        limit.status().code(),
        absl::StrCat(
            "Cannot check the dynamic shared memory request of kernel ",
            kernel_name, ": ", limit.status().message()));
  }
  if (bytes <= *limit) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Kernel ", kernel_name, " requests ", bytes,
                   " bytes of dynamic shared memory, but at most ", *limit,
                   " are available"));
}

absl::StatusOr<int32_t> RocmKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim threads, size_t dynamic_shared_memory_bytes) const {
  int32_t threads_per_block = threads.x * threads.y * threads.z;
  VLOG(0) << "Get kernel block occupancy: " << name()
          << "; threads_per_block: " << threads_per_block
          << "; dynamic_shared_memory_bytes: " << dynamic_shared_memory_bytes;

  std::unique_ptr<ActivateContext> activation = executor_->Activate();

  int max_blocks = 0;
  ABSL_RETURN_IF_ERROR(
      ToStatus(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                   &max_blocks, rocm_function_, threads_per_block,
                   dynamic_shared_memory_bytes),
               "Failed to calculate maximal active blocks per SM"));
  return max_blocks;
}

absl::StatusOr<KernelMetadata> RocmKernel::GetKernelMetadata() {
  KernelMetadata kernel_metadata;
  int value = 0;
  ABSL_RETURN_IF_ERROR(
      FuncGetAttribute(HIP_FUNC_ATTRIBUTE_NUM_REGS, rocm_function_, &value));
  kernel_metadata.set_registers_per_thread(value);

  ABSL_RETURN_IF_ERROR(FuncGetAttribute(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                        rocm_function_, &value));
  kernel_metadata.set_shared_memory_bytes(value);
  return kernel_metadata;
}

void RocmKernel::LoadDynamicSharedMemoryLimit() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  dynamic_shared_memory_limit_bytes_ =
      GetDynamicSharedMemoryLimit(rocm_function_);
}

absl::Status RocmKernel::CheckDynamicSharedMemoryBytes(uint64_t bytes) const {
  return gpu::CheckDynamicSharedMemoryBytes(name(), bytes,
                                            dynamic_shared_memory_limit_bytes_);
}

absl::Status RocmKernel::Launch(const ThreadDim& thread_dims,
                                const BlockDim& block_dims,
                                const std::optional<ClusterDim>& cluster_dims,
                                Stream* stream, const KernelArgs& args) {
  hipFunction_t function = gpu_function();

  // Launch kernels with packed arguments.
  auto launch = [this, &cluster_dims, &thread_dims, &block_dims, &function,
                 stream](const KernelArgsPackedArrayBase& packed) {
    int32_t expected_number_of_arguments =
        Arity() + (packed.number_of_shared_bytes() > 0);

    CHECK_EQ(expected_number_of_arguments, packed.number_of_arguments())
        << "Kernel " << name() << " has " << packed.number_of_arguments()
        << " arguments, but expected " << expected_number_of_arguments
        << "; arity=" << Arity()
        << "; number_of_shared_bytes=" << packed.number_of_shared_bytes();

    void** params = const_cast<void**>(packed.argument_addresses().data());

    return stream->LaunchKernel(thread_dims, block_dims, cluster_dims, function,
                                name(), params, packed.number_of_shared_bytes(),
                                /*use_pdl=*/false);
  };

  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    auto& pack = args_packing();
    if (!pack) {
      return launch(*packed);
    }
    ABSL_ASSIGN_OR_RETURN(auto repacked, pack(*this, *packed));

    return launch(*repacked);
  }

  // For device memory array we rely on a custom kernel arguments packing.
  if (auto* device_mem = DynCast<KernelArgsDeviceAddressArray>(&args)) {
    auto& pack = args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    ABSL_ASSIGN_OR_RETURN(auto packed, pack(*this, *device_mem));
    return launch(*packed);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

}  // namespace gpu
}  // namespace stream_executor
