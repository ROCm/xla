/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"

namespace xla::gpu {

xla::Future<std::unique_ptr<Thunk>> CubinCustomKernelCompiler::Compile(
    Thunk::ThunkInfo thunk_info, LlvmKernelSource kernel_source,
    const std::string& sanitized_kernel_name,
    const emitters::KernelArguments& kernel_arguments,
    const LaunchDimensions& launch_dimensions) {
  if (!thread_pool_) {
    return CompileImpl(std::move(thunk_info), std::move(kernel_source),
                       sanitized_kernel_name, kernel_arguments,
                       launch_dimensions);
  }
  return tsl::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [this, thunk_info = std::move(thunk_info),
       kernel_source = std::move(kernel_source), sanitized_kernel_name,
       kernel_arguments, launch_dimensions]() mutable {
        return CompileImpl(std::move(thunk_info), std::move(kernel_source),
                           sanitized_kernel_name, kernel_arguments,
                           launch_dimensions);
      });
}

xla::Future<LlvmKernelSource> CubinCustomKernelCompiler::CompileMlirToLlvm(
    const se::DeviceDescription& device, const HloModule& hlo_module,
    const std::string& entry_function_name, int unroll_factor,
    MlirKernelSource source, BorrowedMlirContext borrowed_context) {
  if (!thread_pool_) {
    return gpu::CompileMlirToLlvm(device, hlo_module, entry_function_name,
                                  unroll_factor, **borrowed_context,
                                  std::move(source));
  }
  return xla::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [source = std::move(source), device, &hlo_module, entry_function_name,
       unroll_factor,
       borrowed_context = std::move(borrowed_context)]() mutable {
        return gpu::CompileMlirToLlvm(device, hlo_module, entry_function_name,
                                      unroll_factor, **borrowed_context,
                                      std::move(source));
      });
}

xla::Future<std::vector<uint8_t>>
CubinCustomKernelCompiler::CompileToTargetBinary(
    LlvmKernelSource kernel_source) {
  if (!thread_pool_) {
    return CompileToCubinImpl(std::move(kernel_source));
  }
  return xla::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [this, kernel_source = std::move(kernel_source)]() mutable {
        return CompileToCubinImpl(std::move(kernel_source));
      });
}

absl::StatusOr<std::vector<uint8_t>>
CubinCustomKernelCompiler::CompileToCubinImpl(LlvmKernelSource kernel_source) {
  // When compilation is deferred, we do not compile the kernel here. Instead we
  // move its module into the deferred list (to be merged and compiled together
  // by the caller) and return an empty CUBIN, signalling that a KernelThunk
  // should be created for the not-yet-compiled kernel.
  //
  // Once the deferred modules have been consumed (deferral phase ended), we
  // fall through and compile immediately; this is how the merged constants
  // module is compiled.
  {
    absl::MutexLock lock(deferred_modules_mutex_);
    if (defer_compilation_ && !deferral_consumed_) {
      deferred_modules_.push_back(
          std::move(kernel_source).thread_safe_module());
      return std::vector<uint8_t>{};
    }
  }

  llvm::orc::ThreadSafeModule thread_safe_module =
      std::move(kernel_source).thread_safe_module();
  llvm::Module* llvm_module = thread_safe_module.getModuleUnlocked();

  if (pre_optimization_hook()) {
    pre_optimization_hook()(*llvm_module);
  }

  ASSIGN_OR_RETURN(std::vector<uint8_t> cubin,
                   compiler_(*llvm_module, device_info_, debug_options_));
  return cubin;
}

absl::StatusOr<std::unique_ptr<Thunk>> CubinCustomKernelCompiler::CompileImpl(
    Thunk::ThunkInfo thunk_info, LlvmKernelSource kernel_source,
    const std::string& sanitized_kernel_name,
    const emitters::KernelArguments& kernel_arguments,
    const LaunchDimensions& launch_dimensions) {
  ASSIGN_OR_RETURN(std::vector<uint8_t> cubin,
                   CompileToCubinImpl(std::move(kernel_source)));

  return CreateThunkForCubin(std::move(thunk_info), sanitized_kernel_name,
                             std::move(cubin), kernel_arguments,
                             launch_dimensions);
}

absl::StatusOr<std::unique_ptr<Thunk>>
CubinCustomKernelCompiler::CreateThunkForCubin(
    Thunk::ThunkInfo thunk_info, std::string kernel_name,
    std::vector<uint8_t> cubin,
    const emitters::KernelArguments& kernel_arguments,
    const LaunchDimensions& launch_dimensions, int64_t shmem_bytes,
    bool use_pdl) {
  // Compilation was deferred: the kernel is not available as CUBIN and will be
  // loaded by name from the executable at runtime.
  if (cubin.empty()) {
    return std::make_unique<KernelThunk>(
        std::move(thunk_info), std::move(kernel_name), kernel_arguments,
        launch_dimensions, /*cluster_dim=*/std::nullopt, shmem_bytes,
        /*tma_metadata=*/stream_executor::gpu::TmaMetadata{},
        /*zeroed_output_buffer_indices=*/std::vector<int64_t>{}, use_pdl);
  }

  ASSIGN_OR_RETURN(
      CustomKernel custom_kernel,
      kernel::CreateOwnedCubinCustomKernel(
          std::move(kernel_name), std::move(cubin),
          kernel_arguments.args().size(), launch_dimensions.block_counts(),
          launch_dimensions.thread_counts_per_block(), shmem_bytes));

  return std::make_unique<CustomKernelThunk>(
      std::move(thunk_info), std::move(custom_kernel), kernel_arguments,
      use_pdl);
}

xla::Future<TritonWrapperResult> CubinCustomKernelCompiler::CompileTritonToLlvm(
    const absl::string_view kernel_name, const HloModule& hlo_module,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    const llvm::Triple& target_triple, const std::string& data_layout,
    TritonKernelSource triton_source, BorrowedMlirContext borrowed_context,
    bool is_xla_fusion) {
  if (!thread_pool_) {
    return gpu::CompileTritonToLLVM(kernel_name, hlo_module, device_info,
                                    block_level_parameters, target_triple,
                                    data_layout, std::move(triton_source),
                                    **borrowed_context, is_xla_fusion);
  }
  return xla::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [kernel_name = std::string(kernel_name), hlo_module = &hlo_module,
       device_info, block_level_parameters, target_triple, is_xla_fusion,
       data_layout, borrowed_context = std::move(borrowed_context),
       triton_source = std::move(triton_source)]() mutable {
        return gpu::CompileTritonToLLVM(kernel_name, *hlo_module, device_info,
                                        block_level_parameters, target_triple,
                                        data_layout, std::move(triton_source),
                                        **borrowed_context, is_xla_fusion);
      });
}

std::vector<llvm::orc::ThreadSafeModule>
CubinCustomKernelCompiler::ConsumeDeferredModules() {
  absl::MutexLock lock(deferred_modules_mutex_);
  deferral_consumed_ = true;
  return std::move(deferred_modules_);
}

}  // namespace xla::gpu
