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

#ifndef XLA_BACKENDS_GPU_CODEGEN_CUBIN_CUSTOM_KERNEL_COMPILER_H_
#define XLA_BACKENDS_GPU_CODEGEN_CUBIN_CUSTOM_KERNEL_COMPILER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// LlvmIrCompiler abstracts compilation of LLVM IR to target binary.
// Takes the LLVM module, device description, and debug options as input.
// Returns the compiled binary as a vector of bytes or an error status.
using LlvmIrCompiler = absl::AnyInvocable<absl::StatusOr<std::vector<uint8_t>>(
    llvm::Module& module, const stream_executor::DeviceDescription& descr,
    const DebugOptions& opts)>;

// Implementation of KernelCompiler that compiles LLVM IR to CUBIN format using
// a provided compilation function.
//
// Note: CubinCustomKernelCompiler utilizes provided threadpool.
// If threadpool is not provided, the compilation happens
// fully within this call, and the result is returned as an immediately ready
// Future.
//
// If `defer_compilation` is set to true, `Compile` does not compile the kernel
// source at all. Instead it moves the kernel's ThreadSafeModule out of the
// source and stores it in an internal list, returning a KernelThunk that refers
// to the kernel by name immediately (without using the thread pool). The stored
// modules can later be retrieved with `ConsumeDeferredModules` so that the
// caller can merge them into a single module and compile them together.
class CubinCustomKernelCompiler final : public KernelCompiler {
 public:
  CubinCustomKernelCompiler(LlvmIrCompiler compiler,
                            const se::DeviceDescription& gpu_device_info,
                            const DebugOptions& debug_options,
                            tsl::thread::ThreadPool* thread_pool = nullptr,
                            bool defer_compilation = false)
      : compiler_(std::move(compiler)),
        device_info_(gpu_device_info),
        debug_options_(debug_options),
        thread_pool_(thread_pool),
        defer_compilation_(defer_compilation) {}

  xla::Future<std::unique_ptr<Thunk>> Compile(
      Thunk::ThunkInfo thunk_info, LlvmKernelSource kernel_source,
      const std::string& sanitized_kernel_name,
      const emitters::KernelArguments& kernel_arguments,
      const LaunchDimensions& launch_dimensions) override;

  xla::Future<LlvmKernelSource> CompileMlirToLlvm(
      const se::DeviceDescription& device, const HloModule& hlo_module,
      const std::string& entry_function_name, int unroll_factor,
      MlirKernelSource source, BorrowedMlirContext borrowed_context) override;

  xla::Future<std::vector<uint8_t>> CompileToTargetBinary(
      LlvmKernelSource kernel_source) override;

  absl::StatusOr<std::unique_ptr<Thunk>> CreateThunkForCubin(
      Thunk::ThunkInfo thunk_info, std::string kernel_name,
      std::vector<uint8_t> cubin,
      const emitters::KernelArguments& kernel_arguments,
      const LaunchDimensions& launch_dimensions, int64_t shmem_bytes = 0,
      bool use_pdl = false) override;

  xla::Future<TritonWrapperResult> CompileTritonToLlvm(
      absl::string_view kernel_name, const HloModule& hlo_module,
      const se::DeviceDescription& device_info,
      const BlockLevelParameters& block_level_parameters,
      const llvm::Triple& target_triple, const std::string& data_layout,
      TritonKernelSource triton_source, BorrowedMlirContext borrowed_context,
      bool is_xla_fusion) override;

  // Returns the list of kernel modules that were deferred (when
  // `defer_compilation` is true) and clears the internal list. The returned
  // modules are meant to be merged and compiled together by the caller.
  //
  // Calling this also ends the deferral phase: subsequent calls to
  // `CompileToTargetBinary` compile immediately instead of deferring. This lets
  // the caller consume the deferred kernels, merge them into the constants
  // module, and then compile that merged module in a single call.
  std::vector<llvm::orc::ThreadSafeModule> ConsumeDeferredModules() override;

 private:
  absl::StatusOr<std::vector<uint8_t>> CompileToCubinImpl(
      LlvmKernelSource kernel_source);

  absl::StatusOr<std::unique_ptr<Thunk>> CompileImpl(
      Thunk::ThunkInfo thunk_info, LlvmKernelSource kernel_source,
      const std::string& sanitized_kernel_name,
      const emitters::KernelArguments& kernel_arguments,
      const LaunchDimensions& launch_dimensions);

  LlvmIrCompiler compiler_;
  const se::DeviceDescription device_info_;
  const DebugOptions debug_options_;
  tsl::thread::ThreadPool* thread_pool_;
  const bool defer_compilation_;

  // Kernel modules whose compilation was deferred. Guarded by a mutex because
  // compilation may be requested concurrently.
  absl::Mutex deferred_modules_mutex_;
  std::vector<llvm::orc::ThreadSafeModule> deferred_modules_
      ABSL_GUARDED_BY(deferred_modules_mutex_);
  // Set once `ConsumeDeferredModules` is called; disables further deferral so
  // the merged constants module compiles immediately.
  bool deferral_consumed_ ABSL_GUARDED_BY(deferred_modules_mutex_) = false;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_CUBIN_CUSTOM_KERNEL_COMPILER_H_
