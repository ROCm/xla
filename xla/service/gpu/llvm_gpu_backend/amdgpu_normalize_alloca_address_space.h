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

#ifndef XLA_SERVICE_GPU_LLVM_GPU_BACKEND_AMDGPU_NORMALIZE_ALLOCA_ADDRESS_SPACE_H_
#define XLA_SERVICE_GPU_LLVM_GPU_BACKEND_AMDGPU_NORMALIZE_ALLOCA_ADDRESS_SPACE_H_

#include "llvm/IR/PassManager.h"

namespace xla {
namespace gpu {

// LLVM pass that normalizes AMD GPU alloca address spaces to conform to the
// AMDGPU memory model specification.
// See implementation for detailed documentation.
class AMDGPUNormalizeAllocaAddressSpace
    : public llvm::PassInfoMixin<AMDGPUNormalizeAllocaAddressSpace> {
 public:
  llvm::PreservedAnalyses run(llvm::Module& module,
                               llvm::ModuleAnalysisManager& mam);
  
  static llvm::StringRef name() { return "AMDGPUNormalizeAllocaAddressSpace"; }
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_LLVM_GPU_BACKEND_AMDGPU_NORMALIZE_ALLOCA_ADDRESS_SPACE_H_

