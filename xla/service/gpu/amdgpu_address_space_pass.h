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

#ifndef XLA_SERVICE_GPU_AMDGPU_ADDRESS_SPACE_PASS_H_
#define XLA_SERVICE_GPU_AMDGPU_ADDRESS_SPACE_PASS_H_

#include "llvm/IR/Module.h"

namespace xla {
namespace gpu {

// Fixes AMD GPU alloca instructions to use address space 5.
// 
// AMD GPUs require stack allocations (allocas) to be in address space 5,
// not address space 0 (the default). This pass scans the LLVM module for
// alloca instructions in address space 0 and transforms them to use
// address space 5 with appropriate address space casts for compatibility.
//
// This transformation is necessary for correct code generation on AMD GPUs
// and should be run before module verification when targeting AMDGPU.
class AMDGPUAddressSpacePass {
 public:
  // Runs the address space transformation on the given LLVM module.
  // Returns true if the module was modified, false otherwise.
  static bool Run(llvm::Module* module);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AMDGPU_ADDRESS_SPACE_PASS_H_

