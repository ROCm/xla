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

#include "xla/service/gpu/llvm_gpu_backend/amdgpu_normalize_alloca_address_space.h"

#include <vector>

#include "absl/log/log.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

namespace xla {
namespace gpu {

// LLVM pass that normalizes AMD GPU alloca address spaces to conform to the
// AMDGPU memory model specification.
//
// Background:
// The AMDGPU architecture requires stack allocations (alloca instructions) to
// reside in address space 5 (private/local memory), not address space 0 
// (generic/flat memory). Accessing private allocations through generic 
// pointers can result in:
//   - Performance degradation due to address translation overhead
//   - Incorrect behavior on certain GPU architectures
//
// This pass transforms all allocas from AS0 to AS5 and updates their uses
// to work directly with the private address space pointer, eliminating the
// need for address space casts that would negate the performance benefits.
//
// References:
// - AMDGPU Address Spaces Specification:
//   https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#address-spaces
// - AMDGPU Memory Model:
//   https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#memory-model
// - Address Space 5 Definition (Private/Stack):
//   Table 43 in the AMDGPU Usage documentation
//
// See also:
// - xla/codegen/emitters/transforms/lower_tensors.cc for MLIR-level alloca creation
llvm::PreservedAnalyses AMDGPUNormalizeAllocaAddressSpace::run(
    llvm::Module& module, llvm::ModuleAnalysisManager& mam) {
  llvm::Triple target_triple(module.getTargetTriple());
  
  // Only run this pass for AMD GPUs
  if (!target_triple.isAMDGPU()) {
    return llvm::PreservedAnalyses::all();
  }

  VLOG(3) << "AMDGPUNormalizeAllocaAddressSpace: Scanning module '" 
          << module.getName().str() << "' for allocas in address space 0";
  
  std::vector<llvm::AllocaInst*> allocas_to_fix;
  
  // Collect all allocas that need fixing
  for (auto& function : module) {
    for (auto& bb : function) {
      for (auto& inst : bb) {
        if (auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(&inst)) {
          if (alloca->getAddressSpace() == 0) {
            allocas_to_fix.push_back(alloca);
            VLOG(3) << "  Found alloca in function '" << function.getName().str()
                    << "': " << alloca->getName().str() 
                    << " (type: " << *alloca->getAllocatedType() 
                    << ", addrspace: " << alloca->getAddressSpace() << ")";
          }
        }
      }
    }
  }
  
  if (allocas_to_fix.empty()) {
    VLOG(3) << "AMDGPUNormalizeAllocaAddressSpace: No allocas to fix in module '" 
            << module.getName().str() << "'";
    return llvm::PreservedAnalyses::all();
  }
  
  VLOG(3) << "AMDGPUNormalizeAllocaAddressSpace: Fixing " << allocas_to_fix.size() 
          << " alloca(s) in module '" << module.getName().str() << "'";
  
  // Fix each alloca
  for (auto* alloca : allocas_to_fix) {
    llvm::Function* func = alloca->getFunction();
    
    VLOG(3) << "  Replacing alloca '" << alloca->getName().str() 
            << "' in function '" << func->getName().str() << "'";
    
    // Create new alloca in address space 5
    llvm::BasicBlock::iterator insert_pt = func->getEntryBlock().getFirstInsertionPt();
    llvm::AllocaInst* new_alloca = new llvm::AllocaInst(
        alloca->getAllocatedType(),  // Type to allocate
        5,                             // Address space (private/local for AMD GPU)
        alloca->getArraySize(),        // Array size
        alloca->getAlign(),            // Alignment
        alloca->getName() + ".as5");   // Name
    new_alloca->insertBefore(insert_pt);
    
    VLOG(3) << "    Created new alloca '" << new_alloca->getName().str()
            << "' in addrspace " << new_alloca->getAddressSpace();
    
    // Replace all uses directly with the AS5 pointer (no cast to AS0)
    // This ensures optimal performance and correctness for AMD GPUs
    alloca->replaceAllUsesWith(new_alloca);
    alloca->eraseFromParent();
    
    VLOG(3) << "    Successfully replaced alloca with AS5 version (direct access)";
  }
  
  VLOG(3) << "AMDGPUNormalizeAllocaAddressSpace: Successfully fixed " 
          << allocas_to_fix.size() << " alloca(s) in module '" 
          << module.getName().str() << "'";
  
  // We modified the module, so we can't preserve all analyses
  return llvm::PreservedAnalyses::none();
}

}  // namespace gpu
}  // namespace xla

