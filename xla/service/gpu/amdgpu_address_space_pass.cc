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

#include "xla/service/gpu/amdgpu_address_space_pass.h"

#include <vector>

#include "absl/log/log.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

namespace xla {
namespace gpu {

bool AMDGPUAddressSpacePass::Run(llvm::Module* module) {
  VLOG(3) << "Checking module for allocas in wrong address space...";
  
  // Collect all allocas that need to be fixed
  std::vector<llvm::AllocaInst*> allocas_to_fix;
  for (auto& function : *module) {
    for (auto& bb : function) {
      for (auto& inst : bb) {
        if (auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(&inst)) {
          if (alloca->getAddressSpace() == 0) {
            allocas_to_fix.push_back(alloca);
          }
        }
      }
    }
  }
  
  if (allocas_to_fix.empty()) {
    VLOG(3) << "No allocas need fixing";
    return false;
  }
  
  VLOG(3) << "Found " << allocas_to_fix.size() 
          << " allocas to fix";
  
  // Fix each alloca
  for (auto* alloca : allocas_to_fix) {
    VLOG(3) << "Fixing alloca: " << alloca->getName().str() 
            << " in AS" << alloca->getAddressSpace();
    
    // Get the parent function
    llvm::Function* func = alloca->getFunction();
    
    // Create new alloca in address space 5 at the entry block
    llvm::BasicBlock::iterator insert_pt = func->getEntryBlock().getFirstInsertionPt();
    llvm::AllocaInst* new_alloca = new llvm::AllocaInst(
        alloca->getAllocatedType(),  // Type to allocate
        5,                             // Address space
        alloca->getArraySize(),        // Array size
        alloca->getAlign(),            // Alignment
        alloca->getName() + ".as5");   // Name
    new_alloca->insertBefore(insert_pt);
    
    VLOG(3) << "Created new alloca in AS" 
            << new_alloca->getAddressSpace();
    
    // Replace all uses of old alloca directly with AS5 pointer (no cast!)
    // This keeps all memory accesses in AS5 for better performance on AMDGPU
    alloca->replaceAllUsesWith(new_alloca);
    alloca->eraseFromParent();
  }
  
  VLOG(3) << "Successfully fixed " << allocas_to_fix.size() 
          << " allocas";
  
  return true;
}

}  // namespace gpu
}  // namespace xla

