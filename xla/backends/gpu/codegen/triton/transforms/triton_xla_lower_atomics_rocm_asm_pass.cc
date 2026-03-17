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

// ROCm-specific lowering of atomic operations using inline GCN assembly.
// This implementation uses inline assembly for atomic operations to provide
// fine-grained control over memory ordering and synchronization.

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERATOMICSROCMASMPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

absl::string_view GetMemorySemanticStr(triton::MemSemantic semantic) {
  switch (semantic) {
    case triton::MemSemantic::RELAXED:
      return "monotonic";
    case triton::MemSemantic::ACQUIRE:
      return "acquire";
    case triton::MemSemantic::RELEASE:
      return "release";
    case triton::MemSemantic::ACQUIRE_RELEASE:
      return "acq_rel";
  }
}

absl::string_view GetMemSyncScopeStr(triton::MemSyncScope scope) {
  switch (scope) {
    case triton::MemSyncScope::GPU:
      return "agent";
    case triton::MemSyncScope::SYSTEM:
      return "system";
    case triton::MemSyncScope::CTA:
      return "workgroup";
  }
}

// Get SC (Scope) bits for MI300 GPUs based on memory scope
// SC0 | SC1 = System scope (cross-device)
// SC1 = Agent scope (single device, all CUs)
// SC0 = Workgroup scope (CU-local)
// No SC bits = Wavefront scope
std::string GetScopeBits(triton::MemSyncScope scope) {
  switch (scope) {
    case triton::MemSyncScope::SYSTEM:
      return "sc0 sc1";  // System scope
    case triton::MemSyncScope::GPU:
      return "sc1";      // Agent/Device scope
    case triton::MemSyncScope::CTA:
      return "sc0";      // Workgroup scope
  }
  return "";  // Wavefront scope (default)
}

absl::string_view GetComparatorStr(Comparator comparator) {
  switch (comparator) {
    case Comparator::EQ:
      return "eq";
    case Comparator::LT:
      return "lt";
  }
}

mlir::Type GetResultType(mlir::Type ptr_type, PatternRewriter& rewriter) {
  mlir::Type result_type = rewriter.getI32Type();
  auto ranked_tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(ptr_type);
  // Tensor arguments must have tensor result type.
  if (ranked_tensor_type) {
    result_type = mlir::RankedTensorType::get(ranked_tensor_type.getShape(),
                                              rewriter.getI32Type());
  }
  return result_type;
}

LogicalResult LowerAtomicWriteOpROCmAsm(AtomicWriteOp atomic_write,
                                        PatternRewriter& rewriter) {
  mlir::ImplicitLocOpBuilder builder(atomic_write.getLoc(), rewriter);

  mlir::Value ptr = atomic_write.getPtr();
  mlir::Value value = atomic_write.getValue();
  triton::MemSemantic semantic = atomic_write.getMemSyncSemantic();
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::RELEASE) {
    return rewriter.notifyMatchFailure(
        atomic_write, absl::StrFormat("Unsupported memory semantic: %s",
                                      stringifyMemSemantic(semantic)));
  }
  
  triton::MemSyncScope sync_scope = atomic_write.getMemSyncScope();
  std::string scope_bits = GetScopeBits(sync_scope);

  // GCN inline assembly for atomic store
  // Uses SC (Scope) bits for MI300 GPUs to control cache coherency:
  // - sc0 sc1 = System scope (cross-device via PCIe/Infinity Fabric)
  // - sc1 = Agent scope (single device, all CUs)
  // - sc0 = Workgroup scope (CU-local)
  // - (no bits) = Wavefront scope
  //
  // Based on aiter implementation and LLVM SIMemoryLegalizer:
  // - RELEASE semantic requires waiting for prior operations and flushing caches
  // - Uses __scoped_atomic_store_n semantics with proper memory barriers
  
  std::string wait_before = "";
  std::string store_flags = scope_bits.empty() ? "glc" : scope_bits;
  
  if (semantic == triton::MemSemantic::RELEASE) {
    // Wait for all prior memory operations to complete
    wait_before = "s_waitcnt vmcnt(0) lgkmcnt(0)\n    ";
  }
  
  const std::string kAtomicWriteAsmWithMask = absl::StrFormat(R"(
    {
    %sv_cmp_ne_u32 vcc, 0, $3
    s_and_saveexec_b64 s[0:1], vcc
    flat_atomic_store v[$1], $2 %s
    s_or_b64 exec, exec, s[0:1]
    }
  )", wait_before, store_flags);
  
  const std::string kAtomicWriteAsm = absl::StrFormat(R"(
    %sflat_atomic_store v[$1], $2 %s
  )", wait_before, store_flags);

  mlir::Type result_type = GetResultType(ptr.getType(), rewriter);
  mlir::Value mask = atomic_write.getMask();
  if (mask) {
    triton::ElementwiseInlineAsmOp::create(
        builder,
        /*result_types=*/result_type,
        /*asm_string=*/rewriter.getStringAttr(kAtomicWriteAsmWithMask),
        /*constraints=*/rewriter.getStringAttr("=v,v,v,v"),
        /*pure=*/rewriter.getBoolAttr(false),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args=*/mlir::ValueRange{ptr, value, mask});
  } else {
    triton::ElementwiseInlineAsmOp::create(
        builder,
        /*result_types=*/result_type,
        /*asm_string=*/rewriter.getStringAttr(kAtomicWriteAsm),
        /*constraints=*/rewriter.getStringAttr("=v,v,v"),
        /*pure=*/rewriter.getBoolAttr(false),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args=*/mlir::ValueRange{ptr, value});
  }
  // No results to replace; just erase the op.
  rewriter.eraseOp(atomic_write);
  return success();
}

LogicalResult LowerAtomicSpinWaitOpROCmAsm(AtomicSpinWaitOp atomic_wait,
                                           PatternRewriter& rewriter) {
  mlir::ImplicitLocOpBuilder builder(atomic_wait.getLoc(), rewriter);

  mlir::Value ptr = atomic_wait.getPtr();
  mlir::Value expected = atomic_wait.getExpected();
  triton::MemSemantic semantic = atomic_wait.getMemSyncSemantic();
  if (semantic != triton::MemSemantic::RELAXED &&
      semantic != triton::MemSemantic::ACQUIRE) {
    return rewriter.notifyMatchFailure(
        atomic_wait, absl::StrFormat("Unsupported memory semantic: %s",
                                     stringifyMemSemantic(semantic)));
  }
  
  triton::MemSyncScope sync_scope = atomic_wait.getMemSyncScope();
  std::string scope_bits = GetScopeBits(sync_scope);
  absl::string_view comparator = GetComparatorStr(atomic_wait.getComparator());
  
  // GCN inline assembly for atomic spin wait
  // Uses SC (Scope) bits for MI300 GPUs to control cache coherency during spin-wait.
  // Based on aiter's implementation using __scoped_atomic_load_n:
  // - The SC bits ensure proper cache coherency across the specified scope
  // - ACQUIRE semantic requires cache invalidation after the wait completes
  // - This matches the pattern: while(__scoped_atomic_load_n(...) < flag);
  
  std::string load_flags = scope_bits.empty() ? "glc" : scope_bits;
  
  const std::string kAtomicSpinWaitAsm = absl::StrFormat(R"(
    {
    .L_wait_loop_%%=:
      flat_atomic_load v0, v[$1] %s
      v_cmp_%s_u32 vcc, v0, $2
      s_cbranch_vccnz .L_wait_loop_%%=
    }
  )", load_flags, comparator);
  
  const std::string kAtomicSpinWaitAsmWithMask = absl::StrFormat(R"(
    {
    v_cmp_ne_u32 vcc, 0, $3
    s_and_saveexec_b64 s[0:1], vcc
    s_cbranch_execz .L_done_%%=
    .L_wait_loop_%%=:
      flat_atomic_load v0, v[$1] %s
      v_cmp_%s_u32 vcc, v0, $2
      s_cbranch_vccnz .L_wait_loop_%%=
    .L_done_%%=:
    s_or_b64 exec, exec, s[0:1]
    }
  )", load_flags, comparator);
  
  mlir::Type result_type = GetResultType(ptr.getType(), rewriter);
  Value mask = atomic_wait.getMask();
  if (mask) {
    triton::ElementwiseInlineAsmOp::create(
        builder,
        /*result_types=*/result_type,
        /*asm_string=*/rewriter.getStringAttr(kAtomicSpinWaitAsmWithMask),
        /*constraints=*/rewriter.getStringAttr("=v,v,v,v"),
        /*pure=*/rewriter.getBoolAttr(false),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args=*/mlir::ValueRange{ptr, expected, mask});
  } else {
    triton::ElementwiseInlineAsmOp::create(
        builder,
        /*result_types=*/result_type,
        /*asm_string=*/rewriter.getStringAttr(kAtomicSpinWaitAsm),
        /*constraints=*/rewriter.getStringAttr("=v,v,v"),
        /*pure=*/rewriter.getBoolAttr(false),
        /*packed_element=*/rewriter.getI32IntegerAttr(1),
        /*args=*/mlir::ValueRange{ptr, expected});
  }
  rewriter.eraseOp(atomic_wait);
  return success();
}

class TritonXLALowerAtomicsROCmAsmPass
    : public impl::TritonXLALowerAtomicsROCmAsmPassBase<
          TritonXLALowerAtomicsROCmAsmPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerAtomicWriteOpROCmAsm);
    patterns.add(LowerAtomicSpinWaitOpROCmAsm);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerAtomicsROCmAsmPass() {
  return std::make_unique<TritonXLALowerAtomicsROCmAsmPass>();
}

}  // namespace mlir::triton::xla
