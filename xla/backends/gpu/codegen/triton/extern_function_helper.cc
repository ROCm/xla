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

#include "xla/backends/gpu/codegen/triton/extern_function_helper.h"

#include <string>
#include <variant>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace {

// Helper to parse MemSemantic from string
absl::StatusOr<triton::MemSemantic> ParseMemSemantic(
    absl::string_view semantic_str) {
  if (semantic_str == "relaxed") {
    return triton::MemSemantic::RELAXED;
  } else if (semantic_str == "acquire") {
    return triton::MemSemantic::ACQUIRE;
  } else if (semantic_str == "release") {
    return triton::MemSemantic::RELEASE;
  } else if (semantic_str == "acq_rel") {
    return triton::MemSemantic::ACQUIRE_RELEASE;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown memory semantic: %s", semantic_str));
}

// Helper to parse MemSyncScope from string
absl::StatusOr<triton::MemSyncScope> ParseMemSyncScope(
    absl::string_view scope_str) {
  if (scope_str == "system") {
    return triton::MemSyncScope::SYSTEM;
  } else if (scope_str == "gpu") {
    return triton::MemSyncScope::GPU;
  } else if (scope_str == "cta") {
    return triton::MemSyncScope::CTA;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown memory sync scope: %s", scope_str));
}

// Helper to parse Comparator from string
absl::StatusOr<Comparator> ParseComparator(absl::string_view comparator_str) {
  if (comparator_str == "eq") {
    return Comparator::EQ;
  } else if (comparator_str == "lt") {
    return Comparator::LT;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown comparator: %s", comparator_str));
}

// Helper to convert MemSemantic to string
absl::string_view MemSemanticToString(triton::MemSemantic semantic) {
  switch (semantic) {
    case triton::MemSemantic::RELAXED:
      return "relaxed";
    case triton::MemSemantic::ACQUIRE:
      return "acquire";
    case triton::MemSemantic::RELEASE:
      return "release";
    case triton::MemSemantic::ACQUIRE_RELEASE:
      return "acq_rel";
  }
  return "unknown";
}

// Helper to convert MemSyncScope to string
absl::string_view MemSyncScopeToString(triton::MemSyncScope scope) {
  switch (scope) {
    case triton::MemSyncScope::SYSTEM:
      return "system";
    case triton::MemSyncScope::GPU:
      return "gpu";
    case triton::MemSyncScope::CTA:
      return "cta";
  }
  return "unknown";
}

// Helper to convert Comparator to string
absl::string_view ComparatorToString(Comparator comparator) {
  switch (comparator) {
    case Comparator::EQ:
      return "eq";
    case Comparator::LT:
      return "lt";
  }
  return "unknown";
}

}  // namespace

absl::StatusOr<ExternFunctionInstruction> ParseExternFunctionName(
    absl::string_view func_name) {
  // Check for xla_get_thread_id
  if (func_name == "xla_get_thread_id") {
    return GetThreadIdInstruction{};
  }

  // Check for xla_atomic_write_<semantic>_<scope>
  if (absl::StartsWith(func_name, "xla_atomic_write_")) {
    absl::string_view remainder =
        func_name.substr(17);  // Skip "xla_atomic_write_"

    // Find the positions of underscores to split semantic and scope
    size_t first_underscore = remainder.find('_');
    if (first_underscore == absl::string_view::npos) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid atomic write function name: %s", func_name));
    }

    size_t second_underscore = remainder.find('_', first_underscore + 1);
    absl::string_view semantic_str;
    absl::string_view scope_str;

    if (second_underscore == absl::string_view::npos) {
      // Pattern: <semantic>_<scope>
      semantic_str = remainder.substr(0, first_underscore);
      scope_str = remainder.substr(first_underscore + 1);
    } else {
      // Pattern: <semantic_part1>_<semantic_part2>_<scope> (e.g., acq_rel_gpu)
      absl::string_view potential_semantic =
          remainder.substr(0, second_underscore);
      scope_str = remainder.substr(second_underscore + 1);

      // Check if there's another underscore in scope (invalid)
      if (scope_str.find('_') != absl::string_view::npos) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid atomic write function name: %s", func_name));
      }

      semantic_str = potential_semantic;
    }

    auto semantic = ParseMemSemantic(semantic_str);
    if (!semantic.ok()) {
      return semantic.status();
    }

    auto scope = ParseMemSyncScope(scope_str);
    if (!scope.ok()) {
      return scope.status();
    }

    return AtomicWriteInstruction{*semantic, *scope};
  }

  // Check for xla_atomic_spin_wait_<semantic>_<scope>_<comparator>
  if (absl::StartsWith(func_name, "xla_atomic_spin_wait_")) {
    absl::string_view remainder =
        func_name.substr(21);  // Skip "xla_atomic_spin_wait_"

    // Find underscores to split semantic, scope, and comparator
    size_t first_underscore = remainder.find('_');
    if (first_underscore == absl::string_view::npos) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid atomic spin wait function name: %s", func_name));
    }

    size_t second_underscore = remainder.find('_', first_underscore + 1);
    if (second_underscore == absl::string_view::npos) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid atomic spin wait function name: %s", func_name));
    }

    size_t third_underscore = remainder.find('_', second_underscore + 1);

    absl::string_view semantic_str;
    absl::string_view scope_str;
    absl::string_view comparator_str;

    if (third_underscore == absl::string_view::npos) {
      // Pattern: <semantic>_<scope>_<comparator>
      semantic_str = remainder.substr(0, first_underscore);
      scope_str = remainder.substr(first_underscore + 1,
                                   second_underscore - first_underscore - 1);
      comparator_str = remainder.substr(second_underscore + 1);
    } else {
      // Pattern: <semantic_part1>_<semantic_part2>_<scope>_<comparator>
      // (e.g., acq_rel_gpu_eq)
      absl::string_view potential_semantic =
          remainder.substr(0, second_underscore);
      scope_str = remainder.substr(second_underscore + 1,
                                   third_underscore - second_underscore - 1);
      comparator_str = remainder.substr(third_underscore + 1);

      // Check if there are more underscores (invalid)
      if (comparator_str.find('_') != absl::string_view::npos) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid atomic spin wait function name: %s", func_name));
      }

      semantic_str = potential_semantic;
    }

    auto semantic = ParseMemSemantic(semantic_str);
    if (!semantic.ok()) {
      return semantic.status();
    }

    auto scope = ParseMemSyncScope(scope_str);
    if (!scope.ok()) {
      return scope.status();
    }

    auto comparator = ParseComparator(comparator_str);
    if (!comparator.ok()) {
      return comparator.status();
    }

    return AtomicSpinWaitInstruction{*semantic, *scope, *comparator};
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown extern function name: %s", func_name));
}

std::string SerializeExternFunctionName(
    const ExternFunctionInstruction& instruction) {
  return std::visit(
      [](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, GetThreadIdInstruction>) {
          return "xla_get_thread_id";
        } else if constexpr (std::is_same_v<T, AtomicWriteInstruction>) {
          return absl::StrFormat("xla_atomic_write_%s_%s",
                                 MemSemanticToString(arg.semantic),
                                 MemSyncScopeToString(arg.scope));
        } else if constexpr (std::is_same_v<T, AtomicSpinWaitInstruction>) {
          return absl::StrFormat("xla_atomic_spin_wait_%s_%s_%s",
                                 MemSemanticToString(arg.semantic),
                                 MemSyncScopeToString(arg.scope),
                                 ComparatorToString(arg.comparator));
        }
      },
      instruction);
}

absl::Status ValidateMemorySemantic(
    const ExternFunctionInstruction& instruction) {
  return std::visit(
      [](auto&& arg) -> absl::Status {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, GetThreadIdInstruction>) {
          // No memory semantic validation needed for GetThreadId
          return absl::OkStatus();
        } else if constexpr (std::is_same_v<T, AtomicWriteInstruction>) {
          // AtomicWrite only supports RELAXED or RELEASE semantics
          if (arg.semantic != triton::MemSemantic::RELAXED &&
              arg.semantic != triton::MemSemantic::RELEASE) {
            return absl::InvalidArgumentError(
                "AtomicWriteOp only supports RELAXED or RELEASE semantics");
          }
          return absl::OkStatus();
        } else if constexpr (std::is_same_v<T, AtomicSpinWaitInstruction>) {
          // AtomicSpinWait only supports RELAXED or ACQUIRE semantics
          if (arg.semantic != triton::MemSemantic::RELAXED &&
              arg.semantic != triton::MemSemantic::ACQUIRE) {
            return absl::InvalidArgumentError(
                "AtomicSpinWaitOp only supports RELAXED or ACQUIRE semantics");
          }
          return absl::OkStatus();
        }
      },
      instruction);
}

namespace {

// Helper to convert MemSemantic to LLVM AtomicOrdering
LLVM::AtomicOrdering MemSemanticToAtomicOrdering(triton::MemSemantic semantic) {
  switch (semantic) {
    case triton::MemSemantic::RELAXED:
      return LLVM::AtomicOrdering::monotonic;
    case triton::MemSemantic::ACQUIRE:
      return LLVM::AtomicOrdering::acquire;
    case triton::MemSemantic::RELEASE:
      return LLVM::AtomicOrdering::release;
    case triton::MemSemantic::ACQUIRE_RELEASE:
      return LLVM::AtomicOrdering::acq_rel;
  }
  LOG(FATAL) << "Unknown MemSemantic value";
}

// Helper to convert MemSyncScope to LLVM syncscope string for target backend
llvm::StringRef MemSyncScopeToSyncScope(triton::MemSyncScope scope,
                                        TargetBackend target) {
  if (target == TargetBackend::CUDA) {
    // NVPTX memory model (LLVM standard syncscope names)
    switch (scope) {
      case triton::MemSyncScope::SYSTEM:
        return "";  // System scope for cross-GPU visibility
      case triton::MemSyncScope::GPU:
        return "device";
      case triton::MemSyncScope::CTA:
        return "block";
    }
  } else {  // ROCM
    // AMDGPU memory model
    switch (scope) {
      case triton::MemSyncScope::SYSTEM:
        return "";  // System scope for cross-GPU visibility
      case triton::MemSyncScope::GPU:
        return "agent";
      case triton::MemSyncScope::CTA:
        return "workgroup";
    }
  }
  LOG(FATAL) << "Unknown MemSyncScope value";
}

// Create LLVM ops for GetThreadIdInstruction
mlir::Value CreateGetThreadIdOps(const LLVMOpCreationParams& params) {
  auto& builder = params.builder;
  auto i32_type = builder.getI32Type();

  // Create intrinsic call (backend-specific)
  auto intrinsic_name = builder.getStringAttr(
      params.target == TargetBackend::CUDA ? "llvm.nvvm.read.ptx.sreg.tid.x"
                                           : "llvm.amdgcn.workitem.id.x");

  auto intrinsic_call = builder.create<LLVM::CallIntrinsicOp>(
      params.loc, i32_type, intrinsic_name, mlir::ValueRange{});

  return intrinsic_call->getResult(0);
}

// Create LLVM ops for AtomicWriteInstruction
mlir::Value CreateAtomicWriteOps(const AtomicWriteInstruction& instruction,
                                 const LLVMOpCreationParams& params) {
  auto& builder = params.builder;
  auto operands = params.operands;
  auto i32_type = builder.getI32Type();

  // Expected operand layout: [ptr, value, mask?]
  auto addr = operands[0];
  auto value = operands[1];
  mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

  llvm::StringRef syncscope =
      MemSyncScopeToSyncScope(instruction.scope, params.target);
  LLVM::AtomicOrdering ordering =
      MemSemanticToAtomicOrdering(instruction.semantic);

  // Prepare atomic store location
  mlir::Block* exit_block = nullptr;
  if (mask) {
    // Masked atomic: if (mask != 0) { atomic_store } else { nop }
    auto current_block = builder.getBlock();
    auto atomic_block = current_block->splitBlock(builder.getInsertionPoint());
    exit_block = atomic_block->splitBlock(builder.getInsertionPoint());

    // Check mask and branch
    builder.setInsertionPointToEnd(current_block);
    auto zero = builder.create<LLVM::ConstantOp>(params.loc, i32_type,
                                                 builder.getI32IntegerAttr(0));
    auto mask_nonzero = builder.create<LLVM::ICmpOp>(
        params.loc, LLVM::ICmpPredicate::ne, mask, zero);
    builder.create<LLVM::CondBrOp>(params.loc, mask_nonzero, atomic_block,
                                   exit_block);

    // Set insertion point for atomic store
    builder.setInsertionPointToStart(atomic_block);
  }

  // Perform atomic store
  builder.create<LLVM::StoreOp>(
      params.loc, value, addr, /*alignment=*/4, /*isVolatile=*/false,
      /*isNonTemporal=*/false, /*isInvariantGroup=*/false, ordering,
      builder.getStringAttr(syncscope));

  if (mask) {
    // Complete masked path: branch to exit
    builder.create<LLVM::BrOp>(params.loc, exit_block);
    builder.setInsertionPointToStart(exit_block);
  }

  // Return poison value (result not expected to be used)
  return builder.create<LLVM::PoisonOp>(params.loc, i32_type);
}

// Create LLVM ops for AtomicSpinWaitInstruction
mlir::Value CreateAtomicSpinWaitOps(
    const AtomicSpinWaitInstruction& instruction,
    const LLVMOpCreationParams& params) {
  auto& builder = params.builder;
  auto operands = params.operands;
  auto i32_type = builder.getI32Type();

  // Expected operand layout: [ptr, expected, mask?]
  auto addr = operands[0];
  auto expected = operands[1];
  mlir::Value mask = operands.size() > 2 ? operands[2] : mlir::Value{};

  llvm::StringRef syncscope =
      MemSyncScopeToSyncScope(instruction.scope, params.target);
  LLVM::AtomicOrdering ordering =
      MemSemanticToAtomicOrdering(instruction.semantic);

  // acq_rel is not valid for loads (only for RMW operations)
  if (ordering == LLVM::AtomicOrdering::acq_rel) {
    LOG(FATAL) << "acq_rel ordering is not supported for atomic loads. "
               << "Use acquire ordering instead.";
  }

  bool is_lt = (instruction.comparator == Comparator::LT);

  // Create block structure (common for both masked and unmasked)
  auto current_block = builder.getBlock();
  auto loop_block = current_block->splitBlock(builder.getInsertionPoint());
  // Need to set insertion point to loop_block before splitting it
  builder.setInsertionPointToStart(loop_block);
  auto exit_block = loop_block->splitBlock(builder.getInsertionPoint());
  exit_block->addArgument(i32_type, params.loc);

  builder.setInsertionPointToEnd(current_block);

  if (mask) {
    // Masked: conditional branch based on mask (if mask==0, skip loop)
    auto zero = builder.create<LLVM::ConstantOp>(params.loc, i32_type,
                                                 builder.getI32IntegerAttr(0));
    auto mask_nonzero = builder.create<LLVM::ICmpOp>(
        params.loc, LLVM::ICmpPredicate::ne, mask, zero);
    LLVM::CondBrOp::create(builder, params.loc, mask_nonzero, loop_block,
                           mlir::ValueRange{}, exit_block,
                           mlir::ValueRange{zero}, std::nullopt);
  } else {
    // Unmasked: unconditional branch to loop (required terminator)
    LLVM::BrOp::create(builder, params.loc, mlir::ValueRange{}, loop_block);
  }

  // Loop: atomic load + compare + conditional branch
  builder.setInsertionPointToStart(loop_block);
  auto loaded = builder.create<LLVM::LoadOp>(
      params.loc, i32_type, addr, /*alignment=*/4, /*isVolatile=*/false,
      /*isNonTemporal=*/false, /*isInvariant=*/false,
      /*isInvariantGroup=*/false, ordering, builder.getStringAttr(syncscope));

  auto condition =
      is_lt ? builder.create<LLVM::ICmpOp>(params.loc, LLVM::ICmpPredicate::ult,
                                           loaded, expected)
            : builder.create<LLVM::ICmpOp>(params.loc, LLVM::ICmpPredicate::ne,
                                           loaded, expected);

  LLVM::CondBrOp::create(builder, params.loc, condition, loop_block,
                         mlir::ValueRange{}, exit_block,
                         mlir::ValueRange{loaded}, std::nullopt);

  // Return exit block argument
  builder.setInsertionPointToStart(exit_block);
  return exit_block->getArgument(0);
}

}  // namespace

mlir::Value CreateLLVMOpsForInstruction(
    const ExternFunctionInstruction& instruction,
    const LLVMOpCreationParams& params) {
  return std::visit(
      [&params](auto&& arg) -> mlir::Value {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, GetThreadIdInstruction>) {
          return CreateGetThreadIdOps(params);
        } else if constexpr (std::is_same_v<T, AtomicWriteInstruction>) {
          return CreateAtomicWriteOps(arg, params);
        } else if constexpr (std::is_same_v<T, AtomicSpinWaitInstruction>) {
          return CreateAtomicSpinWaitOps(arg, params);
        }
      },
      instruction);
}

}  // namespace mlir::triton::xla
