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
          // AtomicSpinWait supports RELAXED, ACQUIRE, or ACQUIRE_RELEASE
          // semantics
          if (arg.semantic != triton::MemSemantic::RELAXED &&
              arg.semantic != triton::MemSemantic::ACQUIRE &&
              arg.semantic != triton::MemSemantic::ACQUIRE_RELEASE) {
            return absl::InvalidArgumentError(
                "AtomicSpinWaitOp only supports RELAXED, ACQUIRE, or "
                "ACQUIRE_RELEASE semantics");
          }
          return absl::OkStatus();
        }
      },
      instruction);
}

namespace {

// Helper to convert MemSyncScope to PTX scope string
absl::string_view MemSyncScopeToPTXScope(triton::MemSyncScope scope) {
  switch (scope) {
    case triton::MemSyncScope::SYSTEM:
      return "sys";
    case triton::MemSyncScope::GPU:
      return "gpu";
    case triton::MemSyncScope::CTA:
      return "cta";
  }
  LOG(FATAL) << "Unknown MemSyncScope value";
}

// Create LLVM ops for GetThreadIdInstruction
mlir::Value CreateGetThreadIdOps(const LLVMOpCreationParams& params) {
  auto& builder = params.builder;
  auto i32_type = builder.getI32Type();

  // Use inline PTX assembly for CUDA
  const absl::string_view get_tid_asm = R"(
    mov.u32 $0, %tid.x;
  )";
  auto asm_op = builder.create<LLVM::InlineAsmOp>(
      params.loc, i32_type, mlir::ValueRange{},
      builder.getStringAttr(get_tid_asm), builder.getStringAttr("=r"),
      /*has_side_effects=*/mlir::UnitAttr(),
      /*is_align_stack=*/mlir::UnitAttr(),
      LLVM::TailCallKindAttr::get(builder.getContext(),
                                  LLVM::TailCallKind::None),
      /*asm_dialect=*/LLVM::AsmDialectAttr(),
      /*operand_attrs=*/mlir::ArrayAttr());
  return asm_op.getResult(0);
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

  absl::string_view memory_semantic = MemSemanticToString(instruction.semantic);
  absl::string_view scope = MemSyncScopeToPTXScope(instruction.scope);

  // Build PTX inline assembly based on whether mask is present
  if (mask) {
    constexpr absl::string_view kAtomicWriteAsmWithMaskTemplate = R"(
    {
    .reg .pred %%p<>;
    setp.ne.u32 %%p<>, $2, 0;
    @%%p st.global.%s.%s.u32 [$0], $1;
    }
  )";
    std::string atomic_write_asm = absl::StrFormat(
        kAtomicWriteAsmWithMaskTemplate, scope, memory_semantic);
    auto asm_op = builder.create<LLVM::InlineAsmOp>(
        params.loc, i32_type, mlir::ValueRange{addr, value, mask},
        builder.getStringAttr(atomic_write_asm), builder.getStringAttr("l,r,r"),
        /*has_side_effects=*/builder.getUnitAttr(),
        /*is_align_stack=*/nullptr,
        LLVM::TailCallKindAttr::get(builder.getContext(),
                                    LLVM::TailCallKind::None),
        /*asm_dialect=*/nullptr,
        /*operand_attrs=*/nullptr);
    return asm_op.getResult(0);
  } else {
    constexpr absl::string_view kAtomicWriteAsmTemplate = R"(
    st.global.%s.%s.u32 [$0], $1;
  )";
    std::string atomic_write_asm =
        absl::StrFormat(kAtomicWriteAsmTemplate, scope, memory_semantic);
    auto asm_op = builder.create<LLVM::InlineAsmOp>(
        params.loc, i32_type, mlir::ValueRange{addr, value},
        builder.getStringAttr(atomic_write_asm), builder.getStringAttr("l,r"),
        /*has_side_effects=*/builder.getUnitAttr(),
        /*is_align_stack=*/nullptr,
        LLVM::TailCallKindAttr::get(builder.getContext(),
                                    LLVM::TailCallKind::None),
        /*asm_dialect=*/nullptr,
        /*operand_attrs=*/nullptr);
    return asm_op.getResult(0);
  }
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

  absl::string_view memory_semantic = MemSemanticToString(instruction.semantic);
  absl::string_view scope = MemSyncScopeToPTXScope(instruction.scope);
  absl::string_view comparator = ComparatorToString(instruction.comparator);

  // Build PTX inline assembly based on whether mask is present
  if (mask) {
    constexpr absl::string_view kAtomicSpinWaitAsmWithMaskTemplate = R"(
    {
    .reg .pred %%p<2>;
    .reg .b32 %%r<1>;
    setp.ne.u32 %%p0, $2, 0;
    @%%!p0 bra done;
    wait:
      ld.global.%s.%s.u32 %%r0, [$0];
      setp.%s.u32 %%p1, %%r0, $1;
      @%%p1 bra wait;
    done:
    }
  )";
    std::string atomic_wait_asm = absl::StrFormat(
        kAtomicSpinWaitAsmWithMaskTemplate, scope, memory_semantic, comparator);
    auto asm_op = builder.create<LLVM::InlineAsmOp>(
        params.loc, i32_type, mlir::ValueRange{addr, expected, mask},
        builder.getStringAttr(atomic_wait_asm), builder.getStringAttr("l,r,r"),
        /*has_side_effects=*/builder.getUnitAttr(),
        /*is_align_stack=*/nullptr,
        LLVM::TailCallKindAttr::get(builder.getContext(),
                                    LLVM::TailCallKind::None),
        /*asm_dialect=*/nullptr,
        /*operand_attrs=*/nullptr);
    return asm_op.getResult(0);
  } else {
    constexpr absl::string_view kAtomicSpinWaitAsmTemplate = R"(
    {
    .reg .pred %%p<1>;
    .reg .b32 %%r<1>;
    wait:
      ld.global.%s.%s.u32 %%r0, [$0];
      setp.%s.u32 %%p0, %%r0, $1;
      @%%p0 bra wait;
    }
  )";
    std::string atomic_wait_asm = absl::StrFormat(
        kAtomicSpinWaitAsmTemplate, scope, memory_semantic, comparator);
    auto asm_op = builder.create<LLVM::InlineAsmOp>(
        params.loc, i32_type, mlir::ValueRange{addr, expected},
        builder.getStringAttr(atomic_wait_asm), builder.getStringAttr("l,r"),
        /*has_side_effects=*/builder.getUnitAttr(),
        /*is_align_stack=*/nullptr,
        LLVM::TailCallKindAttr::get(builder.getContext(),
                                    LLVM::TailCallKind::None),
        /*asm_dialect=*/nullptr,
        /*operand_attrs=*/nullptr);
    return asm_op.getResult(0);
  }
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
