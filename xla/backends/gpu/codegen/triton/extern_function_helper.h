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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_EXTERN_FUNCTION_HELPER_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_EXTERN_FUNCTION_HELPER_H_

#include <string>
#include <variant>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

// Represents the different types of extern elementwise functions
// used in Triton XLA passes.

// Represents an atomic write operation
struct AtomicWriteInstruction {
  triton::MemSemantic semantic;
  triton::MemSyncScope scope;

  bool operator==(const AtomicWriteInstruction& other) const {
    return semantic == other.semantic && scope == other.scope;
  }
};

// Represents an atomic spin-wait operation
struct AtomicSpinWaitInstruction {
  triton::MemSemantic semantic;
  triton::MemSyncScope scope;
  Comparator comparator;

  bool operator==(const AtomicSpinWaitInstruction& other) const {
    return semantic == other.semantic && scope == other.scope &&
           comparator == other.comparator;
  }
};

// Represents a get thread ID operation
struct GetThreadIdInstruction {
  bool operator==(const GetThreadIdInstruction&) const { return true; }
};

// Variant type that can hold any of the supported instruction types
using ExternFunctionInstruction =
    std::variant<AtomicWriteInstruction, AtomicSpinWaitInstruction,
                 GetThreadIdInstruction>;

// Parses a function name string into an ExternFunctionInstruction variant.
// Returns an error status if the function name is invalid or doesn't match
// any known pattern.
//
// Supported patterns:
// - "xla_atomic_write_<semantic>_<scope>"
// - "xla_atomic_spin_wait_<semantic>_<scope>_<comparator>"
// - "xla_get_thread_id"
//
// Where:
// - <semantic>: relaxed, acquire, release, acq_rel
// - <scope>: system, gpu, cta
// - <comparator>: eq, lt
absl::StatusOr<ExternFunctionInstruction> ParseExternFunctionName(
    absl::string_view func_name);

// Serializes an ExternFunctionInstruction back to its string representation.
// This is the inverse of ParseExternFunctionName.
std::string SerializeExternFunctionName(
    const ExternFunctionInstruction& instruction);

// Validates that the memory semantic is appropriate for the instruction type.
// Returns an error status if validation fails.
absl::Status ValidateMemorySemantic(
    const ExternFunctionInstruction& instruction);

// Target backend for code generation
enum class TargetBackend {
  CUDA,
  ROCM,
};

// Parameters for creating LLVM operations from an instruction
struct LLVMOpCreationParams {
  mlir::OpBuilder& builder;
  mlir::Location loc;
  TargetBackend target;
  mlir::ValueRange
      operands;  // Operands from the call (ptr, value/expected, mask?)
};

// Creates the appropriate LLVM operations for the given instruction.
// This function generates the complete LLVM IR implementation for the
// instruction, including control flow for masked operations and loops for
// spin-wait operations.
//
// Returns the result value that should replace the original call operation.
// For operations that don't produce a meaningful result (like atomic_write),
// returns a poison value.
mlir::Value CreateLLVMOpsForInstruction(
    const ExternFunctionInstruction& instruction,
    const LLVMOpCreationParams& params);

}  // namespace mlir::triton::xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_EXTERN_FUNCTION_HELPER_H_
