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

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class ExternFunctionNameTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.loadDialect<LLVM::LLVMDialect>();
    context_.loadDialect<triton::TritonDialect>();
  }

  mlir::MLIRContext context_;
};

// Test parsing GetThreadId instruction
TEST_F(ExternFunctionNameTest, ParseGetThreadId) {
  auto result = ParseExternFunctionName("xla_get_thread_id");
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(std::holds_alternative<GetThreadIdInstruction>(*result));
}

// Test parsing AtomicWrite instructions
TEST_F(ExternFunctionNameTest, ParseAtomicWriteRelaxedGpu) {
  auto result = ParseExternFunctionName("xla_atomic_write_relaxed_gpu");
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicWriteInstruction>(*result));
  auto& instruction = std::get<AtomicWriteInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::RELAXED);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::GPU);
}

TEST_F(ExternFunctionNameTest, ParseAtomicWriteReleaseSystem) {
  auto result = ParseExternFunctionName("xla_atomic_write_release_system");
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicWriteInstruction>(*result));
  auto& instruction = std::get<AtomicWriteInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::RELEASE);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::SYSTEM);
}

TEST_F(ExternFunctionNameTest, ParseAtomicWriteAcqRelCta) {
  auto result = ParseExternFunctionName("xla_atomic_write_acq_rel_cta");
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicWriteInstruction>(*result));
  auto& instruction = std::get<AtomicWriteInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::ACQUIRE_RELEASE);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::CTA);
}

// Test parsing AtomicSpinWait instructions
TEST_F(ExternFunctionNameTest, ParseAtomicSpinWaitRelaxedGpuEq) {
  auto result = ParseExternFunctionName("xla_atomic_spin_wait_relaxed_gpu_eq");
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicSpinWaitInstruction>(*result));
  auto& instruction = std::get<AtomicSpinWaitInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::RELAXED);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::GPU);
  EXPECT_EQ(instruction.comparator, Comparator::EQ);
}

TEST_F(ExternFunctionNameTest, ParseAtomicSpinWaitAcquireSystemLt) {
  auto result =
      ParseExternFunctionName("xla_atomic_spin_wait_acquire_system_lt");
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicSpinWaitInstruction>(*result));
  auto& instruction = std::get<AtomicSpinWaitInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::ACQUIRE);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::SYSTEM);
  EXPECT_EQ(instruction.comparator, Comparator::LT);
}

TEST_F(ExternFunctionNameTest, ParseAtomicSpinWaitAcqRelCtaEq) {
  auto result = ParseExternFunctionName("xla_atomic_spin_wait_acq_rel_cta_eq");
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicSpinWaitInstruction>(*result));
  auto& instruction = std::get<AtomicSpinWaitInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::ACQUIRE_RELEASE);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::CTA);
  EXPECT_EQ(instruction.comparator, Comparator::EQ);
}

// Test parsing invalid function names
TEST_F(ExternFunctionNameTest, ParseInvalidFunctionName) {
  auto result = ParseExternFunctionName("invalid_function_name");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Unknown extern function name")));
}

TEST_F(ExternFunctionNameTest, ParseInvalidAtomicWrite) {
  auto result = ParseExternFunctionName("xla_atomic_write_invalid_gpu");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Unknown memory semantic")));
}

TEST_F(ExternFunctionNameTest, ParseInvalidScope) {
  auto result = ParseExternFunctionName("xla_atomic_write_relaxed_invalid");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Unknown memory sync scope")));
}

TEST_F(ExternFunctionNameTest, ParseInvalidComparator) {
  auto result =
      ParseExternFunctionName("xla_atomic_spin_wait_relaxed_gpu_invalid");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Unknown comparator")));
}

// Test serialization
TEST_F(ExternFunctionNameTest, SerializeGetThreadId) {
  GetThreadIdInstruction instruction;
  std::string result = SerializeExternFunctionName(instruction);
  EXPECT_EQ(result, "xla_get_thread_id");
}

TEST_F(ExternFunctionNameTest, SerializeAtomicWrite) {
  AtomicWriteInstruction instruction{.semantic = triton::MemSemantic::RELEASE,
                                     .scope = triton::MemSyncScope::GPU};
  std::string result = SerializeExternFunctionName(instruction);
  EXPECT_EQ(result, "xla_atomic_write_release_gpu");
}

TEST_F(ExternFunctionNameTest, SerializeAtomicWriteAcqRel) {
  AtomicWriteInstruction instruction{
      .semantic = triton::MemSemantic::ACQUIRE_RELEASE,
      .scope = triton::MemSyncScope::SYSTEM};
  std::string result = SerializeExternFunctionName(instruction);
  EXPECT_EQ(result, "xla_atomic_write_acq_rel_system");
}

TEST_F(ExternFunctionNameTest, SerializeAtomicSpinWait) {
  AtomicSpinWaitInstruction instruction{
      .semantic = triton::MemSemantic::ACQUIRE,
      .scope = triton::MemSyncScope::CTA,
      .comparator = Comparator::LT};
  std::string result = SerializeExternFunctionName(instruction);
  EXPECT_EQ(result, "xla_atomic_spin_wait_acquire_cta_lt");
}

// Test round-trip (parse then serialize)
TEST_F(ExternFunctionNameTest, RoundTripGetThreadId) {
  std::string original = "xla_get_thread_id";
  auto parsed = ParseExternFunctionName(original);
  ASSERT_THAT(parsed, IsOk());
  std::string serialized = SerializeExternFunctionName(*parsed);
  EXPECT_EQ(original, serialized);
}

TEST_F(ExternFunctionNameTest, RoundTripAtomicWrite) {
  std::string original = "xla_atomic_write_relaxed_gpu";
  auto parsed = ParseExternFunctionName(original);
  ASSERT_THAT(parsed, IsOk());
  std::string serialized = SerializeExternFunctionName(*parsed);
  EXPECT_EQ(original, serialized);
}

TEST_F(ExternFunctionNameTest, RoundTripAtomicSpinWait) {
  std::string original = "xla_atomic_spin_wait_acquire_system_lt";
  auto parsed = ParseExternFunctionName(original);
  ASSERT_THAT(parsed, IsOk());
  std::string serialized = SerializeExternFunctionName(*parsed);
  EXPECT_EQ(original, serialized);
}

// Test memory semantic validation
TEST_F(ExternFunctionNameTest, ValidateGetThreadIdAlwaysValid) {
  GetThreadIdInstruction instruction;
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicWriteRelaxed) {
  AtomicWriteInstruction instruction{.semantic = triton::MemSemantic::RELAXED,
                                     .scope = triton::MemSyncScope::GPU};
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicWriteRelease) {
  AtomicWriteInstruction instruction{.semantic = triton::MemSemantic::RELEASE,
                                     .scope = triton::MemSyncScope::GPU};
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicWriteAcquireInvalid) {
  AtomicWriteInstruction instruction{.semantic = triton::MemSemantic::ACQUIRE,
                                     .scope = triton::MemSyncScope::GPU};
  EXPECT_THAT(ValidateMemorySemantic(instruction),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("RELAXED or RELEASE")));
}

TEST_F(ExternFunctionNameTest, ValidateAtomicSpinWaitRelaxed) {
  AtomicSpinWaitInstruction instruction{
      .semantic = triton::MemSemantic::RELAXED,
      .scope = triton::MemSyncScope::GPU,
      .comparator = Comparator::EQ};
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicSpinWaitAcquire) {
  AtomicSpinWaitInstruction instruction{
      .semantic = triton::MemSemantic::ACQUIRE,
      .scope = triton::MemSyncScope::GPU,
      .comparator = Comparator::EQ};
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicSpinWaitReleaseInvalid) {
  AtomicSpinWaitInstruction instruction{
      .semantic = triton::MemSemantic::RELEASE,
      .scope = triton::MemSyncScope::GPU,
      .comparator = Comparator::EQ};
  EXPECT_THAT(ValidateMemorySemantic(instruction),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("RELAXED or ACQUIRE")));
}

// Test LLVM operation creation - verify correct intrinsics
TEST_F(ExternFunctionNameTest, CreateGetThreadIdOpsCUDA) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  GetThreadIdInstruction instruction;
  LLVMOpCreationParams params{.builder = builder,
                              .loc = builder.getUnknownLoc(),
                              .target = TargetBackend::CUDA,
                              .operands = {}};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);
  EXPECT_TRUE(result.getType().isInteger(32));

  // Verify the intrinsic call was created with correct name
  auto* defining_op = result.getDefiningOp();
  ASSERT_TRUE(defining_op != nullptr);
  auto intrinsic_op = mlir::dyn_cast<LLVM::CallIntrinsicOp>(defining_op);
  ASSERT_TRUE(intrinsic_op != nullptr);
  EXPECT_EQ(intrinsic_op.getIntrin(), "llvm.nvvm.read.ptx.sreg.tid.x");
}

TEST_F(ExternFunctionNameTest, CreateGetThreadIdOpsROCM) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  GetThreadIdInstruction instruction;
  LLVMOpCreationParams params{.builder = builder,
                              .loc = builder.getUnknownLoc(),
                              .target = TargetBackend::ROCM,
                              .operands = {}};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);
  EXPECT_TRUE(result.getType().isInteger(32));

  // Verify the intrinsic call was created with correct name for ROCm
  auto* defining_op = result.getDefiningOp();
  ASSERT_TRUE(defining_op != nullptr);
  auto intrinsic_op = mlir::dyn_cast<LLVM::CallIntrinsicOp>(defining_op);
  ASSERT_TRUE(intrinsic_op != nullptr);
  EXPECT_EQ(intrinsic_op.getIntrin(), "llvm.amdgcn.workitem.id.x");
}

TEST_F(ExternFunctionNameTest, CreateAtomicWriteOpsVerifyOrdering) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto ptr_type = LLVM::LLVMPointerType::get(&context_);
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {ptr_type, i32_type});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  AtomicWriteInstruction instruction{.semantic = triton::MemSemantic::RELEASE,
                                     .scope = triton::MemSyncScope::GPU};

  llvm::SmallVector<mlir::Value> operands = {entry_block->getArgument(0),
                                             entry_block->getArgument(1)};
  LLVMOpCreationParams params{.builder = builder,
                              .loc = builder.getUnknownLoc(),
                              .target = TargetBackend::CUDA,
                              .operands = operands};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);
  EXPECT_TRUE(result.getType().isInteger(32));

  // Add return to complete the function
  builder.create<LLVM::ReturnOp>(builder.getUnknownLoc(), result);

  // Verify a store operation was created with correct ordering and syncscope
  bool found_store = false;
  func.walk([&](LLVM::StoreOp op) {
    found_store = true;
    EXPECT_EQ(op.getOrdering(), LLVM::AtomicOrdering::release);
    EXPECT_EQ(op.getSyncscope().value_or(""), "device");  // CUDA GPU scope
    return mlir::WalkResult::interrupt();
  });
  EXPECT_TRUE(found_store) << "Should have created a store operation";
}

TEST_F(ExternFunctionNameTest, CreateAtomicWriteOpsVerifySyncscopeROCM) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto ptr_type = LLVM::LLVMPointerType::get(&context_);
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {ptr_type, i32_type});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  AtomicWriteInstruction instruction{.semantic = triton::MemSemantic::RELAXED,
                                     .scope = triton::MemSyncScope::CTA};

  llvm::SmallVector<mlir::Value> operands = {entry_block->getArgument(0),
                                             entry_block->getArgument(1)};
  LLVMOpCreationParams params{.builder = builder,
                              .loc = builder.getUnknownLoc(),
                              .target = TargetBackend::ROCM,
                              .operands = operands};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);

  // Add return to complete the function
  builder.create<LLVM::ReturnOp>(builder.getUnknownLoc(), result);

  // Verify a store operation was created with correct ordering and syncscope
  bool found_store = false;
  func.walk([&](LLVM::StoreOp op) {
    found_store = true;
    EXPECT_EQ(op.getOrdering(), LLVM::AtomicOrdering::monotonic);  // RELAXED
    EXPECT_EQ(op.getSyncscope().value_or(""), "workgroup");  // ROCm CTA scope
    return mlir::WalkResult::interrupt();
  });
  EXPECT_TRUE(found_store) << "Should have created a store operation";
}

TEST_F(ExternFunctionNameTest, CreateAtomicSpinWaitOpsVerifyLoop) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto ptr_type = LLVM::LLVMPointerType::get(&context_);
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {ptr_type, i32_type});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  AtomicSpinWaitInstruction instruction{
      .semantic = triton::MemSemantic::ACQUIRE,
      .scope = triton::MemSyncScope::GPU,
      .comparator = Comparator::EQ};

  llvm::SmallVector<mlir::Value> operands = {entry_block->getArgument(0),
                                             entry_block->getArgument(1)};
  LLVMOpCreationParams params{.builder = builder,
                              .loc = builder.getUnknownLoc(),
                              .target = TargetBackend::CUDA,
                              .operands = operands};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);
  EXPECT_TRUE(result.getType().isInteger(32));

  // Add return to complete the function
  builder.create<LLVM::ReturnOp>(builder.getUnknownLoc(), result);

  // Verify loop structure: should have load, icmp, and cond_br operations
  bool found_load = false;
  bool found_icmp = false;
  bool found_cond_br = false;

  func.walk([&](mlir::Operation* op) {
    if (auto load_op = mlir::dyn_cast<LLVM::LoadOp>(op)) {
      found_load = true;
      EXPECT_EQ(load_op.getOrdering(), LLVM::AtomicOrdering::acquire);
      EXPECT_EQ(load_op.getSyncscope().value_or(""),
                "device");  // CUDA GPU scope
    } else if (mlir::isa<LLVM::ICmpOp>(op)) {
      found_icmp = true;
    } else if (mlir::isa<LLVM::CondBrOp>(op)) {
      found_cond_br = true;
    }
  });

  EXPECT_TRUE(found_load) << "Should have atomic load operation";
  EXPECT_TRUE(found_icmp) << "Should have comparison operation";
  EXPECT_TRUE(found_cond_br) << "Should have conditional branch";
}

TEST_F(ExternFunctionNameTest, CreateAtomicSpinWaitOpsVerifyComparator) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto ptr_type = LLVM::LLVMPointerType::get(&context_);
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {ptr_type, i32_type});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  AtomicSpinWaitInstruction instruction{
      .semantic = triton::MemSemantic::RELAXED,
      .scope = triton::MemSyncScope::SYSTEM,
      .comparator = Comparator::LT};

  llvm::SmallVector<mlir::Value> operands = {entry_block->getArgument(0),
                                             entry_block->getArgument(1)};
  LLVMOpCreationParams params{.builder = builder,
                              .loc = builder.getUnknownLoc(),
                              .target = TargetBackend::CUDA,
                              .operands = operands};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);

  // Add return to complete the function
  builder.create<LLVM::ReturnOp>(builder.getUnknownLoc(), result);

  // Verify the comparison uses unsigned less-than predicate
  bool found_icmp = false;
  func.walk([&](LLVM::ICmpOp op) {
    found_icmp = true;
    EXPECT_EQ(op.getPredicate(), LLVM::ICmpPredicate::ult);  // unsigned LT
    return mlir::WalkResult::interrupt();
  });
  EXPECT_TRUE(found_icmp) << "Should have created comparison operation";
}

}  // namespace
}  // namespace mlir::triton::xla
