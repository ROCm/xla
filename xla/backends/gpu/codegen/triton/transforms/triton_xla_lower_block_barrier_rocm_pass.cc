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

// ROCm-specific lowering of BlockBarrierOp for Triton XLA.
// This implementation eliminates the need for thread IDs, following the IRIS
// pattern which uses only program IDs (block IDs) and tensor operations.
//
// Unlike the CUDA implementation which uses thread IDs to limit participation,
// this ROCm version uses tensor operations where each element represents a rank,
// and Triton's SPMD model automatically handles the distribution.

#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERBLOCKBARRIERROCMPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// ROCm-specific block barrier lowering that doesn't use thread IDs.
// This follows the IRIS pattern where synchronization is achieved through
// tensor-based atomic operations without thread-level indexing.
LogicalResult LowerBlockBarrierOpROCm(BlockBarrierOp block_barrier,
                                      PatternRewriter& rewriter) {
  mlir::ImplicitLocOpBuilder builder(block_barrier.getLoc(), rewriter);

  mlir::TypedValue<triton::PointerType> signal_buffers_arg =
      block_barrier.getSignalBuffers();
  const mlir::TypedValue<mlir::IntegerType> rank =
      block_barrier.getDeviceRank();
  const mlir::TypedValue<mlir::IntegerType> signal_value =
      block_barrier.getSignalValue();
  const int32_t world_size = block_barrier.getWorldSize();
  
  // Triton magic constant for global address space
  constexpr int32_t kGlobalAddressSpace = 1;

  const mlir::IntegerType i32_type = builder.getI32Type();
  const mlir::TypedValue<mlir::Type> world_size_op =
      mlir::arith::ConstantOp::create(builder,
                                      builder.getI32IntegerAttr(world_size));
  
  // Get block ID (program ID) - this is what IRIS uses
  const mlir::TypedValue<mlir::IntegerType> block_id =
      triton::GetProgramIdOp::create(builder, 0);

  // ROCm approach: Use tensor operations for all ranks
  // No thread ID check needed - all work is done via tensor operations
  
  // tensor<world_size x i32>
  const auto i32_tensor_type =
      mlir::RankedTensorType::get({world_size}, i32_type);
  // !tt.ptr<i32>
  const auto ptr_to_i32_type = mlir::triton::PointerType::get(
      builder.getI32Type(), kGlobalAddressSpace);
  // !tt.ptr<i64>
  const auto ptr_to_i64_type = mlir::triton::PointerType::get(
      builder.getI64Type(), kGlobalAddressSpace);
  // tensor<world_size x !tt.ptr<i32>>
  const auto tensor_of_ptr_to_i32_type =
      mlir::RankedTensorType::get({world_size}, ptr_to_i32_type);
  // tensor<world_size x !tt.ptr<i64>>
  const auto tensor_of_i64_ptr_type =
      mlir::RankedTensorType::get({world_size}, ptr_to_i64_type);

  // Cast signal buffers to i64 for pointer arithmetic
  auto signal_buffers_i64 = mlir::triton::BitcastOp::create(
      builder, ptr_to_i64_type, signal_buffers_arg);
  
  // Create tensor of signal buffer pointers for all ranks
  auto signal_buffers_tensor = mlir::triton::SplatOp::create(
      builder, tensor_of_i64_ptr_type, signal_buffers_i64);
  
  // Create range [0, world_size) representing all ranks
  auto all_ranks = mlir::triton::MakeRangeOp::create(
      builder, i32_tensor_type, 0, world_size);
  
  // Get pointers to each rank's signal buffer
  auto signal_buffer_ptr = mlir::triton::AddPtrOp::create(
      builder, tensor_of_i64_ptr_type, signal_buffers_tensor, all_ranks);
  
  // Load the signal buffer addresses
  auto signal_buffer_i64 = mlir::triton::LoadOp::create(
      builder,
      /*ptr=*/signal_buffer_ptr,
      /*cache=*/mlir::triton::CacheModifier::NONE,
      /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
      /*isVolatile=*/false);
  
  // Convert to i32 pointers
  auto signal_buffer = mlir::triton::IntToPtrOp::create(
      builder, tensor_of_ptr_to_i32_type, signal_buffer_i64);
  
  // Calculate offset: block_id * world_size + rank
  auto block_offset = mlir::arith::MulIOp::create(
      builder, i32_type, block_id, world_size_op);
  auto block_offset_plus_rank =
      mlir::arith::AddIOp::create(builder, i32_type, block_offset, rank);
  
  // Broadcast to tensor
  auto block_offset_plus_rank_tensor = mlir::triton::SplatOp::create(
      builder, i32_tensor_type, block_offset_plus_rank);
  
  // Calculate signal addresses for all ranks
  auto signal_addresses = mlir::triton::AddPtrOp::create(
      builder, tensor_of_ptr_to_i32_type, signal_buffer,
      block_offset_plus_rank_tensor);
  
  // Signal all ranks (tensor-based atomic write)
  mlir::triton::xla::AtomicWriteOp::create(
      builder,
      /*resultTypes=*/mlir::TypeRange{},
      /*ptr=*/signal_addresses,
      /*signal_value=*/signal_value,
      /*mask=*/mlir::Value{},
      /*scope=*/mlir::triton::MemSyncScope::SYSTEM,
      /*sem=*/mlir::triton::MemSemantic::RELEASE);
  
  // Wait for all ranks to signal
  // Get this rank's signal buffer
  auto read_address_ptr_to_i64 = mlir::triton::AddPtrOp::create(
      builder, signal_buffers_i64.getType(), signal_buffers_i64, rank);
  
  auto read_address_i64 = mlir::triton::LoadOp::create(
      builder,
      /*ptr=*/read_address_ptr_to_i64,
      /*cache=*/mlir::triton::CacheModifier::NONE,
      /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
      /*isVolatile=*/false);
  
  auto read_address = mlir::triton::IntToPtrOp::create(
      builder, ptr_to_i32_type, read_address_i64);
  
  // Calculate wait addresses
  auto read_address_at_block_offset = mlir::triton::AddPtrOp::create(
      builder, ptr_to_i32_type, read_address, block_offset);
  
  auto read_address_at_block_offset_tensor =
      mlir::triton::SplatOp::create(builder, tensor_of_ptr_to_i32_type,
                                    read_address_at_block_offset);
  
  auto wait_addresses = mlir::triton::AddPtrOp::create(
      builder, tensor_of_ptr_to_i32_type,
      read_address_at_block_offset_tensor, all_ranks);
  
  // Wait for all ranks (tensor-based atomic spin-wait)
  mlir::triton::xla::AtomicSpinWaitOp::create(
      builder,
      /*resultTypes=*/mlir::TypeRange{},
      /*ptr=*/wait_addresses,
      /*expected=*/signal_value,
      /*mask=*/mlir::Value{},
      /*scope=*/mlir::triton::MemSyncScope::SYSTEM,
      /*sem=*/mlir::triton::MemSemantic::ACQUIRE,
      /*comparator=*/Comparator::LT);
  
  // Block-level barrier to ensure all threads in the block are synchronized
  builder.create<mlir::triton::gpu::BarrierOp>(triton::gpu::AddrSpace::Local);
  
  rewriter.eraseOp(block_barrier);
  return success();
}

class TritonXLALowerBlockBarrierROCmPass
    : public impl::TritonXLALowerBlockBarrierROCmPassBase<
          TritonXLALowerBlockBarrierROCmPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(LowerBlockBarrierOpROCm);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLALowerBlockBarrierROCmPass() {
  return std::make_unique<TritonXLALowerBlockBarrierROCmPass>();
}

}  // namespace mlir::triton::xla
