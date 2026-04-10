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

#include "xla/backends/gpu/codegen/triton/collective_emitter.h"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace xla::gpu {
namespace {

using ::mlir::Value;
using ::xla::se::gpu::AllReduceStrategy;
using xtile::TensorValue;

namespace ttir = ::mlir::triton;
namespace mtx = ::mlir::triton::xla;
namespace arith = ::mlir::arith;

using ReductionComputationEmitter = absl::AnyInvocable<xtile::TensorValue(
    mlir::ImplicitLocOpBuilder&, xtile::TensorValue, xtile::TensorValue)>;

// The main memory space on a device (HBM).
// Use GPU dialect address space which is platform-independent.
static constexpr auto kGlobalAddressSpace =
    static_cast<std::underlying_type_t<mlir::gpu::AddressSpace>>(
        mlir::gpu::AddressSpace::Global);

// Metadata arguments for the collective emitter.
// device_rank, signal_value, signal_buffers.
static constexpr int32_t kNumCollectiveMetadataArgs = 3;
static constexpr int32_t kNumTileIndexArgs = 1;

struct AllReduceInfo {
  ReductionKind reduction_kind;
  int64_t num_devices;
  int64_t num_elements;
  PrimitiveType element_type;
  AllReduceStrategy all_reduce_strategy;
};

// Common context for all reduce emitters.
struct AllReduceEmitterContext {
  mlir::stablehlo::AllReduceOp op;
  int32_t num_input_output_args;
  int32_t num_scratch_buffers;
  // The entry function of the all reduce op.
  xtile::EntryFuncOp xtile_entry_fn;
  // The input tile to all reduce.
  mlir::Value input_tile;
  // The extract tile op that produced the input tile.
  xtile::ExtractTileOp input_extract;
  AllReduceStrategy strategy;
  int64_t num_elements;
  int64_t input_byte_size;
};

absl::StatusOr<AllReduceEmitterContext> CreateAllReduceEmitterContext(
    mlir::stablehlo::AllReduceOp op) {
  AllReduceEmitterContext ctx;
  if (op.getOperands().size() != 1) {
    return absl::InvalidArgumentError(
        "AllReduce op must have exactly one operand in order to be lowered "
        "to triton.");
  }
  ctx.xtile_entry_fn = op->getParentOfType<xtile::EntryFuncOp>();
  if (!ctx.xtile_entry_fn) {
    return absl::InvalidArgumentError(
        "AllReduce op must be in an XTile entry function in order to be "
        "lowered to triton.");
  }
  // Variadics are not supported yet so we can fix inputs to 1.
  // Which means 2 arguments for input/output one for scratch buffers and
  // 3 metadata arguments. Plus 1 for the tile index for a total of 7.
  ctx.num_input_output_args = op->getNumOperands() * 2;
  ctx.num_scratch_buffers = op->getNumOperands();
  const int32_t expected_num_args =
      ctx.num_input_output_args + ctx.num_scratch_buffers +
      kNumCollectiveMetadataArgs + kNumTileIndexArgs;
  if (ctx.xtile_entry_fn.getNumArguments() != expected_num_args) {
    return absl::InvalidArgumentError(
        absl::StrCat("AllReduce op must have ", expected_num_args,
                     " arguments in order to "
                     "be lowered to triton, but it has ",
                     ctx.xtile_entry_fn.getNumArguments()));
  }
  ctx.input_tile = op->getOperand(0);
  // We assume the input to all reduce is an xtile::ExtractTileOp, or that the
  // parent of the input is an xtile::ExtractTileOp (edge case for booleans).
  ctx.input_extract =
      llvm::dyn_cast<xtile::ExtractTileOp>(ctx.input_tile.getDefiningOp());
  if (!ctx.input_extract &&
      ctx.input_tile.getDefiningOp()->getNumOperands() > 0) {
    // Workaround(i1_to_i8_workaround).
    // Go one place up this is an edge case for booleans
    // Booleans are stored as i8 and then casted to i1 so the tile we get is
    // after the cast. To get the extract tile we need to go one step up.
    ctx.input_extract = llvm::dyn_cast<xtile::ExtractTileOp>(
        ctx.input_tile.getDefiningOp()->getOperand(0).getDefiningOp());
  }
  if (!ctx.input_extract) {
    return absl::InvalidArgumentError(
        "AllReduce op must have an extract tile op as operand in order to be "
        "lowered to triton.");
  }
  llvm::ArrayRef<int64_t> non_tiled_input_shape =
      mlir::cast<mlir::ShapedType>(op.getOperand(0).getType()).getShape();
  ctx.num_elements = Product(non_tiled_input_shape);
  ctx.input_byte_size =
      ctx.num_elements *
      llvm::divideCeil(mlir::cast<mlir::ShapedType>(ctx.input_tile.getType())
                           .getElementTypeBitWidth(),
                       8);
  ctx.strategy = GetAllReduceStrategy(ctx.input_byte_size,
                                      /*is_multimem_enabled=*/false);
  ctx.op = op;
  return ctx;
}

// Returns the AllReduceInfo for the given all-reduce instruction if the
// instruction is supported by the codegen.
std::optional<AllReduceInfo> MaybeBuildAllReduceInfo(
    const HloAllReduceInstruction* all_reduce) {
  if (!all_reduce->GetModule()
           ->config()
           .debug_options()
           .xla_gpu_unsupported_use_all_reduce_one_shot_kernel()) {
    return std::nullopt;
  }
  if (all_reduce->device_list().replica_groups().empty()) {
    VLOG(1) << "Replica groups are empty for " << all_reduce->name()
            << ". Codegen will not be supported.";
    return std::nullopt;
  }
  const int64_t num_devices = all_reduce->device_list().num_devices_per_group();
  const std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(all_reduce->called_computations().front());
  if (!reduction_kind.has_value()) {
    return std::nullopt;
  }
  // TODO(b/383125489): Support variadic all-reduce.
  if (all_reduce->operand_count() > 1) {
    return std::nullopt;
  }
  const int64_t num_elements =
      ShapeUtil::ElementsIn(all_reduce->operand(0)->shape());
  const PrimitiveType element_type =
      all_reduce->operand(0)->shape().element_type();
  const int64_t byte_size =
      num_elements * ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  // TODO(b/457333991): Support twoShot for codegen.
  if (byte_size >
      GetMaxSupportedAllReduceSizeBytes(AllReduceStrategy::kOneShot)) {
    return std::nullopt;
  }
  // NB: We do not codegen multimem kernels for now.
  const AllReduceStrategy all_reduce_strategy =
      GetAllReduceStrategy(byte_size, /*is_multimem_enabled=*/false);
  if (!IsAllReduceKernelSupported(num_devices, num_elements, element_type,
                                  reduction_kind.value(),
                                  all_reduce_strategy)) {
    return std::nullopt;
  }
  return AllReduceInfo{
      /* .reduction_kind= */ reduction_kind.value(),
      /* .num_devices= */ num_devices,
      /* .num_elements= */ num_elements,
      /* .element_type= */ element_type,
      /* .all_reduce_strategy= */ all_reduce_strategy,
  };
}

// The logic here is very naive and assumes a monotonic layout
// where only the last dimension is used as a tiling dimension.
absl::StatusOr<std::optional<BlockLevelFusionConfig>>
GetBlockLevelFusionConfigForAllReduce(
    const se::DeviceDescription& device_info,
    const HloAllReduceInstruction* all_reduce) {
  // Check if the device supports Triton collective codegen
  bool is_supported = false;

  // CUDA: Requires compute capability 9.0+ (Hopper or newer)
  if (device_info.cuda_compute_capability().major >= 9) {
    is_supported = true;
  }

  // ROCm: All versions with Triton support are enabled
  if (device_info.gpu_compute_capability().IsRocm()) {
    is_supported = true;
  }

  if (!is_supported) {
    VLOG(3) << "Collective codegen requires CUDA compute capability >= 9.0 "
            << "or ROCm with Triton support. Codegen will not be supported.";
    return std::nullopt;
  }
  const std::optional<AllReduceInfo> all_reduce_info =
      MaybeBuildAllReduceInfo(all_reduce);
  if (!all_reduce_info.has_value()) {
    return std::nullopt;
  }
  const Shape& output_shape = all_reduce->shape();
  const LaunchDimensions launch_dims = AllReduceLaunchDimensions(
      all_reduce_info->num_elements, all_reduce_info->num_devices,
      all_reduce_info->all_reduce_strategy);
  BlockLevelFusionConfig block_level_config;
  block_level_config.set_num_warps(launch_dims.num_threads_per_block() /
                                   WarpSize(device_info));
  block_level_config.set_num_ctas(1);    // No block-level clustering.
  block_level_config.set_num_stages(1);  // No pipelining of loops.
  Tile* output_tile = block_level_config.add_output_tiles();
  const int64_t rank = output_shape.dimensions().size();

  // Tile sizes are rolled up to power of 2 because this is what triton expects
  // and consequently the tiling infra.
  for (int i = 0; i < rank - 1; ++i) {
    output_tile->add_sizes(llvm::PowerOf2Ceil(output_shape.dimensions(i)));
  }
  // The last dimension is divided amongst blocks.
  if (rank > 0) {
    const int64_t tile_size =
        CeilOfRatio(output_shape.dimensions(rank - 1),
                    absl::implicit_cast<int64_t>(launch_dims.num_blocks()));
    output_tile->add_sizes(llvm::PowerOf2Ceil(tile_size));
  }
  return block_level_config;
}

absl::StatusOr<std::vector<Shape>> GetAllReduceUnmanagedKernelArguments(
    const HloComputation* computation,
    const HloAllReduceInstruction* all_reduce) {
  const int32_t num_devices = all_reduce->device_list().num_devices_per_group();
  std::vector<Shape> unmanaged_arguments;
  unmanaged_arguments.reserve(computation->num_parameters() +
                              kNumCollectiveMetadataArgs);

  // rank and signal_value
  unmanaged_arguments.push_back(ShapeUtil::MakeShape(S32, {}));
  unmanaged_arguments.push_back(ShapeUtil::MakeShape(S32, {}));
  // The shape for signal and scratch buffers does not really matter in the end
  // because this would just be a pointer. For documentation purposes we add
  // the correct shape which would be
  // - num_devices * num_blocks for the signal buffer.
  // - num_devices * shape of the parameter for scratch buffers.
  // Since number of blocks is not known in this context we use a constant.
  static constexpr int32_t kMaxBlocksPerGrid = 24;
  unmanaged_arguments.push_back(
      ShapeUtil::MakeShape(S32, {num_devices, kMaxBlocksPerGrid}));
  // Scratch buffers
  for (const HloInstruction* instr : computation->parameter_instructions()) {
    Shape shape =
        ShapeUtil::InsertDimensionAtIndex(instr->shape(), 0, num_devices);
    unmanaged_arguments.push_back(shape);
  }
  TF_RET_CHECK(unmanaged_arguments.size() ==
               computation->num_parameters() + kNumCollectiveMetadataArgs);
  return unmanaged_arguments;
}

mlir::LogicalResult PopulateReductionComputation(
    mlir::PatternRewriter& rewriter, mlir::stablehlo::AllReduceOp op,
    ReductionComputationEmitter& computation_emitter) {
  // At the moment we expect only one operation in the reduction computation
  // to be relevant.
  auto& reduction_computation_region = op.getComputation();
  int num_ops_to_emit = 0;
  for (auto& block : reduction_computation_region.getBlocks()) {
    for (auto& block_op : block.without_terminator()) {
      if (llvm::dyn_cast<mlir::tensor::ExtractOp>(block_op) ||
          (llvm::dyn_cast<mlir::tensor::FromElementsOp>(block_op))) {
        // These ops are not relevant to the reduction and are just emitted so
        // that we have a valid stablehlo all reduce op.
        // We don't emit them, but they don't count towards our only one op in
        // the reduction computation requirement.
        continue;
      }

      if (block_op.getNumOperands() != 2) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "AllReduce computation must only contain binary operations.");
      }

      if (block_op.getNumResults() != 1) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "AllReduce computation must only contain operations with one "
            "result.");
      }

      computation_emitter = [&](mlir::ImplicitLocOpBuilder& builder,
                                xtile::TensorValue accumulator,
                                xtile::TensorValue next_tile) {
        // Emit a generic binary operation with the given operands.
        mlir::OperationState state(builder.getLoc(), block_op.getName());
        state.addOperands({accumulator, next_tile});
        state.addTypes({accumulator.getType()});
        mlir::Operation* new_op = builder.create(state);
        return mlir::cast<xtile::TensorValue>(new_op->getResult(0));
      };
      num_ops_to_emit++;
    }
  }

  if (num_ops_to_emit != 1) {
    return rewriter.notifyMatchFailure(
        op->getLoc(),
        "AllReduce op must have exactly one relevant operation in order to "
        "be lowered to triton.");
  }

  return mlir::success();
}

class AllReduceEmitter {
 public:
  static mlir::LogicalResult Emit(AllReduceEmitterContext ctx,
                                  mlir::PatternRewriter& rewriter) {
    AllReduceEmitter emitter(std::move(ctx), rewriter);
    if (mlir::failed(emitter.Initialize())) {
      return mlir::failure();
    }

    switch (emitter.ctx_.strategy) {
      case AllReduceStrategy::kOneShot:
        return emitter.EmitOneShot();
      case AllReduceStrategy::kTwoShot:
      case AllReduceStrategy::kMultimem:
        return emitter.rewriter_.notifyMatchFailure(
            emitter.ctx_.op->getLoc(),
            "Two-shot and multi-mem all-reduce are not supported yet for "
            "codegeneration.");
    }
  }

 private:
  AllReduceEmitter(AllReduceEmitterContext ctx, mlir::PatternRewriter& rewriter)
      : ctx_(std::move(ctx)),
        rewriter_(rewriter),
        builder_(ctx_.op->getLoc(), rewriter) {}

  mlir::LogicalResult Initialize() {
    CHECK(!initialized_);
    // NB: This must be done before any other IR is emitted so that we can bail
    // out in case it fails.
    // Otherwise, the IR is considered modified and we end up in an infinite
    // loop.
    if (mlir::failed(PopulateReductionComputation(
            rewriter_, ctx_.op, reduce_computation_emitter_))) {
      return mlir::failure();
    }
    // 1. Opaque arguments. They start after the input/output arguments.
    const int32_t start_idx = ctx_.num_input_output_args;
    device_rank_ = ctx_.xtile_entry_fn.getArgument(start_idx);
    CHECK(device_rank_.getType().isInteger(32));
    signal_value_ = ctx_.xtile_entry_fn.getArgument(start_idx + 1);
    CHECK(signal_value_.getType().isInteger(32));
    // !tt.ptr<!tt.ptr<i32>>
    signal_buffers_ = ctx_.xtile_entry_fn.getArgument(start_idx + 2);
    // !tt.ptr<!tt.ptr<i64>>
    remote_input_buffers_ = ctx_.xtile_entry_fn.getArgument(start_idx + 3);

    // 2. Constants and types.
    llvm::ArrayRef<int64_t> non_tiled_input_shape =
        mlir::cast<mlir::ShapedType>(ctx_.op.getOperand(0).getType())
            .getShape();
    world_size_ = ctx_.op.getReplicaGroups().getShapedType().getDimSize(1);

    elem_type_ = mlir::getElementTypeOrSelf(ctx_.input_tile.getType());
    elem_storage_type_ = xtile::StorageType(elem_type_);
    ptr_to_i64_type_ =
        ttir::PointerType::get(builder_.getI64Type(), kGlobalAddressSpace);
    ptr_to_elem_type_ =
        ttir::PointerType::get(elem_storage_type_, kGlobalAddressSpace);
    remote_memref_type_ =
        mlir::MemRefType::get(non_tiled_input_shape, elem_storage_type_);

    // 3. Emit setup IR.
    remote_input_buffers_i64_ = ttir::BitcastOp::create(
        builder_, ptr_to_i64_type_, remote_input_buffers_);

    const mlir::Type i64_type = builder_.getI64Type();
    const int64_t remote_buffer_size = ctx_.input_byte_size;
    // Check if last bit of signal_value is 0 or 1.
    Value buffer_index = arith::AndIOp::create(
        builder_, i64_type,
        arith::ExtSIOp::create(builder_, i64_type, signal_value_),  // i32->i64
        arith::ConstantOp::create(builder_, i64_type,
                                  builder_.getI64IntegerAttr(1)));
    buffer_offset_ = arith::MulIOp::create(
        builder_, i64_type, buffer_index,
        arith::ConstantOp::create(
            builder_, i64_type,
            builder_.getI64IntegerAttr(remote_buffer_size)));

    initialized_ = true;
    return mlir::success();
  }

  mlir::Value GetBufferPtr(mlir::Value buffer_ptr_base) {
    CHECK(initialized_);
    return ttir::AddPtrOp::create(builder_, ptr_to_elem_type_, buffer_ptr_base,
                                  buffer_offset_);
  }

  mlir::LogicalResult EmitScatter() {
    CHECK(initialized_);
    Value remote_buf_ptr_addr = ttir::AddPtrOp::create(
        builder_, ptr_to_i64_type_, remote_input_buffers_i64_, device_rank_);
    Value remote_buf_i64 =
        ttir::LoadOp::create(builder_, remote_buf_ptr_addr,
                             ttir::CacheModifier::NONE,     //
                             ttir::EvictionPolicy::NORMAL,  //
                             false);                        // isVolatile
    Value remote_buf_ptr_base =
        ttir::IntToPtrOp::create(builder_, ptr_to_elem_type_, remote_buf_i64,
                                 llvm::ArrayRef<mlir::NamedAttribute>{
                                     xtile::GetDivisibilityAttr(builder_)});
    Value remote_buf_ptr = GetBufferPtr(remote_buf_ptr_base);
    Value remote_buf_memref = mtx::PtrToMemrefOp::create(
        builder_, remote_memref_type_, remote_buf_ptr);

    // Workaround(i1_to_i8_workaround) as in fusion_emitter.
    // The parameter extraction casts the storage type to the logical type.
    // But for copying to the remote buffer we need to cast it back to the
    // storage type. Downstream passes should be able to optimize this away.
    Value storage_tile = ctx_.input_tile;
    if (elem_storage_type_ != elem_type_) {
      storage_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder_, ctx_.input_tile, elem_storage_type_));
    }
    xtile::InsertTileOp::create(
        builder_, storage_tile, remote_buf_memref,
        ctx_.input_extract.getOffsets(),
        ctx_.input_extract.getTile().getType().getShape(),
        ctx_.input_extract.getStrides());
    return mlir::success();
  }

  mlir::LogicalResult EmitSync() {
    CHECK(initialized_);
    mtx::BlockBarrierOp::create(builder_, signal_buffers_, device_rank_,
                                signal_value_,
                                builder_.getI32IntegerAttr(world_size_));
    return mlir::success();
  }

  mlir::LogicalResult EmitOneShot() {
    CHECK(initialized_);
    // 1. Scatter phase: Copy local tile to the remote buffer of the current
    // rank.
    if (mlir::failed(EmitScatter())) {
      return mlir::failure();
    }
    // 2. Synchronization phase: Wait for all ranks to complete the scatter.
    if (mlir::failed(EmitSync())) {
      return mlir::failure();
    }
    // 3. Reduce phase: Load tiles from all ranks and reduce them.
    const auto load_tile_for_rank = [&](int64_t rank) {
      Value rank_idx = arith::ConstantOp::create(
          builder_, builder_.getI64Type(), builder_.getI64IntegerAttr(rank));
      Value remote_buf_ptr_addr = ttir::AddPtrOp::create(
          builder_, ptr_to_i64_type_, remote_input_buffers_i64_, rank_idx);
      Value remote_buf_i64 =
          ttir::LoadOp::create(builder_, remote_buf_ptr_addr,
                               ttir::CacheModifier::NONE,     //
                               ttir::EvictionPolicy::NORMAL,  //
                               false);                        // isVolatile
      Value remote_buf_ptr_base =
          ttir::IntToPtrOp::create(builder_, ptr_to_elem_type_, remote_buf_i64);
      Value remote_buf_ptr = GetBufferPtr(remote_buf_ptr_base);
      Value remote_buf_memref = mtx::PtrToMemrefOp::create(
          builder_, remote_memref_type_, remote_buf_ptr);

      auto tensor_type = mlir::RankedTensorType::get(
          ctx_.input_extract.getTile().getType().getShape(),
          elem_storage_type_);

      xtile::TensorValue next_tile = xtile::ExtractTileOp::create(
          builder_, tensor_type, remote_buf_memref,
          ctx_.input_extract.getOffsets(),
          ctx_.input_extract.getTile().getType().getShape(),
          ctx_.input_extract.getStrides());
      // # Workaround(i1_to_i8_workaround) as in fusion_emitter.
      // See fusion emitter for more details.
      if (elem_storage_type_ != elem_type_) {
        next_tile = mlir::cast<xtile::TensorValue>(
            xtile::Cast(builder_, next_tile, elem_type_));
      }
      return next_tile;
    };

    xtile::TensorValue accumulator = load_tile_for_rank(0);
    for (int rank = 1; rank < world_size_; ++rank) {
      xtile::TensorValue next_tile = load_tile_for_rank(rank);
      accumulator =
          reduce_computation_emitter_(builder_, accumulator, next_tile);
    }
    rewriter_.replaceOp(ctx_.op, accumulator.getDefiningOp());
    return mlir::success();
  }

  AllReduceEmitterContext ctx_;
  mlir::PatternRewriter& rewriter_;
  mlir::ImplicitLocOpBuilder builder_;

  ReductionComputationEmitter reduce_computation_emitter_{nullptr};

  Value device_rank_;
  Value signal_value_;
  Value signal_buffers_;
  Value remote_input_buffers_;

  Value remote_input_buffers_i64_;
  Value buffer_offset_;

  int64_t world_size_;

  mlir::Type elem_type_;
  mlir::Type elem_storage_type_;
  ttir::PointerType ptr_to_i64_type_;
  ttir::PointerType ptr_to_elem_type_;
  mlir::MemRefType remote_memref_type_;

  bool initialized_ = false;
};

}  // namespace

llvm::SmallVector<int64_t> GreedyPowerOfTwoTiles(const Shape& output_shape,
                                                 int32_t num_blocks) {
  CHECK_GT(num_blocks, 0) << "num_blocks must be positive. Was " << num_blocks;

  // Triton has a hard limit of 1048576 elements per tensor
  constexpr int64_t kMaxTensorNumElements = 1048576;

  // Rank fits in int32_t.
  const auto rank = static_cast<int32_t>(output_shape.dimensions().size());
  const llvm::ArrayRef<const int64_t> minor_to_major =
      LayoutUtil::MinorToMajor(output_shape);
  llvm::SmallVector<int64_t, 4> tile_sizes(rank);

  // Calculate minimum blocks needed to respect the element limit
  int64_t total_elements = ShapeUtil::ElementsIn(output_shape);
  int64_t min_blocks_for_limit =
      CeilOfRatio(total_elements, kMaxTensorNumElements);

  // Use at least enough blocks to stay under the element limit
  uint64_t effective_num_blocks =
      std::max(static_cast<uint64_t>(num_blocks),
               static_cast<uint64_t>(min_blocks_for_limit));

  VLOG(3) << "GreedyPowerOfTwoTiles: shape=" << output_shape.ToString()
          << " requested_blocks=" << num_blocks
          << " total_elements=" << total_elements
          << " min_blocks_for_limit=" << min_blocks_for_limit
          << " effective_blocks=" << effective_num_blocks;

  // NB: Unsigned because llvm::bit_<> functions expect unsigned.
  uint64_t remaining_blocks = effective_num_blocks;

  // Iterate from most major to most minor to keep memory contiguous for each
  // block.
  for (int32_t i = rank - 1; i >= 0; --i) {
    const auto dim = static_cast<int32_t>(minor_to_major[i]);
    const uint64_t dim_size = output_shape.dimensions(dim);
    // Largest power-of-two k <= std::min(dim_size, remaining_blocks).
    const uint64_t k = std::max(
        uint64_t{1}, llvm::bit_floor(std::min(remaining_blocks, dim_size)));
    // Round up number of tiles in this dimension to power of two.
    tile_sizes[dim] =
        static_cast<int64_t>(llvm::bit_ceil(xla::CeilOfRatio(dim_size, k)));
    const uint64_t blocks_used =
        xla::CeilOfRatio(dim_size, static_cast<uint64_t>(tile_sizes[dim]));

    VLOG(3) << "  dim=" << dim << " dim_size=" << dim_size << " k=" << k
            << " tile_size=" << tile_sizes[dim]
            << " blocks_used=" << blocks_used
            << " remaining_blocks_before=" << remaining_blocks;

    remaining_blocks = xla::FloorOfRatio(remaining_blocks, blocks_used);

    VLOG(3) << "  remaining_blocks_after=" << remaining_blocks;
  }

  // Final check
  int64_t tile_num_elements = Product(tile_sizes);
  VLOG(3) << "Final tile_sizes: [" << absl::StrJoin(tile_sizes, ", ") << "]"
          << " total_tile_elements=" << tile_num_elements;

  if (tile_num_elements > kMaxTensorNumElements) {
    LOG(ERROR) << "Generated tile with " << tile_num_elements << " elements, "
               << "which exceeds Triton's limit of " << kMaxTensorNumElements
               << ". This will cause compilation to fail.";
  }

  return tile_sizes;
}

// Returns the block level fusion config for all-gather if supported.
absl::StatusOr<std::optional<BlockLevelFusionConfig>>
GetBlockLevelFusionConfigForAllGather(
    const se::DeviceDescription& device_info,
    const HloAllGatherInstruction* all_gather) {
  // Check if the Triton AllGather backend is enabled
  if (!all_gather->GetModule()
           ->config()
           .debug_options()
           .xla_gpu_unsupported_use_all_gather_triton_backend()) {
    LOG(INFO) << "AllGather Triton backend is not enabled. "
              << "Set xla_gpu_unsupported_use_all_gather_triton_backend=true "
                 "to enable.";
    return std::nullopt;
  }

  // Check if replica groups are present (required for Triton backend)
  if (!all_gather->device_list()) {
    LOG(INFO) << "AllGather device_list is null for " << all_gather->name()
              << ". This likely means replica_groups were not properly set. "
              << "Codegen will not be supported.";
    return std::nullopt;
  }

  if (all_gather->device_list()->replica_groups().empty()) {
    LOG(INFO) << "Replica groups are empty for " << all_gather->name()
              << ". Codegen will not be supported.";
    return std::nullopt;
  }

  const int64_t num_devices =
      all_gather->device_list()->num_devices_per_group();
  if (!llvm::has_single_bit(static_cast<uint64_t>(num_devices))) {
    VLOG(1) << "Number of devices is not a power of 2 for "
            << all_gather->name() << ". Codegen will not be supported.";
    return std::nullopt;
  }

  // Check if the device supports Triton collective codegen
  bool is_supported = false;

  // CUDA: Requires compute capability 9.0+ (Hopper or newer)
  if (device_info.cuda_compute_capability().major >= 9) {
    is_supported = true;
  }

  // ROCm: All versions with Triton support are enabled
  if (device_info.gpu_compute_capability().IsRocm()) {
    is_supported = true;
  }

  if (!is_supported) {
    LOG(INFO) << "Collective codegen requires CUDA compute capability >= 9.0 "
              << "or ROCm with Triton support. Codegen will not be supported.";
    return std::nullopt;
  }

  // Follow AllReduce pattern: calculate launch dimensions and tile sizes
  const Shape& input_shape = all_gather->operand(0)->shape();
  const int64_t num_elements = ShapeUtil::ElementsIn(input_shape);
  const PrimitiveType element_type = input_shape.element_type();

  // Check if the element type is supported by Triton
  // Triton primarily supports floating-point types and some integer types
  // Unsigned integer types (u32, u16, etc.) are not well supported
  if (element_type == U32 || element_type == U16 || element_type == U64) {
    LOG(INFO) << "AllGather: Unsigned integer type "
              << PrimitiveType_Name(element_type)
              << " is not supported by Triton backend. Falling back to NCCL.";
    return std::nullopt;
  }

  // Check alignment requirement (same as all-reduce)
  // This ensures we fall back to NCCL for shapes that won't work with Triton
  // tiling
  const int64_t alignment_requirement = se::gpu::kNumElementsPerThread;
  if (num_elements % alignment_requirement != 0) {
    LOG(INFO) << "AllGather: Number of elements (" << num_elements
              << ") is not aligned to the alignment requirement ("
              << alignment_requirement << "). Falling back to NCCL.";
    return std::nullopt;
  }

  // Calculate launch dimensions for AllGather (similar to AllReduce but without
  // strategy) For AllGather, each rank processes its own input data, so
  // elements_per_rank = num_elements
  static constexpr int64_t kWarpSize = 32;
  static constexpr uint64_t kMaxThreadsPerBlock = 512;
  static constexpr int64_t kMaxBlocksPerGrid = 32;

  const int64_t elements_per_rank =
      num_elements;  // AllGather doesn't split work like TwoShot
  // Maximum number of threads such that each thread has elements to process
  const int64_t total_threads =
      RoundUpTo(elements_per_rank / se::gpu::kNumElementsPerThread, kWarpSize);
  // Triton expects power of 2 for threads_per_block / threads_per_warp
  const int64_t threads_per_block =
      std::min(kMaxThreadsPerBlock,
               static_cast<uint64_t>(llvm::PowerOf2Ceil(total_threads)));
  const int64_t blocks_per_grid = std::min(
      kMaxBlocksPerGrid, CeilOfRatio(total_threads, threads_per_block));
  const LaunchDimensions launch_dims(blocks_per_grid, threads_per_block);

  // Use input shape for tiling (output shape may not have layout properly set)
  // The tiling is based on processing input-sized chunks from each rank
  BlockLevelFusionConfig block_level_config;
  // Ensure at least 1 warp (minimum valid value)
  const int64_t num_warps = std::max(
      int64_t{1}, static_cast<int64_t>(launch_dims.num_threads_per_block() /
                                       WarpSize(device_info)));
  block_level_config.set_num_warps(num_warps);
  block_level_config.set_num_ctas(1);    // No block-level clustering
  block_level_config.set_num_stages(1);  // No pipelining of loops

  Tile* output_tile = block_level_config.add_output_tiles();
  const llvm::SmallVector<int64_t> tile_sizes =
      GreedyPowerOfTwoTiles(input_shape, launch_dims.num_blocks());
  output_tile->mutable_sizes()->Assign(tile_sizes.begin(), tile_sizes.end());

  VLOG(3) << "Block level fusion config for " << all_gather->name() << ": "
          << block_level_config;
  return block_level_config;
}

absl::StatusOr<std::optional<BlockLevelFusionConfig>>
GetCollectiveBlockLevelFusionConfig(const se::DeviceDescription& device_info,
                                    const HloFusionInstruction* fusion_instr) {
  const HloInstruction* root = fusion_instr->fused_expression_root();

  // For AllGather-start, the fusion root is a GetTupleElement that extracts
  // the output from the AllGather-start tuple
  if (root->opcode() == HloOpcode::kGetTupleElement &&
      root->operand(0)->opcode() == HloOpcode::kAllGatherStart) {
    return GetBlockLevelFusionConfigForAllGather(
        device_info, Cast<HloAllGatherInstruction>(root->operand(0)));
  }

  switch (root->opcode()) {
    case HloOpcode::kAllReduceStart:
      return GetBlockLevelFusionConfigForAllReduce(
          device_info, Cast<HloAllReduceInstruction>(root));
    case HloOpcode::kAllGatherStart:
      return GetBlockLevelFusionConfigForAllGather(
          device_info, Cast<HloAllGatherInstruction>(root));
    default:
      return std::nullopt;
  }
}

absl::StatusOr<bool> TrySetGpuBackendConfigForCollective(
    const se::DeviceDescription& device_info,
    HloFusionInstruction* fusion_instr) {
  TF_ASSIGN_OR_RETURN(
      const std::optional<BlockLevelFusionConfig> block_config,
      GetCollectiveBlockLevelFusionConfig(device_info, fusion_instr));
  if (!block_config.has_value()) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      fusion_instr->backend_config<GpuBackendConfig>());
  gpu_backend_config.mutable_fusion_backend_config()->set_kind(
      kTritonCollectiveFusionKind);
  *gpu_backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() = *std::move(block_config);
  TF_RETURN_IF_ERROR(
      fusion_instr->set_backend_config(std::move(gpu_backend_config)));
  return true;
}

// Returns unmanaged kernel arguments for all-gather (similar to all-reduce).
absl::StatusOr<std::vector<Shape>> GetAllGatherUnmanagedKernelArguments(
    const HloComputation* computation,
    const HloAllGatherInstruction* all_gather) {
  // Check if device_list is valid to prevent segfault
  if (!all_gather->device_list()) {
    return absl::InternalError(absl::StrCat(
        "AllGather instruction ", all_gather->name(),
        " has null device_list. Cannot generate Triton kernel arguments."));
  }

  const int32_t num_devices =
      all_gather->device_list()->num_devices_per_group();

  std::vector<Shape> unmanaged_arguments;
  unmanaged_arguments.reserve(computation->num_parameters() +
                              kNumCollectiveMetadataArgs);
  // rank and signal_value
  unmanaged_arguments.push_back(ShapeUtil::MakeShape(S32, {}));
  unmanaged_arguments.push_back(ShapeUtil::MakeShape(S32, {}));

  // Signal and scratch buffers (same as AllReduce)
  static constexpr int32_t kMaxBlocksPerGrid = 24;
  unmanaged_arguments.push_back(
      ShapeUtil::MakeShape(S32, {num_devices, kMaxBlocksPerGrid}));

  // Scratch buffers for AllGather
  // Note: For AllGather, we need buffers to hold the gathered data
  if (!computation) {
    return absl::InternalError(
        "Computation is null in GetAllGatherUnmanagedKernelArguments");
  }

  for (const HloInstruction* instr : computation->parameter_instructions()) {
    if (!instr) {
      return absl::InternalError(
          "Null parameter instruction in GetAllGatherUnmanagedKernelArguments");
    }
    Shape shape =
        ShapeUtil::InsertDimensionAtIndex(instr->shape(), 0, num_devices);
    unmanaged_arguments.push_back(shape);
  }

  if (unmanaged_arguments.size() !=
      computation->num_parameters() + kNumCollectiveMetadataArgs) {
    return absl::InternalError(
        absl::StrCat("AllGather unmanaged arguments size mismatch. Expected: ",
                     computation->num_parameters() + kNumCollectiveMetadataArgs,
                     ", Got: ", unmanaged_arguments.size()));
  }
  return unmanaged_arguments;
}

absl::StatusOr<std::vector<Shape>> GetCollectiveUnmanagedKernelArguments(
    const HloFusionInstruction* fusion) {
  const HloComputation* computation = fusion->fused_instructions_computation();
  const HloInstruction* root = computation->root_instruction();

  // For AllGather-start wrapped in GetTupleElement
  if (root->opcode() == HloOpcode::kGetTupleElement &&
      root->operand(0)->opcode() == HloOpcode::kAllGatherStart) {
    return GetAllGatherUnmanagedKernelArguments(
        computation, Cast<HloAllGatherInstruction>(root->operand(0)));
  }

  switch (root->opcode()) {
    case HloOpcode::kAllReduceStart:
      return GetAllReduceUnmanagedKernelArguments(
          computation, Cast<HloAllReduceInstruction>(root));
    case HloOpcode::kAllGatherStart:
      return GetAllGatherUnmanagedKernelArguments(
          computation, Cast<HloAllGatherInstruction>(root));
    default:
      return std::vector<Shape>();
  }
}

absl::StatusOr<int32_t> AddCollectiveMetadataArguments(
    llvm::SmallVector<mlir::Type>& fn_arg_types, mlir::ImplicitLocOpBuilder& b,
    const HloComputation* hlo_computation) {
  // rank: i32
  fn_arg_types.push_back(b.getI32Type());
  // signal_value: i32
  fn_arg_types.push_back(b.getI32Type());
  // signal_buffers: !tt.ptr<!tt.ptr<i32>>
  fn_arg_types.push_back(ttir::PointerType::get(
      ttir::PointerType::get(b.getI32Type(), kGlobalAddressSpace),
      kGlobalAddressSpace));
  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    PrimitiveType type = p->shape().element_type();
    mlir::Type ir_type;
    if (type == U16) {
      ir_type = b.getI16Type();
    } else if (type == S4) {
      ir_type = b.getI4Type();
    } else {
      TF_ASSIGN_OR_RETURN(ir_type, xtile::PrimitiveTypeToMlirType(b, type));
    }
    // Also add the remote/scratch buffers for collectives.
    // !tt.ptr<!tt.ptr<type>>
    fn_arg_types.push_back(ttir::PointerType::get(
        ttir::PointerType::get(xtile::StorageType(ir_type),
                               kGlobalAddressSpace),
        kGlobalAddressSpace));
  }
  // num_metadata_args =
  return hlo_computation->num_parameters() + kNumCollectiveMetadataArgs;
}

mlir::LogicalResult RewriteAllReduce(mlir::stablehlo::AllReduceOp op,
                                     mlir::PatternRewriter& rewriter) {
  const mlir::Location loc = op->getLoc();
  absl::StatusOr<AllReduceEmitterContext> maybe_context =
      CreateAllReduceEmitterContext(op);
  if (!maybe_context.ok()) {
    return rewriter.notifyMatchFailure(
        loc, absl::StrCat("Failed to create AllReduceEmitterContext: ",
                          maybe_context.status().message()));
  }
  return AllReduceEmitter::Emit(maybe_context.value(), rewriter);
}

// Context for AllGather emitter
struct AllGatherEmitterContext {
  mlir::stablehlo::AllGatherOp op;
  int32_t num_input_output_args{0};
  int32_t num_scratch_buffers{0};
  xtile::EntryFuncOp xtile_entry_fn;
  xtile::TensorValue input_tile;
  xtile::ExtractTileOp input_extract;
  llvm::SmallVector<int64_t, 4> non_tiled_input_shape;
  PrimitiveType element_type;
  int64_t world_size{0};
  int64_t num_elements{0};
  int64_t all_gather_dimension{0};
};

absl::StatusOr<AllGatherEmitterContext> CreateAllGatherEmitterContext(
    mlir::stablehlo::AllGatherOp op) {
  AllGatherEmitterContext ctx;
  if (op.getOperands().size() != 1) {
    return absl::InvalidArgumentError(
        "AllGather op must have exactly one operand in order to be lowered "
        "to triton.");
  }

  mlir::Type element_type =
      mlir::cast<mlir::ShapedType>(op.getOperand(0).getType()).getElementType();
  ctx.element_type = xla::ConvertMlirTypeToPrimitiveType(element_type);
  if (ctx.element_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) {
    std::string type_string;
    llvm::raw_string_ostream stream(type_string);
    op.getOperand(0).print(stream);
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not convert operand type to a valid PrimitiveType. "
        "Operand Type: %s",
        type_string));
  }

  ctx.xtile_entry_fn = op->getParentOfType<xtile::EntryFuncOp>();
  if (!ctx.xtile_entry_fn) {
    return absl::InvalidArgumentError(
        "AllGather op must be in an XTile entry function in order to be "
        "lowered to triton.");
  }

  ctx.num_input_output_args = op->getNumOperands() * 2;
  ctx.num_scratch_buffers = op->getNumOperands();
  const int32_t expected_num_args =
      ctx.num_input_output_args + ctx.num_scratch_buffers +
      kNumCollectiveMetadataArgs + kNumTileIndexArgs;
  if (ctx.xtile_entry_fn.getNumArguments() != expected_num_args) {
    return absl::InvalidArgumentError(
        absl::StrCat("AllGather op must have ", expected_num_args,
                     " arguments in order to "
                     "be lowered to triton, but it has ",
                     ctx.xtile_entry_fn.getNumArguments()));
  }

  ctx.input_tile = mlir::cast<xtile::TensorValue>(op->getOperand(0));
  ctx.input_extract =
      llvm::dyn_cast<xtile::ExtractTileOp>(ctx.input_tile.getDefiningOp());
  if (!ctx.input_extract &&
      ctx.input_tile.getDefiningOp()->getNumOperands() > 0) {
    ctx.input_extract = llvm::dyn_cast<xtile::ExtractTileOp>(
        ctx.input_tile.getDefiningOp()->getOperand(0).getDefiningOp());
  }
  if (!ctx.input_extract) {
    return absl::InvalidArgumentError(
        "AllGather op must have an extract tile op as operand in order to be "
        "lowered to triton.");
  }

  ctx.non_tiled_input_shape = llvm::SmallVector<int64_t, 4>(
      ctx.input_extract.getSource().getType().getShape());
  ctx.num_elements = Product(ctx.non_tiled_input_shape);
  // Get world_size from replica_groups, not from getAllGatherDim()
  // getAllGatherDim() returns the dimension to gather along (e.g., 0)
  // world_size is the number of devices in the group
  ctx.world_size = op.getReplicaGroups().getShapedType().getDimSize(1);
  ctx.all_gather_dimension = op.getAllGatherDim();
  ctx.op = op;

  return ctx;
}

class AllGatherEmitter {
 public:
  static mlir::LogicalResult Emit(AllGatherEmitterContext ctx,
                                  mlir::PatternRewriter& rewriter,
                                  bool use_swizzled = false,
                                  int64_t group_size_m = 8) {
    AllGatherEmitter emitter(std::move(ctx), rewriter);
    if (auto result = emitter.Initialize(); !result.ok()) {
      LOG(ERROR) << "Failed to initialize AllGatherEmitter: "
                 << result.message();
      return mlir::failure();
    }

    if (use_swizzled) {
      return emitter.EmitAllGatherSwizzled(group_size_m);
    }
    return emitter.EmitAllGather();
  }

 private:
  AllGatherEmitter(AllGatherEmitterContext ctx, mlir::PatternRewriter& rewriter)
      : ctx_(std::move(ctx)),
        rewriter_(rewriter),
        builder_(ctx_.op->getLoc(), rewriter) {}

  absl::Status Initialize() {
    CHECK(!initialized_);

    // 1. Opaque arguments
    const int32_t start_idx = ctx_.num_input_output_args;
    device_rank_ = ctx_.xtile_entry_fn.getArgument(start_idx);
    CHECK(device_rank_.getType().isInteger(32));
    signal_value_ = ctx_.xtile_entry_fn.getArgument(start_idx + 1);
    CHECK(signal_value_.getType().isInteger(32));
    signal_buffers_ = ctx_.xtile_entry_fn.getArgument(start_idx + 2);
    remote_input_buffers_ = ctx_.xtile_entry_fn.getArgument(start_idx + 3);

    // 2. Constants and types
    elem_type_ = mlir::getElementTypeOrSelf(ctx_.input_tile.getType());
    elem_storage_type_ = xtile::StorageType(elem_type_);
    ptr_to_i64_type_ =
        ttir::PointerType::get(builder_.getI64Type(), kGlobalAddressSpace);
    ptr_to_elem_type_ =
        ttir::PointerType::get(elem_storage_type_, kGlobalAddressSpace);
    TF_ASSIGN_OR_RETURN(layout_, xtile::GetPermutationMinorToMajor(
                                     ctx_.input_extract.getSource().getType()));

    // 3. Emit setup IR
    remote_input_buffers_i64_ = ttir::BitcastOp::create(
        builder_, ptr_to_i64_type_, remote_input_buffers_);
    const mlir::Type i64_type = builder_.getI64Type();

    mlir::Value buffer_index = arith::AndIOp::create(
        builder_, i64_type,
        arith::ExtSIOp::create(builder_, i64_type, signal_value_),
        arith::ConstantOp::create(builder_, i64_type,
                                  builder_.getI64IntegerAttr(1)));

    const int64_t buffer_size = xla::RoundUpTo<uint64_t>(
        ctx_.num_elements *
            ShapeUtil::ByteSizeOfPrimitiveType(ctx_.element_type),
        kXlaAllocatedBufferAlignBytes);
    const int64_t elements_per_buffer =
        buffer_size / ShapeUtil::ByteSizeOfPrimitiveType(ctx_.element_type);
    buffer_offset_ = arith::MulIOp::create(
        builder_, i64_type, buffer_index,
        arith::ConstantOp::create(
            builder_, i64_type,
            builder_.getI64IntegerAttr(elements_per_buffer)));

    initialized_ = true;
    return absl::OkStatus();
  }

  mlir::Value GetRemoteBufferPtr(mlir::Value rank_idx) {
    CHECK(initialized_);
    mlir::Value remote_buf_ptr_addr = ttir::AddPtrOp::create(
        builder_, ptr_to_i64_type_, remote_input_buffers_i64_, rank_idx);
    mlir::Value remote_buf_i64 = ttir::LoadOp::create(
        builder_, remote_buf_ptr_addr, ttir::CacheModifier::NONE,
        ttir::EvictionPolicy::NORMAL,
        /*isVolatile=*/false);
    mlir::Value remote_buf_ptr_base =
        ttir::IntToPtrOp::create(builder_, ptr_to_elem_type_, remote_buf_i64,
                                 llvm::ArrayRef<mlir::NamedAttribute>{
                                     xtile::GetDivisibilityAttr(builder_)});
    mlir::Value remote_buf_ptr = ttir::AddPtrOp::create(
        builder_, ptr_to_elem_type_, remote_buf_ptr_base, buffer_offset_);
    return remote_buf_ptr;
  }

  xtile::TensorValue LoadTileForRank(mlir::Value rank_idx,
                                     mlir::ValueRange offsets,
                                     llvm::ArrayRef<int64_t> shape) {
    CHECK(initialized_);
    mlir::Value remote_buf_ptr = GetRemoteBufferPtr(rank_idx);
    auto [ptrs, mask] = triton::CreateTensorOfPointersAndMask(
        builder_, remote_buf_ptr, ctx_.non_tiled_input_shape, layout_, offsets,
        shape, ctx_.input_extract.getStrides(),
        /*reduced_dims=*/{}, shape);

    auto next_tile = mlir::cast<xtile::TensorValue>(
        ttir::LoadOp::create(builder_, ptrs, mask, /*other=*/mlir::Value(),
                             ttir::CacheModifier::NONE,
                             ttir::EvictionPolicy::NORMAL,
                             /*isVolatile=*/false)
            .getResult());

    if (elem_storage_type_ != elem_type_) {
      next_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder_, next_tile, elem_type_));
    }
    return next_tile;
  }

  mlir::LogicalResult EmitCopyToSymmetric(mlir::Value tile_to_store,
                                          mlir::ValueRange offsets,
                                          llvm::ArrayRef<int64_t> shape) {
    CHECK(initialized_);
    mlir::Value remote_buf_ptr = GetRemoteBufferPtr(device_rank_);

    mlir::Value storage_tile = tile_to_store;
    if (elem_storage_type_ != elem_type_) {
      storage_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder_, tile_to_store, elem_storage_type_));
    }

    auto [ptrs, mask] = triton::CreateTensorOfPointersAndMask(
        builder_, remote_buf_ptr, ctx_.non_tiled_input_shape, layout_, offsets,
        shape, ctx_.input_extract.getStrides(),
        /*reduced_dims=*/{}, shape);

    ttir::StoreOp::create(builder_, ptrs, storage_tile, mask,
                          ttir::CacheModifier::NONE,
                          ttir::EvictionPolicy::NORMAL);
    return mlir::success();
  }

  mlir::LogicalResult EmitSync(mlir::Value signal_value) {
    CHECK(initialized_);
    mlir::triton::gpu::BarrierOp::create(builder_,
                                         mlir::triton::gpu::AddrSpace::Local);
    mtx::BlockBarrierOp::create(builder_, signal_buffers_, device_rank_,
                                signal_value,
                                builder_.getI32IntegerAttr(ctx_.world_size));
    return mlir::success();
  }

  mlir::LogicalResult EmitAllGather() {
    CHECK(initialized_);

    llvm::ArrayRef<int64_t> shape = ctx_.input_tile.getType().getShape();

    // Get the block/program ID
    mlir::Value program_id = ttir::GetProgramIdOp::create(builder_, 0);

    // Calculate source rank: program_id % world_size
    // The modulo operation guarantees the result is in [0, world_size-1]
    mlir::Value world_size_i32 =
        arith::ConstantOp::create(builder_, builder_.getI32Type(),
                                  builder_.getI32IntegerAttr(ctx_.world_size));
    mlir::Value source_rank_i32 =
        arith::RemSIOp::create(builder_, program_id, world_size_i32);
    mlir::Value source_rank = arith::ExtSIOp::create(
        builder_, builder_.getI64Type(), source_rank_i32);

    // 1. Copy local tile to symmetric buffer (all ranks do this)
    if (mlir::failed(EmitCopyToSymmetric(
            ctx_.input_tile, ctx_.input_extract.getOffsets(),
            ctx_.input_tile.getType().getShape()))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit copy to symmetric");
    }

    // 2. Synchronization: Wait for all ranks to complete the copy
    if (mlir::failed(EmitSync(signal_value_))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit sync for all-gather");
    }

    // 3. Load the appropriate rank's data based on this block's program ID
    // Calculate the local offset within the source rank's buffer
    // For the gather dimension, offset is always 0 (we load the full input
    // size) For other dimensions, use the tile offset from input_extract
    mlir::ValueRange tile_offsets = ctx_.input_extract.getOffsets();
    llvm::SmallVector<mlir::Value> local_offsets;
    for (size_t i = 0; i < ctx_.non_tiled_input_shape.size(); ++i) {
      if (i == ctx_.all_gather_dimension) {
        // For gather dimension, always start at 0 in the source rank's buffer
        local_offsets.push_back(arith::ConstantOp::create(
            builder_, builder_.getIndexType(), builder_.getIndexAttr(0)));
      } else {
        // For other dimensions, use the tile offset
        local_offsets.push_back(tile_offsets[i]);
      }
    }

    xtile::TensorValue result =
        LoadTileForRank(source_rank, local_offsets, shape);

    // Replace the AllGather op with the loaded result
    rewriter_.replaceOp(ctx_.op, result);

    return mlir::success();
  }

  // Swizzled memory access variant that implements the tiling pattern:
  // num_pid_in_group = GROUP_SIZE_M * num_pid_n
  // group_id = tile_id // num_pid_in_group
  // first_pid_m = group_id * GROUP_SIZE_M
  // group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
  // pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
  // pid_n = (tile_id % num_pid_in_group) // group_size_m
  mlir::LogicalResult EmitAllGatherSwizzled(int64_t group_size_m) {
    CHECK(initialized_);

    llvm::ArrayRef<int64_t> shape = ctx_.input_tile.getType().getShape();

    // Get the tile/block ID
    mlir::Value tile_id = ttir::GetProgramIdOp::create(builder_, 0);

    // Calculate num_pid_m and num_pid_n based on the input shape and tile shape
    // For simplicity, assume 2D tiling with M and N dimensions
    // num_pid_m = ceil(M / BLOCK_SIZE_M)
    // num_pid_n = ceil(N / BLOCK_SIZE_N)

    // For AllGather, we need to determine the number of tiles in each dimension
    // This depends on the tile shape and the total output shape
    const int64_t block_size_m = shape.size() > 0 ? shape[0] : 1;
    const int64_t block_size_n = shape.size() > 1 ? shape[1] : 1;

    // Calculate total number of tiles needed
    const int64_t total_m = ctx_.non_tiled_input_shape[0] * ctx_.world_size;
    const int64_t total_n = ctx_.non_tiled_input_shape.size() > 1
                                ? ctx_.non_tiled_input_shape[1]
                                : 1;

    const int64_t num_pid_m = (total_m + block_size_m - 1) / block_size_m;
    const int64_t num_pid_n = (total_n + block_size_n - 1) / block_size_n;

    // Create constants
    mlir::Value num_pid_m_val = arith::ConstantOp::create(
        builder_, builder_.getI32Type(), builder_.getI32IntegerAttr(num_pid_m));
    mlir::Value num_pid_n_val = arith::ConstantOp::create(
        builder_, builder_.getI32Type(), builder_.getI32IntegerAttr(num_pid_n));
    mlir::Value group_size_m_val =
        arith::ConstantOp::create(builder_, builder_.getI32Type(),
                                  builder_.getI32IntegerAttr(group_size_m));

    // Swizzled tiling calculation
    // num_pid_in_group = GROUP_SIZE_M * num_pid_n
    mlir::Value num_pid_in_group =
        arith::MulIOp::create(builder_, group_size_m_val, num_pid_n_val);

    // group_id = tile_id // num_pid_in_group
    mlir::Value group_id =
        arith::DivSIOp::create(builder_, tile_id, num_pid_in_group);

    // first_pid_m = group_id * GROUP_SIZE_M
    mlir::Value first_pid_m =
        arith::MulIOp::create(builder_, group_id, group_size_m_val);

    // group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    mlir::Value remaining_m =
        arith::SubIOp::create(builder_, num_pid_m_val, first_pid_m);
    mlir::Value cmp = arith::CmpIOp::create(builder_, arith::CmpIPredicate::slt,
                                            remaining_m, group_size_m_val);
    mlir::Value group_size_m_actual =
        arith::SelectOp::create(builder_, cmp, remaining_m, group_size_m_val);

    // tile_id % num_pid_in_group
    mlir::Value tile_id_mod =
        arith::RemSIOp::create(builder_, tile_id, num_pid_in_group);

    // pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    mlir::Value offset_m =
        arith::RemSIOp::create(builder_, tile_id_mod, group_size_m_actual);
    mlir::Value pid_m = arith::AddIOp::create(builder_, first_pid_m, offset_m);

    // pid_n = (tile_id % num_pid_in_group) // group_size_m
    mlir::Value pid_n =
        arith::DivSIOp::create(builder_, tile_id_mod, group_size_m_actual);

    // Determine which rank's data to load based on pid_m
    // Each rank owns M/world_size rows
    const int64_t rows_per_rank = ctx_.non_tiled_input_shape[0];
    mlir::Value rows_per_rank_val = arith::ConstantOp::create(
        builder_, builder_.getI32Type(),
        builder_.getI32IntegerAttr(rows_per_rank / block_size_m));

    mlir::Value source_rank_i32 =
        arith::DivSIOp::create(builder_, pid_m, rows_per_rank_val);
    mlir::Value source_rank = arith::ExtSIOp::create(
        builder_, builder_.getI64Type(), source_rank_i32);

    // 1. Copy local tile to symmetric buffer (all ranks do this)
    if (mlir::failed(EmitCopyToSymmetric(
            ctx_.input_tile, ctx_.input_extract.getOffsets(),
            ctx_.input_tile.getType().getShape()))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit copy to symmetric");
    }

    // 2. Synchronization: Wait for all ranks to complete the copy
    if (mlir::failed(EmitSync(signal_value_))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit sync for all-gather");
    }

    // 3. Calculate offsets based on swizzled pid_m and pid_n
    mlir::ValueRange tile_offsets = ctx_.input_extract.getOffsets();
    llvm::SmallVector<mlir::Value> local_offsets;

    // For M dimension (gather dimension), calculate offset within source rank
    mlir::Value local_pid_m =
        arith::RemSIOp::create(builder_, pid_m, rows_per_rank_val);
    mlir::Value m_offset = arith::MulIOp::create(
        builder_, local_pid_m,
        arith::ConstantOp::create(builder_, builder_.getI32Type(),
                                  builder_.getI32IntegerAttr(block_size_m)));
    local_offsets.push_back(arith::IndexCastOp::create(
        builder_, builder_.getIndexType(), m_offset));

    // For N dimension, use pid_n
    if (ctx_.non_tiled_input_shape.size() > 1) {
      mlir::Value n_offset = arith::MulIOp::create(
          builder_, pid_n,
          arith::ConstantOp::create(builder_, builder_.getI32Type(),
                                    builder_.getI32IntegerAttr(block_size_n)));
      local_offsets.push_back(arith::IndexCastOp::create(
          builder_, builder_.getIndexType(), n_offset));
    }

    // Add remaining dimensions from tile_offsets
    for (size_t i = 2; i < ctx_.non_tiled_input_shape.size(); ++i) {
      local_offsets.push_back(tile_offsets[i]);
    }

    xtile::TensorValue result =
        LoadTileForRank(source_rank, local_offsets, shape);

    // Replace the AllGather op with the loaded result
    rewriter_.replaceOp(ctx_.op, result);

    return mlir::success();
  }

  AllGatherEmitterContext ctx_;
  mlir::PatternRewriter& rewriter_;
  mlir::ImplicitLocOpBuilder builder_;

  mlir::Value device_rank_;
  mlir::Value signal_value_;
  mlir::Value signal_buffers_;
  mlir::Value remote_input_buffers_;
  mlir::Value remote_input_buffers_i64_;
  mlir::Value buffer_offset_;

  llvm::SmallVector<int64_t> layout_;
  mlir::Type elem_type_;
  mlir::Type elem_storage_type_;
  ttir::PointerType ptr_to_i64_type_;
  ttir::PointerType ptr_to_elem_type_;

  bool initialized_ = false;
};

mlir::LogicalResult RewriteAllGather(mlir::stablehlo::AllGatherOp op,
                                     mlir::PatternRewriter& rewriter) {
  const mlir::Location loc = op->getLoc();
  absl::StatusOr<AllGatherEmitterContext> maybe_context =
      CreateAllGatherEmitterContext(op);
  if (!maybe_context.ok()) {
    VLOG(3) << "Failed to create AllGatherEmitterContext: "
            << maybe_context.status().message();
    return rewriter.notifyMatchFailure(
        loc, absl::StrCat("Failed to create AllGatherEmitterContext: ",
                          maybe_context.status().message()));
  }

  VLOG(3) << "AllGatherEmitter::Emit for all-gather";
  return AllGatherEmitter::Emit(maybe_context.value(), rewriter);
}

}  // namespace xla::gpu
