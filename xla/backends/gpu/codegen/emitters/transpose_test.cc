/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */
#include "xla/backends/gpu/codegen/emitters/transpose.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::testing::Contains;
using ::testing::Field;

// The MXFP4 weight-quantization kernel (`fused_convert_transpose.13`): an f32
// weight cast to a 4-bit type in both orientations. $0 is the root element
// type and $1 its layout suffix.
constexpr absl::string_view kConvertTransposeHlo = R"(
HloModule convert_transpose
fused_convert_transpose {
  p0 = f32[1024,4096]{1,0} parameter(0)
  side = f32[1024,128,32]{2,1,0} bitcast(p0)
  side_cvt = $0[1024,128,32]{2,1,0$1} convert(side)
  tr = f32[4096,1024]{1,0} transpose(p0), dimensions={1,0}
  tr_bc = f32[4096,32,32]{2,1,0} bitcast(tr)
  tr_cvt = $0[4096,32,32]{2,1,0$1} convert(tr_bc)
  ROOT t = ($0[1024,128,32]{2,1,0$1}, $0[4096,32,32]{2,1,0$1})
      tuple(side_cvt, tr_cvt)
}
ENTRY main {
  p0 = f32[1024,4096]{1,0} parameter(0)
  ROOT fusion = ($0[1024,128,32]{2,1,0$1}, $0[4096,32,32]{2,1,0$1})
      fusion(p0), kind=kInput, calls=fused_convert_transpose
})";

// The two f32 heroes of the MXFP8 `input_convert_transpose_fusion.1`, cast to
// FP8.
constexpr absl::string_view kTwoF32HeroesHlo = R"(
HloModule two_f32_heroes
fused_convert_transpose {
  p0 = bf16[16384,14336]{1,0} parameter(0)
  p1 = bf16[16384,14336]{1,0} parameter(1)
  a = f32[16384,14336]{1,0} convert(p0)
  b = f32[16384,14336]{1,0} convert(p1)
  ab = f32[16384,14336]{1,0} multiply(a, b)
  tr_a = f32[14336,16384]{1,0} transpose(a), dimensions={1,0}
  tr_ab = f32[14336,16384]{1,0} transpose(ab), dimensions={1,0}
  cvt_a = f8e4m3fn[14336,16384]{1,0} convert(tr_a)
  cvt_ab = f8e4m3fn[14336,16384]{1,0} convert(tr_ab)
  ROOT t = (f8e4m3fn[14336,16384]{1,0}, f8e4m3fn[14336,16384]{1,0})
      tuple(cvt_a, cvt_ab)
}
ENTRY main {
  p0 = bf16[16384,14336]{1,0} parameter(0)
  p1 = bf16[16384,14336]{1,0} parameter(1)
  ROOT fusion = (f8e4m3fn[14336,16384]{1,0}, f8e4m3fn[14336,16384]{1,0})
      fusion(p0, p1), kind=kInput, calls=fused_convert_transpose
})";

class TransposeFusionTest : public HloHardwareIndependentTestBase {
 protected:
  TransposeFusionTest() {
    mlir_context_.appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_.loadAllAvailableDialects();
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  // Builds the transpose emitter for the entry root of `hlo` on `device_info_`
  // and checks that it launches `num_blocks` blocks. If `vector_range` is set,
  // the thread-to-output indexing of the transposed root (index 1) must have a
  // range variable with these bounds, i.e. the vector symbol. If
  // `shared_memory_bytes` is set, the kernel's shared memory tiles must take
  // that many bytes.
  void ExpectLaunch(absl::string_view hlo, int64_t num_blocks,
                    std::optional<Interval> vector_range = std::nullopt,
                    std::optional<int64_t> shared_memory_bytes = std::nullopt) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    const HloFusionInstruction& fusion = *Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    HloFusionAnalysis analysis =
        HloFusionAnalysis::Create(fusion, device_info_);
    ASSERT_EQ(analysis.emitter_fusion_kind(),
              HloFusionAnalysis::EmitterFusionKind::kTranspose);
    std::unique_ptr<MlirKernelEmitter> emitter =
        CreateTransposeFusion(analysis);
    LaunchDimensions launch_dimensions = emitter->launch_dimensions();
    EXPECT_EQ(launch_dimensions.num_blocks(), num_blocks);
    // kNumRows x kTileSize threads, whatever the vector size.
    EXPECT_EQ(launch_dimensions.num_threads_per_block(),
              device_info_.gpu_compute_capability().IsRocm() ? 256 : 128);
    if (vector_range.has_value()) {
      std::optional<IndexingMap> indexing =
          emitter->ComputeThreadIdToOutputIndexing(/*root_index=*/1,
                                                   &mlir_context_);
      ASSERT_TRUE(indexing.has_value());
      EXPECT_THAT(
          indexing->GetRangeVars(),
          Contains(Field(&IndexingMap::Variable::bounds, *vector_range)))
          << *indexing;
    }
    if (shared_memory_bytes.has_value()) {
      ASSERT_OK_AND_ASSIGN(
          mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
          emitter->CreateMLIRModule(mlir_context_, fusion, "fusion",
                                    /*buffer_assignment=*/nullptr));
      int64_t bytes = 0;
      mlir_module->walk([&](AllocateSharedOp allocate) {
        auto type = mlir::cast<mlir::RankedTensorType>(allocate.getType());
        bytes += type.getNumElements() * type.getElementTypeBitWidth() / 8;
      });
      EXPECT_EQ(bytes, *shared_memory_bytes);
    }
  }

  se::DeviceDescription device_info_ = TestGpuDeviceInfo::AMDMI350DeviceInfo();
  mlir::MLIRContext mlir_context_;
};

// On gfx950 the work heuristic alone picks vector size 1 for this fusion: 4096
// blocks of a 32x32 tile, each thread storing one FP4 element, half a byte, of
// each output. The 4-bit rule raises the vector size to 2: a 64x64 tile, 1024
// blocks, and every thread storing whole bytes.
TEST_F(TransposeFusionTest, FourBitRootsGetWholeBytesPerThreadOnMi350) {
  ExpectLaunch(absl::Substitute(kConvertTransposeHlo, "f4e2m1fn", ":E(4)"),
               /*num_blocks=*/1024, Interval{0, 1});
}

// A tiled minor dimension of odd size (1023 rows) cannot hold whole bytes per
// thread, so the rule is skipped and the 32x32 tile stays: 32 x 128 blocks.
TEST_F(TransposeFusionTest, FourBitRootsWithOddTiledDimensionKeepTheTile) {
  constexpr absl::string_view kOddHlo = R"(
HloModule convert_transpose_odd
fused_convert_transpose {
  p0 = f32[1023,4096]{1,0} parameter(0)
  side_cvt = f4e2m1fn[1023,4096]{1,0:E(4)} convert(p0)
  tr = f32[4096,1023]{1,0} transpose(p0), dimensions={1,0}
  tr_cvt = f4e2m1fn[4096,1023]{1,0:E(4)} convert(tr)
  ROOT t = (f4e2m1fn[1023,4096]{1,0:E(4)}, f4e2m1fn[4096,1023]{1,0:E(4)})
      tuple(side_cvt, tr_cvt)
}
ENTRY main {
  p0 = f32[1023,4096]{1,0} parameter(0)
  ROOT fusion = (f4e2m1fn[1023,4096]{1,0:E(4)}, f4e2m1fn[4096,1023]{1,0:E(4)})
      fusion(p0), kind=kInput, calls=fused_convert_transpose
})";
  ExpectLaunch(kOddHlo, /*num_blocks=*/4096);
}

// 2-bit roots are outside the rule and keep the heuristics' 32x32 tile.
TEST_F(TransposeFusionTest, TwoBitRootsKeepTheTile) {
  ExpectLaunch(absl::Substitute(kConvertTransposeHlo, "s2", ":E(2)"),
               /*num_blocks=*/4096);
}

// Without a sub-byte root nothing changes either.
TEST_F(TransposeFusionTest, ByteRootsKeepTheTile) {
  ExpectLaunch(absl::Substitute(kConvertTransposeHlo, "f8e4m3fn", ""),
               /*num_blocks=*/4096);
}

// At vector size 4 the two heroes need 2 x 128 x 129 x 4 bytes, more than half
// of the CU's 160 KiB, so vector size 2: 64x64 tiles, 256 x 224 blocks.
TEST_F(TransposeFusionTest, TwoF32HeroesTakeAtMostHalfTheCuOnMi350) {
  ExpectLaunch(kTwoF32HeroesHlo, /*num_blocks=*/57344, Interval{0, 1},
               /*shared_memory_bytes=*/2 * 64 * 65 * 4);
}

// gfx942's 64 KiB per block already rules out vector size 4.
TEST_F(TransposeFusionTest, TwoF32HeroesOnMi300KeepTheirTile) {
  device_info_ = TestGpuDeviceInfo::AMDMI300DeviceInfo();
  ExpectLaunch(kTwoF32HeroesHlo, /*num_blocks=*/57344, Interval{0, 1},
               /*shared_memory_bytes=*/2 * 64 * 65 * 4);
}

// One f32 hero at vector size 4 fits twice into the CU: 128x128 tiles.
TEST_F(TransposeFusionTest, OneF32HeroKeepsVectorSizeFourOnMi350) {
  constexpr absl::string_view kOneF32HeroHlo = R"(
HloModule one_f32_hero
fused_convert_transpose {
  p0 = bf16[16384,4096]{1,0} parameter(0)
  a = f32[16384,4096]{1,0} convert(p0)
  side_cvt = f8e4m3fn[16384,4096]{1,0} convert(a)
  tr = f32[4096,16384]{1,0} transpose(a), dimensions={1,0}
  tr_cvt = f8e4m3fn[4096,16384]{1,0} convert(tr)
  ROOT t = (f8e4m3fn[16384,4096]{1,0}, f8e4m3fn[4096,16384]{1,0})
      tuple(side_cvt, tr_cvt)
}
ENTRY main {
  p0 = bf16[16384,4096]{1,0} parameter(0)
  ROOT fusion = (f8e4m3fn[16384,4096]{1,0}, f8e4m3fn[4096,16384]{1,0})
      fusion(p0), kind=kInput, calls=fused_convert_transpose
})";
  ExpectLaunch(kOneF32HeroHlo, /*num_blocks=*/4096, Interval{0, 3},
               /*shared_memory_bytes=*/128 * 129 * 4);
}

// CUDA keeps its 4-byte vectors, i.e. vector size 1 for f32: 32x32 tiles.
TEST_F(TransposeFusionTest, TwoF32HeroesOnB200KeepTheirTile) {
  device_info_ = TestGpuDeviceInfo::B200SXMDeviceInfo();
  ExpectLaunch(kTwoF32HeroesHlo, /*num_blocks=*/512 * 448, std::nullopt,
               /*shared_memory_bytes=*/2 * 32 * 33 * 4);
}

}  // namespace
}  // namespace xla::gpu
