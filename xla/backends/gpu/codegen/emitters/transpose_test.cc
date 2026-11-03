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
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
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

class TransposeFusionTest : public HloHardwareIndependentTestBase {
 protected:
  TransposeFusionTest() { RegisterSymbolicExprStorage(&mlir_context_); }

  // Builds the transpose emitter for the entry root of `hlo` on an MI350
  // (gfx950) and returns its launch dimensions. If `vector_range` is set, the
  // thread-to-output indexing of the transposed root (index 1) must have a
  // range variable with these bounds, i.e. the vector symbol.
  void ExpectLaunch(absl::string_view hlo, int64_t num_blocks,
                    std::optional<Interval> vector_range = std::nullopt) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    se::DeviceDescription device_info = TestGpuDeviceInfo::AMDMI350DeviceInfo();
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(
        *module->entry_computation()->root_instruction(), device_info);
    ASSERT_EQ(analysis.emitter_fusion_kind(),
              HloFusionAnalysis::EmitterFusionKind::kTranspose);
    std::unique_ptr<MlirKernelEmitter> emitter =
        CreateTransposeFusion(analysis);
    LaunchDimensions launch_dimensions = emitter->launch_dimensions();
    EXPECT_EQ(launch_dimensions.num_blocks(), num_blocks);
    // kNumRowsRocm x kTileSize threads, whatever the vector size.
    EXPECT_EQ(launch_dimensions.num_threads_per_block(), 256);
    if (vector_range.has_value()) {
      std::optional<IndexingMap> indexing =
          emitter->ComputeThreadIdToOutputIndexing(/*root_index=*/1,
                                                   &mlir_context_);
      ASSERT_TRUE(indexing.has_value());
      EXPECT_THAT(
          indexing->GetRangeVars(),
          Contains(Field(&IndexingMap::Variable::bounds, *vector_range)))
          << indexing->ToString();
    }
  }

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

}  // namespace
}  // namespace xla::gpu
