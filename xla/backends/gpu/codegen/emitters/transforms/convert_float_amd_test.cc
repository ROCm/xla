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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

// The IR of each form: one element per call, or N elements per vector.
constexpr absl::string_view kScalarIr = R"(
CHECK-NOT: x i4>
CHECK: [[WORD:%[0-9]+]] = call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float [[X:%[0-9]+]], float [[X]],
CHECK-NEXT: trunc i32 [[WORD]] to i4
CHECK-NOT: x i4>
)";
constexpr absl::string_view kVector2Ir =
    "CHECK: bitcast i8 %{{[0-9]+}} to <2 x i4>";
constexpr absl::string_view kVector4Ir =
    "CHECK: bitcast i16 %{{[0-9]+}} to <4 x i4>";
constexpr absl::string_view kVector8Ir = R"(
CHECK: call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %{{[0-9]+}}, {{.*}}, i32 3)
CHECK-NEXT: bitcast i32 %{{[0-9]+}} to <8 x i4>
)";
constexpr absl::string_view kVector16Ir =
    "CHECK: bitcast <2 x i32> %{{[0-9]+}} to <16 x i4>";

class ConvertFloatAmdTest
    : public HloInterpreterReferenceMixin<GpuPjRtCodegenTest> {
 protected:
  void SetUp() override {
    const se::RocmComputeCapability* cc =
        device_description().gpu_compute_capability().rocm_compute_capability();
    if (cc == nullptr || !cc->gfx9_mi350()) {
      GTEST_SKIP() << "Requires gfx950.";
    }
  }

  // Runs the emitter whose IR the test checks, not a Triton fusion instead.
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = HloInterpreterReferenceMixin<
        GpuPjRtCodegenTest>::GetDebugOptionsForTest();
    options.set_xla_gpu_experimental_enable_fusion_autotuner(false);
    return options;
  }

  // Bitwise as the interpreter, but NaN gives 0x8 as in ExpandFloatOps.
  void ExpectSameAsInterpreter(absl::string_view hlo, const Literal& input,
                               absl::string_view expected_ir,
                               absl::Span<const int64_t> transpose = {}) {
    EXPECT_OK(CompileAndVerifyIr(hlo, expected_ir));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> reference_module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input}));
    ASSERT_OK_AND_ASSIGN(
        Literal expected,
        reference_runner().Execute(std::move(reference_module), {&input},
                                   /*run_hlo_passes=*/true));
    ASSERT_OK_AND_ASSIGN(Literal source, input.Convert(F32));
    if (!transpose.empty()) {
      source = source.Transpose(transpose).Relayout(
          LayoutUtil::GetDefaultLayoutForRank(transpose.size()));
    }
    absl::Span<const float> x = source.data<float>();
    absl::Span<const tsl::float4_e2m1fn> got =
        result.data<tsl::float4_e2m1fn>();
    absl::Span<const tsl::float4_e2m1fn> want =
        expected.data<tsl::float4_e2m1fn>();
    int64_t mismatches = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      uint8_t got_bits = absl::bit_cast<uint8_t>(got[i]);
      uint8_t want_bits =
          std::isnan(x[i]) ? 0x8 : absl::bit_cast<uint8_t>(want[i]);
      if (got_bits != want_bits && ++mismatches <= 8) {
        ADD_FAILURE() << absl::StrFormat(
            "input %g (0x%08x): got 0x%x, want 0x%x", x[i],
            absl::bit_cast<uint32_t>(x[i]), got_bits, want_bits);
      }
    }
    EXPECT_EQ(mismatches, 0);
  }
};

// Each 16-bit pattern once per 2^16 values, rotated by one per repetition.
template <typename T>
std::vector<T> All16BitValues(int64_t size) {
  std::vector<T> values(size);
  for (int64_t i = 0; i < size; ++i) {
    values[i] = absl::bit_cast<T>(
        static_cast<uint16_t>(i % (1 << 16) + i / (1 << 16)));
  }
  return values;
}

TEST_F(ConvertFloatAmdTest, Bf16ToF4E2M1FNAllValues) {
  Literal input =
      LiteralUtil::CreateR1<bfloat16>(All16BitValues<bfloat16>(1 << 16));
  ExpectSameAsInterpreter(R"hlo(
    ENTRY e {
      p = bf16[65536] parameter(0)
      ROOT c = f4e2m1fn[65536] convert(p)
    })hlo",
                          input, kVector2Ir);
}

TEST_F(ConvertFloatAmdTest, Bf16ToF4E2M1FNOddLength) {
  Literal input =
      LiteralUtil::CreateR1<bfloat16>(All16BitValues<bfloat16>((1 << 16) - 1));
  ExpectSameAsInterpreter(R"hlo(
    ENTRY e {
      p = bf16[65535] parameter(0)
      ROOT c = f4e2m1fn[65535] convert(p)
    })hlo",
                          input, kScalarIr);
}

TEST_F(ConvertFloatAmdTest, F16ToF4E2M1FNAllValues) {
  std::vector<half> values(1 << 16);
  for (int i = 0; i < (1 << 16); ++i) {
    values[i] = absl::bit_cast<half>(static_cast<uint16_t>(i));
  }
  Literal input = LiteralUtil::CreateR1<half>(values);
  ExpectSameAsInterpreter(R"hlo(
    ENTRY e {
      p = f16[65536] parameter(0)
      ROOT c = f4e2m1fn[65536] convert(p)
    })hlo",
                          input, kVector2Ir);
}

// Boundary values, then random bit patterns and random values in 2^-4..2^4.
std::vector<float> F32Inputs() {
  constexpr float kE2M1Values[] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
  std::vector<float> values;
  for (float sign : {1.0f, -1.0f}) {
    for (int i = 0; i < 8; ++i) {
      values.push_back(sign * kE2M1Values[i]);
    }
    for (int i = 0; i < 7; ++i) {
      float midpoint = (kE2M1Values[i] + kE2M1Values[i + 1]) / 2;
      values.push_back(sign * midpoint);
      values.push_back(sign * midpoint * (1.0f + 0x1p-20f));
      values.push_back(sign * midpoint * (1.0f - 0x1p-20f));
    }
    for (float v : {std::nextafter(6.0f, 7.0f), std::nextafter(7.0f, 6.0f),
                    7.0f, std::nextafter(7.0f, 8.0f), 8.0f, 1e30f,
                    std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::denorm_min(),
                    absl::bit_cast<float>(0x00400000u),
                    absl::bit_cast<float>(0x007FFFFFu),
                    std::numeric_limits<float>::min(),
                    static_cast<float>(std::numeric_limits<bfloat16>::max()),
                    static_cast<float>(std::numeric_limits<bfloat16>::min()),
                    static_cast<float>(
                        std::numeric_limits<bfloat16>::denorm_min())}) {
      values.push_back(sign * v);
    }
  }
  for (uint32_t nan : {0x7FC00000u, 0x7F800001u, 0x7FA00000u, 0x7FFFFFFFu,
                       0xFFC00000u, 0xFF800001u, 0xFFA00000u, 0xFFFFFFFFu}) {
    values.push_back(absl::bit_cast<float>(nan));
  }
  std::mt19937 rng(42);
  for (int i = 0; i < (1 << 21); ++i) {
    values.push_back(absl::bit_cast<float>(static_cast<uint32_t>(rng())));
  }
  std::uniform_int_distribution<uint32_t> biased_exponent(127 - 4, 127 + 3);
  for (int i = 0; i < (1 << 21); ++i) {
    uint32_t sign_and_mantissa = static_cast<uint32_t>(rng()) & 0x807FFFFFu;
    values.push_back(absl::bit_cast<float>(sign_and_mantissa |
                                           (biased_exponent(rng) << 23)));
  }
  return values;
}

TEST_F(ConvertFloatAmdTest, F32ToF4E2M1FN) {
  std::vector<float> values = F32Inputs();
  Literal input = LiteralUtil::CreateR1<float>(values);
  ExpectSameAsInterpreter(absl::StrFormat(R"hlo(
    ENTRY e {
      p = f32[%d] parameter(0)
      ROOT c = f4e2m1fn[%d] convert(p)
    })hlo",
                                          values.size(), values.size()),
                          input, kVector4Ir);
}

TEST_F(ConvertFloatAmdTest, Bf16ToF4E2M1FNPacked) {
  std::vector<bfloat16> values(8 * 64);
  for (int i = 0; i < 8 * 64; ++i) {
    values[i] = static_cast<bfloat16>(static_cast<float>(i % 29) / 4 - 3.5f);
  }
  ASSERT_OK_AND_ASSIGN(
      Literal input, LiteralUtil::CreateR1<bfloat16>(values).Reshape({8, 64}));
  ExpectSameAsInterpreter(R"hlo(
    ENTRY e {
      p = bf16[8,64] parameter(0)
      ROOT c = f4e2m1fn[8,64] convert(p)
    })hlo",
                          input, kVector2Ir);
}

// A transpose that keeps the minor dimension stores it as one vector.
TEST_F(ConvertFloatAmdTest, Bf16ToF4E2M1FNVector8) {
  ASSERT_OK_AND_ASSIGN(
      Literal input,
      LiteralUtil::CreateR1<bfloat16>(All16BitValues<bfloat16>(8 << 16))
          .Reshape({512, 128, 8}));
  ExpectSameAsInterpreter(R"hlo(
    ENTRY e {
      p = bf16[512,128,8] parameter(0)
      t = bf16[128,512,8] transpose(p), dimensions={1,0,2}
      ROOT c = f4e2m1fn[128,512,8] convert(t)
    })hlo",
                          input, kVector8Ir, /*transpose=*/{1, 0, 2});
}

TEST_F(ConvertFloatAmdTest, Bf16ToF4E2M1FNVector16) {
  ASSERT_OK_AND_ASSIGN(
      Literal input,
      LiteralUtil::CreateR1<bfloat16>(All16BitValues<bfloat16>(16 << 16))
          .Reshape({512, 128, 16}));
  ExpectSameAsInterpreter(R"hlo(
    ENTRY e {
      p = bf16[512,128,16] parameter(0)
      t = bf16[128,512,16] transpose(p), dimensions={1,0,2}
      ROOT c = f4e2m1fn[128,512,16] convert(t)
    })hlo",
                          input, kVector16Ir, /*transpose=*/{1, 0, 2});
}

constexpr absl::string_view kFp8ScalarIr = R"(
CHECK-NOT: x i8>
CHECK: [[WORD:%[0-9]+]] = call i32 @llvm.amdgcn.cvt.pk.{{fp8|bf8}}.f32(float %{{[0-9]+}}, float undef,
CHECK-NEXT: trunc i32 [[WORD]] to i8
CHECK-NOT: x i8>
)";

struct Fp8PairCase {
  absl::string_view src;
  absl::string_view dst;
  int vector_size;
  absl::string_view vector_ir;
};

class ConvertFloatAmdFp8PairTest
    : public ConvertFloatAmdTest,
      public ::testing::WithParamInterface<Fp8PairCase> {
 protected:
  // Paired conversions in a transpose that keeps the minor dimension against
  // the single conversions of an odd-length convert, for every value and lane.
  template <typename T>
  void ExpectPairedSameAsSingle(const Fp8PairCase& c) {
    int n = c.vector_size;
    ASSERT_OK_AND_ASSIGN(
        Literal input,
        LiteralUtil::CreateR1<T>(All16BitValues<T>(n << 16))
            .Reshape({512, 128, n}));
    std::string paired_hlo = absl::StrFormat(R"hlo(
      ENTRY e {
        p = %1$s[512,128,%3$d] parameter(0)
        t = %1$s[128,512,%3$d] transpose(p), dimensions={1,0,2}
        ROOT c = %2$s[128,512,%3$d] convert(t)
      })hlo",
                                             c.src, c.dst, n);
    EXPECT_OK(CompileAndVerifyIr(paired_hlo, c.vector_ir));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> paired_module,
                         ParseAndReturnVerifiedModule(paired_hlo));
    ASSERT_OK_AND_ASSIGN(Literal paired,
                         Execute(std::move(paired_module), {&input}));

    Literal transposed = input.Transpose({1, 0, 2}).Relayout(
        LayoutUtil::GetDefaultLayoutForRank(3));
    std::vector<T> values(transposed.data<T>().begin(),
                          transposed.data<T>().end());
    values.push_back(T(0));
    Literal single_input = LiteralUtil::CreateR1<T>(values);
    std::string single_hlo = absl::StrFormat(R"hlo(
      ENTRY e {
        p = %1$s[%3$d] parameter(0)
        ROOT c = %2$s[%3$d] convert(p)
      })hlo",
                                             c.src, c.dst, values.size());
    EXPECT_OK(CompileAndVerifyIr(single_hlo, kFp8ScalarIr));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> single_module,
                         ParseAndReturnVerifiedModule(single_hlo));
    ASSERT_OK_AND_ASSIGN(Literal single,
                         Execute(std::move(single_module), {&single_input}));

    const auto* got = static_cast<const uint8_t*>(paired.untyped_data());
    const auto* want = static_cast<const uint8_t*>(single.untyped_data());
    int64_t mismatches = 0;
    for (int64_t i = 0; i < paired.element_count(); ++i) {
      if (got[i] != want[i] && ++mismatches <= 8) {
        ADD_FAILURE() << absl::StrFormat(
            "input 0x%04x at %d: paired 0x%02x, single 0x%02x",
            absl::bit_cast<uint16_t>(values[i]), i, got[i], want[i]);
      }
    }
    EXPECT_EQ(mismatches, 0);
  }
};

TEST_P(ConvertFloatAmdFp8PairTest, PairedSameAsSingle) {
  if (GetParam().src == "bf16") {
    ExpectPairedSameAsSingle<bfloat16>(GetParam());
  } else {
    ExpectPairedSameAsSingle<half>(GetParam());
  }
}

constexpr absl::string_view kFp8Vector2Ir =
    "CHECK: bitcast i16 %{{[0-9]+}} to <2 x i8>";
constexpr absl::string_view kFp8Vector4Ir =
    "CHECK: bitcast <1 x i32> %{{[0-9]+}} to <4 x i8>";
constexpr absl::string_view kFp8Vector8Ir =
    "CHECK: bitcast <2 x i32> %{{[0-9]+}} to <8 x i8>";

INSTANTIATE_TEST_SUITE_P(
    Fp8Pairs, ConvertFloatAmdFp8PairTest,
    ::testing::ValuesIn(std::vector<Fp8PairCase>{
        {"bf16", "f8e4m3fn", 2, kFp8Vector2Ir},
        {"bf16", "f8e4m3fn", 4, kFp8Vector4Ir},
        {"bf16", "f8e4m3fn", 8, kFp8Vector8Ir},
        {"bf16", "f8e5m2", 2, kFp8Vector2Ir},
        {"bf16", "f8e5m2", 4, kFp8Vector4Ir},
        {"bf16", "f8e5m2", 8, kFp8Vector8Ir},
        {"f16", "f8e4m3fn", 2, kFp8Vector2Ir},
        {"f16", "f8e4m3fn", 4, kFp8Vector4Ir},
        {"f16", "f8e4m3fn", 8, kFp8Vector8Ir},
        {"f16", "f8e5m2", 2, kFp8Vector2Ir},
        {"f16", "f8e5m2", 4, kFp8Vector4Ir},
        {"f16", "f8e5m2", 8, kFp8Vector8Ir},
    }),
    [](const ::testing::TestParamInfo<Fp8PairCase>& info) {
      return absl::StrFormat("%s_%s_%d", info.param.src, info.param.dst,
                             info.param.vector_size);
    });

}  // namespace
}  // namespace xla::gpu
