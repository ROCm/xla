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

// End-to-end tests of the Triton TopK kernels on ROCm. Every test forces the
// Triton path and checks that the compiled module took it, then compares the
// results bit for bit: with today's path (flag off), with upstream's u64
// pack -> select -> unpack, or with a host reference.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/backends/gpu/codegen/triton/topk_config.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

// A TopK of `k` over the last dimension of a [batch, n] input; batch 0 means
// a rank-1 input of n elements.
struct TopKShape {
  int64_t batch;
  int64_t n;
  int64_t k;
};

std::string Dims(int64_t batch, int64_t last) {
  return batch == 0 ? absl::StrCat(last) : absl::StrCat(batch, ",", last);
}

std::string TypeName(PrimitiveType type) {
  return primitive_util::LowercasePrimitiveTypeName(type);
}

// A TopK HLO op ("TOTAL" order) or a "TopK" custom call with `order`.
std::string TopKHlo(PrimitiveType type, const TopKShape& shape,
                    absl::string_view order, bool stable) {
  std::string in =
      absl::StrCat(TypeName(type), "[", Dims(shape.batch, shape.n), "]");
  std::string out =
      absl::StrCat("(", TypeName(type), "[", Dims(shape.batch, shape.k),
                   "], s32[", Dims(shape.batch, shape.k), "])");
  std::string topk =
      order == "TOTAL"
          ? absl::Substitute("topk(x), k=$0, largest=true, is_stable=$1",
                             shape.k, stable ? "true" : "false")
          : absl::Substitute(R"(custom-call(x), custom_call_target="TopK", )"
                             R"(backend_config={is_stable = $0, order = "$1"})",
                             stable ? "true" : "false", order);
  return absl::Substitute(R"(
    HloModule topk
    ENTRY main {
      x = $0 parameter(0)
      ROOT t = $1 $2
    })",
                          in, out, topk);
}

class TritonTopKTest : public HloPjRtGpuTestBase {
 protected:
  void SetUp() override {
    HloPjRtGpuTestBase::SetUp();
    if (!device_description().gpu_compute_capability().IsRocm()) {
      GTEST_SKIP() << "The Triton TopK kernels are used on ROCm only.";
    }
  }

  HloModuleConfig ConfigWithTritonTopK(bool triton) const {
    HloModuleConfig config = GetModuleConfigForTest();
    DebugOptions& options = config.mutable_debug_options();
    options.set_xla_gpu_experimental_enable_triton_topk(triton);
    options.set_xla_gpu_experimental_force_triton_topk(triton);
    return config;
  }

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> Compile(
      absl::string_view hlo, const HloModuleConfig& config) {
    ABSL_ASSIGN_OR_RETURN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
    return CreateExecutable(std::move(module), /*run_hlo_passes=*/true);
  }

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CompileWithTritonTopK(
      absl::string_view hlo) {
    return CompileWithTritonTopK(hlo, ConfigWithTritonTopK(true));
  }

  // Compiles `hlo` with `config`, which forces the Triton TopK kernels, and
  // checks that all of its `num_topks` TopKs were routed to them and that no
  // sort is left.
  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CompileWithTritonTopK(
      absl::string_view hlo, const HloModuleConfig& config, int num_topks = 1) {
    ABSL_ASSIGN_OR_RETURN(auto executable, Compile(hlo, config));
    ABSL_ASSIGN_OR_RETURN(const HloModule* optimized,
                          test_runner().HloModuleFromWrapped(executable.get()));
    int triton_topks = 0;
    for (const HloComputation* computation : optimized->computations()) {
      for (const HloInstruction* instr : computation->instructions()) {
        if (instr->opcode() == HloOpcode::kSort) {
          return absl::InternalError(
              absl::StrCat("Sort left in the module: ", instr->ToString()));
        }
        if (instr->opcode() == HloOpcode::kCustomCall &&
            IsTritonTopK(*Cast<HloCustomCallInstruction>(instr))) {
          ++triton_topks;
        }
      }
    }
    if (triton_topks != num_topks) {
      return absl::InternalError(
          absl::StrCat(triton_topks, " TopKs routed to Triton, expected ",
                       num_topks, ":\n", optimized->ToString()));
    }
    return executable;
  }

  absl::StatusOr<Literal> Run(OpaqueExecutable* executable,
                              absl::Span<const Literal* const> arguments) {
    return test_runner().ExecuteWithExecutable(executable, arguments);
  }

  absl::StatusOr<std::vector<Literal>> FakeArguments(absl::string_view hlo) {
    ABSL_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo));
    return MakeFakeArguments(module.get());
  }
};

std::vector<const Literal*> Pointers(const std::vector<Literal>& literals) {
  std::vector<const Literal*> pointers;
  for (const Literal& literal : literals) {
    pointers.push_back(&literal);
  }
  return pointers;
}

// Unsigned integer of the same width as a float type.
template <typename T>
using BitsOf = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;

// The f32 bit pattern of an f32 or bf16 value.
template <typename T>
uint32_t F32Bits(T value) {
  if constexpr (sizeof(T) == 4) {
    return absl::bit_cast<uint32_t>(value);
  } else {
    return uint32_t{absl::bit_cast<uint16_t>(value)} << 16;
  }
}

// Sort key of XLA's TOTAL order (larger key = larger value); PARTIAL order
// treats -0 as +0.
uint32_t OrderKey(uint32_t f32_bits, bool partial) {
  if (partial && f32_bits == 0x80000000u) {
    f32_bits = 0;
  }
  return (f32_bits & 0x80000000u) ? ~f32_bits : (f32_bits | 0x80000000u);
}

// Host reference: the `k` largest elements of each row, ties by lower index.
template <typename T>
Literal ReferenceTopK(const Array2D<T>& input, int64_t k, bool partial) {
  const int64_t batch = input.n1(), n = input.n2();
  Array2D<T> values(batch, k);
  Array2D<int32_t> indices(batch, k);
  std::vector<int32_t> order(n);
  for (int64_t row = 0; row < batch; ++row) {
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
      return OrderKey(F32Bits(input(row, a)), partial) >
             OrderKey(F32Bits(input(row, b)), partial);
    });
    for (int64_t j = 0; j < k; ++j) {
      values(row, j) = input(row, order[j]);
      indices(row, j) = order[j];
    }
  }
  return LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateR2FromArray2D<T>(values),
      LiteralUtil::CreateR2FromArray2D<int32_t>(indices));
}

// Checks the contract of an unstable TopK (design 4.7) against a stable
// reference: the same value bits in order, and every index valid, unique in
// its row and pointing to an element with exactly the returned value bits.
template <typename T>
void ExpectValidUnstableTopK(const Literal& input, const Literal& reference,
                             const Literal& actual, const TopKShape& shape) {
  const int64_t rows = std::max<int64_t>(shape.batch, 1);
  absl::Span<const T> in = input.data<T>();
  absl::Span<const T> ref_values = reference.data<T>({0});
  absl::Span<const T> values = actual.data<T>({0});
  absl::Span<const int32_t> indices = actual.data<int32_t>({1});
  for (int64_t row = 0; row < rows; ++row) {
    absl::flat_hash_set<int32_t> seen;
    for (int64_t j = 0; j < shape.k; ++j) {
      const int64_t pos = row * shape.k + j;
      ASSERT_EQ(absl::bit_cast<BitsOf<T>>(values[pos]),
                absl::bit_cast<BitsOf<T>>(ref_values[pos]))
          << "row " << row << " position " << j;
      const int32_t index = indices[pos];
      ASSERT_GE(index, 0);
      ASSERT_LT(index, shape.n);
      ASSERT_TRUE(seen.insert(index).second)
          << "row " << row << " repeats index " << index;
      ASSERT_EQ(absl::bit_cast<BitsOf<T>>(in[row * shape.n + index]),
                absl::bit_cast<BitsOf<T>>(values[pos]))
          << "row " << row << " index " << index;
    }
  }
}

// ---- L5: flag on vs flag off on random inputs ------------------------------

struct GridCase {
  PrimitiveType type;
  TopKShape shape;
};

std::vector<GridCase> GridCases() {
  const std::vector<std::pair<int64_t, int64_t>> batch_and_n = {
      {0, 7},     {0, 1000}, {0, 16384}, {3, 64},  {129, 33},
      {17, 1024}, {5, 4096}, {2, 16384}, {1000, 8}};
  std::vector<GridCase> cases;
  for (PrimitiveType type : {F32, BF16}) {
    for (auto [batch, n] : batch_and_n) {
      for (int64_t k : {1, 2, 6, 16, 17, 33, 64, 128}) {
        if (static_cast<double>(k) / n < 0.85) {
          cases.push_back({type, {batch, n, k}});
        }
      }
    }
  }
  return cases;
}

std::string GridCaseName(const ::testing::TestParamInfo<GridCase>& info) {
  const GridCase& c = info.param;
  return absl::StrCat(TypeName(c.type), "_b", c.shape.batch, "_n", c.shape.n,
                      "_k", c.shape.k);
}

class TritonTopKGridTest : public TritonTopKTest,
                           public ::testing::WithParamInterface<GridCase> {};

TEST_P(TritonTopKGridTest, StableMatchesTodaysPathBitwise) {
  const GridCase& c = GetParam();
  const std::string hlo = TopKHlo(c.type, c.shape, "TOTAL", /*stable=*/true);
  ASSERT_OK_AND_ASSIGN(auto triton, CompileWithTritonTopK(hlo));
  ASSERT_OK_AND_ASSIGN(auto today, Compile(hlo, ConfigWithTritonTopK(false)));
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args, FakeArguments(hlo));
  ASSERT_OK_AND_ASSIGN(Literal expected, Run(today.get(), Pointers(args)));
  ASSERT_OK_AND_ASSIGN(Literal actual, Run(triton.get(), Pointers(args)));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

TEST_P(TritonTopKGridTest, UnstableIsValidAndDeterministic) {
  const GridCase& c = GetParam();
  const std::string hlo = TopKHlo(c.type, c.shape, "TOTAL", /*stable=*/false);
  ASSERT_OK_AND_ASSIGN(auto triton, CompileWithTritonTopK(hlo));
  ASSERT_OK_AND_ASSIGN(auto today, Compile(hlo, ConfigWithTritonTopK(false)));
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args, FakeArguments(hlo));
  ASSERT_OK_AND_ASSIGN(Literal reference, Run(today.get(), Pointers(args)));
  ASSERT_OK_AND_ASSIGN(Literal first, Run(triton.get(), Pointers(args)));
  ASSERT_OK_AND_ASSIGN(Literal second, Run(triton.get(), Pointers(args)));
  EXPECT_TRUE(LiteralTestUtil::Equal(first, second));
  if (c.type == F32) {
    ExpectValidUnstableTopK<float>(args[0], reference, first, c.shape);
  } else {
    ExpectValidUnstableTopK<bfloat16>(args[0], reference, first, c.shape);
  }
}

INSTANTIATE_TEST_SUITE_P(TritonTopKGrid, TritonTopKGridTest,
                         ::testing::ValuesIn(GridCases()), GridCaseName);

// ---- L5: inside a while body and inside a command buffer -------------------

constexpr absl::string_view kTopKInWhileHlo = R"(
  HloModule topk_in_while

  cond {
    p = (s32[], s32[], f32[16,64], f32[16,6], s32[16,6]) parameter(0)
    i = s32[] get-tuple-element(p), index=0
    trips = s32[] get-tuple-element(p), index=1
    ROOT lt = pred[] compare(i, trips), direction=LT
  }

  body {
    p = (s32[], s32[], f32[16,64], f32[16,6], s32[16,6]) parameter(0)
    i = s32[] get-tuple-element(p), index=0
    trips = s32[] get-tuple-element(p), index=1
    x = f32[16,64] get-tuple-element(p), index=2
    t = (f32[16,6], s32[16,6]) topk(x), k=6, largest=true, is_stable=true
    v = f32[16,6] get-tuple-element(t), index=0
    idx = s32[16,6] get-tuple-element(t), index=1
    one = s32[] constant(1)
    next = s32[] add(i, one)
    neg = f32[16,64] negate(x)
    ROOT r = (s32[], s32[], f32[16,64], f32[16,6], s32[16,6])
        tuple(next, trips, neg, v, idx)
  }

  ENTRY main {
    x = f32[16,64] parameter(0)
    trips = s32[] parameter(1)
    zero = s32[] constant(0)
    fzero = f32[] constant(0)
    v0 = f32[16,6] broadcast(fzero), dimensions={}
    i0 = s32[16,6] broadcast(zero), dimensions={}
    init = (s32[], s32[], f32[16,64], f32[16,6], s32[16,6])
        tuple(zero, trips, x, v0, i0)
    w = (s32[], s32[], f32[16,64], f32[16,6], s32[16,6]) while(init),
        condition=cond, body=body
    v = f32[16,6] get-tuple-element(w), index=3
    idx = s32[16,6] get-tuple-element(w), index=4
    ROOT out = (f32[16,6], s32[16,6]) tuple(v, idx)
  })";

TEST_F(TritonTopKTest, TopKInWhileBody) {
  ASSERT_OK_AND_ASSIGN(auto triton, CompileWithTritonTopK(kTopKInWhileHlo));
  ASSERT_OK_AND_ASSIGN(const HloModule* optimized,
                       test_runner().HloModuleFromWrapped(triton.get()));
  bool topk_in_while_body = false;
  for (const HloComputation* computation : optimized->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kWhile) {
        continue;
      }
      for (const HloInstruction* body_instr :
           instr->while_body()->instructions()) {
        topk_in_while_body |=
            body_instr->opcode() == HloOpcode::kCustomCall &&
            IsTritonTopK(*Cast<HloCustomCallInstruction>(body_instr));
      }
    }
  }
  EXPECT_TRUE(topk_in_while_body) << optimized->ToString();

  ASSERT_OK_AND_ASSIGN(auto today,
                       Compile(kTopKInWhileHlo, ConfigWithTritonTopK(false)));
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       FakeArguments(kTopKInWhileHlo));
  args[1] = LiteralUtil::CreateR0<int32_t>(3);
  ASSERT_OK_AND_ASSIGN(Literal expected, Run(today.get(), Pointers(args)));
  ASSERT_OK_AND_ASSIGN(Literal actual, Run(triton.get(), Pointers(args)));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

// True if a kernel thunk of the dumped thunk sequence is nested in a top-level
// command buffer thunk.
bool HasKernelInCommandBuffer(absl::string_view dump) {
  bool in_command_buffer = false;
  for (absl::string_view line : absl::StrSplit(dump, '\n')) {
    absl::string_view trimmed = absl::StripLeadingAsciiWhitespace(line);
    if (trimmed.empty()) {
      continue;
    }
    if (trimmed.data() == line.data()) {
      in_command_buffer = absl::StrContains(trimmed, "kCommandBuffer");
    } else if (in_command_buffer &&
               absl::StrContains(trimmed, "kCustomKernel")) {
      return true;
    }
  }
  return false;
}

TEST_F(TritonTopKTest, TopKInCommandBuffer) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  HloModuleConfig config = ConfigWithTritonTopK(true);
  config.mutable_debug_options().set_xla_dump_to(dump_dir.path());
  config.mutable_debug_options().set_xla_gpu_graph_min_graph_size(1);
  const std::string hlo = TopKHlo(F32, {1024, 64, 6}, "TOTAL", /*stable=*/true);
  ASSERT_OK_AND_ASSIGN(auto triton, CompileWithTritonTopK(hlo, config));

  std::vector<std::string> dump_files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_dir.path(),
                        "*thunk_sequence_after_thunk_passes*.txt"),
      &dump_files));
  ASSERT_GE(dump_files.size(), 1);
  std::string dump;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), dump_files[0], &dump));
  EXPECT_TRUE(HasKernelInCommandBuffer(dump)) << dump;

  ASSERT_OK_AND_ASSIGN(auto today, Compile(hlo, ConfigWithTritonTopK(false)));
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args, FakeArguments(hlo));
  ASSERT_OK_AND_ASSIGN(Literal expected, Run(today.get(), Pointers(args)));
  // Runs 1 and 2 record the command buffer; run 3 updates it with new buffers.
  for (int run = 0; run < 3; ++run) {
    std::vector<Literal> copies;
    for (const Literal& arg : args) {
      copies.push_back(arg.Clone());
    }
    ASSERT_OK_AND_ASSIGN(Literal actual, Run(triton.get(), Pointers(copies)));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual)) << "run " << run;
  }
}

// ---- Special-value inputs (L6, L7) ----------------------------------------

enum class Pattern { kSpecial, kTies, kAllEqual };

// f32 bit patterns of the special values of topk_special_floats_test.cc, plus
// signaling NaNs of both signs.
constexpr uint32_t kSpecialF32[] = {
    0xFF800000u, 0x7F800000u, 0x80100000u, 0x00000000u, 0x7FC00000u,
    0x7FE00001u, 0x80000000u, 0xC2810000u, 0x7F800000u, 0xFFC00000u,
    0xFFE00001u, 0xC2810000u, 0x80000000u, 0x00020000u, 0x80010000u,
    0xFFE00000u, 0x00000000u, 0xFF800000u, 0x42810000u, 0x7FE00000u,
    0x3C4D0000u, 0x00700000u, 0x80100000u, 0x00020000u, 0x80010000u,
    0x7F800001u, 0xFF800001u, 0x7FA00000u, 0xFFA00000u};
// The same values as bf16 (upper halves), with bf16 signaling NaNs.
constexpr uint16_t kSpecialBf16[] = {
    0xFF80u, 0x7F80u, 0x8010u, 0x0000u, 0x7FC0u, 0x7FE1u, 0x8000u, 0xC281u,
    0x7F80u, 0xFFC0u, 0xFFE1u, 0xC281u, 0x8000u, 0x0002u, 0x8001u, 0xFFE0u,
    0x0000u, 0xFF80u, 0x4281u, 0x7FE0u, 0x3C4Du, 0x0070u, 0x8010u, 0x0002u,
    0x8001u, 0x7F81u, 0xFF81u, 0x7FA0u, 0xFFA0u};

bool IsSignalingNan(uint32_t f32_bits) {
  return (f32_bits & 0x7F800000u) == 0x7F800000u &&
         (f32_bits & 0x007FFFFFu) != 0 && (f32_bits & 0x00400000u) == 0;
}

template <typename T>
T FromBits(uint32_t bits) {
  return absl::bit_cast<T>(static_cast<BitsOf<T>>(bits));
}

// Rows start with a rotation of the special values and are filled with the
// smallest NaN (as in topk_special_floats_test.cc), or hold few distinct
// values with ±0, or one value.
template <typename T>
Array2D<T> MakeInput(Pattern pattern, int64_t batch, int64_t n,
                     bool with_signaling_nans) {
  std::vector<uint32_t> special;
  if constexpr (sizeof(T) == 4) {
    special.assign(std::begin(kSpecialF32), std::end(kSpecialF32));
  } else {
    special.assign(std::begin(kSpecialBf16), std::end(kSpecialBf16));
  }
  if (!with_signaling_nans) {
    special.erase(std::remove_if(special.begin(), special.end(),
                                 [](uint32_t bits) {
                                   return IsSignalingNan(
                                       sizeof(T) == 4 ? bits : bits << 16);
                                 }),
                  special.end());
  }
  const uint32_t smallest_nan = sizeof(T) == 4 ? 0xFFFFFFFFu : 0xFFFFu;
  const uint32_t one = sizeof(T) == 4 ? 0x3F800000u : 0x3F80u;
  const uint32_t neg_zero = sizeof(T) == 4 ? 0x80000000u : 0x8000u;
  const uint32_t ties[] = {one, 0, neg_zero, one, 0};
  Array2D<T> input(batch, n);
  for (int64_t row = 0; row < batch; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      uint32_t bits = smallest_nan;
      switch (pattern) {
        case Pattern::kSpecial:
          if (col < static_cast<int64_t>(special.size())) {
            bits = special[(col + row) % special.size()];
          }
          break;
        case Pattern::kTies:
          bits = ties[(col * 7 + row) % 5];
          break;
        case Pattern::kAllEqual:
          bits = neg_zero;
          break;
      }
      input(row, col) = FromBits<T>(bits);
    }
  }
  return input;
}

struct SpecialCase {
  PrimitiveType type;
  bool partial;
  Pattern pattern;
  TopKShape shape;
};

std::vector<SpecialCase> SpecialCases(bool all_patterns) {
  std::vector<SpecialCase> cases;
  for (PrimitiveType type : {F32, BF16}) {
    for (bool partial : {false, true}) {
      for (Pattern pattern :
           {Pattern::kSpecial, Pattern::kTies, Pattern::kAllEqual}) {
        if (!all_patterns && pattern != Pattern::kSpecial) {
          continue;
        }
        for (auto [batch, n] : std::vector<std::pair<int64_t, int64_t>>{
                 {1, 32}, {8, 1024}, {3, 16384}}) {
          for (int64_t k : {2, 17, 64}) {
            if (static_cast<double>(k) / n < 0.85) {
              cases.push_back({type, partial, pattern, {batch, n, k}});
            }
          }
        }
      }
    }
  }
  return cases;
}

std::string SpecialCaseName(const ::testing::TestParamInfo<SpecialCase>& info) {
  const SpecialCase& c = info.param;
  const char* pattern = c.pattern == Pattern::kSpecial ? "special"
                        : c.pattern == Pattern::kTies  ? "ties"
                                                       : "all_equal";
  return absl::StrCat(TypeName(c.type), c.partial ? "_partial" : "_total", "_",
                      pattern, "_b", c.shape.batch, "_n", c.shape.n, "_k",
                      c.shape.k);
}

template <typename T>
Literal MakeInputLiteral(const SpecialCase& c, bool with_signaling_nans) {
  return LiteralUtil::CreateR2FromArray2D<T>(
      MakeInput<T>(c.pattern, c.shape.batch, c.shape.n, with_signaling_nans));
}

// ---- L7: special values against the host reference -------------------------

class TritonTopKSpecialValuesTest
    : public TritonTopKTest,
      public ::testing::WithParamInterface<SpecialCase> {};

TEST_P(TritonTopKSpecialValuesTest, MatchesHostReferenceBitwise) {
  const SpecialCase& c = GetParam();
  const std::string hlo =
      TopKHlo(c.type, c.shape, c.partial ? "PARTIAL" : "TOTAL",
              /*stable=*/true);
  ASSERT_OK_AND_ASSIGN(auto triton, CompileWithTritonTopK(hlo));
  Literal input, expected;
  if (c.type == F32) {
    Array2D<float> array =
        MakeInput<float>(c.pattern, c.shape.batch, c.shape.n, true);
    input = LiteralUtil::CreateR2FromArray2D<float>(array);
    expected = ReferenceTopK(array, c.shape.k, c.partial);
  } else {
    Array2D<bfloat16> array =
        MakeInput<bfloat16>(c.pattern, c.shape.batch, c.shape.n, true);
    input = LiteralUtil::CreateR2FromArray2D<bfloat16>(array);
    expected = ReferenceTopK(array, c.shape.k, c.partial);
  }
  ASSERT_OK_AND_ASSIGN(Literal actual, Run(triton.get(), {&input}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

INSTANTIATE_TEST_SUITE_P(TritonTopKSpecialValues, TritonTopKSpecialValuesTest,
                         ::testing::ValuesIn(SpecialCases(true)),
                         SpecialCaseName);

// ---- L6: equal to upstream's u64 pack -> select -> unpack -------------------

// The HLO of upstream's u64 adapter (BuildPackF32ToU64, BuildUnpackU64ToF32 in
// topk_specializer.cc), op for op, around a u64 TopK that becomes a sort.
std::string PackSelectUnpackHlo(PrimitiveType type, const TopKShape& shape,
                                bool partial) {
  const std::string in = Dims(shape.batch, shape.n);
  const std::string out = Dims(shape.batch, shape.k);
  const std::string to_f32 =
      type == F32 ? "x32 = f32[$0] copy(x)" : "x32 = f32[$0] convert(x)";
  const std::string from_f32 =
      type == F32 ? "vals = f32[$1] copy(v32)" : "vals = bf16[$1] convert(v32)";
  const std::string pack_low = partial ? R"(
      c7f = u32[] constant(2147483647)
      b7f = u32[$0] broadcast(c7f), dimensions={}
      iota_neg = u32[$0] subtract(b7f, iota)
      c1 = u32[] constant(1)
      b1 = u32[$0] broadcast(c1), dimensions={}
      iota_shifted = u32[$0] shift-left(iota_neg, b1)
      is_nz_pred = pred[$0] compare(u32_val, b8), direction=EQ
      is_nz = u32[$0] convert(is_nz_pred)
      high = u32[$0] add(radix, is_nz)
      low = u32[$0] or(iota_shifted, is_nz))"
                                       : R"(
      cff = u32[] constant(4294967295)
      bff = u32[$0] broadcast(cff), dimensions={}
      high = u32[$0] copy(radix)
      low = u32[$0] subtract(bff, iota))";
  const std::string unpack_key = partial ? R"(
      low_k = u32[$1] convert(keys)
      uc1 = u32[] constant(1)
      ub1 = u32[$1] broadcast(uc1), dimensions={}
      nz_k = u32[$1] and(low_k, ub1)
      radix_k = u32[$1] subtract(radix_k0, nz_k))"
                                         : R"(
      radix_k = u32[$1] copy(radix_k0))";
  return absl::Substitute(absl::StrCat(R"(
    HloModule pack_select_unpack
    ENTRY main {
      x = $2[$0] parameter(0)
      )",
                                       to_f32, R"(
      s32_val = s32[$0] bitcast-convert(x32)
      c31 = s32[] constant(31)
      b31 = s32[$0] broadcast(c31), dimensions={}
      smeared = s32[$0] shift-right-arithmetic(s32_val, b31)
      smeared_u32 = u32[$0] bitcast-convert(smeared)
      c8 = u32[] constant(2147483648)
      b8 = u32[$0] broadcast(c8), dimensions={}
      mask = u32[$0] or(smeared_u32, b8)
      u32_val = u32[$0] bitcast-convert(x32)
      radix = u32[$0] xor(u32_val, mask)
      iota = u32[$0] iota(), iota_dimension=$3)",
                                       pack_low, R"(
      high64 = u64[$0] convert(high)
      c32 = u64[] constant(32)
      b32 = u64[$0] broadcast(c32), dimensions={}
      top = u64[$0] shift-left(high64, b32)
      low64 = u64[$0] convert(low)
      packed = u64[$0] or(top, low64)
      t = (u64[$1], s32[$1]) topk(packed), k=$4, largest=true
      keys = u64[$1] get-tuple-element(t), index=0
      idx = s32[$1] get-tuple-element(t), index=1
      kc32 = u64[] constant(32)
      kb32 = u64[$1] broadcast(kc32), dimensions={}
      rshift = u64[$1] shift-right-logical(keys, kb32)
      radix_k0 = u32[$1] convert(rshift))",
                                       unpack_key, R"(
      mc31 = u32[] constant(31)
      mb31 = u32[$1] broadcast(mc31), dimensions={}
      msb = u32[$1] shift-right-logical(radix_k, mb31)
      mc1 = u32[] constant(1)
      mb1 = u32[$1] broadcast(mc1), dimensions={}
      msb_minus_one = u32[$1] subtract(msb, mb1)
      mc8 = u32[] constant(2147483648)
      mb8 = u32[$1] broadcast(mc8), dimensions={}
      unmask = u32[$1] or(msb_minus_one, mb8)
      orig = u32[$1] xor(radix_k, unmask)
      v32 = f32[$1] bitcast-convert(orig)
      )",
                                       from_f32, R"(
      ROOT r = ($2[$1], s32[$1]) tuple(vals, idx)
    })"),
                          in, out, TypeName(type), shape.batch == 0 ? 0 : 1,
                          shape.k);
}

class TritonTopKEquivalenceTest
    : public TritonTopKTest,
      public ::testing::WithParamInterface<SpecialCase> {};

// Signaling NaNs are left out: upstream's bf16 <-> f32 converts quiet them,
// while the Triton kernel returns the input bits (design T4).
TEST_P(TritonTopKEquivalenceTest, MatchesUpstreamPackSelectUnpackBitwise) {
  const SpecialCase& c = GetParam();
  ASSERT_OK_AND_ASSIGN(
      auto triton, CompileWithTritonTopK(
                       TopKHlo(c.type, c.shape, c.partial ? "PARTIAL" : "TOTAL",
                               /*stable=*/true)));
  ASSERT_OK_AND_ASSIGN(auto upstream,
                       Compile(PackSelectUnpackHlo(c.type, c.shape, c.partial),
                               ConfigWithTritonTopK(false)));
  Literal input = c.type == F32 ? MakeInputLiteral<float>(c, false)
                                : MakeInputLiteral<bfloat16>(c, false);
  ASSERT_OK_AND_ASSIGN(Literal expected, Run(upstream.get(), {&input}));
  ASSERT_OK_AND_ASSIGN(Literal actual, Run(triton.get(), {&input}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

INSTANTIATE_TEST_SUITE_P(TritonTopKEquivalence, TritonTopKEquivalenceTest,
                         ::testing::ValuesIn(SpecialCases(false)),
                         SpecialCaseName);

}  // namespace
}  // namespace xla::gpu
