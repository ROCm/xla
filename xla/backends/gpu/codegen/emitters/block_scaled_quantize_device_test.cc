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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/backends/gpu/transforms/block_scaled_quantize_swizzle_folding.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/computation_layout.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "re2/re2.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kReducers = R"(
max_bf16 {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT m = bf16[] maximum(a, b)
}

max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}
)";

// How the quantization is written: as the fused computation the pass builds
// after layout normalization (bitcasts), or as the unoptimized HLO that today's
// pipeline compiles (reshapes, transposes and bitcast-converts).
enum class Form { kFused, kUnfused };

// How a column quantization reaches its reduce in the fused form (both occur in
// the step 0 dumps): through layout bitcasts with the reduce result transposed,
// or through a transpose of |x|.
enum class Column { kNone, kNormalized, kTransposeFirst };

struct Quantize {
  PrimitiveType source;
  PrimitiveType data;
  int64_t rows;
  int64_t cols;
  bool rms_norm_prologue = false;
  // The 2^-127 of the decodes as tensor operands, as in the forward body where
  // it is a loop-carried broadcast.
  bool tiny_operand = false;
  bool row = true;
  Column column = Column::kNone;
  // The encode's ratio is the amax itself, so that a block of amax 2^127 or
  // the source type's maximum gets the scale byte 254.
  bool unit_ratio = false;
};

// JAX's E8M0 encode of `$PRE_ratio` input `$AMAXF` and the decode, divide,
// clamp and convert of `$XNAME`, as XLA has simplified them (step 0 dump of
// Llama3-8B).
constexpr absl::string_view kScaleAndData = R"(
  $PRE_inv = f32[] constant($INV)
  $PRE_inv_b = f32[$BLKS] broadcast($PRE_inv), dimensions={}
  $PRE_ratio = f32[$BLKS] multiply($AMAXF, $PRE_inv_b)
  $PRE_bits = u32[$BLKS] $BITCAST($PRE_ratio)
  $PRE_c_mant = u32[] constant(8388607)
  $PRE_mant_mask = u32[$BLKS] broadcast($PRE_c_mant), dimensions={}
  $PRE_mant = u32[$BLKS] and($PRE_bits, $PRE_mant_mask)
  $PRE_c0 = u32[] constant(0)
  $PRE_zero = u32[$BLKS] broadcast($PRE_c0), dimensions={}
  $PRE_gt = pred[$BLKS] compare($PRE_mant, $PRE_zero), direction=GT
  $PRE_c23 = u32[] constant(23)
  $PRE_s23 = u32[$BLKS] broadcast($PRE_c23), dimensions={}
  $PRE_exp = u32[$BLKS] shift-right-logical($PRE_bits, $PRE_s23)
  $PRE_c254 = u32[] constant(254)
  $PRE_e254 = u32[$BLKS] broadcast($PRE_c254), dimensions={}
  $PRE_ne = pred[$BLKS] compare($PRE_exp, $PRE_e254), direction=NE
  $PRE_and0 = pred[$BLKS] and($PRE_gt, $PRE_ne)
  $PRE_eq0 = pred[$BLKS] compare($PRE_exp, $PRE_zero), direction=EQ
  $PRE_c_half = u32[] constant(4194304)
  $PRE_half = u32[$BLKS] broadcast($PRE_c_half), dimensions={}
  $PRE_le = pred[$BLKS] compare($PRE_mant, $PRE_half), direction=LE
  $PRE_and1 = pred[$BLKS] and($PRE_eq0, $PRE_le)
  $PRE_not = pred[$BLKS] not($PRE_and1)
  $PRE_round_up = pred[$BLKS] and($PRE_and0, $PRE_not)
  $PRE_c1 = u32[] constant(1)
  $PRE_one = u32[$BLKS] broadcast($PRE_c1), dimensions={}
  $PRE_exp1 = u32[$BLKS] add($PRE_exp, $PRE_one)
  $PRE_exp_sel = u32[$BLKS] select($PRE_round_up, $PRE_exp1, $PRE_exp)
  $PRE_scale = u8[$BLKS] convert($PRE_exp_sel)
  $PRE_scale32 = u32[$BLKS] convert($PRE_scale)
  $PRE_shl = u32[$BLKS] shift-left($PRE_scale32, $PRE_s23)
  $PRE_pow = f32[$BLKS] $BITCAST($PRE_shl)
  $PRE_cf0 = f32[] constant(0)
  $PRE_fzero = f32[$BLKS] broadcast($PRE_cf0), dimensions={}
  $PRE_is_zero = pred[$BLKS] compare($PRE_pow, $PRE_fzero), direction=EQ
$TINY  $PRE_decoded = f32[$BLKS] select($PRE_is_zero, $PRE_tiny_b, $PRE_pow)
  $PRE_bs = f32[$VIEW] broadcast($PRE_decoded), dimensions={0,1}
  $PRE_q = f32[$VIEW] divide($XNAME, $PRE_bs)
  $PRE_lo = f32[] constant(-$MAX)
  $PRE_lo_b = f32[$VIEW] broadcast($PRE_lo), dimensions={}
  $PRE_hi = f32[] constant($MAX)
  $PRE_hi_b = f32[$VIEW] broadcast($PRE_hi), dimensions={}
  $PRE_cl = f32[$VIEW] clamp($PRE_lo_b, $PRE_q, $PRE_hi_b)
  $PRE_data = $DTYPE[$VIEW]$DLAYOUT convert($PRE_cl)
)";

std::string Shape(absl::string_view type, int64_t a, int64_t b, int64_t c = 0) {
  return c == 0 ? absl::StrCat(type, "[", a, ",", b, "]")
                : absl::StrCat(type, "[", a, ",", b, ",", c, "]");
}

struct Body {
  std::string text;
  std::string root_shape;
};

// The row and/or column quantization of the source, in blocks of 32.
Body QuantizeBody(const Quantize& q, Form form) {
  const bool f32 = q.source == F32;
  const bool fp4 = q.data == F4E2M1FN;
  const bool fused = form == Form::kFused;
  const std::string src = primitive_util::LowercasePrimitiveTypeName(q.source);
  const std::string dtype = primitive_util::LowercasePrimitiveTypeName(q.data);
  const std::string data_layout = fp4 && fused ? "{2,1,0:E(4)}" : "";
  const int64_t m = q.rows, n = q.cols, mb = m / 32, nb = n / 32;
  const std::string rc = Shape(src, m, n);
  auto scale_and_data = [&](absl::string_view prefix, std::string blocks,
                            std::string view, absl::string_view amax_f32,
                            absl::string_view x, std::string tiny) {
    std::string text(kScaleAndData);
    absl::StrReplaceAll({{"$PRE", prefix},
                         {"$BLKS", blocks},
                         {"$VIEW", view},
                         {"$AMAXF", amax_f32},
                         {"$XNAME", x},
                         {"$INV", q.unit_ratio ? "1"
                                  : fp4        ? "0.166666672"
                                               : "0.00223214296"},
                         {"$MAX", fp4 ? "6" : "448"},
                         {"$DTYPE", dtype},
                         {"$DLAYOUT", data_layout},
                         {"$BITCAST", fused ? "bitcast" : "bitcast-convert"},
                         {"$TINY", tiny}},
                        &text);
    return text;
  };
  auto tiny_constant = [&](absl::string_view prefix, absl::string_view blocks) {
    return absl::StrCat("  ", prefix,
                        "_tiny = f32[] constant(5.87747175e-39)\n  ", prefix,
                        "_tiny_b = f32[", blocks, "] broadcast(", prefix,
                        "_tiny), dimensions={}\n");
  };
  std::string text;
  if (q.rms_norm_prologue) {
    // The RMSNorm apply in the form today's pipeline computes it: in f32,
    // with one rounding to the source type.
    absl::StrAppend(
        &text, "  x = ", rc, " parameter(0)\n  rsqrt = f32[", m,
        "] parameter(1)\n  w = ", src, "[", n,
        "] parameter(2)\n  x_f32 = ", Shape("f32", m, n),
        " convert(x)\n  rsqrt_b = ", Shape("f32", m, n),
        " broadcast(rsqrt), dimensions={0}\n  normed = ", Shape("f32", m, n),
        " multiply(x_f32, rsqrt_b)\n  w_f32 = f32[", n,
        "] convert(w)\n  w_b = ", Shape("f32", m, n),
        " broadcast(w_f32), dimensions={1}\n",
        f32 ? absl::StrCat("  p0 = ", rc, " multiply(normed, w_b)\n")
            : absl::StrCat("  prod = ", Shape("f32", m, n),
                           " multiply(normed, w_b)\n  p0 = ", rc,
                           " convert(prod)\n"));
  } else {
    absl::StrAppend(&text, "  p0 = ", rc, " parameter(0)\n");
  }
  const std::string reshape = fused ? "bitcast" : "reshape";
  absl::StrAppend(&text, "  abs = ", rc, " abs(p0)\n  ninf = ", src,
                  "[] constant(-inf)\n");
  const bool uses_xf =
      (q.row && !q.rms_norm_prologue) || (fused && q.column != Column::kNone);
  if (!f32 && uses_xf) {
    absl::StrAppend(&text, "  xf = ", Shape("f32", m, n), " convert(p0)\n");
  }
  const std::string xf = f32 ? "p0" : "xf";
  std::vector<std::string> data_roots, scale_roots, data_shapes, scale_shapes;
  if (q.row) {
    const std::string view = absl::StrCat(m, ",", nb, ",32");
    const std::string blocks = absl::StrCat(m, ",", nb);
    if (q.rms_norm_prologue && !f32) {
      absl::StrAppend(&text, "  r_xs = ", Shape(src, m, nb, 32), " ", reshape,
                      "(p0)\n  r_x = ", Shape("f32", m, nb, 32),
                      " convert(r_xs)\n");
    } else {
      absl::StrAppend(&text, "  r_x = ", Shape("f32", m, nb, 32), " ", reshape,
                      "(", xf, ")\n");
    }
    absl::StrAppend(&text, "  r_ab = ", Shape(src, m, nb, 32), " ", reshape,
                    "(abs)\n  r_amax = ", Shape(src, m, nb),
                    " reduce(r_ab, ninf), dimensions={2}, to_apply=max_", src,
                    "\n");
    if (!f32) {
      absl::StrAppend(&text, "  r_amax_f32 = ", Shape("f32", m, nb),
                      " convert(r_amax)\n");
    }
    const std::string tiny =
        q.tiny_operand
            ? absl::StrCat("  r_tiny_b = f32[", blocks, "] parameter(",
                           q.rms_norm_prologue ? 3 : 1, ")\n")
            : tiny_constant("r", blocks);
    absl::StrAppend(
        &text, scale_and_data("r", blocks, view, f32 ? "r_amax" : "r_amax_f32",
                              "r_x", tiny));
    data_roots.push_back("r_data");
    scale_roots.push_back("r_scale");
    data_shapes.push_back(Shape(dtype, m, nb, 32) + data_layout);
    scale_shapes.push_back(Shape("u8", m, nb));
  }
  if (q.column != Column::kNone) {
    const std::string view = absl::StrCat(n, ",", mb, ",32");
    const std::string blocks = absl::StrCat(n, ",", mb);
    std::string amax_f32 = "c_amax_f32";
    if (!fused) {
      absl::StrAppend(
          &text, "  c_t = ", Shape(src, n, m),
          " transpose(p0), dimensions={1,0}\n  c_v = ", Shape(src, n, mb, 32),
          " reshape(c_t)\n  c_abs = ", Shape(src, n, mb, 32),
          " abs(c_v)\n  c_amax = ", Shape(src, n, mb),
          " reduce(c_abs, ninf), dimensions={2}, "
          "to_apply=max_",
          src, "\n");
      if (!f32) {
        absl::StrAppend(&text, "  c_x = ", Shape("f32", n, mb, 32),
                        " convert(c_v)\n");
      }
    } else {
      absl::StrAppend(&text, "  c_xb = ", Shape("f32", mb, 32, n), " bitcast(",
                      xf, ")\n  c_x = ", Shape("f32", n, mb, 32),
                      " transpose(c_xb), dimensions={2,0,1}\n");
      if (q.column == Column::kNormalized) {
        absl::StrAppend(
            &text, "  c_ab0 = ", Shape(src, n, mb, 32),
            "{0,2,1} bitcast(abs)\n", "  c_ab = ", Shape(src, mb, 32, n),
            " bitcast(c_ab0)\n", "  c_amax0 = ", Shape(src, mb, n),
            " reduce(c_ab, ninf), dimensions={1}, to_apply=max_", src, "\n",
            "  c_at0 = ", Shape(src, n, mb), "{0,1} bitcast(c_amax0)\n",
            "  c_at1 = ", Shape(src, mb, n), " bitcast(c_at0)\n");
        if (!f32) {
          absl::StrAppend(&text, "  c_af = ", Shape("f32", mb, n),
                          " convert(c_at1)\n");
        }
        absl::StrAppend(&text, "  c_amax_f32 = ", Shape("f32", n, mb),
                        " transpose(", f32 ? "c_at1" : "c_af",
                        "), dimensions={1,0}\n");
      } else {
        absl::StrAppend(&text, "  c_ab = ", Shape(src, mb, 32, n),
                        " bitcast(abs)\n  c_at = ", Shape(src, n, mb, 32),
                        " transpose(c_ab), dimensions={2,0,1}\n  c_amax = ",
                        Shape(src, n, mb),
                        " reduce(c_at, ninf), dimensions={2}, to_apply=max_",
                        src, "\n");
      }
    }
    if (!fused || q.column == Column::kTransposeFirst) {
      if (f32) {
        amax_f32 = "c_amax";
      } else {
        absl::StrAppend(&text, "  c_amax_f32 = ", Shape("f32", n, mb),
                        " convert(c_amax)\n");
      }
    }
    absl::StrAppend(
        &text,
        scale_and_data(
            "c", blocks, view, amax_f32, !fused && f32 ? "c_v" : "c_x",
            q.tiny_operand
                ? absl::StrCat("  c_tiny_b = f32[", blocks, "] parameter(",
                               (q.rms_norm_prologue ? 3 : 1) + (q.row ? 1 : 0),
                               ")\n")
                : tiny_constant("c", blocks)));
    data_roots.push_back("c_data");
    scale_roots.push_back("c_scale");
    data_shapes.push_back(Shape(dtype, n, mb, 32) + data_layout);
    scale_shapes.push_back(Shape("u8", n, mb));
  }
  absl::c_copy(scale_roots, std::back_inserter(data_roots));
  absl::c_copy(scale_shapes, std::back_inserter(data_shapes));
  const std::string root_shape =
      absl::StrCat("(", absl::StrJoin(data_shapes, ", "), ")");
  absl::StrAppend(&text, "  ROOT t = ", root_shape, " tuple(",
                  absl::StrJoin(data_roots, ", "), ")\n");
  return {text, root_shape};
}

// The shapes of the 2^-127 operands: the row scale's, then the column's.
std::vector<std::vector<int64_t>> TinyShapes(const Quantize& q) {
  std::vector<std::vector<int64_t>> shapes;
  if (q.row) {
    shapes.push_back({q.rows, q.cols / 32});
  }
  if (q.column != Column::kNone) {
    shapes.push_back({q.cols, q.rows / 32});
  }
  return shapes;
}

std::string FusedModule(const Quantize& q) {
  const std::string src = primitive_util::LowercasePrimitiveTypeName(q.source);
  std::string params =
      absl::StrCat("  p = ", src, "[", q.rows, ",", q.cols, "] parameter(0)\n");
  std::string operands = "p";
  if (q.rms_norm_prologue) {
    absl::StrAppend(&params, "  r = f32[", q.rows,
                    "] parameter(1)\n  w = ", src, "[", q.cols,
                    "] parameter(2)\n");
    operands = "p, r, w";
  }
  if (q.tiny_operand) {
    int64_t number = q.rms_norm_prologue ? 3 : 1;
    for (const std::vector<int64_t>& shape : TinyShapes(q)) {
      absl::StrAppend(&params, "  tiny", number, " = f32[", shape[0], ",",
                      shape[1], "] parameter(", number, ")\n");
      absl::StrAppend(&operands, ", tiny", number);
      ++number;
    }
  }
  Body body = QuantizeBody(q, Form::kFused);
  return absl::StrCat(
      "HloModule fused\n", kReducers, "\nfused_quantize {\n", body.text,
      "}\n\nENTRY main {\n", params, "  ROOT f = ", body.root_shape, " fusion(",
      operands,
      "), kind=kCustom, calls=fused_quantize, "
      "backend_config={\"fusion_backend_config\":{\"kind\":\"__block_"
      "scaled_quantize\"}}\n}\n");
}

std::string UnfusedModule(const Quantize& q) {
  return absl::StrCat("HloModule unfused\n", kReducers, "\nENTRY main {\n",
                      QuantizeBody(q, Form::kUnfused).text, "}\n");
}

// The values of the source's blocks, one class per test input.
enum class InputClass {
  kRandom,
  kZeros,
  kSubnormalAmax,
  kAmaxIs448TimesPowerOfTwo,
  kAmaxIs6TimesPowerOfTwo,
  kInf,
  kNan,
  kHugeAmax,
  kScaleSubstitution,
  kMixed,
};

constexpr InputClass kAllClasses[] = {
    InputClass::kRandom,
    InputClass::kZeros,
    InputClass::kSubnormalAmax,
    InputClass::kAmaxIs448TimesPowerOfTwo,
    InputClass::kAmaxIs6TimesPowerOfTwo,
    InputClass::kInf,
    InputClass::kNan,
    InputClass::kHugeAmax,
    InputClass::kScaleSubstitution,
    InputClass::kMixed,
};

absl::string_view ClassName(InputClass c) {
  switch (c) {
    case InputClass::kRandom:
      return "random";
    case InputClass::kZeros:
      return "zeros";
    case InputClass::kSubnormalAmax:
      return "subnormal_amax";
    case InputClass::kAmaxIs448TimesPowerOfTwo:
      return "amax_448x2^k";
    case InputClass::kAmaxIs6TimesPowerOfTwo:
      return "amax_6x2^k";
    case InputClass::kInf:
      return "inf";
    case InputClass::kNan:
      return "nan";
    case InputClass::kHugeAmax:
      return "amax_ge_2^127";
    case InputClass::kScaleSubstitution:
      return "scale_2^-127_substitution";
    case InputClass::kMixed:
      return "mixed";
  }
}

// Fills one block of 32 values of class `c`.
void FillBlock(InputClass c, float type_max, std::mt19937& rng,
               absl::Span<float> block) {
  std::normal_distribution<float> normal;
  std::uniform_int_distribution<int> exponent(-30, 30);
  std::uniform_int_distribution<int> position(0, block.size() - 1);
  std::bernoulli_distribution sign;
  auto fill_random = [&](float magnitude) {
    for (float& value : block) {
      value = normal(rng) * magnitude;
    }
  };
  auto set_amax = [&](float amax) {
    for (float& value : block) {
      value = std::clamp(value, -amax, amax);
    }
    block[position(rng)] = sign(rng) ? -amax : amax;
  };
  switch (c) {
    case InputClass::kRandom:
      fill_random(std::ldexp(1.0f, exponent(rng)));
      return;
    case InputClass::kZeros:
      for (float& value : block) {
        value = sign(rng) ? -0.0f : 0.0f;
      }
      return;
    case InputClass::kSubnormalAmax:
      fill_random(1e-40f);
      return;
    case InputClass::kAmaxIs448TimesPowerOfTwo: {
      const float amax = std::ldexp(448.0f, exponent(rng));
      fill_random(amax / 4);
      set_amax(amax);
      return;
    }
    case InputClass::kAmaxIs6TimesPowerOfTwo: {
      const float amax = std::ldexp(6.0f, exponent(rng));
      fill_random(amax / 4);
      set_amax(amax);
      return;
    }
    case InputClass::kInf:
      fill_random(std::ldexp(1.0f, exponent(rng)));
      block[position(rng)] = sign(rng) ? -std::numeric_limits<float>::infinity()
                                       : std::numeric_limits<float>::infinity();
      return;
    case InputClass::kNan:
      fill_random(std::ldexp(1.0f, exponent(rng)));
      block[position(rng)] = std::numeric_limits<float>::quiet_NaN();
      return;
    case InputClass::kHugeAmax: {
      const float amax = sign(rng) ? type_max : std::ldexp(1.0f, 127);
      fill_random(amax / 8);
      set_amax(amax);
      return;
    }
    case InputClass::kScaleSubstitution: {
      std::uniform_int_distribution<int> tiny(-133, -118);
      const float amax = std::ldexp(1.0f, tiny(rng));
      fill_random(amax / 4);
      set_amax(amax);
      return;
    }
    case InputClass::kMixed: {
      std::uniform_int_distribution<int> pick(0, 8);
      FillBlock(kAllClasses[pick(rng)], type_max, rng, block);
      return;
    }
  }
}

// The source with blocks of 32 of class `c`, along its rows or, with
// `column_blocks`, along its columns.
Literal MakeSource(PrimitiveType type, int64_t rows, int64_t cols, InputClass c,
                   int seed, bool column_blocks = false) {
  std::mt19937 rng(seed);
  std::vector<float> blocks(rows * cols);
  const float type_max =
      type == BF16 ? static_cast<float>(std::numeric_limits<bfloat16>::max())
                   : std::numeric_limits<float>::max();
  for (int64_t start = 0; start < rows * cols; start += 32) {
    FillBlock(c, type_max, rng, absl::MakeSpan(blocks).subspan(start, 32));
  }
  std::vector<float> values = blocks;
  if (column_blocks) {
    for (int64_t m = 0; m < rows; ++m) {
      for (int64_t n = 0; n < cols; ++n) {
        values[m * cols + n] = blocks[n * rows + m];
      }
    }
  }
  Literal literal(ShapeUtil::MakeShape(type, {rows, cols}));
  if (type == BF16) {
    absl::Span<bfloat16> data = literal.data<bfloat16>();
    for (int64_t i = 0; i < rows * cols; ++i) {
      data[i] = static_cast<bfloat16>(values[i]);
    }
  } else {
    absl::c_copy(values, literal.data<float>().begin());
  }
  return literal;
}

Literal MakeRandom(PrimitiveType type, absl::Span<const int64_t> dims,
                   float magnitude, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(0.5f, 1.5f);
  Literal literal(ShapeUtil::MakeShape(type, dims));
  const int64_t n = ShapeUtil::ElementsIn(literal.shape());
  for (int64_t i = 0; i < n; ++i) {
    const float value = uniform(rng) * magnitude;
    if (type == BF16) {
      literal.data<bfloat16>()[i] = static_cast<bfloat16>(value);
    } else {
      literal.data<float>()[i] = value;
    }
  }
  return literal;
}

std::vector<std::pair<InputClass, bool>> Cross(
    absl::Span<const InputClass> classes, absl::Span<const bool> fills) {
  std::vector<std::pair<InputClass, bool>> result;
  for (InputClass c : classes) {
    for (bool fill : fills) {
      result.push_back({c, fill});
    }
  }
  return result;
}

class BlockScaledQuantizeDeviceTest
    : public HloPjRtGpuTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType>> {
 protected:
  // Compiles `q` as one `__block_scaled_quantize` fusion and through today's
  // pipeline, runs both on each input class and expects bitwise equal outputs.
  void RunAndCompareExactly(const Quantize& q,
                            absl::Span<const InputClass> classes,
                            absl::Span<const bool> column_fills = {false}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> fused,
                         ParseAndReturnVerifiedModule(FusedModule(q)));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> unfused,
                         ParseAndReturnVerifiedModule(UnfusedModule(q)));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> fused_executable,
                         CreateExecutable(std::move(fused),
                                          /*run_hlo_passes=*/false));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> unfused_executable,
                         CreateExecutable(std::move(unfused),
                                          /*run_hlo_passes=*/true));
    for (auto [c, column_fill] : Cross(classes, column_fills)) {
      SCOPED_TRACE(
          absl::StrCat(ClassName(c), column_fill ? " in columns" : ""));
      std::vector<Literal> arguments;
      arguments.push_back(MakeSource(q.source, q.rows, q.cols, c,
                                     static_cast<int>(c), column_fill));
      if (q.rms_norm_prologue) {
        arguments.push_back(MakeRandom(F32, {q.rows}, 1.0f, 1));
        arguments.push_back(MakeRandom(q.source, {q.cols}, 1.0f, 2));
      }
      for (const std::vector<int64_t>& shape :
           q.tiny_operand ? TinyShapes(q)
                          : std::vector<std::vector<int64_t>>{}) {
        Literal tiny(ShapeUtil::MakeShape(F32, shape));
        absl::Span<float> values = tiny.data<float>();
        std::fill(values.begin(), values.end(), std::ldexp(1.0f, -127));
        arguments.push_back(std::move(tiny));
      }
      ASSERT_OK_AND_ASSIGN(Literal expected,
                           test_runner().ExecuteWithExecutable(
                               unfused_executable.get(), arguments));
      ASSERT_OK_AND_ASSIGN(Literal actual,
                           test_runner().ExecuteWithExecutable(
                               fused_executable.get(), arguments));
      EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual))
          << "input class " << ClassName(c)
          << (column_fill ? " in column blocks" : "");
    }
  }
};

TEST_P(BlockScaledQuantizeDeviceTest, RowSmall) {
  auto [source, data] = GetParam();
  RunAndCompareExactly({source, data, 64, 128}, kAllClasses);
}

TEST_P(BlockScaledQuantizeDeviceTest, RowWithRmsNormPrologue) {
  auto [source, data] = GetParam();
  RunAndCompareExactly({source, data, 64, 128, /*rms_norm_prologue=*/true},
                       {InputClass::kRandom, InputClass::kMixed});
}

TEST_P(BlockScaledQuantizeDeviceTest, RowWithScaleOperand) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 64, 128};
  q.tiny_operand = true;
  RunAndCompareExactly(q, {InputClass::kRandom, InputClass::kZeros,
                           InputClass::kScaleSubstitution, InputClass::kMixed});
}

// bf16 [32, 96] leaves half of the second workgroup outside the source; its
// lanes evaluate the scale region, which reads the 2^-127 operand.
TEST_P(BlockScaledQuantizeDeviceTest, RowWithScaleOperandPartialWorkgroup) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 32, 96};
  q.tiny_operand = true;
  RunAndCompareExactly(q, {InputClass::kRandom, InputClass::kScaleSubstitution,
                           InputClass::kMixed});
}

TEST_P(BlockScaledQuantizeDeviceTest, RowWithTopScaleByte) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 64, 128};
  q.unit_ratio = true;
  RunAndCompareExactly(q, kAllClasses);
}

TEST_P(BlockScaledQuantizeDeviceTest, RowLlama8bActivation) {
  auto [source, data] = GetParam();
  if (source != BF16) {
    GTEST_SKIP() << "The 8B activation shape is tested in bf16 only.";
  }
  RunAndCompareExactly({source, data, 16384, 4096},
                       {InputClass::kRandom, InputClass::kMixed});
}

TEST_P(BlockScaledQuantizeDeviceTest, ColumnNormalized) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 128, 256};
  q.row = false;
  q.column = Column::kNormalized;
  RunAndCompareExactly(q, kAllClasses, {true});
}

TEST_P(BlockScaledQuantizeDeviceTest, ColumnTransposeFirst) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 128, 256};
  q.row = false;
  q.column = Column::kTransposeFirst;
  RunAndCompareExactly(q, kAllClasses, {true});
}

// 96 rows: the last workgroup tile has one band and three column blocks
// outside the source.
TEST_P(BlockScaledQuantizeDeviceTest, ColumnPartialTile) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 96, 192};
  q.row = false;
  q.column = Column::kNormalized;
  RunAndCompareExactly(q, {InputClass::kRandom, InputClass::kMixed}, {true});
}

TEST_P(BlockScaledQuantizeDeviceTest, ColumnPartialTileWithScaleOperand) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 96, 192};
  q.row = false;
  q.column = Column::kNormalized;
  q.tiny_operand = true;
  RunAndCompareExactly(
      q,
      {InputClass::kRandom, InputClass::kScaleSubstitution, InputClass::kMixed},
      {true});
}

// 256 columns take the 8-wave tile; its last tile has three bands.
TEST_P(BlockScaledQuantizeDeviceTest, ColumnPartialWideTileWithScaleOperand) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 96, 256};
  q.row = false;
  q.column = Column::kNormalized;
  q.tiny_operand = true;
  RunAndCompareExactly(q, kAllClasses, {false, true});
}

TEST_P(BlockScaledQuantizeDeviceTest, BothNormalized) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 128, 256};
  q.column = Column::kNormalized;
  RunAndCompareExactly(q, kAllClasses, {false, true});
}

TEST_P(BlockScaledQuantizeDeviceTest, BothTransposeFirst) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 128, 256};
  q.column = Column::kTransposeFirst;
  RunAndCompareExactly(q, kAllClasses, {false, true});
}

TEST_P(BlockScaledQuantizeDeviceTest, BothPartialTile) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 192, 192};
  q.column = Column::kNormalized;
  RunAndCompareExactly(q, {InputClass::kRandom, InputClass::kMixed},
                       {false, true});
}

TEST_P(BlockScaledQuantizeDeviceTest, BothPartialTileWithScaleOperand) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 192, 192};
  q.column = Column::kNormalized;
  q.tiny_operand = true;
  RunAndCompareExactly(
      q,
      {InputClass::kRandom, InputClass::kScaleSubstitution, InputClass::kMixed},
      {false, true});
}

TEST_P(BlockScaledQuantizeDeviceTest, BothPartialWideTileWithScaleOperand) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 192, 256};
  q.column = Column::kNormalized;
  q.tiny_operand = true;
  RunAndCompareExactly(q, kAllClasses, {false, true});
}

TEST_P(BlockScaledQuantizeDeviceTest, BothWithTopScaleByte) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 128, 256};
  q.column = Column::kNormalized;
  q.unit_ratio = true;
  RunAndCompareExactly(q, kAllClasses, {false, true});
}

TEST_P(BlockScaledQuantizeDeviceTest, BothLlama8bActivation) {
  auto [source, data] = GetParam();
  if (source != BF16) {
    GTEST_SKIP() << "The 8B activation shape is tested in bf16 only.";
  }
  Quantize q{source, data, 16384, 4096};
  q.column = Column::kNormalized;
  RunAndCompareExactly(q, {InputClass::kRandom, InputClass::kMixed},
                       {false, true});
}

// The producer of the quantized value, as JAX emits it in Llama3-8B (step 0
// dump before optimizations).
enum class Producer {
  // A parameter.
  kNone,
  // RMSNorm apply in bf16: x * rsqrt(sum(x^2) / n + eps) rounded to bf16, times
  // the bf16 norm weight sliced from an f32 stack.
  kRmsNorm,
  // silu(a) * b in bf16.
  kSiluMul,
  // An f32 weight sliced from a stack and converted to bf16.
  kWeight,
  // An attention gradient [b, h, s, d] seen as [b * s, h * d], like dk.
  kHeadsTokensDim,
  // An attention gradient [b, h, d, s] seen as [b * s, h * d], like dv.
  kHeadsDimTokens,
  // A [b, s, h, d] value seen as [b * s, h * d]: the column stride spans h and
  // d, as for Wq, the attention output and dq.
  kTokensHeadsDim,
};

// Variants of JAX's `quantize` the pipeline tests compile.
struct PipelineVariant {
  // The decode's 2^-127 comes from a tensor operand, as the loop-carried one
  // of the forward body.
  bool tiny_operand = false;
  // The divide, clamp and convert run in bf16.
  bool bf16_data_region = false;
};

// JAX's `quantize` of `source` along its rows, or along its columns after a
// transpose, through its optimization barrier. Appends the barrier's outputs.
std::string JaxQuantize(absl::string_view prefix, absl::string_view source,
                        int64_t rows, int64_t cols, bool column,
                        PrimitiveType data, absl::string_view tiny,
                        bool bf16_data_region,
                        std::vector<std::string>& outputs,
                        std::vector<std::string>& shapes) {
  const std::string dtype = primitive_util::LowercasePrimitiveTypeName(data);
  const bool fp4 = data == F4E2M1FN;
  const int64_t m = column ? cols : rows, n = column ? rows : cols;
  const std::string blocks = absl::StrCat(m, ",", n / 32);
  const std::string view = absl::StrCat(m, ",", n / 32, ",32");
  std::string input(source);
  std::string text;
  if (column) {
    input = absl::StrCat(prefix, "_in");
    text = absl::StrCat("  ", input, " = bf16[", m, ",", n, "] transpose(",
                        source, "), dimensions={1,0}\n");
  }
  absl::StrAppend(&text, "  $P_v = bf16[$VIEW] reshape(", input, ")\n");
  absl::StrAppend(&text, R"(
  $P_abs = bf16[$VIEW] abs($P_v)
  $P_ninf = bf16[] constant(-inf)
  $P_amax = bf16[$BLKS] reduce($P_abs, $P_ninf), dimensions={2}, to_apply=max_bf16
  $P_amax_f32 = f32[$BLKS] convert($P_amax)
  $P_max = f32[] constant($MAX)
  $P_max_b = f32[$BLKS] broadcast($P_max), dimensions={}
  $P_ratio = f32[$BLKS] divide($P_amax_f32, $P_max_b)
  $P_bits = u32[$BLKS] bitcast-convert($P_ratio)
  $P_c_mant = u32[] constant(8388607)
  $P_mant_mask = u32[$BLKS] broadcast($P_c_mant), dimensions={}
  $P_mant = u32[$BLKS] and($P_bits, $P_mant_mask)
  $P_c0 = u32[] constant(0)
  $P_zero = u32[$BLKS] broadcast($P_c0), dimensions={}
  $P_gt = pred[$BLKS] compare($P_mant, $P_zero), direction=GT
  $P_c23 = u32[] constant(23)
  $P_s23 = u32[$BLKS] broadcast($P_c23), dimensions={}
  $P_exp = u32[$BLKS] shift-right-logical($P_bits, $P_s23)
  $P_c254 = u32[] constant(254)
  $P_e254 = u32[$BLKS] broadcast($P_c254), dimensions={}
  $P_ne = pred[$BLKS] compare($P_exp, $P_e254), direction=NE
  $P_and0 = pred[$BLKS] and($P_gt, $P_ne)
  $P_eq0 = pred[$BLKS] compare($P_exp, $P_zero), direction=EQ
  $P_c_half = u32[] constant(4194304)
  $P_half = u32[$BLKS] broadcast($P_c_half), dimensions={}
  $P_le = pred[$BLKS] compare($P_mant, $P_half), direction=LE
  $P_and1 = pred[$BLKS] and($P_eq0, $P_le)
  $P_not = pred[$BLKS] not($P_and1)
  $P_round_up = pred[$BLKS] and($P_and0, $P_not)
  $P_c1 = u32[] constant(1)
  $P_one = u32[$BLKS] broadcast($P_c1), dimensions={}
  $P_exp1 = u32[$BLKS] add($P_exp, $P_one)
  $P_exp_sel = u32[$BLKS] select($P_round_up, $P_exp1, $P_exp)
  $P_scale = u8[$BLKS] convert($P_exp_sel)
  $P_scale32 = u32[$BLKS] convert($P_scale)
  $P_shl = u32[$BLKS] shift-left($P_scale32, $P_s23)
  $P_pow = f32[$BLKS] bitcast-convert($P_shl)
  $P_cf0 = f32[] constant(0)
  $P_fzero = f32[$BLKS] broadcast($P_cf0), dimensions={}
  $P_is_zero = pred[$BLKS] compare($P_pow, $P_fzero), direction=EQ
$TINYDEF  $P_decoded = f32[$BLKS] select($P_is_zero, $TINYB, $P_pow)
$DATAREGION  $P_data = $DTYPE[$VIEW] convert($P_cl)
  $P_data_out = $DTYPE[$DM,$DN] reshape($P_data)
  $P_scale_out = f8e8m0fnu[$BLKS] bitcast-convert($P_scale)
  $P_pair = ($DTYPE[$DM,$DN], f8e8m0fnu[$BLKS]) tuple($P_data_out, $P_scale_out)
  $P_barrier = ($DTYPE[$DM,$DN], f8e8m0fnu[$BLKS]) opt-barrier($P_pair)
  $P_data_q = $DTYPE[$DM,$DN] get-tuple-element($P_barrier), index=0
  $P_scale_q = f8e8m0fnu[$BLKS] get-tuple-element($P_barrier), index=1
)");
  const std::string tiny_definition =
      tiny.empty() ? std::string(R"(  $P_tiny = f32[] constant(5.87747175e-39)
  $P_tiny_b = f32[$BLKS] broadcast($P_tiny), dimensions={}
)")
                   : "";
  const std::string data_region =
      bf16_data_region
          ? R"(  $P_dec_bf16 = bf16[$BLKS] convert($P_decoded)
  $P_bs = bf16[$VIEW] broadcast($P_dec_bf16), dimensions={0,1}
  $P_q = bf16[$VIEW] divide($P_v, $P_bs)
  $P_lo = bf16[] constant(-$MAX)
  $P_lo_b = bf16[$VIEW] broadcast($P_lo), dimensions={}
  $P_hi = bf16[] constant($MAX)
  $P_hi_b = bf16[$VIEW] broadcast($P_hi), dimensions={}
  $P_cl = bf16[$VIEW] clamp($P_lo_b, $P_q, $P_hi_b)
)"
          : R"(  $P_bs = f32[$VIEW] broadcast($P_decoded), dimensions={0,1}
  $P_xf = f32[$VIEW] convert($P_v)
  $P_q = f32[$VIEW] divide($P_xf, $P_bs)
  $P_lo = f32[] constant(-$MAX)
  $P_lo_b = f32[$VIEW] broadcast($P_lo), dimensions={}
  $P_hi_b = f32[$VIEW] broadcast($P_max), dimensions={}
  $P_cl = f32[$VIEW] clamp($P_lo_b, $P_q, $P_hi_b)
)";
  absl::StrReplaceAll({{"$TINYDEF", tiny_definition},
                       {"$TINYB", tiny.empty() ? absl::StrCat(prefix, "_tiny_b")
                                               : std::string(tiny)},
                       {"$DATAREGION", data_region}},
                      &text);
  absl::StrReplaceAll({{"$P", prefix},
                       {"$VIEW", view},
                       {"$BLKS", blocks},
                       {"$MAX", fp4 ? "6" : "448"},
                       {"$DTYPE", dtype},
                       {"$DM", absl::StrCat(m)},
                       {"$DN", absl::StrCat(n)}},
                      &text);
  outputs.push_back(absl::StrCat(prefix, "_data_q"));
  outputs.push_back(absl::StrCat(prefix, "_scale_q"));
  shapes.push_back(absl::StrCat(dtype, "[", m, ",", n, "]"));
  shapes.push_back(absl::StrCat("f8e8m0fnu[", blocks, "]"));
  return text;
}

int64_t NumProducerParameters(Producer producer) {
  switch (producer) {
    case Producer::kRmsNorm:
      return 4;
    case Producer::kSiluMul:
    case Producer::kWeight:
      return 2;
    default:
      return 1;
  }
}

// A bf16 [rows, cols] value from `producer`, named `s`, quantized along its
// rows and/or its columns.
std::string PipelineModule(Producer producer, int64_t rows, int64_t cols,
                           bool row, bool column, PrimitiveType data,
                           const PipelineVariant& variant) {
  const std::string rc = absl::StrCat(rows, ",", cols);
  std::string text;
  switch (producer) {
    case Producer::kNone:
      text = absl::StrCat("  s = bf16[", rc, "] parameter(0)\n");
      break;
    case Producer::kRmsNorm:
      text = absl::StrReplaceAll(R"(  x = bf16[$RC] parameter(0)
  sumsq = f32[$ROWS] parameter(1)
  stack = f32[4,$COLS] parameter(2)
  index = s32[] parameter(3)
  zero_index = s32[] constant(0)
  w_slice = f32[1,$COLS] dynamic-slice(stack, index, zero_index), dynamic_slice_sizes={1,$COLS}
  w_f32 = f32[$COLS] reshape(w_slice)
  w = bf16[$COLS] convert(w_f32)
  n = f32[] constant($COLS)
  n_b = f32[$ROWS] broadcast(n), dimensions={}
  mean = f32[$ROWS] divide(sumsq, n_b)
  eps = f32[] constant(1e-05)
  eps_b = f32[$ROWS] broadcast(eps), dimensions={}
  var = f32[$ROWS] add(mean, eps_b)
  inv = f32[$ROWS] rsqrt(var)
  inv_b = f32[$RC] broadcast(inv), dimensions={0}
  x_f32 = f32[$RC] convert(x)
  normed = f32[$RC] multiply(x_f32, inv_b)
  normed_bf16 = bf16[$RC] convert(normed)
  w_b = bf16[$RC] broadcast(w), dimensions={1}
  s = bf16[$RC] multiply(w_b, normed_bf16)
)",
                                 {{"$RC", rc},
                                  {"$ROWS", absl::StrCat(rows)},
                                  {"$COLS", absl::StrCat(cols)}});
      break;
    case Producer::kSiluMul:
      text = absl::StrReplaceAll(R"(  a = bf16[$RC] parameter(0)
  b = bf16[$RC] parameter(1)
  one = bf16[] constant(1)
  one_b = bf16[$RC] broadcast(one), dimensions={}
  neg = bf16[$RC] negate(a)
  e = bf16[$RC] exponential(neg)
  den = bf16[$RC] add(e, one_b)
  sigmoid = bf16[$RC] divide(one_b, den)
  silu = bf16[$RC] multiply(a, sigmoid)
  s = bf16[$RC] multiply(silu, b)
)",
                                 {{"$RC", rc}});
      break;
    case Producer::kWeight:
      text = absl::StrReplaceAll(R"(  stack = f32[4,$RC] parameter(0)
  index = s32[] parameter(1)
  zero_index = s32[] constant(0)
  w_slice = f32[1,$RC] dynamic-slice(stack, index, zero_index, zero_index), dynamic_slice_sizes={1,$RC}
  w_f32 = f32[$RC] reshape(w_slice)
  s = bf16[$RC] convert(w_f32)
)",
                                 {{"$RC", rc}});
      break;
    case Producer::kHeadsTokensDim:
    case Producer::kHeadsDimTokens: {
      // rows = 2 * tokens, cols = 4 * 128.
      const bool tokens_minor = producer == Producer::kHeadsDimTokens;
      const int64_t tokens = rows / 2;
      text = absl::StrCat("  g = bf16[2,4,", tokens_minor ? 128 : tokens, ",",
                          tokens_minor ? tokens : 128,
                          "] parameter(0)\n  g_t = bf16[2,", tokens,
                          ",4,128] transpose(g), dimensions=",
                          tokens_minor ? "{0,3,1,2}" : "{0,2,1,3}",
                          "\n  s = bf16[", rc, "] reshape(g_t)\n");
      break;
    }
    case Producer::kTokensHeadsDim:
      text =
          absl::StrCat("  g = bf16[2,", rows / 2, ",", cols / 128,
                       ",128] parameter(0)\n  s = bf16[", rc, "] reshape(g)\n");
      break;
  }
  std::string tiny;
  if (variant.tiny_operand) {
    tiny = "tiny";
    absl::StrAppend(&text, "  tiny = f32[", rows, ",", cols / 32,
                    "] parameter(", NumProducerParameters(producer), ")\n");
  }
  std::vector<std::string> outputs, shapes;
  if (row) {
    absl::StrAppend(
        &text, JaxQuantize("r", "s", rows, cols, /*column=*/false, data, tiny,
                           variant.bf16_data_region, outputs, shapes));
  }
  if (column) {
    absl::StrAppend(
        &text,
        JaxQuantize("c", "s", rows, cols, /*column=*/true, data, /*tiny=*/"",
                    variant.bf16_data_region, outputs, shapes));
  }
  return absl::StrCat("HloModule pipeline\n", kReducers, "\nENTRY main {\n",
                      text, "  ROOT t = (", absl::StrJoin(shapes, ", "),
                      ") tuple(", absl::StrJoin(outputs, ", "), ")\n}\n");
}

// The arguments of `PipelineModule`, the quantized value's inputs of class `c`.
std::vector<Literal> PipelineArguments(Producer producer, int64_t rows,
                                       int64_t cols, InputClass c,
                                       bool column_blocks,
                                       const PipelineVariant& variant) {
  std::vector<Literal> arguments;
  const int seed = static_cast<int>(c);
  switch (producer) {
    case Producer::kNone:
      arguments.push_back(MakeSource(BF16, rows, cols, c, seed, column_blocks));
      break;
    case Producer::kRmsNorm:
      arguments.push_back(MakeSource(BF16, rows, cols, c, seed, column_blocks));
      arguments.push_back(MakeRandom(F32, {rows}, cols, seed + 1));
      arguments.push_back(MakeRandom(F32, {4, cols}, 1.0f, seed + 2));
      arguments.push_back(LiteralUtil::CreateR0<int32_t>(2));
      break;
    case Producer::kSiluMul:
      arguments.push_back(MakeSource(BF16, rows, cols, c, seed, column_blocks));
      arguments.push_back(MakeRandom(BF16, {rows, cols}, 1.0f, seed + 1));
      break;
    case Producer::kHeadsTokensDim:
    case Producer::kHeadsDimTokens: {
      // The blocks of the gradient's own layout, not of the [b * s, h * d]
      // view.
      const bool tokens_minor = producer == Producer::kHeadsDimTokens;
      Literal flat =
          MakeSource(BF16, rows * cols / (tokens_minor ? rows / 2 : 128),
                     tokens_minor ? rows / 2 : 128, c, seed, column_blocks);
      arguments.push_back(*flat.Reshape({2, 4, tokens_minor ? 128 : rows / 2,
                                         tokens_minor ? rows / 2 : 128}));
      break;
    }
    case Producer::kTokensHeadsDim:
      arguments.push_back(*MakeSource(BF16, rows, cols, c, seed, column_blocks)
                               .Reshape({2, rows / 2, cols / 128, 128}));
      break;
    case Producer::kWeight: {
      Literal slice = MakeSource(F32, rows, cols, c, seed, column_blocks);
      Literal stack(ShapeUtil::MakeShape(F32, {4, rows, cols}));
      absl::Span<float> values = stack.data<float>();
      for (int64_t i = 0; i < 4; ++i) {
        absl::c_copy(slice.data<float>(), values.begin() + i * rows * cols);
      }
      arguments.push_back(std::move(stack));
      arguments.push_back(LiteralUtil::CreateR0<int32_t>(2));
      break;
    }
  }
  if (variant.tiny_operand) {
    Literal tiny(ShapeUtil::MakeShape(F32, {rows, cols / 32}));
    absl::Span<float> values = tiny.data<float>();
    std::fill(values.begin(), values.end(), std::ldexp(1.0f, -127));
    arguments.push_back(std::move(tiny));
  }
  return arguments;
}

const HloFusionInstruction* FindBlockScaledQuantizeFusion(
    const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kFusion) {
        continue;
      }
      absl::StatusOr<GpuBackendConfig> config =
          instr->backend_config<GpuBackendConfig>();
      if (config.ok() && config->fusion_backend_config().kind() ==
                             kBlockScaledQuantizeFusionKind) {
        return Cast<HloFusionInstruction>(instr);
      }
    }
  }
  return nullptr;
}

int64_t CountBlockScaledQuantizeFusions(const HloModule& module) {
  int64_t count = 0;
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kFusion) {
        continue;
      }
      absl::StatusOr<GpuBackendConfig> config =
          instr->backend_config<GpuBackendConfig>();
      count += config.ok() && config->fusion_backend_config().kind() ==
                                  kBlockScaledQuantizeFusionKind;
    }
  }
  return count;
}

class BlockScaledQuantizeFlagTest : public HloPjRtGpuTestBase {};

TEST_F(BlockScaledQuantizeFlagTest, LeavesModuleWithoutQuantizationUnchanged) {
  constexpr absl::string_view kHlo = R"(
HloModule no_quantization

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}

ENTRY main {
  x = bf16[256,512] parameter(0)
  w = bf16[512,256] parameter(1)
  x_f32 = f32[256,512] convert(x)
  sq = f32[256,512] multiply(x_f32, x_f32)
  zero = f32[] constant(0)
  sum = f32[256] reduce(sq, zero), dimensions={1}, to_apply=add_f32
  sum_b = f32[256,512] broadcast(sum), dimensions={0}
  normed = f32[256,512] divide(x_f32, sum_b)
  normed_bf16 = bf16[256,512] convert(normed)
  y = bf16[256,256] dot(normed_bf16, w), lhs_contracting_dims={1},
      rhs_contracting_dims={0}
  y_f32 = f32[256,256] convert(y)
  y_abs = f32[256,256] abs(y_f32)
  ninf = f32[] constant(-inf)
  y_max = f32[256] reduce(y_abs, ninf), dimensions={1}, to_apply=max_f32
  y_max_b = f32[256,256] broadcast(y_max), dimensions={0}
  ROOT out = f32[256,256] divide(y_f32, y_max_b)
})";
  HloModuleConfig off_config = GetModuleConfigForTest();
  HloModuleConfig on_config = GetModuleConfigForTest();
  on_config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_block_scaled_quantize_fusion(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> off,
                       GetOptimizedModule(kHlo, off_config));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> on,
                       GetOptimizedModule(kHlo, on_config));
  EXPECT_EQ(off->ToString(HloPrintOptions::Fingerprint()),
            on->ToString(HloPrintOptions::Fingerprint()));
}

// The weight shards of FSDP: a [512, cols] slice of an f32 stack, rounded to
// bf16, with column blocks whose data keep the source's order, and with row
// blocks too if `row`; the step 0 form of an 8-way sharded 8B weight.
std::string ShardBody(PrimitiveType data, int64_t cols, bool row, bool fused) {
  const bool fp4 = data == F4E2M1FN;
  std::string text = R"(
  stack = f32[2,512,$N] parameter(0)
  index = s32[] parameter(1)
  zero_index = s32[] constant(0)
  ds = f32[1,512,$N] dynamic-slice(stack, index, zero_index, zero_index), dynamic_slice_sizes={1,512,$N}
  ds_bf16 = bf16[1,512,$N] convert(ds)
  abs = f32[1,512,$N] abs(ds)
  abs_bf16 = bf16[1,512,$N] convert(abs)
  ninf = bf16[] constant(-inf)
  ninf_f32 = f32[] convert(ninf)
  lo = f32[] constant(-$MAX)
  hi = f32[] constant($MAX)
  inv = f32[] constant($INV)
  k_mant = u32[] constant(8388607)
  k0 = u32[] constant(0)
  k23 = u32[] constant(23)
  k254 = u32[] constant(254)
  k_half = u32[] constant(4194304)
  k1 = u32[] constant(1)
  f0 = f32[] constant(0)
  tiny = f32[] constant(5.87747175e-39)
)";
  // The encode and decode of `$P_amax_f32` over the blocks `$B`.
  constexpr absl::string_view kScale = R"(
  $P_inv_b = f32[$B] broadcast(inv), dimensions={}
  $P_ratio = f32[$B] multiply($P_amax_f32, $P_inv_b)
  $P_bits = u32[$B] $BITCAST($P_ratio)
  $P_mant_mask = u32[$B] broadcast(k_mant), dimensions={}
  $P_mant = u32[$B] and($P_bits, $P_mant_mask)
  $P_zero = u32[$B] broadcast(k0), dimensions={}
  $P_gt = pred[$B] compare($P_mant, $P_zero), direction=GT
  $P_s23 = u32[$B] broadcast(k23), dimensions={}
  $P_exp = u32[$B] shift-right-logical($P_bits, $P_s23)
  $P_e254 = u32[$B] broadcast(k254), dimensions={}
  $P_ne = pred[$B] compare($P_exp, $P_e254), direction=NE
  $P_and0 = pred[$B] and($P_gt, $P_ne)
  $P_eq0 = pred[$B] compare($P_exp, $P_zero), direction=EQ
  $P_half = u32[$B] broadcast(k_half), dimensions={}
  $P_le = pred[$B] compare($P_mant, $P_half), direction=LE
  $P_and1 = pred[$B] and($P_eq0, $P_le)
  $P_not = pred[$B] not($P_and1)
  $P_round_up = pred[$B] and($P_and0, $P_not)
  $P_one = u32[$B] broadcast(k1), dimensions={}
  $P_exp1 = u32[$B] add($P_exp, $P_one)
  $P_exp_sel = u32[$B] select($P_round_up, $P_exp1, $P_exp)
  $P_scale = u8[$B] convert($P_exp_sel)
  $P_scale32 = u32[$B] convert($P_scale)
  $P_shl = u32[$B] shift-left($P_scale32, $P_s23)
  $P_pow = f32[$B] $BITCAST($P_shl)
  $P_fzero = f32[$B] broadcast(f0), dimensions={}
  $P_is_zero = pred[$B] compare($P_pow, $P_fzero), direction=EQ
  $P_tiny_b = f32[$B] broadcast(tiny), dimensions={}
  $P_decoded = f32[$B] select($P_is_zero, $P_tiny_b, $P_pow)
  $P_bs = f32[$V] broadcast($P_decoded), dimensions={$BDIMS}
  $P_q = f32[$V] divide($P_x, $P_bs)
  $P_lo_b = f32[$V] broadcast(lo), dimensions={}
  $P_hi_b = f32[$V] broadcast(hi), dimensions={}
  $P_cl = f32[$V] clamp($P_lo_b, $P_q, $P_hi_b)
  $P_data = $DTYPE[$V]$DLAYOUT convert($P_cl)
)";
  auto scale = [&](absl::string_view prefix, absl::string_view blocks,
                   absl::string_view view, absl::string_view dims) {
    return absl::StrReplaceAll(kScale, {{"$P", prefix},
                                        {"$BITCAST", "$BITCAST"},
                                        {"$BDIMS", dims},
                                        {"$B", blocks},
                                        {"$V", view}});
  };
  std::string roots, shapes;
  if (row) {
    absl::StrAppend(&text, R"(
  r_xs = bf16[512,$NB,32] $RESHAPE(ds_bf16)
  r_x = f32[512,$NB,32] convert(r_xs)
  r_ab = bf16[512,$NB,32] $RESHAPE(abs_bf16)
  r_abf = f32[512,$NB,32] convert(r_ab)
  r_amax = f32[512,$NB] reduce(r_abf, ninf_f32), dimensions={2}, to_apply=max_f32
  r_amax_bf16 = bf16[512,$NB] convert(r_amax)
  r_amax_f32 = f32[512,$NB] convert(r_amax_bf16)
)",
                    scale("r", "512,$NB", "512,$NB,32", "0,1"));
    roots = "r_data, ";
    shapes = "$DTYPE[512,$NB,32]$DLAYOUT, ";
  }
  absl::StrAppend(&text, R"(
  c_xs = bf16[16,32,$N] $RESHAPE(ds_bf16)
  c_x = f32[16,32,$N] convert(c_xs)
)",
                  fused ? R"(
  c_ab0 = bf16[1,$N,16,32]{1,3,2,0} bitcast(abs_bf16)
  c_ab1 = bf16[$N,16,32]{0,2,1} bitcast(c_ab0)
  c_ab = bf16[16,32,$N] bitcast(c_ab1)
)"
                        : R"(
  c_ab = bf16[16,32,$N] reshape(abs_bf16)
)",
                  R"(
  c_abf = f32[16,32,$N] convert(c_ab)
  c_amax = f32[16,$N] reduce(c_abf, ninf_f32), dimensions={1}, to_apply=max_f32
  c_amax_bf16 = bf16[16,$N] convert(c_amax)
  c_amax_f32 = f32[16,$N] convert(c_amax_bf16)
)",
                  scale("c", "16,$N", "16,32,$N", "0,2"));
  absl::StrAppend(&roots, "c_data, ", row ? "r_scale, " : "", "c_scale");
  absl::StrAppend(&shapes, "$DTYPE[16,32,$N]$DLAYOUT, ",
                  row ? "u8[512,$NB], " : "", "u8[16,$N]");
  absl::StrAppend(&text, "  ROOT t = (", shapes, ") tuple(", roots, ")\n");
  return absl::StrReplaceAll(
      text, {{"$NB", absl::StrCat(cols / 32)},
             {"$N", absl::StrCat(cols)},
             {"$MAX", fp4 ? "6" : "448"},
             {"$INV", fp4 ? "0.166666672" : "0.00223214296"},
             {"$DTYPE", primitive_util::LowercasePrimitiveTypeName(data)},
             {"$DLAYOUT", fp4 && fused ? "{2,1,0:E(4)}" : ""},
             {"$RESHAPE", fused ? "bitcast" : "reshape"},
             {"$BITCAST", fused ? "bitcast" : "bitcast-convert"}});
}

constexpr absl::string_view kMaxF32 = R"(
max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}
)";

std::string ShardFusedModule(PrimitiveType data, int64_t cols, bool row) {
  std::string body = ShardBody(data, cols, row, /*fused=*/true);
  std::string root_shape = body.substr(body.find("ROOT t = ") + 9);
  root_shape = root_shape.substr(0, root_shape.find(" tuple("));
  return absl::StrCat(
      "HloModule shard\n", kMaxF32, "\nfused_quantize {", body,
      "}\n\nENTRY main {\n  stack = f32[2,512,", cols,
      "] parameter(0)\n  index = s32[] parameter(1)\n  ROOT f = ", root_shape,
      " fusion(stack, index), kind=kCustom, calls=fused_quantize, "
      "backend_config={\"fusion_backend_config\":{\"kind\":\"__block_"
      "scaled_quantize\"}}\n}\n");
}

std::string ShardUnfusedModule(PrimitiveType data, int64_t cols, bool row) {
  return absl::StrCat("HloModule shard\n", kMaxF32, "\nENTRY main {",
                      ShardBody(data, cols, row, /*fused=*/false), "}\n");
}

class BlockScaledQuantizeShardTest
    : public HloPjRtGpuTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, int64_t, bool>> {};

// 16 block rows; 256 columns take the 8-wave tile, 192 the 4-wave one.
TEST_P(BlockScaledQuantizeShardTest, ColumnDataInSourceOrder) {
  auto [data, cols, row] = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> fused,
      ParseAndReturnVerifiedModule(ShardFusedModule(data, cols, row)));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> unfused,
      ParseAndReturnVerifiedModule(ShardUnfusedModule(data, cols, row)));
  // The fusion keeps the bf16 roundings of its HLO, so today's pipeline must
  // keep them too.
  unfused->mutable_config()
      .mutable_debug_options()
      .set_xla_allow_excess_precision(false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> fused_executable,
                       CreateExecutable(std::move(fused),
                                        /*run_hlo_passes=*/false));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> unfused_executable,
                       CreateExecutable(std::move(unfused),
                                        /*run_hlo_passes=*/true));
  constexpr bool kFills[] = {false, true};
  for (auto [c, column_fill] :
       Cross(absl::MakeConstSpan(kAllClasses), absl::MakeConstSpan(kFills))) {
    SCOPED_TRACE(absl::StrCat(ClassName(c), column_fill ? " in columns" : ""));
    Literal stack(ShapeUtil::MakeShape(F32, {2, 512, cols}));
    for (int64_t slice = 0; slice < 2; ++slice) {
      Literal source = MakeSource(F32, 512, cols, c,
                                  static_cast<int>(c) * 2 + slice, column_fill);
      absl::Span<const float> values = source.data<float>();
      absl::c_copy(values, stack.data<float>().begin() + slice * 512 * cols);
    }
    std::vector<Literal> arguments;
    arguments.push_back(std::move(stack));
    arguments.push_back(LiteralUtil::CreateR0<int32_t>(1));
    ASSERT_OK_AND_ASSIGN(Literal expected,
                         test_runner().ExecuteWithExecutable(
                             unfused_executable.get(), arguments));
    ASSERT_OK_AND_ASSIGN(
        Literal actual,
        test_runner().ExecuteWithExecutable(fused_executable.get(), arguments));
    std::vector<Literal> expected_outputs = expected.DecomposeTuple();
    std::vector<Literal> actual_outputs = actual.DecomposeTuple();
    for (int64_t i = 0; i < expected_outputs.size(); ++i) {
      EXPECT_TRUE(
          LiteralTestUtil::Equal(expected_outputs[i], actual_outputs[i]))
          << "output " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    BlockScaledQuantizeShardTests, BlockScaledQuantizeShardTest,
    ::testing::Combine(::testing::Values(F8E4M3FN, F4E2M1FN),
                       ::testing::Values(192, 256), ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<PrimitiveType, int64_t, bool>>&
           info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          "_", std::get<1>(info.param), std::get<2>(info.param) ? "_pair" : "");
    });

// The hipBLASLt 32x8 swizzle of `scale` as `SwizzleHipblasLtScale32x8` builds
// it, which the swizzle folding pass recognizes.
HloInstruction* Swizzle32x8(HloInstruction* scale) {
  const xla::Shape& shape = scale->shape();
  const int64_t rows = shape.dimensions(0);
  const int64_t cols = shape.dimensions(1);
  HloComputation::Builder builder("hipblaslt_mx_scale_32x8");
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "scale"));
  HloInstruction* split =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShapeWithDescendingLayout(
              F8E8M0FNU, {rows / 32, 2, 16, cols / 8, 2, 4}),
          input));
  HloInstruction* transpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShapeWithDescendingLayout(
              F8E8M0FNU, {rows / 32, cols / 8, 4, 16, 2, 2}),
          split, {0, 3, 5, 2, 4, 1}));
  HloInstruction* result = builder.AddInstruction(
      HloInstruction::CreateReshape(shape, transpose));
  HloComputation* computation =
      scale->GetModule()->AddEmbeddedComputation(builder.Build(result));
  return scale->parent()->AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {scale}, computation));
}

// Reads the u8 scale results of the fusion at the root of `module` as E8M0 and
// swizzles each one that the hipBLASLt 32x8 layout supports.
void SwizzleScaleResults(HloModule* module) {
  HloComputation* entry = module->entry_computation();
  HloInstruction* fusion = entry->root_instruction();
  std::vector<HloInstruction*> results;
  for (int64_t i = 0; i < fusion->shape().tuple_shapes().size(); ++i) {
    HloInstruction* result = entry->AddInstruction(
        HloInstruction::CreateGetTupleElement(fusion, i));
    const xla::Shape& shape = result->shape();
    if (shape.element_type() == U8 && shape.dimensions(0) % 32 == 0 &&
        shape.dimensions(1) % 8 == 0) {
      result = Swizzle32x8(
          entry->AddInstruction(HloInstruction::CreateBitcast(
              ShapeUtil::ChangeElementType(shape, F8E8M0FNU), result)));
    }
    results.push_back(result);
  }
  entry->set_root_instruction(
      entry->AddInstruction(HloInstruction::CreateTuple(results)),
      /*accept_different_shape=*/true);
  *module->mutable_entry_computation_layout() =
      ComputationLayout(entry->ComputeProgramShape(), /*ignore_layouts=*/false);
}

// Stores the u8[a, b] scale results of `computation` through reshape,
// transpose and reshape, in an order other than the hipBLASLt one.
absl::Status PermuteScaleResults(HloComputation* computation) {
  HloInstruction* root = computation->root_instruction();
  for (int64_t i = 0; i < root->operand_count(); ++i) {
    HloInstruction* scale = root->mutable_operand(i);
    if (scale->shape().element_type() != U8) {
      continue;
    }
    const int64_t a = scale->shape().dimensions(0);
    const int64_t b = scale->shape().dimensions(1);
    HloInstruction* split = computation->AddInstruction(
        HloInstruction::CreateReshape(ShapeUtil::MakeShapeWithDescendingLayout(
                                          U8, {a / 4, 4, b / 2, 2}),
                                      scale));
    HloInstruction* transpose =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShapeWithDescendingLayout(U8, {b / 2, a / 4, 2, 4}),
            split, {2, 0, 3, 1}));
    ABSL_RETURN_IF_ERROR(root->ReplaceOperandWith(
        i, computation->AddInstruction(
               HloInstruction::CreateReshape(scale->shape(), transpose))));
  }
  return absl::OkStatus();
}

// Scale results stored in the hipBLASLt 32x8 order by the emitter, after the
// swizzle folding, against natural-order scales and the separate swizzle.
class BlockScaledQuantizeSwizzleTest : public BlockScaledQuantizeDeviceTest {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options =
        BlockScaledQuantizeDeviceTest::GetDebugOptionsForTest();
    options.set_xla_gpu_experimental_enable_fusion_autotuner(false);
    return options;
  }

  void RunAndCompareSwizzled(const Quantize& q,
                             absl::Span<const InputClass> classes,
                             absl::Span<const bool> column_fills = {false}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> natural,
                         ParseAndReturnVerifiedModule(FusedModule(q)));
    SwizzleScaleResults(natural.get());
    std::unique_ptr<HloModule> folded = natural->Clone();
    ASSERT_THAT(BlockScaledQuantizeSwizzleFolding().Run(folded.get()),
                absl_testing::IsOkAndHolds(true));
    EXPECT_FALSE(absl::c_any_of(folded->entry_computation()->instructions(),
                                [](const HloInstruction* instr) {
                                  return instr->IsLoopFusion();
                                }));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> natural_executable,
                         CreateExecutable(std::move(natural),
                                          /*run_hlo_passes=*/false));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> folded_executable,
                         CreateExecutable(std::move(folded),
                                          /*run_hlo_passes=*/false));
    for (auto [c, column_fill] : Cross(classes, column_fills)) {
      SCOPED_TRACE(
          absl::StrCat(ClassName(c), column_fill ? " in columns" : ""));
      std::vector<Literal> arguments;
      arguments.push_back(MakeSource(q.source, q.rows, q.cols, c,
                                     static_cast<int>(c), column_fill));
      for (const std::vector<int64_t>& shape :
           q.tiny_operand ? TinyShapes(q)
                          : std::vector<std::vector<int64_t>>{}) {
        Literal tiny(ShapeUtil::MakeShape(F32, shape));
        absl::Span<float> values = tiny.data<float>();
        std::fill(values.begin(), values.end(), std::ldexp(1.0f, -127));
        arguments.push_back(std::move(tiny));
      }
      ASSERT_OK_AND_ASSIGN(Literal expected,
                           test_runner().ExecuteWithExecutable(
                               natural_executable.get(), arguments));
      ASSERT_OK_AND_ASSIGN(Literal actual,
                           test_runner().ExecuteWithExecutable(
                               folded_executable.get(), arguments));
      EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
    }
  }
};

TEST_P(BlockScaledQuantizeSwizzleTest, Row) {
  auto [source, data] = GetParam();
  RunAndCompareSwizzled(Quantize{source, data, 128, 256}, kAllClasses);
}

// 96 rows leave the last row workgroup partial; a 32x8 scale needs its rows to
// be a multiple of 32, so no grid is partial inside a block of 32 rows.
TEST_P(BlockScaledQuantizeSwizzleTest, RowPartialGrid) {
  auto [source, data] = GetParam();
  RunAndCompareSwizzled(Quantize{source, data, 96, 512},
                        {InputClass::kRandom, InputClass::kMixed});
}

TEST_P(BlockScaledQuantizeSwizzleTest, RowWithScaleOperand) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 128, 256};
  q.tiny_operand = true;
  RunAndCompareSwizzled(
      q, {InputClass::kRandom, InputClass::kScaleSubstitution,
          InputClass::kMixed});
}

TEST_P(BlockScaledQuantizeSwizzleTest, ColumnNormalized) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 256, 256};
  q.row = false;
  q.column = Column::kNormalized;
  RunAndCompareSwizzled(q, kAllClasses, {false, true});
}

TEST_P(BlockScaledQuantizeSwizzleTest, ColumnTransposeFirst) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 256, 256};
  q.row = false;
  q.column = Column::kTransposeFirst;
  RunAndCompareSwizzled(q, kAllClasses, {false, true});
}

// 320 columns leave the last column tile partial.
TEST_P(BlockScaledQuantizeSwizzleTest, ColumnPartialTile) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 256, 320};
  q.row = false;
  q.column = Column::kNormalized;
  RunAndCompareSwizzled(q, {InputClass::kRandom, InputClass::kMixed},
                        {false, true});
}

TEST_P(BlockScaledQuantizeSwizzleTest, BothNormalized) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 256, 256};
  q.column = Column::kNormalized;
  RunAndCompareSwizzled(q, kAllClasses, {false, true});
}

TEST_P(BlockScaledQuantizeSwizzleTest, BothTransposeFirst) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 256, 256};
  q.column = Column::kTransposeFirst;
  RunAndCompareSwizzled(q, kAllClasses, {false, true});
}

TEST_P(BlockScaledQuantizeSwizzleTest, BothWithScaleOperand) {
  auto [source, data] = GetParam();
  Quantize q{source, data, 256, 256};
  q.column = Column::kNormalized;
  q.tiny_operand = true;
  RunAndCompareSwizzled(
      q,
      {InputClass::kRandom, InputClass::kScaleSubstitution, InputClass::kMixed},
      {false, true});
}

INSTANTIATE_TEST_SUITE_P(
    BlockScaledQuantizeSwizzleTests, BlockScaledQuantizeSwizzleTest,
    ::testing::Combine(::testing::Values(BF16, F32),
                       ::testing::Values(F8E4M3FN, F4E2M1FN)),
    [](const ::testing::TestParamInfo<std::tuple<PrimitiveType, PrimitiveType>>&
           info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          "_",
          primitive_util::LowercasePrimitiveTypeName(std::get<1>(info.param)));
    });

// The shards' scale results in another order, against today's pipeline with
// the same order.
class BlockScaledQuantizeShardPermutationTest
    : public BlockScaledQuantizeShardTest {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options =
        BlockScaledQuantizeShardTest::GetDebugOptionsForTest();
    options.set_xla_gpu_experimental_enable_fusion_autotuner(false);
    return options;
  }
};

TEST_P(BlockScaledQuantizeShardPermutationTest, PermutedScales) {
  auto [data, cols, row] = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> fused,
      ParseAndReturnVerifiedModule(ShardFusedModule(data, cols, row)));
  ASSERT_OK(PermuteScaleResults(fused->entry_computation()
                                    ->root_instruction()
                                    ->fused_instructions_computation()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> unfused,
      ParseAndReturnVerifiedModule(ShardUnfusedModule(data, cols, row)));
  ASSERT_OK(PermuteScaleResults(unfused->entry_computation()));
  unfused->mutable_config()
      .mutable_debug_options()
      .set_xla_allow_excess_precision(false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> fused_executable,
                       CreateExecutable(std::move(fused),
                                        /*run_hlo_passes=*/false));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> unfused_executable,
                       CreateExecutable(std::move(unfused),
                                        /*run_hlo_passes=*/true));
  for (InputClass c : {InputClass::kRandom, InputClass::kMixed,
                       InputClass::kScaleSubstitution}) {
    SCOPED_TRACE(ClassName(c));
    Literal stack(ShapeUtil::MakeShape(F32, {2, 512, cols}));
    for (int64_t slice = 0; slice < 2; ++slice) {
      Literal source =
          MakeSource(F32, 512, cols, c, static_cast<int>(c) * 2 + slice);
      absl::c_copy(source.data<float>(),
                   stack.data<float>().begin() + slice * 512 * cols);
    }
    std::vector<Literal> arguments;
    arguments.push_back(std::move(stack));
    arguments.push_back(LiteralUtil::CreateR0<int32_t>(1));
    ASSERT_OK_AND_ASSIGN(Literal expected,
                         test_runner().ExecuteWithExecutable(
                             unfused_executable.get(), arguments));
    ASSERT_OK_AND_ASSIGN(
        Literal actual,
        test_runner().ExecuteWithExecutable(fused_executable.get(), arguments));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
  }
}

INSTANTIATE_TEST_SUITE_P(
    BlockScaledQuantizeShardPermutationTests,
    BlockScaledQuantizeShardPermutationTest,
    ::testing::Combine(::testing::Values(F8E4M3FN, F4E2M1FN),
                       ::testing::Values(192, 256), ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<PrimitiveType, int64_t, bool>>&
           info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          "_", std::get<1>(info.param), std::get<2>(info.param) ? "_pair" : "");
    });

// An optimized module and the executable that owns it.
using OptimizedModule =
    std::pair<const HloModule*, std::unique_ptr<OpaqueExecutable>>;

// The acceptance test: the same HLO compiled through all HLO passes with the
// block-scaled quantize fusion on and off, with and without excess precision.
class BlockScaledQuantizePipelineTest
    : public HloPjRtGpuTestBase,
      public ::testing::WithParamInterface<std::tuple<PrimitiveType, bool>> {
 protected:
  void RunAndCompareThroughPipeline(Producer producer, int64_t rows,
                                    int64_t cols, bool row, bool column,
                                    absl::Span<const InputClass> classes,
                                    const PipelineVariant& variant = {}) {
    auto [data, excess_precision] = GetParam();
    const std::string hlo =
        PipelineModule(producer, rows, cols, row, column, data, variant);
    HloModuleConfig off_config = GetModuleConfigForTest();
    off_config.mutable_debug_options().set_xla_allow_excess_precision(
        excess_precision);
    HloModuleConfig on_config = off_config;
    on_config.mutable_debug_options()
        .set_xla_gpu_experimental_enable_block_scaled_quantize_fusion(true);
    ASSERT_OK_AND_ASSIGN(OptimizedModule off,
                         GetOptimizedModuleForExecutable(hlo, off_config));
    ASSERT_OK_AND_ASSIGN(OptimizedModule on,
                         GetOptimizedModuleForExecutable(hlo, on_config));
    EXPECT_EQ(CountBlockScaledQuantizeFusions(*off.first), 0);
    ASSERT_EQ(CountBlockScaledQuantizeFusions(*on.first), 1)
        << on.first->ToString();
    // A pair is one fusion of both orientations, not a row alone.
    EXPECT_EQ(FindBlockScaledQuantizeFusion(*on.first)
                  ->fused_expression_root()
                  ->operand_count(),
              row && column ? 4 : 2);
    constexpr bool kFills[] = {false, true};
    for (auto [c, column_fill] :
         Cross(classes, absl::MakeConstSpan(kFills, column ? 2 : 1))) {
      SCOPED_TRACE(
          absl::StrCat(ClassName(c), column_fill ? " in columns" : ""));
      std::vector<Literal> arguments =
          PipelineArguments(producer, rows, cols, c, column_fill, variant);
      ASSERT_OK_AND_ASSIGN(
          Literal expected,
          test_runner().ExecuteWithExecutable(off.second.get(), arguments));
      ASSERT_OK_AND_ASSIGN(Literal actual, test_runner().ExecuteWithExecutable(
                                               on.second.get(), arguments));
      EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
    }
  }
};

int64_t CountScaleSwizzles(const HloModule& module) {
  int64_t count = 0;
  for (const HloInstruction* instr :
       module.entry_computation()->instructions()) {
    count += instr->IsLoopFusion() &&
             absl::StartsWith(instr->name(), "hipblaslt_mx_scale_32x8");
  }
  return count;
}

class BlockScaledQuantizeScaledDotTest : public HloPjRtGpuTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
    options.set_xla_gpu_experimental_enable_fusion_autotuner(false);
    options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
    options.set_xla_gpu_enable_triton_gemm(true);
    options.set_xla_gpu_blas_max_algorithms(4);
    options.set_xla_autotuner_preferred_backend(autotuner::HIPBLASLT);
    options.set_xla_gpu_experimental_hipblaslt_mx_scale_layout(
        DebugOptions::HIPBLASLT_MX_SCALE_LAYOUT_PRESWIZZLED_32X8);
    return options;
  }
};

// An MX scaled-dot on hipBLASLt with 32x8 scales, its lhs and lhs scale from a
// quantize fusion: as the autotuner leaves it, with the lhs scale swizzled by
// a kernel of its own, and after the swizzle folding.
TEST_F(BlockScaledQuantizeScaledDotTest, SwizzleFoldedIntoQuantizeFusion) {
  const auto* rocm =
      device_description().gpu_compute_capability().rocm_compute_capability();
  if (rocm == nullptr || !rocm->gfx9_mi350()) {
    GTEST_SKIP() << "32x8 hipBLASLt scales require gfx950.";
  }
  constexpr int64_t kM = 256, kN = 256, kK = 512;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(R"hlo(
    ENTRY main {
      a = f8e4m3fn[256,512] parameter(0)
      b = f8e4m3fn[256,512] parameter(1)
      sa = f8e8m0fnu[256,16] parameter(2)
      sb = f8e8m0fnu[256,16] parameter(3)
      ROOT dot = f32[256,256] scaled-dot(a, b, sa, sb),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })hlo"));
  ASSERT_EQ(CountScaleSwizzles(*optimized), 2) << optimized->ToString();

  // Parameters 0 and 2 become the data and scale results of a quantize fusion
  // of a bf16 parameter 0; parameter 2 stays, unused.
  std::string quantize = FusedModule(Quantize{BF16, F8E4M3FN, kM, kK});
  quantize = absl::StrReplaceAll(quantize.substr(0, quantize.find("ENTRY")),
                                 {{"HloModule fused\n", ""},
                                  {"max_bf16", "e2e_max_bf16"},
                                  {"max_f32", "e2e_max_f32"},
                                  {"fused_quantize", "e2e_quantize"}});
  std::string text = absl::StrReplaceAll(
      optimized->ToString(), {{"is_scheduled=true, ", ""}});
  ASSERT_TRUE(RE2::Replace(
      &text, R"(entry_computation_layout=\{\(f8e4m3fn\[256,512\]\{1,0\})",
      "entry_computation_layout={(bf16[256,512]{1,0}"));
  text.insert(text.find("\nENTRY ") + 1, quantize);
  const RE2 lhs(
      R"((%[a-z_.0-9]+) = f8e4m3fn\[256,512\]\{1,0\} parameter\(0\))");
  const RE2 lhs_scale(
      R"((%[a-z_.0-9]+) = f8e8m0fnu\[256,16\]\{1,0\} parameter\(2\))");
  std::string lhs_name, lhs_scale_name;
  ASSERT_TRUE(RE2::PartialMatch(text, lhs, &lhs_name)) << text;
  ASSERT_TRUE(RE2::PartialMatch(text, lhs_scale, &lhs_scale_name)) << text;
  const size_t body = text.find("{\n", text.find("\nENTRY ")) + 2;
  text.insert(body,
              "  %x = bf16[256,512]{1,0} parameter(0)\n  %q = (f8e4m3fn[256,16,"
              "32]{2,1,0}, u8[256,16]{1,0}) fusion(%x), kind=kCustom, "
              "calls=%e2e_quantize, backend_config={\"fusion_backend_config\":"
              "{\"kind\":\"__block_scaled_quantize\"}}\n  %q_data = "
              "f8e4m3fn[256,16,32]{2,1,0} get-tuple-element(%q), index=0\n  "
              "%q_scale = u8[256,16]{1,0} get-tuple-element(%q), index=1\n");
  ASSERT_TRUE(RE2::Replace(
      &text, lhs,
      absl::StrCat(lhs_name, " = f8e4m3fn[256,512]{1,0} bitcast(%q_data)")));
  ASSERT_TRUE(RE2::Replace(
      &text, lhs_scale,
      absl::StrCat("%sa_unused = f8e8m0fnu[256,16]{1,0} parameter(2)\n  ",
                   lhs_scale_name,
                   " = f8e8m0fnu[256,16]{1,0} bitcast(%q_scale)")));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> natural,
                       ParseAndReturnVerifiedModule(text));
  std::unique_ptr<HloModule> folded = natural->Clone();
  ASSERT_THAT(BlockScaledQuantizeSwizzleFolding().Run(folded.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_EQ(CountScaleSwizzles(*natural), 2);
  EXPECT_EQ(CountScaleSwizzles(*folded), 1) << folded->ToString();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> natural_executable,
                       CreateExecutable(std::move(natural),
                                        /*run_hlo_passes=*/false));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> folded_executable,
                       CreateExecutable(std::move(folded),
                                        /*run_hlo_passes=*/false));

  std::mt19937 rng(7);
  std::uniform_int_distribution<int> magnitude(0, 0x7e);
  std::uniform_int_distribution<int> exponent(124, 130);
  for (InputClass c : {InputClass::kRandom, InputClass::kMixed,
                       InputClass::kScaleSubstitution}) {
    SCOPED_TRACE(ClassName(c));
    Literal rhs(ShapeUtil::MakeShape(F8E4M3FN, {kN, kK}));
    for (tsl::float8_e4m3fn& value : rhs.data<tsl::float8_e4m3fn>()) {
      value = tsl::float8_e4m3fn::FromRep(
          static_cast<uint8_t>(magnitude(rng) | (rng() & 0x80)));
    }
    Literal rhs_scale(ShapeUtil::MakeShape(F8E8M0FNU, {kN, kK / 32}));
    for (tsl::float8_e8m0fnu& value : rhs_scale.data<tsl::float8_e8m0fnu>()) {
      value =
          tsl::float8_e8m0fnu::FromRep(static_cast<uint8_t>(exponent(rng)));
    }
    std::vector<Literal> arguments;
    arguments.push_back(MakeSource(BF16, kM, kK, c, static_cast<int>(c)));
    arguments.push_back(std::move(rhs));
    arguments.push_back(
        Literal(ShapeUtil::MakeShape(F8E8M0FNU, {kM, kK / 32})));
    arguments.push_back(std::move(rhs_scale));
    ASSERT_OK_AND_ASSIGN(Literal expected,
                         test_runner().ExecuteWithExecutable(
                             natural_executable.get(), arguments));
    ASSERT_OK_AND_ASSIGN(Literal actual,
                         test_runner().ExecuteWithExecutable(
                             folded_executable.get(), arguments));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
  }
}

constexpr InputClass kPipelineClasses[] = {
    InputClass::kRandom,
    InputClass::kZeros,
    InputClass::kSubnormalAmax,
    InputClass::kAmaxIs448TimesPowerOfTwo,
    InputClass::kAmaxIs6TimesPowerOfTwo,
    InputClass::kInf,
    InputClass::kNan,
    InputClass::kHugeAmax,
    InputClass::kScaleSubstitution,
    InputClass::kMixed};

TEST_P(BlockScaledQuantizePipelineTest, RowOfParameter) {
  RunAndCompareThroughPipeline(Producer::kNone, 256, 512, true, false,
                               kPipelineClasses);
}

TEST_P(BlockScaledQuantizePipelineTest, RowOfRmsNorm) {
  RunAndCompareThroughPipeline(Producer::kRmsNorm, 256, 512, true, false,
                               {InputClass::kRandom, InputClass::kMixed});
}

TEST_P(BlockScaledQuantizePipelineTest, RowOfSiluMul) {
  RunAndCompareThroughPipeline(Producer::kSiluMul, 256, 512, true, false,
                               {InputClass::kRandom, InputClass::kMixed});
}

TEST_P(BlockScaledQuantizePipelineTest, RowOfWeight) {
  RunAndCompareThroughPipeline(Producer::kWeight, 256, 512, true, false,
                               kPipelineClasses);
}

TEST_P(BlockScaledQuantizePipelineTest, ColumnOfParameter) {
  RunAndCompareThroughPipeline(Producer::kNone, 256, 512, false, true,
                               kPipelineClasses);
}

TEST_P(BlockScaledQuantizePipelineTest, ColumnOfWeight) {
  RunAndCompareThroughPipeline(Producer::kWeight, 256, 512, false, true,
                               kPipelineClasses);
}

TEST_P(BlockScaledQuantizePipelineTest, PairOfParameter) {
  RunAndCompareThroughPipeline(Producer::kNone, 256, 512, true, true,
                               kPipelineClasses);
}

TEST_P(BlockScaledQuantizePipelineTest, PairOfWeight) {
  RunAndCompareThroughPipeline(Producer::kWeight, 256, 512, true, true,
                               kPipelineClasses);
}

TEST_P(BlockScaledQuantizePipelineTest, PairOfSiluMul) {
  RunAndCompareThroughPipeline(Producer::kSiluMul, 256, 512, true, true,
                               {InputClass::kRandom, InputClass::kMixed});
}

TEST_P(BlockScaledQuantizePipelineTest, PairOfHeadsTokensDim) {
  RunAndCompareThroughPipeline(Producer::kHeadsTokensDim, 512, 512, true, true,
                               kPipelineClasses);
}

TEST_P(BlockScaledQuantizePipelineTest, PairOfHeadsDimTokens) {
  RunAndCompareThroughPipeline(Producer::kHeadsDimTokens, 512, 512, true, true,
                               kPipelineClasses);
}

// 192 columns take the 4-wave tile.
TEST_P(BlockScaledQuantizePipelineTest, PairOfParameterNarrowTile) {
  RunAndCompareThroughPipeline(Producer::kNone, 256, 192, true, true,
                               kPipelineClasses);
}

TEST_P(BlockScaledQuantizePipelineTest, PairOfTokensHeadsDim) {
  RunAndCompareThroughPipeline(Producer::kTokensHeadsDim, 512, 512, true, true,
                               kPipelineClasses);
}

// The forward A1 and A2: the RMSNorm apply, with the decode's 2^-127 read from
// a tensor operand.
TEST_P(BlockScaledQuantizePipelineTest, RowOfRmsNormWithScaleOperand) {
  RunAndCompareThroughPipeline(Producer::kRmsNorm, 256, 512, true, false,
                               {InputClass::kRandom, InputClass::kMixed},
                               {.tiny_operand = true});
}

TEST_P(BlockScaledQuantizePipelineTest, RowOfParameterWithScaleOperand) {
  RunAndCompareThroughPipeline(Producer::kNone, 256, 512, true, false,
                               kPipelineClasses, {.tiny_operand = true});
}

TEST_P(BlockScaledQuantizePipelineTest, RowWithBf16DataRegion) {
  RunAndCompareThroughPipeline(Producer::kNone, 256, 512, true, false,
                               kPipelineClasses, {.bf16_data_region = true});
}

INSTANTIATE_TEST_SUITE_P(
    BlockScaledQuantizePipelineTests, BlockScaledQuantizePipelineTest,
    ::testing::Combine(::testing::Values(F8E4M3FN, F4E2M1FN),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<PrimitiveType, bool>>& info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          std::get<1>(info.param) ? "" : "_no_excess_precision");
    });

INSTANTIATE_TEST_SUITE_P(
    BlockScaledQuantizeDeviceTests, BlockScaledQuantizeDeviceTest,
    ::testing::Combine(::testing::Values(BF16, F32),
                       ::testing::Values(F8E4M3FN, F4E2M1FN)),
    [](const ::testing::TestParamInfo<std::tuple<PrimitiveType, PrimitiveType>>&
           info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          "_",
          primitive_util::LowercasePrimitiveTypeName(std::get<1>(info.param)));
    });

}  // namespace
}  // namespace xla::gpu
