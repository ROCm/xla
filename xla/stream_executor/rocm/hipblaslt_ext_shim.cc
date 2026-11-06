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

#include "xla/stream_executor/rocm/hipblaslt_ext_shim.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "rocm/include/hipblaslt/hipblaslt.h"

// hipBLASLt's C++ API passes libstdc++ types, so this file must be built
// against libstdc++, which libstdcxx_cc_library in the BUILD file ensures.
#if defined(_LIBCPP_VERSION)
#error "hipblaslt_ext_shim.cc must be built against libstdc++."
#endif

// Bind the libstdc++ runtime functions used here to libstdc++.so.6 by symbol
// version, also when this file is linked into code built against libc++:
// unversioned, they could resolve to libc++'s runtime instead, mixing the two
// (e.g. operator new from one, delete from the other). Versions are those of
// the oldest libstdc++ in use (GCC 8), except where the symbol is newer.
// hipblaslt_ext_shim_libstdcxx_stub.c lists the same symbols; keep them in
// sync.
#define XLA_LIBSTDCXX_SYMBOL(symbol, version) \
  __asm__(".symver " #symbol ", " #symbol "@" version)
XLA_LIBSTDCXX_SYMBOL(_Znwm, "GLIBCXX_3.4");
XLA_LIBSTDCXX_SYMBOL(_ZdlPv, "GLIBCXX_3.4");
XLA_LIBSTDCXX_SYMBOL(_ZdlPvm, "CXXABI_1.3.9");
XLA_LIBSTDCXX_SYMBOL(_ZTVN10__cxxabiv117__class_type_infoE, "CXXABI_1.3");
XLA_LIBSTDCXX_SYMBOL(_ZNSt8ios_base4InitC1Ev, "GLIBCXX_3.4");
XLA_LIBSTDCXX_SYMBOL(_ZNSt8ios_base4InitD1Ev, "GLIBCXX_3.4");
XLA_LIBSTDCXX_SYMBOL(_ZSt17__throw_bad_allocv, "GLIBCXX_3.4");
XLA_LIBSTDCXX_SYMBOL(_ZSt20__throw_length_errorPKc, "GLIBCXX_3.4");
// Only referenced with GCC 11+ headers.
XLA_LIBSTDCXX_SYMBOL(_ZSt28__throw_bad_array_new_lengthv, "GLIBCXX_3.4.29");

namespace {

std::vector<int64_t> ToVector(const int64_t* values, size_t count) {
  return std::vector<int64_t>(values, values + count);
}

// The problem type given to the constructor.
hipblaslt_ext::GemmProblemType GetProblemType(
    hipblaslt_ext::GroupedGemm& gemm) {
  return gemm.getProblemTypes()[0];
}

}  // namespace

extern "C" {

hipblaslt_ext::GroupedGemm* XlaHipblasLtExt_GroupedGemmCreate(
    hipblasLtHandle_t handle, hipblasOperation_t op_a, hipblasOperation_t op_b,
    hipDataType type_a, hipDataType type_b, hipDataType type_c,
    hipDataType type_d, hipblasComputeType_t compute_type) {
  return new hipblaslt_ext::GroupedGemm(handle, op_a, op_b, type_a, type_b,
                                        type_c, type_d, compute_type);
}

hipblasStatus_t XlaHipblasLtExt_GroupedGemmSetProblem(
    hipblaslt_ext::GroupedGemm& gemm, size_t group_count, const int64_t* m,
    const int64_t* n, const int64_t* k, const int64_t* batch_count,
    const int64_t* lda, const int64_t* ldb, const int64_t* ldc,
    const int64_t* ldd, const int64_t* stride_a, const int64_t* stride_b,
    const int64_t* stride_c, const int64_t* stride_d,
    const hipblaslt_ext::GemmEpilogue* epilogues,
    const hipblaslt_ext::GemmInputs* inputs) {
  if (group_count == 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  std::vector<int64_t> v_m = ToVector(m, group_count);
  std::vector<int64_t> v_n = ToVector(n, group_count);
  std::vector<int64_t> v_k = ToVector(k, group_count);
  std::vector<int64_t> v_batch = ToVector(batch_count, group_count);
  std::vector<int64_t> v_lda = ToVector(lda, group_count);
  std::vector<int64_t> v_ldb = ToVector(ldb, group_count);
  std::vector<int64_t> v_ldc = ToVector(ldc, group_count);
  std::vector<int64_t> v_ldd = ToVector(ldd, group_count);
  std::vector<int64_t> v_stride_a = ToVector(stride_a, group_count);
  std::vector<int64_t> v_stride_b = ToVector(stride_b, group_count);
  std::vector<int64_t> v_stride_c = ToVector(stride_c, group_count);
  std::vector<int64_t> v_stride_d = ToVector(stride_d, group_count);
  std::vector<hipblaslt_ext::GemmEpilogue> v_epilogues(epilogues,
                                                       epilogues + group_count);
  std::vector<hipblaslt_ext::GemmInputs> v_inputs(inputs, inputs + group_count);

  // Like hipBLASLt's own setProblem overloads without problem type.
  hipblaslt_ext::GemmProblemType problem_type = GetProblemType(gemm);
  return gemm.setProblem(v_m, v_n, v_k, v_batch, v_lda, v_ldb, v_ldc, v_ldd,
                         v_stride_a, v_stride_b, v_stride_c, v_stride_d,
                         v_epilogues, v_inputs, problem_type);
}

hipblasStatus_t XlaHipblasLtExt_GroupedGemmGetAllAlgos(
    hipblasLtHandle_t handle, hipblaslt_ext::GroupedGemm& gemm,
    XlaHipblasLtExt_HeuristicResultsCallback callback, void* context) {
  hipblaslt_ext::GemmProblemType p = GetProblemType(gemm);
  std::vector<hipblasLtMatmulHeuristicResult_t> results;
  hipblasStatus_t status = hipblaslt_ext::getAllAlgos(
      handle, hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM, p.getOpA(),
      p.getOpB(), p.getTypeA(), p.getTypeB(), p.getTypeC(), p.getTypeD(),
      p.getTypeCompute(), results);
  if (status == HIPBLAS_STATUS_SUCCESS) {
    callback(context, results.data(), results.size());
  }
  return status;
}

}  // extern "C"
