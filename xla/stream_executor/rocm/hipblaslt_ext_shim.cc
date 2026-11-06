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

struct XlaHipblasLtExt_GroupedGemm {
  XlaHipblasLtExt_GroupedGemm(hipblasLtHandle_t handle,
                              const XlaHipblasLtExt_ProblemType& problem)
      : gemm(handle, problem.op_a, problem.op_b, problem.type_a, problem.type_b,
             problem.type_c, problem.type_d, problem.compute_type) {}

  hipblaslt_ext::GroupedGemm gemm;
};

namespace {

std::vector<int64_t> ToVector(const int64_t* values, size_t count) {
  return std::vector<int64_t>(values, values + count);
}

}  // namespace

extern "C" {

XlaHipblasLtExt_GroupedGemm* XlaHipblasLtExt_GroupedGemmCreate(
    hipblasLtHandle_t handle, const XlaHipblasLtExt_ProblemType& problem_type) {
  return new XlaHipblasLtExt_GroupedGemm(handle, problem_type);
}

void XlaHipblasLtExt_GroupedGemmDestroy(XlaHipblasLtExt_GroupedGemm* gemm) {
  delete gemm;
}

hipblasStatus_t XlaHipblasLtExt_GroupedGemmSetProblem(
    XlaHipblasLtExt_GroupedGemm& gemm, size_t group_count, const int64_t* m,
    const int64_t* n, const int64_t* k, const int64_t* batch_count,
    const int64_t* lda, const int64_t* ldb, const int64_t* ldc,
    const int64_t* ldd, const int64_t* stride_a, const int64_t* stride_b,
    const int64_t* stride_c, const int64_t* stride_d,
    const XlaHipblasLtExt_Epilogue* epilogues,
    const XlaHipblasLtExt_GemmInputs* inputs) {
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

  std::vector<hipblaslt_ext::GemmEpilogue> v_epilogues(group_count);
  std::vector<hipblaslt_ext::GemmInputs> v_inputs(group_count);
  for (size_t i = 0; i < group_count; ++i) {
    v_epilogues[i].setMode(epilogues[i].mode);
    if (epilogues[i].has_bias_data_type != 0) {
      v_epilogues[i].setBiasDataType(epilogues[i].bias_data_type);
    }
    const XlaHipblasLtExt_GemmInputs& in = inputs[i];
    if (in.a != nullptr) {
      v_inputs[i].setA(in.a);
    }
    if (in.b != nullptr) {
      v_inputs[i].setB(in.b);
    }
    if (in.c != nullptr) {
      v_inputs[i].setC(in.c);
    }
    if (in.d != nullptr) {
      v_inputs[i].setD(in.d);
    }
    if (in.alpha != nullptr) {
      v_inputs[i].setAlpha(in.alpha);
    }
    if (in.beta != nullptr) {
      v_inputs[i].setBeta(in.beta);
    }
    if (in.bias != nullptr) {
      v_inputs[i].setBias(in.bias);
    }
  }

  // The problem type given to the constructor, as hipBLASLt's own setProblem
  // overloads without problem type pass it.
  std::vector<hipblaslt_ext::GemmProblemType> problem_types =
      gemm.gemm.getProblemTypes();
  return gemm.gemm.setProblem(v_m, v_n, v_k, v_batch, v_lda, v_ldb, v_ldc,
                              v_ldd, v_stride_a, v_stride_b, v_stride_c,
                              v_stride_d, v_epilogues, v_inputs,
                              problem_types[0]);
}

hipblasStatus_t XlaHipblasLtExt_GroupedGemmGetAllAlgos(
    hipblasLtHandle_t handle, const XlaHipblasLtExt_ProblemType& p,
    XlaHipblasLtExt_HeuristicResultsCallback callback, void* context) {
  std::vector<hipblasLtMatmulHeuristicResult_t> results;
  hipblasStatus_t status = hipblaslt_ext::getAllAlgos(
      handle, hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM, p.op_a, p.op_b,
      p.type_a, p.type_b, p.type_c, p.type_d, p.compute_type, results);
  if (status == HIPBLAS_STATUS_SUCCESS) {
    callback(context, results.data(), results.size());
  }
  return status;
}

void XlaHipblasLtExt_GroupedGemmSetMaxWorkspaceBytes(
    XlaHipblasLtExt_GroupedGemm& gemm, size_t workspace_bytes) {
  gemm.gemm.setMaxWorkspaceBytes(workspace_bytes);
}

hipblasStatus_t XlaHipblasLtExt_GroupedGemmIsAlgoSupported(
    XlaHipblasLtExt_GroupedGemm& gemm, hipblasLtMatmulAlgo_t& algo,
    size_t& workspace_size) {
  return gemm.gemm.isAlgoSupported(algo, workspace_size);
}

hipblasStatus_t XlaHipblasLtExt_GroupedGemmInitialize(
    XlaHipblasLtExt_GroupedGemm& gemm, const hipblasLtMatmulAlgo_t& algo,
    void* workspace) {
  return gemm.gemm.initialize(algo, workspace);
}

hipblasStatus_t
XlaHipblasLtExt_GroupedGemmGetDefaultValueForDeviceUserArguments(
    XlaHipblasLtExt_GroupedGemm& gemm, void* host_user_args) {
  return gemm.gemm.getDefaultValueForDeviceUserArguments(host_user_args);
}

hipblasStatus_t XlaHipblasLtExt_GroupedGemmRun(
    XlaHipblasLtExt_GroupedGemm& gemm, void* device_user_args,
    hipStream_t stream) {
  return gemm.gemm.run(device_user_args, stream);
}

int XlaHipblasLtExt_GetIndexFromAlgo(const hipblasLtMatmulAlgo_t& algo) {
  // hipblaslt_ext::getIndexFromAlgo takes a non-const reference.
  hipblasLtMatmulAlgo_t copy = algo;
  return hipblaslt_ext::getIndexFromAlgo(copy);
}

}  // extern "C"
