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

struct XlaHipblasLtGroupedGemm {
  XlaHipblasLtGroupedGemm(hipblasLtHandle_t handle,
                          const XlaHipblasLtProblemType& problem)
      : problem_type(problem),
        gemm(handle, problem.op_a, problem.op_b, problem.type_a, problem.type_b,
             problem.type_c, problem.type_d, problem.compute_type) {}

  XlaHipblasLtProblemType problem_type;
  hipblaslt_ext::GroupedGemm gemm;
};

namespace {

hipblaslt_ext::GemmProblemType ToGemmProblemType(
    const XlaHipblasLtProblemType& p) {
  return hipblaslt_ext::GemmProblemType(p.op_a, p.op_b, p.type_a, p.type_b,
                                        p.type_c, p.type_d, p.compute_type);
}

std::vector<int64_t> ToVector(const int64_t* values, size_t count) {
  return std::vector<int64_t>(values, values + count);
}

}  // namespace

extern "C" {

hipblasStatus_t XlaHipblasLtGroupedGemmCreate(
    hipblasLtHandle_t handle, const XlaHipblasLtProblemType* problem_type,
    XlaHipblasLtGroupedGemm** gemm) {
  if (problem_type == nullptr || gemm == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  *gemm = new XlaHipblasLtGroupedGemm(handle, *problem_type);
  return HIPBLAS_STATUS_SUCCESS;
}

void XlaHipblasLtGroupedGemmDestroy(XlaHipblasLtGroupedGemm* gemm) {
  delete gemm;
}

hipblasStatus_t XlaHipblasLtGroupedGemmSetProblem(
    XlaHipblasLtGroupedGemm* gemm, size_t group_count, const int64_t* m,
    const int64_t* n, const int64_t* k, const int64_t* batch_count,
    const int64_t* lda, const int64_t* ldb, const int64_t* ldc,
    const int64_t* ldd, const int64_t* stride_a, const int64_t* stride_b,
    const int64_t* stride_c, const int64_t* stride_d,
    const XlaHipblasLtEpilogue* epilogues,
    const XlaHipblasLtGemmInputs* inputs) {
  if (gemm == nullptr || group_count == 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  for (const int64_t* array : {m, n, k, batch_count, lda, ldb, ldc, ldd,
                               stride_a, stride_b, stride_c, stride_d}) {
    if (array == nullptr) {
      return HIPBLAS_STATUS_INVALID_VALUE;
    }
  }
  if (epilogues == nullptr || inputs == nullptr) {
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
    const XlaHipblasLtGemmInputs& in = inputs[i];
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

  hipblaslt_ext::GemmProblemType problem =
      ToGemmProblemType(gemm->problem_type);
  return gemm->gemm.setProblem(v_m, v_n, v_k, v_batch, v_lda, v_ldb, v_ldc,
                               v_ldd, v_stride_a, v_stride_b, v_stride_c,
                               v_stride_d, v_epilogues, v_inputs, problem);
}

hipblasStatus_t XlaHipblasLtGroupedGemmGetAllAlgos(
    hipblasLtHandle_t handle, const XlaHipblasLtProblemType* problem_type,
    XlaHipblasLtHeuristicResultsCallback callback, void* context) {
  if (problem_type == nullptr || callback == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  const XlaHipblasLtProblemType& p = *problem_type;
  std::vector<hipblasLtMatmulHeuristicResult_t> results;
  hipblasStatus_t status = hipblaslt_ext::getAllAlgos(
      handle, hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM, p.op_a, p.op_b,
      p.type_a, p.type_b, p.type_c, p.type_d, p.compute_type, results);
  if (status == HIPBLAS_STATUS_SUCCESS) {
    callback(context, results.data(), results.size());
  }
  return status;
}

void XlaHipblasLtGroupedGemmSetMaxWorkspaceBytes(XlaHipblasLtGroupedGemm* gemm,
                                                 size_t workspace_bytes) {
  gemm->gemm.setMaxWorkspaceBytes(workspace_bytes);
}

hipblasStatus_t XlaHipblasLtGroupedGemmIsAlgoSupported(
    XlaHipblasLtGroupedGemm* gemm, hipblasLtMatmulAlgo_t* algo,
    size_t* workspace_size) {
  if (gemm == nullptr || algo == nullptr || workspace_size == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  return gemm->gemm.isAlgoSupported(*algo, *workspace_size);
}

hipblasStatus_t XlaHipblasLtGroupedGemmInitialize(
    XlaHipblasLtGroupedGemm* gemm, const hipblasLtMatmulAlgo_t* algo,
    void* workspace) {
  if (gemm == nullptr || algo == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  return gemm->gemm.initialize(*algo, workspace);
}

hipblasStatus_t XlaHipblasLtGroupedGemmGetDefaultValueForDeviceUserArguments(
    XlaHipblasLtGroupedGemm* gemm, void* host_user_args) {
  if (gemm == nullptr || host_user_args == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  return gemm->gemm.getDefaultValueForDeviceUserArguments(host_user_args);
}

hipblasStatus_t XlaHipblasLtGroupedGemmRun(XlaHipblasLtGroupedGemm* gemm,
                                           void* device_user_args,
                                           hipStream_t stream) {
  if (gemm == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  return gemm->gemm.run(device_user_args, stream);
}

int XlaHipblasLtGetIndexFromAlgo(const hipblasLtMatmulAlgo_t* algo) {
  hipblasLtMatmulAlgo_t copy = *algo;
  return hipblaslt_ext::getIndexFromAlgo(copy);
}

}  // extern "C"
