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
                          const XlaHipblasLtProblemType& type)
      : handle(handle),
        type(type),
        gemm(handle, type.op_a, type.op_b, type.type_a, type.type_b,
             type.type_c, type.type_d, type.compute_type) {}

  hipblasLtHandle_t handle;
  XlaHipblasLtProblemType type;
  hipblaslt_ext::GroupedGemm gemm;
};

namespace {

// The value of `field` in each GEMM of the group described by `desc`.
std::vector<int64_t> PerGroup(const XlaHipblasLtGroupedGemmDesc& desc,
                              int64_t XlaHipblasLtGemmShape::*field) {
  std::vector<int64_t> values(desc.group_count, desc.rest.*field);
  values[0] = desc.first.*field;
  return values;
}

}  // namespace

extern "C" {

hipblasStatus_t XlaHipblasLtGroupedGemmCreate(
    hipblasLtHandle_t handle, const XlaHipblasLtGroupedGemmDesc* desc,
    XlaHipblasLtGroupedGemm** gemm, XlaHipblasLtUserArgsEpilogue* defaults) {
  if (desc->group_count == 0) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }
  auto grouped_gemm =
      std::make_unique<XlaHipblasLtGroupedGemm>(handle, desc->type);

  hipblaslt_ext::GemmEpilogue epilogue;
  epilogue.setMode(desc->epilogue);
  // hipBLASLt reads the real pointers from the device user arguments, but
  // setProblem wants non-null ones.
  const void* dummy = reinterpret_cast<const void*>(~uintptr_t{0});
  hipblaslt_ext::GemmInputs inputs;
  inputs.setA(dummy);
  inputs.setB(dummy);
  inputs.setC(dummy);
  inputs.setD(dummy);
  inputs.setAlpha(&desc->alpha);
  inputs.setBeta(&desc->beta);
  if (desc->has_bias != 0) {
    epilogue.setBiasDataType(desc->bias_type);
    inputs.setBias(dummy);
  }

  using S = XlaHipblasLtGemmShape;
  std::vector<int64_t> m = PerGroup(*desc, &S::m);
  std::vector<int64_t> n = PerGroup(*desc, &S::n);
  std::vector<int64_t> k = PerGroup(*desc, &S::k);
  std::vector<int64_t> batch_count = PerGroup(*desc, &S::batch_count);
  std::vector<int64_t> lda = PerGroup(*desc, &S::lda);
  std::vector<int64_t> ldb = PerGroup(*desc, &S::ldb);
  std::vector<int64_t> ldc = PerGroup(*desc, &S::ldc);
  std::vector<int64_t> ldd = PerGroup(*desc, &S::ldd);
  std::vector<int64_t> stride_a = PerGroup(*desc, &S::stride_a);
  std::vector<int64_t> stride_b = PerGroup(*desc, &S::stride_b);
  std::vector<int64_t> stride_c = PerGroup(*desc, &S::stride_c);
  std::vector<int64_t> stride_d = PerGroup(*desc, &S::stride_d);
  std::vector<hipblaslt_ext::GemmEpilogue> epilogues(desc->group_count,
                                                     epilogue);
  std::vector<hipblaslt_ext::GemmInputs> all_inputs(desc->group_count, inputs);
  const XlaHipblasLtProblemType& t = desc->type;
  hipblaslt_ext::GemmProblemType problem(t.op_a, t.op_b, t.type_a, t.type_b,
                                         t.type_c, t.type_d, t.compute_type);
  hipblasStatus_t status = grouped_gemm->gemm.setProblem(
      m, n, k, batch_count, lda, ldb, ldc, ldd, stride_a, stride_b, stride_c,
      stride_d, epilogues, all_inputs, problem);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }

  std::vector<hipblaslt_ext::UserArguments> user_args(desc->group_count);
  status = grouped_gemm->gemm.getDefaultValueForDeviceUserArguments(
      user_args.data());
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  defaults->activation_type = user_args[0].activationType;
  defaults->bias_type = user_args[0].biasType;
  defaults->act0 = user_args[0].act0;
  defaults->act1 = user_args[0].act1;

  *gemm = grouped_gemm.release();
  return HIPBLAS_STATUS_SUCCESS;
}

void XlaHipblasLtGroupedGemmDestroy(XlaHipblasLtGroupedGemm* gemm) {
  delete gemm;
}

hipblasStatus_t XlaHipblasLtGroupedGemmGetAlgos(
    XlaHipblasLtGroupedGemm* gemm, size_t max_workspace_bytes,
    hipblasLtMatmulHeuristicResult_t* results, size_t capacity, size_t* count) {
  *count = 0;
  gemm->gemm.setMaxWorkspaceBytes(max_workspace_bytes);
  const XlaHipblasLtProblemType& t = gemm->type;
  std::vector<hipblasLtMatmulHeuristicResult_t> all;
  hipblasStatus_t status = hipblaslt_ext::getAllAlgos(
      gemm->handle, hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM, t.op_a,
      t.op_b, t.type_a, t.type_b, t.type_c, t.type_d, t.compute_type, all);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return status;
  }
  for (hipblasLtMatmulHeuristicResult_t& result : all) {
    if (*count == capacity) {
      break;
    }
    size_t workspace_size = 0;
    if (result.state == HIPBLAS_STATUS_SUCCESS &&
        gemm->gemm.isAlgoSupported(result.algo, workspace_size) ==
            HIPBLAS_STATUS_SUCCESS) {
      results[(*count)++] = result;
    }
  }
  return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t XlaHipblasLtGroupedGemmInitialize(
    XlaHipblasLtGroupedGemm* gemm, const hipblasLtMatmulAlgo_t* algo,
    void* workspace) {
  return gemm->gemm.initialize(*algo, workspace);
}

hipblasStatus_t XlaHipblasLtGroupedGemmRun(XlaHipblasLtGroupedGemm* gemm,
                                           void* device_user_args,
                                           hipStream_t stream) {
  return gemm->gemm.run(device_user_args, stream);
}

size_t XlaHipblasLtUserArgumentsSize(void) {
  return sizeof(hipblaslt_ext::UserArguments);
}

int XlaHipblasLtGetIndexFromAlgo(const hipblasLtMatmulAlgo_t* algo) {
  hipblasLtMatmulAlgo_t copy = *algo;
  return hipblaslt_ext::getIndexFromAlgo(copy);
}

}  // extern "C"
