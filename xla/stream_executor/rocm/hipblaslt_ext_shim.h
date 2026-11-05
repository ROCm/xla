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

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIPBLASLT_EXT_SHIM_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIPBLASLT_EXT_SHIM_H_

// C interface to the parts of hipBLASLt's C++ extension API
// (hipblaslt-ext.hpp) that XLA uses for grouped GEMM.
//
// hipBLASLt is built against libstdc++, and some of its extension functions
// take std::vector arguments or return them, so they cannot be called from
// code built against another C++ standard library (such as libc++). The
// implementation, hipblaslt_ext_shim.cc, is always built against libstdc++
// (see libstdcxx_cc_library). This interface passes only pointers, integers
// and C structs, and keeps the hipblaslt_ext::GroupedGemm object behind an
// opaque handle, so it can be used from code built against either library.
//
// TODO: Replace with hipBLASLt's own C API for grouped GEMM once it exists.

#include <stddef.h>
#include <stdint.h>

#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hipblaslt/hipblaslt.h"

// The implementation is compiled with hidden visibility, so that none of its
// libstdc++ template instantiations are exported; only this interface is.
#define XLA_HIPBLASLT_SHIM_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

// A hipblaslt_ext::GroupedGemm.
typedef struct XlaHipblasLtGroupedGemm XlaHipblasLtGroupedGemm;

// Transposes, data types and compute type shared by all GEMMs in a group.
typedef struct {
  hipblasOperation_t op_a;
  hipblasOperation_t op_b;
  hipDataType type_a;
  hipDataType type_b;
  hipDataType type_c;
  hipDataType type_d;
  hipblasComputeType_t compute_type;
} XlaHipblasLtProblemType;

// Sizes and strides of one GEMM in a group.
typedef struct {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t batch_count;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t ldd;
  int64_t stride_a;
  int64_t stride_b;
  int64_t stride_c;
  int64_t stride_d;
} XlaHipblasLtGemmShape;

// A grouped GEMM whose GEMMs share everything but their shapes, and all but
// the first share their shape too. The data pointers are not part of it: they
// are passed to XlaHipblasLtGroupedGemmRun in the device user arguments.
typedef struct {
  XlaHipblasLtProblemType type;
  size_t group_count;
  XlaHipblasLtGemmShape first;  // Group 0.
  XlaHipblasLtGemmShape rest;   // Groups 1 to group_count - 1.
  hipblasLtEpilogue_t epilogue;
  int has_bias;
  hipDataType bias_type;  // Only used if has_bias is nonzero.
  float alpha;
  float beta;
} XlaHipblasLtGroupedGemmDesc;

// The epilogue fields of hipblaslt_ext::UserArguments.
typedef struct {
  int32_t activation_type;
  int32_t bias_type;
  float act0;
  float act1;
} XlaHipblasLtUserArgsEpilogue;

// Creates a grouped GEMM for `desc` and returns the epilogue fields of its
// first default device user arguments in `defaults`. `handle` must outlive it.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t XlaHipblasLtGroupedGemmCreate(
    hipblasLtHandle_t handle, const XlaHipblasLtGroupedGemmDesc* desc,
    XlaHipblasLtGroupedGemm** gemm, XlaHipblasLtUserArgsEpilogue* defaults);

// Destroys `gemm`. NULL is allowed.
XLA_HIPBLASLT_SHIM_EXPORT void XlaHipblasLtGroupedGemmDestroy(
    XlaHipblasLtGroupedGemm* gemm);

// Writes up to `capacity` heuristic results whose algorithms `gemm` supports
// within `max_workspace_bytes` to `results`, and their number to `count`.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t XlaHipblasLtGroupedGemmGetAlgos(
    XlaHipblasLtGroupedGemm* gemm, size_t max_workspace_bytes,
    hipblasLtMatmulHeuristicResult_t* results, size_t capacity, size_t* count);

// hipblaslt_ext::GemmInstance::initialize with device user arguments.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t XlaHipblasLtGroupedGemmInitialize(
    XlaHipblasLtGroupedGemm* gemm, const hipblasLtMatmulAlgo_t* algo,
    void* workspace);

// hipblaslt_ext::GroupedGemm::run with device user arguments.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t XlaHipblasLtGroupedGemmRun(
    XlaHipblasLtGroupedGemm* gemm, void* device_user_args, hipStream_t stream);

// sizeof(hipblaslt_ext::UserArguments).
XLA_HIPBLASLT_SHIM_EXPORT size_t XlaHipblasLtUserArgumentsSize(void);

// hipblaslt_ext::getIndexFromAlgo.
XLA_HIPBLASLT_SHIM_EXPORT int XlaHipblasLtGetIndexFromAlgo(
    const hipblasLtMatmulAlgo_t* algo);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIPBLASLT_EXT_SHIM_H_
