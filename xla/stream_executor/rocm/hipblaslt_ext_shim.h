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

// Interface to the parts of hipBLASLt's C++ extension API (hipblaslt-ext.hpp)
// that XLA uses for grouped GEMM.
//
// hipBLASLt is built against libstdc++, and some of its extension functions
// take std::vector arguments or return them, so they cannot be called from
// code built against another C++ standard library (such as libc++). The
// implementation, hipblaslt_ext_shim.cc, is always built against libstdc++
// (see libstdcxx_cc_library). This interface passes only references,
// pointers, integers and plain structs, and keeps the
// hipblaslt_ext::GroupedGemm object behind an opaque type, so it can be used
// from code built against either library. Its functions have C linkage, so
// their symbols do not depend on C++ name mangling either.
//
// hipblaslt_ext::GemmEpilogue and hipblaslt_ext::GemmInputs are passed as they
// are: all their members are defined in hipBLASLt, and their only data is a
// pointer to their implementation.
//
// TODO: Replace with hipBLASLt's own C API for grouped GEMM once it exists.

#include <cstddef>
#include <cstdint>

#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "rocm/include/hipblaslt/hipblaslt.h"

// The implementation is compiled with hidden visibility, so that none of its
// libstdc++ template instantiations are exported; only this interface is.
#define XLA_HIPBLASLT_SHIM_EXPORT __attribute__((visibility("default")))

extern "C" {

// A hipblaslt_ext::GroupedGemm.
struct XlaHipblasLtExt_GroupedGemm;

// Transposes, data types and compute type shared by all GEMMs in a group.
struct XlaHipblasLtExt_ProblemType {
  hipblasOperation_t op_a;
  hipblasOperation_t op_b;
  hipDataType type_a;
  hipDataType type_b;
  hipDataType type_c;
  hipDataType type_d;
  hipblasComputeType_t compute_type;
};

// Creates a grouped GEMM for `problem_type`, to be destroyed with
// XlaHipblasLtExt_GroupedGemmDestroy. `handle` must outlive it.
XLA_HIPBLASLT_SHIM_EXPORT XlaHipblasLtExt_GroupedGemm*
XlaHipblasLtExt_GroupedGemmCreate(
    hipblasLtHandle_t handle, const XlaHipblasLtExt_ProblemType& problem_type);

// Destroys `gemm`. Null is allowed.
XLA_HIPBLASLT_SHIM_EXPORT void XlaHipblasLtExt_GroupedGemmDestroy(
    XlaHipblasLtExt_GroupedGemm* gemm);

// hipblaslt_ext::GroupedGemm::setProblem. Every array has `group_count`
// elements; the problem type is the one given at creation.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t XlaHipblasLtExt_GroupedGemmSetProblem(
    XlaHipblasLtExt_GroupedGemm& gemm, size_t group_count, const int64_t* m,
    const int64_t* n, const int64_t* k, const int64_t* batch_count,
    const int64_t* lda, const int64_t* ldb, const int64_t* ldc,
    const int64_t* ldd, const int64_t* stride_a, const int64_t* stride_b,
    const int64_t* stride_c, const int64_t* stride_d,
    const hipblaslt_ext::GemmEpilogue* epilogues,
    const hipblaslt_ext::GemmInputs* inputs);

// Receives the results of XlaHipblasLtExt_GroupedGemmGetAllAlgos. `results` is
// only valid during the call.
typedef void (*XlaHipblasLtExt_HeuristicResultsCallback)(
    void* context, const hipblasLtMatmulHeuristicResult_t* results,
    size_t count);

// hipblaslt_ext::getAllAlgos for grouped GEMM. On success, calls
// `callback(context, results, count)` once with all results.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t
XlaHipblasLtExt_GroupedGemmGetAllAlgos(
    hipblasLtHandle_t handle, const XlaHipblasLtExt_ProblemType& problem_type,
    XlaHipblasLtExt_HeuristicResultsCallback callback, void* context);

// hipblaslt_ext::GemmInstance::setMaxWorkspaceBytes.
XLA_HIPBLASLT_SHIM_EXPORT void XlaHipblasLtExt_GroupedGemmSetMaxWorkspaceBytes(
    XlaHipblasLtExt_GroupedGemm& gemm, size_t workspace_bytes);

// hipblaslt_ext::GemmInstance::isAlgoSupported. Updates `algo`.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t
XlaHipblasLtExt_GroupedGemmIsAlgoSupported(XlaHipblasLtExt_GroupedGemm& gemm,
                                           hipblasLtMatmulAlgo_t& algo,
                                           size_t& workspace_size);

// hipblaslt_ext::GemmInstance::initialize with device user arguments.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t XlaHipblasLtExt_GroupedGemmInitialize(
    XlaHipblasLtExt_GroupedGemm& gemm, const hipblasLtMatmulAlgo_t& algo,
    void* workspace);

// hipblaslt_ext::GroupedGemm::getDefaultValueForDeviceUserArguments: writes
// one hipblaslt_ext::UserArguments per GEMM to the host buffer
// `host_user_args`.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t
XlaHipblasLtExt_GroupedGemmGetDefaultValueForDeviceUserArguments(
    XlaHipblasLtExt_GroupedGemm& gemm, void* host_user_args);

// hipblaslt_ext::GroupedGemm::run with device user arguments.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t
XlaHipblasLtExt_GroupedGemmRun(XlaHipblasLtExt_GroupedGemm& gemm,
                               void* device_user_args, hipStream_t stream);

// hipblaslt_ext::getIndexFromAlgo.
XLA_HIPBLASLT_SHIM_EXPORT int XlaHipblasLtExt_GetIndexFromAlgo(
    const hipblasLtMatmulAlgo_t& algo);

}  // extern "C"

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIPBLASLT_EXT_SHIM_H_
