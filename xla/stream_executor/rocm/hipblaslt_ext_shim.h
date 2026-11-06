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

// The parts of hipBLASLt's C++ extension API (hipblaslt-ext.hpp) for grouped
// GEMM that pass standard library types.
//
// hipBLASLt is built against libstdc++, so its functions that take or return
// std::vector cannot be called from code built against another C++ standard
// library (such as libc++). The implementation, hipblaslt_ext_shim.cc, is
// always built against libstdc++ (see libstdcxx_cc_library) and wraps them in
// functions that pass only references, pointers and integers. They have C
// linkage, so their symbols do not depend on C++ name mangling either.
//
// The rest of hipblaslt-ext.hpp can be used directly: hipblaslt_ext::
// GroupedGemm's other members, GemmEpilogue and GemmInputs are defined in
// hipBLASLt and pass no standard library types, and UserArguments is a POD.
// A GroupedGemm, whose members are libstdc++ types, must only be created by
// XlaHipblasLtExt_GroupedGemmCreate and deleted through its virtual
// destructor, which hipBLASLt defines.
//
// TODO: Replace with hipBLASLt's own C API for grouped GEMM once it exists.

#include <cstddef>
#include <cstdint>

#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "rocm/include/hipblaslt/hipblaslt.h"

// The implementation is compiled with hidden visibility, so that none of its
// libstdc++ template instantiations are exported; only this interface is.
#define XLA_HIPBLASLT_SHIM_EXPORT __attribute__((visibility("default")))

extern "C" {

// new hipblaslt_ext::GroupedGemm(handle, op_a, ...). Allocated and constructed
// with libstdc++, like hipBLASLt's own objects; delete it as usual.
XLA_HIPBLASLT_SHIM_EXPORT hipblaslt_ext::GroupedGemm*
XlaHipblasLtExt_GroupedGemmCreate(hipblasLtHandle_t handle,
                                  hipblasOperation_t op_a,
                                  hipblasOperation_t op_b, hipDataType type_a,
                                  hipDataType type_b, hipDataType type_c,
                                  hipDataType type_d,
                                  hipblasComputeType_t compute_type);

// hipblaslt_ext::GroupedGemm::setProblem. Every array has `group_count`
// elements; the problem type is the one given at creation.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t XlaHipblasLtExt_GroupedGemmSetProblem(
    hipblaslt_ext::GroupedGemm& gemm, size_t group_count, const int64_t* m,
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

// hipblaslt_ext::getAllAlgos for grouped GEMMs of `gemm`'s problem type. On
// success, calls `callback(context, results, count)` once with all results.
XLA_HIPBLASLT_SHIM_EXPORT hipblasStatus_t
XlaHipblasLtExt_GroupedGemmGetAllAlgos(
    hipblasLtHandle_t handle, hipblaslt_ext::GroupedGemm& gemm,
    XlaHipblasLtExt_HeuristicResultsCallback callback, void* context);

}  // extern "C"

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIPBLASLT_EXT_SHIM_H_
