/* Copyright 2023 The OpenXLA Authors.
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
#include "xla/stream_executor/rocm/hip_blas_lt.h"

#include <algorithm>
#include <any>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "rocm/rocm_config.h"
#if TF_HIPBLASLT

#include "Eigen/Core"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "rocm/include/hip/library_types.h"
#include "rocm/include/hipblas/hipblas.h"
#include "rocm/include/hipblaslt/hipblaslt.h"
#include "rocm/include/rocblas/internal/rocblas-types.h"
#include "tsl/platform/ml_dtypes.h"
#include "xla/primitive_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/rocm/hip_blas_utils.h"
#include "xla/stream_executor/rocm/hipblaslt_wrapper.h"
#include "xla/stream_executor/rocm/rocm_blas.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"

#define SET_ATTR(setter, handle, attr, value) \
  ToStatus(setter(handle, attr, &value, sizeof(decltype(value))), #setter)

// hipblasLtMatmulDescGetAttribute does not allow nullptr for the last
// argument (size_t* sizeWritten)
#define GET_ATTR(getter, handle, attr, ValueT)                          \
  [&]() -> absl::StatusOr<ValueT> {                                     \
    ValueT value;                                                       \
    size_t size;                                                        \
    TF_RETURN_IF_ERROR(ToStatus(                                        \
        getter(handle, attr, &value, sizeof(ValueT), &size), #getter)); \
    return std::move(value);                                            \
  }()

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                       \
  if (error != HIPBLAS_STATUS_SUCCESS) {                                   \
    fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, \
            __LINE__);                                                     \
    fprintf(stderr, "\n");                                                 \
    exit(EXIT_FAILURE);                                                    \
  }
#endif

namespace stream_executor {

namespace rocm {

using ::xla::complex128;
using ::xla::complex64;

using namespace hipblaslt_ext;

namespace {

template <typename T>
absl::Status SetAttr(hipblasLtMatrixLayout_t handle,
                     hipblasLtMatrixLayoutAttribute_t attr, T value) {
  return SET_ATTR(wrap::hipblasLtMatrixLayoutSetAttribute, handle, attr, value);
}

template <typename T>
absl::StatusOr<T> GetAttr(hipblasLtMatrixLayout_t handle,
                          hipblasLtMatrixLayoutAttribute_t attr) {
  return GET_ATTR(wrap::hipblasLtMatrixLayoutGetAttribute, handle, attr, T);
}

template <typename T>
absl::Status SetAttr(hipblasLtMatmulDesc_t handle,
                     hipblasLtMatmulDescAttributes_t attr, T value) {
  return SET_ATTR(wrap::hipblasLtMatmulDescSetAttribute, handle, attr, value);
}

template <typename T>
absl::StatusOr<T> GetAttr(hipblasLtMatmulDesc_t handle,
                          hipblasLtMatmulDescAttributes_t attr) {
  return GET_ATTR(wrap::hipblasLtMatmulDescGetAttribute, handle, attr, T);
}

template <typename T>
absl::Status SetAttr(hipblasLtMatmulPreference_t handle,
                     hipblasLtMatmulPreferenceAttributes_t attr, T value) {
  return SET_ATTR(wrap::hipblasLtMatmulPreferenceSetAttribute, handle, attr,
                  value);
}

static absl::StatusOr<hipblasLtEpilogue_t> AsHipblasLtEpilogue(
    gpu::BlasLt::Epilogue epilogue) {
  switch (epilogue) {
    case gpu::BlasLt::Epilogue::kDefault:
      return HIPBLASLT_EPILOGUE_DEFAULT;
    case gpu::BlasLt::Epilogue::kReLU:
      return HIPBLASLT_EPILOGUE_RELU;
    case gpu::BlasLt::Epilogue::kBias:
      return HIPBLASLT_EPILOGUE_BIAS;
    case gpu::BlasLt::Epilogue::kBiasThenReLU:
      return HIPBLASLT_EPILOGUE_RELU_BIAS;
    case gpu::BlasLt::Epilogue::kGELU:
      return HIPBLASLT_EPILOGUE_GELU;
#if TF_ROCM_VERSION >= 60000
    case gpu::BlasLt::Epilogue::kGELUWithAux:
      return HIPBLASLT_EPILOGUE_GELU_AUX;
    case gpu::BlasLt::Epilogue::kBiasThenGELU:
      return HIPBLASLT_EPILOGUE_GELU_BIAS;
    case gpu::BlasLt::Epilogue::kBiasThenGELUWithAux:
      return HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
#endif
#if TF_ROCM_VERSION >= 70000
    case gpu::BlasLt::Epilogue::kSILU:
      return HIPBLASLT_EPILOGUE_SWISH_EXT;
    case gpu::BlasLt::Epilogue::kBiasThenSILU:
      return HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT;
#endif
    default:
      return absl::InternalError("Unsupported epilogue: " +
                                 std::to_string((int)epilogue));
  }
}

absl::Status debug_print_userArgs(StreamExecutor *executor,
                                  DeviceMemoryBase &d_userArgs,
                                  size_t group_count) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<MemoryAllocation> host_userArgs,
                      executor->HostMemoryAllocate(
                          group_count * sizeof(hipblaslt_ext::UserArguments)));
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(
      host_userArgs.get()->opaque(), d_userArgs,
      group_count * sizeof(hipblaslt_ext::UserArguments)));
  hipblaslt_ext::UserArguments *h_userArgs =
      static_cast<hipblaslt_ext::UserArguments *>(
          host_userArgs.get()->opaque());
  for (int i = 0; i < group_count; i++) {
    std::cout << "m[" << i << "] = " << h_userArgs[i].m << std::endl;
    std::cout << "n[" << i << "] = " << h_userArgs[i].n << std::endl;
    std::cout << "k[" << i << "] = " << h_userArgs[i].k << std::endl;
    std::cout << "batch[" << i << "] = " << h_userArgs[i].batch << std::endl;
    std::cout << "a[" << i << "] = " << h_userArgs[i].a << std::endl;
    std::cout << "b[" << i << "] = " << h_userArgs[i].b << std::endl;
    std::cout << "c[" << i << "] = " << h_userArgs[i].c << std::endl;
    std::cout << "d[" << i << "] = " << h_userArgs[i].d << std::endl;
    std::cout << "strideA1[" << i << "] = " << h_userArgs[i].strideA1
              << std::endl;
    std::cout << "strideA2[" << i << "] = " << h_userArgs[i].strideA2
              << std::endl;
    std::cout << "strideB1[" << i << "] = " << h_userArgs[i].strideB1
              << std::endl;
    std::cout << "strideB2[" << i << "] = " << h_userArgs[i].strideB2
              << std::endl;
    std::cout << "strideC1[" << i << "] = " << h_userArgs[i].strideC1
              << std::endl;
    std::cout << "strideC2[" << i << "] = " << h_userArgs[i].strideC2
              << std::endl;
    std::cout << "strideD1[" << i << "] = " << h_userArgs[i].strideD1
              << std::endl;
    std::cout << "strideD2[" << i << "] = " << h_userArgs[i].strideD2
              << std::endl;
    std::cout << "alpha = " << *reinterpret_cast<float *>(h_userArgs[i].alpha)
              << std::endl;
    std::cout << "beta = " << *reinterpret_cast<float *>(h_userArgs[i].beta)
              << std::endl;
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status BlasLt::Init() {
  hipblasLtHandle_t blas_lt;
  SE_HIPBLAS_RETURN_IF_ERROR(wrap::hipblasLtCreate(&blas_lt));
  absl::MutexLock lock(mu_);
  blas_lt_.reset(blas_lt);
  return absl::OkStatus();
}

/*static*/ absl::StatusOr<BlasLt::MatrixLayout> BlasLt::MatrixLayout::Create(
    const gpu::MatrixLayout &m) {
  TF_ASSIGN_OR_RETURN(auto type, gpu::AsBlasDataType(m.dtype));

  auto hipblas_data_type_ = AsHipblasDataType(type);
  hipblasLtMatrixLayout_t hip_layout;
  SE_HIPBLAS_RETURN_IF_ERROR(wrap::hipblasLtMatrixLayoutCreate(
      &hip_layout, hipblas_data_type_, m.num_rows, m.num_cols,
      m.leading_dim_stride));
  // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatrixLayout layout(hip_layout, hipblas_data_type_);
  if (m.order != gpu::MatrixLayout::Order::kColumnMajor)
    return absl::InternalError("HipblasLT does not support row-major matrices");
  TF_RETURN_IF_ERROR(SetAttr(hip_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                             static_cast<int32_t>(m.batch_size)));

  VLOG(2) << "BlasLt::MatrixLayout::Create type: " << (int)type
          << " rows: " << m.num_rows << " cols: " << m.num_cols
          << " batch_size: " << m.batch_size
          << " leading_dim_stride: " << m.leading_dim_stride
          << " batch_stride: " << m.batch_stride;

  TF_RETURN_IF_ERROR(SetAttr(hip_layout,
                             HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                             m.batch_stride));
  return std::move(layout);
}

/*static*/ absl::StatusOr<BlasLt::MatmulDesc> BlasLt::MatmulDesc::Create(
    blas::ComputationType compute_type, blas::DataType scale_type,
    blas::Transpose trans_a, blas::Transpose trans_b, Epilogue epilogue,
    PointerMode pointer_mode) {
  hipblasLtMatmulDesc_t hip_desc;
  VLOG(2) << "BlasLt::MatmulDesc::Create compute_type: " << int(compute_type)
          << " scale_type: " << int(scale_type)
          << " epilogue: " << int(epilogue) << " trans_a: " << int(trans_a)
          << " trans_b: " << int(trans_b) << " pointer_mode "
          << int(pointer_mode);
  auto hip_scale_type = AsHipblasDataType(scale_type);
  auto hip_compute_type = AsHipblasComputeType(compute_type);
  SE_HIPBLAS_RETURN_IF_ERROR(wrap::hipblasLtMatmulDescCreate(
      &hip_desc, hip_compute_type, hip_scale_type));

  int32_t bias_flag =
      static_cast<int32_t>(epilogue) & static_cast<int32_t>(Epilogue::kBias);
  // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatmulDesc desc(hip_desc, hip_compute_type, hip_scale_type,
                          bias_flag != 0);
  if (pointer_mode != PointerMode::kHost) {
    return absl::InternalError("hipblaslt does not support device pointers");
  }

  TF_RETURN_IF_ERROR(SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                             AsHipblasOperation(trans_a)));
  TF_RETURN_IF_ERROR(SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                             AsHipblasOperation(trans_b)));
  TF_ASSIGN_OR_RETURN(hipblasLtEpilogue_t epi, AsHipblasLtEpilogue(epilogue));
  TF_RETURN_IF_ERROR(SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, epi));
  return std::move(desc);
}

auto BlasLt::MatmulPlan::GetAlgorithms(const Stream *stream,
                                       size_t max_algorithm_count,
                                       size_t max_workspace_size) const
    -> absl::StatusOr<std::vector<MatmulAlgorithm>> {
  max_algorithm_count = std::min(max_algorithm_count, size_t{INT_MAX});
  std::vector<hipblasLtMatmulHeuristicResult_t> results(max_algorithm_count);
  {
    auto blas_lt = static_cast<BlasLt *>(gpu::BlasLt::Get(stream));
    absl::MutexLock lock(blas_lt->mu_);
    TF_RET_CHECK(blas_lt->blas_lt_ != nullptr);

    hipblasLtMatmulPreference_t hip_preference;
    SE_HIPBLAS_RETURN_IF_ERROR(
        wrap::hipblasLtMatmulPreferenceCreate(&hip_preference));

    // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
    Owned<hipblasLtMatmulPreference_t> preference(
        hip_preference, wrap::hipblasLtMatmulPreferenceDestroy);

    TF_RETURN_IF_ERROR(SetAttr<uint64_t>(
        hip_preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        max_workspace_size));

    std::unique_ptr<ActivateContext> activation = blas_lt->parent_->Activate();

    // hipBlasLt requires setting the bias pointer (even a dummy one), otherwise
    // no algorithms can be found for "bias epilogues". This is to be removed
    // later when this limitation is gone.
    if (op_desc_.has_bias_epilogue()) {
      static int64_t dummy_pointer = 0xACEBALL;
      TF_RETURN_IF_ERROR(SetAttr(
          op_desc_.get(), HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dummy_pointer));
    }

    // hipBlasLt requires setting the a/b scale pointer (even a dummy one),
    // otherwise no algorithms can be found for "a/b scaling". This is to be
    // removed later when this limitation is gone.
    auto IsFP8 = [&](const MatrixLayout &layout) -> bool {
      return layout.type() == HIP_R_8F_E5M2_FNUZ ||
             layout.type() == HIP_R_8F_E4M3_FNUZ ||
             layout.type() == HIP_R_8F_E5M2 || layout.type() == HIP_R_8F_E4M3;
    };
    if (IsFP8(a_desc_) && IsFP8(b_desc_)) {
      static int64_t dummy_pointer = 0xACEBALL;
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                 &dummy_pointer));
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                 &dummy_pointer));
    }

    int found_algorithm_count = 0;
    auto error = wrap::hipblasLtMatmulAlgoGetHeuristic(
        blas_lt->blas_lt_.get(), op_desc_.get(), a_desc_.get(), b_desc_.get(),
        c_desc_.get(), d_desc_.get(), preference.get(), max_algorithm_count,
        results.data(), &found_algorithm_count);
    if (error != 0) {
      VLOG(0) << "hipblasLtMatmulAlgoGetHeuristic returned " << (int)error;
      SE_HIPBLAS_RETURN_IF_ERROR(error);
    }
    results.resize(found_algorithm_count);
  }  // end mutex block

  std::vector<MatmulAlgorithm> algorithms;
  algorithms.reserve(results.size());
  for (const hipblasLtMatmulHeuristicResult_t &result : results) {
    if (result.state == HIPBLAS_STATUS_SUCCESS) {  // Skip failed algos.
      algorithms.push_back({result.algo, result.workspaceSize});
    }
  }
  return std::move(algorithms);
}

auto BlasLt::GetMatmulPlan(const gpu::GemmConfig &cfg, Epilogue epilogue) const
    -> absl::StatusOr<MatmulPlanPtr> {
  auto lhs_layout = cfg.lhs_layout, rhs_layout = cfg.rhs_layout,
       output_layout = cfg.output_layout, c_layout = cfg.c_layout;

  // cublasLt matmul requires batch sizes to be equal. If only one operand has a
  // batch, the other will be broadcast (as its batch_stride == 0).
  size_t batch_size = std::max(lhs_layout.batch_size, rhs_layout.batch_size);
  lhs_layout.batch_size = batch_size;
  rhs_layout.batch_size = batch_size;

  bool must_swap_operands =
      MakeOutputColumnMajor(lhs_layout, rhs_layout, output_layout, &c_layout);

  // Do not transpose either input. Note the cuBLASLt documentation somewhat
  // incorrectly claims "A must be transposed and B non-transposed" when A and B
  // are FP8 (https://docs.nvidia.com/cuda/cublas/#cublasltmatmul). In reality,
  // this is only true if A and B are column-major. If A is row-major, A must
  // *not* be transposed, and if B is row-major, B must be transposed. We never
  // transpose A or B, and expect the caller to ensure A is row-major and B is
  // column when A and B are FP8.
  auto trans_a = lhs_layout.transpose, trans_b = rhs_layout.transpose;

  if (xla::primitive_util::IsF8Type(lhs_layout.dtype) &&
      lhs_layout.order == gpu::MatrixLayout::Order::kColumnMajor) {
    return xla::Internal("The F8 LHS must be column-major");
  }
  if (xla::primitive_util::IsF8Type(rhs_layout.dtype) &&
      rhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    return xla::Internal("The F8 RHS must be row-major");
  }

  TF_ASSIGN_OR_RETURN(auto output_dtype,
                      gpu::AsBlasDataType(output_layout.dtype));

  auto compute_type = cfg.compute_type;
  if (!compute_type) {  // obtain compute_type unless provided by the user
    TF_ASSIGN_OR_RETURN(compute_type,
                        gpu::GetBlasComputationType(
                            cfg.precision_algorithm, lhs_layout.dtype,
                            output_layout.dtype, cfg.compute_precision));
  }

  if (lhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    trans_a = blas::Transpose::kTranspose;
    lhs_layout.Transpose();
  }
  if (rhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    trans_b = blas::Transpose::kTranspose;
    rhs_layout.Transpose();
  }

  TF_ASSIGN_OR_RETURN(
      auto op_desc,
      MatmulDesc::Create(*compute_type,
                         gpu::GetScaleType(output_dtype, *compute_type),
                         trans_a, trans_b, epilogue));

  TF_ASSIGN_OR_RETURN(auto a_desc, MatrixLayout::Create(lhs_layout));
  TF_ASSIGN_OR_RETURN(auto b_desc, MatrixLayout::Create(rhs_layout));
  TF_ASSIGN_OR_RETURN(auto c_desc, MatrixLayout::Create(c_layout));
  TF_ASSIGN_OR_RETURN(auto d_desc, MatrixLayout::Create(output_layout));

#if TF_ROCM_VERSION >= 60000
  // Currently, the default bias data type in hipblasLt is the same with output
  // data type for fp8 matmul, which is different from cublasLt. This is a
  // workaround to match cublasLt behavior.
  if (epilogue == gpu::BlasLt::Epilogue::kBias) {
    auto a_dtype = a_desc.type();
    auto b_dtype = b_desc.type();

    auto bias_dtype = d_desc.type();
    if ((a_dtype == HIP_R_8F_E4M3_FNUZ || a_dtype == HIP_R_8F_E5M2_FNUZ) &&
        (b_dtype == HIP_R_8F_E4M3_FNUZ || b_dtype == HIP_R_8F_E5M2_FNUZ)) {
      auto d_dtype = d_desc.type();
      if (d_dtype == HIP_R_32F) {
        bias_dtype = HIP_R_16BF;
      }

      if (bias_dtype != d_dtype) {
        TF_RETURN_IF_ERROR(SetAttr(
            op_desc.get(), HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, bias_dtype));
      }
    }
  }
#endif  // TF_ROCM_VERSION >= 60000

  return std::make_unique<MatmulPlan>(std::move(op_desc), std::move(a_desc),
                                      std::move(b_desc), std::move(c_desc),
                                      std::move(d_desc), cfg.alpha, cfg.beta,
                                      must_swap_operands);
}

absl::Status BlasLt::MatmulPlan::DoMatmul(
    Stream *stream, const void *alpha, const void *beta,
    const gpu::BlasLt::MemoryArgs &args,
    blas::ProfileResult *profile_result) const {
  if (!algorithm_.has_value()) {
    return absl::InternalError(
        "Algorithm must be set before calling DoMatMul!");
  }
  DeviceMemoryBase a = args.a, b = args.b;
  if (must_swap_operands_) {
    std::swap(a, b);
  }

  auto blas_lt = static_cast<BlasLt *>(gpu::BlasLt::Get(stream));
  TF_RET_CHECK(blas_lt != nullptr);
  absl::Status status =
      blas_lt->parent_->RecordApiTrace(StreamExecutor::GemmCallTrace{
          StreamExecutor::GemmCallTrace::GemmType::kBlasLt, 0, a.size(),
          b.size()});
  std::unique_ptr<EventBasedTimer> timer;

  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   profile_result->warmup_run_executed()));
  }

  void *workspace_addr = nullptr;
  uint64_t workspace_size = algorithm_->workspace_size;
  if (workspace_size > 0) {
    if (args.scratch_allocator != nullptr) {
      TF_ASSIGN_OR_RETURN(
          DeviceMemory<uint8_t> alloc,
          args.scratch_allocator->AllocateBytes(workspace_size));
      workspace_addr = gpu::GpuMemoryMutable(&alloc);
    } else {
      workspace_addr = args.workspace.opaque();
      size_t new_size = args.workspace.size();
      TF_RET_CHECK(workspace_addr != nullptr && new_size >= workspace_size);
      workspace_size = new_size;
    }
  }

  auto palgo = std::any_cast<hipblasLtMatmulAlgo_t>(&algorithm_->opaque_algo);
  {
    absl::MutexLock lock(blas_lt->mu_);
    TF_RET_CHECK(blas_lt->blas_lt_ != nullptr);
    // We must set the bias and aux pointers while holding the mutex, to avoid a
    // potential race condition from multiple threads sharing the same plan.
    if (op_desc_.has_bias_epilogue() && args.bias != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                 args.bias.opaque()));
    }

#if TF_ROCM_VERSION >= 60000
    if (args.a_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                 args.a_scale.opaque()));
    }
    if (args.b_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                 args.b_scale.opaque()));
    }
    if (args.c_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                 args.c_scale.opaque()));
    }
    if (args.d_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                 args.d_scale.opaque()));
    }
#else
    if (!(args.a_scale == nullptr && args.b_scale == nullptr &&
          args.c_scale == nullptr && args.d_scale == nullptr)) {
      return absl::InternalError("hipblaslt does not support scale");
    }
#endif

    if (args.d_amax != nullptr) {
      return absl::InternalError("hipblaslt does not support amax");
    }

    if (args.aux != nullptr) {
      return absl::InternalError(
          "hipblaslt does not support auxiliary inputs / outputs");
    }

    std::unique_ptr<ActivateContext> activation = blas_lt->parent_->Activate();

    if (palgo != nullptr) {
      SE_HIPBLAS_RETURN_IF_ERROR(wrap::hipblasLtMatmul(
          blas_lt->blas_lt_.get(), op_desc_.get(), alpha, a.opaque(),
          a_desc_.get(), b.opaque(), b_desc_.get(), beta, args.c.opaque(),
          c_desc_.get(), args.d.opaque(), d_desc_.get(), palgo, workspace_addr,
          workspace_size,
          absl::bit_cast<hipStream_t>(
              stream->platform_specific_handle().stream)));
    } else {
      return absl::InternalError("hipblaslt: Invalid algorithm type");
    }
  }

  typedef struct __attribute__((packed, aligned(8))) _rocblaslt_matmul_algo {
    uint8_t data[8] = {0};
    bool fallback = false;
    size_t max_workspace_bytes = 0;
  } rocblaslt_matmul_algo;

  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed, timer->GetElapsedDuration());
    // set algorithm ID to be unique (otherwise it gets kDefaultAlgorithm ID)
    auto roc_algo = (const rocblaslt_matmul_algo *)palgo;
    auto pindex = (int *)roc_algo->data;
    profile_result->set_algorithm(static_cast<blas::AlgorithmType>(*pindex));
    profile_result->set_is_valid(true);
    profile_result->set_elapsed_time_in_ms(absl::ToDoubleMilliseconds(elapsed));
  }
  return absl::OkStatus();
}

absl::Status BlasLt::MatmulPlan::ExecuteOnStream(
    Stream *stream, const gpu::BlasLt::MemoryArgs &args,
    blas::ProfileResult *profile_result) const {
  auto wrapped_matmul = [&](auto scale) {
    using Scale = decltype(scale);
    Scale salpha;
    if constexpr (std::is_same_v<Scale, xla::complex64> ||
                  std::is_same_v<Scale, xla::complex128>) {
      salpha = static_cast<Scale>(alpha_);
    } else {
      salpha = static_cast<Scale>(alpha_.real());
    }
    Scale sbeta = static_cast<Scale>(beta_);
    return DoMatmul(stream, &salpha, &sbeta, args, profile_result);
  };

  std::tuple operand_types{a_desc_.type(), b_desc_.type(), c_desc_.type(),
                           d_desc_.type()};

#define TYPED_MATMUL(Scale, ATYPE, BTYPE, CTYPE, DTYPE)          \
  if (operand_types == std::tuple{ATYPE, BTYPE, CTYPE, DTYPE}) { \
    return wrapped_matmul(Scale{});                              \
  }

// FP8 compatible types combinations (Full table in
// https://github.com/ROCm/hipBLASLt/blob/develop/docs/api-reference.rst?plain=1)
#if TF_ROCM_VERSION >= 60000
  TYPED_MATMUL(float, HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ, HIP_R_16F,
               HIP_R_16F)
  TYPED_MATMUL(float, HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ, HIP_R_32F,
               HIP_R_32F)

  TYPED_MATMUL(float, HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E5M2_FNUZ, HIP_R_16F,
               HIP_R_16F)
  TYPED_MATMUL(float, HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E5M2_FNUZ, HIP_R_32F,
               HIP_R_32F)

  TYPED_MATMUL(float, HIP_R_8F_E5M2_FNUZ, HIP_R_8F_E4M3_FNUZ, HIP_R_16F,
               HIP_R_16F)
  TYPED_MATMUL(float, HIP_R_8F_E5M2_FNUZ, HIP_R_8F_E4M3_FNUZ, HIP_R_32F,
               HIP_R_32F)
#endif

#if TF_ROCM_VERSION >= 60200
  TYPED_MATMUL(float, HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ, HIP_R_16BF,
               HIP_R_16BF)
  TYPED_MATMUL(float, HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E5M2_FNUZ, HIP_R_16BF,
               HIP_R_16BF)
  TYPED_MATMUL(float, HIP_R_8F_E5M2_FNUZ, HIP_R_8F_E4M3_FNUZ, HIP_R_16BF,
               HIP_R_16BF)
  TYPED_MATMUL(float, HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ,
               HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ)
  TYPED_MATMUL(float, HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E5M2_FNUZ,
               HIP_R_8F_E5M2_FNUZ, HIP_R_8F_E5M2_FNUZ)
  TYPED_MATMUL(float, HIP_R_8F_E5M2_FNUZ, HIP_R_8F_E4M3_FNUZ,
               HIP_R_8F_E5M2_FNUZ, HIP_R_8F_E5M2_FNUZ)
#endif

#if TF_ROCM_VERSION >= 60300
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_16BF)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_8F_E4M3)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16F, HIP_R_8F_E4M3)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16F, HIP_R_16F)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_32F, HIP_R_32F)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_8F_E4M3,
               HIP_R_8F_E4M3)

  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_16BF, HIP_R_16BF)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_16BF, HIP_R_8F_E4M3)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_16BF, HIP_R_8F_E5M2)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_16F, HIP_R_8F_E4M3)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_16F, HIP_R_8F_E5M2)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_16F, HIP_R_16F)
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_32F, HIP_R_32F)

  TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_16BF)
  TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_8F_E4M3)
  TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_8F_E5M2)
  TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_16F, HIP_R_8F_E4M3)
  TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_16F, HIP_R_8F_E5M2)
  TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_16F, HIP_R_16F)
  TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_8F_E4M3, HIP_R_32F, HIP_R_32F)
#endif

  // Other data types:
  TYPED_MATMUL(float, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF)
  TYPED_MATMUL(float, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F)
  TYPED_MATMUL(float, HIP_R_16BF, HIP_R_16BF, HIP_R_32F, HIP_R_32F)
  TYPED_MATMUL(float, HIP_R_16F, HIP_R_16F, HIP_R_32F, HIP_R_32F)
  TYPED_MATMUL(float, HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F)
  TYPED_MATMUL(double, HIP_R_64F, HIP_R_64F, HIP_R_64F, HIP_R_64F)
  TYPED_MATMUL(complex64, HIP_C_32F, HIP_C_32F, HIP_C_32F, HIP_C_32F)
  TYPED_MATMUL(complex128, HIP_C_64F, HIP_C_64F, HIP_C_64F, HIP_C_64F)

#undef TYPED_MATMUL

  return xla::Internal("Unexpected dtype");
}

auto BlasLt::GetGroupedMatmulPlan(gpu::GroupedGemmConfig &cfg,
                                  std::vector<Epilogue> epilogues) const
    -> absl::StatusOr<GroupedMatmulPlanPtr> {
  auto lhs_layout = cfg.lhs_layout, rhs_layout = cfg.rhs_layout,
       output_layout = cfg.output_layout, c_layout = cfg.c_layout;

  bool must_swap_operands =
      MakeOutputColumnMajor(lhs_layout, rhs_layout, output_layout, &c_layout);

  auto lda = lhs_layout.leading_dim_stride;
  auto ldb = rhs_layout.leading_dim_stride;
  auto ldc = c_layout.leading_dim_stride;
  auto ldd = output_layout.leading_dim_stride;

  auto trans_a = lhs_layout.transpose, trans_b = rhs_layout.transpose;
  if (lhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    trans_a = blas::Transpose::kTranspose;
    lhs_layout.Transpose();
  }
  if (rhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    trans_b = blas::Transpose::kTranspose;
    rhs_layout.Transpose();
  }

  auto m = cfg.m, n = cfg.n;
  auto type_a = cfg.type_a, type_b = cfg.type_b;
  auto batch_stride_a = (m * cfg.k);
  auto batch_stride_b = (n * cfg.k);
  if (cfg.rhs_group_dimension.has_value()) {
    batch_stride_b *= cfg.group_count;
  }

  if (must_swap_operands) {
    std::swap(type_a, type_b);
    std::swap(m, n);
    std::swap(batch_stride_a, batch_stride_b);
  }

  auto compute_type = cfg.compute_type;
  if (!compute_type) {  // obtain compute_type unless provided by the user
    TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type_a,
                        gpu::AsXlaPrimitiveType(type_a));
    TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type_d,
                        gpu::AsXlaPrimitiveType(cfg.type_d));
    TF_ASSIGN_OR_RETURN(
        compute_type,
        gpu::GetBlasComputationType(cfg.precision_algorithm, primitive_type_a,
                                    primitive_type_d, cfg.compute_precision));
  }
  if (!compute_type) {
    return absl::InternalError(
        "This algorithm requires a non-zero compute_type!");
  }

  auto plan =
      std::make_unique<GroupedMatmulPlan>(std::move(cfg), must_swap_operands);

  plan->grouped_gemm_ = std::make_unique<GroupedGemm>(
      blas_lt_.get(), AsHipblasOperation(lhs_layout.transpose),
      AsHipblasOperation(rhs_layout.transpose), AsHipblasDataType(type_a),
      AsHipblasDataType(type_b), AsHipblasDataType(plan->cfg_.type_c),
      AsHipblasDataType(plan->cfg_.type_d),
      AsHipblasComputeType(*compute_type));
  auto &ggemm = plan->grouped_gemm_;

  std::vector<int64_t> v_m(plan->cfg_.group_count, m),
      v_n(plan->cfg_.group_count, n), v_k(plan->cfg_.group_count, cfg.k),
      v_batch_count(plan->cfg_.group_count, cfg.batch_count),
      v_lda(plan->cfg_.group_count, lda), v_ldb(plan->cfg_.group_count, ldb),
      v_ldc(plan->cfg_.group_count, ldc), v_ldd(plan->cfg_.group_count, ldd),
      v_strideA(plan->cfg_.group_count, batch_stride_a),
      v_strideB(plan->cfg_.group_count, batch_stride_b),
      v_strideC(plan->cfg_.group_count, (m * n)),
      v_strideD(plan->cfg_.group_count, (m * n));

  switch (plan->cfg_.ragged_mode) {
    case gpu::RaggedDotMode::kRaggedNonContracting: {
      if (must_swap_operands) {
        // ragged dimension in the n dimension
        std::fill(v_n.begin() + 1, v_n.end(), 1);
      } else {
        std::fill(v_m.begin() + 1, v_m.end(), 1);
      }
      break;
    }
    case gpu::RaggedDotMode::kRaggedContracting: {
      std::fill(v_k.begin() + 1, v_k.end(), 1);
      break;
    }
    case gpu::RaggedDotMode::kRaggedBatch: {
      std::fill(v_batch_count.begin() + 1, v_batch_count.end(), 1);
      break;
    }
  }

  // TODO: recover GemmEpilogues from args
  std::vector<GemmEpilogue> epilogue(plan->cfg_.group_count);
  std::vector<GemmInputs> inputs(plan->cfg_.group_count);

  // TODO Improve alpha and beta conversion (similarly to done in the
  // MatmulPlan)
  float salpha = plan->cfg_.alpha.real();
  float sbeta = plan->cfg_.beta;
  for (int64_t i = 0; i < plan->cfg_.group_count; i++) {
    epilogue[i].setMode(HIPBLASLT_EPILOGUE_DEFAULT);
    inputs[i].setA(reinterpret_cast<void *>(~0ULL));
    inputs[i].setB(reinterpret_cast<void *>(~0ULL));
    inputs[i].setC(reinterpret_cast<void *>(~0ULL));
    inputs[i].setD(reinterpret_cast<void *>(~0ULL));
    inputs[i].setAlpha(static_cast<void *>(&salpha));
    inputs[i].setBeta(static_cast<void *>(&sbeta));
  }

  GemmProblemType problem(AsHipblasOperation(trans_a),
                          AsHipblasOperation(trans_b),
                          AsHipblasDataType(type_a), AsHipblasDataType(type_b),
                          AsHipblasDataType(plan->cfg_.type_c),
                          AsHipblasDataType(plan->cfg_.type_d),
                          AsHipblasComputeType(*compute_type));

  // Set the Matrix orders does not seem to change anything.
  // This unexpectec behavior worth to be further investigated.
  // For the moment, we do not defined a specific order and
  // go with the default order (i.e., COLUMN-MAJOR)
  // problem.setOrderA(HIPBLASLT_ORDER_COL);
  // problem.setOrderB(HIPBLASLT_ORDER_COL);
  {
    absl::MutexLock lock(&mu_);
    SE_HIPBLAS_RETURN_IF_ERROR(ggemm->setProblem(
        v_m, v_n, v_k, v_batch_count, v_lda, v_ldb, v_ldc, v_ldd, v_strideA,
        v_strideB, v_strideC, v_strideD, epilogue, inputs, problem));
  }  // end block

  return absl::StatusOr<GroupedMatmulPlanPtr>(std::move(plan));
}

auto BlasLt::GroupedMatmulPlan::GetAlgorithms(const Stream *stream,
                                              size_t max_algorithm_count,
                                              size_t max_workspace_size) const
    -> absl::StatusOr<std::vector<MatmulAlgorithm>> {
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;

  auto blas_lt = static_cast<BlasLt *>(gpu::BlasLt::Get(stream));
  absl::MutexLock lock(&blas_lt->mu_);

  std::unique_ptr<ActivateContext> activation = blas_lt->parent_->Activate();

  auto problem = grouped_gemm_->getProblemTypes()[0];

  SE_HIPBLAS_RETURN_IF_ERROR(
      getAllAlgos(blas_lt->blas_lt_.get(), GemmType::HIPBLASLT_GROUPED_GEMM,
                  problem.getOpA(), problem.getOpB(), problem.getTypeA(),
                  problem.getTypeB(), problem.getTypeC(), problem.getTypeD(),
                  problem.getTypeCompute(), heuristicResult));

  VLOG(2) << "Total heuristics found: " << heuristicResult.size();
  std::vector<MatmulAlgorithm> algorithms;
  algorithms.reserve(heuristicResult.size());
  for (const hipblasLtMatmulHeuristicResult_t &result : heuristicResult) {
    if (result.state == HIPBLAS_STATUS_SUCCESS) {  // Skip failed algos.
      algorithms.push_back({result.algo, result.workspaceSize});
    }
  }
  return std::move(algorithms);
}

absl::Status BlasLt::GroupedMatmulPlan::ExecuteOnStream(
    Stream *stream, const MemoryArgs &args,
    blas::ProfileResult *profile_result) const {
  if (!algorithm_.has_value()) {
    return absl::InternalError(
        "Algorithm must be set before calling DoMatMul!");
  }
  typedef struct __attribute__((packed, aligned(8))) _rocblaslt_matmul_algo {
    uint8_t data[8] = {0};
    bool fallback = false;
    size_t max_workspace_bytes = 0;
  } rocblaslt_matmul_algo;

  auto palgo = std::any_cast<hipblasLtMatmulAlgo_t>(&algorithm_->opaque_algo);
  if (palgo == nullptr) {
    return absl::InternalError("Wrong GGEMM algorithm instance !");
  }
  auto roc_algo = (const rocblaslt_matmul_algo *)palgo;
  auto pindex = (int *)roc_algo->data;
  auto blas_lt = static_cast<BlasLt *>(gpu::BlasLt::Get(stream));
  absl::MutexLock lock(&blas_lt->mu_);

  // NOTE: it could be that workspace is no longer valid after
  // this function returns !!!!

  SE_HIPBLAS_RETURN_IF_ERROR(
      grouped_gemm_->initialize(*palgo, (void *)args.workspace.opaque()));

  auto ByteWidth = [](blas::DataType ty) -> size_t {
    switch (ty) {
      case blas::DataType::kInt8:
        return 1;
      case blas::DataType::kBF16:
      case blas::DataType::kHalf:
        return 2;
      case blas::DataType::kFloat:
      case blas::DataType::kInt32:
      case blas::DataType::kComplexFloat:
        return 4;
      case blas::DataType::kDouble:
      case blas::DataType::kComplexDouble:
        return 8;
      default:
        LOG(FATAL) << "Unknown DataType " << static_cast<int32_t>(ty);
    }
  };

  DeviceMemoryBase a = args.a, b = args.b;
  auto type_a = cfg_.type_a, type_b = cfg_.type_b;
  auto m = cfg_.m, n = cfg_.n;
  if (must_swap_operands_) {
    std::swap(a, b);
    std::swap(type_a, type_b);
    std::swap(m, n);
  }

  size_t byte_width_elem_a = ByteWidth(type_a);
  size_t byte_width_elem_b = ByteWidth(type_b);
  size_t byte_width_elem_c = ByteWidth(cfg_.type_c);
  size_t byte_width_elem_d = ByteWidth(cfg_.type_d);

  // Get the default hipblaslt_ext::UserArguments
  auto executor = blas_lt->parent_;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<MemoryAllocation> host_allocation,
      executor->HostMemoryAllocate(cfg_.group_count *
                                   sizeof(hipblaslt_ext::UserArguments)));
  hipblaslt_ext::UserArguments *userArgs =
      static_cast<hipblaslt_ext::UserArguments *>(
          host_allocation.get()->opaque());
  grouped_gemm_->getDefaultValueForDeviceUserArguments(userArgs);

  auto group_size_bytewidth =
      (cfg_.ragged_mode != gpu::RaggedDotMode::kRaggedBatch)
          ? static_cast<size_t>(args.aux.size() /
                                (cfg_.group_count * cfg_.batch_count))
          : static_cast<size_t>(args.aux.size() / cfg_.group_count);

  // Copy group_sizes from Device to Host
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<MemoryAllocation> host_group_sizes,
      executor->HostMemoryAllocate(cfg_.group_count * group_size_bytewidth));
  TF_RETURN_IF_ERROR(
      executor->SynchronousMemcpy(host_group_sizes.get()->opaque(), args.aux,
                                  cfg_.group_count * group_size_bytewidth));
  // Note: The group size are considered the same accross the batch.
  // Indeed, different group sizes for different batch would required to divide
  // our group-gemm into more sub-groups and set the GEMM config for each
  // sub-group by hand.
  auto get_group_value_at =
      [group_size_bytewidth](
          const std::unique_ptr<MemoryAllocation> &host_group_sizes,
          size_t index) {
        if (group_size_bytewidth == 8) {
          return static_cast<int64_t>(
              static_cast<int64_t *>(host_group_sizes.get()->opaque())[index]);
        }
        return static_cast<int64_t>(
            static_cast<int32_t *>(host_group_sizes.get()->opaque())[index]);
      };

  // Compute args on the host
  userArgs[0].a = a.opaque();
  userArgs[0].b = b.opaque();
  userArgs[0].c = args.d.opaque();
  userArgs[0].d = args.d.opaque();
  if (cfg_.ragged_mode == gpu::RaggedDotMode::kRaggedBatch) {
    userArgs[0].batch = get_group_value_at(host_group_sizes, 0);
  } else if (cfg_.ragged_mode == gpu::RaggedDotMode::kRaggedContracting) {
    userArgs[0].k = get_group_value_at(host_group_sizes, 0);
  } else {
    if (must_swap_operands_) {
      // The ragged matrix has been set as operand B.
      userArgs[0].n = get_group_value_at(host_group_sizes, 0);
    } else {
      userArgs[0].m = get_group_value_at(host_group_sizes, 0);
    }
  }

  VLOG(2) << "cfg.lhs_stride_ragged_dim = " << cfg_.lhs_stride_ragged_dim;
  VLOG(2) << "cfg.rhs_stride_group_dim = " << cfg_.rhs_stride_group_dim;
  VLOG(2) << "cfg.output_stride_ragged_dim = " << cfg_.output_stride_ragged_dim;
  switch (cfg_.ragged_mode) {
    case gpu::RaggedDotMode::kRaggedNonContracting: {
      for (size_t i = 1; i < cfg_.group_count; i++) {
        if (must_swap_operands_) {
          userArgs[i].n = get_group_value_at(host_group_sizes, i);
          userArgs[i].b = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].b) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.lhs_stride_ragged_dim * byte_width_elem_b)));
          userArgs[i].a = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(a.opaque()) +
              (i * cfg_.rhs_stride_group_dim * byte_width_elem_a)));
        } else {
          userArgs[i].m = get_group_value_at(host_group_sizes, i);
          userArgs[i].a = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].a) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.lhs_stride_ragged_dim * byte_width_elem_a)));
          userArgs[i].b = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(b.opaque()) +
              (i * cfg_.rhs_stride_group_dim * byte_width_elem_b)));
        }
        userArgs[i].c = static_cast<void *>(
            static_cast<uint8_t *>(userArgs[i - 1].c) +
            (get_group_value_at(host_group_sizes, i - 1) *
             cfg_.output_stride_ragged_dim * byte_width_elem_c));
        userArgs[i].d = static_cast<void *>(
            static_cast<uint8_t *>(userArgs[i - 1].d) +
            (get_group_value_at(host_group_sizes, i - 1) *
             cfg_.output_stride_ragged_dim * byte_width_elem_d));
      }
      break;
    }
    case gpu::RaggedDotMode::kRaggedContracting: {
      for (size_t i = 1; i < cfg_.group_count; i++) {
        if (must_swap_operands_) {
          userArgs[i].b = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].b) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.lhs_stride_ragged_dim * byte_width_elem_b)));
          userArgs[i].a = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].a) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.rhs_stride_group_dim * byte_width_elem_a)));
        } else {
          userArgs[i].a = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].a) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.lhs_stride_ragged_dim * byte_width_elem_a)));
          userArgs[i].b = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].b) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.rhs_stride_group_dim * byte_width_elem_b)));
        }
        userArgs[i].k = get_group_value_at(host_group_sizes, i);
        userArgs[i].c = static_cast<void *>(
            static_cast<uint8_t *>(args.c.opaque()) +
            (i * cfg_.batch_count * userArgs[i].strideC2 * byte_width_elem_c));
        userArgs[i].d = static_cast<void *>(
            static_cast<uint8_t *>(args.d.opaque()) +
            (i * cfg_.batch_count * userArgs[i].strideD2 * byte_width_elem_d));
      }
      break;
    }
    case gpu::RaggedDotMode::kRaggedBatch: {
      for (size_t i = 1; i < cfg_.group_count; i++) {
        if (must_swap_operands_) {
          userArgs[i].b = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].b) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.lhs_stride_ragged_dim * byte_width_elem_b)));
          userArgs[i].a = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].a) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.rhs_stride_group_dim * byte_width_elem_a)));
        } else {
          userArgs[i].a = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].a) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.lhs_stride_ragged_dim * byte_width_elem_a)));
          userArgs[i].b = static_cast<void *>(const_cast<uint8_t *>(
              static_cast<const uint8_t *>(userArgs[i - 1].b) +
              (get_group_value_at(host_group_sizes, i - 1) *
               cfg_.rhs_stride_group_dim * byte_width_elem_b)));
        }
        userArgs[i].batch = get_group_value_at(host_group_sizes, i);
        userArgs[i].c = static_cast<void *>(
            static_cast<uint8_t *>(userArgs[i - 1].c) +
            (get_group_value_at(host_group_sizes, i - 1) *
             cfg_.output_stride_ragged_dim * byte_width_elem_c));
        userArgs[i].d = static_cast<void *>(
            static_cast<uint8_t *>(userArgs[i - 1].d) +
            (get_group_value_at(host_group_sizes, i - 1) *
             cfg_.output_stride_ragged_dim * byte_width_elem_d));
      }
      break;
    }
  }

  // Copy arguments to device memory
  DeviceMemoryBase d_userArgs = executor->Allocate(
      cfg_.group_count * sizeof(hipblaslt_ext::UserArguments), 0 /*device*/);
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(
      &d_userArgs, userArgs,
      cfg_.group_count * sizeof(hipblaslt_ext::UserArguments)));

  TF_RET_CHECK(blas_lt != nullptr);
  absl::Status status =
      blas_lt->parent_->RecordApiTrace(StreamExecutor::GemmCallTrace{
          StreamExecutor::GemmCallTrace::GemmType::kBlasLt, 0,
          m * cfg_.k * cfg_.batch_count,
          cfg_.k * n * cfg_.batch_count * cfg_.group_count});
  std::unique_ptr<EventBasedTimer> timer;

  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   profile_result->warmup_run_executed()));
  }

  SE_HIPBLAS_RETURN_IF_ERROR(
      grouped_gemm_->run(d_userArgs.opaque(), gpu::AsGpuStreamValue(stream)));
  executor->SynchronizeAllActivity();

  // The profiling has not been tested yet
  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed, timer->GetElapsedDuration());
    // set algorithm ID to be unique (otherwise it gets kDefaultAlgorithm ID)
    profile_result->set_algorithm(static_cast<blas::AlgorithmType>(*pindex));
    profile_result->set_is_valid(true);
    profile_result->set_elapsed_time_in_ms(absl::ToDoubleMilliseconds(elapsed));
  }
  return absl::OkStatus();
}
}  // namespace rocm

}  // namespace stream_executor

#endif  // TF_HIPBLASLT
