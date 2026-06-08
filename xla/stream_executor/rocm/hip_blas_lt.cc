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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "rocm/include/hip/library_types.h"
#include "rocm/include/hipblas/hipblas.h"
#include "rocm/include/hipblaslt/hipblaslt.h"
#include "rocm/include/rocblas/internal/rocblas-types.h"
#include "xla/primitive_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/rocm/hip_blas_utils.h"
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

namespace stream_executor {

namespace rocm {

using ::xla::complex128;
using ::xla::complex64;

namespace {

bool IsBiasEpilogue(const hipblaslt_ext::GemmEpilogue& epi) {
  switch (epi.getMode()) {
    case HIPBLASLT_EPILOGUE_BIAS:
    case HIPBLASLT_EPILOGUE_RELU_BIAS:
    case HIPBLASLT_EPILOGUE_GELU_BIAS:
    case HIPBLASLT_EPILOGUE_GELU_AUX_BIAS:
    case HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT:
      return true;
    default:
      return false;
  }
}

// This is a special operator == which does not compare workspace fields!
bool operator==(const gpu::BlasLt::MemoryArgs& lhs,
                const gpu::BlasLt::MemoryArgs& rhs) {
  return (lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c &&
          lhs.d == rhs.d && lhs.bias == rhs.bias && lhs.aux == rhs.aux &&
          lhs.a_scale == rhs.a_scale && lhs.b_scale == rhs.b_scale &&
          lhs.c_scale == rhs.c_scale && lhs.d_scale == rhs.d_scale &&
          lhs.d_amax == rhs.d_amax);
}

std::ostream& operator<<(std::ostream& os,
                         const BlasLt::MatmulPlan::Config& cfg) {
  os << "m: " << cfg.m << " n: " << cfg.n << " k: " << cfg.k
     << " batch: " << cfg.batch_count << " lda/b/c/d: " << cfg.lda << '/'
     << cfg.ldb << '/' << cfg.ldc << '/' << cfg.ldd
     << " stridea/b/c/d: " << cfg.strideA << '/' << cfg.strideB << '/'
     << cfg.strideC << '/' << cfg.strideD
     << " epi: " << (int)cfg.epilogue.getMode()
     << " scale_mode: " << static_cast<int>(cfg.scale_mode);
  return os;
}

template <typename T>
absl::Status SetAttr(hipblasLtMatrixLayout_t handle,
                     hipblasLtMatrixLayoutAttribute_t attr, T value) {
  return SET_ATTR(hipblasLtMatrixLayoutSetAttribute, handle, attr, value);
}

template <typename T>
absl::StatusOr<T> GetAttr(hipblasLtMatrixLayout_t handle,
                          hipblasLtMatrixLayoutAttribute_t attr) {
  return GET_ATTR(hipblasLtMatrixLayoutGetAttribute, handle, attr, T);
}

template <typename T>
absl::Status SetAttr(hipblasLtMatmulDesc_t handle,
                     hipblasLtMatmulDescAttributes_t attr, T value) {
  return SET_ATTR(hipblasLtMatmulDescSetAttribute, handle, attr, value);
}

template <typename T>
absl::StatusOr<T> GetAttr(hipblasLtMatmulDesc_t handle,
                          hipblasLtMatmulDescAttributes_t attr) {
  return GET_ATTR(hipblasLtMatmulDescGetAttribute, handle, attr, T);
}

template <typename T>
absl::Status SetAttr(hipblasLtMatmulPreference_t handle,
                     hipblasLtMatmulPreferenceAttributes_t attr, T value) {
  return SET_ATTR(hipblasLtMatmulPreferenceSetAttribute, handle, attr, value);
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

}  // namespace

absl::Status BlasLt::Init() {
  hipblasLtHandle_t handle;
  SE_HIPBLAS_RETURN_IF_ERROR(hipblasLtCreate(&handle));
  absl::MutexLock lock(mu_);
  handle_.reset(handle);
  return absl::OkStatus();
}

/*static*/ absl::StatusOr<BlasLt::MatrixLayout> BlasLt::MatrixLayout::Create(
    const gpu::MatrixLayout& m) {
  TF_ASSIGN_OR_RETURN(auto type, gpu::AsBlasDataType(m.dtype));

  auto hipblas_data_type_ = AsHipblasDataType(type);
  hipblasLtMatrixLayout_t hip_layout;
  SE_HIPBLAS_RETURN_IF_ERROR(
      hipblasLtMatrixLayoutCreate(&hip_layout, hipblas_data_type_, m.num_rows,
                                  m.num_cols, m.leading_dim_stride));
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
    PointerMode pointer_mode, gpu::ScaleMode scale_mode) {
  hipblasLtMatmulDesc_t hip_desc;
  VLOG(2) << "BlasLt::MatmulDesc::Create compute_type: " << int(compute_type)
          << " scale_type: " << int(scale_type)
          << " epilogue: " << int(epilogue) << " trans_a: " << int(trans_a)
          << " trans_b: " << int(trans_b) << " pointer_mode "
          << int(pointer_mode)
          << " scale_mode: " << static_cast<int>(scale_mode);
  auto hip_scale_type = AsHipblasDataType(scale_type);
  auto hip_compute_type = AsHipblasComputeType(compute_type);
  SE_HIPBLAS_RETURN_IF_ERROR(
      hipblasLtMatmulDescCreate(&hip_desc, hip_compute_type, hip_scale_type));

  int32_t bias_flag =
      static_cast<int32_t>(epilogue) & static_cast<int32_t>(Epilogue::kBias);
  // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatmulDesc desc(hip_desc, hip_compute_type, hip_scale_type,
                          bias_flag != 0, scale_mode);
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

auto BlasLt::MatmulPlan::GetAlgorithms(size_t max_algorithm_count,
                                       size_t max_workspace_size) const
    -> absl::StatusOr<std::vector<MatmulAlgorithm>> {
  max_algorithm_count = std::min(max_algorithm_count, size_t{INT_MAX});
  std::vector<MatmulAlgorithm> algorithms;

  {
    absl::MutexLock lock(blas_lt_.mu_);
    TF_RET_CHECK(blas_lt_.handle_ != nullptr);
    auto activation = blas_lt_.executor_->Activate();

    // hipBlasLt requires setting the a/b scale and bias pointers (even a dummy
    // one), otherwise no algorithms can be found for "a/b scaling" and "bias
    // epilogues". This is to be removed later when this limitation is gone
    // (this probably will never happen).
    static int64_t dummy_pointer = 0xACEBALL;
    gpu::BlasLt::MemoryArgs args;
    args.a = DeviceMemoryBase(&dummy_pointer);
    args.b = args.a;
    args.c = args.a;
    args.d = args.a;
    args.bias = args.a;  // need to set bias pointer explicitly
    args.a_scale = args.a;
    args.b_scale = args.a;

    // if (IsBiasEpilogue(cfg_.epilogue)) {
    //   args.bias = args.a; // need to set bias pointer explicitly
    // }
    auto problem = gemm_.getProblemTypes();

    TF_RETURN_IF_ERROR(MaybeSetMemoryArgs(args));
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    results.reserve(max_algorithm_count);

    // int found_algorithm_count = 0;
    // auto error = hipblasLtMatmulAlgoGetHeuristic(
    //     blas_lt_.handle_.get(), op_desc_.get(), a_desc_.get(), b_desc_.get(),
    //     c_desc_.get(), d_desc_.get(), preference.get(), max_algorithm_count,
    //     results.data(), &found_algorithm_count);
    SE_HIPBLAS_RETURN_IF_ERROR(hipblaslt_ext::getAllAlgos(
        blas_lt_.handle_.get(), hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        problem.getOpA(), problem.getOpB(), problem.getTypeA(),
        problem.getTypeB(), problem.getTypeC(), problem.getTypeD(),
        problem.getTypeCompute(), results));
    VLOG(1) << "Total heuristics found: " << results.size();
    algorithms.reserve(std::min(results.size(), max_algorithm_count));

    for (auto& res : results) {
      size_t workspace_size = 0;
      if (gemm_.isAlgoSupported(res.algo, workspace_size) ==
              HIPBLAS_STATUS_SUCCESS &&
          workspace_size <= max_workspace_size) {
        algorithms.push_back({res.algo, workspace_size});
        if (algorithms.size() >= max_algorithm_count) break;
      }
    }
    VLOG(1) << "Total algos found: " << algorithms.size();
  }  // end mutex block
  return algorithms;
}

absl::StatusOr<BlasLt::MatmulPlanPtr> BlasLt::GetMatmulPlan(
    const gpu::GemmConfig& cfg, Epilogue epilogue) const {
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

  auto IsScaledType = [](xla::PrimitiveType dtype) {
    return xla::primitive_util::IsF8Type(dtype) || dtype == xla::F4E2M1FN;
  };
  if (IsScaledType(lhs_layout.dtype) &&
      lhs_layout.order == gpu::MatrixLayout::Order::kColumnMajor) {
    return xla::Internal("The F8/MX LHS must be row-major");
  }
  if (IsScaledType(rhs_layout.dtype) &&
      rhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    return xla::Internal("The F8/MX RHS must be column-major");
  }

  TF_ASSIGN_OR_RETURN(auto output_dtype,
                      gpu::AsBlasDataType(output_layout.dtype));

  auto compute_type = cfg.compute_type;
  if (!compute_type) {  // obtain compute_type unless provided by the user
    TF_ASSIGN_OR_RETURN(
        compute_type,
        gpu::GetBlasComputationType(
            cfg.precision_algorithm, lhs_layout.dtype, output_layout.dtype,
            cfg.compute_precision,
            executor_->GetDeviceDescription().gpu_compute_capability()));
  }

  if (lhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    trans_a = blas::Transpose::kTranspose;
    lhs_layout.Transpose();
  }
  if (rhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    trans_b = blas::Transpose::kTranspose;
    rhs_layout.Transpose();
  }

  VLOG(0) << "Creating MatmulPlan LHS: "
          << (trans_a == blas::Transpose::kTranspose ? "T" : "N")
          << " RHS: " << (trans_b == blas::Transpose::kTranspose ? "T" : "N")
          << " Scale Mode: " << (int)cfg.scale_mode;

  auto hip_trans_a = AsHipblasOperation(trans_a),
       hip_trans_b = AsHipblasOperation(trans_b);

  auto hip_type = [](const auto& m) -> absl::StatusOr<hipDataType> {
    TF_ASSIGN_OR_RETURN(auto type, gpu::AsBlasDataType(m.dtype));
    return AsHipblasDataType(type);
  };
  TF_ASSIGN_OR_RETURN(auto a_type, hip_type(lhs_layout));
  TF_ASSIGN_OR_RETURN(auto b_type, hip_type(rhs_layout));
  TF_ASSIGN_OR_RETURN(auto c_type, hip_type(c_layout));
  TF_ASSIGN_OR_RETURN(auto d_type, hip_type(output_layout));

  auto scale_type = gpu::GetScaleType(output_dtype, *compute_type);
  auto hip_scale_type = AsHipblasDataType(scale_type);
  auto hip_compute_type = AsHipblasComputeType(*compute_type);

  (void)hip_scale_type;  // HIPBLASLT_DATATYPE_INVALID

  hipblaslt_ext::GemmEpilogue gemm_epi;
  TF_ASSIGN_OR_RETURN(auto mode, AsHipblasLtEpilogue(epilogue));
  gemm_epi.setMode(mode);

  switch (cfg.scale_mode) {
    case gpu::ScaleMode::kNone:
    case gpu::ScaleMode::kTensorScaling:
      // Not really necessary but for completeness
      gemm_epi.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
      gemm_epi.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
      break;
    case gpu::ScaleMode::kBlockScaling:
      gemm_epi.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
      gemm_epi.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
      break;
  }

#if TF_ROCM_VERSION >= 60000
  // Currently, the default bias data type in hipblasLt is the same with output
  // data type for fp8 matmul, which is different from cublasLt. This is a
  // workaround to match cublasLt behavior.
  if (IsBiasEpilogue(gemm_epi)) {
    auto bias_data_type = d_type;
    if ((a_type == HIP_R_8F_E4M3_FNUZ || a_type == HIP_R_8F_E5M2_FNUZ) &&
        (b_type == HIP_R_8F_E4M3_FNUZ || b_type == HIP_R_8F_E5M2_FNUZ)) {
      bias_data_type = d_type == HIP_R_32F ? HIP_R_16BF : d_type;
    }
    gemm_epi.setBiasDataType(bias_data_type);
  }
#endif  // TF_ROCM_VERSION >= 60000

  MatmulPlan::Config internal_cfg{
      .m = output_layout.num_rows,
      .n = output_layout.num_cols,
      .k = (hip_trans_a == HIPBLAS_OP_N ? lhs_layout.num_cols
                                        : lhs_layout.num_rows),
      .batch_count = lhs_layout.batch_size,
      .lda = lhs_layout.leading_dim_stride,
      .ldb = rhs_layout.leading_dim_stride,
      .ldc = c_layout.leading_dim_stride,
      .ldd = output_layout.leading_dim_stride,
      .strideA = lhs_layout.batch_stride,
      .strideB = rhs_layout.batch_stride,
      .strideC = c_layout.batch_stride,
      .strideD = output_layout.batch_stride,
      .scale_mode = cfg.scale_mode,
      .epilogue = gemm_epi};

  absl::MutexLock lock(mu_);
  hipblaslt_ext::Gemm gemm(handle_.get(), hip_trans_a, hip_trans_b, a_type,
                           b_type, c_type, d_type, hip_compute_type);

  auto plan = std::make_unique<MatmulPlan>(*this, std::move(gemm), internal_cfg,
                                           must_swap_operands);

  auto assign_alpha_beta = [&](auto scale) {
    using Scale = decltype(scale);
    static_assert(sizeof(Scale) <= MatmulPlan::kMaxScaleBytes,
                  "Scale type must fit in kMaxScaleBytes");
    auto* palpha = reinterpret_cast<Scale*>(&plan->alpha_[0]);
    if constexpr (std::is_same_v<Scale, xla::complex64> ||
                  std::is_same_v<Scale, xla::complex128>) {
      *palpha = static_cast<Scale>(cfg.alpha);
    } else {
      *palpha = static_cast<Scale>(cfg.alpha.real());
    }
    auto* pbeta = reinterpret_cast<Scale*>(&plan->beta_[0]);
    *pbeta = static_cast<Scale>(cfg.beta);
  };

  std::tuple operand_types{a_type, b_type, c_type, d_type};

  // clang-format off
#define TYPED_MATMUL(Scale, ATYPE, BTYPE, CTYPE, DTYPE)               \
  } else if (operand_types == std::tuple{ATYPE, BTYPE, CTYPE, DTYPE}) { \
    assign_alpha_beta(Scale{});
  // clang-format on

  if (false) {  // This is needed to avoid compiler error for the else clause.
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

#if TF_ROCM_VERSION >= 70000
    // MX FP4 (F4E2M1FN) type combinations
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_32F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_32F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_32F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_16F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_16F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_16F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_16BF, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_16BF, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_4F_E2M1, HIP_R_16BF, HIP_R_16F)

    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_32F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_32F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_32F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_16F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_16F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_16F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_16F)

    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_32F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_32F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_32F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_16F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_16F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_16F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_16BF, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_16BF, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_4F_E2M1, HIP_R_8F_E5M2, HIP_R_16BF, HIP_R_16F)

    TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_32F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_32F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_32F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_16F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_16F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_16F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_16BF, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_16BF, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_4F_E2M1, HIP_R_16BF, HIP_R_16F)

    TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_32F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_32F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_32F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_16F, HIP_R_16F)
    TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_16F, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_16F, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_16BF, HIP_R_16BF)
    TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_16BF, HIP_R_32F)
    TYPED_MATMUL(float, HIP_R_8F_E5M2, HIP_R_4F_E2M1, HIP_R_16BF, HIP_R_16F)
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
  } else {
    return xla::Internal("Unexpected operand types for hipblaslt matmul");
  }
#undef TYPED_MATMUL
  return plan;
}

absl::Status BlasLt::MatmulPlan::MaybeSetMemoryArgs(
    const gpu::BlasLt::MemoryArgs& args) const {
  if (saved_args_ == args) {
    return absl::OkStatus();
  }
  auto a = args.a, b = args.b, a_scale = args.a_scale, b_scale = args.b_scale;
  if (must_swap_operands_) {
    std::swap(a, b);
    std::swap(a_scale, b_scale);
  }

  if (!(args.d_amax == nullptr)) {
    return absl::InternalError("hipblaslt does not support amax");
  }
  hipblaslt_ext::GemmInputs inputs;
  inputs.setA(a.opaque());
  inputs.setB(b.opaque());
  inputs.setC(args.c.opaque());
  inputs.setD(args.d.opaque());
  inputs.setAlpha(const_cast<void*>((const void*)&alpha_));
  inputs.setBeta(const_cast<void*>((const void*)&beta_));
  inputs.setBias(args.bias.opaque());
  inputs.setScaleA(a_scale.opaque());
  inputs.setScaleB(b_scale.opaque());
  inputs.setScaleC(args.c_scale.opaque());
  inputs.setScaleD(args.d_scale.opaque());
  inputs.setScaleAux(nullptr);
  inputs.setScaleAlphaVec(nullptr);
  inputs.setAux(args.aux.opaque());
  inputs.setAmaxD(args.d_amax.opaque());

  auto problem = gemm_.getProblemTypes();
  auto epi = cfg_.epilogue;
  SE_HIPBLAS_RETURN_IF_ERROR(
      gemm_.setProblem(cfg_.m, cfg_.n, cfg_.k, cfg_.batch_count, cfg_.lda,
                       cfg_.ldb, cfg_.ldc, cfg_.ldd, cfg_.strideA, cfg_.strideB,
                       cfg_.strideC, cfg_.strideD, epi, inputs, problem));

  saved_args_ = args;
  algorithm_is_dirty_ = true;  // this force a call to Gemm::initialize()
  return absl::OkStatus();
}

absl::Status BlasLt::MatmulPlan::ExecuteOnStream(
    Stream* stream, const gpu::BlasLt::MemoryArgs& args,
    blas::ProfileResult* profile_result) const {
  if (!algorithm_.has_value()) {
    return absl::InternalError(
        "Algorithm must be set before calling ExecuteOnStream!");
  }
  DeviceAddressBase a = args.a, b = args.b;
  DeviceAddressBase a_scale = args.a_scale, b_scale = args.b_scale;
  if (must_swap_operands_) {
    std::swap(a, b);
    std::swap(a_scale, b_scale);
  }

  absl::Status status =
      blas_lt_.executor_->RecordApiTrace(StreamExecutor::GemmCallTrace{
          StreamExecutor::GemmCallTrace::GemmType::kBlasLt, 0, a.size(),
          b.size()});

  std::unique_ptr<EventBasedTimer> timer;
  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   profile_result->warmup_run_executed()));
  }

  void* workspace_addr = nullptr;
  uint64_t workspace_size = workspace_size_;
  if (workspace_size > 0) {
    if (args.scratch_allocator != nullptr) {
      TF_ASSIGN_OR_RETURN(
          DeviceAddress<uint8_t> alloc,
          args.scratch_allocator->AllocateBytes(workspace_size));
      workspace_addr = gpu::GpuMemoryMutable(&alloc);
    } else {
      workspace_addr = args.workspace.opaque();
      size_t new_size = args.workspace.size();
      TF_RET_CHECK(workspace_addr != nullptr && new_size >= workspace_size);
      workspace_size = new_size;
    }
  }

  {
    absl::MutexLock lock(blas_lt_.mu_);
    TF_RET_CHECK(blas_lt_.handle_ != nullptr);
    std::unique_ptr<ActivateContext> activation =
        blas_lt_.executor_->Activate();
    TF_RETURN_IF_ERROR(MaybeSetMemoryArgs(args));

    auto hip_stream =
        absl::bit_cast<hipStream_t>(stream->platform_specific_handle().stream);
    if (saved_args_.workspace.opaque() != workspace_addr ||
        algorithm_is_dirty_) {
      SE_HIPBLAS_RETURN_IF_ERROR(
          gemm_.initialize(*algorithm_, workspace_addr, true, hip_stream));
      algorithm_is_dirty_ = false;
      saved_args_.workspace = DeviceAddressBase(workspace_addr);
    }
    SE_HIPBLAS_RETURN_IF_ERROR(gemm_.run(hip_stream));
  }

  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed, timer->GetElapsedDuration());
    profile_result->set_algorithm(static_cast<blas::AlgorithmType>(
        hipblaslt_ext::getIndexFromAlgo(*algorithm_)));
    profile_result->set_is_valid(true);
    profile_result->set_elapsed_time_in_ms(absl::ToDoubleMilliseconds(elapsed));
  }
  return absl::OkStatus();
}

}  // namespace rocm

}  // namespace stream_executor

#endif  // TF_HIPBLASLT
