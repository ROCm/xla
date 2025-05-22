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

#include "rocm/rocm_config.h"
#if TF_HIPBLASLT

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "Eigen/Core"
#include "rocm/include/hip/library_types.h"
#include "rocm/include/hipblas/hipblas.h"
#include "rocm/include/hipblaslt/hipblaslt.h"
#include "rocm/include/rocblas/internal/rocblas-types.h"
#include "xla/primitive_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/rocm/hip_blas_utils.h"
#include "xla/stream_executor/rocm/hipblaslt_wrapper.h"
#include "xla/stream_executor/rocm/rocm_blas.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/ml_dtypes.h"

namespace stream_executor {

namespace rocm {

using ::xla::complex128;
using ::xla::complex64;

namespace {

// This is a special operator == which does not compare workspace fields!
bool operator ==(const gpu::BlasLt::MemoryArgs& lhs, 
                                      const gpu::BlasLt::MemoryArgs& rhs) {
  return (lhs.a == rhs.a && lhs.b == rhs.b && 
          lhs.c == rhs.c && lhs.d == rhs.d && 
          lhs.bias == rhs.bias && lhs.aux == rhs.aux && 
          lhs.a_scale == rhs.a_scale && lhs.b_scale == rhs.b_scale && 
          lhs.c_scale == rhs.c_scale && lhs.d_scale == rhs.d_scale && 
          lhs.d_amax == rhs.d_amax);
}

std::ostream& operator <<(std::ostream& os, const BlasLt::MatmulPlan::Config& cfg) {

  os << "m: " << cfg.m << " n: " << cfg.n << " k: " << cfg.k
      << " batch: " << cfg.batch_count 
      << " lda/b/c/d: " << cfg.lda << '/' << cfg.ldb << '/'
      << cfg.ldc << '/' << cfg.ldd
      << " stridea/b/c/d: " << cfg.strideA << '/' << cfg.strideB << '/'
      << cfg.strideC << '/' << cfg.strideD 
      << " epi: " << (int)cfg.epilogue.mode;
  return os;
}

hipblasLtEpilogue_t AsHipblasLtEpilogue(gpu::BlasLt::Epilogue epilogue) {
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
#if TF_ROCM_VERSION >= 60500
    case gpu::BlasLt::Epilogue::kSILU:
      return HIPBLASLT_EPILOGUE_SWISH_EXT;
    case gpu::BlasLt::Epilogue::kBiasThenSILU:
      return HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT;
#endif
    default:
      LOG(FATAL) << "Unsupported epilogue: " << std::to_string((int)epilogue);
  }
}

bool IsBiasEpilogue(const hipblaslt_ext::GemmEpilogue& epi) {
  switch (epi.mode) {
    case HIPBLASLT_EPILOGUE_BIAS:
    case HIPBLASLT_EPILOGUE_RELU_BIAS:
    case HIPBLASLT_EPILOGUE_GELU_BIAS:
    case HIPBLASLT_EPILOGUE_GELU_AUX_BIAS:
      return true;
    default:
      return false;
  }
}

}  // namespace

absl::Status BlasLt::Init() {
  hipblasLtHandle_t blas_lt;
  SE_HIPBLAS_RETURN_IF_ERROR(wrap::hipblasLtCreate(&blas_lt));
  absl::MutexLock lock(&mu_);
  blas_lt_.reset(blas_lt);
  return absl::OkStatus();
}

auto BlasLt::MatmulPlan::GetAlgorithms(const Stream* stream, 
                                       size_t max_algorithm_count,
                                       size_t max_workspace_size) const
    -> absl::StatusOr<std::vector<MatmulAlgorithm>> {
  max_algorithm_count = std::min(max_algorithm_count, size_t{INT_MAX});
  std::vector<MatmulAlgorithm> algorithms;

  {
    auto blas_lt = static_cast< BlasLt *>(gpu::BlasLt::Get(stream));
    absl::MutexLock lock(&blas_lt->mu_);
    auto activation = blas_lt->parent_->Activate();

    // hipBlasLt requires setting the bias pointer (even a dummy one), otherwise
    // no algorithms can be found for "bias epilogues".
    static int64_t dummy_pointer = 0xACEBALL;
    gpu::BlasLt::MemoryArgs args;
    args.a = DeviceMemoryBase(&dummy_pointer);
    args.b = args.a;
    args.c = args.a;
    args.d = args.a;

    if (IsBiasEpilogue(cfg_.epilogue)) {
      args.bias = args.a; // need to set bias pointer explicitly
    }
    auto problem = gemm_.getProblemTypes();
    // hipBlasLt requires setting the a/b scale pointer (even a dummy one),
    // otherwise no algorithms can be found for "a/b scaling". This is to be
    // removed later when this limitation is gone.
    auto IsFP8 = [&](auto hip_type) {
      return hip_type == HIP_R_8F_E5M2_FNUZ || hip_type == HIP_R_8F_E4M3_FNUZ;
    };
    if (IsFP8(problem.type_a) && IsFP8(problem.type_b)) {
      args.a_scale = args.a;
      args.b_scale = args.a;
    }

    TF_RETURN_IF_ERROR(MaybeSetMemoryArgs(args));
    std::vector<hipblasLtMatmulHeuristicResult_t> results;

    SE_HIPBLAS_RETURN_IF_ERROR(hipblaslt_ext::getAllAlgos(
        blas_lt->blas_lt_.get(), hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        problem.op_a, problem.op_b, 
        problem.type_a, problem.type_b, 
        problem.type_c, problem.type_d, 
        problem.type_compute, results));
    algorithms.reserve(std::min(results.size(), max_algorithm_count));
    VLOG(1) << "Total heuristics found: " << results.size();

    for (auto& res : results) {
      size_t workspace_size = 0;
      if (gemm_.isAlgoSupported(res.algo, workspace_size) 
             == HIPBLAS_STATUS_SUCCESS && workspace_size <= max_workspace_size) {

        algorithms.push_back({ res.algo, workspace_size }); 
        // TODO we need to add actual algorithm ID here!
        //     static_cast< blas::AlgorithmType >
        // (hipblaslt_ext::getIndexFromAlgo(res.algo))});
        if (algorithms.size() >= max_algorithm_count) break;
      }
    }
    VLOG(1) << "Total algos found: " << algorithms.size();
  }  // end mutex block
  return std::move(algorithms);
}

auto BlasLt::GetMatmulPlan(const gpu::GemmConfig& cfg, Epilogue epilogue) const
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

  auto hip_trans_a = AsHipblasOperation(trans_a),
       hip_trans_b = AsHipblasOperation(trans_b);

  auto hip_type = [](const auto& m) -> absl::StatusOr< hipDataType > {
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

  hipblaslt_ext::GemmEpilogue gemm_epi;
  gemm_epi.mode = AsHipblasLtEpilogue(epilogue);
  
  // Currently, the default bias data type in hipblasLt is the same with output
  // data type for fp8 matmul, which is different from cublasLt. This is a
  // workaround to match cublasLt behavior.
  if (IsBiasEpilogue(gemm_epi)) {
    gemm_epi.bias_data_type = d_type;
    if ((a_type == HIP_R_8F_E4M3_FNUZ || a_type == HIP_R_8F_E5M2_FNUZ) &&
        (b_type == HIP_R_8F_E4M3_FNUZ || b_type == HIP_R_8F_E5M2_FNUZ)) {
      gemm_epi.bias_data_type = d_type == HIP_R_32F ? HIP_R_16BF : d_type;
    }
  }

  MatmulPlan::Config internal_cfg {
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
    .epilogue = gemm_epi
  };

  // VLOG(1) << "trans_a/b: " << (int)trans_a << '/' << (int)trans_b << 
  //     " gemm: " << internal_cfg;

  Scale alpha, beta;
  auto set_alpha_beta = [&alpha, &beta, &cfg](auto scale) {
    using InputScale = decltype(scale);
    if constexpr (std::is_same_v<InputScale, xla::complex64> ||
                  std::is_same_v<InputScale, xla::complex128>) {
      alpha = static_cast<InputScale>(cfg.alpha);
    } else {
      alpha = static_cast<InputScale>(cfg.alpha.real());
    }
    beta = static_cast<InputScale>(cfg.beta);
  };

  std::tuple operand_types{a_type, b_type, c_type, d_type};

#define TYPED_MATMUL(Scale, ATYPE, BTYPE, CTYPE, DTYPE)          \
  if (operand_types == std::tuple{ATYPE, BTYPE, CTYPE, DTYPE}) { \
    set_alpha_beta(Scale{});                                     \
  } else

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
  TYPED_MATMUL(float, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_8F_E4M3)

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
  { // else block
    return xla::Internal("GetMatmulPlan: unexpected dtype");
  }
#undef TYPED_MATMUL

  hipblaslt_ext::Gemm gemm(blas_lt_.get(), hip_trans_a, hip_trans_b, 
        a_type, b_type, c_type, d_type, hip_compute_type);

  return std::make_unique<MatmulPlan>(std::move(gemm), internal_cfg, 
                            alpha, beta, must_swap_operands);
}

absl::Status BlasLt::MatmulPlan::MaybeSetMemoryArgs(
      const gpu::BlasLt::MemoryArgs& args) const {
  
  if (saved_args_ == args) {
    return absl::OkStatus();
  }
  DeviceMemoryBase a = args.a, b = args.b;
  if (must_swap_operands_) {
    std::swap(a, b);
  }

#if TF_ROCM_VERSION < 60000
  if (!(args.a_scale == nullptr && args.b_scale == nullptr &&
        args.c_scale == nullptr && args.d_scale == nullptr)) {
    return absl::InternalError("hipblaslt does not support scale");
  }
#endif
  if (!(args.d_amax == nullptr)) {
    return absl::InternalError("hipblaslt does not support amax");
  }
  hipblaslt_ext::GemmInputs inputs = {
      .a = a.opaque(),
      .b = b.opaque(),
      .c = args.c.opaque(),
      .d = args.d.opaque(),
      .alpha = const_cast< void *>((const void *)&alpha_),
      .beta = const_cast< void *>((const void *)&beta_),
      .bias = args.bias.opaque(),
      .scaleA = args.a_scale.opaque(),
      .scaleB = args.b_scale.opaque(),
      .scaleC = args.c_scale.opaque(),
      .scaleD = args.d_scale.opaque(),
      .scaleAux = nullptr,
      .scaleAlphaVec = nullptr,
      .aux = args.aux.opaque()
  };

  auto problem = gemm_.getProblemTypes();
  SE_HIPBLAS_RETURN_IF_ERROR(gemm_.setProblem(
      cfg_.m, cfg_.n, cfg_.k,
      cfg_.batch_count,
      cfg_.lda, cfg_.ldb, cfg_.ldc, cfg_.ldd,
      cfg_.strideA, cfg_.strideB, cfg_.strideC, cfg_.strideD,
      cfg_.epilogue,
      inputs, problem));

  saved_args_ = args;
  algorithm_is_dirty_ = true;   // this force a call to Gemm::initialize()
  return absl::OkStatus();
}

absl::Status BlasLt::MatmulPlan::SetAlgorithm(
        const MatmulAlgorithm& algorithm) {
  
  auto *ptr = std::any_cast<hipblasLtMatmulAlgo_t>(&algorithm.opaque_algo);
  if (ptr == nullptr) {
    return absl::InternalError("SetAlgorithm: invalid opaque algorithm type!");
  }
  algorithm_ = algorithm;
  algorithm_is_dirty_ = true;
  return absl::OkStatus();
}

absl::Status BlasLt::MatmulPlan::ExecuteOnStream(
    Stream* stream, const gpu::BlasLt::MemoryArgs& args,
    blas::ProfileResult* profile_result) const {

  if (!algorithm_.has_value()) {
    return absl::InternalError(
        "Algorithm must be set before calling ExecuteOnStream!");
  }

  auto blas_lt = static_cast<BlasLt*>(gpu::BlasLt::Get(stream));
  TF_RET_CHECK(blas_lt != nullptr);
  absl::Status status =
      blas_lt->parent_->RecordApiTrace(StreamExecutor::GemmCallTrace{
          StreamExecutor::GemmCallTrace::GemmType::kBlasLt, 0, 
          args.a.size(), args.b.size()});

  std::unique_ptr<EventBasedTimer> timer;
  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   profile_result->warmup_run_executed()));
  }

  void* workspace_addr = nullptr;
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
  // SetAlgorithm ensures that algorithm is correctly set
  auto palgo = std::any_cast<hipblasLtMatmulAlgo_t>(&algorithm_->opaque_algo);
  {
    absl::MutexLock lock(&blas_lt->mu_);
    auto activation = blas_lt->parent_->Activate();
    TF_RETURN_IF_ERROR(MaybeSetMemoryArgs(args));

    auto hip_stream = absl::bit_cast<hipStream_t>(
               stream->platform_specific_handle().stream);
    if (saved_args_.workspace.opaque() != workspace_addr || algorithm_is_dirty_) {
      SE_HIPBLAS_RETURN_IF_ERROR(
           gemm_.initialize(*palgo, workspace_addr, true, hip_stream));
      algorithm_is_dirty_ = false;
      saved_args_.workspace = DeviceMemoryBase(workspace_addr);
    }

    SE_HIPBLAS_RETURN_IF_ERROR(gemm_.run(hip_stream));
  } // end block

  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed, timer->GetElapsedDuration());
    profile_result->set_algorithm(static_cast<blas::AlgorithmType>(
              hipblaslt_ext::getIndexFromAlgo(*palgo)));
    profile_result->set_is_valid(true);
    profile_result->set_elapsed_time_in_ms(absl::ToDoubleMilliseconds(elapsed));
  }
  return absl::OkStatus();
}

}  // namespace rocm

}  // namespace stream_executor

#endif  // TF_HIPBLASLT
