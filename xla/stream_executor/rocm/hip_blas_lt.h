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

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_

#include <cstddef>
#include <utility>

#include "absl/status/status.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/host_or_device_scalar.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"

#if TF_HIPBLASLT

#include "xla/stream_executor/rocm/hip_blas_utils.h"

namespace stream_executor {

namespace rocm {

class BlasLt : public gpu::BlasLt {
  template <typename T>
  using Owned =
      std::unique_ptr<std::remove_pointer_t<T>, hipblasStatus_t (*)(T)>;

 public:
  using Scale = std::variant< float, double, xla::complex64, xla::complex128 >;

  struct MatmulPlan : public gpu::BlasLt::MatmulPlan {

    struct Config {
      int64_t m, n, k, batch_count;
      int64_t lda, ldb, ldc, ldd;
      int64_t strideA, strideB, strideC, strideD;
      hipblaslt_ext::GemmEpilogue epilogue;
    };

    MatmulPlan(hipblaslt_ext::Gemm&& gemm,
               const Config& cfg,
               Scale alpha, Scale beta,
               bool must_swap_operands)
        : gemm_(std::move(gemm)), cfg_(cfg),
          alpha_(alpha), beta_(beta),
          must_swap_operands_(must_swap_operands),
          algorithm_is_dirty_(true) {}

    ~MatmulPlan() override = default;

    absl::Status ExecuteOnStream(
        Stream* stream, const gpu::BlasLt::MemoryArgs& args,
        blas::ProfileResult* profile_result) const override;

    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        const Stream* stream, size_t max_algorithm_count,
        size_t max_workspace_size) const override;

    absl::Status SetAlgorithm(const MatmulAlgorithm& algorithm) override;

  protected:
    absl::Status MaybeSetMemoryArgs(
        const gpu::BlasLt::MemoryArgs& args) const;

  private:
    mutable hipblaslt_ext::Gemm gemm_;
    mutable Config cfg_;
    Scale alpha_, beta_;
    bool must_swap_operands_;
    mutable bool algorithm_is_dirty_;
    mutable std::optional<MatmulAlgorithm> algorithm_;  // selected algorithm
    mutable gpu::BlasLt::MemoryArgs saved_args_;
  };  // class MatmulPlan

  explicit BlasLt(StreamExecutor* parent)
      : parent_(parent), blas_lt_(nullptr, wrap::hipblasLtDestroy) {}

  absl::Status Init() override;

  absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const gpu::GemmConfig& cfg,
                                              Epilogue epilogue) const override;

  ~BlasLt() override = default;

 private:
  StreamExecutor* parent_;
  mutable absl::Mutex mu_;
  Owned<hipblasLtHandle_t> blas_lt_ ABSL_GUARDED_BY(mu_);
};

}  // namespace rocm
}  // namespace stream_executor

#endif  // TF_HIPBLASLT
#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
