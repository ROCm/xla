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

#ifndef XLA_STREAM_EXECUTOR_GPU_BLAS_LT_MATMUL_ALGORITHM_H_
#define XLA_STREAM_EXECUTOR_GPU_BLAS_LT_MATMUL_ALGORITHM_H_

#include <cstdint>
#include <tuple>
#include <variant>

#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.h"
#include "xla/stream_executor/gpu/blas_lt_matmul_algorithm.pb.h"

namespace stream_executor::gpu {

// ROCm-specific serializable description of a `BlasLtMatmulAlgorithm`: the
// index of a hipBLASLt solution in the loaded library.
struct HipblasLtMatmulAlgorithm {
  int32_t solution_index = 0;

  bool operator==(const HipblasLtMatmulAlgorithm& other) const {
    return solution_index == other.solution_index;
  }
  bool operator!=(const HipblasLtMatmulAlgorithm& other) const {
    return !(*this == other);
  }
};

// Platform-agnostic, serializable representation of a BlasLt matmul algorithm
// (paired with the backend library version) for autotuning persistence and AOT
// compilation.
struct BlasLtMatmulAlgorithm {
  using Algorithm =
      std::variant<cuda::CublasLtMatmulAlgorithm, HipblasLtMatmulAlgorithm>;

  int64_t library_version = 0;
  Algorithm algorithm;

  bool operator==(const BlasLtMatmulAlgorithm& other) const {
    return std::tie(library_version, algorithm) ==
           std::tie(other.library_version, other.algorithm);
  }
  bool operator!=(const BlasLtMatmulAlgorithm& other) const {
    return !(*this == other);
  }

  static absl::StatusOr<BlasLtMatmulAlgorithm> FromProto(
      const BlasLtMatmulAlgorithmProto& proto);
  BlasLtMatmulAlgorithmProto ToProto() const;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_BLAS_LT_MATMUL_ALGORITHM_H_
