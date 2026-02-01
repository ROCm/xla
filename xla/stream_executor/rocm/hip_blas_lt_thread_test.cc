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


#include "xla/stream_executor/rocm/hip_blas_lt.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_executor.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"


namespace stream_executor::rocm {

namespace {

using gpu::BlasLt;
using gpu::GemmConfig;
using gpu::MatrixLayout;

// Helper to create a BF16 GemmConfig for testing
GemmConfig CreateBF16GemmConfig(int64_t M, int64_t N, int64_t K) {
    MatrixLayout layout_a(xla::PrimitiveType::BF16, M, K,
                        MatrixLayout::Order::kRowMajor, 1, K, 0,
                        blas::Transpose::kNoTranspose);
    MatrixLayout layout_b(xla::PrimitiveType::BF16, K, N,
                        MatrixLayout::Order::kRowMajor, 1, N, 0,
                        blas::Transpose::kNoTranspose);
    MatrixLayout layout_c(xla::PrimitiveType::BF16, M, N,
                        MatrixLayout::Order::kRowMajor, 1, N, 0,
                        blas::Transpose::kNoTranspose);

    return GemmConfig{
        layout_a,
        layout_b,
        layout_c,
        layout_c,
        {1.0, 0.0},
        0.0,
        0,
        xla::PrecisionConfig::ALG_UNSET,
        std::nullopt,
        false,
        false,
        false,
        std::nullopt
    };
}

// Test concurrent algorithm queries
// This test is designed to catch data races in hipblaslt's
// MasterSolutionLibrary and ContractionSolution.
TEST(HipBlasLtThreadTest, ConcurrentGetAlgorithms) {
    auto platform_status = PlatformManager::PlatformWithId(kROCmPlatformId);
    if (!platform_status.ok()) {
    GTEST_SKIP() << "ROCm platform not available";
    }

    auto* platform = platform_status.value();
    auto executor_status = platform->ExecutorForDevice(0);
    if (!executor_status.ok()) {
    GTEST_SKIP() << "No GPU device available";
    }

    auto* executor = executor_status.value();
    auto stream_status = executor->CreateStream();
    if (!stream_status.ok()) {
    GTEST_SKIP() << "Failed to create stream";
    }
    auto stream = std::move(stream_status.value());

    // Create GemmConfig for BF16 (which triggered the original issue)
    GemmConfig config = CreateBF16GemmConfig(1024, 1024, 1024);

    // Get a matmul plan
    auto plan_status = BlasLt::GetMatmulPlan(stream.get(), config,
                                            BlasLt::Epilogue::kDefault);
    if (!plan_status.ok()) {
    GTEST_SKIP() << "Failed to create matmul plan: " << plan_status.status();
    }
    auto plan = std::move(plan_status.value());

    // Higher thread count and iterations increase chance of hitting race conditions
    constexpr int kNumThreads = 32;
    constexpr int kIterationsPerThread = 500;

    std::atomic<int> errors{0};
    std::atomic<int> completed{0};
    std::string first_error;
    std::mutex error_mutex;
    
    {
      // Scope the ThreadPool so it destructs (waits for completion) before we check results
      tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "blas_lt_test",
                                          kNumThreads);
      for (int t = 0; t < kNumThreads; ++t) {
        thread_pool.Schedule([&]() {
          for (int i = 0; i < kIterationsPerThread; ++i) {
            // GetAlgorithms triggers loadLibrary and getSolutionByIndex
            // which had the data race
            auto algorithms = plan->GetAlgorithms(stream.get(), 
                                                  /*max_algorithm_count=*/10,
                                                  /*max_workspace_size=*/1 << 20);
            if (!algorithms.ok()) {
              {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) {
                  first_error = algorithms.status().ToString();
                }
              }
              errors.fetch_add(1);
              continue;
            }
            if (algorithms->empty()) {
              {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) {
                  first_error = "GetAlgorithms returned empty list";
                }
              }
              errors.fetch_add(1);
              continue;
            }
            completed.fetch_add(1);
          }
        });
      }
      // ThreadPool destructor waits for completion here
    }

    EXPECT_EQ(errors.load(), 0) << "Thread safety issues detected. First error: " << first_error;
    EXPECT_EQ(completed.load(), kNumThreads * kIterationsPerThread) 
        << "First error: " << first_error;
}

// Test concurrent access with different problem sizes (different solution indices)
TEST(HipBlasLtThreadTest, ConcurrentDifferentProblemSizes) {
    auto platform_status = PlatformManager::PlatformWithId(kROCmPlatformId);
    if (!platform_status.ok()) {
    GTEST_SKIP() << "ROCm platform not available";
    }

    auto* platform = platform_status.value();
    auto executor_status = platform->ExecutorForDevice(0);
    if (!executor_status.ok()) {
    GTEST_SKIP() << "No GPU device available";
    }

    auto* executor = executor_status.value();
    auto stream_status = executor->CreateStream();
    if (!stream_status.ok()) {
    GTEST_SKIP() << "Failed to create stream";
    }
    auto stream = std::move(stream_status.value());

    // Different problem sizes trigger different solution indices
    // This exercises lazy loading of different solution library files
    std::vector<std::tuple<int64_t, int64_t, int64_t>> sizes = {
        {128, 128, 128},   {256, 256, 256},   {512, 512, 512},
        {1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096},
        {128, 256, 512},   {512, 128, 256},
    };

    std::atomic<int> errors{0};
    std::atomic<int> completed{0};
    std::string first_error;
    std::mutex error_mutex;

    // Higher iterations increase chance of hitting race conditions
    constexpr int kIterationsPerSize = 200;

    {
      // Scope the ThreadPool so it destructs (waits for completion) before we check results
      tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "blas_lt_sizes",
                                          sizes.size());

      for (const auto& [M, N, K] : sizes) {
        thread_pool.Schedule([&, M = M, N = N, K = K]() {
          GemmConfig config = CreateBF16GemmConfig(M, N, K);

          auto plan_status = BlasLt::GetMatmulPlan(stream.get(), config,
                                                   BlasLt::Epilogue::kDefault);
          if (!plan_status.ok()) {
            // Some sizes may not be supported, that's OK
            return;
          }
          auto plan = std::move(plan_status.value());

          for (int i = 0; i < kIterationsPerSize; ++i) {
            auto algorithms = plan->GetAlgorithms(stream.get(),
                                                  /*max_algorithm_count=*/10,
                                                  /*max_workspace_size=*/1 << 20);
            if (!algorithms.ok()) {
              {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) {
                  first_error = algorithms.status().ToString();
                }
              }
              errors.fetch_add(1);
              continue;
            }
            if (algorithms->empty()) {
              {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) {
                  first_error = "GetAlgorithms returned empty list";
                }
              }
              errors.fetch_add(1);
              continue;
            }
            completed.fetch_add(1);
          }
        });
      }
      // ThreadPool destructor waits for completion here
    }

    EXPECT_EQ(errors.load(), 0) << "Thread safety issues detected. First error: " << first_error;
    // Not all sizes may succeed, so we just check no errors occurred
}

}  // namespace
}  // namespace stream_executor::rocm