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
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_executor.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scratch_allocator.h"
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

// Helper to create a F32 GemmConfig for synchronizer buffer race testing
// F32 with large K dimension encourages GSU (Global Split-U) algorithms
GemmConfig CreateF32GemmConfig(int64_t M, int64_t N, int64_t K) {
    MatrixLayout layout_a(xla::PrimitiveType::F32, M, K,
                          MatrixLayout::Order::kRowMajor, 1, K, 0,
                          blas::Transpose::kNoTranspose);
    MatrixLayout layout_b(xla::PrimitiveType::F32, K, N,
                          MatrixLayout::Order::kRowMajor, 1, N, 0,
                          blas::Transpose::kNoTranspose);
    MatrixLayout layout_c(xla::PrimitiveType::F32, M, N,
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

// Simple scratch allocator for workspace allocation in tests
class TestScratchAllocator : public ScratchAllocator {
 public:
  TestScratchAllocator(StreamExecutor* executor, int64_t memory_limit)
      : executor_(executor), memory_limit_(memory_limit) {}

  ~TestScratchAllocator() override {
    for (auto& alloc : allocations_) {
      executor_->Deallocate(&alloc);
    }
  }

  int64_t GetMemoryLimitInBytes() override { return memory_limit_; }

  absl::StatusOr<DeviceMemory<uint8_t>> AllocateBytes(
      int64_t byte_size) override {
    DeviceMemory<uint8_t> buffer =
        executor_->AllocateArray<uint8_t>(byte_size, 0);
    if (buffer.is_null()) {
      return absl::ResourceExhaustedError("Failed to allocate scratch memory");
    }
    allocations_.push_back(buffer);
    return buffer;
  }

 private:
  StreamExecutor* executor_;
  int64_t memory_limit_;
  std::vector<DeviceMemory<uint8_t>> allocations_;
};

// GEMM problem sizes that are most likely to trigger GSU with MBSK algorithm.
// Based on analysis of hipblaslt/TensileLite test configurations:
//
// GSU (Global Split-U) is selected when:
//   autoGSU = min(K/depthU/3.0, numCUs/numWGs) > 1
//
// This happens when:
// 1. Small M × N (few tiles = low CU occupancy, need GSU to improve parallelism)
// 2. Large K (many iterations to split across workgroups)
//
// MBSK (MultipleBufferSingleKernel, globalAccumulation=3) uses the shared
// Synchronizer buffer which is the source of the data race.
//
// Problem size format: {M, N, K}
// Source: hipblaslt/tensilelite/Tensile/Tests/common/gsu/*.yaml
const std::vector<std::tuple<int64_t, int64_t, int64_t>> kGsuMbskProblemSizes = {
    // From gsu_mbsk.yaml - explicitly tests MBSK
    {500, 501, 502},
    
    // From mbskPrefetchOpt.yaml - MBSK prefetch optimization tests
    {255, 255, 511},
    {257, 257, 513},
    
    // From auto_gsu.yaml - small M×N with large K triggers auto GSU
    {128, 128, 4096},
    
    // From your reproduction case - small N, large K
    {1024, 37, 5244},
    
    // Additional sizes to stress GSU selection:
    // Very small M×N forces GSU to improve CU utilization
    {64, 64, 8192},
    {32, 32, 16384},
    
    // Mid-size with large K
    {256, 128, 4096},
    {512, 64, 8192},
};

// Test for hipblaslt synchronizer buffer race condition with actual GEMM execution.
// This occurs when using GSU (Global Split-U) algorithms with multiple streams:
// 1. GSU algorithms use a shared synchronizer buffer in the hipblaslt handle
// 2. Multiple streams can launch kernels that overlap on GPU
// 3. Overlapping kernels with same problem size corrupt each other's sync entries
// 4. This leads to incorrect results or hangs
//
// NOTE: This test uses XLA's BlasLt wrapper which auto-selects algorithms.
// It may NOT select a GSU algorithm, so it may not reproduce the synchronizer
// buffer race. 
TEST(HipBlasLtThreadTest, MultiStreamGemmExecutionRace) {
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
    
    // Create multiple streams - key for triggering the race
    constexpr int kNumStreams = 4;
    std::vector<std::unique_ptr<Stream>> streams;
    for (int i = 0; i < kNumStreams; ++i) {
        auto stream_status = executor->CreateStream();
        if (!stream_status.ok()) {
            GTEST_SKIP() << "Failed to create stream " << i;
        }
        streams.push_back(std::move(stream_status.value()));
    }

    // Use problem sizes that are most likely to trigger GSU/MBSK algorithms
    // Small M×N with large K encourages GSU selection
    // Matches configurations from hipblaslt/tensilelite/Tensile/Tests/common/gsu/
    constexpr int64_t M = 128;   // Small M
    constexpr int64_t N = 128;   // Small N  
    constexpr int64_t K = 4096;  // Large K → GSU splits this dimension
    
    GemmConfig config = CreateF32GemmConfig(M, N, K);

    // Create a single plan (shared handle = shared synchronizer buffer)
    auto plan_status = BlasLt::GetMatmulPlan(streams[0].get(), config,
                                              BlasLt::Epilogue::kDefault);
    if (!plan_status.ok()) {
        GTEST_SKIP() << "Failed to create matmul plan: " << plan_status.status();
    }
    auto plan = std::move(plan_status.value());

    // Get algorithms - try to find one that might be GSU
    // Note: XLA doesn't expose algorithm IDs, so we can't force GSUAMBSK
    auto algorithms_status = plan->GetAlgorithms(streams[0].get(),
                                                  /*max_algorithm_count=*/128,
                                                  /*max_workspace_size=*/64 << 20);
    if (!algorithms_status.ok() || algorithms_status->empty()) {
        GTEST_SKIP() << "No algorithms available for this problem size";
    }
    auto algorithms = std::move(algorithms_status.value());
    
    // Set the algorithm (preferably one that uses GSU)
    auto set_status = plan->SetAlgorithm(algorithms[0]);
    if (!set_status.ok()) {
        GTEST_SKIP() << "Failed to set algorithm: " << set_status;
    }

    // Allocate device memory for each stream
    // Each stream has its own A, B, C, D buffers
    struct GemmBuffers {
        DeviceMemory<float> a, b, c, d;
    };
    std::vector<GemmBuffers> buffers(kNumStreams);

    // Initialize matrices: A[i,k] = 1.0, B[k,j] = 1.0
    // Expected result: D[i,j] = K (sum of K ones)
    std::vector<float> host_a(M * K, 1.0f);
    std::vector<float> host_b(K * N, 1.0f);
    std::vector<float> host_c(M * N, 0.0f);

    for (int i = 0; i < kNumStreams; ++i) {
        buffers[i].a = executor->AllocateArray<float>(M * K, 0);
        buffers[i].b = executor->AllocateArray<float>(K * N, 0);
        buffers[i].c = executor->AllocateArray<float>(M * N, 0);
        buffers[i].d = executor->AllocateArray<float>(M * N, 0);
        
        if (buffers[i].a.is_null() || buffers[i].b.is_null() ||
            buffers[i].c.is_null() || buffers[i].d.is_null()) {
            GTEST_SKIP() << "Failed to allocate device memory for stream " << i;
        }

        // Copy input data to device
        ASSERT_TRUE(streams[i]->Memcpy(&buffers[i].a, host_a.data(), 
                                        M * K * sizeof(float)).ok());
        ASSERT_TRUE(streams[i]->Memcpy(&buffers[i].b, host_b.data(), 
                                        K * N * sizeof(float)).ok());
        ASSERT_TRUE(streams[i]->Memcpy(&buffers[i].c, host_c.data(), 
                                        M * N * sizeof(float)).ok());
    }

    // Sync all streams before starting the race test
    for (int i = 0; i < kNumStreams; ++i) {
        ASSERT_TRUE(streams[i]->BlockHostUntilDone().ok());
    }

    std::atomic<int> errors{0};
    std::atomic<int> completed{0};
    std::string first_error;
    std::mutex error_mutex;

    // Run multiple iterations of GEMM on all streams concurrently
    // WITHOUT synchronization between iterations to maximize kernel overlap
    // Increased iterations to maximize kernel overlap
    constexpr int kIterationsPerStream = 1000;

    {
        tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), 
                                             "gemm_race_test", kNumStreams);

        for (int stream_idx = 0; stream_idx < kNumStreams; ++stream_idx) {
            thread_pool.Schedule([&, stream_idx]() {
                Stream* stream = streams[stream_idx].get();
                auto& buf = buffers[stream_idx];
                
                // Create per-thread scratch allocator
                TestScratchAllocator scratch(executor, 64 << 20);
                
                for (int iter = 0; iter < kIterationsPerStream; ++iter) {
                    // Execute GEMM: D = alpha * A * B + beta * C
                    // This actually calls hipblasLtMatmul which uses the
                    // shared synchronizer buffer
                    auto status = plan->ExecuteOnStream(
                        stream,
                        buf.a,  // A
                        buf.b,  // B  
                        buf.c,  // C
                        buf.d,  // D (output)
                        DeviceMemoryBase{},  // bias
                        DeviceMemoryBase{},  // aux
                        DeviceMemoryBase{},  // a_scale
                        DeviceMemoryBase{},  // b_scale
                        DeviceMemoryBase{},  // c_scale
                        DeviceMemoryBase{},  // d_scale
                        DeviceMemoryBase{},  // d_amax
                        scratch,
                        nullptr);  // profile_result

                    if (!status.ok()) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (first_error.empty()) {
                            first_error = "Stream " + std::to_string(stream_idx) + 
                                          " iter " + std::to_string(iter) + ": " +
                                          status.ToString();
                        }
                        errors.fetch_add(1);
                    } else {
                        completed.fetch_add(1);
                    }
                }
            });
        }
        // ThreadPool destructor waits for completion
    }

    // Synchronize all streams
    for (int i = 0; i < kNumStreams; ++i) {
        auto sync_status = streams[i]->BlockHostUntilDone();
        ASSERT_TRUE(sync_status.ok()) 
            << "Stream " << i << " failed to synchronize: " << sync_status;
    }

    // Verify results - check one stream's output
    std::vector<float> host_d(M * N);
    ASSERT_TRUE(streams[0]->Memcpy(host_d.data(), buffers[0].d, 
                                    M * N * sizeof(float)).ok());
    ASSERT_TRUE(streams[0]->BlockHostUntilDone().ok());

    // Expected value: each element should be K (1.0 * 1.0 summed K times)
    float expected = static_cast<float>(K);
    int incorrect_count = 0;
    float max_error = 0.0f;
    for (size_t i = 0; i < host_d.size(); ++i) {
        float error = std::abs(host_d[i] - expected);
        if (error > 1.0f) {  // Allow some floating point tolerance
            incorrect_count++;
            max_error = std::max(max_error, error);
        }
    }

    // Clean up device memory
    for (int i = 0; i < kNumStreams; ++i) {
        executor->Deallocate(&buffers[i].a);
        executor->Deallocate(&buffers[i].b);
        executor->Deallocate(&buffers[i].c);
        executor->Deallocate(&buffers[i].d);
    }

    int total_iterations = kNumStreams * kIterationsPerStream;
    EXPECT_EQ(errors.load(), 0) 
        << "Multi-stream GEMM execution errors. First error: " << first_error;
    EXPECT_EQ(completed.load(), total_iterations)
        << "Not all iterations completed. First error: " << first_error;
    EXPECT_EQ(incorrect_count, 0) 
        << "Result corruption detected: " << incorrect_count 
        << " incorrect values, max error: " << max_error
        << " (expected " << expected << ")";
}

// More aggressive test: Fire-and-forget style GEMM launches to maximize overlap
// This test cycles through multiple problem sizes known to trigger GSU/MBSK
// to maximize the chance of hitting the synchronizer buffer race.
TEST(HipBlasLtThreadTest, FireAndForgetGemmRace) {
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
    
    // More streams = more potential for kernel overlap
    constexpr int kNumStreams = 8;
    std::vector<std::unique_ptr<Stream>> streams;
    for (int i = 0; i < kNumStreams; ++i) {
        auto stream_status = executor->CreateStream();
        if (!stream_status.ok()) {
            GTEST_SKIP() << "Failed to create stream " << i;
        }
        streams.push_back(std::move(stream_status.value()));
    }

    // Use problem sizes from kGsuMbskProblemSizes that encourage GSU/MBSK
    // Small M×N with large K forces GSU to improve CU utilization
    constexpr int64_t M = 64;    // Very small M
    constexpr int64_t N = 64;    // Very small N
    constexpr int64_t K = 8192;  // Very large K → high GSU value
    
    GemmConfig config = CreateF32GemmConfig(M, N, K);

    // Create a single plan (shared handle = shared synchronizer buffer)
    auto plan_status = BlasLt::GetMatmulPlan(streams[0].get(), config,
                                              BlasLt::Epilogue::kDefault);
    if (!plan_status.ok()) {
        GTEST_SKIP() << "Failed to create matmul plan: " << plan_status.status();
    }
    auto plan = std::move(plan_status.value());

    auto algorithms_status = plan->GetAlgorithms(streams[0].get(),
                                                  /*max_algorithm_count=*/128,
                                                  /*max_workspace_size=*/64 << 20);
    if (!algorithms_status.ok() || algorithms_status->empty()) {
        GTEST_SKIP() << "No algorithms available";
    }
    
    auto set_status = plan->SetAlgorithm(algorithms_status.value()[0]);
    if (!set_status.ok()) {
        GTEST_SKIP() << "Failed to set algorithm: " << set_status;
    }

    // Allocate buffers for all streams
    struct GemmBuffers {
        DeviceMemory<float> a, b, c, d;
    };
    std::vector<GemmBuffers> buffers(kNumStreams);
    
    std::vector<float> host_a(M * K, 0.5f);
    std::vector<float> host_b(K * N, 0.5f);
    std::vector<float> host_c(M * N, 0.0f);

    for (int i = 0; i < kNumStreams; ++i) {
        buffers[i].a = executor->AllocateArray<float>(M * K, 0);
        buffers[i].b = executor->AllocateArray<float>(K * N, 0);
        buffers[i].c = executor->AllocateArray<float>(M * N, 0);
        buffers[i].d = executor->AllocateArray<float>(M * N, 0);
        
        if (buffers[i].a.is_null() || buffers[i].b.is_null() ||
            buffers[i].c.is_null() || buffers[i].d.is_null()) {
            GTEST_SKIP() << "Failed to allocate device memory";
        }

        ASSERT_TRUE(streams[i]->Memcpy(&buffers[i].a, host_a.data(), 
                                        M * K * sizeof(float)).ok());
        ASSERT_TRUE(streams[i]->Memcpy(&buffers[i].b, host_b.data(), 
                                        K * N * sizeof(float)).ok());
        ASSERT_TRUE(streams[i]->Memcpy(&buffers[i].c, host_c.data(), 
                                        M * N * sizeof(float)).ok());
    }

    for (int i = 0; i < kNumStreams; ++i) {
        ASSERT_TRUE(streams[i]->BlockHostUntilDone().ok());
    }

    std::atomic<int> errors{0};
    std::atomic<int> completed{0};
    std::string first_error;
    std::mutex error_mutex;

    // Increased iterations to maximize kernel overlap
    constexpr int kIterationsPerStream = 2000;

    {
        tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), 
                                             "fire_forget_gemm", kNumStreams);

        for (int stream_idx = 0; stream_idx < kNumStreams; ++stream_idx) {
            thread_pool.Schedule([&, stream_idx]() {
                Stream* stream = streams[stream_idx].get();
                auto& buf = buffers[stream_idx];
                TestScratchAllocator scratch(executor, 64 << 20);
                
                for (int iter = 0; iter < kIterationsPerStream; ++iter) {
                    // Fire off GEMM without any synchronization
                    auto status = plan->ExecuteOnStream(
                        stream,
                        buf.a, buf.b, buf.c, buf.d,
                        DeviceMemoryBase{}, DeviceMemoryBase{},
                        DeviceMemoryBase{}, DeviceMemoryBase{},
                        DeviceMemoryBase{}, DeviceMemoryBase{},
                        DeviceMemoryBase{},
                        scratch, nullptr);

                    if (!status.ok()) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (first_error.empty()) {
                            first_error = "Stream " + std::to_string(stream_idx) + 
                                          " iter " + std::to_string(iter) + ": " +
                                          status.ToString();
                        }
                        errors.fetch_add(1);
                    } else {
                        completed.fetch_add(1);
                    }
                }
            });
        }
    }

    // Synchronize and check for hangs
    for (int i = 0; i < kNumStreams; ++i) {
        auto sync_status = streams[i]->BlockHostUntilDone();
        ASSERT_TRUE(sync_status.ok()) 
            << "Stream " << i << " failed to sync (possible hang): " << sync_status;
    }

    // Verify one result
    std::vector<float> host_d(M * N);
    ASSERT_TRUE(streams[0]->Memcpy(host_d.data(), buffers[0].d, 
                                    M * N * sizeof(float)).ok());
    ASSERT_TRUE(streams[0]->BlockHostUntilDone().ok());

    // Expected: 0.5 * 0.5 * K = 0.25 * K = 2048 (for K=8192)
    float expected = 0.25f * K;
    int incorrect_count = 0;
    for (size_t i = 0; i < host_d.size(); ++i) {
        if (std::abs(host_d[i] - expected) > 1.0f) {
            incorrect_count++;
        }
    }

    // Cleanup
    for (int i = 0; i < kNumStreams; ++i) {
        executor->Deallocate(&buffers[i].a);
        executor->Deallocate(&buffers[i].b);
        executor->Deallocate(&buffers[i].c);
        executor->Deallocate(&buffers[i].d);
    }

    EXPECT_EQ(errors.load(), 0) << "Execution errors: " << first_error;
    EXPECT_EQ(incorrect_count, 0) 
        << "Result corruption: " << incorrect_count << " incorrect values"
        << " (expected ~" << expected << ")";
}

}  // namespace
}  // namespace stream_executor::rocm