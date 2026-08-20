/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/autotuner/gpu_profiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;

constexpr absl::string_view kGemmBackendConfig =
    R"(backend_config={"gemm_backend_config":{"dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],"rhs_contracting_dimensions":["1"]}}})";

class MockExecutable : public Executable {
 public:
  explicit MockExecutable(std::shared_ptr<HloModule> module, int duration_ns,
                          bool should_fail = false,
                          bool write_past_allocated_buffer = false)
      : Executable(module),
        duration_ns_(duration_ns),
        should_fail_(should_fail),
        write_past_allocated_buffer_(write_past_allocated_buffer) {}
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    if (!arguments.empty()) {
      first_input_addresses_.push_back(
          arguments[0].Buffer(/*index=*/{}).AsDeviceAddress().opaque());
    }
    if (should_fail_) {
      return absl::InternalError("MockExecutable failed as requested.");
    }
    ExecutionProfile* profile = run_options->run_options().execution_profile();
    if (profile != nullptr) {
      // Only timed runs are handed a profile, so the sequence indexes timed
      // runs and the warm-up does not consume an entry.
      profile->set_compute_time_ns(duration_sequence_.empty()
                                       ? duration_ns_
                                       : duration_sequence_[timed_runs_++ %
                                                            duration_sequence_
                                                                .size()]);
    }
    if (write_past_allocated_buffer_) {
      ABSL_RETURN_IF_ERROR(WriteOutOfBounds(*run_options));
    }
    const Shape& result_shape =
        module().entry_computation()->root_instruction()->shape();
    return ExecutionOutput(result_shape, result_shape,
                           run_options->run_options().allocator(),
                           run_options->run_options().device_ordinal());
  }

  // Address of the first input buffer of every execution, in order. Lets a
  // test see which rotating input set each run was handed.
  const std::vector<const void*>& first_input_addresses() const {
    return first_input_addresses_;
  }

  // Reports these durations in order, one per timed run, instead of the fixed
  // duration. Lets a test drive the median across runs.
  void SetDurationSequence(std::vector<int> durations_ns) {
    duration_sequence_ = std::move(durations_ns);
  }

  int timed_runs() const { return timed_runs_; }

 private:
  // Simulates a kernel that writes past the end of an allocated buffer:
  // allocates a buffer through the run's allocator and then writes a few
  // bytes past the end of it. When the allocator handed to us is a
  // redzone-wrapping allocator (as GpuProfiler::Profile uses during its
  // warm-up run when ProfileOptions.redzone_padding_bytes > 0), this lands
  // in the mapped post-redzone instead of faulting, so it can be detected by
  // CheckRedzones() instead of crashing the process.
  absl::Status WriteOutOfBounds(
      const ServiceExecutableRunOptions& run_options) {
    constexpr int64_t kBufferBytes = 1024;
    constexpr int64_t kOverrunBytes = 64;
    se::DeviceAddressAllocator* allocator =
        run_options.run_options().allocator();
    ABSL_ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> buffer,
        allocator->Allocate(run_options.run_options().device_ordinal(),
                            kBufferBytes));
    se::DeviceAddressBase oob_region(
        static_cast<char*>(buffer->opaque()) + kBufferBytes, kOverrunBytes);
    return run_options.run_options().stream()->MemZero(&oob_region,
                                                       kOverrunBytes);
  }

  int duration_ns_;
  bool should_fail_;
  bool write_past_allocated_buffer_;
  std::vector<const void*> first_input_addresses_;
  std::vector<int> duration_sequence_;
  int timed_runs_ = 0;
};

absl::StatusOr<ScopedShapedBuffer> CreateTestBuffer(
    se::DeviceAddressAllocator* allocator, se::StreamExecutor* stream_exec,
    se::Stream* stream, int32_t value) {
  Shape test_shape = ShapeUtil::MakeShape(S32, {});
  ABSL_ASSIGN_OR_RETURN(auto* transfer_manager,
                   TransferManager::GetForPlatform(stream_exec->GetPlatform()));
  ABSL_ASSIGN_OR_RETURN(ScopedShapedBuffer output,
                   transfer_manager->AllocateScopedShapedBuffer(
                       test_shape, allocator, stream_exec->device_ordinal()));
  Literal literal = LiteralUtil::CreateR0<int32_t>(value);
  ABSL_RETURN_IF_ERROR(
      transfer_manager->TransferLiteralToDevice(stream, literal, output));
  return output;
}

absl::StatusOr<ScopedShapedBuffer> CreateTupleTestBuffer(
    se::DeviceAddressAllocator* allocator, se::StreamExecutor* stream_exec,
    se::Stream* stream, int32_t value1, int32_t value2) {
  Shape test_shape = ShapeUtil::MakeShape(S32, {});
  Shape test_shape_tuple = ShapeUtil::MakeTupleShape({test_shape, test_shape});
  ABSL_ASSIGN_OR_RETURN(auto* transfer_manager,
                   TransferManager::GetForPlatform(stream_exec->GetPlatform()));
  ABSL_ASSIGN_OR_RETURN(
      ScopedShapedBuffer output,
      transfer_manager->AllocateScopedShapedBuffer(
          test_shape_tuple, allocator, stream_exec->device_ordinal()));
  Literal literal1 = LiteralUtil::CreateR0<int32_t>(value1);
  Literal literal2 = LiteralUtil::CreateR0<int32_t>(value2);
  Literal tuple_literal = LiteralUtil::MakeTuple({&literal1, &literal2});
  ABSL_RETURN_IF_ERROR(
      transfer_manager->TransferLiteralToDevice(stream, tuple_literal, output));
  return output;
}

class GpuProfilerTest : public HloHardwareIndependentTestBase {
 public:
  GpuProfilerTest() {
    se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
    std::vector<se::StreamExecutor*> executors =
        PlatformUtil::GetStreamExecutors(platform).value();
    stream_exec_ = executors[0];
    allocator_ =
        std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
            stream_exec_);
  }

  absl::StatusOr<int64_t> GetScratchBytes(absl::string_view hlo_text) {
    NVPTXCompiler compiler;
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    module->mutable_config()
        .mutable_debug_options()
        .clear_xla_gpu_enable_command_buffer();
    ABSL_ASSIGN_OR_RETURN(auto gpu_executable,
                     compiler.RunBackend(std::move(module), stream_exec_,
                                         GpuCompiler::CompileOptions()));
    auto profiler =
        GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<InputBuffers> buffers,
                     profiler->CreateInputBuffers(gpu_executable.get()));
    ABSL_ASSIGN_OR_RETURN(ProfileResult profile,
                     profiler->Profile(gpu_executable.get(), *buffers));
    return profile.scratch_bytes;
  }

  se::StreamExecutor* stream_exec_;
  std::unique_ptr<se::DeviceAddressAllocator> allocator_;
};

TEST_F(GpuProfilerTest, CreateInputBuffersAndProfile) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(1000));
  EXPECT_EQ(profile.output_buffer->on_device_shape(),
            ShapeUtil::MakeShape(S32, {}));
  EXPECT_EQ(profile.scratch_bytes, 0);
}

TEST_F(GpuProfilerTest, ProfileWithRotatingBuffers) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  ProfileOptions options;
  options.rotating_buffer_bytes = 1024 * 1024;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(1000));
  EXPECT_EQ(profile.scratch_bytes, 0);
}

// The rotating sets must live at distinct addresses, or they would not evict
// each other from the cache, and they must hold identical bytes, or candidates
// reading different sets would produce legitimately different outputs and the
// output clustering in ConfigRunner would put every candidate in its own
// cluster.
TEST_F(GpuProfilerTest, RotatingBuffersAreDistinctCopiesOfTheSameData) {
  constexpr int64_t kElements = 1024;
  constexpr int64_t kBufferBytes = kElements * sizeof(float);
  constexpr int kExpectedSets = 3;
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      ROOT add = f32[1024] add(p0, p1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);

  ProfileOptions options;
  options.should_init_buffers = true;
  // Two inputs per set, so this budget asks for exactly kExpectedSets.
  options.rotating_buffer_bytes = kExpectedSets * 2 * kBufferBytes;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));

  const std::vector<RedzoneBuffers>& sets =
      static_cast<GpuInputBuffers*>(buffers.get())->redzone_buffers;
  ASSERT_EQ(sets.size(), kExpectedSets);

  ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  for (int input = 0; input < 2; ++input) {
    std::vector<float> reference(kElements);
    std::vector<const void*> seen_addresses;
    for (const RedzoneBuffers& set : sets) {
      ASSERT_EQ(set.input_buffers().size(), 2);
      const se::DeviceAddressBase& buffer = set.input_buffers()[input];
      ASSERT_EQ(static_cast<int64_t>(buffer.size()), kBufferBytes);
      EXPECT_THAT(seen_addresses, ::testing::Not(::testing::Contains(
                                      buffer.opaque())));
      seen_addresses.push_back(buffer.opaque());

      std::vector<float> host(kElements);
      TF_ASSERT_OK(stream->Memcpy(host.data(), buffer, kBufferBytes));
      TF_ASSERT_OK(stream->BlockHostUntilDone());
      if (&set == &sets.front()) {
        reference = host;
      } else {
        EXPECT_EQ(host, reference);
      }
    }
    // The buffers really were initialized, so the equality above is not
    // trivially comparing two runs of zeros.
    EXPECT_THAT(reference, ::testing::Contains(::testing::Ne(0.0f)));
  }
}

// The index has to advance on every execution, not once per candidate. If the
// warm-up and the timed run shared a set, the warm-up would pull exactly the
// timed run's data into the cache and the rotation would achieve nothing.
TEST_F(GpuProfilerTest, RotationAdvancesOnEveryExecutionIncludingWarmUp) {
  constexpr int64_t kBufferBytes = 1024 * sizeof(float);
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT n = f32[1024] negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);

  ProfileOptions options;
  // A single input per set, so this budget asks for exactly three sets.
  options.rotating_buffer_bytes = 3 * kBufferBytes;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_EQ(
      static_cast<GpuInputBuffers*>(buffers.get())->redzone_buffers.size(), 3);

  TF_ASSERT_OK(profiler->Profile(&mock_executable, *buffers).status());
  const std::vector<const void*>& addresses =
      mock_executable.first_input_addresses();
  // Warm-up run and timed run, on sets 0 and 1.
  ASSERT_EQ(addresses.size(), 2);
  EXPECT_NE(addresses[0], addresses[1]);

  // The next candidate continues the cycle rather than restarting it, so with
  // three sets it lands on 2 and then wraps back to 0.
  TF_ASSERT_OK(profiler->Profile(&mock_executable, *buffers).status());
  ASSERT_EQ(addresses.size(), 4);
  EXPECT_NE(addresses[1], addresses[2]);
  EXPECT_NE(addresses[2], addresses[3]);
  EXPECT_EQ(addresses[3], addresses[0]);
}

// ConfigRunner clusters candidates by comparing their outputs against each
// other, so two runs that read different rotating sets must still produce
// bit-identical outputs. If they did not, every candidate would land in its
// own cluster and the correctness machinery would collapse.
TEST_F(GpuProfilerTest, RotationKeepsOutputsIdenticalAcrossRuns) {
  constexpr int64_t kBufferBytes = 1024 * sizeof(float);
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      ROOT a = f32[1024] add(p0, p1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       PlatformUtil::GetDefaultPlatform());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler,
                       Compiler::GetForPlatform(platform->id()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  module->mutable_config()
      .mutable_debug_options()
      .clear_xla_gpu_enable_command_buffer();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                       compiler->RunBackend(std::move(module), stream_exec_,
                                            GpuCompiler::CompileOptions()));

  ProfileOptions options;
  options.should_init_buffers = true;
  options.rotating_buffer_bytes = 3 * 2 * kBufferBytes;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(executable.get()));
  ASSERT_EQ(
      static_cast<GpuInputBuffers*>(buffers.get())->redzone_buffers.size(), 3);

  // The timed runs of these two calls read sets 1 and 0 respectively.
  ASSERT_OK_AND_ASSIGN(ProfileResult first,
                       profiler->Profile(executable.get(), *buffers));
  ASSERT_OK_AND_ASSIGN(ProfileResult second,
                       profiler->Profile(executable.get(), *buffers));
  ASSERT_TRUE(first.output_buffer.has_value());
  ASSERT_TRUE(second.output_buffer.has_value());
  TF_EXPECT_OK(profiler->CheckOutputBuffer(*first.output_buffer,
                                           *second.output_buffer,
                                           /*rtol=*/0.0));
}

// The flush kernel runs on the same stream as the executable and must not be
// charged to the candidate. The reported duration comes from an event based
// timer started inside the executable, so the mock's fixed duration should come
// back unchanged.
TEST_F(GpuProfilerTest, CacheFlushDoesNotChangeReportedDuration) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT n = f32[1024] negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, /*duration_ns=*/1000);

  ProfileOptions options;
  options.cache_flush_bytes = 64 * 1024 * 1024;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(1000));

  // Both executions still happen; flushing must not swallow the warm-up.
  EXPECT_EQ(mock_executable.first_input_addresses().size(), 2);
}

// A flush budget that cannot be allocated must degrade to no flushing rather
// than fail compilation.
TEST_F(GpuProfilerTest, ImpossibleCacheFlushBudgetDegradesGracefully) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, /*duration_ns=*/1000);

  ProfileOptions options;
  options.cache_flush_bytes = int64_t{1} << 46;  // 64 TiB
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_NE(profiler, nullptr);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(1000));
}

TEST_F(GpuProfilerTest, MultipleTimedRunsReportTheMedian) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, /*duration_ns=*/0);
  // Deliberately out of order, and with an outlier at each end, so a mean
  // (3000) or a min (1000) would both give a different answer than the
  // median.
  mock_executable.SetDurationSequence({5000, 1000, 3000, 2000, 4000});

  ProfileOptions options;
  options.num_timed_runs = 5;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));

  EXPECT_EQ(mock_executable.timed_runs(), 5);
  EXPECT_EQ(profile.duration, absl::Nanoseconds(3000));
  EXPECT_TRUE(profile.output_buffer.has_value());
}

TEST_F(GpuProfilerTest, SingleTimedRunIsTheDefault) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, /*duration_ns=*/1000);
  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(mock_executable.timed_runs(), 1);
  EXPECT_EQ(profile.duration, absl::Nanoseconds(1000));
}

TEST_F(GpuProfilerTest, RotationIsDisabledByDefault) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT n = f32[1024] negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  EXPECT_EQ(
      static_cast<GpuInputBuffers*>(buffers.get())->redzone_buffers.size(), 1);
}

TEST_F(GpuProfilerTest, RejectsCandidateThatWritesPastAllocatedBuffer) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, /*duration_ns=*/1000,
                                 /*should_fail=*/false,
                                 /*write_past_allocated_buffer=*/true);

  ProfileOptions options;
  options.redzone_padding_bytes = 1024;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  EXPECT_THAT(profiler->Profile(&mock_executable, *buffers),
              StatusIs(absl::StatusCode::kInternal,
                       ::testing::HasSubstr("Redzone mismatch")));
}

TEST_F(GpuProfilerTest, FailingExecutablesReturnStatus) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, /*duration_ns=*/0,
                                 /*should_fail=*/true);

  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  EXPECT_THAT(profiler->Profile(&mock_executable, *buffers),
              StatusIs(absl::StatusCode::kInternal));
}

class GpuProfilerTestWithRedzonePadding
    : public GpuProfilerTest,
      public ::testing::WithParamInterface<int> {};

TEST_P(GpuProfilerTestWithRedzonePadding, CheckInputBuffers) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  ProfileOptions options;
  options.redzone_padding_bytes = GetParam();
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  TF_EXPECT_OK(profiler->CheckInputBuffers(*buffers));
}

INSTANTIATE_TEST_SUITE_P(GpuProfilerTestWithRedzonePadding,
                         GpuProfilerTestWithRedzonePadding,
                         ::testing::Values(0, 1024));

TEST_F(GpuProfilerTest, CheckOutputBufferWhenBuffersAreSame) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());

  ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer output,
                       CreateTestBuffer(allocator.get(), stream_exec_,
                                        stream.get(), /*value=*/1));
  ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer reference,
                       CreateTestBuffer(allocator.get(), stream_exec_,
                                        stream.get(), /*value=*/1));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kOk));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWhenBuffersAreDifferent) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer output,
                       CreateTestBuffer(allocator.get(), stream_exec_,
                                        stream.get(), /*value=*/1));
  ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer reference,
                       CreateTestBuffer(allocator.get(), stream_exec_,
                                        stream.get(), /*value=*/2));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWithTupleShapeAreSame) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());

  ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer output,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/2));
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer reference,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/2));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kOk));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWithTupleShapeAreDifferent) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());

  ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer reference,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/2));
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer output_error_in_first_element,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/0, /*value2=*/2));
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer output_error_in_second_element,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/3));
  EXPECT_THAT(profiler->CheckOutputBuffer(output_error_in_first_element,
                                          reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(profiler->CheckOutputBuffer(output_error_in_second_element,
                                          reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(GpuProfilerTest, CheckScratchBytesArePopulated) {
  constexpr int64_t kScratchBytes = 26738688;
  std::string hlo_text = absl::Substitute(R"hlo(
    HloModule gemm_fusion_dot.1
    ENTRY %entry_computation (lhs: bf16[3072,512], rhs: bf16[3840,3072]) -> bf16[512,3840] {
      %lhs = bf16[3072,512]{1,0} parameter(0)
      %rhs = bf16[3840,3072]{1,0} parameter(1)
      %custom-call.1 = (bf16[512,3840]{0,1}, s8[$1]{0}) custom-call(%lhs, %rhs), custom_call_target="__cublas$$lt$$matmul", $0
      ROOT %get-tuple-element = bf16[512,3840]{0,1} get-tuple-element(%custom-call.1), index=0
    }
  )hlo",
                                          kGemmBackendConfig, kScratchBytes);
  EXPECT_THAT(GetScratchBytes(hlo_text), IsOkAndHolds(kScratchBytes));
}

TEST_F(GpuProfilerTest, CheckScratchBytesAreDeDuplicated) {
  constexpr int64_t kScratchBytes = 26738688;
  std::string hlo_text = absl::Substitute(R"hlo(
    HloModule gemm_fusion_dot.2
    ENTRY %entry_computation (lhs: bf16[3072,512], rhs: bf16[3840,3072]) -> (bf16[512,3840], bf16[512,3840]) {
      %lhs = bf16[3072,512]{1,0} parameter(0)
      %rhs = bf16[3840,3072]{1,0} parameter(1)
      %custom-call.1 = (bf16[512,3840]{0,1}, s8[$1]{0}) custom-call(%lhs, %rhs), custom_call_target="__cublas$$lt$$matmul", $0
      %val1 = bf16[512,3840]{0,1} get-tuple-element(%custom-call.1), index=0

      %custom-call.2 = (bf16[512,3840]{0,1}, s8[$1]{0}) custom-call(%lhs, %rhs), custom_call_target="__cublas$$lt$$matmul", $0
      %val2 = bf16[512,3840]{0,1} get-tuple-element(%custom-call.2), index=0

      ROOT %tuple = (bf16[512,3840]{0,1}, bf16[512,3840]{0,1}) tuple(%val1, %val2)
    }
  )hlo",
                                          kGemmBackendConfig, kScratchBytes);

  EXPECT_THAT(GetScratchBytes(hlo_text), IsOkAndHolds(kScratchBytes));
}

}  // namespace

}  // namespace gpu
}  // namespace xla
