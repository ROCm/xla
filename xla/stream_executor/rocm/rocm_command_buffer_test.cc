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

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

using absl_testing::StatusIs;

// Static shared memory of StaticAndDynShmemKernel (gpu_test_kernels_lib.cu.h).
constexpr int64_t kStaticSharedMemoryBytes = 16 * 1024;
constexpr uint32_t kThreads = 256;
constexpr char kKernelName[] = "StaticAndDynShmemKernel";

class RocmCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(Platform * platform,
                         PlatformManager::PlatformWithName("ROCM"));
    ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    ASSERT_OK_AND_ASSIGN(fatbin_, GetGpuTestKernelsFatbin("ROCM"));
    // What a launch can use besides the kernel's static shared memory. HIP
    // accepts node requests up to the device limit, so requests between the
    // two are only caught by RocmCommandBuffer.
    limit_ = executor_->GetDeviceDescription().shared_memory_per_block_optin() -
             kStaticSharedMemoryBytes;
    buf_ = executor_->AllocateArray<uint8_t>(kThreads);
    for (uint32_t i = 0; i < kThreads; ++i) {
      input_[i] = static_cast<uint8_t>(i * i + 3);
    }
    for (uint32_t i = 0; i < kThreads; ++i) {
      expected_[i] = static_cast<uint8_t>(input_[i] + input_[kThreads - 1 - i]);
    }
  }

  // Runs `cmd_buffer` on a fresh copy of the input and checks the result.
  void SubmitAndCheck(CommandBuffer* cmd_buffer) {
    ASSERT_OK(stream_->Memcpy(&buf_, input_.data(), kThreads));
    ASSERT_OK(cmd_buffer->Submit(stream_.get()));
    std::vector<uint8_t> output(kThreads);
    ASSERT_OK(stream_->Memcpy(output.data(), buf_, kThreads));
    ASSERT_OK(stream_->BlockHostUntilDone());
    EXPECT_EQ(output, expected_);
  }

  StreamExecutor* executor_ = nullptr;
  std::unique_ptr<Stream> stream_;
  std::vector<uint8_t> fatbin_;
  int64_t limit_ = 0;
  DeviceAddress<uint8_t> buf_;
  std::vector<uint8_t> input_ = std::vector<uint8_t>(kThreads);
  std::vector<uint8_t> expected_ = std::vector<uint8_t>(kThreads);
};

TEST_F(RocmCommandBufferTest, KernelDynamicSharedMemoryAboveKernelLimit) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> kernel,
      executor_->LoadKernel(KernelLoaderSpec::CreateCudaCubinInMemorySpec(
          fatbin_, kKernelName, /*arity=*/1)));
  ASSERT_EQ(kernel->metadata().shared_memory_bytes(), kStaticSharedMemoryBytes);

  ASSERT_OK_AND_ASSIGN(auto cmd_buffer, executor_->CreateCommandBuffer(
                                            CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(
      const CommandBuffer::Command* launch,
      cmd_buffer->CreateLaunch(ThreadDim(kThreads), BlockDim(1), {}, *kernel,
                               *PackKernelArgs(limit_, buf_), {}));
  ASSERT_OK(cmd_buffer->Finalize());
  ASSERT_NO_FATAL_FAILURE(SubmitAndCheck(cmd_buffer.get()));

  ASSERT_OK(cmd_buffer->Update());
  EXPECT_THAT(
      cmd_buffer->UpdateLaunch(launch, ThreadDim(kThreads), BlockDim(1), {},
                               *kernel, *PackKernelArgs(limit_ + 256, buf_)),
      StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK_AND_ASSIGN(auto other, executor_->CreateCommandBuffer(
                                       CommandBuffer::Mode::kPrimary));
  EXPECT_THAT(other->CreateLaunch(ThreadDim(kThreads), BlockDim(1), {}, *kernel,
                                  *PackKernelArgs(limit_ + 256, buf_), {}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(RocmCommandBufferTest, NativeKernelDynamicSharedMemoryAboveKernelLimit) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  hipModule_t module = nullptr;
  ASSERT_EQ(hipModuleLoadData(&module, fatbin_.data()), hipSuccess);
  // The command buffers reference a function from `module`, so they must be
  // destroyed before it is unloaded, on every exit path.
  std::unique_ptr<CommandBuffer> cmd_buffer;
  std::unique_ptr<CommandBuffer> other;
  absl::Cleanup unload_module = [&] {
    cmd_buffer.reset();
    other.reset();
    EXPECT_EQ(hipModuleUnload(module), hipSuccess);
  };
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipModuleGetFunction(&function, module, kKernelName), hipSuccess);
  int static_bytes = 0;
  ASSERT_EQ(hipFuncGetAttribute(&static_bytes,
                                HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function),
            hipSuccess);
  ASSERT_EQ(static_bytes, kStaticSharedMemoryBytes);
  NativeKernel kernel{function, kKernelName};

  ASSERT_OK_AND_ASSIGN(cmd_buffer, executor_->CreateCommandBuffer(
                                       CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(
      const CommandBuffer::Command* launch,
      cmd_buffer->CreateLaunch(ThreadDim(kThreads), BlockDim(1), {}, kernel,
                               *PackKernelArgs(limit_, buf_), {}));
  ASSERT_OK(cmd_buffer->Finalize());
  ASSERT_NO_FATAL_FAILURE(SubmitAndCheck(cmd_buffer.get()));

  ASSERT_OK(cmd_buffer->Update());
  EXPECT_THAT(
      cmd_buffer->UpdateLaunch(launch, ThreadDim(kThreads), BlockDim(1), {},
                               kernel, *PackKernelArgs(limit_ + 256, buf_)),
      StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK_AND_ASSIGN(
      other, executor_->CreateCommandBuffer(CommandBuffer::Mode::kPrimary));
  EXPECT_THAT(other->CreateLaunch(ThreadDim(kThreads), BlockDim(1), {}, kernel,
                                  *PackKernelArgs(limit_ + 256, buf_), {}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace stream_executor::gpu
