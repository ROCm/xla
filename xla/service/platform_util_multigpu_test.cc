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

#include <memory>
#include <set>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;

// As when device 1's first convolution initializes while device 0 records a
// command buffer. Only meaningful while nothing earlier in this process creates
// device 1's DNN support.
TEST(PlatformUtilMultiGpuTest,
     CreatingDnnSupportLeavesAnotherDevicesCaptureIntact) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       PlatformUtil::GetPlatform("gpu"));
  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }
  ASSERT_OK_AND_ASSIGN(
      std::vector<se::StreamExecutor*> executors,
      PlatformUtil::GetStreamExecutors(platform, std::set<int>{0, 1}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executors[0]->CreateStream());

  se::dnn::DnnSupport* dnn = nullptr;
  EXPECT_THAT(se::TraceCommandBufferFactory::Create(
                  executors[0], stream.get(),
                  [&](se::Stream*) {
                    std::unique_ptr<tsl::Thread> thread(
                        tsl::Env::Default()->StartThread(
                            tsl::ThreadOptions(), "dnn_support",
                            [&] { dnn = executors[1]->AsDnn(); }));
                    return absl::OkStatus();
                  },
                  se::CommandBuffer::Mode::kPrimary),
              IsOk());
  EXPECT_NE(dnn, nullptr);
}

}  // namespace
}  // namespace xla
