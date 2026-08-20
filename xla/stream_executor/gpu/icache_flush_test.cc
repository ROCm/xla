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

#include "xla/stream_executor/gpu/icache_flush.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace gpu {
namespace {

bool IsRocm() { return GpuPlatformName() == "ROCM"; }

StreamExecutor* GetStreamExecutor() {
  Platform* platform =
      PlatformManager::PlatformWithName(GpuPlatformName()).value();
  return platform->ExecutorForDevice(0).value();
}

TEST(IcacheFlushTest, FlushesWithoutError) {
  if (!IsRocm()) {
    GTEST_SKIP() << "Instruction cache flushing is only implemented for ROCm.";
  }
  StreamExecutor* stream_exec = GetStreamExecutor();

  TF_ASSERT_OK_AND_ASSIGN(IcacheFlusher flusher,
                          IcacheFlusher::Create(stream_exec));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          stream_exec->CreateStream());

  // Repeated flushes on the same stream must all succeed; this is how the
  // autotuner uses it.
  TF_ASSERT_OK(flusher.Flush(stream.get()));
  TF_ASSERT_OK(flusher.Flush(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
}

TEST(IcacheFlushTest, CreateFailsOnPlatformsWithoutAFlushKernel) {
  if (IsRocm()) {
    GTEST_SKIP() << "ROCm does provide a flush kernel.";
  }
  absl::StatusOr<IcacheFlusher> flusher =
      IcacheFlusher::Create(GetStreamExecutor());
  EXPECT_FALSE(flusher.ok());
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
