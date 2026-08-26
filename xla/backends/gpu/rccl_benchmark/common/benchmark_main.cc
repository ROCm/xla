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

// Entry point shared by every case in this directory.
//
// RCCL snapshots its environment when the library loads, which is before main
// runs. Setting NCCL_DEBUG from here therefore has no effect at all: the
// directory gets created, no log is ever written, and every case reports that
// it could not confirm its path. Rather than depend on the caller getting this
// right, main re-executes itself once with the environment in place.
//
// The same constraint is why cases that need different library parameters have
// to run as separate processes instead of as separate tests in one binary.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/backends/gpu/rccl_benchmark/common/path_assert.h"

namespace {

// Set once the environment has been prepared, so the re-exec happens at most
// once no matter how the binary was invoked.
constexpr char kEnvReadyMarker[] = "RCCL_BENCHMARK_ENV_READY";

}  // namespace

int main(int argc, char** argv) {
  const bool environment_ready = std::getenv(kEnvReadyMarker) != nullptr;

  const absl::Status status =
      xla::gpu::rccl_benchmark::ConfigureRcclDebugLogging();
  if (!status.ok()) {
    std::fprintf(stderr,
                 "Failed to configure RCCL debug logging: %s\n"
                 "Without it no case can confirm which code path ran, so the "
                 "binary refuses to report results.\n",
                 status.ToString().c_str());
    return 2;
  }

  if (!environment_ready) {
    setenv(kEnvReadyMarker, "1", /*overwrite=*/1);
    execv("/proc/self/exe", argv);
    // execv only returns on failure.
    std::fprintf(stderr,
                 "Failed to re-execute with RCCL debug logging configured. "
                 "Set NCCL_DEBUG, NCCL_DEBUG_SUBSYS and NCCL_DEBUG_FILE before "
                 "starting this binary instead.\n");
    return 2;
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
