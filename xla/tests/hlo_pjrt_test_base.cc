/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tests/hlo_pjrt_test_base.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/tests/aot_utils.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/pjrt_client_registry.h"

namespace xla {
namespace {

// Returns a shared_ptr to a GPU PjRt client that is created once and shared
// across all tests in this process.
//
// Ownership schema:
//
//   static shared_ptr<PjRtClient> kClient  <-- PRIMARY OWNER (refcount >= 1)
//             |
//             +-- per-test: shared_ptr<PjRtClient>   copy of kClient
//                           stored in HloRunnerPjRt::pjrt_client_
//                           (refcount += 1 during fixture lifetime)
//                           destroyed at fixture teardown (refcount -= 1)
//                           client NOT freed as long as kClient is alive
//
//   At process exit: kClient is destroyed (static destructor)
//                    → refcount drops to 0 → client is properly freed ✓
//
// Since gtest runs tests sequentially within a process, the shared client is
// never accessed concurrently and requires no synchronization.
// With Bazel test sharding each shard is a separate OS process and gets its
// own independent cached client, so sharding is fully compatible.
std::shared_ptr<PjRtClient> GetCachedPjRtClientForTest() {
  CHECK(ShouldUsePjRt())
      << "PjRt is required for tests extending HloPjRtTestBase.";
  // The static-local shared_ptr is initialized at most once (C++11 guarantee).
  // The GPU client it owns is created on the first fixture construction and
  // destroyed by the static destructor at process exit.
  static const std::shared_ptr<PjRtClient> kClient = []() {
    absl::StatusOr<std::unique_ptr<PjRtClient>> client =
        GetGlobalPjRtClientTestFactory().Get()();
    CHECK_OK(client.status())
        << "Failed to create PjRt client. " << client.status();
    return std::move(*client);
  }();
  return kClient;  // Copies the shared_ptr: refcount += 1 for the caller.
}

HloRunnerAgnosticTestBaseOptions BuildOptions(HloPjRtTestBaseOptions options) {
  HloRunnerAgnosticTestBaseOptions new_options;
  new_options.verifier_layout_sensitive = options.verifier_layout_sensitive;
  new_options.allow_mixed_precision_in_hlo_verifier =
      options.allow_mixed_precision_in_hlo_verifier;
  new_options.instruction_can_change_layout_func =
      std::move(options.instruction_can_change_layout_func);
  new_options.swallow_execution_errors =
      HasPjRtAotAwareSwallowExecutionErrors();
  return new_options;
}
}  // namespace

HloPjRtTestBase::HloPjRtTestBase(HloPjRtTestBaseOptions options)
    : HloPjRtTestBase(GetCachedPjRtClientForTest(), std::move(options)) {}

HloPjRtTestBase::HloPjRtTestBase(std::shared_ptr<PjRtClient> client,
                                 HloPjRtTestBaseOptions options)
    // Use HloRunnerPjRt's shared_ptr constructor so that HloRunnerPjRt
    // holds a copy of the shared_ptr (refcount += 1). When the test fixture
    // is torn down, HloRunnerPjRt is destroyed and its copy of the shared_ptr
    // is released (refcount -= 1), but the static kClient in
    // GetCachedPjRtClientForTest() still holds a reference, so the
    // PjRtClient is NOT freed between tests.
    : HloRunnerAgnosticTestBase(
          std::make_unique<HloRunnerPjRt>(client),
          GetGlobalPjRtClientTestFactory().GetDeviceShapeRepresentationFn(
              client.get()),
          GetGlobalPjRtClientTestFactory().GetDeviceShapeSizeFn(client.get()),
          BuildOptions(std::move(options))) {}

HloPjRtTestBase::HloPjRtTestBase(
    DeviceShapeRepresentationFn device_shape_representation_fn,
    DeviceShapeSizeFn device_shape_size_fn, std::unique_ptr<PjRtClient> client,
    HloPjRtTestBaseOptions options)
    : HloRunnerAgnosticTestBase(MakeHloRunnerPjRtAotAware(std::move(client)),
                                std::move(device_shape_representation_fn),
                                std::move(device_shape_size_fn),
                                BuildOptions(std::move(options))) {}

}  // namespace xla
