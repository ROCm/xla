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

// Returns a raw pointer to a GPU PjRt client that is created once and reused
// across all tests in this process. Ownership is held by the static unique_ptr
// inside this function, which ensures the client is properly destroyed at
// process exit. This avoids the expensive GPU context teardown and
// re-initialization that would otherwise happen between every test fixture
// construction and destruction.
// Since gtest runs tests sequentially within a process, the shared
// client is never accessed concurrently and requires no synchronization.
// With Bazel test sharding each shard is a separate OS process and gets its
// own independent cached client, so sharding is fully compatible.
PjRtClient* GetCachedPjRtClientForTest() {
  CHECK(ShouldUsePjRt())
      << "PjRt is required for tests extending HloPjRtTestBase.";
  // The static-local unique_ptr is initialized at most once (C++11 guarantee).
  // The GPU client it owns is created on the first fixture construction and
  // destroyed by the static destructor at process exit.
  static const std::unique_ptr<PjRtClient> kClient = []() {
    absl::StatusOr<std::unique_ptr<PjRtClient>> client =
        GetGlobalPjRtClientTestFactory().Get()();
    CHECK_OK(client.status())
        << "Failed to create PjRt client. " << client.status();
    return std::move(*client);
  }();
  return kClient.get();
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

HloPjRtTestBase::HloPjRtTestBase(PjRtClient* client,
                                 HloPjRtTestBaseOptions options)
    // Use HloRunnerPjRt's non-owning constructor (PjRtClient* overload) so
    // that when the test fixture is torn down the cached PjRtClient is NOT
    // freed. The static unique_ptr in GetCachedPjRtClientForTest() retains
    // ownership and the client remains alive for the next test fixture to
    // reuse. MakeHloRunnerPjRtAotAware() is bypassed here because it requires
    // a unique_ptr<PjRtClient> (owning); AoT mode is therefore not supported
    // on the cached-client path.
    : HloRunnerAgnosticTestBase(
          std::make_unique<HloRunnerPjRt>(client),
          GetGlobalPjRtClientTestFactory().GetDeviceShapeRepresentationFn(
              client),
          GetGlobalPjRtClientTestFactory().GetDeviceShapeSizeFn(client),
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
