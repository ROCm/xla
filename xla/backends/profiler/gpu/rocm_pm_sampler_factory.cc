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

#include "xla/backends/profiler/gpu/rocm_pm_sampler_factory.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "xla/backends/profiler/gpu/rocm_pm_sampler.h"
#include "xla/backends/profiler/gpu/rocm_pm_sampler_impl.h"

namespace xla {
namespace profiler {

absl::StatusOr<std::unique_ptr<RocmPmSampler>> CreateRocmPmSampler(
    const std::vector<rocprofiler_context_id_t>& gpu_contexts,
    const std::vector<rocprofiler_agent_id_t>& gpu_agents,
    const RocmPmSamplerOptions& options) {
  auto sampler_or =
      RocmPmSamplerImpl::Create(gpu_contexts, gpu_agents, options);
  if (!sampler_or.ok()) {
    return sampler_or.status();
  }
  return std::unique_ptr<RocmPmSampler>(std::move(sampler_or).value());
}

}  // namespace profiler
}  // namespace xla
