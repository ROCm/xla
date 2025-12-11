/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/profiler/gpu/distributed_profiler_context.h"

#include "tsl/platform/logging.h"

namespace xla {
namespace profiler {

DistributedProfilerContextManager& DistributedProfilerContextManager::Get() {
  static DistributedProfilerContextManager instance;
  return instance;
}

void DistributedProfilerContextManager::SetDistributedContext(
    const DistributedProfilerContext& ctx) {
  absl::MutexLock lock(&mu_);
  
  if (context_set_) {
    LOG(WARNING) << "Distributed profiler context already set. "
                 << "Ignoring new context. "
                 << "(node_id=" << ctx.node_id << ", num_nodes=" << ctx.num_nodes << ")";
    return;
  }
  
  context_ = ctx;
  context_set_ = true;
  
  LOG(INFO) << "Distributed profiler context set: "
            << "node_id=" << ctx.node_id 
            << ", num_nodes=" << ctx.num_nodes
            << ", addresses=" << ctx.node_addresses.size()
            << ", neighbors=" << ctx.neighbors.size()
            << ", in_neighbors=" << ctx.in_neighbors.size()
            << ", output_dir=" << ctx.output_dir;
}

std::optional<DistributedProfilerContext> 
DistributedProfilerContextManager::GetDistributedContext() const {
  absl::MutexLock lock(&mu_);
  return context_;
}

bool DistributedProfilerContextManager::HasDistributedContext() const {
  absl::MutexLock lock(&mu_);
  return context_set_;
}

void DistributedProfilerContextManager::ResetContext() {
  absl::MutexLock lock(&mu_);
  context_.reset();
  context_set_ = false;
  LOG(INFO) << "Distributed profiler context reset";
}

}  // namespace profiler
}  // namespace xla

