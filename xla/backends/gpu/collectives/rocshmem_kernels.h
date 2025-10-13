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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_ROCSHMEM_KERNELS_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_ROCSHMEM_KERNELS_H_

#include "absl/status/status.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "third_party/rocshmem/rocshmem.hpp"
#include "third_party/rocshmem/rocshmem_COLL.hpp"

namespace rocshmem {

using stream_executor::gpu::GpuStreamHandle;

using longlong = long long; // needed for macro substitution

template < class Type >
absl::Status allreduce_on_stream(rocshmem_team_t *teams, size_t max_teams, 
       Type *dest, const Type *source, size_t nreduce, 
       xla::ReductionKind kind, GpuStreamHandle stream);

template < class Type >
absl::Status get_nbi_on_stream(int peer, Type *dest, const Type *source, 
            size_t nelems, GpuStreamHandle stream);

template < class Type >
absl::Status put_nbi_on_stream(int peer, Type *dest, const Type *source, 
            size_t nelems, GpuStreamHandle stream);

void rocshmem_quiet_on_stream(GpuStreamHandle stream);

} // namespace rocshmem

#endif // XLA_BACKENDS_GPU_COLLECTIVES_ROCSHMEM_KERNELS_H_
