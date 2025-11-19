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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NAN_CHECK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NAN_CHECK_H_

#include "absl/status/statusor.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

enum class NaNCheckerResult : uint32_t {
  OK = 0,
  NaN = 1,
  Inf = 2,
  LargeMagnitude = 3,
};

absl::Status LaunchNanCheckKernel(se::Stream* stream,
                                  const se::DeviceMemoryBase& buffer,
                                  const PrimitiveType element_type,
                                  float threshold, bool verbose,
                                  se::DeviceMemory<uint32_t>& nan_signal);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_NAN_CHECK_H_
