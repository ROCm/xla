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

#include "xla/primitive_util.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/gpu/nan_check_kernel_lib.cu.h"
#include "xla/stream_executor/platform/initialize.h"

namespace stream_executor::rocm {


namespace {

static void RegisterNanCheckKernelRocmImpl() {
  auto register_kernel = [&](auto primitive_type_constant) {
    gpu::RegisterNanCheckKernelParametrized<
        xla::primitive_util::NativeTypeOf<primitive_type_constant()>>(
        stream_executor::rocm::kROCmPlatformId);
  };
  // TODO(rocm): All fp types? 
  xla::primitive_util::FloatingPointTypeForEach(register_kernel);
}

}  // namespace
}  // namespace stream_executor::rocm

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    RegisterNanCheckKernelRocm,
    stream_executor::rocm::RegisterNanCheckKernelRocmImpl());
