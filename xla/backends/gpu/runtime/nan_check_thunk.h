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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NAN_CHECK_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NAN_CHECK_THUNK_H_

#include <cstdint>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

class NanCheckThunk : public Thunk {
 public:
  NanCheckThunk(ThunkInfo thunk_info, 
             const HloInstruction::InstructionVector& operands,
                std::vector<BufferAllocation::Slice>&& buffers);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  absl::Status Postprocess(se::Stream* stream, 
    absl::InlinedVector<se::DeviceMemoryBase, 4>&& buffers,
    std::atomic<uint32_t>& nan_signal);


  HloInstruction::InstructionVector operands_;
  std::vector<BufferAllocation::Slice> buffers_;
  int64_t run_no_ = 0;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NAN_CHECK_THUNK_H_
