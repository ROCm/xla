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

#ifndef XLA_STREAM_EXECUTOR_ROCM_CIRCULAR_VMM_POOL_H_
#define XLA_STREAM_EXECUTOR_ROCM_CIRCULAR_VMM_POOL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// A circular pool of VMM slots for update-free command buffer execution.
//
// Pre-allocates N physical memory chunks at startup, each permanently mapped to
// its own virtual address range. Iterations cycle through slots: iteration i
// uses slot (i % N). The GPU signals slot completion via hipStreamWriteValue64
// to coherent host memory; the CPU checks a monotonic timeline counter with a
// non-blocking memory read before reusing a slot.
//
// After the first N iterations (warmup), per-iteration overhead is zero: no
// hipMemMap, no hipMemUnmap, no hipEventSynchronize. Each slot has stable VA
// addresses, so HIP graphs recorded against a slot are replayed indefinitely.
class CircularVmmPool {
 public:
  // Per-buffer layout within each slot.
  struct BufferLayout {
    uint64_t offset;  // Byte offset within the slot's VA range.
    uint64_t size;    // Logical size requested by the caller.
  };

  // Creates a circular pool with `num_slots` slots. Each slot holds one
  // physical chunk large enough for all buffers described by `buffer_sizes`,
  // rounded up to the device's allocation granularity. Each chunk is mapped to
  // its own reserved VA range with P2P access enabled.
  static absl::StatusOr<std::unique_ptr<CircularVmmPool>> Create(
      StreamExecutor* executor, absl::Span<const uint64_t> buffer_sizes,
      int num_slots);

  // Returns the pre-computed device addresses for the given iteration's slot.
  // Blocks (spin-waits) if the slot is still in use by a previous iteration.
  absl::StatusOr<std::vector<DeviceAddressBase>> AcquireNextSlot(
      uint64_t iteration);

  // Enqueues a GPU-side timeline write so the CPU knows when the GPU is done
  // with this iteration's slot. Must be called after ExecuteThunksImpl returns.
  absl::Status ReleaseSlot(Stream* stream, uint64_t iteration);

  int num_slots() const { return num_slots_; }

  ~CircularVmmPool();
  CircularVmmPool(const CircularVmmPool&) = delete;
  CircularVmmPool& operator=(const CircularVmmPool&) = delete;

 private:
  struct Slot {
    std::unique_ptr<MemoryAllocation> physical;
    std::unique_ptr<MemoryReservation> va_range;
    MemoryReservation::ScopedMapping mapping;
    std::vector<DeviceAddressBase> buffer_addresses;
  };

  CircularVmmPool(StreamExecutor* executor, int num_slots,
                  std::vector<Slot> slots, std::vector<BufferLayout> layout,
                  volatile uint64_t* timeline, void* timeline_host_ptr);

  StreamExecutor* executor_;
  int num_slots_;
  std::vector<Slot> slots_;
  std::vector<BufferLayout> layout_;

  // Monotonic timeline counter in coherent host memory. The GPU advances it
  // via hipStreamWriteValue64 after each iteration; the CPU reads it to
  // determine when a slot is safe to reuse.
  volatile uint64_t* timeline_;
  void* timeline_host_ptr_;  // Raw pointer for hipHostFree.
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_CIRCULAR_VMM_POOL_H_
