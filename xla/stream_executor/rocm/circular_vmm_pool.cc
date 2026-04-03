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

#include "xla/stream_executor/rocm/circular_vmm_pool.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <sched.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_memory_reservation.h"
#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {

CircularVmmPool::CircularVmmPool(StreamExecutor* executor, int num_slots,
                                 std::vector<Slot> slots,
                                 std::vector<BufferLayout> layout,
                                 volatile uint64_t* timeline,
                                 void* timeline_host_ptr)
    : executor_(executor),
      num_slots_(num_slots),
      slots_(std::move(slots)),
      layout_(std::move(layout)),
      timeline_(timeline),
      timeline_host_ptr_(timeline_host_ptr) {}

CircularVmmPool::~CircularVmmPool() {
  slots_.clear();
  if (timeline_host_ptr_ != nullptr) {
    auto status = ToStatus(wrap::hipFree(timeline_host_ptr_),
                           "hipFree for circular VMM pool timeline");
    if (!status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
}

absl::StatusOr<std::unique_ptr<CircularVmmPool>> CircularVmmPool::Create(
    StreamExecutor* executor, absl::Span<const uint64_t> buffer_sizes,
    int num_slots) {
  if (num_slots < 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("num_slots must be >= 1, got %d", num_slots));
  }
  if (buffer_sizes.empty()) {
    return absl::InvalidArgumentError("buffer_sizes must not be empty");
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();

  // Query allocation granularity.
  hipDevice_t device;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGet(&device, executor->device_ordinal())));

  hipMemAllocationProp props = {};
  props.type = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device;
  props.requestedHandleTypes = hipMemHandleTypeNone;

  size_t granularity = 0;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipMemGetAllocationGranularity(
      &granularity, &props, hipMemAllocationGranularityRecommended)));

  // Compute per-buffer offsets and total slot size.
  std::vector<BufferLayout> layout;
  layout.reserve(buffer_sizes.size());
  uint64_t total_slot_size = 0;
  for (uint64_t size : buffer_sizes) {
    layout.push_back({total_slot_size, size});
    total_slot_size += xla::RoundUpTo<uint64_t>(size, granularity);
  }

  if (total_slot_size == 0) {
    return absl::InvalidArgumentError("Total slot size is zero");
  }

  // Allocate signal memory for the monotonic timeline counter.
  // hipStreamWriteValue64 on AMD hardware requires the target pointer to be
  // allocated with hipMallocSignalMemory.
  void* timeline_host_ptr = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipExtMallocWithFlags(&timeline_host_ptr,
                                           sizeof(uint64_t),
                                           hipMallocSignalMemory),
               "hipExtMallocWithFlags for circular VMM pool timeline"));
  *static_cast<volatile uint64_t*>(timeline_host_ptr) = 0;

  // Create N slots, each with its own physical chunk and VA range.
  std::vector<Slot> slots;
  slots.reserve(num_slots);
  for (int i = 0; i < num_slots; ++i) {
    TF_ASSIGN_OR_RETURN(auto physical,
                        RocmRawMemoryAllocation::Create(executor,
                                                        total_slot_size));
    TF_ASSIGN_OR_RETURN(auto va_range,
                        RocmMemoryReservation::Create(executor,
                                                      total_slot_size));

    TF_ASSIGN_OR_RETURN(
        auto mapping,
        va_range->MapTo(0, 0, physical->address().size(), *physical));

    void* va_base = va_range->address().opaque();
    std::vector<DeviceAddressBase> buffer_addresses;
    buffer_addresses.reserve(layout.size());
    for (const auto& buf : layout) {
      void* addr = static_cast<char*>(va_base) + buf.offset;
      buffer_addresses.emplace_back(addr, buf.size);
    }

    LOG(INFO) << absl::StrFormat(
        "CircularVmmPool slot %d: VA base=%p, physical size=%d, "
        "%d buffers",
        i, va_base, total_slot_size, buffer_addresses.size());

    slots.push_back({std::move(physical), std::move(va_range),
                     std::move(mapping), std::move(buffer_addresses)});
  }

  return absl::WrapUnique(new CircularVmmPool(
      executor, num_slots, std::move(slots), std::move(layout),
      static_cast<volatile uint64_t*>(timeline_host_ptr), timeline_host_ptr));
}

absl::StatusOr<std::vector<DeviceAddressBase>> CircularVmmPool::AcquireNextSlot(
    uint64_t iteration) {
  int slot_idx = iteration % num_slots_;

  // The slot was last used at iteration (iteration - num_slots_). It is safe
  // to reuse when the GPU has signaled completion of that earlier iteration.
  if (iteration >= static_cast<uint64_t>(num_slots_)) {
    uint64_t required = iteration - num_slots_ + 1;
    uint64_t completed = __atomic_load_n(timeline_, __ATOMIC_ACQUIRE);
    constexpr int kMaxSpinIterations = 1000;
    constexpr auto kTimeout = absl::Seconds(30);
    auto deadline = absl::Now() + kTimeout;
    int spin_count = 0;
    while (completed < required) {
      if (++spin_count > kMaxSpinIterations) {
        if (absl::Now() > deadline) {
          return absl::DeadlineExceededError(absl::StrFormat(
              "CircularVmmPool: timed out waiting for slot %d "
              "(required=%d, completed=%d)",
              iteration % num_slots_, required, completed));
        }
        sched_yield();
        spin_count = 0;
      }
      completed = __atomic_load_n(timeline_, __ATOMIC_ACQUIRE);
    }
  }

  return slots_[slot_idx].buffer_addresses;
}

absl::Status CircularVmmPool::ReleaseSlot(Stream* stream, uint64_t iteration) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  hipStream_t hip_stream =
      static_cast<hipStream_t>(stream->platform_specific_handle().stream);
  uint64_t signal_value = iteration + 1;
  void* timeline_ptr = reinterpret_cast<void*>(
      const_cast<uint64_t*>(timeline_));
  return ToStatus(
      wrap::hipStreamWriteValue64(hip_stream, timeline_ptr, signal_value, 0),
      "hipStreamWriteValue64 for circular VMM pool");
}

}  // namespace stream_executor::gpu
