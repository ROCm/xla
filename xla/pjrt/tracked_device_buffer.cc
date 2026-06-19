/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/pjrt/tracked_device_buffer.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/future.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se_raw_buffer.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"

namespace xla {

ShapedBuffer RawSEDeviceMemory::AsShapedBuffer(
    PjRtDevice* device, const Shape& on_device_shape) const {
  ShapedBuffer shaped_buffer(on_device_shape, device->local_device_id().value(),
                             device->local_hardware_id().value());
  ShapeTree<se::DeviceAddressBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  CHECK(iterator != shaped_buffer.buffers().end());
  iterator->second = mem();
  ++iterator;
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

class AllocatedRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  AllocatedRawSEDeviceMemory(se::DeviceAddressBase value,
                             LocalDeviceState* local_device,
                             se::DeviceAddressAllocator* allocator)
      : RawSEDeviceMemory(value),
        allocator_(allocator),
        local_device_(local_device) {
    if (local_device_->allocation_model() ==
        LocalDeviceState::kComputeSynchronized) {
      sync_point_ = local_device_->GetNextComputeStreamSyncPoint();
    }
  }

  ~AllocatedRawSEDeviceMemory() override {
    if (!allocator_) {
      return;
    }
    const int device_ordinal = local_device_->local_device_id().value();
    se::DeviceAddressBase memory = mem();
    se::DeviceAddressAllocator* allocator = allocator_;
    // [A/B/C gate] XLA_ROCM_DEFER_FREE selects the deallocation strategy so a
    // single build can benchmark all three:
    //   0 = immediate host-side free (original; races under async dispatch)
    //   1 = defer via compute_stream DoHostCallback (correct, but
    //   hipLaunchHostFunc
    //       is stream-ordered so it STALLS the compute stream per free)
    //   2 = defer via ThenExecuteCallback (correct; the free runs on a separate
    //       callback stream that WaitFor()s the compute stream, so the compute
    //       stream is NOT stalled -- only a cheap event is recorded)
    // Default = 2 (cheapest correct option).
    static const int free_mode = [] {
      const char* v = std::getenv("XLA_ROCM_DEFER_FREE");
      if (v == nullptr) return 2;
      if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0) return 0;
      if (std::strcmp(v, "1") == 0) return 1;
      return 2;
    }();
    if (free_mode == 0) {
      // [Poison-on-free clincher] XLA_ROCM_POISON_FREE=<hex32> scribbles a
      // recognizable pattern into the chunk on the COMPUTE stream just before
      // returning it to BFC. Compute-stream ordering means this is strictly
      // after the buffer's last legitimate use (which was enqueued before this
      // destructor ran), so it never corrupts correct single-stream reuse.
      // It DOES corrupt the buggy path: if the freed chunk is read back or
      // reused by another stream while our kernel is still in flight, the
      // consumer observes the poison instead of silent zeros. Recommended
      // pattern 7fc00000 = f32 quiet-NaN, so the failing conv test prints NaN
      // (unmistakable use-after-free) rather than the silent 0.0.
      static const uint32_t poison = [] {
        const char* v = std::getenv("XLA_ROCM_POISON_FREE");
        if (v == nullptr || v[0] == '\0') return 0u;
        return static_cast<uint32_t>(std::strtoul(v, nullptr, 16));
      }();
      if (poison != 0 && memory.size() >= 4) {
        se::Stream* cs = local_device_->compute_stream();
        se::DeviceAddressBase scribble(memory.opaque(),
                                       memory.size() & ~uint64_t{3});
        absl::Status ps = cs->Memset32(&scribble, poison, scribble.size());
        if (!ps.ok()) {
          LOG(ERROR) << "Poison-on-free Memset32 failed: " << ps;
        }
      }
      absl::Status status = allocator->Deallocate(device_ordinal, memory);
      if (!status.ok()) {
        LOG(ERROR) << "Buffer deallocation failed: " << status;
      }
      return;
    }
    if (free_mode == 2) {
      // [ROCm conv-zero fix v5 / callback-stream deferred free] Defer the
      // actual free until the compute stream drains, so the chunk is NOT
      // returned to BFC (and then reused + zeroed by a sibling op's MIOpen
      // SetTensor(0)) while a kernel that uses this buffer is still in flight.
      // Unlike mode 1 (DoHostCallback directly on the compute stream, which
      // stalls it per free), ThenExecuteCallback runs the free on a separate
      // callback stream that WaitFor()s the compute stream, so the compute
      // stream only records a cheap event and is not stalled. The host never
      // blocks, so BFC stays asynchronous.
      absl::Status s = local_device_->ThenExecuteCallback(
          local_device_->compute_stream(),
          [allocator, device_ordinal, memory]() {
            absl::Status st = allocator->Deallocate(device_ordinal, memory);
            if (!st.ok()) {
              LOG(ERROR) << "Deferred (callback-stream) deallocation failed: "
                         << st;
            }
          },
          nullptr, "DeferredFree");
      if (s.ok()) {
        return;
      }
      // Fall back to immediate free if the callback could not be enqueued.
      absl::Status status = allocator->Deallocate(device_ordinal, memory);
      if (!status.ok()) {
        LOG(ERROR) << "Buffer deallocation failed: " << status;
      }
      return;
    }
    // free_mode == 1: defer via DoHostCallback on the compute stream.
    se::Stream* stream = local_device_->compute_stream();
    absl::Status cb_status =
        stream->DoHostCallback([allocator, device_ordinal, memory]() {
          absl::Status status = allocator->Deallocate(device_ordinal, memory);
          if (!status.ok()) {
            LOG(ERROR) << "Deferred buffer deallocation failed: " << status;
          }
        });
    if (!cb_status.ok()) {
      // Fallback: free immediately if the callback could not be enqueued.
      absl::Status status = allocator->Deallocate(device_ordinal, memory);
      if (!status.ok()) {
        LOG(ERROR) << "Buffer deallocation failed: " << status;
      }
    }
  }

  void UnsafeReleaseMemory() override { allocator_ = nullptr; }

  absl::StatusOr<BufferSequencingEventRef> GetDefinitionEvent(
      AsyncWorkRunner* async_work_runner, bool nullptr_if_past) const override {
    if (sync_point_ != std::numeric_limits<size_t>::max()) {
      return local_device_->GetEventForComputeStreamSyncPoint(
          sync_point_, async_work_runner, nullptr_if_past);
    }
    return BufferSequencingEventRef();
  }

 private:
  se::DeviceAddressAllocator* allocator_;
  LocalDeviceState* local_device_;
  size_t sync_point_ = std::numeric_limits<size_t>::max();
};

tsl::AsyncValueRef<RawSEDeviceMemory> RawSEDeviceMemory::Create(
    se::DeviceAddressBase value, LocalDeviceState* local_device,
    se::DeviceAddressAllocator* allocator) {
  return tsl::MakeAvailableAsyncValueRef<AllocatedRawSEDeviceMemory>(
      value, local_device, allocator);
}

/*static*/ void RawSEDeviceMemory::ConstructDelayed(
    tsl::AsyncValueRef<RawSEDeviceMemory> buf, se::DeviceAddressBase value,
    LocalDeviceState* local_device, se::DeviceAddressAllocator* allocator) {
  tsl::Cast<AllocatedRawSEDeviceMemory>(buf).emplace(value, local_device,
                                                     allocator);
}

/*static*/ tsl::AsyncValueRef<RawSEDeviceMemory>
RawSEDeviceMemory::CreateDelayedMemory() {
  return tsl::MakeUnconstructedAsyncValueRef<AllocatedRawSEDeviceMemory>();
}

class ForeignRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  ForeignRawSEDeviceMemory(se::DeviceAddressBase value,
                           absl::AnyInvocable<void() &&> on_delete_callback)
      : RawSEDeviceMemory(value),
        on_delete_callback_(std::move(on_delete_callback)) {}

  ~ForeignRawSEDeviceMemory() override { std::move(on_delete_callback_)(); }

  void UnsafeReleaseMemory() override {
    LOG(FATAL) << "ForeignRawSEDeviceMemory cannot be donated.";
  }

 private:
  absl::AnyInvocable<void() &&> on_delete_callback_;
};

class SlicedRawSEDeviceMemory : public RawSEDeviceMemory {
 public:
  SlicedRawSEDeviceMemory(se::DeviceAddressBase value,
                          tsl::AsyncValueRef<RawSEDeviceMemory> base)
      : RawSEDeviceMemory(value), base_(base) {}

  void UnsafeReleaseMemory() override {
    LOG(FATAL) << "SlicedRawSEDeviceMemory cannot be donated.";
  }

 private:
  tsl::AsyncValueRef<RawSEDeviceMemory> base_;
};

tsl::AsyncValueRef<RawSEDeviceMemory> RawSEDeviceMemory::CreateForeign(
    se::DeviceAddressBase value,
    absl::AnyInvocable<void() &&> on_delete_callback) {
  return tsl::MakeAvailableAsyncValueRef<ForeignRawSEDeviceMemory>(
      value, std::move(on_delete_callback));
}

tsl::AsyncValueRef<RawSEDeviceMemory> RawSEDeviceMemory::CreateSlice(
    tsl::AsyncValueRef<RawSEDeviceMemory> base, size_t offset, size_t size) {
  size_t src_size = base->mem().size();
  if (offset <= src_size && size <= src_size - offset) {
    return tsl::MakeAvailableAsyncValueRef<SlicedRawSEDeviceMemory>(
        se::DeviceAddressBase(
            reinterpret_cast<char*>(base->mem().opaque()) + offset, size),
        base);
  }
  return tsl::MakeErrorAsyncValueRef(absl::InvalidArgumentError(
      absl::StrFormat("Error when slicing: [%d,%d) in array of size %d", offset,
                      offset + size, src_size)));
}

}  // namespace xla
