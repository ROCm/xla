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

#include "xla/stream_executor/integrations/tf_allocator_adapter.h"

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/layout.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/numbers.h"

namespace stream_executor {

// ---------------------------------------------------------------------------
// [ROCm BFC reuse tracer] Diagnostic-only allocation ledger.
//
// Purpose: prove/disprove that the shared BFC pool hands the *same* device
// byte range to a new consumer right after a previous owner freed it, which is
// the precondition for the conv-zero (BFC async-dealloc reuse) race documented
// in MLSE/"Stabilizing JAX CI pytest runs", issue #4.
//
// Enabled only when XLA_ROCM_ALLOC_TRACE=1. Zero cost when unset (one relaxed
// atomic load gate). Every alloc/free that flows through MultiDeviceAdapter --
// which includes BOTH PjRt execution buffers AND the conv autotuner's
// RedzoneAllocator buffers, since they share this allocator -- is recorded.
//
// We emit a loud one-line [BFC-REUSE] record only when a fresh allocation
// returns a pointer that exactly matches (or overlaps) a range freed very
// recently, including the host time delta and the freeing vs allocating thread
// ids. A small delta + different thread is the smoking gun: the chunk went back
// to the free list and was re-served while the freeing thread's compute stream
// had not necessarily drained.
// ---------------------------------------------------------------------------
namespace {

bool AllocTraceEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("XLA_ROCM_ALLOC_TRACE");
    return v != nullptr && v[0] != '0' && v[0] != '\0';
  }();
  return enabled;
}

uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint64_t ThisThreadId() {
  return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

class BfcReuseTracer {
 public:
  static BfcReuseTracer& Get() {
    static BfcReuseTracer* tracer = new BfcReuseTracer();
    return *tracer;
  }

  void OnAlloc(int dev, const void* ptr, uint64_t size) {
    if (dev < 0 || dev >= kMaxDevices) return;
    const uint64_t seq = seq_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t now = NowNs();
    const uint64_t tid = ThisThreadId();
    const uintptr_t a = reinterpret_cast<uintptr_t>(ptr);
    absl::MutexLock lock(&mu_);
    auto& freed = freed_[dev];
    // Find the most recent free whose range overlaps [a, a+size).
    auto it = freed.upper_bound(a);  // first free-start > a
    if (it != freed.begin()) {
      --it;  // candidate free-start <= a
      const uintptr_t f_beg = it->first;
      const FreeRec& fr = it->second;
      const uintptr_t f_end = f_beg + fr.size;
      const uintptr_t a_end = a + size;
      const bool overlap = a < f_end && f_beg < a_end;
      if (overlap) {
        const double dt_us = (now - fr.t_ns) / 1000.0;
        LOG(ERROR) << absl::StrFormat(
            "[BFC-REUSE] dev=%d realloc ptr=0x%x size=%d alloc_seq=%d "
            "tid=0x%x <- reuses freed [0x%x,+%d) free_seq=%d free_tid=0x%x "
            "dt_us=%.1f exact=%d cross_thread=%d",
            dev, a, size, seq, tid, f_beg, fr.size, fr.seq, fr.tid, dt_us,
            (a == f_beg && size == fr.size) ? 1 : 0, (tid != fr.tid) ? 1 : 0);
        freed.erase(it);
      }
    }
    if (VlogLedger()) {
      LOG(ERROR) << absl::StrFormat(
          "[ALLOC] dev=%d ptr=0x%x size=%d seq=%d tid=0x%x t_ns=%d", dev, a,
          size, seq, tid, now);
    }
  }

  void OnFree(int dev, const void* ptr, uint64_t size) {
    if (dev < 0 || dev >= kMaxDevices) return;
    const uint64_t seq = seq_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t now = NowNs();
    const uint64_t tid = ThisThreadId();
    const uintptr_t a = reinterpret_cast<uintptr_t>(ptr);
    absl::MutexLock lock(&mu_);
    auto& freed = freed_[dev];
    freed[a] = FreeRec{size, seq, tid, now};
    // Bound memory: keep only the most recent kMaxFreed records per device.
    if (freed.size() > kMaxFreed) {
      // Drop the oldest by sequence: linear scan is fine, this is debug-only.
      auto oldest = freed.begin();
      for (auto i = freed.begin(); i != freed.end(); ++i) {
        if (i->second.seq < oldest->second.seq) oldest = i;
      }
      freed.erase(oldest);
    }
    if (VlogLedger()) {
      LOG(ERROR) << absl::StrFormat(
          "[FREE]  dev=%d ptr=0x%x size=%d seq=%d tid=0x%x t_ns=%d", dev, a,
          size, seq, tid, now);
    }
  }

 private:
  struct FreeRec {
    uint64_t size;
    uint64_t seq;
    uint64_t tid;
    uint64_t t_ns;
  };
  static constexpr size_t kMaxFreed = 4096;
  static constexpr int kMaxDevices = 16;
  static bool VlogLedger() {
    static const bool v = [] {
      const char* e = std::getenv("XLA_ROCM_ALLOC_TRACE");
      return e != nullptr && e[0] == '2';  // =2 dumps the full ledger.
    }();
    return v;
  }
  absl::Mutex mu_;
  std::map<uintptr_t, FreeRec> freed_[kMaxDevices];  // per device ordinal
  std::atomic<uint64_t> seq_{0};
};

}  // namespace

TfAllocatorAdapter::TfAllocatorAdapter(tsl::Allocator* wrapped, Stream* stream,
                                       size_t min_alignment,
                                       tsl::AllocationEnd allocation_end)
    : DeviceAddressAllocator(CHECK_NOTNULL(stream)->parent()->GetPlatform()),
      wrapped_(wrapped),
      stream_(stream),
      min_alignment_(min_alignment),
      allocation_end_(allocation_end) {}

TfAllocatorAdapter::TfAllocatorAdapter(tsl::Allocator* wrapped,
                                       const Platform* platform,
                                       size_t min_alignment,
                                       tsl::AllocationEnd allocation_end)
    : DeviceAddressAllocator(platform),
      wrapped_(wrapped),
      stream_(nullptr),
      min_alignment_(min_alignment),
      allocation_end_(allocation_end) {}

TfAllocatorAdapter::~TfAllocatorAdapter() {}

absl::StatusOr<ScopedDeviceAddress<uint8_t>> TfAllocatorAdapter::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  tsl::AllocationAttributes attrs;
  attrs.retry_on_failure = retry_on_failure;
  attrs.allocation_end = allocation_end_;
  void* data = nullptr;
  if (size != 0) {
    data = wrapped_->AllocateRaw(min_alignment_, size, attrs);
    if (data == nullptr) {
      return MemoryAllocationError(
          size, memory_space == xla::Layout::kHostMemorySpace);
    }
  }
  return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(data, size),
                                      device_ordinal, this);
}

absl::Status TfAllocatorAdapter::Deallocate(int device_ordinal,
                                            DeviceAddressBase mem) {
  wrapped_->DeallocateRaw(mem.opaque());
  return absl::OkStatus();
}

absl::StatusOr<Stream*> TfAllocatorAdapter::GetStream(int device_ordinal) {
  CHECK(stream_ != nullptr) << "GetStream requires a non-null stream";
  CHECK_EQ(stream_->parent()->device_ordinal(), device_ordinal);
  return stream_;
}

absl::StatusOr<tsl::Allocator*> TfAllocatorAdapter::GetAllocator(
    int device_ordinal) {
  if (stream_ && stream_->parent()->device_ordinal() != device_ordinal) {
    return absl::InternalError(
        absl::StrCat("stream_->parent()->device_ordinal() ",
                     stream_->parent()->device_ordinal(),
                     " not equal to device_ordinal ", device_ordinal));
  }
  return wrapped_;
}

//===----------------------------------------------------------------------===//
// MultiDeviceAdapter
//===----------------------------------------------------------------------===//

static int GetDeviceOrdinal(const MultiDeviceAdapter::AllocatorInfo& info) {
  return info.device_ordinal.has_value()
             ? *info.device_ordinal
             : CHECK_NOTNULL(info.stream)->parent()->device_ordinal();
}

MultiDeviceAdapter::MultiDeviceAdapter(const Platform* platform,
                                       std::vector<AllocatorInfo> allocators)
    : DeviceAddressAllocator(platform) {
  // Sort allocators by device ordinal and memory space to get user-friendly
  // logging below. It doesn't change the runtime behavior.
  absl::c_sort(allocators, [](const AllocatorInfo& a, const AllocatorInfo& b) {
    return std::make_pair(a.memory_space, GetDeviceOrdinal(a)) <
           std::make_pair(b.memory_space, GetDeviceOrdinal(b));
  });

  for (AllocatorInfo& info : allocators) {
    std::vector<std::shared_ptr<TfAllocatorAdapter>>& per_device_allocators =
        memory_space_to_per_device_allocators_[info.memory_space];
    int device_ordinal = GetDeviceOrdinal(info);
    if (per_device_allocators.size() <= device_ordinal) {
      per_device_allocators.resize(device_ordinal + 1);
    }
    CHECK(!per_device_allocators[device_ordinal]);
    if (info.stream != nullptr) {
      per_device_allocators[device_ordinal] =
          std::make_shared<TfAllocatorAdapter>(info.allocator.get(),
                                               info.stream, info.min_alignment,
                                               info.allocation_end);
    } else {
      per_device_allocators[device_ordinal] =
          std::make_shared<TfAllocatorAdapter>(
              info.allocator.get(), info.platform, info.min_alignment,
              info.allocation_end);
    }
    VLOG(3) << absl::StrFormat(
        "MultiDeviceAdapter: device_ordinal=%d memory_space=%d "
        "min_alignment=%d",
        device_ordinal, info.memory_space, info.min_alignment);
    allocators_.push_back(std::move(info.allocator));
  }
}

absl::StatusOr<ScopedDeviceAddress<uint8_t>> MultiDeviceAdapter::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  auto it = memory_space_to_per_device_allocators_.find(memory_space);
  CHECK(it != memory_space_to_per_device_allocators_.end());
  CHECK_LT(device_ordinal, it->second.size());
  ASSIGN_OR_RETURN(auto result,
                   it->second[device_ordinal]->Allocate(
                       device_ordinal, size, retry_on_failure, memory_space));

  if (AllocTraceEnabled()) {
    BfcReuseTracer::Get().OnAlloc(device_ordinal, result->opaque(),
                                  result->size());
  }

  absl::MutexLock lock(mu_);
  buffer_memory_spaces_[{device_ordinal, result->opaque()}] = memory_space;
  return result;
}

absl::StatusOr<std::shared_ptr<TfAllocatorAdapter>>
MultiDeviceAdapter::GetDefaultAllocator(int device_ordinal) {
  auto it = memory_space_to_per_device_allocators_.find(0);
  if (it == memory_space_to_per_device_allocators_.end() ||
      device_ordinal < 0 || device_ordinal >= it->second.size() ||
      !it->second[device_ordinal]) {
    return absl::InternalError(absl::StrCat(
        "No default allocator found for device ordinal ", device_ordinal));
  }
  return it->second[device_ordinal];
}

absl::Status MultiDeviceAdapter::Deallocate(int device_ordinal,
                                            DeviceAddressBase mem) {
  if (mem.opaque() == nullptr) {
    return absl::OkStatus();
  }
  if (AllocTraceEnabled()) {
    BfcReuseTracer::Get().OnFree(device_ordinal, mem.opaque(), mem.size());
  }
  int64_t memory_space;
  {
    absl::MutexLock lock(mu_);
    auto it = buffer_memory_spaces_.find({device_ordinal, mem.opaque()});
    if (it == buffer_memory_spaces_.end()) {
      // There might be situation when device memory was allocated somewhere
      // outside of the current allocator. For backward compatibility in
      // this case we are falling back to the first allocator to deallocate
      // the memory.
      // See b/325527293 for more details.
      ASSIGN_OR_RETURN(auto allocator, GetDefaultAllocator(device_ordinal));
      return allocator->Deallocate(device_ordinal, mem);
    }
    memory_space = it->second;
    buffer_memory_spaces_.erase(it);
  }

  auto it = memory_space_to_per_device_allocators_.find(memory_space);
  CHECK(it != memory_space_to_per_device_allocators_.end());
  CHECK_GE(device_ordinal, 0);
  CHECK_LT(device_ordinal, it->second.size());
  if (it->second[device_ordinal] == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "No allocator found for device ordinal %d and memory space %d",
        device_ordinal, memory_space));
  }
  return it->second[device_ordinal]->Deallocate(device_ordinal, mem);
}

absl::StatusOr<Stream*> MultiDeviceAdapter::GetStream(int device_ordinal) {
  ASSIGN_OR_RETURN(auto allocator, GetDefaultAllocator(device_ordinal));
  return allocator->GetStream(device_ordinal);
}

absl::StatusOr<tsl::Allocator*> MultiDeviceAdapter::GetAllocator(
    int device_ordinal) {
  ASSIGN_OR_RETURN(auto allocator, GetDefaultAllocator(device_ordinal));
  return allocator->GetAllocator(device_ordinal);
}

//===----------------------------------------------------------------------===//
// Error helpers
//===----------------------------------------------------------------------===//

static constexpr absl::string_view kMemoryAllocationErrorPayloadKey =
    "tf-allocator-allocation-error";

absl::Status MemoryAllocationError(uint64_t size, bool is_host_mem) {
  constexpr absl::string_view kHostMemoryExplanation =
      " Please set the environment variable "
      "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB to allocate larger "
      "host memory than the default 64 GB.";

  absl::Status status = absl::ResourceExhaustedError(
      absl::StrCat("Out of ", (is_host_mem ? "host " : ""),
                   "memory while trying to allocate ",
                   tsl::strings::HumanReadableNumBytes(size), ".",
                   (is_host_mem ? kHostMemoryExplanation : "")));
  status.SetPayload(kMemoryAllocationErrorPayloadKey, absl::Cord());
  return status;
}

bool IsMemoryAllocationError(absl::Status status) {
  return status.GetPayload(kMemoryAllocationErrorPayloadKey).has_value();
}

}  // namespace stream_executor
