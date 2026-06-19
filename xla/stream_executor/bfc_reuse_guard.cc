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

#include "xla/stream_executor/bfc_reuse_guard.h"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor {
namespace {

uint64_t ThisThreadId() {
  return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

}  // namespace

class BfcReuseGuard::Impl {
 public:
  void OnGuardedFree(int dev, const void* ptr, uint64_t size,
                     Stream* compute_stream) {
    if (dev < 0 || dev >= kMaxDevices || compute_stream == nullptr ||
        ptr == nullptr) {
      return;
    }
    StreamExecutor* executor = compute_stream->parent();
    absl::MutexLock lock(&mu_);
    std::unique_ptr<Event> ev = AcquireEvent(executor);
    if (ev == nullptr) {
      return;  // best-effort: skip guarding this free.
    }
    // Record on the COMPUTE stream: completes only after the buffer's last use
    // (already enqueued before this destructor ran) finishes on the GPU.
    if (!compute_stream->RecordEvent(ev.get()).ok()) {
      ReleaseEvent(std::move(ev));
      return;
    }
    auto& active = active_[dev];
    if (active.size() >= kMaxActive) {
      auto oldest = active.begin();
      for (auto it = active.begin(); it != active.end(); ++it) {
        if (it->second.seq < oldest->second.seq) oldest = it;
      }
      ReleaseEvent(std::move(oldest->second.ev));
      active.erase(oldest);
    }
    const uint64_t seq = seq_++;
    active[reinterpret_cast<uintptr_t>(ptr)] =
        Rec{size, seq, ThisThreadId(), std::move(ev)};
  }

  void OnAlloc(int dev, const void* ptr, uint64_t size) {
    if (dev < 0 || dev >= kMaxDevices || ptr == nullptr) {
      return;
    }
    const uintptr_t a = reinterpret_cast<uintptr_t>(ptr);
    absl::MutexLock lock(&mu_);
    auto& active = active_[dev];
    auto it = active.upper_bound(a);
    if (it == active.begin()) {
      return;
    }
    --it;
    const uintptr_t f_beg = it->first;
    Rec& r = it->second;
    const uintptr_t f_end = f_beg + r.size;
    const uintptr_t a_end = a + size;
    if (!(a < f_end && f_beg < a_end)) {
      return;  // no overlap with the nearest freed range.
    }
    const Event::Status st = r.ev->PollForStatus();
    if (st == Event::Status::kPending) {
      LOG(ERROR) << absl::StrFormat(
          "[BFC-UAF] dev=%d realloc ptr=0x%x size=%d reuses freed [0x%x,+%d) "
          "free_seq=%d free_tid=0x%x while its COMPUTE-STREAM guard event is "
          "STILL PENDING -> the previous owner's kernel is in flight on the "
          "GPU. Immediate free re-served live device memory; stream-deferred "
          "free (XLA_ROCM_DEFER_FREE=2) would have withheld this chunk.",
          dev, a, size, f_beg, r.size, r.seq, r.tid);
    }
    ReleaseEvent(std::move(r.ev));
    active.erase(it);
  }

 private:
  struct Rec {
    uint64_t size;
    uint64_t seq;
    uint64_t tid;
    std::unique_ptr<Event> ev;
  };
  static constexpr int kMaxDevices = 16;
  static constexpr size_t kMaxActive = 4096;
  static constexpr size_t kMaxPool = 4096;

  std::unique_ptr<Event> AcquireEvent(StreamExecutor* executor) {
    if (!pool_.empty()) {
      std::unique_ptr<Event> e = std::move(pool_.back());
      pool_.pop_back();
      return e;
    }
    absl::StatusOr<std::unique_ptr<Event>> e = executor->CreateEvent();
    if (!e.ok()) return nullptr;
    return std::move(*e);
  }
  void ReleaseEvent(std::unique_ptr<Event> e) {
    if (e != nullptr && pool_.size() < kMaxPool) {
      pool_.push_back(std::move(e));
    }
  }

  absl::Mutex mu_;
  std::map<uintptr_t, Rec> active_[kMaxDevices];
  std::vector<std::unique_ptr<Event>> pool_;
  uint64_t seq_ = 0;
};

bool BfcReuseGuard::Enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("XLA_ROCM_FREE_GUARD");
    return v != nullptr && v[0] != '0' && v[0] != '\0';
  }();
  return enabled;
}

BfcReuseGuard& BfcReuseGuard::Get() {
  static BfcReuseGuard* guard = new BfcReuseGuard();
  return *guard;
}

BfcReuseGuard::BfcReuseGuard() : impl_(new Impl()) {}
BfcReuseGuard::~BfcReuseGuard() { delete impl_; }

void BfcReuseGuard::OnGuardedFree(int device_ordinal, const void* ptr,
                                  uint64_t size, Stream* compute_stream) {
  if (!Enabled()) return;
  impl_->OnGuardedFree(device_ordinal, ptr, size, compute_stream);
}

void BfcReuseGuard::OnAlloc(int device_ordinal, const void* ptr,
                            uint64_t size) {
  if (!Enabled()) return;
  impl_->OnAlloc(device_ordinal, ptr, size);
}

}  // namespace stream_executor
