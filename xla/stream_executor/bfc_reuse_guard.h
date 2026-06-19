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

#ifndef XLA_STREAM_EXECUTOR_BFC_REUSE_GUARD_H_
#define XLA_STREAM_EXECUTOR_BFC_REUSE_GUARD_H_

#include <cstdint>

// ---------------------------------------------------------------------------
// [ROCm conv-zero GPU-proven use-after-free guard]  (diagnostic only)
//
// Goal: prove that the shared BFC pool re-serves a device byte range to a new
// consumer WHILE the previous owner's kernel is still in flight on the GPU --
// the exact hazard the stream-deferred free (XLA_ROCM_DEFER_FREE=2) closes.
//
// Mechanism (host-side, non-blocking):
//   * At free time (PjRt buffer destructor, which holds the compute stream) we
//     record a GPU event on the COMPUTE stream. The buffer's last use was
//     enqueued before the destructor ran, so the event completes IFF that last
//     use (and everything before it) has finished on the GPU.
//   * At alloc time (MultiDeviceAdapter::Allocate) we check whether the freshly
//     returned pointer overlaps a still-tracked freed range; if so we poll its
//     guard event. PENDING => the GPU has not drained past the previous owner's
//     last use, i.e. the BFC just handed out memory that is still live on the
//     device. That is logged as [BFC-UAF].
//
// Enabled only when XLA_ROCM_FREE_GUARD=1; otherwise every entry point is a
// cheap no-op. Only meaningful with immediate free (XLA_ROCM_DEFER_FREE=0); the
// deferred free never returns the chunk until the event would already be
// complete, so it produces no [BFC-UAF] lines -- that contrast is the proof.
// ---------------------------------------------------------------------------

namespace stream_executor {

class Stream;

class BfcReuseGuard {
 public:
  // True iff XLA_ROCM_FREE_GUARD is set to a non-zero value (cached).
  static bool Enabled();

  static BfcReuseGuard& Get();

  // Records a compute-stream guard event for the freed range [ptr, ptr+size)
  // and starts tracking it. Must be called from the PjRt buffer destructor,
  // BEFORE the chunk is returned to the allocator, with the device's compute
  // stream. No-op when disabled.
  void OnGuardedFree(int device_ordinal, const void* ptr, uint64_t size,
                     Stream* compute_stream);

  // Checks a freshly-returned allocation against tracked freed ranges and emits
  // [BFC-UAF] if it reuses one whose guard event is still pending. No-op when
  // disabled.
  void OnAlloc(int device_ordinal, const void* ptr, uint64_t size);

 private:
  BfcReuseGuard();
  ~BfcReuseGuard();
  BfcReuseGuard(const BfcReuseGuard&) = delete;
  BfcReuseGuard& operator=(const BfcReuseGuard&) = delete;

  class Impl;
  Impl* impl_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_BFC_REUSE_GUARD_H_
