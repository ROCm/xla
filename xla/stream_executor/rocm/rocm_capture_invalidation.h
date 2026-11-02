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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_CAPTURE_INVALIDATION_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_CAPTURE_INVALIDATION_H_

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// Returns whether a synchronous HIP call (hipMemset) that one thread makes on
// `sync_device` invalidates a stream capture that another thread has in
// progress on `capture_device`. Runs on threads of its own, so the caller's
// current device and last HIP error are left alone, and ends the capture it
// starts.
absl::StatusOr<bool> SyncCallInvalidatesCapture(int capture_device,
                                                int sync_device);

// Creating a device's DNN support makes synchronous HIP calls. Up to
// ROCm/rocm-systems#12274, HIP invalidates every stream capture in the process
// on such a call, not only the captures on the calling device, so DNN support
// created lazily while another device records a command buffer breaks that
// recording.
//
// If `executors` spans more than one device, checks once per process, with
// SyncCallInvalidatesCapture, that a call on one device invalidates a capture
// on that device and not one on another device. Unless both hold, creates the
// DNN support of every executor now, one at a time. Call it only after every
// executor's Init() has returned: MIOpen's first handle loads a library whose
// HIP code objects take comgr's lock under the dynamic loader's lock, and a
// device still in Init() can hold comgr's lock while waiting for the loader's.
void CreateDnnSupportUnlessCaptureInvalidationIsDeviceScoped(
    absl::Span<StreamExecutor* const> executors);

// As above, but asks `sync_call_invalidates_capture`, on every call.
void CreateDnnSupportUnlessCaptureInvalidationIsDeviceScoped(
    absl::Span<StreamExecutor* const> executors,
    absl::FunctionRef<absl::StatusOr<bool>(int capture_device, int sync_device)>
        sync_call_invalidates_capture);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_CAPTURE_INVALIDATION_H_
