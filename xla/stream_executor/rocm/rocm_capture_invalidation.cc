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

#include "xla/stream_executor/rocm/rocm_capture_invalidation.h"

#include <cstddef>
#include <memory>

#include "absl/base/no_destructor.h"
#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"

namespace stream_executor::gpu {
namespace {

constexpr size_t kSyncCallBytes = 4;

void LogIfError(hipError_t result, absl::string_view detail) {
  if (absl::Status status = ToStatus(result, detail); !status.ok()) {
    LOG(ERROR) << status;
  }
}

absl::StatusOr<bool> SyncCallInvalidatesCaptureOnThisThread(int capture_device,
                                                            int sync_device) {
  void* buffer = nullptr;
  ABSL_RETURN_IF_ERROR(ToStatus(hipSetDevice(sync_device), "hipSetDevice"));
  ABSL_RETURN_IF_ERROR(
      ToStatus(hipMalloc(&buffer, kSyncCallBytes), "hipMalloc"));
  absl::Cleanup free_buffer = [buffer] {
    LogIfError(hipFree(buffer), "hipFree");
  };

  hipStream_t stream = nullptr;
  ABSL_RETURN_IF_ERROR(ToStatus(hipSetDevice(capture_device), "hipSetDevice"));
  ABSL_RETURN_IF_ERROR(
      ToStatus(hipStreamCreateWithFlags(&stream, hipStreamDefault),
               "hipStreamCreateWithFlags"));
  absl::Cleanup destroy_stream = [stream] {
    LogIfError(hipStreamDestroy(stream), "hipStreamDestroy");
  };
  ABSL_RETURN_IF_ERROR(
      ToStatus(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
               "hipStreamBeginCapture"));

  hipError_t sync_result = hipErrorUnknown;
  std::unique_ptr<tsl::Thread> sync_thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "hip_sync_call", [&] {
        sync_result = hipSetDevice(sync_device);
        if (sync_result == hipSuccess) {
          sync_result = hipMemset(buffer, 0, kSyncCallBytes);
        }
      }));
  sync_thread.reset();

  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  hipError_t query_result = hipStreamIsCapturing(stream, &capture_status);
  hipGraph_t graph = nullptr;
  hipError_t end_result = hipStreamEndCapture(stream, &graph);
  if (graph != nullptr) {
    LogIfError(hipGraphDestroy(graph), "hipGraphDestroy");
  }
  ABSL_RETURN_IF_ERROR(ToStatus(query_result, "hipStreamIsCapturing"));

  if (capture_status == hipStreamCaptureStatusActive &&
      end_result == hipSuccess && sync_result == hipSuccess) {
    return false;
  }
  if (capture_status == hipStreamCaptureStatusInvalidated &&
      end_result == hipErrorStreamCaptureInvalidated &&
      sync_result != hipSuccess) {
    return true;
  }
  return absl::InternalError(absl::StrCat(
      "Inconsistent stream capture after a synchronous HIP call: status ",
      static_cast<int>(capture_status), ", hipStreamEndCapture ",
      hipGetErrorName(end_result), ", hipMemset ",
      hipGetErrorName(sync_result)));
}

// Returns whether a synchronous HIP call invalidates only the captures on the
// calling device.
absl::StatusOr<bool> IsCaptureInvalidationDeviceScoped(
    int device, int other_device,
    absl::FunctionRef<absl::StatusOr<bool>(int, int)>
        sync_call_invalidates_capture) {
  ABSL_ASSIGN_OR_RETURN(bool same_device_invalidated,
                        sync_call_invalidates_capture(device, device));
  if (!same_device_invalidated) {
    return absl::FailedPreconditionError(
        "A synchronous HIP call left a capture on its own device intact, so "
        "invalidation cannot be observed");
  }
  ABSL_ASSIGN_OR_RETURN(bool other_device_invalidated,
                        sync_call_invalidates_capture(device, other_device));
  return !other_device_invalidated;
}

void CreateDnnSupportUnlessDeviceScoped(
    absl::Span<StreamExecutor* const> executors,
    absl::FunctionRef<absl::StatusOr<bool>(int device, int other_device)>
        is_capture_invalidation_device_scoped) {
  if (executors.size() < 2) {
    return;
  }
  absl::StatusOr<bool> device_scoped = is_capture_invalidation_device_scoped(
      executors[0]->device_ordinal(), executors[1]->device_ordinal());
  if (device_scoped.ok() && *device_scoped) {
    VLOG(1) << "HIP invalidates stream captures per device; DNN support stays "
               "lazy";
    return;
  }
  if (device_scoped.ok()) {
    VLOG(1) << "HIP invalidates stream captures on every device; creating DNN "
               "support for "
            << executors.size() << " devices";
  } else {
    LOG(WARNING) << "Creating DNN support for " << executors.size()
                 << " devices now: " << device_scoped.status();
  }
  for (StreamExecutor* executor : executors) {
    // Not fatal: programs without convolutions must still start.
    if (executor->AsDnn() == nullptr) {
      LOG(WARNING) << "Failed to create DNN support for device "
                   << executor->device_ordinal();
    }
  }
}

}  // namespace

absl::StatusOr<bool> SyncCallInvalidatesCapture(int capture_device,
                                                int sync_device) {
  absl::StatusOr<bool> result;
  std::unique_ptr<tsl::Thread> thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "hip_capture_probe", [&] {
        result =
            SyncCallInvalidatesCaptureOnThisThread(capture_device, sync_device);
      }));
  thread.reset();
  return result;
}

void CreateDnnSupportUnlessCaptureInvalidationIsDeviceScoped(
    absl::Span<StreamExecutor* const> executors) {
  CreateDnnSupportUnlessDeviceScoped(
      executors, [](int device, int other_device) {
        // How far invalidation reaches is a property of the HIP runtime, so
        // the first answer holds for every pair of devices.
        static const absl::NoDestructor<absl::StatusOr<bool>> device_scoped(
            IsCaptureInvalidationDeviceScoped(device, other_device,
                                              SyncCallInvalidatesCapture));
        return *device_scoped;
      });
}

void CreateDnnSupportUnlessCaptureInvalidationIsDeviceScoped(
    absl::Span<StreamExecutor* const> executors,
    absl::FunctionRef<absl::StatusOr<bool>(int capture_device, int sync_device)>
        sync_call_invalidates_capture) {
  CreateDnnSupportUnlessDeviceScoped(
      executors, [&](int device, int other_device) {
        return IsCaptureInvalidationDeviceScoped(device, other_device,
                                                 sync_call_invalidates_capture);
      });
}

}  // namespace stream_executor::gpu
