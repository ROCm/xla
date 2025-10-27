/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime_intrinsics.h"

#include <cstdint>
#include <string>
#include <regex>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/nan_check.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_finder.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"


namespace xla {

namespace {

std::string GetGpuPlatformName() {
  return absl::AsciiStrToUpper(
      PlatformUtil::CanonicalPlatformName("gpu").value());
}

absl::Status AssertOnGpu(void* stream_handle, void* buffer,
                         absl::string_view error_msg) {
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithName(GetGpuPlatformName()));
  TF_ASSIGN_OR_RETURN(se::Stream * stream,
                      stream_executor::FindStream(platform, stream_handle));
  if (!stream) {
    return Internal("Stream not found for: %p", stream_handle);
  }

  int8_t expected = false;
  int64_t byte_size = sizeof(int8_t);
  CHECK_EQ(byte_size, ShapeUtil::ByteSizeOfPrimitiveType(PrimitiveType::PRED));
  TF_RETURN_IF_ERROR(stream->Memcpy(
      &expected, se::DeviceMemoryBase{buffer, static_cast<uint64_t>(byte_size)},
      byte_size));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  if (!static_cast<bool>(expected)) {
    return Internal("%s", error_msg);
  }

  return absl::OkStatus();
}

void AssertionCustomCall(void* stream_handle, void** buffers,
                         const char* opaque, int opaque_len,
                         XlaCustomCallStatus* status) {
  absl::Status s =
      AssertOnGpu(stream_handle, buffers[0],
                  absl::string_view{opaque, static_cast<uint64_t>(opaque_len)});
  if (!s.ok()) {
    auto msg = s.message();
    XlaCustomCallStatusSetFailure(status, msg.data(), msg.size());
  }
}

void NopReturnTokenCustomCall(void* stream_handle, void** buffers,
                              const char* opaque, int opaque_len,
                              XlaCustomCallStatus* status) {
  VLOG(1) << "NopReturnTokenCustomCall called.";
}

absl::Status NanCheckCustomCall(
    se::Stream* stream, ffi::AnyBuffer buffer,
    xla::ffi::Result<xla::ffi::Buffer<xla::TOKEN>> res,
    std::string_view msg) {

  static absl::Mutex nan_signal_map_mutex;
  static absl::flat_hash_map<se::Stream*, se::DeviceMemory<uint32_t>> nan_signal_map;
  static std::atomic<const se::MemoryAllocation*> abort_lock;

  se::DeviceMemory<uint32_t> nan_signal;
  {
    // TODO(rocm) Move this into Prepare to make it command buffer safe
    absl::MutexLock lock(&nan_signal_map_mutex);
    auto it = nan_signal_map.find(stream);
    if (TF_PREDICT_FALSE(it == nan_signal_map.end())) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<se::MemoryAllocation> signal_buffer,
          stream->parent()->HostMemoryAllocate(sizeof(uint32_t)));
      nan_signal = se::DeviceMemory<uint32_t>::MakeFromByteSize(
          signal_buffer->opaque(), sizeof(uint32_t));
      TF_RETURN_IF_ERROR(stream->MemZero(&nan_signal, sizeof(uint32_t)));
      nan_signal_map.emplace(stream, nan_signal);
      signal_buffer.release();  // TODO(rocm) Leak it for now
    } else {
      nan_signal = it->second;
    }
  }

  TF_RETURN_IF_ERROR(gpu::LaunchNanCheckKernel(
      stream, buffer.device_memory(), buffer.element_type(), nan_signal));

  return stream->DoHostCallback(
      [_device_ordinal = stream->parent()->device_ordinal(),
       _msg = std::string(msg),  // TODO(rocm): Do we need to make a defensive copy here
       &_nan_signal =
           *reinterpret_cast<std::atomic<uint32_t>*>(nan_signal.opaque())]() {
        if (TF_PREDICT_FALSE(_nan_signal.load(std::memory_order_relaxed) != 0)) {
          _nan_signal.store(0, std::memory_order_relaxed);
          static auto filter = []() -> std::optional<std::regex> {
            const char* pattern = std::getenv("TF_ROCM_NAN_CHECK_FILTER");
            if (pattern == nullptr || std::strlen(pattern) == 0) {
              return std::nullopt;
            }
            return std::regex(pattern);
          }();
          if (!filter || !std::regex_search(_msg, *filter)) {
            LOG(FATAL) << _msg << " on GPU " << _device_ordinal;;
          }
        }
      });
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kXlaGpuNanCheckCustomCall, NanCheckCustomCall,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<xla::ffi::Buffer<xla::TOKEN>>()
                           .Attr<absl::string_view>("msg"));
                          /* {xla::ffi::Traits::kCmdBufferCompatible} */

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kXlaGpuNanCheckCustomCallTag,
                         GetGpuPlatformName(), kXlaGpuNanCheckCustomCall);



XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    std::string(kXlaGpuAssertCustomCallTag), AssertionCustomCall,
    GetGpuPlatformName());

// This allows measuring exported HLOs where kOutfeed and kSendDone has been
// replaced with NopReturnToken. In that case the runtime of the original
// kOutfeed and kSendDone operations is not measured.
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    std::string(kNopReturnTokenCustomCallTarget), NopReturnTokenCustomCall,
    GetGpuPlatformName());

}  // namespace xla
