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

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/roc_mori_collectives.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"  // IWYU pragma: keep
#include "xla/primitive_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

struct MoriCommState {
  std::shared_ptr<Communicator> communicator;
};

absl::StatusOr<std::shared_ptr<Communicator>> GetOrCreateCommunicator() {
  static auto* mu = new absl::Mutex;
  static auto* shared_comm = new std::shared_ptr<Communicator>;

  absl::MutexLock lock(mu);
  if (!*shared_comm) {
    auto* collectives = gpu::MoriCollectives::Default();
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Communicator> comm,
                        collectives->CreateCommunicator());
    *shared_comm = std::move(comm);
  }
  return *shared_comm;
}

absl::StatusOr<std::unique_ptr<MoriCommState>> MoriInstantiate() {
  TF_ASSIGN_OR_RETURN(auto communicator, GetOrCreateCommunicator());
  auto state = std::make_unique<MoriCommState>();
  state->communicator = std::move(communicator);
  return state;
}

absl::StatusOr<PrimitiveType> DecodeDType(int32_t dtype) {
  PrimitiveType primitive_type = static_cast<PrimitiveType>(dtype);
  if (!primitive_util::IsArrayType(primitive_type)) {
    return InvalidArgument("Unsupported Mori dtype attr value: %d", dtype);
  }
  return primitive_type;
}

absl::Status MoriSendExecute(xla::ffi::AnyBuffer recv_buffer,
                             xla::ffi::AnyBuffer send_buffer, int32_t peer,
                             int32_t dtype, size_t count, se::Stream* stream,
                             MoriCommState* state) {
  TF_RET_CHECK(peer >= 0) << "peer rank must be non-negative, got " << peer;
  TF_ASSIGN_OR_RETURN(PrimitiveType primitive_type, DecodeDType(dtype));

  auto recv = se::DeviceAddressBase(recv_buffer.untyped_data(),
                                    recv_buffer.size_bytes());
  auto send = se::DeviceAddressBase(send_buffer.untyped_data(),
                                    send_buffer.size_bytes());
  TF_RETURN_IF_ERROR(state->communicator
                         ->Send(recv, send, primitive_type, count, RankId(peer),
                                gpu::GpuCollectives::On(*stream))
                         .Await());
  return state->communicator->Quiet(gpu::GpuCollectives::On(*stream));
}

absl::Status MoriRecvExecute(xla::ffi::AnyBuffer recv_buffer,
                             xla::ffi::AnyBuffer send_buffer, int32_t peer,
                             int32_t dtype, size_t count, se::Stream* stream,
                             MoriCommState* state) {
  TF_RET_CHECK(peer >= 0) << "peer rank must be non-negative, got " << peer;
  TF_ASSIGN_OR_RETURN(PrimitiveType primitive_type, DecodeDType(dtype));

  auto recv = se::DeviceAddressBase(recv_buffer.untyped_data(),
                                    recv_buffer.size_bytes());
  auto send = se::DeviceAddressBase(send_buffer.untyped_data(),
                                    send_buffer.size_bytes());
  TF_RETURN_IF_ERROR(state->communicator
                         ->Recv(recv, send, primitive_type, count, RankId(peer),
                                gpu::GpuCollectives::On(*stream))
                         .Await());
  return state->communicator->Quiet(gpu::GpuCollectives::On(*stream));
}

XLA_FFI_DEFINE_HANDLER(kMoriInstantiate, MoriInstantiate,
                       xla::ffi::Ffi::BindInstantiate());

XLA_FFI_DEFINE_HANDLER(
    kMoriSend, MoriSendExecute,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()            // recv_buffer
        .Arg<xla::ffi::AnyBuffer>()            // send_buffer
        .Attr<int32_t>("peer")                 // rank to communicate with
        .Attr<int32_t>("dtype")                // xla::PrimitiveType enum value
        .Attr<size_t>("count")                 // element count
        .Ctx<xla::ffi::Stream>()               // se::Stream*
        .Ctx<xla::ffi::State<MoriCommState>>());

XLA_FFI_DEFINE_HANDLER(
    kMoriRecv, MoriRecvExecute,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()            // recv_buffer
        .Arg<xla::ffi::AnyBuffer>()            // send_buffer
        .Attr<int32_t>("peer")                 // rank to communicate with
        .Attr<int32_t>("dtype")                // xla::PrimitiveType enum value
        .Attr<size_t>("count")                 // element count
        .Ctx<xla::ffi::Stream>()               // se::Stream*
        .Ctx<xla::ffi::State<MoriCommState>>());

XLA_FFI_REGISTER_HANDLER(
    xla::ffi::GetXlaFfiApi(), "xla.gpu.ext.mori_send", "ROCM",
    {/*instantiate=*/kMoriInstantiate, /*prepare=*/nullptr,
     /*initialize=*/nullptr, /*execute=*/kMoriSend});

XLA_FFI_REGISTER_HANDLER(
    xla::ffi::GetXlaFfiApi(), "xla.gpu.ext.mori_recv", "ROCM",
    {/*instantiate=*/kMoriInstantiate, /*prepare=*/nullptr,
     /*initialize=*/nullptr, /*execute=*/kMoriRecv});

}  // namespace
}  // namespace xla
