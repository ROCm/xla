/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/nan_check_thunk.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <regex>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/nan_check.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace {

static absl::Status ExecuteOnSeparateThread(
    absl::AnyInvocable<absl::Status() &&> callback) {
  absl::Status status;
  {
    tsl::thread::ThreadPool one_shot_pool(tsl::Env::Default(),
                                          tsl::ThreadOptions(), "one_shot", 1);

    one_shot_pool.Schedule(
        [&status, &callback]() { status = std::move(callback)(); });
  }
  return status;
}
}  // namespace

namespace xla {
namespace gpu {

NanCheckThunk::NanCheckThunk(ThunkInfo thunk_info, HloInstruction* instruction,
                             std::vector<BufferAllocation::Slice>&& buffers)
    : Thunk(Kind::kNanCheck, thunk_info),
      instruction_(instruction),
      buffers_(std::move(buffers)) {}

absl::Status NanCheckThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers(buffers_.size());
  for (size_t i = 0; i < buffers_.size(); i++) {
    buffers[i] = allocs.GetDeviceAddress(buffers_[i]);
  }

  static absl::Mutex nan_signal_map_mutex;
  static absl::flat_hash_map<se::Stream*, se::DeviceMemory<uint32_t>>
      nan_signal_map;

  se::DeviceMemory<uint32_t> nan_signal;
  {
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

  TF_RETURN_IF_ERROR(nan_checker_(stream, buffers[0], nan_signal));

  return stream->DoHostCallback(
      [this, _stream = stream,
       &_nan_signal =
           *reinterpret_cast<std::atomic<uint32_t>*>(nan_signal.opaque()),
       _buffers = std::move(buffers)]() {
        Postprocess(_stream, _nan_signal, _buffers);
      });
}

void NanCheckThunk::Postprocess(
    se::Stream* stream, std::atomic<uint32_t>& nan_signal,
    const absl::InlinedVector<se::DeviceMemoryBase, 4>& buffers) {
  if (TF_PREDICT_TRUE(nan_signal.load(std::memory_order_relaxed) == 0)) {
    return;
  }

  nan_signal.store(0, std::memory_order_relaxed);

  static auto filter = []() -> std::optional<std::regex> {
    const char* pattern = std::getenv("TF_ROCM_NAN_CHECK_FILTER");
    if (pattern == nullptr || std::strlen(pattern) == 0) {
      return std::nullopt;
    }
    return std::regex(pattern);
  }();

  auto print_options = HloPrintOptions::ShortParsable()
                           .set_print_operand_shape(true)
                           .set_print_extra_attributes(true);

  HloInstruction* source = instruction_;
  int64_t index = 0;
  if (source->opcode() == HloOpcode::kGetTupleElement) {
    index = source->tuple_index();
    source = source->mutable_operand(0);
  }

  auto msg = absl::StrFormat("Computation %s/%s: NaN found in result %d of %s",
                             source->parent()->parent()->name(),
                             source->parent()->name(), index,
                             source->ToString(print_options));

  if (filter && std::regex_search(msg, *filter)) {
    return;
  }

  auto dump_snapshoot = [&]() -> absl::Status {
    LOG(INFO) << "Dumping snapshoot for " << source->name();
    std::unique_ptr<HloModule> module;
    if (instruction_ == source) {
      module = ExtractInstructionIntoNewModule(*source);
    } else {
      module = ExtractProducerConsumerIntoNewModule(*source, *instruction_);
    }

    HloSnapshot snapshot;
    snapshot.set_execution_platform("gpu");
    *snapshot.mutable_hlo()->mutable_hlo_module() = module->ToProto();

    Literal output_literal(instruction_->shape());
    std::vector<Literal> input_literals;
    input_literals.reserve(source->operands().size());

    TF_RETURN_IF_ERROR(ExecuteOnSeparateThread([&]() {
      TF_ASSIGN_OR_RETURN(auto transfer_stream_owned,
                          stream->parent()->CreateStream());

      // TODO(rocm): Why does the destructor hang? Leak it for now;
      auto transfer_stream = transfer_stream_owned.release();

      TF_RETURN_IF_ERROR(transfer_stream->Memcpy(output_literal.untyped_data(),
                                                 buffers[0],
                                                 output_literal.size_bytes()));

      size_t buffer_index = 1;
      for (auto operand : source->operands()) {
        if (!operand->shape().IsArray()) {
          return absl::InternalError(
              "Cannot take a snapshoot with non array input");
        }

        if (operand->opcode() == HloOpcode::kConstant) {
          input_literals.emplace_back(operand->literal().Clone());
        } else {
          input_literals.emplace_back(operand->shape());
          if (buffer_index == buffers.size()) {
            return absl::InternalError("Not enough data to take a snapshoot");
          }
          TF_RETURN_IF_ERROR(transfer_stream->Memcpy(
              input_literals.back().untyped_data(), buffers[buffer_index++],
              input_literals.back().size_bytes()));
        }

        TF_RETURN_IF_ERROR(transfer_stream->BlockHostUntilDone());
      }
    }));

    *snapshot.mutable_result() = output_literal.ToProto();

    for (const auto& literal : input_literals) {
      *snapshot.add_arguments() = literal.ToProto();
    }

    auto filename = absl::StrFormat("%s.%d.hlo_snapshot.pb", module->name(),
                                    tsl::Env::Default()->NowMicros());
    std::string pb;
    if (!tsl::SerializeToStringDeterministic(snapshot, &pb)) {
      return absl::InternalError("Failed to serialize hlo snapshoot");
    }
    TF_RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), filename, pb));
    LOG(ERROR) << "Saved snapshoot to " << filename;
    return absl::OkStatus();
  };

  auto status = dump_snapshoot();

  if (!status.ok()) {
    LOG(ERROR) << "Failed to save hlo snapshoot: " << status.message();
  }

  LOG(FATAL) << msg << " on GPU " << stream->parent()->device_ordinal();
}

absl::Status NanCheckThunk::Initialize(const InitializeParams& params) {
  nan_checker_ = [element_type = instruction_->shape().element_type()](
                     se::Stream* stream, se::DeviceMemoryBase buffer,
                     se::DeviceMemory<uint32_t> nan_signal) -> absl::Status {
    return gpu::LaunchNanCheckKernel(stream, buffer, element_type, nan_signal);
  };

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
