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
#include "xla/tsl/util/env_var.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {

namespace {

static auto env_threshold = []() {
  auto value = std::numeric_limits< float >::infinity();
  TF_CHECK_OK(tsl::ReadFloatFromEnvVar("TF_ROCM_NAN_CHECK_MAG_THRESHOLD",
                                  /*default_val=*/value, &value));
  VLOG(0) << "NaN checker magnitude threshold " << value;
  return value;
}();

static auto env_verbose = []() {
  bool value = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_NAN_CHECK_VERBOSE",
                                  /*default_val=*/value, &value));
  return value;
}();

static auto env_check_device = []() {
  int64_t device_id = -1;
  TF_CHECK_OK(tsl::ReadInt64FromEnvVar("TF_ROCM_NAN_CHECK_DEVICE",
                                         /*default_val=*/-1, &device_id));
  return device_id;
}();

static auto env_filter = []() -> std::optional<std::regex> {
  std::string pattern;
  TF_CHECK_OK(tsl::ReadStringFromEnvVar("TF_ROCM_NAN_CHECK_FILTER",
                                          /*default_val=*/"", &pattern));
  if (pattern.empty()) {
    return std::nullopt;
  }
  return std::regex(pattern);
}();

static auto env_check_count = []() -> std::atomic<int64_t> {
  int64_t count;
  TF_CHECK_OK(tsl::ReadInt64FromEnvVar("TF_ROCM_NAN_CHECK_COUNT",
                                         /*default_val=*/1, &count));
  return count;
}();

static absl::Mutex nan_signal_map_mutex;
static absl::flat_hash_map<se::Stream*, se::DeviceMemory<uint32_t>>
      nan_signal_map;

}  // namespace

NanCheckThunk::NanCheckThunk(ThunkInfo thunk_info, 
                          const HloInstruction::InstructionVector& operands,
                          std::vector<BufferAllocation::Slice>&& buffers)
    : Thunk(Kind::kNanCheck, thunk_info),
      operands_(operands),
      buffers_(std::move(buffers)) {}

absl::Status NanCheckThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers(buffers_.size());
  bool oo = stream->parent()->device_ordinal()==0;

  auto source = operands_[0];
  if (source->opcode() == HloOpcode::kGetTupleElement) {
    source = source->mutable_operand(0);
  }

  //if (oo) VLOG(0) << source->ToString() << " total bufs " << buffers_.size();
  for (size_t i = 0; i < buffers_.size(); i++) {
    buffers[i] = allocs.GetDeviceAddress(buffers_[i]);
    //if (oo) VLOG(0) << i << " buf size: " << buffers[i].size();
  }

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

  auto element_type = operands_[0]->shape().element_type();
  TF_RETURN_IF_ERROR(gpu::LaunchNanCheckKernel(
        stream, buffers[0], element_type, env_threshold, env_verbose, nan_signal));

  return Postprocess(stream, std::move(buffers), 
           *reinterpret_cast<std::atomic<uint32_t>*>(nan_signal.opaque()));
}

absl::Status NanCheckThunk::Postprocess(se::Stream* stream, 
    absl::InlinedVector<se::DeviceMemoryBase, 4>&& buffers,
    std::atomic<uint32_t>& nan_signal) {

  auto sigval = static_cast< NaNCheckerResult >(
              nan_signal.load(std::memory_order_relaxed));
  run_no_++; // always increment run_no to distinguish different invocations
  if (TF_PREDICT_TRUE(sigval == NaNCheckerResult::OK)) return absl::OkStatus();

  fflush(stdout);
  nan_signal.store(0, std::memory_order_relaxed);

  auto device_ordinal = stream->parent()->device_ordinal();
  if (env_check_device >= 0 && env_check_device != device_ordinal) {
    return absl::OkStatus();
  }
  auto print_options = HloPrintOptions::ShortParsable()
                           .set_print_operand_shape(true)
                           .set_print_extra_attributes(true);

  HloInstruction* source = operands_[0];
  int64_t index = 0;
  if (source->opcode() == HloOpcode::kGetTupleElement) {
    index = source->tuple_index();
    source = source->mutable_operand(0);
  }

  auto instr_str = source->ToString(print_options);
  absl::string_view what = sigval == NaNCheckerResult::NaN ?
            ": NaN in result " : sigval == NaNCheckerResult::Inf ?
            ": Inf in result " : ": large mag in result ";

  auto msg = absl::StrCat(source->parent()->parent()->name(), "/",
                  source->parent()->name(), what, index, " of ", instr_str);
  
  auto dump_snapshot = [&]() -> absl::Status {
    auto* env = tsl::Env::Default();
    std::vector<std::string> tempdirs;
    env->GetLocalTempDirectories(&tempdirs);
    if (tempdirs.empty()) {
      return absl::InternalError("Env TMPDIR/TEST_TMPDIR must be set to enable Hlo snapshots!");
    }

    LOG(INFO) << "Dumping snapshot for " << source->name();
    std::unique_ptr<HloModule> module;
    if (operands_[0] == source) {
      module = ExtractInstructionIntoNewModule(*source);
    } else {
      module = ExtractProducerConsumerIntoNewModule(*source, *operands_[0]);
    }

    HloSnapshot snapshot;
    snapshot.set_execution_platform("gpu");
    *snapshot.mutable_hlo()->mutable_hlo_module() = module->ToProto();

    std::vector<Literal> literals;
    literals.reserve(operands_.size());

    int buf_idx = 0;
    for (auto op : operands_) {
      if (op->opcode() == HloOpcode::kConstant) {
        literals.emplace_back(op->literal().Clone());
        continue;
      }
      auto& L = literals.emplace_back(op->shape());
      TF_RETURN_IF_ERROR(stream->Memcpy(L.untyped_data(), buffers[buf_idx++],
                        L.size_bytes()));
    }
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    for (uint32_t i = 0; i < literals.size(); i++) {
      if (i == 0) {
        *snapshot.mutable_result() = literals[i].ToProto();
      } else {
        *snapshot.add_arguments() = literals[i].ToProto();
      }
    }

    auto filename = tsl::io::JoinPath(tempdirs[0], 
          absl::StrFormat("%s-dev%d-%d.hlo.pb", module->name(),
                device_ordinal, run_no_));
    std::string pb;
    if (!tsl::SerializeToStringDeterministic(snapshot, &pb)) {
      return absl::InternalError("Failed to serialize hlo snapshoot");
    }
    TF_RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), filename, pb));
    LOG(ERROR) << "Saved snapshoot to " << filename;
    return absl::OkStatus();
  };

  int64_t count = env_check_count.fetch_sub(1);
  LOG(INFO) << "COUNT " << count;
  bool do_abort = count <= 1;

  LOG(ERROR) << msg << " on GPU " << device_ordinal;
  if (!env_filter.has_value() || std::regex_search(instr_str, *env_filter)) {
    if (auto s = dump_snapshot(); !s.ok()) {
      LOG(ERROR) << "Failed to save hlo snapshot: " << s.message();
    }
  }
  if (do_abort) {
    std::exit(1);
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
