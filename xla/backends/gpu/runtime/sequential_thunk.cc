/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/sequential_thunk.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"


#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {

enum {
  eProfilePrintOut = 1,
  eProfileDumpToFile = 2,
  eProfileAny = 3,
};

static int64_t GetThunksProfiling()
{
  static int64_t value = [] {
    int64_t value = 0;
    tsl::ReadInt64FromEnvVar("XLA_THUNKS_PROFILING", value, &value).IgnoreError();
    return value;
  }();
  return value;
}


/*

Need to fill in latencies:

  reduce-scatter-done.3 = (bf16[4,384]{0,1}, bf16[4,384]{0,1}, bf16[4,1152,256]{2,0,1}) async-done(reduce-scatter-start.3), metadata={scheduling_name="reduce-scatter-done.3"}
  get-tuple-element.53 = bf16[4,1152,256]{2,0,1} get-tuple-element(reduce-scatter-done.3), index=2, metadata={scheduling_name="get-tuple-element.53"}
  bitcast.5832 = bf16[4,256,1152]{1,0,2} bitcast(get-tuple-element.53), metadata={scheduling_name="bitcast.5832"}
  loop_convert_fusion.3 = bf16[1152,4,256]{2,1,0} fusion(bitcast.5832, param.28.0), kind=kLoop, calls=fused_convert.3, metadata={deduplicated_name="loop_convert_fusion.2" scheduling_name="loop_convert_fusion.3"}
  get-tuple-element.52 = bf16[4,384]{0,1} get-tuple-element(reduce-scatter-done.3), index=1, metadata={scheduling_name="get-tuple-element.52"}
  get-tuple-element.51 = bf16[4,384]{0,1} get-tuple-element(reduce-scatter-done.3), index=0, metadata={scheduling_name="get-tuple-element.51"}


ProfileGuidedLatencyEstimator::ProfileGuidedLatencyEstimator(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const tensorflow::profiler::ProfiledInstructionsProto& proto,
    std::unique_ptr<ProfileStatisticsAggregator> aggregator)
    : config_(config),
      latency_estimator_(std::move(latency_estimator)),
      aggregator_(std::move(aggregator)) {
  const int cycles_per_microsecond = latency_estimator_->CyclesPerMicrosecond();
  for (const auto& instr_cost : proto.costs()) {
    instr_map_[instr_cost.name()] =
        ProfileInfo{instr_cost.cost_us() * cycles_per_microsecond};
  }
  for (const auto& latency : proto.latencies()) {
    auto it = instr_map_.insert(std::make_pair(latency.source(), ProfileInfo{}))
                  .first;
    it->second.latencies[latency.target()] =
        latency.latency_us() * cycles_per_microsecond;
  }
}
*/

struct ProfileInstructionsCreator {
  struct Latency {
    float beg, end;
  };
  struct CostStats {
    uint64_t total;
    double sum_cost;
  };

  // skip first N points!
  static constexpr uint64_t s_num_warmup_runs = 8;

  ProfileInstructionsCreator() = default;
  ~ProfileInstructionsCreator() {
    SaveProfileProto("/data/xla/pgle_profile").IgnoreError();
  }

  void AddCost(se::Stream *stream, absl::string_view name, double usec) {
    // auto dev_id = stream->parent()->device_ordinal();
    absl::MutexLock _(&mu_);
    auto [it, added] = hlo_latency_map_.emplace(name, CostStats{});
    it->second.total++;
    if(it->second.total > s_num_warmup_runs) {
      it->second.sum_cost += usec;
    }
  }

  absl::Status SaveProfileProto(const std::string& path) {
    VLOG(0) << "Dumping profile info to file: " << path;
    tensorflow::profiler::ProfiledInstructionsProto result;
    {
      absl::MutexLock _(&mu_);
      for (const auto& [name, stats] : hlo_latency_map_) {
        int64_t diff = stats.total - s_num_warmup_runs;
        if (diff <= 0) {
          VLOG(0) << name << " was not run enough iterations!";
          continue;
        }
        auto *cost = result.add_costs();
        cost->set_name(name);
        cost->set_cost_us(stats.sum_cost / diff);
      }
    }
    auto *env = tsl::Env::Default();
    return tsl::WriteTextProto(env, absl::StrCat(path, ".pbtxt"), result);
  }

private:
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, CostStats> hlo_latency_map_ ABSL_GUARDED_BY(mu_);
};


static ProfileInstructionsCreator s_ProfileCreator;

SequentialThunk::SequentialThunk(ThunkInfo thunk_info, ThunkSequence thunks)
    : Thunk(Kind::kSequential, thunk_info), thunks_(std::move(thunks)) {}

std::string SequentialThunk::ToString(int indent) const {
  const std::string indent_str(indent * 2, ' ');
  if (thunks_.empty()) return indent_str + "No thunks.";

  auto thunk_with_longest_kind = absl::c_max_element(
      thunks_,
      [](const std::unique_ptr<Thunk>& a, const std::unique_ptr<Thunk>& b) {
        return Thunk::KindToString(a->kind()).length() <
               Thunk::KindToString(b->kind()).length();
      });
  int64_t max_thunk_kind_len =
      Thunk::KindToString(thunk_with_longest_kind->get()->kind()).length();
  std::string result;
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    // Write out the thunk kind, padded out to max_thunk_kind_len.
    absl::string_view kind_str = Thunk::KindToString(thunk->kind());
    absl::StrAppend(&result, indent_str, kind_str,
                    std::string(max_thunk_kind_len - kind_str.length(), ' '),
                    "\t");
    absl::StrAppend(&result, thunk->ToString(indent + 1));
    absl::StrAppend(&result, "\n");
  }
  return result;
}

absl::Status SequentialThunk::Prepare(
    const PrepareParams& params, ResourceRequestsInterface& resource_requests) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::Initialize(const InitializeParams& params) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

static bool IsCollectiveDone(Thunk::Kind kind) {
  switch (kind) {
    case Thunk::kNcclAllGatherDone:
    case Thunk::kNcclAllReduceDone:
    case Thunk::kNcclCollectiveBroadcastDone:
    case Thunk::kNcclCollectivePermuteDone:
    case Thunk::kNcclReduceScatterDone:
    case Thunk::kNcclAllToAllDone:
    case Thunk::kNcclSendDone:
    case Thunk::kNcclRecvDone:
    case Thunk::kNcclGroupDone:
      return true;
    default:;
  }
  return false;
}

absl::Status SequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
  std::optional<tsl::profiler::ScopedAnnotation> seq_annotation =
      GetKernelAnnotation(profile_annotation());

  int64_t eprofile = GetThunksProfiling();
  if (eprofile & eProfilePrintOut) {
    VLOG(0) << "Executing thunks: #" << thunks_.size();
    VLOG(0) << "=============================================================";
  }

  for (const std::unique_ptr<Thunk>& thunk : thunks_) {

    uint64_t start_tm = 0;
    if (eprofile & eProfileAny) {
      start_tm = tsl::Env::Default()->NowMicros();
    }

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(thunk->profile_annotation());
    if (params.mock_collectives && thunk->IsCollective()) {
      continue;
    }
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));

    if (eprofile & eProfileAny) {
      auto instr_name = thunk->profile_annotation();
      auto kind = thunk->kind();

      TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
      auto name = thunk->ToString(0);
      if (name.empty()) name = thunk->KindToString(kind);
      auto tm = tsl::Env::Default()->NowMicros() - start_tm;

      if (eprofile & eProfilePrintOut) {
        VLOG(0) << instr_name << ": " << name << " time: " << tm << " usec";
      }
      if (eprofile & eProfileDumpToFile) {
        s_ProfileCreator.AddCost(params.stream, 
                                    instr_name, (double)tm);
      }

/*
2025-04-15 15:39:33.426918: I xla/backends/gpu/runtime/sequential_thunk.cc:243] reduce-scatter.5.1: kNcclReduceScatterStart time: 145 usec
2025-04-15 15:39:33.426927: I xla/backends/gpu/runtime/sequential_thunk.cc:243] reduce-scatter.5.1: kNcclReduceScatterStart time: 155 usec
2025-04-15 15:39:33.426940: I xla/backends/gpu/runtime/sequential_thunk.cc:243] reduce-scatter.5.1: kNcclReduceScatterStart time: 128 usec
2025-04-15 15:39:33.427048: I xla/backends/gpu/runtime/sequential_thunk.cc:243] reduce-scatter-done.2: kNcclReduceScatterDone time: 123 usec
2025-04-15 15:39:33.427052: I xla/backends/gpu/runtime/sequential_thunk.cc:243] reduce-scatter-done.2: kNcclReduceScatterDone time: 155 usec

2025-04-15 16:33:04.746693: I xla/service/latency_hiding_scheduler.cc:2002] GetLatencyBetween: reduce-scatter-start.3 and reduce-scatter-done.3
2025-04-15 16:33:04.746694: I xla/service/profile_guided_latency_estimator.cc:92] PGLE found async wrapped instruction: reduce-scatter.8 in reduce-scatter-start.3
2025-04-15 16:33:04.746695: I xla/service/profile_guided_latency_estimator.cc:98] PGLE did NOT find wrapped instruction name or async start. From: reduce-scatter-start.3

  struct ProfileInfo {
    std::optional<TimeCost> cost;
    // Latencies to other instruction with this instruction as source.
    absl::flat_hash_map<std::string, TimeCost> latencies;
  };
  absl::flat_hash_map<std::string, ProfileInfo> instr_map_;
  instr_map_['reduce-scatter.8'] = ProfileInfo{std::nullopt, {'reduce-scatter-done.3', latency-val} }

*/

// suppose we hit reduce-scatter-done.2 which is a wait instruction for 'reduce-scatter.5.1' which is 
// async_wrapped_instruction() for reduce-scatter-start.2

      if (IsCollectiveDone(kind)) {
        auto pos = instr_name.find("-done"); // this must be fine!!
        auto start_name = std::string(instr_name.substr(0, pos)) + "-start";
        start_name += instr_name.substr(pos + 5);
        VLOG(0) << "start name " << start_name;
      }
    }
  }
  return absl::OkStatus();
}

void SequentialThunk::ForAllThunks(
    absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    thunk->ForAllThunks(fn);
  }
}

}  // namespace gpu
}  // namespace xla
