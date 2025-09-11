/* Copyright 2023 The OpenXLA Authors.

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

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"

#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;
using ::testing::NotNull;

// Makes a DeviceAssignment device#i to replica_id #i.
DeviceAssignment MakeDeviceAssn(int64_t num_replicas) {
  DeviceAssignment assn(/*replica_count=*/num_replicas,
                        /*computation_count=*/1);
  for (int64_t i = 0; i < num_replicas; ++i) {
    assn(i, 0) = i;
  }
  return assn;
}

class CollectiveOpsTestE2E : public HloTestBase {
 public:
  CollectiveOpsTestE2E() {
    replacements_[kF8E4M3DatatypePlaceholder] =
        IsCuda() ? "f8e4m3fn" : "f8e4m3fnuz";
    replacements_[kF8E5M2DatatypePlaceholder] =
        IsCuda() ? "f8e5m2" : "f8e5m2fnuz";
  }

  bool IsCuda() {
    return std::holds_alternative<se::CudaComputeCapability>(Capability());
  }

  const se::GpuComputeCapability& Capability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }

  bool HasFp8Support() {
    if (IsCuda()) {
      return std::get<se::CudaComputeCapability>(Capability()).IsAtLeast(8, 9);
    }
    return std::get<se::RocmComputeCapability>(Capability())
               .has_fp8_support() &&
           GetDebugOptionsForTest().xla_gpu_enable_cublaslt();
  }

  void CollectiveOpsVerifyF8Matmul(absl::string_view hlo_text,
                                   const DebugOptions& options) {
    if (!HasFp8Support()) {
      return;
    }
    const int64_t kNumReplicas = 1;
    const int64_t kNumPartitions = 4;

    HloModuleConfig config =
        GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
    config.set_debug_options(options);
    config.set_num_partitions(kNumPartitions);
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_text, config));

    TF_ASSERT_OK_AND_ASSIGN(auto executable,
                            CreateExecutable(std::move(module),
                                             /*run_hlo_passes=*/true));
    EXPECT_TRUE(executable->has_module());
    std::vector<HloInstruction*> gemm_ops =
        FindInstructions(&executable->module(), HloOpcode::kCustomCall);
    for (HloInstruction* gemm_op : gemm_ops) {
      EXPECT_EQ(gemm_op->custom_call_target(), "__cublas$lt$matmul$f8");
    }
  }

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(Executable* executable,
                                                         int64_t num_replicas) {
    DeviceAssignment device_assignment = MakeDeviceAssn(num_replicas);
    return HloTestBase::ExecuteReplicated(
        /*executable_provider*/ [&](int64_t) { return executable; },
        /*argument_count_provider*/ [](int64_t) { return 0; },
        /*argument_provider*/ [](int64_t, int64_t) { return nullptr; },
        num_replicas, /*run_hlo_passes=*/false, &device_assignment);
  }

  bool IsAsync(const HloInstruction* inst) {
    return !inst->backend_config<gpu::GpuBackendConfig>()
                .value()
                .collective_backend_config()
                .is_sync();
  }

 protected:
  absl::flat_hash_map<absl::string_view, absl::string_view> replacements_;

 private:
  static constexpr const char* kF8E4M3DatatypePlaceholder{"<<F8E4M3>>"};
  static constexpr const char* kF8E5M2DatatypePlaceholder{"<<F8E5M2>>"};
};

// E2E tests for collective ops. These will generally verify some HLO transform
// for collectives (for example, sync -> async conversion) and correct
// execution of the transformed HLO.

// E2E test for collectives with flags set. Has constructor arguments specifying
// whether to enable/disable async collectives, and to set the memcpy_local_p2p
// flag. Subclasses pass in constructor arguments based on GetParam().
class CollectiveOpsWithFlagsBase : public CollectiveOpsTestE2E {
 public:
  CollectiveOpsWithFlagsBase(bool enable_async, bool enable_p2p_memcpy)
      : enable_async_(enable_async),
        enable_p2p_memcpy_(enable_p2p_memcpy),
        num_devices_(backend().device_count()) {
    VLOG(1) << "Running with " << num_devices_ << " devices";
  }

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();

    // Disable autotuning which is unnecessary.
    debug_options.set_xla_gpu_autotune_level(0);

    // Enable or disable all async collectives based on test parameter.
    if (!enable_async_) {
      for (auto option :
           {DebugOptions::NOOP, DebugOptions::ALLREDUCE,
            DebugOptions::ALLGATHER, DebugOptions::REDUCESCATTER,
            DebugOptions::COLLECTIVEBROADCAST, DebugOptions::ALLTOALL,
            DebugOptions::COLLECTIVEPERMUTE, DebugOptions::RAGGEDALLTOALL}) {
        debug_options.add_xla_gpu_disable_async_collectives(option);
      }
    }
    debug_options.add_xla_disable_hlo_passes(
        "gpu-convert-async-collectives-to-sync");
    if (enable_p2p_memcpy_) {
      debug_options.set_xla_gpu_use_memcpy_local_p2p(true);
    }
    return debug_options;
  }

  absl::StatusOr<std::unique_ptr<Executable>> CreateExecutable(
      absl::string_view hlo_string, int64_t num_replicas) {
    HloModuleConfig config =
        GetModuleConfigForTest(/*replica_count=*/num_replicas);

    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_string, config));
    return CreateExecutable(std::move(module),
                            /*run_hlo_passes=*/true);
  }
  using CollectiveOpsTestE2E::CreateExecutable;
  const bool enable_async_;
  const bool enable_p2p_memcpy_;
  const int64_t num_devices_;
};

class AllReduceTest
    : public CollectiveOpsWithFlagsBase,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {
 public:
  struct InputsOutputs {
    std::vector<Literal> inputs;
    std::vector<Literal> expected_outputs;

    [[nodiscard]] std::vector<std::vector<Literal*>> InputLiteralPtrs() {
      std::vector<std::vector<Literal*>> result;
      for (auto& input : inputs) {
        result.push_back(std::vector<Literal*>{&input});
      }
      return result;
    }
  };

  AllReduceTest()
      : CollectiveOpsWithFlagsBase(std::get<0>(GetParam()),
                                   /*enable_p2p_memcpy=*/false) {}

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();

    // opts.set_xla_gpu_unsupported_use_all_reduce_one_shot_kernel(
    //     std::get<1>(GetParam()));

    return opts;
  }

  static absl::StatusOr<InputsOutputs> BuildTestInputsOutputs(
      HloModule& module, int64_t num_replicas, int64_t num_iterations) {
    std::vector<Array<float>> inputs;
    std::vector<Literal> input_literals;
    const int64_t num_elements =
        module.entry_computation()->root_instruction()->shape().dimensions()[0];
    for (int i = 0; i < num_replicas; ++i) {
      auto& input = inputs.emplace_back(Array<float>({num_elements}));
      input.FillRandom(1.0f, 10.0f, /*seed=*/i);
      input_literals.push_back(LiteralUtil::CreateFromArray(input));
    }
    std::vector<Array<float>> expected_outputs(num_replicas,
                                               Array<float>({num_elements}));
    std::vector<Literal> expected_output_literals;
    const HloInstruction* const instr =
        FindInstruction(&module, HloOpcode::kAllReduce);
    if (instr == nullptr) {
      return absl::InvalidArgumentError(
          "Instruction 'all-reduce' not found in module.");
    }
    const std::vector<ReplicaGroup>& replica_groups =
        instr->device_list().replica_groups();
    // Map each device to set of replica groups it belongs to.
    std::vector<std::vector<int64_t>> device_to_groups(num_replicas);
    for (const auto& replica_group : replica_groups) {
      const auto& replica_ids = replica_group.replica_ids();
      for (int64_t replica : replica_group.replica_ids()) {
        CHECK_EQ(device_to_groups[replica].size(), 0);
        device_to_groups[replica].assign(replica_ids.begin(),
                                         replica_ids.end());
      }
    }
    // Sum inputs from each replica group
    for (int i = 0; i < num_replicas; ++i) {
      expected_outputs[i].Each(
          [&](absl::Span<const int64_t> indices, float* val) {
            for (const int64_t replica : device_to_groups[i]) {
              *val += inputs[replica](indices);
            }
            // Each iteration after the first,the output is doubled.
            *val *= std::pow(device_to_groups[i].size(), num_iterations - 1);
          });
    }
    for (auto& expected_output : expected_outputs) {
      expected_output_literals.push_back(
          LiteralUtil::CreateFromArray(expected_output));
    }
    return InputsOutputs{std::move(input_literals),
                         std::move(expected_output_literals)};
  }
};

TEST_P(AllReduceTest, AsyncAllReduceInsideWhile_F32_2GPUs) {
  const int64_t kNumElements = 32;
  const int64_t kNumIterations = 3;
  const absl::string_view kReplicaGroups = "{0,1}";
  const auto kModuleStr = absl::StrFormat(
      R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  while_condition {
    limit = s32[] constant(%1$d)
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    ROOT result = pred[] compare(loop_counter, limit), direction=LT
  }

  while_body {
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    tensor = get-tuple-element(params), index=1
    out0 = f32[%2$d] all-reduce(tensor), to_apply=apply_op,
      replica_groups={%3$s}
    new_loop_counter = s32[] add(loop_counter, s32[] constant(1))
    ROOT result = (s32[], f32[%2$d]{0}) tuple(new_loop_counter, out0)
  }

  ENTRY test_computation {
    param_0 = f32[%2$d] parameter(0)
    while_init = (s32[], f32[%2$d]{0}) tuple(s32[] constant(0), param_0)
    while_result = (s32[], f32[%2$d]{0})
      while(while_init), condition=while_condition, body=while_body
    ROOT result = get-tuple-element(while_result), index=1
  }
  )",
      kNumIterations, kNumElements, kReplicaGroups);

  const int64_t kNumReplicas = 2;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  int64_t num_elements =
      module->entry_computation()->root_instruction()->shape().dimensions()[0];

  Array<float> input1({num_elements}), input2({num_elements});
  input1.FillRandom(1.0f, 10.0f, /*seed=*/0);
  input2.FillRandom(1.0f, 10.0f, /*seed=*/1);
  Array<float> expected_output({num_elements});
  expected_output.Each([&](absl::Span<const int64_t> indices, float* val) {
    *val =
        (input1(indices) + input2(indices)) * std::pow(2, kNumIterations - 1);
  });

  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);
  Literal expected_output_literal =
      LiteralUtil::CreateFromArray(expected_output);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      HloTestBase::ExecuteReplicated(std::move(module),
                                     {{&input_literal1}, {&input_literal2}},
                                     /*num_replicas=*/kNumReplicas,
                                     /*run_hlo_passes=*/true,
                                     /*device_assignment=*/nullptr));
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[1]));
}

TEST_P(AllReduceTest, AsyncAllReduce_BF16_2GPUs) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  apply_op {
    x = bf16[] parameter(0)
    y = bf16[] parameter(1)
    ROOT apply_op = bf16[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = bf16[65536] parameter(0)
    ROOT all-reduce = bf16[65536] all-reduce(param_0), to_apply=apply_op, replica_groups={{0,1}}
  }
  )";

  const int64_t kNumReplicas = 2;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  int64_t num_elements =
      module->entry_computation()->root_instruction()->shape().dimensions()[0];

  Array<bfloat16> input1({num_elements}), input2({num_elements});
  input1.FillRandom(static_cast<bfloat16>(1.0f), 10.0f, /*seed=*/0);
  input2.FillRandom(static_cast<bfloat16>(1.0f), 10.0f, /*seed=*/1);
  Array<bfloat16> expected_output({num_elements});
  expected_output.Each([&](absl::Span<const int64_t> indices, bfloat16* val) {
    *val = input1(indices) + input2(indices);
  });

  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);
  Literal expected_output_literal =
      LiteralUtil::CreateFromArray(expected_output);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      HloTestBase::ExecuteReplicated(std::move(module),
                                     {{&input_literal1}, {&input_literal2}},
                                     /*num_replicas=*/kNumReplicas,
                                     /*run_hlo_passes=*/true,
                                     /*device_assignment=*/nullptr));
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[1]));
}

TEST_P(AllReduceTest, AsyncAllReduce_PRED_2GPUs) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  apply_op {
    x = pred[] parameter(0)
    y = pred[] parameter(1)
    ROOT apply_op = pred[] or(x, y)
  }

  ENTRY test_computation {
    param_0 = pred[65536] parameter(0)
    ROOT all-reduce = pred[65536] all-reduce(param_0), to_apply=apply_op, replica_groups={{0,1}}
  }
  )";

  const int64_t kNumReplicas = 2;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  int64_t num_elements =
      module->entry_computation()->root_instruction()->shape().dimensions()[0];

  Array<bool> input1({num_elements}), input2({num_elements});
  input1.FillRandomBool(/*seed=*/0);
  input2.FillRandomBool(/*seed=*/1);
  Array<bool> expected_output({num_elements});
  expected_output.Each([&](absl::Span<const int64_t> indices, bool* val) {
    *val = input1(indices) | input2(indices);
  });

  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);
  Literal expected_output_literal =
      LiteralUtil::CreateFromArray(expected_output);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      HloTestBase::ExecuteReplicated(std::move(module),
                                     {{&input_literal1}, {&input_literal2}},
                                     /*num_replicas=*/kNumReplicas,
                                     /*run_hlo_passes=*/true,
                                     /*device_assignment=*/nullptr));
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[1]));
}

TEST_P(AllReduceTest, AsyncAllReduce_8GPUs_AllReplicasOneGroup) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = f32[65536] parameter(0)
    ROOT all-reduce = f32[65536] all-reduce(param_0), to_apply=apply_op,
      replica_groups={{0,1,2,3,4,5,6,7}}
  }
  )";

  const int64_t kNumReplicas = 8;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      BuildTestInputsOutputs(*module, kNumReplicas, /*num_iterations=*/1));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      HloTestBase::ExecuteReplicated(std::move(module),
                                     /*arguments=*/test_io.InputLiteralPtrs(),
                                     /*num_replicas=*/kNumReplicas,
                                     /*run_hlo_passes=*/true,
                                     /*device_assignment=*/nullptr));
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    // NB: nccl accumulation order can be different from expected calculations
    // leading to differences in the results (floating point imprecision).
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-4}))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllReduceTest, AsyncAllReduce_8GPUs_2ReplicasPerGroup) {
  const int64_t kNumElements = 65536;
  const int64_t kNumIterations = 3;
  const auto kModuleStr = absl::StrFormat(
      R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  while_condition {
    limit = s32[] constant(%1$d)
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    ROOT result = pred[] compare(loop_counter, limit), direction=LT
  }

  while_body {
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    tensor = get-tuple-element(params), index=1
    out0 = f32[%2$d] all-reduce(tensor), to_apply=apply_op,
      replica_groups={{0,4},{1,5},{2,6},{3,7}}
    new_loop_counter = s32[] add(loop_counter, s32[] constant(1))
    ROOT result = (s32[], f32[%2$d]{0}) tuple(new_loop_counter, out0)
  }

  ENTRY test_computation {
    param_0 = f32[%2$d] parameter(0)
    while_init = (s32[], f32[%2$d]{0}) tuple(s32[] constant(0), param_0)
    while_result = (s32[], f32[%2$d]{0})
      while(while_init), condition=while_condition, body=while_body
    ROOT result = get-tuple-element(while_result), index=1
  }
  )",
      kNumIterations, kNumElements);

  const int64_t kNumReplicas = 8;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      BuildTestInputsOutputs(*module, kNumReplicas, kNumIterations));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      HloTestBase::ExecuteReplicated(std::move(module),
                                     /*arguments=*/test_io.InputLiteralPtrs(),
                                     /*num_replicas=*/kNumReplicas,
                                     /*run_hlo_passes=*/true,
                                     /*device_assignment=*/nullptr));
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Equal(test_io.expected_outputs[i], results[i]))
        << "ExpectedOutput != Result at index " << i;
  }
}

std::string GetAsyncTestName(bool is_async) {
  return is_async ? "async" : "sync";
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceTest, AllReduceTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<bool, bool>>& info) {
      return absl::StrCat(GetAsyncTestName(std::get<0>(info.param)), "_",
                          std::get<1>(info.param) ? "one_shot" : "nccl");
    });

}  // namespace
}  // namespace xla