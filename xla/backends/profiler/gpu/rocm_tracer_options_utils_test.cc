/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/profiler/gpu/rocm_tracer_options_utils.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using ::tensorflow::ProfileOptions;
using ::tensorflow::profiler::XSpace;
using ::testing::Contains;
using ::testing::IsEmpty;
using ::testing::SizeIs;

void SetInt(ProfileOptions& options, absl::string_view key, int64_t value) {
  ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_int64_value(value);
  (*options.mutable_advanced_configuration())[std::string(key)] = config_value;
}

void SetBool(ProfileOptions& options, absl::string_view key, bool value) {
  ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_bool_value(value);
  (*options.mutable_advanced_configuration())[std::string(key)] = config_value;
}

void SetString(ProfileOptions& options, absl::string_view key,
               absl::string_view value) {
  ProfileOptions::AdvancedConfigValue config_value;
  config_value.set_string_value(std::string(value));
  (*options.mutable_advanced_configuration())[std::string(key)] = config_value;
}

MATCHER_P(MentionsKey, key, absl::StrCat("mentions the key '", key, "'")) {
  return absl::StrContains(arg, absl::StrCat("'", key, "'"));
}

// Sentinel defaults, chosen so that "the field was never touched" is
// distinguishable from "the field was set to a plausible value".
constexpr uint64_t kTracerAnnotationSentinel = 111;
constexpr uint64_t kCallbackSentinel = 222;
constexpr uint64_t kActivitySentinel = 333;
constexpr uint64_t kCollectorAnnotationSentinel = 444;
constexpr uint32_t kNumGpusSentinel = 8;

struct Fixture {
  ProfileOptions options;
  RocmTracerOptions tracer = {kTracerAnnotationSentinel};
  RocmTraceCollectorOptions collector = {kCallbackSentinel, kActivitySentinel,
                                         kCollectorAnnotationSentinel,
                                         kNumGpusSentinel};
  RocmTracerOptionDiagnostics diagnostics;

  void Run() {
    UpdateRocmTracerOptionsFromProfilerOptions(options, tracer, collector,
                                               diagnostics);
  }

  // True when no option field was modified.
  bool AllFieldsUntouched() const {
    return tracer.max_annotation_strings == kTracerAnnotationSentinel &&
           collector.max_callback_api_events == kCallbackSentinel &&
           collector.max_activity_api_events == kActivitySentinel &&
           collector.max_annotation_strings == kCollectorAnnotationSentinel &&
           collector.num_gpus == kNumGpusSentinel;
  }
};

TEST(RocmTracerOptionsUtilsTest, EmptyMapIsANoOp) {
  Fixture f;
  f.Run();

  EXPECT_TRUE(f.AllFieldsUntouched());
  EXPECT_TRUE(f.diagnostics.empty());
}

TEST(RocmTracerOptionsUtilsTest, SetsCallbackEventLimit) {
  Fixture f;
  SetInt(f.options, "gpu_max_callback_api_events", 4242);
  f.Run();

  EXPECT_EQ(f.collector.max_callback_api_events, 4242);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, SetsActivityEventLimit) {
  Fixture f;
  SetInt(f.options, "gpu_max_activity_api_events", 4242);
  f.Run();

  EXPECT_EQ(f.collector.max_activity_api_events, 4242);
  // Wired silently: the cap that reads this field is not yet enforced, but
  // fixing that requires no further plumbing, so there is no warning to
  // retract later.
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, SetsBothAnnotationLimitsFromOneKey) {
  Fixture f;
  SetInt(f.options, "gpu_max_annotation_strings", 777);
  f.Run();

  EXPECT_EQ(f.tracer.max_annotation_strings, 777);
  EXPECT_EQ(f.collector.max_annotation_strings, 777);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
}

TEST(RocmTracerOptionsUtilsTest, SetsNumGpusAndWarnsAboutPostHocDrop) {
  Fixture f;
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 2);
  f.Run();

  // The reset-to-all for out-of-range values lives in the caller, so the raw
  // value is visible here.
  EXPECT_EQ(f.collector.num_gpus, 2);
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  ASSERT_THAT(f.diagnostics.warnings, SizeIs(1));
  EXPECT_TRUE(absl::StrContains(f.diagnostics.warnings[0], "post-hoc"));
}

TEST(RocmTracerOptionsUtilsTest, UnknownKeyIsReportedByName) {
  Fixture f;
  SetInt(f.options, "gpu_max_callbac_api_events", 4242);  // realistic typo
  f.Run();

  ASSERT_THAT(f.diagnostics.errors, SizeIs(1));
  EXPECT_THAT(f.diagnostics.errors[0], MentionsKey("gpu_max_callbac_api_events"));
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest, AllUnknownKeysAreReported) {
  Fixture f;
  SetInt(f.options, "not_a_key", 1);
  SetBool(f.options, "also_not_a_key", true);
  SetString(f.options, "still_not_a_key", "x");
  f.Run();

  // Every unknown key is named. A parser that returns on the first failure --
  // as the CUPTI one does -- reports only one of these.
  EXPECT_THAT(f.diagnostics.errors, SizeIs(3));
  EXPECT_THAT(f.diagnostics.errors, Contains(MentionsKey("not_a_key")));
  EXPECT_THAT(f.diagnostics.errors, Contains(MentionsKey("also_not_a_key")));
  EXPECT_THAT(f.diagnostics.errors, Contains(MentionsKey("still_not_a_key")));
}

TEST(RocmTracerOptionsUtilsTest, WrongTypeIsReportedOnce) {
  Fixture f;
  SetBool(f.options, "gpu_max_callback_api_events", true);  // documented int64
  f.Run();

  // Exactly one message. tsl::profiler::SetValue returns its type error before
  // erasing the key from the working set, so a parser that forgets the
  // explicit erase reports this key twice: once as a type error and once as
  // unrecognised.
  ASSERT_THAT(f.diagnostics.errors, SizeIs(1));
  EXPECT_THAT(f.diagnostics.errors[0],
              MentionsKey("gpu_max_callback_api_events"));
  EXPECT_THAT(f.diagnostics.warnings, IsEmpty());
  EXPECT_EQ(f.collector.max_callback_api_events, kCallbackSentinel);
}

TEST(RocmTracerOptionsUtilsTest, UnimplementedKeysAreAcceptedWithWarnings) {
  Fixture f;
  SetBool(f.options, "gpu_enable_nvtx_tracking", true);
  SetBool(f.options, "gpu_enable_cupti_activity_graph_trace", true);
  SetBool(f.options, "gpu_dump_graph_node_mapping", true);
  SetString(f.options, "gpu_pm_sample_counters", "SQ_WAVES");
  SetInt(f.options, "gpu_pm_sample_interval_us", 500);
  SetInt(f.options, "gpu_pm_sample_buffer_size_per_gpu_mb", 64);
  SetBool(f.options, "gpu_aggregated_tracing", true);
  f.Run();

  // A script written against CUDA must not fail on ROCm just because a key is
  // not implemented here yet.
  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(7));
  EXPECT_THAT(f.diagnostics.warnings,
              Contains(MentionsKey("gpu_enable_nvtx_tracking")));
  EXPECT_THAT(f.diagnostics.warnings,
              Contains(MentionsKey("gpu_pm_sample_counters")));
  EXPECT_TRUE(f.AllFieldsUntouched());
}

TEST(RocmTracerOptionsUtilsTest, WrongTypeOnAnUnimplementedKeyStillErrors) {
  Fixture f;
  SetString(f.options, "gpu_pm_sample_interval_us", "500");  // documented int64
  f.Run();

  // Type-checking the unimplemented keys is not decorative: a user should hear
  // about this now, not in the release that implements the key.
  ASSERT_THAT(f.diagnostics.errors, SizeIs(1));
  EXPECT_THAT(f.diagnostics.errors[0], MentionsKey("gpu_pm_sample_interval_us"));
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(1));
}

TEST(RocmTracerOptionsUtilsTest, AllDocumentedKeysAreRecognised) {
  // The ten keys published at openxla.org/xprof/advanced_profiler_options,
  // with their documented types. A user copying the page verbatim must not see
  // a single error on ROCm. If the page grows an eleventh key, this test fails
  // and someone has to look at it -- which is the point.
  Fixture f;
  SetInt(f.options, "gpu_max_callback_api_events", 2 * 1024 * 1024);
  SetInt(f.options, "gpu_max_activity_api_events", 2 * 1024 * 1024);
  SetInt(f.options, "gpu_max_annotation_strings", 1024 * 1024);
  SetInt(f.options, "gpu_num_chips_to_profile_per_task", 4);
  SetBool(f.options, "gpu_enable_nvtx_tracking", true);
  SetBool(f.options, "gpu_enable_cupti_activity_graph_trace", true);
  SetBool(f.options, "gpu_dump_graph_node_mapping", true);
  SetString(f.options, "gpu_pm_sample_counters", "SQ_WAVES,GRBM_GUI_ACTIVE");
  SetInt(f.options, "gpu_pm_sample_interval_us", 500);
  SetInt(f.options, "gpu_pm_sample_buffer_size_per_gpu_mb", 64);
  f.Run();

  EXPECT_THAT(f.diagnostics.errors, IsEmpty());
  // Six unimplemented keys, plus the post-hoc caveat on num_chips.
  EXPECT_THAT(f.diagnostics.warnings, SizeIs(7));
  EXPECT_EQ(f.collector.max_callback_api_events, 2 * 1024 * 1024);
  EXPECT_EQ(f.tracer.max_annotation_strings, 1024 * 1024);
  EXPECT_EQ(f.collector.num_gpus, 4);
}

TEST(RocmTracerOptionsUtilsTest, AppendOptionDiagnosticsWritesToXSpace) {
  RocmTracerOptionDiagnostics diagnostics;
  diagnostics.errors = {"first error", "second error"};
  diagnostics.warnings = {"a warning"};

  XSpace space;
  AppendOptionDiagnostics(diagnostics, &space);

  ASSERT_EQ(space.errors_size(), 2);
  EXPECT_EQ(space.errors(0), "first error");
  EXPECT_EQ(space.errors(1), "second error");
  ASSERT_EQ(space.warnings_size(), 1);
  EXPECT_EQ(space.warnings(0), "a warning");
}

TEST(RocmTracerOptionsUtilsTest, AppendOptionDiagnosticsToleratesEmptyAndNull) {
  XSpace space;
  AppendOptionDiagnostics(RocmTracerOptionDiagnostics{}, &space);
  EXPECT_EQ(space.errors_size(), 0);
  EXPECT_EQ(space.warnings_size(), 0);

  RocmTracerOptionDiagnostics diagnostics;
  diagnostics.errors = {"dropped on the floor"};
  AppendOptionDiagnostics(diagnostics, nullptr);  // must not crash
}

}  // namespace
}  // namespace profiler
}  // namespace xla
