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

// Tests for roctx_utils.cc — the ROCm dual-push implementation of
// nvtx_utils.h. Verifies that RangePush/RangePop populate AnnotationStack
// (Pipeline A) and emit roctx markers (Pipeline B).

#include <string>

#include "rocm/include/rocprofiler-sdk-roctx/roctx.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace tsl {
namespace profiler {
namespace {

// RAII guard that enables AnnotationStack on construction and disables
// on destruction so tests don't leak enabled state.
class AnnotationStackGuard {
 public:
  AnnotationStackGuard() { AnnotationStack::Enable(true); }
  ~AnnotationStackGuard() { AnnotationStack::Enable(false); }
};

TEST(RoctxUtils, DefaultProfilerDomainReturnsNonNull) {
  ProfilerDomainHandle domain = DefaultProfilerDomain();
  ASSERT_NE(domain, nullptr)
      << "On ROCm builds, DefaultProfilerDomain() must return non-null "
         "to trigger the domain path in PushAnnotation";
}

TEST(RoctxUtils, RangePushPopulatesAnnotationStack) {
  auto domain = DefaultProfilerDomain();
  ASSERT_NE(domain, nullptr);

  AnnotationStackGuard guard;
  RangePush(domain, "test_op");
  EXPECT_EQ(AnnotationStack::Get(), "test_op");
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, NestedPushPopMaintainsAnnotationStack) {
  auto domain = DefaultProfilerDomain();
  ASSERT_NE(domain, nullptr);

  AnnotationStackGuard guard;

  RangePush(domain, "outer");
  EXPECT_EQ(AnnotationStack::Get(), "outer");

  RangePush(domain, "inner");
  EXPECT_EQ(AnnotationStack::Get(), "outer::inner");

  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "outer");

  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, PushPopWithAnnotationStackDisabled) {
  auto domain = DefaultProfilerDomain();
  ASSERT_NE(domain, nullptr);

  // AnnotationStack is NOT enabled. RangePush/RangePop must not crash
  // and must not populate the stack.
  RangePush(domain, "ignored_op");
  EXPECT_EQ(AnnotationStack::Get(), "");
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, ScopedAnnotationIntegration) {
  // Full chain: ScopedAnnotation → PushAnnotation →
  // DefaultProfilerDomain() non-null → RangePush → dual-push.
  AnnotationStackGuard guard;
  {
    ScopedAnnotation annotation("my_kernel");
    EXPECT_EQ(AnnotationStack::Get(), "my_kernel");
    {
      ScopedAnnotation nested("inner_kernel");
      EXPECT_EQ(AnnotationStack::Get(), "my_kernel::inner_kernel");
    }
    EXPECT_EQ(AnnotationStack::Get(), "my_kernel");
  }
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, ScopedAnnotationDisabledStackDoesNotCrash) {
  // ScopedAnnotation with stack disabled — must not crash.
  {
    ScopedAnnotation annotation("disabled_op");
    EXPECT_EQ(AnnotationStack::Get(), "");
  }
  EXPECT_EQ(AnnotationStack::Get(), "");
}

TEST(RoctxUtils, DirectRoctxCallsDoNotCrash) {
  // Verify direct roctx API calls work (linked at build time).
  int depth = roctxRangePushA("roctx_utils_test_label");
  EXPECT_GE(depth, 0);
  int pop_result = roctxRangePop();
  EXPECT_GE(pop_result, 0);
}

TEST(RoctxUtils, DirectRoctxMarkDoesNotCrash) {
  roctxMarkA("roctx_utils_test_mark");
}

TEST(RoctxUtils, DirectRoctxNestedRanges) {
  int d0 = roctxRangePushA("level_0");
  EXPECT_EQ(d0, 0);
  int d1 = roctxRangePushA("level_1");
  EXPECT_EQ(d1, 1);
  int d2 = roctxRangePushA("level_2");
  EXPECT_EQ(d2, 2);

  EXPECT_EQ(roctxRangePop(), 2);
  EXPECT_EQ(roctxRangePop(), 1);
  EXPECT_EQ(roctxRangePop(), 0);
}

TEST(RoctxUtils, DetailRangePushIsNoOp) {
  auto domain = DefaultProfilerDomain();
  ASSERT_NE(domain, nullptr);
  detail::RangePush(domain, nullptr, 0, nullptr, 0);
}

TEST(RoctxUtils, RegisterStringReturnsNull) {
  auto domain = DefaultProfilerDomain();
  ASSERT_NE(domain, nullptr);
  StringHandle handle = RegisterString(domain, "test_string");
  EXPECT_EQ(handle, nullptr);
}

TEST(RoctxUtils, RegisterSchemaReturnsZero) {
  auto domain = DefaultProfilerDomain();
  ASSERT_NE(domain, nullptr);
  uint64_t schema_id = RegisterSchema(domain, nullptr);
  EXPECT_EQ(schema_id, 0);
}

TEST(RoctxUtils, DualPushBothPipelinesSimultaneously) {
  // Verify that a single RangePush populates AnnotationStack AND emits
  // an roctx call without crashing. This is the core dual-push contract.
  auto domain = DefaultProfilerDomain();
  ASSERT_NE(domain, nullptr);

  AnnotationStackGuard guard;

  // RangePush should do both: AnnotationStack + roctxRangePushA.
  RangePush(domain, "dual_push_test");
  EXPECT_EQ(AnnotationStack::Get(), "dual_push_test");

  // RangePop should do both: AnnotationStack + roctxRangePop.
  RangePop(domain);
  EXPECT_EQ(AnnotationStack::Get(), "");
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
