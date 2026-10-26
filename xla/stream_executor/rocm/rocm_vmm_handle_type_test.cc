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

#include "xla/stream_executor/rocm/rocm_vmm_handle_type.h"

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_device_address_vmm_allocator.h"
#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/rocm/rocm_vmm_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

// Exports `handle` as a POSIX file descriptor exactly the way RCCL's symmetric
// memory path does (symMemoryObtain -> ncclProxyClientGetFdBlocking), and
// immediately closes it. This is the property the exportable handle type
// exists for: before the fix the allocation carried hipMemHandleTypeNone and
// this export is impossible, which made RCCL block forever waiting for a file
// descriptor that could never be produced.
hipError_t TryExportAsPosixFd(hipMemGenericAllocationHandle_t handle) {
  int fd = -1;
  hipError_t result = hipMemExportToShareableHandle(
      &fd, handle, hipMemHandleTypePosixFileDescriptor, 0ULL);
  if (result != hipSuccess) {
    return result;
  }
  EXPECT_GE(fd, 0) << "Export succeeded but produced an invalid descriptor.";
  if (fd >= 0) {
    close(fd);
  }
  return hipSuccess;
}

class RocmVmmHandleTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("ROCM");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "ROCM platform not available";
    }
    platform_ = platform_or.value();

    auto executor_or = platform_->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "ROCM executor not available: " << executor_or.status();
    }
    executor_ = executor_or.value();

    std::unique_ptr<ActivateContext> activation = executor_->Activate();
    ASSERT_OK(ToStatus(hipDeviceGet(&device_, executor_->device_ordinal()),
                       "hipDeviceGet"));
  }

  // Skips the calling test below ROCm 10.0, or on a driver that refuses
  // exportable allocations. RocmVmmHandleType() falls back to
  // hipMemHandleTypeNone in both cases, so the exportability assertions below
  // do not apply; the fallback itself is covered by
  // RequestedHandleTypeMatchesProbe.
  void SkipIfExportableHandlesUnsupported() {
    std::unique_ptr<ActivateContext> activation = executor_->Activate();
    if (RocmVmmHandleType(device_) != hipMemHandleTypePosixFileDescriptor) {
      GTEST_SKIP() << "Driver does not support exportable VMM handles on "
                      "device "
                   << executor_->device_ordinal();
    }
  }

  Platform* platform_ = nullptr;
  StreamExecutor* executor_ = nullptr;
  hipDevice_t device_ = 0;
};

// The probe accepts POSIX file descriptor handles on hardware that supports
// them. A ROCm 10.0+ device that fails this is not a test bug - it means
// symmetric memory cannot work there at all.
TEST_F(RocmVmmHandleTypeTest, ProbeSelectsExportableHandlesWhenSupported) {
  if (TF_ROCM_VERSION < 100000) {
    GTEST_SKIP() << "Exportable VMM handles are gated off below ROCm 10.0; "
                    "the probe does not run.";
  }
  if (getenv("XLA_ROCM_VMM_EXPORTABLE_HANDLES") != nullptr) {
    GTEST_SKIP() << "XLA_ROCM_VMM_EXPORTABLE_HANDLES overrides the probe; see "
                    "rocm_vmm_handle_type_disabled_test.";
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();

  hipMemAllocationProp props = {};
  props.type = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device_;
  props.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;

  size_t granularity = 0;
  hipMemGenericAllocationHandle_t handle = nullptr;
  bool driver_supports_exportable =
      hipMemGetAllocationGranularity(&granularity, &props,
                                     hipMemAllocationGranularityMinimum) ==
          hipSuccess &&
      granularity != 0 &&
      hipMemCreate(&handle, granularity, &props, 0ULL) == hipSuccess;
  if (driver_supports_exportable) {
    EXPECT_EQ(hipMemRelease(handle), hipSuccess);
  }

  EXPECT_EQ(RocmVmmHandleType(device_) == hipMemHandleTypePosixFileDescriptor,
            driver_supports_exportable);
}

TEST_F(RocmVmmHandleTypeTest, RequestedHandleTypeMatchesProbe) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();

  hipMemAllocationProp props = MakeRocmVmmAllocationProps(device_);

  EXPECT_EQ(props.requestedHandleTypes, RocmVmmHandleType(device_));
  EXPECT_EQ(props.type, hipMemAllocationTypePinned);
  EXPECT_EQ(props.location.type, hipMemLocationTypeDevice);
  EXPECT_EQ(props.location.id, device_);
}

// The handle type is probed once and cached; every later query must agree with
// the first, or allocation sites and granularity queries drift apart.
TEST_F(RocmVmmHandleTypeTest, HandleTypeIsStableAcrossCalls) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();

  hipMemAllocationHandleType first = RocmVmmHandleType(device_);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(RocmVmmHandleType(device_), first);
    EXPECT_EQ(MakeRocmVmmAllocationProps(device_).requestedHandleTypes, first);
  }
}

// Every device the platform exposes must agree, otherwise a multi-GPU job
// would have some ranks registerable and some not - the asymmetric case that
// deadlocks instead of failing.
TEST_F(RocmVmmHandleTypeTest, AllDevicesAgreeOnHandleType) {
  SkipIfExportableHandlesUnsupported();

  for (int ordinal = 0; ordinal < platform_->VisibleDeviceCount(); ++ordinal) {
    ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                         platform_->ExecutorForDevice(ordinal));
    std::unique_ptr<ActivateContext> activation = executor->Activate();
    hipDevice_t device = 0;
    ASSERT_OK(ToStatus(hipDeviceGet(&device, ordinal), "hipDeviceGet"));
    EXPECT_EQ(RocmVmmHandleType(device), hipMemHandleTypePosixFileDescriptor)
        << "Device " << ordinal
        << " does not request exportable handles while device 0 does.";
  }
}

// This is the allocation that hung: RocmDeviceAddressVmmAllocator creates its
// physical memory through RocmRawMemoryAllocation::Create.
TEST_F(RocmVmmHandleTypeTest, RawMemoryAllocationHandleIsExportable) {
  SkipIfExportableHandlesUnsupported();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmRawMemoryAllocation> allocation,
      RocmRawMemoryAllocation::Create(executor_, /*size=*/1024 * 1024));
  ASSERT_NE(allocation->GetHandle(), nullptr);

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  EXPECT_EQ(TryExportAsPosixFd(allocation->GetHandle()), hipSuccess)
      << "VMM allocation handle cannot be exported as a POSIX file "
         "descriptor; RCCL symmetric memory registration would hang.";
}

// An exported descriptor that cannot be imported back is of no use to the NCCL
// proxy, so check the full round trip rather than just the export call.
TEST_F(RocmVmmHandleTypeTest, ExportedDescriptorCanBeImported) {
  SkipIfExportableHandlesUnsupported();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmRawMemoryAllocation> allocation,
      RocmRawMemoryAllocation::Create(executor_, /*size=*/1024 * 1024));

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  int fd = -1;
  ASSERT_EQ(
      hipMemExportToShareableHandle(&fd, allocation->GetHandle(),
                                    hipMemHandleTypePosixFileDescriptor, 0ULL),
      hipSuccess);
  ASSERT_GE(fd, 0);

  hipMemGenericAllocationHandle_t imported = nullptr;
  hipError_t import_result = hipMemImportFromShareableHandle(
      &imported, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
      hipMemHandleTypePosixFileDescriptor);
  close(fd);

  ASSERT_EQ(import_result, hipSuccess);
  ASSERT_NE(imported, nullptr);
  EXPECT_EQ(hipMemRelease(imported), hipSuccess);
}

TEST_F(RocmVmmHandleTypeTest, VmmAllocatorMemoryIsExportable) {
  SkipIfExportableHandlesUnsupported();

  RocmVmmAllocator allocator(executor_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator.Allocate(/*size=*/1024 * 1024));
  ASSERT_NE(allocation->address().opaque(), nullptr);

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(
      hipMemRetainAllocationHandle(&handle, allocation->address().opaque()),
      hipSuccess);

  EXPECT_EQ(TryExportAsPosixFd(handle), hipSuccess);
  EXPECT_EQ(hipMemRelease(handle), hipSuccess);
}

// The path symmetric memory actually takes: a collective-space buffer from the
// kVmm allocator, retained and exported the way RCCL does after
// ncclCommWindowRegister.
TEST_F(RocmVmmHandleTypeTest, CollectiveAllocationIsExportable) {
  SkipIfExportableHandlesUnsupported();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor_->CreateStream());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmDeviceAddressVmmAllocator> allocator,
      RocmDeviceAddressVmmAllocator::Create(executor_, stream.get()));

  ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), /*size=*/393216,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kCollective)));
  ASSERT_FALSE(scoped_address.is_null());

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipMemRetainAllocationHandle(&handle, scoped_address->opaque()),
            hipSuccess);

  EXPECT_EQ(TryExportAsPosixFd(handle), hipSuccess)
      << "Collective-space VMM buffer is not registerable; this is the "
         "configuration that deadlocked in ncclProxyClientGetFdBlocking.";
  EXPECT_EQ(hipMemRelease(handle), hipSuccess);
}

// MakeRocmVmmAllocationProps() exists so that the site querying granularity
// and the site allocating cannot disagree. Checking that a one-byte request is
// padded to exactly the granularity reported for the same properties pins that
// down: if a caller reverted to hand-built properties with a different handle
// type and the driver reported a different granularity, this would fail.
TEST_F(RocmVmmHandleTypeTest, PaddingMatchesGranularityForRequestedHandles) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();

  hipMemAllocationProp props = MakeRocmVmmAllocationProps(device_);
  size_t granularity = 0;
  ASSERT_OK(ToStatus(hipMemGetAllocationGranularity(
      &granularity, &props, hipMemAllocationGranularityRecommended)));
  ASSERT_GT(granularity, 0u);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmRawMemoryAllocation> allocation,
                       RocmRawMemoryAllocation::Create(executor_, /*size=*/1));
  EXPECT_EQ(allocation->address().size(), granularity);
}

}  // namespace
}  // namespace stream_executor::gpu
