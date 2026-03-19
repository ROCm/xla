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

#include "xla/stream_executor/rocm/rocm_core_info_table.h"

#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

void CheckPeakOpsPerNs(const DeviceDescription& device_info,
                       bool is_matrix_unit, xla::PrimitiveType dtype,
                       double expected_tflops) {
  const ExecutionUnitDescription* eu_descr =
      is_matrix_unit ? device_info.matrix_unit_description()
                     : device_info.scalar_unit_description();

  ASSERT_NE(eu_descr, nullptr);

  std::optional<ExecutionUnitDescription::RateInfo> dtype_rates =
      eu_descr->GetRateInfo(dtype);
  ASSERT_TRUE(dtype_rates.has_value());

  double flops_per_ns_per_unit = dtype_rates->clock_rate_ghz *
                                 dtype_rates->ops_per_clock *
                                 2;  // FMA is 2 ops.
  int64_t n_compute_units =
      device_info.core_count() * dtype_rates->units_per_core;

  float ops_per_ns = flops_per_ns_per_unit * n_compute_units;

  // Allow for 2% error to account for imprecise estimates.
  EXPECT_NEAR(ops_per_ns / 1000.0, expected_tflops, expected_tflops * 0.02)
      << "Failed for dtype: " << xla::PrimitiveType_Name(dtype);
}

// Helper to create a DeviceDescription for a specific ROCm GPU.
DeviceDescription MakeDeviceDesc(const char* gcn_arch, int core_count,
                                 float clock_rate_ghz) {
  DeviceDescription desc;
  RocmComputeCapability cc(gcn_arch);
  desc.set_gpu_compute_capability(GpuComputeCapability(cc));
  desc.set_core_count(core_count);
  desc.set_clock_rate_ghz(clock_rate_ghz);
  desc.set_fpus_per_core(GetFpusPerCore(cc));
  FillExecutionUnitDesc(cc, clock_rate_ghz, desc);
  return desc;
}

// MI210 (gfx90a): 104 CUs, 1.7 GHz.
// Official peak per GCD: FP16 matrix ~362 TFLOPS, FP64 matrix ~90.5 TFLOPS.
TEST(RocmCoreInfoTableTest, CalculatePeakOpsPerNsMI210) {
  DeviceDescription mi210 = MakeDeviceDesc("gfx90a", 104, 1.7);

  // Matrix unit (MFMA) tests.
  CheckPeakOpsPerNs(mi210, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F16, 362.1);
  CheckPeakOpsPerNs(mi210, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::BF16, 362.1);
  CheckPeakOpsPerNs(mi210, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F32, 181.0);
  CheckPeakOpsPerNs(mi210, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F64, 90.5);

  // Scalar unit (Stream Processor) tests.
  CheckPeakOpsPerNs(mi210, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F16, 45.3);
  CheckPeakOpsPerNs(mi210, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F32, 22.6);
  // gfx90a has unique 1:1 FP64:FP32 scalar ratio.
  CheckPeakOpsPerNs(mi210, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F64, 22.6);
  CheckPeakOpsPerNs(mi210, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::S32, 22.6);
}

// MI300X (gfx942): 304 CUs, 2.1 GHz.
// Official peak: FP8=2615 TFLOPS, FP16=1307 TFLOPS, FP64=163 TFLOPS.
TEST(RocmCoreInfoTableTest, CalculatePeakOpsPerNsMI300X) {
  DeviceDescription mi300x = MakeDeviceDesc("gfx942", 304, 2.1);

  // Matrix unit (MFMA) tests.
  CheckPeakOpsPerNs(mi300x, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F8E4M3FN, 2615.0);
  CheckPeakOpsPerNs(mi300x, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::S8, 2615.0);
  CheckPeakOpsPerNs(mi300x, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F16, 1307.0);
  CheckPeakOpsPerNs(mi300x, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::BF16, 1307.0);
  CheckPeakOpsPerNs(mi300x, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F32, 654.0);
  CheckPeakOpsPerNs(mi300x, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F64, 163.4);

  // Scalar unit tests.
  CheckPeakOpsPerNs(mi300x, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F32, 163.4);
  CheckPeakOpsPerNs(mi300x, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F64, 81.7);
  CheckPeakOpsPerNs(mi300x, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::S32, 81.7);
}

// MI100 (gfx908): 120 CUs, 1.502 GHz.
// Official peak: FP16 matrix=184.6 TFLOPS, FP32 matrix=92.3 TFLOPS.
TEST(RocmCoreInfoTableTest, CalculatePeakOpsPerNsMI100) {
  DeviceDescription mi100 = MakeDeviceDesc("gfx908", 120, 1.502);

  CheckPeakOpsPerNs(mi100, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F16, 184.6);
  CheckPeakOpsPerNs(mi100, /*is_matrix_unit=*/true,
                    xla::PrimitiveType::F32, 92.3);

  // No FP64 matrix on CDNA 1.
  EXPECT_EQ(mi100.matrix_unit_description()->GetRateInfo(xla::F64),
            std::nullopt);

  // Scalar unit tests.
  CheckPeakOpsPerNs(mi100, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F32, 23.1);
  // FP64 at 1:2 rate.
  CheckPeakOpsPerNs(mi100, /*is_matrix_unit=*/false,
                    xla::PrimitiveType::F64, 11.5);
}

TEST(RocmCoreInfoTableTest, GetFpusPerCore) {
  // Backward-compatible values.
  EXPECT_EQ(GetFpusPerCore(RocmComputeCapability("gfx906")), 64);
  EXPECT_EQ(GetFpusPerCore(RocmComputeCapability("gfx900")), 64);
  EXPECT_EQ(GetFpusPerCore(RocmComputeCapability("gfx908")), 128);
  EXPECT_EQ(GetFpusPerCore(RocmComputeCapability("gfx90a")), 128);
  EXPECT_EQ(GetFpusPerCore(RocmComputeCapability("gfx942")), 128);
  EXPECT_EQ(GetFpusPerCore(RocmComputeCapability("gfx950")), 128);
  EXPECT_EQ(GetFpusPerCore(RocmComputeCapability("gfx1100")), 128);
}

TEST(RocmCoreInfoTableTest, UnsupportedArchHasNoExecutionUnitDesc) {
  DeviceDescription desc = MakeDeviceDesc("gfx906", 60, 1.8);
  EXPECT_EQ(desc.scalar_unit_description(), nullptr);
  EXPECT_EQ(desc.matrix_unit_description(), nullptr);
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
