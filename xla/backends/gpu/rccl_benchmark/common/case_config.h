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

#ifndef XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_CASE_CONFIG_H_
#define XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_CASE_CONFIG_H_

#include <cstdint>
#include <string>

namespace xla::gpu::rccl_benchmark {

// Knobs a case reads from the environment.
//
// The shape of a case - how many operations share a group, whether they are
// submitted together - is fixed in C++, because that is what the case is about.
// The size and the expected library behaviour are environment-driven, because
// the runner sweeps them and because the values that make a case meaningful
// depend on the RCCL build under test rather than on the test source.
struct CaseConfig {
  // Bytes each rank contributes. For an AllGather the resulting transfer is
  // this times the number of ranks, which is the quantity RCCL compares
  // against its feature thresholds.
  int64_t per_rank_bytes = 0;

  // Poison region placed on each side of every payload.
  int64_t guard_bytes = 0;

  // What the runner asserts the library will do. Checked against the debug log
  // after the collective runs; a mismatch fails the case instead of reporting
  // an unearned pass.
  bool expect_warp_speed = false;

  // How many times to execute the collective within one process. Repeats catch
  // corruption that only appears once a communicator has been reused.
  int repeats = 1;

  std::string Describe() const;
};

CaseConfig CaseConfigFromEnv();

}  // namespace xla::gpu::rccl_benchmark

#endif  // XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_CASE_CONFIG_H_
