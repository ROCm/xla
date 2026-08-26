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

#ifndef XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_GUARDED_BUFFER_H_
#define XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_GUARDED_BUFFER_H_

#include <cstdint>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"

namespace xla::gpu::rccl_benchmark {

// A device allocation laid out as [guard | payload | guard].
//
// The collective under test is only ever handed the payload sub-range; the
// guard regions are filled with a position-dependent poison pattern and checked
// after execution. The failure this exists for writes outside the range it was
// given, and without guards that shows up far downstream as a NaN loss several
// steps later. With guards the same failure is reported as "this buffer was
// overwritten N bytes past its end", which is the difference between a test
// that detects a problem and a test that locates one.
//
// Guards do not replace fault detection. A misrouted descriptor can address
// memory far enough away to trigger a hardware memory fault before it ever
// touches an adjacent guard, which kills the process instead of corrupting it.
// Both signals are needed; the runner treats a faulting child as its own
// failure class.
struct GuardedRegion {
  int64_t guard_bytes = 0;
  int64_t payload_bytes = 0;

  int64_t payload_offset() const { return guard_bytes; }
  int64_t total_bytes() const { return 2 * guard_bytes + payload_bytes; }
};

// Guard size used unless a case overrides it. Large enough to catch the modest
// overruns that a wrong element count produces, small enough that a case with
// several large buffers still fits comfortably.
inline constexpr int64_t kDefaultGuardBytes = 1 << 20;  // 1 MiB

// Returns the poison byte for `offset` within an allocation. The value depends
// on the offset so that a block of guard bytes copied to the wrong place is
// still detected.
uint8_t GuardByte(int64_t offset);

// Raw byte transfers. The shared multi-GPU helpers only speak float; these
// cases carry bfloat16 payloads and compare them bit-for-bit, so they need to
// move opaque bytes.
absl::Status WriteDeviceBytes(se::Stream& stream, se::DeviceAddressBase buffer,
                              absl::Span<const uint8_t> bytes);
absl::StatusOr<std::vector<uint8_t>> ReadDeviceBytes(
    se::Stream& stream, se::DeviceAddressBase buffer, int64_t num_bytes);

// Writes guards plus `payload_words` into `buffer`. `payload_words` must cover
// exactly `region.payload_bytes`.
absl::Status WriteGuardedBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                const GuardedRegion& region,
                                absl::Span<const uint16_t> payload_words);

// Writes guards plus a payload consisting of `fill_word` repeated.
absl::Status WriteGuardedBufferFilled(se::Stream& stream,
                                      se::DeviceAddressBase buffer,
                                      const GuardedRegion& region,
                                      uint16_t fill_word);

// Reads the whole allocation, guards included, so both can be checked from one
// transfer.
absl::StatusOr<std::vector<uint8_t>> ReadGuardedBuffer(
    se::Stream& stream, se::DeviceAddressBase buffer,
    const GuardedRegion& region);

// Fails if either guard region of `image` was modified. `label` names the
// buffer in the error message.
absl::Status CheckGuards(absl::Span<const uint8_t> image,
                         const GuardedRegion& region, absl::string_view label);

// Fails if the payload of `image` differs from `expected_words`. The message
// decodes both words so it names which buffer and rank the wrong data came
// from.
absl::Status CheckPayload(absl::Span<const uint8_t> image,
                          const GuardedRegion& region,
                          absl::Span<const uint16_t> expected_words,
                          absl::string_view label);

// As above, but the expected word at each index is produced on demand. An
// AllGather destination is `num_ranks` times the size of its input, and with
// several such buffers per device the materialized expectation would be the
// largest allocation in the test for no benefit.
absl::Status CheckPayloadGenerated(
    absl::Span<const uint8_t> image, const GuardedRegion& region,
    absl::FunctionRef<uint16_t(int64_t)> expected_word_at,
    absl::string_view label);

}  // namespace xla::gpu::rccl_benchmark

#endif  // XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_GUARDED_BUFFER_H_
