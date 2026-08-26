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

#include "xla/backends/gpu/rccl_benchmark/common/guarded_buffer.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/rccl_benchmark/common/data_pattern.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu::rccl_benchmark {
namespace {

// Number of differing bytes reported before the message is truncated. Enough to
// see the shape of the corruption without producing an unreadable log.
constexpr int kMaxReportedMismatches = 8;

// Checks the half-open byte range [begin, end) of `image` against the poison
// pattern. `distance_description` renders how far a given offset lies from the
// payload, which is the number a reader actually wants when triaging.
absl::Status CheckRegionUntouched(
    absl::Span<const uint8_t> image, int64_t begin, int64_t end,
    absl::string_view label, absl::string_view which,
    absl::FunctionRef<std::string(int64_t)> distance_description) {
  int64_t first_bad = -1;
  int64_t bad_count = 0;
  for (int64_t offset = begin; offset < end; ++offset) {
    if (image[offset] != GuardByte(offset)) {
      if (first_bad < 0) {
        first_bad = offset;
      }
      ++bad_count;
    }
  }
  if (bad_count == 0) {
    return absl::OkStatus();
  }
  return absl::DataLossError(absl::StrFormat(
      "%s: %s guard was overwritten; %d of %d guard bytes changed, first at "
      "allocation offset %d (%s); saw 0x%02x, expected 0x%02x. An "
      "out-of-bounds write by the collective is the expected cause.",
      label, which, bad_count, end - begin, first_bad,
      distance_description(first_bad), image[first_bad], GuardByte(first_bad)));
}

}  // namespace

uint8_t GuardByte(int64_t offset) {
  // Cheap position-dependent poison: a guard block relocated by any non-zero
  // amount no longer matches.
  return static_cast<uint8_t>(0xA5u ^ (static_cast<uint64_t>(offset) * 31u));
}

absl::Status WriteDeviceBytes(se::Stream& stream, se::DeviceAddressBase buffer,
                              absl::Span<const uint8_t> bytes) {
  RETURN_IF_ERROR(stream.Memcpy(&buffer, bytes.data(), bytes.size()));
  return stream.BlockHostUntilDone();
}

absl::StatusOr<std::vector<uint8_t>> ReadDeviceBytes(
    se::Stream& stream, se::DeviceAddressBase buffer, int64_t num_bytes) {
  std::vector<uint8_t> bytes(num_bytes);
  RETURN_IF_ERROR(stream.Memcpy(bytes.data(), buffer, num_bytes));
  RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return bytes;
}

absl::Status WriteGuardedBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                const GuardedRegion& region,
                                absl::Span<const uint16_t> payload_words) {
  const int64_t payload_bytes =
      static_cast<int64_t>(payload_words.size()) * sizeof(uint16_t);
  if (payload_bytes != region.payload_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "payload_words covers %d bytes but the region declares %d",
        payload_bytes, region.payload_bytes));
  }

  std::vector<uint8_t> image(region.total_bytes());
  for (int64_t offset = 0; offset < region.payload_offset(); ++offset) {
    image[offset] = GuardByte(offset);
  }
  std::memcpy(image.data() + region.payload_offset(), payload_words.data(),
              payload_bytes);
  for (int64_t offset = region.payload_offset() + payload_bytes;
       offset < region.total_bytes(); ++offset) {
    image[offset] = GuardByte(offset);
  }
  return WriteDeviceBytes(stream, buffer, image);
}

absl::Status WriteGuardedBufferFilled(se::Stream& stream,
                                      se::DeviceAddressBase buffer,
                                      const GuardedRegion& region,
                                      uint16_t fill_word) {
  std::vector<uint16_t> payload(region.payload_bytes / sizeof(uint16_t),
                                fill_word);
  return WriteGuardedBuffer(stream, buffer, region, payload);
}

absl::StatusOr<std::vector<uint8_t>> ReadGuardedBuffer(
    se::Stream& stream, se::DeviceAddressBase buffer,
    const GuardedRegion& region) {
  return ReadDeviceBytes(stream, buffer, region.total_bytes());
}

absl::Status CheckGuards(absl::Span<const uint8_t> image,
                         const GuardedRegion& region, absl::string_view label) {
  if (static_cast<int64_t>(image.size()) != region.total_bytes()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: image is %d bytes, region declares %d", label,
                        image.size(), region.total_bytes()));
  }
  const int64_t payload_begin = region.payload_offset();
  const int64_t payload_end = payload_begin + region.payload_bytes;
  RETURN_IF_ERROR(CheckRegionUntouched(
      image, 0, payload_begin, label, "leading", [&](int64_t offset) {
        return absl::StrFormat("%d bytes before the payload",
                               payload_begin - offset);
      }));
  return CheckRegionUntouched(
      image, payload_end, region.total_bytes(), label, "trailing",
      [&](int64_t offset) {
        return absl::StrFormat("%d bytes past the end of the payload",
                               offset - payload_end);
      });
}

absl::Status CheckPayload(absl::Span<const uint8_t> image,
                          const GuardedRegion& region,
                          absl::Span<const uint16_t> expected_words,
                          absl::string_view label) {
  const int64_t expected_bytes =
      static_cast<int64_t>(expected_words.size()) * sizeof(uint16_t);
  if (expected_bytes != region.payload_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: expected_words covers %d bytes but the region declares %d", label,
        expected_bytes, region.payload_bytes));
  }
  return CheckPayloadGenerated(
      image, region, [&](int64_t i) { return expected_words[i]; }, label);
}

absl::Status CheckPayloadGenerated(
    absl::Span<const uint8_t> image, const GuardedRegion& region,
    absl::FunctionRef<uint16_t(int64_t)> expected_word_at,
    absl::string_view label) {
  if (static_cast<int64_t>(image.size()) != region.total_bytes()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: image is %d bytes, region declares %d", label,
                        image.size(), region.total_bytes()));
  }

  const uint16_t* actual = reinterpret_cast<const uint16_t*>(
      image.data() + region.payload_offset());
  const int64_t num_words = region.payload_bytes / sizeof(uint16_t);

  int64_t mismatches = 0;
  std::string detail;
  for (int64_t i = 0; i < num_words; ++i) {
    const uint16_t expected = expected_word_at(i);
    if (actual[i] == expected) {
      continue;
    }
    ++mismatches;
    if (mismatches <= kMaxReportedMismatches) {
      absl::StrAppendFormat(&detail, "\n  word %d: got %s, expected %s", i,
                            DescribeWord(actual[i]), DescribeWord(expected));
    }
  }
  if (mismatches == 0) {
    return absl::OkStatus();
  }
  if (mismatches > kMaxReportedMismatches) {
    absl::StrAppendFormat(&detail, "\n  ... and %d more",
                          mismatches - kMaxReportedMismatches);
  }
  return absl::DataLossError(
      absl::StrFormat("%s: %d of %d payload words are wrong.%s", label,
                      mismatches, num_words, detail));
}

}  // namespace xla::gpu::rccl_benchmark
