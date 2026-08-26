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

#include "xla/backends/gpu/rccl_benchmark/common/data_pattern.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"

namespace xla::gpu::rccl_benchmark {
namespace {

// Folds all bytes of the element index into eight bits. A plain `index & 0xFF`
// would make any shift by a multiple of 256 elements invisible; xor-folding the
// higher bytes back in breaks that aliasing for the shifts a misrouted work
// descriptor is likely to produce.
uint8_t PositionTag(int64_t index) {
  uint64_t bits = static_cast<uint64_t>(index);
  bits ^= bits >> 8;
  bits ^= bits >> 16;
  bits ^= bits >> 32;
  return static_cast<uint8_t>(bits & 0xFF);
}

}  // namespace

uint16_t PatternWord(int buffer_id, int rank, int64_t index) {
  const uint16_t buffer_bits = static_cast<uint16_t>((buffer_id & 0x7) << 12);
  const uint16_t rank_bits = static_cast<uint16_t>((rank & 0x7) << 8);
  return static_cast<uint16_t>(buffer_bits | rank_bits | PositionTag(index));
}

std::vector<uint16_t> MakeSourcePattern(int buffer_id, int rank,
                                        int64_t num_elements) {
  std::vector<uint16_t> words(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    words[i] = PatternWord(buffer_id, rank, i);
  }
  return words;
}

std::vector<uint16_t> MakeGatheredPattern(int buffer_id, int num_ranks,
                                          int64_t elements_per_rank) {
  std::vector<uint16_t> words(num_ranks * elements_per_rank);
  int64_t out = 0;
  for (int rank = 0; rank < num_ranks; ++rank) {
    for (int64_t i = 0; i < elements_per_rank; ++i) {
      words[out++] = PatternWord(buffer_id, rank, i);
    }
  }
  return words;
}

uint16_t UnwrittenPayloadWord() {
  // Bit 15 is set, which PatternWord() never produces because buffer ids are
  // masked to 0..7.
  return 0xBEEF;
}

std::string DescribeWord(uint16_t word) {
  if (word == UnwrittenPayloadWord()) {
    return "<unwritten>";
  }
  if ((word & 0x8000) != 0) {
    return absl::StrFormat("<foreign 0x%04x>", word);
  }
  return absl::StrFormat("buffer=%d rank=%d tag=0x%02x", (word >> 12) & 0x7,
                         (word >> 8) & 0x7, word & 0xFF);
}

}  // namespace xla::gpu::rccl_benchmark
