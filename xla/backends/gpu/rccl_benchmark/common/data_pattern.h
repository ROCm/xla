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

#ifndef XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_DATA_PATTERN_H_
#define XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_DATA_PATTERN_H_

#include <cstdint>
#include <string>
#include <vector>

namespace xla::gpu::rccl_benchmark {

// Deterministic 16-bit payload words for collectives under test.
//
// Every word encodes which buffer and which rank produced it. That matters for
// grouped collectives specifically: the most likely corruption mode when
// several operations share one RCCL group is that a buffer is served by the
// wrong work descriptor, so two same-shaped AllGathers can swap or share their
// data. A payload that only depended on the element index would compare equal
// after such a swap and the test would pass while the data was wrong.
//
// Word layout (bit 15 is the most significant bit):
//
//   bits 15..12  buffer id   (0..7)
//   bits 11.. 8  rank        (0..7)
//   bits  7.. 0  position tag, folded from the element index
//
// Restricting buffer id and rank to 0..7 keeps every word <= 0x77FF, which is
// never a bfloat16 infinity or NaN. The words are compared bit-for-bit, so this
// is not strictly required, but it keeps the payload printable as ordinary
// finite values when a case is debugged by hand.
inline constexpr int kMaxPatternBufferId = 8;
inline constexpr int kMaxPatternRank = 8;

// Returns the payload word for `index` of `buffer_id` contributed by `rank`.
uint16_t PatternWord(int buffer_id, int rank, int64_t index);

// Returns `num_elements` payload words as they are written by `rank` into the
// source buffer of `buffer_id`.
std::vector<uint16_t> MakeSourcePattern(int buffer_id, int rank,
                                        int64_t num_elements);

// Returns the words an AllGather over `num_ranks` ranks must produce in the
// destination buffer of `buffer_id`: rank 0's contribution followed by rank 1's
// and so on.
std::vector<uint16_t> MakeGatheredPattern(int buffer_id, int num_ranks,
                                          int64_t elements_per_rank);

// Returns a filler word that no PatternWord() can produce, used to poison a
// destination buffer before the collective runs so that untouched output is
// distinguishable from correctly written output.
uint16_t UnwrittenPayloadWord();

// Renders `word` as "buffer=N rank=M tag=0xTT", or as "<unwritten>" for the
// poison word. Used in failure messages so a mismatch immediately names the
// buffer and rank the data actually came from.
std::string DescribeWord(uint16_t word);

}  // namespace xla::gpu::rccl_benchmark

#endif  // XLA_BACKENDS_GPU_RCCL_BENCHMARK_COMMON_DATA_PATTERN_H_
