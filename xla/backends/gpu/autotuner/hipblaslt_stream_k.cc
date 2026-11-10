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

#include "xla/backends/gpu/autotuner/hipblaslt_stream_k.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace xla {
namespace gpu {

std::optional<int> HipblasLtStreamKMode(absl::string_view name) {
  // Tokens are "_"-separated; "SK" followed only by digits is the StreamK
  // parameter (SKFTR0, SKWS0, SKXCCM0, ... are other stream-K settings).
  for (absl::string_view token : absl::StrSplit(name, '_')) {
    if (token.size() < 3 || !absl::StartsWith(token, "SK")) {
      continue;
    }
    absl::string_view digits = token.substr(2);
    bool all_digits = true;
    for (char c : digits) {
      all_digits &= absl::ascii_isdigit(static_cast<unsigned char>(c)) != 0;
    }
    int mode = 0;
    if (all_digits && absl::SimpleAtoi(digits, &mode)) {
      return mode;
    }
  }
  return std::nullopt;
}

std::vector<int> SelectNonStreamKSolutions(
    absl::Span<const std::string> names) {
  std::vector<int> keep;
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    std::optional<int> mode = HipblasLtStreamKMode(names[i]);
    if (!mode.has_value() || *mode == 0) {
      keep.push_back(i);
    }
  }
  if (keep.empty()) {
    keep.reserve(names.size());
    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
      keep.push_back(i);
    }
  }
  return keep;
}

}  // namespace gpu
}  // namespace xla
