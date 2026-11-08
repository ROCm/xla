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

// Prints the autotune cache fingerprint (GetHloFingerprint, the file name of a
// per-fusion autotune cache entry) of every fusion and custom call in an HLO
// text dump, e.g. <module>.before_config_assignment.txt. One line per
// instruction: fingerprint, name, then the fused computation's dot (if any)
// on the next tab-separated column.
//
// Usage: autotune_fingerprints <hlo.txt>

#include <cstdio>
#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "xla/backends/autotuner/autotune_fingerprint.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/env.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <hlo.txt>\n", argv[0]);
    return 2;
  }
  std::string text;
  absl::Status s = tsl::ReadFileToString(tsl::Env::Default(), argv[1], &text);
  if (!s.ok()) {
    std::fprintf(stderr, "%s\n", s.ToString().c_str());
    return 1;
  }
  absl::StatusOr<std::unique_ptr<xla::HloModule>> module =
      xla::ParseAndReturnUnverifiedModule(text);
  if (!module.ok()) {
    std::fprintf(stderr, "%s\n", module.status().ToString().c_str());
    return 1;
  }
  for (const xla::HloComputation* comp : (*module)->computations()) {
    for (const xla::HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() != xla::HloOpcode::kFusion &&
          instr->opcode() != xla::HloOpcode::kCustomCall) {
        continue;
      }
      std::string dot;
      if (instr->opcode() == xla::HloOpcode::kFusion) {
        for (const xla::HloInstruction* f :
             instr->fused_instructions_computation()->instructions()) {
          if (f->opcode() == xla::HloOpcode::kDot) {
            xla::HloPrintOptions o = xla::HloPrintOptions::ShortParsable();
            o.set_print_operand_shape(true);
            o.set_print_metadata(false);
            dot = f->ToString(o);
          }
        }
      }
      std::string line = absl::StrCat(
          xla::AutotuneFingerprintToString(xla::GetHloFingerprint(*instr)),
          "\t", instr->name(), "\t", comp->name(), "\t", instr->shape().ToString(true),
          "\t", absl::StrReplaceAll(dot, {{"\t", " "}, {"\n", " "}}));
      std::printf("%s\n", line.c_str());
    }
  }
  return 0;
}
