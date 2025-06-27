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

#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace pjrt {

namespace {
absl::StatusOr<xla::CompileOptions> ParseCompileOptions(
    absl::string_view options_str) {
  xla::CompileOptionsProto options_proto;
  if (!options_proto.ParseFromArray(options_str.data(), options_str.size())) {
    return absl::InvalidArgumentError(
        "PJRT_Client_Compile: failed to deserialize CompileOptionsProto");
  }
  return xla::CompileOptions::FromProto(options_proto);
}

}  // namespace

std::vector<std::string> ConvertCharBufferToCppStrings(
    const char** char_buffers, const size_t* char_buffer_sizes,
    size_t num_strings) {
  assert(char_buffers != nullptr);

  std::vector<std::string> cpp_strings;
  cpp_strings.reserve(num_strings);
  for (size_t i = 0; i < num_strings; ++i) {
    cpp_strings.push_back(std::string(char_buffers[i], char_buffer_sizes[i]));
  }

  return cpp_strings;
}

void ConvertCppStringsToCharBuffer(const std::vector<std::string>& strings,
                                   const char*** char_buffers,
                                   const size_t** char_buffer_sizes,
                                   size_t* num_strings) {
  *num_strings = strings.size();
  const char** buffer_pointers = new const char*[*num_strings];
  size_t* buffer_sizes = new size_t[*num_strings];

  for (size_t i = 0; i < *num_strings; ++i) {
    size_t string_data_size = strings[i].size();
    char* string_buffer = new char[string_data_size];
    memcpy(string_buffer, strings[i].data(), string_data_size);

    buffer_pointers[i] = string_buffer;
    buffer_sizes[i] = string_data_size;
  }
  *char_buffers = buffer_pointers;
  *char_buffer_sizes = buffer_sizes;
}

static PJRT_Error* PJRT_PhaseCompile_Run_Phase(
    PJRT_PhaseCompile_Run_Phase_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Run_Phase_Args",
      PJRT_PhaseCompile_Run_Phase_Args_STRUCT_SIZE, args->struct_size));

  // Extract the phases to run from the input buffer.
  std::vector<std::string> phases_to_run = ConvertCharBufferToCppStrings(
      args->phases_to_run, args->phases_to_run_sizes, args->num_phases_to_run);

  // Extract the input programs from the input buffer.
  auto programs_in = ConvertCharBufferToCppStrings(args->input_programs,
                                                   args->input_programs_sizes,
                                                   args->num_input_programs);
  std::vector<xla::PjRtPartialProgramProto> programs_in_protos;
  for (const auto& program_in : programs_in) {
    xla::PjRtPartialProgramProto partial_program;
    partial_program.ParseFromString(program_in);
    programs_in_protos.push_back(partial_program);
  }

  // Parse the compile options.
  PJRT_ASSIGN_OR_RETURN(
      xla::CompileOptions options,
      ParseCompileOptions(absl::string_view(args->compile_options,
                                            args->compile_options_size)));

  // Run the partial compile phase.
  if (args->phase_compiler == nullptr) {
    return new PJRT_Error{absl::InternalError(
        "PJRT_PhaseCompile_Run_Phase: phase compiler is null")};
  }
  PJRT_ASSIGN_OR_RETURN(
      absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> programs_out,
      args->phase_compiler->compiler->RunPhases(options, programs_in_protos,
                                                *args->topology->topology,
                                                phases_to_run));
  if (!programs_out.ok()) {
    return new PJRT_Error{programs_out.status()};
  }

  // Combine the output programs into a single output buffer.
  std::vector<std::string> serialized_programs_out;
  serialized_programs_out.reserve(programs_out->size());
  for (const auto& partial_program : *programs_out) {
    serialized_programs_out.push_back(partial_program.SerializeAsString());
  }

  ConvertCppStringsToCharBuffer(serialized_programs_out, &args->output_programs,
                                &args->output_programs_sizes,
                                &args->num_output_programs);

  return nullptr;
}

static PJRT_Error* PJRT_PhaseCompile_Get_Phase_Names(
    PJRT_PhaseCompile_Get_PhaseNames_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Get_Phase_Names_Args",
      PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE, args->struct_size));

  // Get the phase names from the compiler.
  if (args->phase_compiler == nullptr) {
    return new PJRT_Error{absl::InternalError(
        "PJRT_PhaseCompile_Get_Phase_Names: phase compiler is null")};
  }
  PJRT_ASSIGN_OR_RETURN(absl::StatusOr<std::vector<std::string>> phase_names,
                        args->phase_compiler->compiler->GetPhaseNames());
  if (!phase_names.ok()) {
    return new PJRT_Error{phase_names.status()};
  }

  // Copy the phase names to the output buffer.
  ConvertCppStringsToCharBuffer(*phase_names, &args->phase_names,
                                &args->phase_names_sizes,
                                &args->num_phase_names);
  return nullptr;
}

static void PJRT_PhaseCompile_C_Buffers_Destroy(
    PJRT_PhaseCompile_C_Buffers_Destroy_Args* args) {
  assert(args->char_buffers != nullptr);
  assert(args->char_buffer_sizes != nullptr);

  for (size_t i = 0; i < args->num_char_buffers; ++i) {
    delete[] args->char_buffers[i];
  }
  delete[] args->char_buffer_sizes;
  delete[] args->char_buffers;
}

PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next, PJRT_PhaseCompile_Get_Compiler get_compiler,
    PJRT_PhaseCompile_Destroy_Compiler destroy_compiler) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_PhaseCompile_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_PhaseCompile,
          /*next=*/next,
      },
      /*phase_compile_get_compiler=*/get_compiler,
      /*phase_compile_destroy_compiler=*/destroy_compiler,
      /*phase_compile_run_phase=*/PJRT_PhaseCompile_Run_Phase,
      /*phase_compile_get_phase_names=*/
      PJRT_PhaseCompile_Get_Phase_Names,
      /*phase_compile_c_buffers_destroy=*/PJRT_PhaseCompile_C_Buffers_Destroy,
  };
}
}  // namespace pjrt
