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

#include "xla/backends/gpu/autotuner/triton/triton_configs.h"

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/autotuner/triton/embed_default_configs.h"
#include "xla/service/gpu/matmul_utils.h"

namespace xla::gpu {
namespace {

std::vector<TritonGemmConfig> ParseConfig(absl::string_view config_str) {
  TritonGemmConfigsProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(config_str, &proto))
      << config_str;
  std::vector<TritonGemmConfig> configs;
  for (const auto& config_proto : proto.config()) {
    absl::StatusOr<TritonGemmConfig> config =
        TritonGemmConfig::FromProto(config_proto);
    CHECK_OK(config);
    configs.push_back(*config);
  }
  return configs;
};

absl::string_view GetDefaultConfigStr(absl::string_view filename) {
  // embed_files generates get_<stem>() functions where stem is filename without
  // extension
  if (filename == "a100.txtpb") {
    return configs::get_a100();
  } else if (filename == "b200.txtpb") {
    return configs::get_b200();
  } else if (filename == "sm120.txtpb") {
    return configs::get_sm120();
  } else if (filename == "cuda.txtpb") {
    return configs::get_cuda();
  } else if (filename == "rocm.txtpb") {
    return configs::get_rocm();
  } else if (filename == "h100.txtpb") {
    return configs::get_h100();
  } else if (filename == "mi300.txtpb") {
    return configs::get_mi300();
  } else if (filename == "mi350.txtpb") {
    return configs::get_mi350();
  }
  LOG(FATAL) << "Embedded file not found: " << filename;
}

}  // namespace

const std::vector<TritonGemmConfig>& GetTritonConfigsForPlatform(
    TritonConfigsPlatform platform) {
  static const absl::NoDestructor<
      absl::flat_hash_map<TritonConfigsPlatform, std::vector<TritonGemmConfig>>>
      kConfigs({{TritonConfigsPlatform::kAmpere,
                 ParseConfig(GetDefaultConfigStr("a100.txtpb"))},
                {TritonConfigsPlatform::kBlackwell,
                 ParseConfig(GetDefaultConfigStr("b200.txtpb"))},
                {TritonConfigsPlatform::kBlackwellConsumer,
                 ParseConfig(GetDefaultConfigStr("sm120.txtpb"))},
                {TritonConfigsPlatform::kDefaultCuda,
                 ParseConfig(GetDefaultConfigStr("cuda.txtpb"))},
                {TritonConfigsPlatform::kDefaultRocm,
                 ParseConfig(GetDefaultConfigStr("rocm.txtpb"))},
                {TritonConfigsPlatform::kHopper,
                 ParseConfig(GetDefaultConfigStr("h100.txtpb"))},
                {TritonConfigsPlatform::kMI300,
                 ParseConfig(GetDefaultConfigStr("mi300.txtpb"))},
                {TritonConfigsPlatform::kMI350,
                 ParseConfig(GetDefaultConfigStr("mi350.txtpb"))}});
  return kConfigs->at(platform);
}

}  // namespace xla::gpu
