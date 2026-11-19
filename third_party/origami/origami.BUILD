# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # MIT
)

exports_files(["LICENSE.md"])

# CMake's generate_export_header output. ORIGAMI_STATIC makes every macro empty,
# which is what we want for static linkage into XLA.
genrule(
    name = "export_header",
    outs = ["include/origami/origami_export.h"],
    cmd = """cat > $@ <<'EOF'
#ifndef ORIGAMI_EXPORT_H
#define ORIGAMI_EXPORT_H

#define ORIGAMI_EXPORT
#define ORIGAMI_NO_EXPORT
#define ORIGAMI_DEPRECATED __attribute__((__deprecated__))
#define ORIGAMI_DEPRECATED_EXPORT ORIGAMI_DEPRECATED
#define ORIGAMI_DEPRECATED_NO_EXPORT ORIGAMI_DEPRECATED

#endif  // ORIGAMI_EXPORT_H
EOF
""",
)

cc_library(
    name = "origami",
    srcs = glob([
        "src/origami/*.cpp",
        "src/simulator/tensilelite/*.cpp",
    ]),
    hdrs = glob([
        "include/origami/**/*.hpp",
    ]) + [":export_header"],
    copts = [
        "-std=c++17",
        "-fexceptions",
        # Not our code; its warnings are not actionable here.
        "-w",
    ],
    defines = ["ORIGAMI_STATIC"],
    includes = ["include"],
    # hardware.hpp includes <hip/hip_runtime.h>; hardware.cpp calls
    # hipGetDeviceProperties, though XLA only uses the GPU-less arch path.
    deps = ["@local_config_rocm//rocm:hip"],
)
