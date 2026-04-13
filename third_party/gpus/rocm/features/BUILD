# Copyright 2025 The OpenXLA Authors.
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
# ==============================================================================

"""ROCm-specific toolchain features.

This file defines only ROCm-specific features. Generic features like
sysroot, PIC, C++ standard, etc. are reused from rules_ml_toolchain.
"""

load(
    "@rules_ml_toolchain//third_party/rules_cc_toolchain/features:features.bzl",
    "cc_feature",
    "cc_toolchain_sysroot_feature",
)

package(default_visibility = ["//visibility:public"])

# =============================================================================
# ROCm-specific features
# =============================================================================

# ROCm-specific preprocessor definitions
cc_feature(
    name = "rocm_defines",
    compiler_flags = [
        "-DTENSORFLOW_USE_ROCM=1",
        "-D__HIP_PLATFORM_AMD__",
        "-DEIGEN_USE_HIP",
        "-DUSE_ROCM",
    ],
    enabled = True,
)

# Hermetic ROCm includes - prevents using system HIP headers
cc_feature(
    name = "rocm_hermetic",
    compiler_flags = ["-nohipinc"],
    enabled = True,
)

# Warning settings specific to ROCm compilation
cc_feature(
    name = "rocm_warnings",
    compiler_flags = [
        "-Wno-unused-result",
    ],
    enabled = True,
)

# =============================================================================
# Generic toolchain features (defined locally to avoid visibility issues)
# =============================================================================

# Hermetic includes - no system headers
cc_feature(
    name = "hermetic_libraries",
    cc_flags = ["-nostdinc++"],
    compiler_flags = ["-nostdinc"],
    enabled = True,
    linker_flags = [
        "-nostdlib",
        "-nodefaultlibs",
    ],
)

# Standard warnings
cc_feature(
    name = "warnings",
    compiler_flags = [
        "-Wall",
        "-fcolor-diagnostics",
    ],
    enabled = True,
)

# Reproducible builds - avoid absolute paths in output
cc_feature(
    name = "reproducible",
    compiler_flags = [
        "-no-canonical-prefixes",
    ],
    enabled = True,
)

# C++17 standard
cc_feature(
    name = "c++17",
    cc_flags = [
        "-std=c++17",
    ],
    provides = ["cc_standard"],
)

# Use LLD linker
cc_feature(
    name = "use_lld",
    enabled = True,
    linker_flags = [
        "-fuse-ld=lld",
    ],
)

# PIC support
cc_feature(
    name = "supports_pic",
    enabled = False,
)

cc_feature(
    name = "position_independent_code",
    compiler_flags = [
        "-fPIC",
    ],
    enabled = True,
    expand_if_available = "pic",
    implies = [
        ":supports_pic",
    ],
)

# Sysroot configuration for Linux x86_64
cc_toolchain_sysroot_feature(
    name = "sysroot_linux_x86_64",
    enabled = True,
    sysroot = "@sysroot_linux_x86_64//:sysroot",
    target = "x86_64-linux-gnu",
)
