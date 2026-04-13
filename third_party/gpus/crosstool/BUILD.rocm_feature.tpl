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

"""Feature-based ROCm toolchain configuration.

This file defines a ROCm toolchain using the feature-based architecture,
similar to how CUDA is implemented in rules_ml_toolchain. This approach
allows paths (sysroot, clang, ROCm) to be resolved dynamically at analysis
time via Bazel labels, rather than being hardcoded at repository rule time.
"""

load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load("//third_party/gpus/rocm/features:rocm_hipcc_feature.bzl", "rocm_hipcc_feature")
load("//third_party/gpus/rocm/features:toolchain_config.bzl", "rocm_toolchain_config")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# =============================================================================
# ROCm HIPcc Feature
# =============================================================================
# This feature sets environment variables (HIPCC_PATH, ROCM_PATH, etc.)
# that are read by the hipcc_wrapper script.

rocm_hipcc_feature(
    name = "rocm_hipcc_feature",
    amdgpu_targets = [%{rocm_amdgpu_targets}],
    enabled = True,
    host_compiler = "@llvm_linux_x86_64//:clang",
    rocm_toolkit = "@local_config_rocm//rocm:rocm_root",
    version = "%{rocm_version}",
)

# =============================================================================
# Wrapper Scripts for Toolchain
# =============================================================================
# Tool paths map to wrapper scripts that read environment variables set by features.

ROCM_TOOLS = {
    "gcc": "//third_party/gpus/crosstool/wrappers:hipcc_wrapper",
    "cpp": "//third_party/gpus/crosstool/wrappers:clang++",
    "ld": "//third_party/gpus/crosstool/wrappers:ld",
    "ar": "//third_party/gpus/crosstool/wrappers:ar",
    "gcov": "//third_party/gpus/crosstool/wrappers:idler",
    "llvm-cov": "//third_party/gpus/crosstool/wrappers:idler",
    "nm": "//third_party/gpus/crosstool/wrappers:idler",
    "objdump": "//third_party/gpus/crosstool/wrappers:idler",
    "strip": "//third_party/gpus/crosstool/wrappers:idler",
}

# =============================================================================
# Filegroups for Toolchain Files
# =============================================================================

filegroup(
    name = "wrappers",
    srcs = ["//third_party/gpus/crosstool/wrappers:all"],
)

filegroup(
    name = "rocm_toolkit_files",
    srcs = ["@local_config_rocm//rocm:toolchain_data"],
)

filegroup(
    name = "compiler_deps",
    srcs = [
        ":wrappers",
        ":rocm_toolkit_files",
        "@llvm_linux_x86_64//:clang",
        "@llvm_linux_x86_64//:clang++",
        "@llvm_linux_x86_64//:includes",
        "@sysroot_linux_x86_64//:sysroot",
    ],
)

filegroup(
    name = "linker_deps",
    srcs = [
        ":compiler_deps",
        "@llvm_linux_x86_64//:ld",
    ],
)

filegroup(
    name = "ar_deps",
    srcs = [
        ":wrappers",
        "@llvm_linux_x86_64//:ar",
    ],
)

filegroup(
    name = "all_deps",
    srcs = [
        ":compiler_deps",
        ":linker_deps",
        ":ar_deps",
    ],
)

filegroup(
    name = "empty",
    srcs = [],
)

# =============================================================================
# Feature List
# =============================================================================
# These features are composed to create the complete toolchain configuration.

FEATURES = [
    # ROCm-specific feature
    ":rocm_hipcc_feature",

    # ROCm defines and compiler settings
    "//third_party/gpus/rocm/features:rocm_defines",
    "//third_party/gpus/rocm/features:hermetic",
    "//third_party/gpus/rocm/features:warnings",
    "//third_party/gpus/rocm/features:c++17",
    "//third_party/gpus/rocm/features:pic",
]

# =============================================================================
# Toolchain Configuration
# =============================================================================

rocm_toolchain_config(
    name = "rocm_feature_toolchain_config",
    archiver = "@llvm_linux_x86_64//:ar",
    c_compiler = "@llvm_linux_x86_64//:clang",
    cc_compiler = "@llvm_linux_x86_64//:clang++",
    compiler_features = FEATURES,
    linker = "@llvm_linux_x86_64//:ld",
    strip_tool = "@llvm_linux_x86_64//:strip",
    target_cpu = "x86_64",
    target_system_name = "local",
    tool_paths = {
        "gcc": "wrappers/hipcc_wrapper",
        "cpp": "wrappers/clang++",
        "ld": "wrappers/ld",
        "ar": "wrappers/ar",
        "gcov": "wrappers/idler",
        "llvm-cov": "wrappers/idler",
        "nm": "wrappers/idler",
        "objdump": "wrappers/idler",
        "strip": "wrappers/idler",
    },
    toolchain_identifier = "rocm_feature_toolchain",
)

cc_toolchain(
    name = "rocm_feature_toolchain",
    all_files = ":all_deps",
    ar_files = ":ar_deps",
    as_files = ":compiler_deps",
    compiler_files = ":compiler_deps",
    dwp_files = ":empty",
    linker_files = ":linker_deps",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":rocm_feature_toolchain_config",
    toolchain_identifier = "rocm_feature_toolchain",
)

# =============================================================================
# Toolchain Registration
# =============================================================================

toolchain(
    name = "toolchain-linux-x86_64-rocm-feature",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    toolchain = ":rocm_feature_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
