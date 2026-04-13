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

This file defines a ROCm toolchain using the feature-based architecture
from rules_ml_toolchain. Paths are resolved dynamically at analysis time
via Bazel labels, rather than being hardcoded.
"""

load("@local_config_rocm//rocm:build_defs.bzl", "rocm_gpu_architectures", "rocm_version_number")
load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load(
    "@rules_ml_toolchain//third_party/rules_cc_toolchain/features:cc_toolchain_import.bzl",
    "cc_toolchain_import",
)
load(
    "@rules_ml_toolchain//third_party/rules_cc_toolchain/features:features.bzl",
    "cc_toolchain_import_feature",
)
load(
    "@rules_ml_toolchain//third_party/rules_cc_toolchain:toolchain_config.bzl",
    "cc_toolchain_config",
)
load("//third_party/gpus/rocm/features:rocm_hipcc_feature.bzl", "rocm_hipcc_feature")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# =============================================================================
# ROCm HIPcc Feature
# =============================================================================
# This feature sets environment variables (HIPCC_PATH, ROCM_PATH, etc.)
# that are read by the hipcc_wrapper script.

rocm_hipcc_feature(
    name = "rocm_hipcc_feature",
    amdgpu_targets = rocm_gpu_architectures(),
    enabled = True,
    host_compiler = "@llvm_linux_x86_64//:clang",
    rocm_toolkit = "@local_config_rocm//rocm:rocm_root",
    version = str(rocm_version_number()),
)

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
        "@sysroot_linux_x86_64//:includes_c",
        "@sysroot_linux_x86_64//:includes_system",
    ],
)

filegroup(
    name = "linker_deps",
    srcs = [
        ":compiler_deps",
        "@llvm_linux_x86_64//:ld",
        "@llvm_linux_x86_64//:distro_libs",
        "@sysroot_linux_x86_64//:startup_libs",
        "@sysroot_linux_x86_64//:syslibs",
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
        ":ar_deps",
        ":compiler_deps",
        ":linker_deps",
    ],
)

filegroup(
    name = "empty",
    srcs = [],
)

# =============================================================================
# Library Imports
# =============================================================================
# Import sysroot libraries using cc_toolchain_import from rules_ml_toolchain.
# This provides -L library paths and -l library flags automatically.

cc_toolchain_import(
    name = "sysroot_imports",
    deps = [
        # LLVM includes must come before sysroot includes for #include_next to work
        # (e.g., clang's limits.h does #include_next <limits.h> to find sysroot's limits.h)
        "@llvm_linux_x86_64//:includes",
        "@llvm_linux_x86_64//:libclang_rt",
        "@sysroot_linux_x86_64//:includes_c",
        "@sysroot_linux_x86_64//:includes_system",
        "@sysroot_linux_x86_64//:syslibs",
    ],
)

cc_toolchain_import(
    name = "startup_libs",
    deps = [
        "@sysroot_linux_x86_64//:startup_libs",
    ],
)

cc_toolchain_import_feature(
    name = "sysroot_imports_feature",
    enabled = True,
    toolchain_import = ":sysroot_imports",
)

cc_toolchain_import_feature(
    name = "startup_libs_feature",
    enabled = True,
    toolchain_import = ":startup_libs",
)

# =============================================================================
# Feature List
# =============================================================================
# Compose features from rules_ml_toolchain with ROCm-specific features.

FEATURES = [
    # Hermetic libraries feature (must come first before imports)
    "//third_party/gpus/rocm/features:hermetic_libraries",

    # Library imports
    ":sysroot_imports_feature",
    ":startup_libs_feature",

    # ROCm-specific features
    ":rocm_hipcc_feature",
    "//third_party/gpus/rocm/features:rocm_defines",
    "//third_party/gpus/rocm/features:rocm_hermetic",
    "//third_party/gpus/rocm/features:rocm_warnings",

    # Generic toolchain features (defined locally in rocm/features)
    "//third_party/gpus/rocm/features:warnings",
    "//third_party/gpus/rocm/features:reproducible",
    "//third_party/gpus/rocm/features:c++17",
    "//third_party/gpus/rocm/features:use_lld",
    "//third_party/gpus/rocm/features:supports_pic",
    "//third_party/gpus/rocm/features:position_independent_code",

    # Sysroot configuration
    "//third_party/gpus/rocm/features:sysroot_linux_x86_64",
]

# =============================================================================
# Toolchain Configuration
# =============================================================================

cc_toolchain_config(
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
        "ar": "wrappers/ar",
        "cpp": "wrappers/clang++",
        "gcc": "wrappers/hipcc_wrapper",  # Use hipcc for ROCm compilation
        "gcov": "wrappers/idler",
        "ld": "wrappers/ld",
        "llvm-cov": "wrappers/idler",
        "nm": "wrappers/idler",
        "objdump": "wrappers/idler",
        "strip": "wrappers/idler",
    },
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
