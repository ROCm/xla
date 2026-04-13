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

"""Wrapper scripts for ROCm feature-based toolchain.

The common wrappers (ar, clang, ld, idler) are simple pass-through scripts
identical to those in rules_ml_toolchain. Only hipcc_wrapper is ROCm-specific.
"""

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all",
    srcs = [
        "ar",
        "clang",
        "clang++",
        "hipcc_wrapper",
        "idler",
        "ld",
    ],
)

exports_files([
    "ar",
    "clang",
    "clang++",
    "hipcc_wrapper",
    "idler",
    "ld",
])
