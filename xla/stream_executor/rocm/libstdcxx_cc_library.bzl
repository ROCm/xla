# Copyright 2026 The OpenXLA Authors.
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
"""Builds a cc_library against libstdc++, whatever the rest of the build uses.

ROCm libraries such as hipBLASLt are built against libstdc++ and have C++ APIs
that pass standard library types (std::vector, std::string, ...). Code built
against libc++ cannot call those APIs. libstdcxx_cc_library builds a target
against libstdc++ so it can call them, and exposes it to the rest of the build
like any cc_library. The target must only expose a C interface: its headers are
included by code built against the other standard library.
"""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

_STDLIB = "@rules_ml_toolchain//common:stdlib"
_STATIC_LIBCXX = "@rules_ml_toolchain//common:static_libcxx"

def _libstdcxx_transition_impl(settings, attr):
    _ignore = (settings, attr)  # @unused
    return {
        _STDLIB: "libstdc++",
        # Link libstdc++ dynamically: the process already loads libstdc++.so.6
        # for the ROCm libraries, and a second, static copy is not needed.
        _STATIC_LIBCXX: False,
    }

_libstdcxx_transition = transition(
    implementation = _libstdcxx_transition_impl,
    inputs = [],
    outputs = [_STDLIB, _STATIC_LIBCXX],
)

def _libstdcxx_cc_library_impl(ctx):
    dep = ctx.attr.dep[0]
    return [
        dep[CcInfo],
        DefaultInfo(files = dep[DefaultInfo].files),
    ]

libstdcxx_cc_library = rule(
    implementation = _libstdcxx_cc_library_impl,
    doc = "Forwards `dep`, a cc_library built against libstdc++.",
    attrs = {
        "dep": attr.label(
            cfg = _libstdcxx_transition,
            mandatory = True,
            providers = [CcInfo],
            doc = "The cc_library to build against libstdc++. Tag it " +
                  "\"manual\" so that wildcard builds do not also build it " +
                  "against the default standard library.",
        ),
    },
    provides = [CcInfo],
)
