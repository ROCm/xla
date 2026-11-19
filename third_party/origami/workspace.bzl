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

"""Origami, AMD's analytic GEMM cost model (MIT).

Built from source rather than linked from the ROCm install: ROCm ships
liborigami.so as a private runtime dependency of hipBLASLt, with no headers and
no presence guarantee across installs.

Origami lives in the ROCm monorepo under shared/origami, so strip_prefix selects
that subtree out of the full rocm-libraries tarball.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Bump deliberately: the cost model's predictions change with it.
ORIGAMI_COMMIT = "0d9c1802af3f78cc348ed0e34c1dfabcd5b427a0"
ORIGAMI_SHA256 = ""

def repo():
    tf_http_archive(
        name = "origami",
        build_file = "//third_party/origami:origami.BUILD",
        sha256 = ORIGAMI_SHA256,
        strip_prefix = "rocm-libraries-{c}/shared/origami".format(c = ORIGAMI_COMMIT),
        urls = tf_mirror_urls(
            "https://github.com/ROCm/rocm-libraries/archive/{c}.tar.gz".format(
                c = ORIGAMI_COMMIT,
            ),
        ),
    )
