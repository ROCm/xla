#!/usr/bin/env python3
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
# ============================================================================
"""The ROCm stacks the nightly XLA UT and JAX jobs test.

rocm_xla_ci_nightly.yml runs this once and passes `stacks` on to
rocm_xla_ci_jax.yml, so both jobs test the same builds, the TheRock nightly
included. rocm_xla_ci_jax.yml runs it itself only when it is started on its
own, by a PR label, with no stacks passed in.

A stack names one ROCm build twice: XLA builds against its tarball
(`distro_url`, `distro_hash`), JAX installs it with pip (`index_url`,
`version`).

Writes two single-line JSON lists to $GITHUB_OUTPUT (stdout when unset):
  stacks     the stacks, for the JAX job
  xla_modes  the XLA UT matrix: every stack on the default RBE cluster, plus
             the BuildBarn variant of BUILDBARN_STACK
"""

import json
import os
import subprocess
import sys

NIGHTLY_INDEX_URL = "https://nightly.repo.amd.com/rocm/whl-next/"
# The release line whose nightlies are tested. The nightly tarball URL has
# changed between release lines before, so moving to the next line is a
# deliberate edit here rather than whatever the index lists newest.
NIGHTLY_LINE = "10.3"

RUNNER = "amd-do-linux-xla-gpu-gfx950-dpx-4"
BUILDBARN_STACK = "therock_release_10.2"


def detect_nightly_version():
  script = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "detect-therock-version.sh"
  )
  env = dict(
      os.environ,
      GPU_TARGET="device-all",
      THEROCK_INDEX_URL=NIGHTLY_INDEX_URL,
      ROCM_SDK_VERSION=NIGHTLY_LINE,
  )
  env.pop("GITHUB_OUTPUT", None)
  result = subprocess.run(
      ["bash", script], env=env, check=True, capture_output=True, text=True
  )
  sys.stderr.write(result.stderr)
  for line in result.stdout.splitlines():
    key, _, value = line.partition("=")
    if key == "therock_version" and value:
      return value
  sys.exit(f"{script} printed no therock_version:\n{result.stdout}")


def make_stacks(nightly):
  return [
      {
          "name": "therock_lts_release_7.14.1",
          "distro_url": "https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-multiarch-7.14.1.tar.gz",
          "distro_hash": "no_hash",
          "index_url": "https://repo.amd.com/rocm/whl-multi-arch/rocm_sdk_core-7.14.1-py3-none-linux_x86_64.whl",
          "version": "7.14.1",
      },
      {
          "name": "therock_release_10.2",
          "distro_url": "https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-10.2.0.tar.gz",
          "distro_hash": "6ac0b54de1a410b9727c37607c4c2cb6b4795c3db0a9b01393bccc8d8d01e87b",
          "index_url": "https://stable.repo.amd.com/rocm/whl-next/",
          "version": "10.2.0",
      },
      {
          "name": "therock_nightly",
          "distro_url": f"https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-{nightly}.tar.gz",
          # Resolved fresh every run, so there is no digest to pin; `version`
          # says which build this was.
          "distro_hash": "no_hash",
          "index_url": NIGHTLY_INDEX_URL,
          "version": nightly,
      },
  ]


def make_xla_modes(stacks):
  modes = [dict(s, runner=RUNNER, rbe_config="rocm_rbe") for s in stacks]
  buildbarn = {s["name"]: s for s in stacks}[BUILDBARN_STACK]
  modes.append(
      dict(
          buildbarn,
          name=f"{BUILDBARN_STACK}_bb",
          runner=RUNNER,
          rbe_config="rocm_rbe_dev",
      )
  )
  return modes


def main():
  stacks = make_stacks(detect_nightly_version())
  outputs = {"stacks": stacks, "xla_modes": make_xla_modes(stacks)}
  for mode in outputs["xla_modes"]:
    print(
        f"{mode['name']:28s} {mode['version']:18s} {mode['rbe_config']}",
        file=sys.stderr,
    )
  lines = "".join(
      f"{key}={json.dumps(value, separators=(',', ':'))}\n"
      for key, value in outputs.items()
  )
  path = os.environ.get("GITHUB_OUTPUT")
  if path:
    with open(path, "a") as f:
      f.write(lines)
  else:
    sys.stdout.write(lines)


if __name__ == "__main__":
  main()
