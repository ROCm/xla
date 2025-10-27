# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import pathlib
import argparse

from transformer_engine import transformer_engine_jax
from xla.tools.multihost_hlo_runner import py_hlo_multihost_runner

def _register_transformer_engine_custom_calls():
  for name, value in transformer_engine_jax.registrations().items():
    try:
      py_hlo_multihost_runner.register_custom_call_target(
          name, value, platform="ROCM", api_version=1
      )
    except:
      pass

def main():
    parser = argparse.ArgumentParser(description="Run the specified hlo_file.")
    parser.add_argument("hlo_file", help="Path to the input file")
    parser.add_argument("-o", help="Path to output literal file")

    args = parser.parse_args()
  
    _register_transformer_engine_custom_calls()
    
    config = py_hlo_multihost_runner.PyHloRunnerConfig()
    config.input_format = py_hlo_multihost_runner.InputFormat.Text
    config.hlo_argument_mode = (
        py_hlo_multihost_runner.ModuleArgumentMode.UseRandomInputs
    )
    if args.o:
      config.dump_output_literal_to = args.o
    
    os.environ['NVTE_FUSED_ATTN'] = '1'
    os.environ['NVTE_FUSED_ATTN_CK'] = '1'
    os.environ['NVTE_CK_USES_FWD_V3'] = '1'
    os.environ['NVTE_CK_USES_BWD_V3'] = '1'
    os.environ['NVTE_CK_IS_V3_ATOMIC_FP32'] = '1'
    os.environ['NVTE_CK_HOW_V3_BF16_CVT'] = '1'
    
    py_hlo_multihost_runner.RunHloFiles([args.hlo_file], config)


if __name__ == "__main__":
  main()
