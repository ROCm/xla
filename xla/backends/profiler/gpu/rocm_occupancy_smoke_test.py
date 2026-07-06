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
"""Python smoke tests for ROCm theoretical occupancy in XPlane profiles.

These tests run real JAX kernels on AMD GPU hardware, capture a profiler trace
via jax.profiler.trace, load the resulting XSpace proto, and assert that the
theoretical occupancy stats (theoretical_occupancy_pct, occupancy_min_grid_size,
occupancy_suggested_block_size) are present and sane in kernel events.

Usage:
  python3 rocm_occupancy_smoke_test.py          # run all tests
  python3 rocm_occupancy_smoke_test.py -v       # verbose

Requirements:
  - AMD GPU with ROCm driver
  - JAX built with this branch's rocprofiler-sdk occupancy changes
  - absl-py  (pip install absl-py)
"""

import glob
import os
import tempfile
import unittest

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.profiler as jprofiler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_trace(fn, *, num_warmup=1):
  """Warm up fn, then capture one profiler trace and return ProfileData."""
  # Warmup: compile the computation and let any lazy initialisation finish.
  for _ in range(num_warmup):
    fn()

  with tempfile.TemporaryDirectory() as trace_dir:
    with jprofiler.trace(trace_dir):
      fn()

    pb_files = glob.glob(
        os.path.join(trace_dir, '**', '*.xplane.pb'), recursive=True
    )
    if not pb_files:
      raise FileNotFoundError(
          f'No .xplane.pb found under {trace_dir}; '
          'JAX profiler may not have written trace data.'
      )
    return jprofiler.ProfileData.from_file(pb_files[0])


def _gpu_kernel_events(profile_data):
  """Yield (plane_name, event) for every kernel event on any GPU plane."""
  for plane in profile_data.planes:
    if not plane.name.startswith('/device:GPU:'):
      continue
    for line in plane.lines:
      for event in line.events:
        yield plane.name, event


def _occupancy_stats(event):
  """Return the occupancy-related stats from an event as a dict."""
  stats = dict(event.stats)
  return {
      k: v
      for k, v in stats.items()
      if k in (
          'theoretical_occupancy_pct',
          'occupancy_min_grid_size',
          'occupancy_suggested_block_size',
          'kernel_details',
      )
  }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class RocmOccupancySmokeTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    devices = jax.devices('gpu')
    if not devices:
      raise unittest.SkipTest('No AMD GPU devices found')
    cls.device = devices[0]

  # -- Test 1: basic matmul --------------------------------------------------

  def test_matmul_kernel_has_occupancy_stats(self):
    """A large matmul should produce a kernel event with occupancy stats."""
    def workload():
      a = jnp.ones((1024, 1024), dtype=jnp.float32, device=self.device)
      b = jnp.ones((1024, 1024), dtype=jnp.float32, device=self.device)
      c = a @ b
      c.block_until_ready()

    profile = _run_trace(workload)

    found_occ = False
    found_kernel_with_details = False

    for plane_name, event in _gpu_kernel_events(profile):
      occ = _occupancy_stats(event)

      if 'kernel_details' in occ:
        found_kernel_with_details = True

      if 'theoretical_occupancy_pct' in occ:
        found_occ = True
        pct = occ['theoretical_occupancy_pct']
        self.assertIsInstance(pct, float,
            f'theoretical_occupancy_pct must be float, got {type(pct)}')
        self.assertGreater(pct, 0.0,
            f'Occupancy must be > 0 for a running kernel; got {pct}')
        self.assertLessEqual(pct, 100.0,
            f'Occupancy cannot exceed 100%; got {pct}')

        if 'occupancy_min_grid_size' in occ:
          self.assertGreater(occ['occupancy_min_grid_size'], 0,
              'occupancy_min_grid_size must be positive')
        if 'occupancy_suggested_block_size' in occ:
          self.assertGreater(occ['occupancy_suggested_block_size'], 0,
              'occupancy_suggested_block_size must be positive')

    self.assertTrue(found_kernel_with_details,
        'No GPU kernel event with kernel_details found in the trace. '
        'The profiler may not be capturing kernel dispatches.')
    self.assertTrue(found_occ,
        'No GPU kernel event carried theoretical_occupancy_pct. '
        'Check that:\n'
        '  1. KernelEvent() in rocm_tracer.cc populates func_ptr from '
        'kernel_info_[kinfo.kernel_id].data.kernel_object\n'
        '  2. GetDeviceCapabilities() stores max_waves_per_cu_ and '
        'wave_front_size_ from the rocprofiler agent\n'
        '  3. CreateXEvent() calls hipFuncGetAttributes + GetOccupancy '
        'and emits kTheoreticalOccupancyPct')

  # -- Test 2: kernel_details string contains occ_pct ------------------------

  def test_kernel_details_string_contains_occ_pct(self):
    """The kernel_details stat string must embed a non-zero occ_pct field."""
    def workload():
      x = jnp.ones((512, 512), dtype=jnp.float32, device=self.device)
      y = jnp.sin(x)
      y.block_until_ready()

    profile = _run_trace(workload)

    found_nonzero = False
    for _, event in _gpu_kernel_events(profile):
      occ = _occupancy_stats(event)
      details = occ.get('kernel_details', '')
      if not isinstance(details, str):
        continue
      if 'occ_pct:' not in details:
        continue
      # Parse the value after "occ_pct:"
      try:
        tail = details.split('occ_pct:', 1)[1]
        pct_str = tail.split()[0]
        pct = float(pct_str)
      except (IndexError, ValueError):
        continue
      if pct > 0.0:
        found_nonzero = True
        break

    self.assertTrue(found_nonzero,
        'No kernel event had a non-zero occ_pct in its kernel_details string. '
        'The occ_pct field is always written; if it is always 0 then '
        'occupancy computation is failing silently.')

  # -- Test 3: all three occupancy stat keys present -------------------------

  def test_all_three_occupancy_stats_present(self):
    """All three occupancy stats must appear together on the same event."""
    def workload():
      a = jnp.ones((2048, 512), dtype=jnp.bfloat16, device=self.device)
      b = jnp.ones((512, 2048), dtype=jnp.bfloat16, device=self.device)
      c = (a @ b).astype(jnp.float32)
      c.block_until_ready()

    profile = _run_trace(workload)

    found_all_three = False
    for _, event in _gpu_kernel_events(profile):
      occ = _occupancy_stats(event)
      if all(k in occ for k in (
          'theoretical_occupancy_pct',
          'occupancy_min_grid_size',
          'occupancy_suggested_block_size',
      )):
        found_all_three = True
        break

    self.assertTrue(found_all_three,
        'No single kernel event carried all three occupancy stats '
        '(theoretical_occupancy_pct, occupancy_min_grid_size, '
        'occupancy_suggested_block_size). They should all be written '
        'together when hipFuncGetAttributes succeeds.')

  # -- Test 4: reduction kernel ----------------------------------------------

  def test_reduction_kernel_has_occupancy(self):
    """A reduction (different kernel shape) should also carry occupancy."""
    def workload():
      x = jnp.ones((1024 * 1024,), dtype=jnp.float32, device=self.device)
      s = jnp.sum(x)
      s.block_until_ready()

    profile = _run_trace(workload)

    any_occ = any(
        'theoretical_occupancy_pct' in dict(ev.stats)
        for _, ev in _gpu_kernel_events(profile)
    )
    self.assertTrue(any_occ,
        'Reduction kernel did not produce a theoretical_occupancy_pct stat.')

  # -- Test 5: occupancy value is stable across two traces -------------------

  def test_occupancy_is_deterministic(self):
    """Same workload run twice should produce the same occupancy value."""
    def workload():
      a = jnp.ones((512, 512), dtype=jnp.float32, device=self.device)
      b = (a @ a)
      b.block_until_ready()

    def first_occ_pct(profile):
      for _, ev in _gpu_kernel_events(profile):
        s = dict(ev.stats)
        if 'theoretical_occupancy_pct' in s:
          return s['theoretical_occupancy_pct']
      return None

    p1 = _run_trace(workload)
    p2 = _run_trace(workload)

    v1 = first_occ_pct(p1)
    v2 = first_occ_pct(p2)

    if v1 is None or v2 is None:
      self.skipTest(
          'No occupancy stat found — prerequisite check covered by '
          'test_matmul_kernel_has_occupancy_stats')

    self.assertAlmostEqual(v1, v2, places=6,
        msg=f'Occupancy should be deterministic: {v1} vs {v2}')


if __name__ == '__main__':
  absltest.main()
