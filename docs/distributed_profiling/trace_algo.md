# Trace Alignment Algorithm Plan

## 0. Clarifications
- The **R timeline** is *not* GPU-only. ROCm tracer timestamps (CUTPI timeline)
  are used for **all** events—host callback lines, device kernels, and memcpy.
- The **S timeline** is the host-side probing/monitoring clock collected via
  snapshot pairs and network probing (alpha/beta windows).
- Once a corrected timestamp `E_r_c` is produced, the neighboring snapshot
  pair used to map `E_r` → `S` may no longer bracket `E_r_c`. We therefore
  re-run the bracketing search after each correction stage.
- GraphCalc offset files contain absolute (unnormalized) wall-clock midpoints
  for node 0 plus per-node `offset_ns`. All XSpace timestamps, by contrast, are
  normalized relative to `start_walltime_ns` / `start_gpu_ns`, so transforms
  must add/subtract these baselines when hopping between representations.

## 1. Executive Summary
| Item | Description |
|------|-------------|
| Goal | Convert per-node traces written on the shared ROCm tracer timeline into a globally aligned view using probe-based offsets. |
| Inputs | XSpace per node (timeline **R**), snapshot pairs `(S,R)`, alpha/beta graph output (`offset_snapshots`). |
| Outputs | Corrected XSpace per node (same schema) ready for merge; diagnostics summarizing interpolation/extrapolation coverage. |
| Implementation | New alignment library + CLI, plus incremental changes to XPlane utilities. |

## 2. File-Level Plan

### 2.1 New Files
| File | Purpose |
|------|---------|
| `xla/backends/profiler/gpu/trace_alignment.h/.cc` | Library implementing interpolation pipelines described below. |
| `docs/trace_alignment_flow.md` | Living spec (this doc + diagrams). |
| `tools/trace_combiner.py` | **Implemented:** Python implementation of the alignment algorithm and combiner logic. |

### 2.2 Existing Files to Update
| File | Change |
|------|--------|
| `xla/tsl/profiler/utils/xplane_utils.cc` | Add helpers to walk events in timestamp order, annotate correction stats, and normalize after correction. |
| `xla/tsl/profiler/utils/xplane_builder.cc` | Provide API for bulk timestamp rewrite with monotonicity guards. |
| `xla/python/profiler.cc` | Expose alignment knob via Python (e.g., `combine_traces(..., apply_alignment=True)`). |
| `third_party/tsl/tsl/profiler/protobuf/xplane.proto` | Optional: embed correction metadata (e.g., `StatType::kTimestampCorrectionNs`). |

## 3. Data Structures
```cpp
struct SnapshotPair {
  // Normalized deltas from the per-node start timestamps (as emitted by the
  // tracer and stored in snapshot files). sys_ns is relative to start_walltime,
  // tracer_ns relative to start_gpu.
  uint64_t sys_ns;
  uint64_t tracer_ns;
};

struct OffsetSample {
  // Parsed directly from GraphCalc JSONL (unnormalized host clock on node 0).
  // For node j we reconstruct M_jk = M_0k + offset_ns.
  uint64_t midpoint_sys_ns;  // M_0k
  double offset_ns;          // O_jk (positive → node j appears ahead of node 0)
  double slope_ppm;          // optional alpha component (drift)
};

struct AlignmentInputs {
  std::vector<SnapshotPair> snapshot_pairs;
  std::vector<OffsetSample> offset_samples;
  tsl::profiler::XSpace* space;  // mutable
  std::string node_id;
  uint64_t start_host_ns;  // S_hi from XSpace metadata
  uint64_t start_dev_ns;   // S_di from XSpace metadata
};

struct AlignmentStats {
  uint64_t events_corrected;
  uint64_t events_clamped_monotonic;
  uint64_t snapshot_extrapolations;
  uint64_t offset_extrapolations;
};
```

## 4. Algorithm (Step-by-step)

### 4.1 Preprocessing
1. Sort snapshots by `tracer_ns` (R).
2. Sort offsets by `midpoint_sys_ns` (S).
3. Precompute segment slopes for faster interpolation:
   ```
   snapshot_slope = (S_n - S_p) / (R_n - R_p)
   offset_slope  = (O_n - O_p) / (M_n - M_p)
   ```
4. Build iterators per XLine to exploit chronological ordering.

### 4.2 Event Processing Loop
For each event timestamp `E_r`:
1. **Locate snapshot segment** such that `R_p ≤ E_r ≤ R_n`.
   - If `E_r < R_0`, extrapolate using first slope; record diagnostic.
2. **Map R→S:** `E_s = S_p + snapshot_slope * (E_r - R_p)`.
3. **Locate offset segment** for `E_s`.
4. **Apply offset:** `O = O_p + offset_slope * (E_s - M_p)`.
   - If slope includes α (ppm), adjust: `O = O_p + slope_ppm/1e6 * (E_s - M_p)`.
   - `E_g = E_s + O`.
5. **Re-evaluate snapshot segment** for `E_g`.
   - Because `E_g` may sit outside `[S_p, S_n]`, repeat step 1 with `E_g`.
6. **Map S→R:** `E_r_new = R_p' + ((E_g - S_p') / snapshot_slope')`.
7. **Monotonicity guard:** If `E_r_new ≤ prev_timestamp_on_line`, set
   `E_r_new = prev + 1`. Log clamp counts.
8. **Write back** new timestamp (and optionally `StatType::kTimestampCorrectionNs`
   = `E_r_new - E_r`).

### 4.3 Handling Neighbor Changes After Correction
- After step 4 (offset applied), we treat `E_g` as a new value on S and **repeat
  segment selection**. This ensures we never rely on stale `R_p/R_n` that no
  longer bound the corrected timestamp. Implementation detail:
  ```cpp
  auto segment = snapshot_index.LocateBySysTime(E_g);
  ```

### 4.4 Complexity
- Each locate is `O(log N)` but we maintain index cursors for amortized `O(1)`.
- Total runtime dominated by event visits: `O(E)` where `E` is event count.

### 4.5 Normalization & Host/Device Transform Rules
Let node *i* have host timeline `H_i`, device timeline `D_i`, and start
timestamps `S_hi` / `S_di` captured in the XSpace metadata. When processing
events we need to hop between normalized values (stored in the trace) and the
absolute times referenced by the offset JSON.

1. **Midpoints from offsets:** The JSONL provides `M_0k = (window_start +
   window_end)/2` for node 0 in absolute wall-clock units. For node *j*, the
   midpoint in the same round is `M_jk = M_0k + O_jk` where `O_jk` is the
   `offset_ns` value. These remain unnormalized until we subtract `S_h0`.

2. **Host events (line on `/host:` plane)**
   - De-normalize: `e_i_raw = e_i_norm + S_hi`.
   - Locate neighboring midpoints `M_ik`, `M_i(k+1)` by bisecting in the
     unnormalized domain. Each midpoint stores the corresponding node 0 value.
   - Interpolate using node 0’s midpoints to obtain `e_0_raw` on `H_0`.
   - Re-normalize: `e_corrected = e_0_raw - S_h0` and write back to the event.

3. **Device events (line on `/device:` plane)**
   - De-normalize on device: `e_di_raw = e_di_norm + S_di`.
   - Use snapshot pairs `(P_hik, P_dik)` to hop from device to host on node *i*
     (R→S). That yields `e_hi_raw`.
   - Apply the host transform above to obtain `e_h0_raw`.
   - Use the same snapshot segment to hop from host back to device (S→R) on node
     *i*, then subtract `S_di` to get the corrected device timestamp.

4. **Why double bisect?** After applying offsets the corrected host time may no
   longer lie inside the original snapshot bracket, so we must re-run the
   segment search before mapping back to the device timeline (step 5 in §4.2).

## 5. CLI + Tooling

### 5.1 New CLI (`xprof_trace_align`)
```
xprof_trace_align \
  --trace /tmp/node0.xspace \
  --snapshot_pairs /tmp/node0.snapshot.jsonl \
  --offsets /tmp/node0.offsets.jsonl \
  --output /tmp/node0.aligned.xspace
```
- Implementation inside `xla/python/profiler_tools/trace_combiner_cli.cc` or pure Python.
- Reuses `AlignmentInputs` API for programmatic use.

### 5.2 Diagnostics File
- `node0.alignment_stats.json` with counts + min/max correction applied.

## 6. Testing
| Test | Description |
|------|-------------|
| `trace_alignment_test.cc` | Synthetic snapshots/offsets; verify mapping math + monotonicity. |
| `trace_alignment_integration_test.cc` | Replay 2-node traces, ensure combined result matches expected skew removal. |
| Regression | Compare corrected vs baseline using golden JSON metrics. |
| **Validation** | Use `check_collective_happens_before_violations` (Python prototype) to verify that aligned collective ops (e.g. AllReduce) do not exhibit impossible timing (e.g. end before start). |

## 7. Open Questions
1. Should slope (ppm) input model be additive (β only) or linear (α, β)?
2. Do we need to preserve the original `E_r` as a stat for tooling?
3. How to handle missing offset windows (skip vs. hold last value)?

## 8. Python Implementation (`tools/trace_combiner.py`)

In parallel with the C++ plan, a production-ready Python tool lives at
`tools/trace_combiner.py`. It currently provides:

- **Alignment + Merge:** Reads per-node XSpaces (with embedded snapshot plane),
  interpolates offsets, and rewrites timestamps before merging.
- **Algorithm Fidelity:** Implements the full R→S→Offset→R remapping logic described in §4, including monotonicity guards.
- **Compatibility Fix:** Implements plane renaming logic to ensure the combined multi-node trace can be ingested by standard XLA tools like `trace_viewer` (which only recognize standard plane prefixes like `/device:GPU:`).
    - Maps all GPU planes to unique global IDs: `/device:GPU:{node*stride + local_id}`.
    - Maps all Host planes to unique "GPU" device tracks: `/device:GPU:Host_Node_{id}` to ensure visualization.
