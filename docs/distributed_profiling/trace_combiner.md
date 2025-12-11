# Trace Combiner Plan

## 0. Executive Summary
| Item | Description |
|------|-------------|
| Goal | Produce a single trace artifact that shows every node’s activity aligned on the corrected ROCm tracer timeline. |
| Scope | Ingest per-node traces (already aligned via `trace_alignment`), snapshot artifacts, and global offsets → emit combined XSpace + diagnostics. |
| Deliverables | CLI tool (`xprof_trace_combine`), reusable library (`trace_combiner.h/.cc` OR Python equivalent), documentation, unit tests. |

## 1. Inputs / Outputs
| Input | Notes |
|-------|-------|
| `trace_files[]` | Post-alignment XSpace files per node (timestamps already corrected on **R**). |
| `snapshot_pairs[]` | For provenance; embedded into metadata plane so viewers know which host clock segments were used. |
| `offset_snapshots` | GraphCalc results per node (e.g. JSONL lines like `{"round_id":3,"window_id":3,...,"offset_ns":-750.8,"drift_ppm":0.4362}`) used for metadata + validation. |
| `Config` | Mapping between node ids, hostnames, GPU ordinals (from `DistributedProfilerContext`). |

| Output | Description |
|--------|-------------|
| `combined.xspace` | Single XSpace containing all host/device planes with node prefixes. |
| `combined.trace.json.gz` | Optional trace viewer export. |
| `combined.metadata.json` | Summary metrics (per-node skew residual, missing data). |

## 2. File-Level Plan

### 2.1 New Files
| File | Description |
|------|-------------|
| `xla/backends/profiler/gpu/trace_combiner.h/.cc` | (Optional) C++ Library that orchestrates ingestion, tagging, merging, and metadata emission. |
| `xla/python/tools/trace_combiner_cli.cc` | (Optional) C++ CLI entrypoint. |
| `tools/trace_combiner.py` | **Implemented:** Python-based implementation for flexibility and ease of integration with analysis scripts. |
| `docs/trace_combiner_flow.md` | Expanded doc referencing this plan. |

### 2.2 Updates to Existing Files
| File | Change |
|------|--------|
| `xla/python/profiler.cc` | Expose `combine_traces()` API for Python consumers (e.g., `jax.profiler.combine_traces`). |
| `xla/tsl/profiler/utils/xplane_utils.cc` | Add helper `TagPlanesWithNodeId(XSpace*, int node_id)` and `AppendMetadataPlane(...)`. |
| `xla/tsl/profiler/rpc/client/save_profile.cc` | Include snapshot + offset artifact paths in metadata so CLI can find them automatically. |
| `DOCUMENTATION_INDEX.md` | Link to new docs. |

## 3. Detailed Pipeline

### Step 1: Input Resolution
1. Accept `--traces`, `--snapshots`, `--offsets`, `--run_metadata`.
2. Validate cardinality (same number of trace/snapshot entries).
3. Cross-check run id + node id across artifacts. Fail if mismatch.

### Step 2: Alignment Verification
- Treat any `*.xplane.pb` artifact as the standard XSpace input (same schema as
  “xspace”). If a node’s trace hasn’t been run through the alignment library
  yet, invoke the `trace_alignment` step here before merging.

### Step 3: Plane Tagging
1. For each XSpace:
   - Prefix plane names: `/node:<node_id>/device:GPU:<ordinal>`.
   - For host planes, prefix `/node:<node_id>/host_threads`.
   - Add plane-level stats:
     ```
     StatType::kNodeId      = node_id
     StatType::kHostname    = "<hostname>"
     StatType::kClockSkewNs = residual computed during alignment
     ```
2. Inject snapshot + offset metadata:
   - Add metadata plane `/node:<id>/clock_metadata` containing snapshot counts,
     sampling period, offset variance.
3. When producing the **combined** trace:
   - Keep node 0's `/host:CPU` plane as-is to preserve standard tooling behavior.
   - Remap host planes for nodes `> 0` to synthetic GPU planes named
     `/device:GPU:<10000 * node_id>` so converters that only recognize device
     planes (e.g., `xspace_to_trace_events`) can render them.
   - Remap GPU planes to `/device:GPU:<node_id * 64 + local_gpu_id>` to maintain
     uniqueness across nodes.

### Step 4: Merge
1. Initialize destination `XSpace combined`.
2. Append planes from each node using `MergePlanes(...)`.
3. Run `SortXSpace` and `NormalizeTimeStamps(&combined, min_timestamp_ns)`.

### Step 5: Diagnostics
- Build JSON summary:
  ```json
  {
    "nodes": [
      { "id": 0, "snapshots": 42, "offset_windows": 10,
        "max_correction_ns": 1200, "missing_snapshots": 0 }
    ]
  }
  ```
- Write to `combined.metadata.json`.

### Step 6: Export
- Serialize combined XSpace.
- Optionally call `ConvertXSpaceToTraceEvents` for `trace.json.gz`.

## 4. CLI UX
```
xprof_trace_combine \
  --run_dir /tmp/prof/run123 \
  --output /tmp/prof/run123/combined.xspace \
  --auto_discover
```
- `--auto_discover` scans `run_dir` for `*.xplane.pb`, `*.snapshot_pairs.pb`,
  `offsets/*.jsonl`.
- Flags to override prefixes, e.g., `--plane_prefix=/clusterA`.

## 5. Testing Strategy
| Test | Scenario |
|------|----------|
| `trace_combiner_test.py` | Feed 2 synthetic nodes; assert plane counts, stats, metadata. |
| Integration | Use real traces from two hosts; verify viewer renders multi-node timeline. |
| Failure | Missing snapshot file → expect descriptive error. |
| Validation | Use `analyze_collective_operations` to check happens-before relations between nodes. |

## 6. Open Questions
1. Should combined trace also include per-node copies of snapshot data, or just metadata stats?
2. Large clusters may need streaming merge—do we need incremental write to avoid high memory usage?
3. Do we enforce consistent `StatType::kGroupId` across nodes or leave as-is?

## 7. Python Implementation (`tools/trace_combiner.py`)

The python tool `tools/trace_combiner.py` is a production-ready implementation of the alignment and combination logic.

### Usage
```bash
./tools/trace_combiner.py \
  --traces node0.xplane.pb node1.xplane.pb \
  --snapshots node0.snapshots.pb node1.snapshots.pb \
  --offsets offsets.jsonl \
  --output combined.xplane.pb \
  [--no_correction] \
  [--print_events]
```

### Features
- **Alignment + Merge:** Reads per-node XSpaces, interpolates timestamps using snapshot pairs and offset windows, and rewrites timestamps before merging.
- **JSONL Offset Support:** Understands both legacy midpoint files and the GraphCalc JSONL format (`meta` header + `window_start_ns` / `window_end_ns`) and remaps windows into each node’s clock domain automatically.
- **Monotonicity Guard:** Ensures timestamp monotonicity is preserved after correction.
- **Trace Viewer Compatibility:** Renames planes to ensure compatibility with standard XLA conversion tools (`xspace_to_trace_events`):
    - **GPU Planes:** Renamed to `/device:GPU:<global_id>` (where `global_id = node_id * 64 + local_id`).
    - **Host Planes:** Node 0 stays on `/host:CPU`, while nodes `> 0` are remapped to `/device:GPU:<10000 * node_id>` so each node’s host activity shows up as a distinct device track.
- **Configurable Correction:** Pass `--no_correction` to skip timestamp adjustment and simply merge/rename planes (useful for debugging raw traces).
- **Event-level Debugging:** `--print_events` logs every event's metadata name and aligned timestamp while processing, making it easier to spot anomalies.

## 8. Collective Happens-Before Validation

To verify alignment quality we compare collective operations (e.g., AllReduce, AllGather) across nodes and count **violations**—cases where one node's collective kernel finishes *before* another node's matching collective starts, which is physically impossible.

### 8.1 New Tool: `tools/validate_collectives.py`

| Item | Description |
|------|-------------|
| **Purpose** | Load one or more XSpace files (raw or combined) and report collective timing violations. |
| **Inputs** | List of `*.xplane.pb` files (each representing a node or the combined trace). |
| **Outputs** | JSON summary + human-readable report (violations, overlaps, warnings). |

### 8.2 CLI Usage
```bash
# Compare raw (uncorrected) traces
./tools/validate_collectives.py \
  --traces node0.xplane.pb node1.xplane.pb \
  --output violations_raw.json

# Compare after correction
./tools/validate_collectives.py \
  --traces combined.xplane.pb \
  --output violations_corrected.json
```

### 8.3 Integration with `trace_combiner.py`
Add a `--validate` flag to run the happens-before check automatically after combining:
```bash
./tools/trace_combiner.py \
  --traces node0.xplane.pb node1.xplane.pb \
  --offsets offsets.jsonl \
  --output combined.xplane.pb \
  --validate
```
When enabled, the tool will:
1. Run alignment + merge as usual.
2. Call `analyze_collective_operations()` on the combined XSpace.
3. Print summary: `Violations: X, Overlaps: Y, Warnings: Z`.
4. Optionally write detailed report to `combined.violations.json`.

### 8.4 Algorithm (from `xprof_analysis.py`)
1. **Extract RCCL events:** For each GPU plane, find events whose metadata name contains `rccl` and extract the `hlo_op` stat.
2. **Group by HLO op:** Build a DataFrame per node with `(hlo_op, start_ns, end_ns)`.
3. **Pairwise comparison:** For each unique collective HLO op and each pair of nodes, compare the i-th invocation:
   - **Violation:** `node_a.end < node_b.start` or vice-versa.
   - **Overlap:** intervals intersect (expected for correct collectives).
   - **Warning:** no matching event or mismatched counts.
4. **Aggregate:** Sum violations/overlaps/warnings across all ops and pairs.

### 8.5 Expected Outcome
| Scenario | Violations | Notes |
|----------|------------|-------|
| Raw traces (no correction) | High (100s–1000s) | Clock skew causes apparent non-overlap. |
| Corrected traces | 0 (ideal) | Alignment should eliminate false violations. |
| Residual violations | Low (<5) | May indicate real timing anomalies or insufficient offset data. |

### 8.6 File-Level Changes
| File | Change |
|------|--------|
| `tools/validate_collectives.py` | New script implementing `get_rccl_hlo_df`, `check_collective_happens_before_violations`, `analyze_collective_operations`. |
| `tools/trace_combiner.py` | Add `--validate` flag; call validation after merge. |
| `trace_combiner.md` | This section. |

### 8.7 Testing
| Test | Scenario |
|------|----------|
| `validate_collectives_test.py` | Synthetic 2-node XSpaces with known overlap/violation counts. |
| Integration | Run on real 2-node traces before/after correction; assert violation count drops. |

## 9. Development & Maintenance

### 9.1 Regenerating Protocol Buffers
The Python tools rely on `tools/xplane_pb2.py` which is generated from the XLA/TensorFlow protobuf definitions. If the XSpace schema changes (e.g., in `third_party/tsl/tsl/profiler/protobuf/xplane.proto`), you must regenerate the Python bindings:

```bash
# Assuming you are in the project root
protoc -I=third_party/tsl \
       --python_out=tools/ \
       third_party/tsl/tsl/profiler/protobuf/xplane.proto
```

Ensure `protoc` is installed and available in your path.
