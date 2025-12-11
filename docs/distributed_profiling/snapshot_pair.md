# Snapshot Pair Plan

> **Format Note:** This spec mirrors the structure used in
> `master_alpha_sync.md`: exec summary, requirements, per-file change list,
> testing, and open questions.

## 1. Executive Summary
- **Goal:** Capture `(sys_clock_ns, RocmTracer::GetTimestamp())` pairs every
  4 s per distributed node so later stages can align traces collected on the
  **R timeline** (the internal ROCm tracer clock shared by host + device
  events) with local host clocks (**S timeline**, also used by the probing
  stack).
- **Scope:** Snapshots are only needed when `InitializeDistributedSync()` is
  active. The ROCm collector owns sampling/storage/export, but the distributed
  sync path is responsible for starting/stopping the sampling thread.
- **Output:** A per-node artifact (JSONL or proto) referenced by future trace
  combiners and graph-calculation jobs.

## 2. Requirements
| Requirement | Details |
|-------------|---------|
| Cadence | Default 4 s; configurable via `tensorflow::ProfileOptions.advanced_configuration["snapshot_pair_period_ms"]`. |
| Coverage | Start only when `RocmTraceCollectorImpl::InitializeDistributedSync()` succeeds (i.e., distributed mode). Stop when distributed sync shuts down or tracing disables. |
| Clock Source | `sys_clock_ns` uses `EnvTime::NowNanos()` (CLOCK_REALTIME). `rocm_ts_ns` uses `RocmTracer::GetTimestamp()`. |
| Atomicity | Read both clocks back-to-back under a lightweight mutex to ensure `<5 µs` skew per snapshot. |
| Storage | thread-safe ring buffer sized for `(profiling_duration / cadence) + headroom`. |
| Export | Write stable artifact per host (embedded in XSpace as `/host:snapshots` plane). |
| Telemetry | Emit counters (snapshots_taken, missed_deadline) to assist validation. |

## 3. Implementation Plan by File

### 3.1 `xla/backends/profiler/gpu/rocm_trace_collector_impl.{h,cc}`
| Change | Detail |
|--------|--------|
| Data model | Add `struct ClockSnapshot { uint64_t sys_ns; uint64_t rocm_ns; };` and `std::vector<ClockSnapshot> snapshots_ ABSL_GUARDED_BY(snapshot_mutex_);`. |
| API | Add `void StartSnapshotThread(); void StopSnapshotThread();`. These are called only from distributed-sync initialization/teardown, not every run. |
| Thread | Launch `snapshot_thread_` that loops while `running_` flag true: collect pair, sleep `snapshot_period`. Reuse `absl::SleepFor`. |
| Export | Extend `Export(XSpace* space)` to append `/host:snapshots` plane with collected pairs. |

### 3.2 `xla/backends/profiler/gpu/rocm_profiler_sdk.cc`
| Change | Detail |
|--------|--------|
| Distributed Sync | When `collector->InitializeDistributedSync()` succeeds, immediately invoke `collector_->StartSnapshotThread(snapshot_period_ms)`. |
| Teardown | In `Export()`/`Disable()`, stop the snapshot thread only if distributed sync is active. |

### 3.3 `xla/tsl/profiler/rpc/client/save_profile.cc`
| Change | Detail |
|--------|--------|
| Tool data | Add helper `SaveSnapshotPairs(run_dir, host, const SnapshotPairsProto&)`. |
| RPC | Extend `ProfileResponse.tool_data` list to include `snapshot_pairs.pb`. |

### 3.4 `third_party/tsl/tsl/profiler/protobuf/xplane.proto`
| Change | Detail |
|--------|--------|
| Proto | Introduce `message SnapshotPair { uint64 sys_clock_ns = 1; uint64 tracer_clock_ns = 2; }` and `message SnapshotPairs { repeated SnapshotPair pairs = 1; uint64 sampling_period_ns = 2; }`. |

### 3.5 `xla/python/profiler.cc`
| Change | Detail |
|--------|--------|
| Export-to-TensorBoard | Expose snapshot artifacts via `ProfilerSession::stop()` result dictionary so Python callers can pass snapshot files to the combiner. |

### 3.6 `docs`
- Update `DOCUMENTATION_INDEX.md` with link to this spec.

## 4. Sampling + Storage Flow
```
Distributed run detected?
  ├─ InitializeDistributedSync()
  │    ├─ success → StartSnapshotThread()
  │    │     ├─ while (running_) every period:
  │    │     │     snapshot.sys_ns = EnvTime::NowNanos();
  │    │     │     snapshot.rocm_ns = RocmTracer::GetTimestamp();
  │    │     │     snapshots_.push_back(snapshot);
  │    │     └─ tracks dropped_samples counter
  │    └─ failure → no snapshot thread
  └─ ...

Teardown
  ├─ StopSnapshotThread() (only if started)
  ├─ Flush()
  └─ Export() -> SaveSnapshotPairs()
```

## 5. Testing
| Test | Description |
|------|-------------|
| Unit (`snapshot_pair_test.cc`) | Fake clock sources to ensure cadence + storage invariants. |
| Integration | Run single-node profiling for 12 s; expect ≥3 samples exported. |
| Stress | Force `snapshot_period_ms=10` to verify low-overhead path and bounding of buffer. |

## 6. Open Questions
1. Should we allow per-node override of cadence (via env var `XLA_ROCM_SNAPSHOT_PERIOD_MS`)?
2. Large runs might accumulate thousands of samples; do we need compression?
3. Is JSONL sufficient, or do we require binary proto for downstream streaming?



