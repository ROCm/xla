# ROCm Latency Check Tool

A single-source, HIP-runtime-only micro-benchmark for measuring the **end-to-end
per-request latency** of a DL/LLM-style dispatch pipeline and for **comparing
that latency across different ROCm versions**. It uses only the HIP runtime.

Each request runs the pipeline:

```
H2D  ->  timedBusyKernel  ->  D2D  ->  timedBusyKernel  ->  D2H
```

and the full round-trip is timed. Many consumer threads borrow/return GPU slots
to emulate a TF/JAX/XLA session pool. The tool reports a latency percentile
histogram (min / mean / p50 / p75 / p95 / p98 / p99 / p99.9) plus throughput
(QPS), and can append every report to a CSV that is stamped with the ROCm/HIP
version and the full run configuration.

## Why this pipeline

The compute op is a **fixed-duration on-GPU busy kernel** (`timedBusyKernel`,
which spins on `wall_clock64` for ~80us), not a library GEMM. Using a
deterministic compute duration is deliberate: it removes kernel/algorithm
selection as a variable, so run-to-run and **ROCm-version-to-version**
differences are attributable to **HIP runtime + memory-copy overhead** — which
is exactly what this regression tool is meant to track.

The two busy-kernel stages mimic a real DL/LLM step:

1. **H2D** — load the input tensor onto the device.
2. **timedBusyKernel** — stand-in for the input compute/transpose that produces
   the tensor which is then moved.
3. **D2D** — a device-to-device copy (buffer copy / layout / KV-cache on a single
   GPU, or a cross-GPU transfer with `--d2d-peer`).
4. **timedBusyKernel** — stand-in for the follow-on compute that consumes the
   moved tensor.
5. **D2H** — read the result back to the host.

Because there are **two** busy kernels per request, the synthetic compute portion
of each request is ~2x a single stage (~160us total at the default duration).

The copies use XLA main's typed HIP driver-style async APIs so the tool
exercises the same runtime paths XLA does:

- `hipMemcpyHtoDAsync` for H2D
- `hipMemcpyDtoDAsync` for D2D
- `hipMemcpyDtoHAsync` for D2H

## How it works

- `num-gpus x streams-per-gpu` slots are created; each slot owns one HIP stream
  on which its H2D, compute, D2D, and D2H are serialized (matching TF's aliased
  StreamGroup behavior).
- Producer threads enqueue requests; consumer threads dequeue, borrow a free
  slot, run the pipeline, synchronize, record the wall-clock latency, and return
  the slot.
- Reports are **cumulative** (computed over all requests since start) and are
  printed every `--report-interval-sec`; a final report is printed at the end.
- The run stops after `--duration-sec`; it is not infinite.

## Building

Single source file, no build system — compile with `hipcc` (no math libraries to
link):

```bash
hipcc -O3 -std=c++17 rocm_latency_check_tool.cpp -o rocm_latency_check_tool
```

Requirements:

- An AMD GPU (or several) and a ROCm installation providing `hipcc`. No hipBLASLt.

To build against a **specific ROCm version** (this is the core of the regression
flow), invoke that version's `hipcc`. The binary embeds the HIP version it was
built with, which is exactly the version under test:

```bash
/opt/rocm-7.2.1/bin/hipcc -O3 -std=c++17 rocm_latency_check_tool.cpp \
  -o rocm_latency_check_tool_7.2.1
```

## Usage

Run with defaults:

```bash
./rocm_latency_check_tool
```

List all flags:

```bash
./rocm_latency_check_tool --help
```

Single-GPU run (D2D is an intra-device copy, so this works on one GPU):

```bash
./rocm_latency_check_tool --num-gpus 1 --duration-sec 30 --csv results.csv --label 7.2.1
```

Cross-GPU D2D run (requires at least 2 GPUs):

```bash
./rocm_latency_check_tool --num-gpus 2 --d2d-peer --duration-sec 30 --csv results.csv --label 7.2.1
```

Flags accept both `--flag value` and `--flag=value`.

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--num-gpus N` | `4` | Number of GPUs to use. Must be `<=` the number of visible devices. |
| `--streams-per-gpu N` | `4` | Per-GPU slots/streams. Total slots = `num-gpus * streams-per-gpu`. |
| `--duration-sec N` | `10` | Total measured run length, in seconds. |
| `--report-interval-sec N` | `3` | How often a (cumulative) report is printed / written to CSV. |
| `--target-qps N` | `0` | Paced target QPS. `0` = max throughput (no pacing). |
| `--consumer-threads N` | `16` | Number of consumer threads competing for slots. |
| `--max-queue-size N` | `10000` | Per-consumer bounded request queue size. |
| `--reduced-d2h` | on | Copy back only `--reduced-d2h-bytes` (eval-style small output). |
| `--no-reduced-d2h` | | Copy back the full output matrix instead. |
| `--reduced-d2h-bytes N` | `11800` | Bytes copied back when reduced-D2H is enabled. |
| `--d2d-bytes N` | `0` | Bytes for the D2D copy. `0` = mirror the input (B) size. |
| `--d2d-peer` | off | Make D2D a cross-GPU copy to the neighbor device. Requires `--num-gpus >= 2`. |
| `--csv PATH` | disabled | Append one row per report to this CSV file (see below). |
| `--label STR` | auto | ROCm release tag stamped into the CSV `rocm_version` column (e.g. `7.2.1`). If omitted, the compiled HIP version is used. |
| `-h`, `--help` | | Print help and exit. |

### D2D: single-GPU vs cross-GPU

By default the D2D stage is an **intra-device** buffer-to-buffer copy, so the
whole tool runs on a single GPU. With `--d2d-peer`, the D2D destination is a
buffer on the **neighbor GPU** `(g+1) % num_gpus`; the tool enables peer access
at startup and the copy becomes a cross-GPU transfer. `--d2d-peer` therefore
requires `--num-gpus >= 2`.

## The compute kernel is fixed by design

The busy kernel's parameters are **compile-time constants** (no flags, no
environment variables), so the compute baseline is fixed and fully reproducible
across ROCm versions:

- duration ≈ 80us (`wall_clock64` spin)
- 256 threads per block
- blocks = auto (GPU CU count)
- yields with `s_sleep` while spinning

## Reading the output

Each console report looks like:

```text
Final Benchmark Statistics ==================================
-- Latency Histogram (us)
             count = 12261
               min = 2391
               max = 25777
              mean = 4687.99
            stddev = 798.48
            median = 4344.00
              75% <= 4406.25
              95% <= 4754.45
              98% <= 14536.06
              99% <= 16994.74
            99.9% <= 22018.08
-- Throughput
       wall_time_s = 6.0040
       total_iters = 12261
               QPS = 2042.15
```

The startup banner echoes the resolved configuration (pipeline, D2D/D2H modes,
busy-kernel config) and the detected ROCm/HIP/runtime/driver versions, so every
console log is self-describing and reproducible.

## CSV output

When `--csv PATH` is set, the tool writes one row per periodic report
(`phase=interval`) plus one final row (`phase=final`). If the file already has
content the header is **not** rewritten and rows are appended, so you can point
many runs (across ROCm versions) at the same file and accumulate them.

Every row is fully self-describing — it carries the ROCm/HIP version, the full
run configuration, and the latency/throughput stats — so the CSV can be loaded
directly into pandas/Excel and filtered or pivoted by `rocm_version` / `label`.

Columns:

| Column | Meaning |
| --- | --- |
| `timestamp_iso` | Local wall-clock time the row was written. |
| `rocm_version` | `--label` if set, else the compiled HIP version string. |
| `hip_version` | Compile-time `HIP_VERSION_MAJOR.MINOR.PATCH`. |
| `hip_runtime_version` | Runtime `hipRuntimeGetVersion()`. |
| `hip_driver_version` | Runtime `hipDriverGetVersion()`. |
| `label` | Raw `--label` value (empty if not passed). |
| `phase` | `interval` for periodic snapshots, `final` for the end-of-run row. |
| `elapsed_sec` | Seconds elapsed when the row was written. |
| `duration_sec`, `report_interval_sec` | Run cadence config. |
| `num_gpus`, `streams_per_gpu`, `total_slots` | Slot topology. |
| `consumer_threads`, `target_qps`, `max_queue_size` | Dispatch config. |
| `reduced_d2h`, `reduced_d2h_bytes` | D2H mode (`1`/`0`) and reduced byte count. |
| `d2d_bytes` | Requested D2D size (`0` = mirror input B size). |
| `d2d_peer` | `1` if the D2D is a cross-GPU (peer) copy, else `0`. |
| `compute_op` | Always `timedBusyKernel`. |
| `busy_kernel_ns`, `busy_kernel_threads`, `busy_kernel_blocks`, `busy_kernel_use_sleep` | Fixed busy-kernel config (documents the compute baseline; `blocks=0` = auto). |
| `work_m`, `work_n`, `work_k` | Workload shape that determines the copy sizes. |
| `count`, `qps` | Requests measured so far and throughput. |
| `min_us`, `max_us`, `mean_us`, `stddev_us` | Latency summary (microseconds). |
| `p50_us`, `p75_us`, `p95_us`, `p98_us`, `p99_us`, `p999_us` | Latency percentiles (microseconds). |

## Regression workflow: comparing ROCm versions

1. Build one binary per ROCm version using that version's `hipcc` (see
   [Building](#building)).
2. Run each binary with identical flags, writing into the **same** CSV and
   tagging each with a distinct `--label`:

```bash
./rocm_latency_check_tool_7.1.0 --duration-sec 30 --csv latency.csv --label 7.1.0
./rocm_latency_check_tool_7.2.1 --duration-sec 30 --csv latency.csv --label 7.2.1
```

3. Compare the `final` rows (or the percentile columns) across `label`s to spot
   latency/throughput regressions between ROCm versions.
