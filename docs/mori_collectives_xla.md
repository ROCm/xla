# MORI Collectives in XLA: Design, Interface, and Early Evaluation

*Draft — results tables to be filled once MaxText / microbenchmark numbers
land. Written against the `pemeliya/collectives-bench` line of work (opt-in
MORI backend + `XLA_GPU_COLLECTIVE_TIMING` profiler).*

---

## Why this matters

On multi-GPU training, a large fraction of step time is collective
communication: all-reduce of gradients, all-gather of weights under FSDP,
reduce-scatter of shards, and so on. XLA already speaks that vocabulary as
HLO (`all-reduce`, `all-gather`, …). On ROCm the default host library behind
those ops is **RCCL**.

**MORI** (Modular RDMA Interface) is an alternative ROCm collectives stack.
XLA can select it at runtime without changing the model or the HLO shape of
the program. This note explains:

1. what MORI is (briefly),
2. how its collectives path is designed,
3. how it plugs into XLA next to RCCL,
4. small JAX examples that stress **compute ↔ collective overlap**,
5. how we plan to measure MaxText training vs RCCL (numbers TBD).

---

## What is MORI?

MORI is AMD’s modular GPU communication framework. Rather than a single
monolithic collectives library, it layers reusable pieces:

| Layer | Role |
| --- | --- |
| **CCO** (Collective Communication Object) | Explicit communicator (`ccoComm`) with GPU-initiated one-sided ops |
| **Transports** | **LSA** — intra-node load/store over XGMI symmetric addresses; **GDA** — cross-node GPU-direct RDMA; **SDMA** — copy-engine transfers with device-visible signals |
| **Collectives facade** | Higher-level AllReduce / AllGather / … kernels used by XLA |
| **Symmetric heap** | Static collective memory pool so peer addresses are known after init |

For XLA we care most about CCO + the collectives facade: XLA creates a world
of PEs once, allocates collective buffers from the symmetric heap, and
launches facade kernels on HIP streams.

Further reading on the MORI side: the
[MORI CCO Guide](https://rocm.github.io/mori/MORI-CCO-GUIDE.html).

---

## Collectives design (MORI side)

### Explicit communicators, not a process-global singleton

Each `ccoComm` is independently created. A process can hold multiple views
(sub-communicators) over contiguous PE ranges. XLA maps a `CliqueKey` to a
facade via `CollectivesFacade::Get(rank, num_ranks, rank_to_pe)`.

### Eager world init → symmetric heap early

MORI’s collective memory (`CollectivesFacade::Allocate`) needs every PE to
finish CCO bootstrap first. XLA therefore **eagerly** initializes all local
devices during client setup (`MoriCollectives::InitializeTopology`):

- **Single process:** one clique; PE == local device ordinal.
- **Multi-process:** process 0 publishes a UID under `mori_shmem_global_uid`
  in the KV store; every PE across the world init concurrently so the socket
  bootstrap can rendezvous.

That makes `MemorySpace::kCollective` allocations valid *before* any
executable runs — important for PjRt / StreamExecutor collective buffers.

### What ops XLA drives today

Through `MoriCommunicator` → `CollectivesFacade::Run*`:

| Supported | Not yet |
| --- | --- |
| AllReduce, ReduceScatter, AllGather | Broadcast |
| AllToAll, CollectivePermute | `ncclCommSplit`-style arbitrary splits |
| Send / Recv, Barrier, Quiet / Fence | — |

Async collectives are marked via `GpuCommunicator::SetAsync` so the facade
(and the timing profiler) can distinguish compute-stream vs async-stream
launches.

### Pool sizing (env)

| Variable | Meaning |
| --- | --- |
| `MORI_COLLECTIVES_POOL_SIZE_IN_MB` | Symmetric heap size (default 2048) |
| `MORI_COLLECTIVES_MAX_STAGING_IN_MB` | Staging cap (`0` → MORI default) |
| `MORI_ENABLE_SDMA` | Prefer SDMA path where applicable |

---

## XLA interface: same registry, different implementation

XLA’s GPU collectives surface is `xla::gpu::GpuCollectives`, registered per
platform and implementation name:

```text
XLA_COLLECTIVES_REGISTER("ROCM", "rccl",  100, ...);  // default
XLA_COLLECTIVES_REGISTER("ROCM", "mori", -100, ...);  // opt-in
```

Resolution (`GpuCollectives::Resolve`):

1. Explicit override on `GpuExecutableRunOptions`, else
2. `--xla_gpu_collectives_implementation=<name>`, else
3. Highest-priority registration for the platform (RCCL on ROCm).

So enabling MORI is a flag, not a rebuild of the model:

```bash
export XLA_FLAGS="--xla_gpu_collectives_implementation=mori"
```

Key files under `xla/backends/gpu/collectives/`:

| File | Role |
| --- | --- |
| `mori_collectives.{h,cc}` | Topology / PE init, allocate, communicator factory |
| `mori_communicator.{h,cc}` | Map XLA collective API → facade `Run*` |
| `mori_kernels.cu.cc` | HIP TU that compiles device kernels (`MORI_KERNELS_IMPL`) |
| `rccl_collectives.cc` | Default RCCL path for comparison |

Vendored MORI sources are pulled hermetically via
`third_party/roc_mori/workspace.bzl` (`@roc_mori`).

### Conceptual difference vs RCCL

| | RCCL (default) | MORI (opt-in) |
| --- | --- | --- |
| ID model | Per-clique ID store | One eager world UID (+ contiguous PE facades) |
| Collective memory | `ncclMemAlloc`-style | Symmetric heap after eager init |
| Host library | Mature general-purpose | CCO + facade kernels in-process |
| Goal in XLA today | Production default | Evaluate latency / overlap / SDMA tradeoffs |

---

## Small JAX examples (compute overlapping a collective)

MaxText numbers are still in flight. Until they land, the cheapest way to
exercise the stack — and the latency-hiding scheduler — is a **tiny multi-GPU
JAX program** where an all-reduce is independent of a large matmul for long
enough that XLA can overlap them.

### 0. Prerequisites

```bash
# 2+ AMD GPUs visible to JAX/ROCm
export JAX_PLATFORMS=rocm

# A: RCCL (default) — omit the flag or set explicitly
export XLA_FLAGS="--xla_gpu_collectives_implementation=rccl"

# B: MORI
export XLA_FLAGS="--xla_gpu_collectives_implementation=mori"
# optional:
# export MORI_ENABLE_SDMA=1
# export MORI_COLLECTIVES_POOL_SIZE_IN_MB=2048

# Skip first N executable runs per device, then time collectives at exit
export XLA_GPU_COLLECTIVE_TIMING=5
```

Run the same script under A and B; compare wall time of the timed loop **and**
the stderr table printed when the process exits.

### 1. Overlap sketch: independent all-reduce + matmul

The dependency graph matters. If the matmul consumes the all-reduce output,
there is nothing to overlap. The pattern below keeps them independent until a
final combine:

```python
"""overlap_ar_matmul.py — pedagogical multi-GPU overlap microbenchmark.

  python -m jax.distributed.launch --nproc_per_node=2 overlap_ar_matmul.py
or a single process with multiple local devices via jax.devices().
"""

from functools import partial
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def main():
  devices = jax.devices()
  assert len(devices) >= 2, "need >= 2 GPUs"
  mesh = Mesh(devices[:2], ("data",))

  # Per-rank activation shard and a fat matmul that does *not* depend on the
  # all-reduce result until the end.
  m, k, n = 8192, 8192, 8192
  x = jnp.ones((m // mesh.size, k), dtype=jnp.bfloat16)
  w = jnp.ones((k, n), dtype=jnp.bfloat16)
  x = jax.device_put(x, NamedSharding(mesh, P("data", None)))
  w = jax.device_put(w, NamedSharding(mesh, P(None, None)))

  @jax.jit
  def step(x, w):
    # Gradient-style reduction across the data axis (becomes all-reduce).
    reduced = jax.lax.with_sharding_constraint(
        jax.lax.psum(x, "data"), NamedSharding(mesh, P(None, None))
    )
    # Independent compute: matmul on the *local* shard, not on `reduced`.
    y = x @ w
    # Touch both results so neither is DCE'd; forces a join after overlap.
    return y.sum() + reduced.sum()

  # Warmup (excluded by XLA_GPU_COLLECTIVE_TIMING skip count).
  for _ in range(5):
    step(x, w).block_until_ready()

  iters = 20
  t0 = time.perf_counter()
  for _ in range(iters):
    out = step(x, w)
  out.block_until_ready()
  dt = time.perf_counter() - t0
  print(f"devices={mesh.size} iters={iters} wall_s={dt:.4f} "
        f"per_iter_ms={1e3 * dt / iters:.3f}")

if __name__ == "__main__":
  main()
```

What to look for:

- **Wall time / iter** under RCCL vs MORI (same shapes, same flags otherwise).
- Profiler stderr: `Async collectives` should show `kAllReduce` (or similar)
  while sync totals stay smaller if overlap is working.
- If async totals are ~0, the collective may have stayed on the compute
  stream — try larger `m,k,n`, or inspect the HLO for `all-reduce-start` /
  `all-reduce-done` after latency-hiding scheduling.

### 2. FSDP-flavored all-gather + local GEMM

A second useful microbenchmark mimics weight all-gather under FSDP: gather a
sharded weight, then GEMM. Overlap is harder here because the GEMM depends on
the gather — still valuable to compare **exposed** collective time (sync
bucket) between backends:

```python
@jax.jit
def fspd_like(x, w_shard):
  # all-gather weights along "model"
  w = jax.lax.all_gather(w_shard, "model", tiled=True)
  return x @ w
```

Sweep message sizes (change shard shapes) and read the profiler’s
power-of-two size buckets — that is where MORI vs RCCL bandwidth/latency
tradeoffs show up before you invest in a full MaxText run.

---

## Measuring collectives: `XLA_GPU_COLLECTIVE_TIMING`

This branch adds an opt-in process-wide profiler
(`xla/backends/gpu/runtime/collective_profiler.*`):

```bash
export XLA_GPU_COLLECTIVE_TIMING=5   # skip first 5 runs of each executable/device
```

At exit it prints something like:

```text
===== XLA GPU collective timing (devices: D, skipped first N runs ...) =====
--- Sync collectives (compute stream, exposed) ---
kAllReduce: calls=… total=… ms (…%) avg=… us
  [1 MiB, 2 MiB)  calls=… total=… ms  avg=… us  min=… us  max=… us
SYNC TOTAL: …
--- Async collectives (async stream, may overlap compute) ---
…
ASYNC TOTAL: …
TOTAL: … ms
=====
```

Notes:

- Times are GPU-event based on the collective’s stream (includes peer wait).
- Sync vs async split matches whether the thunk ran under an async start/done
  scope (`SetAsync`).
- Collectives lowered into **command buffers are not timed** — disable
  collective command buffers if you need them in the table.

---

## MaxText training vs RCCL (numbers TBD)

### Method (locked)

1. Fix hardware, ROCm, MaxText commit, model, parallel layout (TP/DP/FSDP),
   batch, seq length.
2. Run **A**: default RCCL (`xla_gpu_collectives_implementation=rccl` or unset).
3. Run **B**: MORI (`=mori`), same everything else; record `MORI_*` envs.
4. Warmup then steady-state: step time and/or tokens/s.
5. Both runs with `XLA_GPU_COLLECTIVE_TIMING=<warmup>` for the stderr table.
6. Optionally dump a short profiler excerpt for the dominant size buckets.

### Results template

| Config | Backend | GPUs / nodes | Step time (ms) | Tokens/s | SYNC TOTAL (ms) | ASYNC TOTAL (ms) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| *TBD* | RCCL | | | | | | |
| *TBD* | MORI | | | | | | |

Dominant ops (fill from profiler):

| Op | Size bucket | RCCL avg (us) | MORI avg (us) | Δ |
| --- | --- | --- | --- | --- |
| AllReduce | | | | |
| AllGather | | | | |
| ReduceScatter | | | | |

### What we will claim only after numbers exist

- End-to-end training throughput delta (MORI − RCCL).
- Whether gains come from lower exposed sync time, better async overlap, or
  both.
- Any size regimes where RCCL remains ahead (expected; document honestly).

---

## Caveats

- MORI is **opt-in** (registry priority −100); RCCL remains the ROCm default.
- Subcommunicators today are contiguous PE ranges, not arbitrary NCCL splits.
- Broadcast over MORI is unimplemented in the XLA communicator.
- Profiler skip count is per `(executable, device)` — set it high enough to
  clear MaxText compile / autotune noise.
- Do not compare against fused GEMM+collective MORI operators from serving
  stacks without saying so; that is a different integration path than the XLA
  collectives backend described here.

---

## Where the code lives

- Backend: `xla/backends/gpu/collectives/mori_*.cc`
- Flag: `xla_gpu_collectives_implementation` in `xla/xla.proto`
- Profiler: `xla/backends/gpu/runtime/collective_profiler.*`
- MORI pin: `third_party/roc_mori/workspace.bzl`

---

## Next edits to this draft

1. Paste MaxText A/B numbers into the results tables.
2. Attach one real `XLA_GPU_COLLECTIVE_TIMING` dump (RCCL + MORI) from the
   overlap microbenchmark and from MaxText.
3. Optionally promote the JAX scripts to a small `docs/snippets/` or
   `examples/` path once they are validated on MI300/MI350 hardware.
