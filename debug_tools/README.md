# debug_tools

This directory hosts debugging utilities used when triaging issues while running Maxtext-TE/JAX/TF workloads on AMD GPUs (ROCm)

## Scope

The tools collected here are organised around two modes:

### 1. Hang investigation

Helpers that drive `gdb`, `rocgdb`, `umr`, the ROCR Debug Agent
(`librocm-debug-agent`), and Remote Debug Lib (RDL) to inspect a hung
process and its GPU state. Typical use cases:

- Attaching `gdb` / `rocgdb` to a stuck XLA worker and capturing host- and
  device-side backtraces across all ranks.
- Inspecting in-flight HIP streams, kernels, and queues via `rocgdb`.
- Dumping GPU register state, wavefront status, ring-buffer contents, and
  shader-engine activity with [`umr`](https://gitlab.freedesktop.org/tomstdenis/umr)
  (AMD's User Mode Register debugger) to determine whether a kernel is
  genuinely stuck, waiting on a fence, or blocked on an RDMA / SDMA
  completion.
- Loading the [ROCR Debug Agent](https://rocm.docs.amd.com/projects/rocr_debug_agent/en/latest/)
  (`HSA_TOOLS_LIB=librocm-debug-agent.so.2`) into the ROCr runtime to dump
  the state of every AMD GPU wavefront on a queue error (memory violation,
  `s_trap 2`, illegal instruction) or on demand via
  `kill -s SIGQUIT <pid>` (equivalently `Ctrl-\`). This is the lightest-
  weight way to get a per-wavefront snapshot of a hung XLA / MORI job
  without re-running under a debugger.
- Using Remote Debug Lib (RDL) for multi-tenancy VM environments where
  direct `umr` / kernel-level access is not available. RDL exposes a
  remote-attach surface so the host or a privileged sidecar can collect
  the same hang-triage signals (GPU queues, kernel state, fences) from
  inside a guest VM that lacks the host privileges `umr` would normally
  require.

### 2. NaN / numerical-correctness investigation

Tools for catching and localising NaN (and other anomalous numerical outputs)
produced by HLO computations and collective ops. They cover:

- Scanning intermediate buffers for NaN / Inf and reporting the first offending
  HLO instruction, buffer index, and rank.
- Comparing per-rank outputs across collective boundaries to isolate whether a
  NaN originates from compute or from communication.
- Lightweight wrappers for re-running a captured execution under stricter
  numerical checks.

## Layout

Individual tools live in their own subdirectories with self-contained usage
notes. See each subdirectory's `README.md` for invocation details.
