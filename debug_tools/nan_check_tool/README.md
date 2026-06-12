# NaN Check Tool

A lightweight [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) tool that makes AMD GPU kernels **trap on the first invalid floating-point operation** (i.e. the operation that first produces a `NaN`). When paired with a GPU debugger such as `rocgdb`, this lets you pinpoint the exact kernel, wavefront, and instruction where a `NaN` (and, optionally, an infinity/overflow) is generated.

To get the faulting PC to point at the exact ALU instruction that produced the `NaN` (rather than somewhere downstream), enable **precise ALU exceptions** in your debugger — see the [`precise-alu-exceptions`](#precise-alu-exceptions) section below.

## How it works

`rocprofiler-sdk` notifies the tool whenever a code object is loaded and whenever a device kernel symbol is registered. For every (non-internal) kernel, the tool:

1. Allocates a small **trampoline** in executable GPU memory.
2. Writes a short ISA sequence into the trampoline that, before jumping to the real kernel entry point, sets bits in the hardware `MODE` register:
   - `DX10_CLAMP = 0` — disables clamping so invalid results are not silently sanitized.
   - `EXCP_EN.INVALID = 1` — enables the *invalid operation* floating-point exception (the source of `NaN`s).
   - Optionally `EXCP_EN.OVERFLOW = 1` — also trap on overflow/infinity (see `NAN_CHECK_OVERFLOW`).

When an enabled exception fires on the GPU, the wavefront traps. A debugger attached to the process then stops at the offending instruction, so you can inspect the call site, registers, and source line responsible for the bad value.


> **Note:** This tool targets `gfx9`-class GPUs (e.g. MI200/MI300). It checks the agent ISA name for `gfx9` during code-object load.

## Building

The tool is built as a shared library that gets loaded into your application by `rocprofiler-sdk`.

```bash
hipcc -fpic -g3 -fvisibility=hidden -fno-exceptions -shared \
  -lrocprofiler-sdk nan_check_tool.cpp -o nan_check_tool.so
```

Requirements:

- A ROCm installation that provides `hipcc`, the HSA runtime headers (`hsa/hsa.h`, `hsa/hsa_ext_amd.h`), and `rocprofiler-sdk` (we recommended at least ROCm-7.1.x).
- A `gfx9`-class AMD GPU.

## Usage

Set `ROCP_TOOL_LIBRARIES` to the built `.so` so that `rocprofiler-sdk` loads the tool into your program, then run your program under a GPU debugger.

### With `rocgdb`

```bash
ROCP_TOOL_LIBRARIES=$PWD/nan_check_tool.so \
NAN_CHECK_VERBOSE=1 \
rocgdb --args ./my_program
```

When a `NaN`-producing instruction executes, the GPU traps and `rocgdb` stops at the faulting instruction. Enable precise ALU exceptions first so the trap reports the exact instruction:

```text
(rocgdb) set amdgpu precise-alu-exceptions on
(rocgdb) run
(rocgdb) info threads
(rocgdb) bt
(rocgdb) x/i $pc
```

#### Example: trapping on a `NaN` and disassembling the faulting kernel

You can also drive everything from a single command line by passing `-ex`
commands to `rocgdb`. For example, to run a script, stop on the first invalid
operation, and disassemble the kernel that trapped:

```bash
ROCP_TOOL_LIBRARIES=debug_tools/nan_check_tool/nan_check_tool.so \
rocgdb -ex "set pagination off" -ex "run" -ex "disassemble" \
  --args python3 test_nan.py
```

When the `NaN` is produced, the GPU traps and `rocgdb` stops at the faulting
instruction (marked with `=>` in the disassembly). The output looks like:

```text
Thread 908 "python3" received signal SIGFPE, Arithmetic exception.
Warning: precise ALU exception reporting is not enabled, reported location
may not be accurate.  See "show amdgpu precise-alu-exceptions".
[Switching to thread 908, lane 0 (AMDGPU Lane 1:64:1:1/0 (0,0,0)[0,0,0])]
0x00007ffdfc575694 in loop_log_fusion () from memory://900366#offset=0x7f867a4d29a0&size=4128
Dump of assembler code for function loop_log_fusion:
   0x00007ffdfc575600 <+0>:     s_load_dwordx4 s[4:7], s[0:1], 0x0
   0x00007ffdfc575608 <+8>:     v_lshlrev_b32_e32 v0, 2, v0
   0x00007ffdfc57560c <+12>:    v_mov_b32_e32 v2, 0xb7000000
   0x00007ffdfc575614 <+20>:    s_mov_b32 s1, 0x800000
   0x00007ffdfc57561c <+28>:    s_waitcnt lgkmcnt(0)
   0x00007ffdfc575620 <+32>:    global_load_dword v1, v0, s[4:5]
   0x00007ffdfc575628 <+40>:    s_load_dword s0, s[6:7], 0x0
   0x00007ffdfc575630 <+48>:    s_waitcnt lgkmcnt(0)
   0x00007ffdfc575634 <+52>:    v_mul_f32_e32 v2, s0, v2
   0x00007ffdfc575638 <+56>:    s_mov_b32 s0, 0x3f317217
   0x00007ffdfc575640 <+64>:    s_waitcnt vmcnt(0)
   0x00007ffdfc575644 <+68>:    v_add_f32_e32 v1, v1, v2
   0x00007ffdfc575648 <+72>:    v_cmp_gt_f32_e32 vcc, s1, v1
   0x00007ffdfc57564c <+76>:    s_mov_b32 s1, 0x7f800000
   0x00007ffdfc575654 <+84>:    s_nop 0
   0x00007ffdfc575658 <+88>:    v_cndmask_b32_e64 v2, 0, 32, vcc
   0x00007ffdfc575660 <+96>:    v_ldexp_f32 v1, v1, v2
   0x00007ffdfc575668 <+104>:   v_log_f32_e32 v1, v1
   0x00007ffdfc57566c <+108>:   v_mov_b32_e32 v2, 0x41b17218
   0x00007ffdfc575674 <+116>:   v_cndmask_b32_e32 v2, 0, v2, vcc
   0x00007ffdfc575678 <+120>:   v_mul_f32_e32 v3, 0x3f317217, v1
   0x00007ffdfc575680 <+128>:   v_fma_f32 v4, v1, s0, -v3
   0x00007ffdfc575688 <+136>:   v_fmamk_f32 v4, v1, 0x3377d1cf, v4
   0x00007ffdfc575690 <+144>:   v_add_f32_e32 v3, v3, v4
=> 0x00007ffdfc575694 <+148>:   v_cmp_lt_f32_e64 s[0:1], |v1|, s1
   0x00007ffdfc57569c <+156>:   s_nop 1
   0x00007ffdfc5756a0 <+160>:   v_cndmask_b32_e64 v1, v1, v3, s[0:1]
   0x00007ffdfc5756a8 <+168>:   v_sub_f32_e32 v1, v1, v2
   0x00007ffdfc5756ac <+172>:   global_store_dword v0, v1, s[4:5]
   0x00007ffdfc5756b4 <+180>:   s_endpgm
End of assembler dump.
```

Because precise ALU exceptions were not enabled in this run, the reported
location may be slightly off. Add
`-ex "set amdgpu precise-alu-exceptions on"` before `-ex "run"` to make the
trap point at the precise faulting instruction (see
[`precise-alu-exceptions`](#precise-alu-exceptions)).

## precise-alu-exceptions

By default, GPU ALU exceptions are reported imprecisely: the wavefront may continue past the instruction that produced the `NaN` before it traps, so the reported PC does not point at the real culprit. Enabling **precise ALU exceptions** forces the hardware to report the exception synchronously at the faulting instruction, which is what makes this tool useful for locating the exact source of a `NaN`.

- In `rocgdb`, enable it before running:

```text
(rocgdb) set amdgpu precise-alu-exceptions on
```

Precise ALU exceptions add runtime overhead, so enable them only while debugging.

## Environment variables

| Variable | Effect |
| --- | --- |
| `ROCP_TOOL_LIBRARIES` | Path(s) to the tool `.so`. Required so `rocprofiler-sdk` loads the tool. |
| `NAN_CHECK_VERBOSE` | If set to anything other than `0`, logs each patched kernel (name, kernel object, trampoline address) and page-free counts to stderr. |
| `NAN_CHECK_OVERFLOW` | If set to anything other than `0`, also enables the **overflow** exception in addition to *invalid*, so the GPU also traps on overflow/infinity. |

## Caveats

Keep the following limitations in mind when interpreting traps:

- **Transient `NaN`s may cause false positives.** Some kernels are deliberately
  written to compute a value that is `NaN` on some lanes and then discard it
  (e.g. via a select/mask), so the `NaN` never affects the final result. The
  tool still traps on the arithmetic that produced it. This is rare, but it
  happens, so be prepared to recognize a trap on a value that is later thrown
  away.

- **Only `NaN`s produced by arithmetic are caught.** The tool relies on the
  hardware *invalid operation* floating-point exception, so it traps only when
  an instruction *generates* an invalid result, for example:
  - `0 * inf`
  - `inf - inf`
  - `0 / 0` or `inf / inf`
  - `sqrt` of a negative number
  - operations on a signaling `NaN` (sNaN)

  It does **not** trap when a junk/`qNaN` bit pattern is simply read from memory
  (or otherwise materialized) and then fed into arithmetic that happens to
  propagate it — propagating an existing `NaN` is not an invalid operation, so
  no exception fires. To find those, you have to trace back to where the bad
  bit pattern originated.
